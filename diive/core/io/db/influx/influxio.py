"""
CORE.IO.DB.INFLUX.INFLUXIO: INFLUXDB DOWNLOAD / UPLOAD / DELETE ENGINE
=====================================================================

The in-diive InfluxDB v2 I/O engine. :class:`InfluxIO` connects to an InfluxDB
described by a config directory (see :mod:`diive.core.io.db.influx.config`) and
provides download, upload, delete and schema-browsing.

This is a clean port of the former external ``dbc-influxdb`` package, folded into
diive so the optional ``db`` dependency group needs only the real third-party
client (``influxdb-client``). Improvements over the original: diive console
output (no ``logging``), no ``pytz`` / ``dateutil`` dependency, lazy client
imports (importing this module never requires the ``db`` extra), and the
data-version / units schema helpers the GUI explorer needs.

Required units are the caller's responsibility; all timestamps in the database
are stored in UTC.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from diive.core.io.db.influx import fluxql
from diive.core.io.db.influx.common import TAGS, convert_ts_to_timezone
from diive.core.io.db.influx.config import get_conf_filetypes, read_configfile, read_dbconf
from diive.core.io.db.influx.connection import get_client, get_delete_api, get_query_api
from diive.core.times.times import describe_time_resolution
from diive.core.utils.console import VERBOSE_PROGRESS, detail, info, success, warn

#: Records per synchronous write request in ``upload_singlevar``.
_WRITE_CHUNK_ROWS = 5000

#: Tags ``upload_singlevar`` needs: the delete is limited to one site and data
#: version, and ``download`` finds the variable by its ``varname`` tag.
_REQUIRED_TAGS = ['site', 'varname', 'data_version']
# A record stored without gain or offset was not scaled: gain 1, offset 0. Written as
# the raw buckets store them ('1.0' and '0.0' in nearly all raw data).
_TAG_DEFAULTS = {'gain': '1.0', 'offset': '0.0'}


class InfluxIO:
    """InfluxDB v2 connection backed by ``influxdb-client``.

    Constructed from a config-directory path (``dirconf``); the secret
    url/org/token are read from that directory's ``<dirconf>_secret`` sibling.
    Construction also pings the server, so a successfully built instance is live.
    """

    def __init__(self, dirconf: str, verbose: int = VERBOSE_PROGRESS) -> None:
        self.dirconf = Path(dirconf)
        self.verbose = verbose

        self.conf_filetypes, \
            self.conf_unitmapper, \
            self.conf_dirs, \
            self.conf_db = self._read_configs()

        self.test_connection()

    @staticmethod
    def _format_utc_offset(timezone_offset_to_utc_hours: int | float) -> str:
        """Format an hour offset as an ISO 8601 UTC offset string.

        Supports negative and fractional offsets; fractional hours are rounded
        to the nearest minute. e.g. ``1`` -> ``'+01:00'``, ``-5`` -> ``'-05:00'``,
        ``10`` -> ``'+10:00'``, ``5.5`` -> ``'+05:30'``.
        """
        total_minutes = round(timezone_offset_to_utc_hours * 60)
        sign = '+' if total_minutes >= 0 else '-'
        hours, minutes = divmod(abs(total_minutes), 60)
        return f"{sign}{hours:02d}:{minutes:02d}"

    def _add_timestamp_utc(self, timestamp_index, timezone_offset_to_utc_hours) -> pd.DatetimeIndex:
        """Localize a naive timestamp index to a fixed UTC offset.

        *timezone_offset_to_utc_hours* is the offset of the local timestamps to
        UTC, e.g. 1 for CET (UTC+01:00, winter time) or -5 for UTC-05:00;
        fractional hours are allowed (5.5 gives UTC+05:30). The timestamps keep
        their values and get this offset, so InfluxDB can store them in UTC.
        """
        # Needs to be in format '2022-05-27 00:00:00+01:00' for InfluxDB
        utc_str = self._format_utc_offset(timezone_offset_to_utc_hours)
        timestamp_index_utc = timestamp_index.tz_localize(utc_str)
        return timestamp_index_utc

    def upload_singlevar(self,
                         var_df: DataFrame,
                         to_bucket: str,
                         to_measurement: str,
                         timezone_offset_to_utc_hours: int | float,
                         delete_from_db_before_upload: bool = True,
                         delete_start: str | None = None,
                         delete_stop: str | None = None) -> None:
        """Upload single variable to database.

        The database needs to know the timezone because all data in the db are
        stored in UTC/GMT.

        Args:
            var_df: contains measured variable data and tags (data_detailed)
            to_bucket: name of database bucket
            to_measurement: name of measurement, e.g. 'TA'
            timezone_offset_to_utc_hours: e.g. 1, see the docstring of ``_add_timestamp_utc`` for details
            delete_from_db_before_upload: delete existing data before uploading. The delete is
                limited to the same site, measurement, variable name and data version as *var_df*,
                so *var_df* must hold exactly one ``site`` and one ``data_version``. Implemented to
                avoid duplicate uploads of the same data in cases where data remained the same, but
                one of the tags has changed.
            delete_start: start of the delete range, e.g. '2024-03-01 00:00:00', in the same
                local time as the index of *var_df* (same format and timezone frame as the
                *start* of ``download``). Default ``None`` uses the first timestamp of *var_df*.
            delete_stop: end of the delete range, e.g. '2024-04-01 00:00:00'. Default ``None``
                uses the last timestamp of *var_df*.

        The deleted range is ``[delete_start, delete_stop]`` with both bounds included:
        InfluxDB deletes from the earliest to the latest given time. Passing the same
        START and STOP as to ``download`` (which reads ``[start, stop)``) therefore also
        deletes a record stamped exactly at STOP. A ValueError is raised before anything
        is deleted if *var_df* is empty, has timestamps outside the delete range, or if
        its ``site`` tag holds several sites.

        Records with a NaN value are not written. If all values are NaN, the period is
        still deleted (the old values were rejected as well) and a warning says that
        nothing was written.

        Writes are synchronous, in chunks of ``_WRITE_CHUNK_ROWS`` records. A failed
        write raises a RuntimeError that says whether the period was already deleted.

        Note:
            *var_df* must contain one data column (the variable) plus tag
            columns from :data:`diive.core.io.db.influx.common.TAGS`. The tags
            ``site``, ``varname`` and ``data_version`` are required. A missing
            ``gain`` is written as ``'1.0'`` and a missing ``offset`` as ``'0.0'``
            (raw data stored without them were not scaled). Any other tag may be
            missing: the written points then do not carry it, as in the
            downloaded data. Do not add it as an empty column: the client can
            write an empty string as the tag value ``nan``.

        Returns:
            Nothing, only uploads to database.
        """
        from influxdb_client.client.write_api import SYNCHRONOUS  # lazy: only upload needs it

        if var_df.empty:
            raise ValueError("var_df has no records, nothing to upload. Nothing was deleted.")

        # Work on a copy so the caller's DataFrame is not mutated (the index is
        # localized to UTC below).
        var_df = var_df.copy()

        data_cols = var_df.columns.to_list()

        # Other tags may be missing (raw data stored without them). The
        # serializer skips absent tag columns, so the points get the same tag
        # set as the downloaded data (L300). Gain and offset get their neutral
        # values instead, as the data were not scaled.
        cols_not_in_data = [l for l in _REQUIRED_TAGS if l not in data_cols]
        if len(cols_not_in_data) > 0:
            raise ValueError(f"Data do not contain required tag columns: {cols_not_in_data}")
        for tag, default in _TAG_DEFAULTS.items():
            if tag not in data_cols:
                var_df[tag] = default
            else:
                var_df[tag] = var_df[tag].where(var_df[tag].notna(), default)
        data_cols = var_df.columns.to_list()

        # Detect field name (variable name)
        # The field name is the name of the column that is not part of the tags
        field = [l for l in data_cols if l not in TAGS]
        if len(field) > 1:
            raise ValueError(f"Only one field (variable name) allowed, found {field}.")

        if (delete_start is not None or delete_stop is not None) and not delete_from_db_before_upload:
            raise ValueError("delete_start and delete_stop need delete_from_db_before_upload=True.")

        start = stop = None
        if delete_from_db_before_upload:
            first, last = var_df.index.min(), var_df.index.max()
            start = str(first) if delete_start is None else delete_start
            stop = str(last) if delete_stop is None else delete_stop
            outside = (var_df.index < pd.Timestamp(start)) | (var_df.index > pd.Timestamp(stop))
            if outside.any():
                raise ValueError(
                    f"var_df has {int(outside.sum())} timestamps outside the delete range "
                    f"[{start}, {stop}] (data from {first} to {last}). Nothing was deleted.")
            data_version = list(set(var_df['data_version'].tolist()))
            if len(data_version) > 1:
                raise ValueError('Multiple data versions not supported')
            data_version = data_version[0]
            # Without the site, a bucket shared by several sites would lose this
            # variable for every site in the uploaded period.
            site = list(set(var_df['site'].tolist()))
            if len(site) != 1:
                raise ValueError(f"Exactly one site required for the delete before upload, found {site}.")
            site = site[0]
            # Tag extraction joins differing tag values with commas, so a single
            # value like 'ch-hon,ch-oth' still means several sites.
            if ',' in str(site):
                raise ValueError(f"The site tag '{site}' joins several sites with commas. "
                                 f"Upload one site at a time. Nothing was deleted.")
            self.delete(bucket=to_bucket, measurements=[to_measurement],
                        start=start, stop=stop,
                        timezone_offset_to_utc_hours=timezone_offset_to_utc_hours,
                        data_version=data_version, fields=field, site=site)

        deleted_msg = (f" The period {start} to {stop} (both included) was deleted from "
                       f"bucket {to_bucket} before the write." if delete_from_db_before_upload else "")

        # NaN records are not written (the serializer would skip them anyway).
        # Dropping them here gives the exact number of points.
        var_df = var_df[var_df[field[0]].notna()]
        n_points = len(var_df)
        if n_points == 0:
            warn(f"All values of {field[0]} are NaN: nothing was written to bucket {to_bucket}."
                 f"{deleted_msg} The old values in that period were rejected as well, so they "
                 f"stay deleted.", verbose=self.verbose)
            return

        # Add timezone info to timestamp
        var_df.index = self._add_timestamp_utc(timestamp_index=var_df.index,
                                               timezone_offset_to_utc_hours=timezone_offset_to_utc_hours)

        info("Connecting to database ...", verbose=self.verbose)
        info(f"--> UPLOAD TO DATABASE BUCKET {to_bucket}:  {field}", verbose=self.verbose)

        # Synchronous writes raise HTTP errors here. The batching WriteApi only
        # logged them, so a failed write after the delete went unnoticed (L227).
        n_written = 0
        with get_client(conf_db=self.conf_db) as client, \
                client.write_api(write_options=SYNCHRONOUS) as write_api:
            try:
                for i in range(0, n_points, _WRITE_CHUNK_ROWS):
                    chunk = var_df.iloc[i:i + _WRITE_CHUNK_ROWS]
                    write_api.write(to_bucket,
                                    record=chunk,
                                    data_frame_measurement_name=to_measurement,
                                    data_frame_tag_columns=TAGS,
                                    write_precision='s')
                    n_written += len(chunk)
            except Exception as err:
                raise RuntimeError(
                    f"Upload of {field[0]} to bucket {to_bucket} failed after {n_written} of "
                    f"{n_points} points: {str(err).strip()}.{deleted_msg} Re-run the upload.") from err

        success(f"Upload finished: {n_points} points of {field[0]} written to bucket {to_bucket}.",
                verbose=self.verbose)

    def download(self,
                 bucket: str,
                 start: str,
                 stop: str,
                 timezone_offset_to_utc_hours: int | float,
                 data_version: list | str | None = None,
                 measurements: list | None = None,
                 fields: list | None = None,
                 verify_freq: str | bool = False) -> tuple[DataFrame, dict, dict]:
        """
        Get data from database between 'start' and 'stop' dates

        The exact 'stop' date is NOT included.

        The args 'start' and 'stop' dates are in relation to the same timezone
        as defined in 'timezone_offset_to_utc_hours'. This means that if the offset
        is e.g. '1' for CET, then the dates are also understood to be in the same
        timezone, e.g. CET.

        A variable stored with several tag sets (e.g. a changed ``gain``) comes back
        as several Flux tables, which the client may put into one DataFrame. Each
        DataFrame is split by its ``table`` column first. At shared timestamps the table that starts later wins,
        i.e. the newer setup. Tables that start at the same time are ordered by their
        tag values (as strings, in :data:`~diive.core.io.db.influx.common.TAGS`
        order). A warning reports timestamps where the tables hold different values.

        Args:
            bucket: name of bucket in database
            measurements: list of measurements in database, e.g. ['TA', 'SW']
            fields: list of fields (variable names)
            start: start date, e.g. '2022-07-04 00:30:00'
            stop: stop date, e.g. '2022-07-05 12:00:00'
            timezone_offset_to_utc_hours: convert the UTC timestamp from the
                database to this timezone offset, e.g. if '1' then data are downloaded
                and returned with the timestamp 'UTC+01:00', i.e. UTC + 1 hour, which
                corresponds to CET (winter time)
            data_version: version ID of the data that should be downloaded,
                e.g. ['meteoscreening']. If given as a string it is converted to a list
                with the string as the list element.
            verify_freq: checks if the downloaded data has the expected frequency, given
                as str in the format of pandas frequency strings, e.g., '30min' for 30-minute
                data. If the inferred frequency does not match, a warning is printed.
        """
        if isinstance(data_version, str):
            data_version = [data_version]

        fields_str = fields if fields else "ALL"
        measurements_str = measurements if measurements else "ALL"
        info(f"DOWNLOADING from bucket {bucket}: variables {fields_str} "
             f"from measurements {measurements_str}, data version {data_version}, "
             f"between {start} and {stop} (timezone offset to UTC {timezone_offset_to_utc_hours})",
             verbose=self.verbose)

        # InfluxDB needs ISO 8601 date format (in requested timezone) for query
        start_iso = self._convert_datestr_to_iso8601(datestr=start,
                                                     timezone_offset_to_utc_hours=timezone_offset_to_utc_hours)
        stop_iso = self._convert_datestr_to_iso8601(datestr=stop,
                                                    timezone_offset_to_utc_hours=timezone_offset_to_utc_hours)

        # Assemble query
        bucketstring = fluxql.bucketstring(bucket=bucket)
        rangestring = fluxql.rangestring(start=start_iso, stop=stop_iso)

        # Measurements
        if measurements:
            measurementstring = fluxql.filterstring(queryfor='_measurement', querylist=measurements, logic='or')
        else:
            measurementstring = ''  # Empty means all measurements

        # Fields
        if fields:
            fieldstring = fluxql.filterstring(queryfor='_field', querylist=fields, logic='or')
        else:
            fieldstring = ''  # Empty means all fields

        pivotstring = fluxql.pivotstring()

        if data_version:
            dataversionstring = fluxql.filterstring(queryfor='data_version', querylist=data_version, logic='or')
            querystring = f"{bucketstring} {rangestring} {measurementstring} " \
                          f"{dataversionstring} {fieldstring} {pivotstring}"
        else:
            querystring = f"{bucketstring} {rangestring} {measurementstring} " \
                          f"{fieldstring} {pivotstring}"

        # Run database query
        tables = self._query_df(querystring)  # DataFrame or list of DataFrames
        detail(f"Used querystring: {querystring}", verbose=self.verbose)

        # In case only one single variable is downloaded, the query returns
        # a single dataframe. If multiple variables are downloaded, the query
        # returns a list of dataframes. To keep these two options consistent,
        # single dataframes are converted to a list, in which case the list
        # contains only one element: the dataframe of the single variable.
        tables = [tables] if not isinstance(tables, list) else tables

        # The client puts all tables with the same columns into one DataFrame,
        # so two tag sets of one variable can arrive together. Split them first,
        # so the merge rule below sees every tag set as its own table (L299).
        tables = [series for t in tables for series in self._split_series(t)]

        # No data in the requested range: query_data_frame returns an empty
        # DataFrame (without the expected columns). Return empty results in that
        # case instead of raising when accessing '_measurement' below.
        tables = [t for t in tables if not t.empty and '_measurement' in t.columns]
        if not tables:
            info("No data found for the requested query.", verbose=self.verbose)
            return DataFrame(), {}, {}

        # Each table in tables contains data for one variable
        found_measurements = []
        tables_per_var = {}  # Var name -> tables of that var (one per tag set)
        data_detailed = {}  # Stores variables and their tags
        data_simple = DataFrame()  # Stores variables
        for ix, table in enumerate(tables):

            found_measurement = list(set(table['_measurement'].tolist()))
            if len(found_measurement) != 1:
                raise ValueError(f"Found {len(found_measurement)} measurements, but only one allowed")
            found_measurements.append(found_measurement[0])

            # Queries are always returned w/ UTC timestamp
            # Create timestamp columns
            table.rename(columns={"_time": "TIMESTAMP_UTC_END"}, inplace=True)
            table['TIMESTAMP_END'] = table['TIMESTAMP_UTC_END'].copy()

            # TIMEZONE: convert timestamp index to required timezone
            table['TIMESTAMP_END'] = convert_ts_to_timezone(
                timezone_offset_to_utc_hours=timezone_offset_to_utc_hours,
                timestamp_index=table['TIMESTAMP_END'])

            # Remove timezone info in timestamp from TIMESTAMP_END
            # -> download clean timestamp without timestamp info
            table['TIMESTAMP_END'] = table['TIMESTAMP_END'].dt.tz_localize(None)  # Timezone!

            # Set TIMESTAMP_END as the main index
            table.set_index("TIMESTAMP_END", inplace=True)
            table.sort_index(inplace=True, kind='stable')

            # Remove duplicated timestamps within this one table (one tag set),
            # keeping the last entry. Duplicates across tables (e.g. an old and a
            # new logger file with different tags) are merged further below.
            table = table[~table.index.duplicated(keep='last')]

            # Remove timezone info from UTC timestamp, header already states it's UTC
            table['TIMESTAMP_UTC_END'] = table['TIMESTAMP_UTC_END'].dt.tz_localize(None)  # Timezone!

            # Detect of which variable the frame contains data
            # Here it is useful that the variable name is also available as tag 'varname'.
            list_of_fields = list(set(table['varname'].tolist()))

            # Current table must contain one single variable name
            if len(list_of_fields) != 1:
                raise ValueError(f"Expected one field, got {list_of_fields}")

            field_in_table = list_of_fields[0]
            key = field_in_table

            # Keep all columns that are either the field or database tags
            keepcols = [col for col in table.columns if col in TAGS]
            keepcols.append(key)
            table = table[keepcols].copy()

            # Variables with different sets of tags are downloaded in their own
            # table, so one var name can come in several tables.
            tables_per_var.setdefault(key, []).append(table)

        # Merge the tables of each variable by start time, then tag values. At
        # shared timestamps the later table wins, i.e. the newer setup. Influx's
        # table order is not used, so the result does not depend on it.
        for key, var_tables in tables_per_var.items():
            var_tables = sorted(var_tables, key=self._tag_sort_key)
            merged = var_tables[0]
            for table in var_tables[1:]:
                self._warn_conflicting_tables(merged, table, key)
                # Whole rows, so the tags always belong to the kept value.
                keep_old = merged.index.difference(table.index[table[key].notna()])
                merged = pd.concat([merged.loc[keep_old],
                                    table.drop(index=table.index.intersection(keep_old))]).sort_index()
            data_detailed[key] = merged

            # Collect variables without tags in a separate (simplified) dataframe.
            # This dataframe only contains the timestamp and the data column of each var.
            data_simple = data_simple.combine_first(pd.DataFrame(merged[key]))

        # Info
        info(f"Downloaded data for {len(data_detailed)} variables:", verbose=self.verbose)
        for key, frame in data_detailed.items():
            # Detected from the timestamps, with the freq tag next to it: a wrong tag
            # shows here already, before the screening warns about it.
            tags = ', '.join(str(v) for v in pd.unique(frame['freq'].dropna())) if 'freq' in frame else ''
            tag_text = f" (freq tag: {tags})" if tags else " (no freq tag)"
            info(f"<-- {key}: {len(frame):,} records from {frame.index[0]} to {frame.index[-1]}, "
                 f"time resolution {describe_time_resolution(frame.index)}{tag_text}",
                 verbose=self.verbose)

        if not measurements:
            found_measurements = list(set(found_measurements))
            measurements = found_measurements

        assigned_measurements = self._detect_measurement_for_field(bucket=bucket,
                                                                   measurementslist=measurements,
                                                                   varnameslist=list(data_detailed.keys()))

        if verify_freq and not data_simple.empty:
            self._verify_freq(data_index=data_simple.index, expected_freq=verify_freq)

        return data_simple, data_detailed, assigned_measurements

    @staticmethod
    def _split_series(frame: DataFrame) -> list[DataFrame]:
        """Split a query result frame into one frame per Flux table.

        ``query_data_frame`` starts a new DataFrame only when the columns change,
        so one frame can hold several tables (series). The ``table`` column holds
        the Flux table id. A frame without that column is split by its
        measurement and tag values.
        """
        if frame.empty:
            return [frame]
        if 'table' in frame.columns:
            by = ['table']
        else:
            by = [c for c in frame.columns if c == '_measurement' or c in TAGS]
        if not by:
            return [frame]
        return [group.copy() for _, group in frame.groupby(by, sort=True, dropna=False)]

    @staticmethod
    def _tag_sort_key(table: DataFrame) -> tuple:
        """Sort key of a downloaded table: its first timestamp, then its tag values as strings."""
        return (table.index.min(),) + tuple(
            str(table[tag].iloc[0]) if tag in table.columns else '' for tag in TAGS)

    def _warn_conflicting_tables(self, merged: DataFrame, table: DataFrame, key: str) -> None:
        """Warn if *table* and *merged* hold different values of *key* at shared timestamps."""
        shared = merged.index.intersection(table.index)
        old = merged.loc[shared, key]
        new = table.loc[shared, key]
        conflict = old.notna() & new.notna() & (old != new)
        if not conflict.any():
            return
        stamps = conflict.index[conflict]
        tagdiffs = []
        for tag in TAGS:
            old_vals = sorted(set(merged.loc[stamps, tag].astype(str))) if tag in merged.columns else ['<none>']
            new_vals = sorted(set(table.loc[stamps, tag].astype(str))) if tag in table.columns else ['<none>']
            if old_vals != new_vals:
                tagdiffs.append(f"{tag} {new_vals} kept over {old_vals}")
        warn(f"{key}: {len(stamps)} timestamps hold different values in tables with different "
             f"tags ({'; '.join(tagdiffs)}). Kept the values of the table that starts later.",
             verbose=self.verbose)

    def _verify_freq(self, data_index: pd.DatetimeIndex, expected_freq: str) -> None:
        """Warn if the inferred frequency of the data does not match *expected_freq*."""
        inferred_freq = pd.infer_freq(data_index)
        if inferred_freq is None:
            warn("Could not infer a frequency from the downloaded data "
                 f"(expected '{expected_freq}').", verbose=self.verbose)
        elif inferred_freq != expected_freq:
            warn(f"Inferred data frequency '{inferred_freq}' does not match "
                 f"expected frequency '{expected_freq}'.", verbose=self.verbose)
        else:
            info(f"Verified data frequency: '{inferred_freq}'.", verbose=self.verbose)

    def delete(self,
               bucket: str,
               measurements: list | bool,
               start: str,
               stop: str,
               timezone_offset_to_utc_hours: int | float,
               data_version: str,
               fields: list | bool,
               site: str | None = None) -> None:
        """
        Delete data from bucket

        Args:
            bucket: name of bucket in database
            measurements: list or True
                If list, list of measurements in database, e.g. ['TA', 'SW']
                If True, all *fields* in all *measurements* will be deleted.
                The measurement list is then looked up in the database, restricted
                to *data_version*; a ValueError is raised if that lookup finds
                nothing, so a delete can never report success without acting.
            fields: list or True
                If list, list of fields (variable names) to delete
                If True, all data in *fields* in *measurements* will be deleted.
            start: start datetime, e.g. '2022-07-04 00:30:00'
            stop: stop datetime, e.g. '2022-07-05 12:00:00'
                The delete range includes both bounds: InfluxDB's storage engine
                deletes everything in ``[start, stop]``. Note this differs from a
                Flux ``range(start:, stop:)`` *query*, which excludes *stop* - the
                same pair of bounds therefore covers one more record when deleting
                than when reading the data back.
            timezone_offset_to_utc_hours: the timezone of *start* and *stop* datetimes.
                Necessary because the database always stores data with UTC timestamps.
                For example, if data were originally recorded using CET (winter time),
                which corresponds to UTC+01:00, and all data between 1 Jun 2024 00:30 CET and
                2 Jun 2024 12:00 CET should be deleted, then *timezone_offset_to_utc_hours=1*.
            data_version: version ID of the data that should be deleted,
                e.g. 'meteoscreening_diive', 'raw', 'myID', ...
            site: if given, only data with this ``site`` tag are deleted, e.g. 'CH-DAV'.
                ``None`` (default) deletes regardless of site.

        Examples:

            Delete all variables across all measurements:
                measurements=True, fields=True

            Delete all variables of a specific measurement:
                measurements=['TA'], fields=True

            Delete specific variables in specific measurements:
                measurements=['TA', 'SW'], fields=['TA_T1_1_1', 'SW_T1_1_1']

            Delete specific variables across all measurements:
                measurements=True, fields=['TA_T1_1_1', 'SW_T1_1_1']
                This basically searches the variables across all measurements
                and then deletes them.

        docs:
        - https://influxdb-client.readthedocs.io/en/stable/usage.html#delete-data
        - https://docs.influxdata.com/influxdb/v2/reference/syntax/delete-predicate/
        """
        # Validate destructive inputs up front. Anything falsy-but-not-True
        # (None, False, empty list) is rejected so we never silently delete
        # nothing while logging a successful deletion, and never iterate over a
        # non-iterable bool.
        if not measurements:
            raise ValueError("`measurements` must be a non-empty list of measurement "
                             "names or True (all measurements).")
        if not fields:
            raise ValueError("`fields` must be a non-empty list of field names or "
                             "True (all fields).")

        # InfluxDB needs ISO 8601 date format (in requested timezone) for query
        start_iso = self._convert_datestr_to_iso8601(datestr=start,
                                                     timezone_offset_to_utc_hours=timezone_offset_to_utc_hours)
        stop_iso = self._convert_datestr_to_iso8601(datestr=stop,
                                                    timezone_offset_to_utc_hours=timezone_offset_to_utc_hours)

        # Check if measurements is boolean and True
        measurements_all = False
        if measurements and isinstance(measurements, bool):
            # Scoped to *data_version*: measurements holding no data of this
            # version have nothing to delete anyway, and the version-filtered
            # query looks at the full history instead of the schema default.
            measurements = self.show_measurements_in_bucket(bucket=bucket, data_version=data_version,
                                                            verbose=False)
            measurements_all = True
            if not measurements:
                raise ValueError(
                    f"No measurements found for data_version '{data_version}' in bucket "
                    f"'{bucket}'. Nothing was deleted. Check the bucket and data version "
                    f"names, e.g. with show_data_versions_in_bucket(bucket='{bucket}').")

        # Run database query
        with get_client(self.conf_db) as client:
            delete_api = get_delete_api(client)

            # Delete
            kwargs = dict(start=start_iso, stop=stop_iso, bucket=bucket)
            site_str = f' AND site="{site}"' if site is not None else ''
            for measurement in measurements:

                # Delete all variables (fields) in measurement
                if fields and isinstance(fields, bool):
                    predicate_str = (f'_measurement="{measurement}" '
                                     f'AND data_version="{data_version}"'
                                     f'{site_str}')
                    delete_api.delete(predicate=predicate_str, **kwargs)

                # Delete given variables (fields) in measurement
                elif isinstance(fields, list):
                    for field in fields:
                        predicate_str = (f'_measurement="{measurement}" '
                                         f'AND varname="{field}" '
                                         f'AND data_version="{data_version}"'
                                         f'{site_str}')
                        delete_api.delete(predicate=predicate_str, **kwargs)

        # Report the measurements actually targeted, not the requested `True`,
        # so a delete that hit nothing cannot read like a successful wipe.
        if measurements_all:
            measurements_str = f"ALL ({len(measurements)}) {measurements}"
        elif isinstance(measurements, list):
            measurements_str = measurements
        else:
            measurements_str = None

        if fields and isinstance(fields, bool):
            fields_str = "ALL"
        elif isinstance(fields, list):
            fields_str = fields
        else:
            fields_str = None

        site_msg = f" for site {site}" if site is not None else ""
        info(f"Deleted variables {fields_str}{site_msg} between {start_iso} and {stop_iso} "
             f"from measurements {measurements_str} in bucket {bucket}.", verbose=self.verbose)

    def show_configs_unitmapper(self) -> dict:
        return self.conf_unitmapper

    def show_configs_dirs(self) -> dict:
        return self.conf_dirs

    def show_configs_filetypes(self) -> dict:
        return self.conf_filetypes

    def show_config_for_filetype(self, filetype: str) -> dict:
        return self.conf_filetypes[filetype]

    def show_fields_in_measurement(self, bucket: str, measurement: str,
                                   data_version: str | None = None, days: int = 9999,
                                   verbose: bool = True) -> list:
        """Show fields (variable names) in measurement, optionally narrowed to one
        *data_version*."""
        if data_version:
            query = fluxql.tag_values(
                bucket=bucket, tag='_field',
                conditions={'_measurement': measurement, 'data_version': data_version},
                days=days)
        else:
            query = fluxql.fields_in_measurement(bucket=bucket, measurement=measurement, days=days)
        fieldslist = self._values_from_query(query)
        if verbose:
            info(f"Fields in measurement {measurement} of bucket {bucket}:", verbose=self.verbose)
            for ix, f in enumerate(fieldslist, 1):
                detail(f"#{ix}  {bucket}  {measurement}  {f}", verbose=self.verbose)
            info(f"Found {len(fieldslist)} fields in measurement {measurement} "
                 f"of bucket {bucket}.", verbose=self.verbose)
        return fieldslist

    def show_fields_in_bucket(self, bucket: str, measurement: str | None = None,
                              verbose: bool = True) -> list:
        """Show fields (variable names) in bucket (optional: for specific measurement)"""
        if measurement is not None:
            return self.show_fields_in_measurement(bucket=bucket, measurement=measurement, verbose=verbose)
        query = fluxql.fields_in_bucket(bucket=bucket)
        fieldslist = self._values_from_query(query)
        if verbose:
            info(f"Fields in bucket {bucket}:", verbose=self.verbose)
            for ix, f in enumerate(fieldslist, 1):
                detail(f"#{ix}  {bucket}  {f}", verbose=self.verbose)
            info(f"Found {len(fieldslist)} variables (fields) in bucket {bucket}.", verbose=self.verbose)
        return fieldslist

    def show_measurements_in_bucket(self, bucket: str, data_version: str | None = None,
                                    verbose: bool = True) -> list:
        """Show measurements in bucket, optionally narrowed to one *data_version*."""
        if data_version:
            query = fluxql.tag_values(
                bucket=bucket, tag='_measurement', conditions={'data_version': data_version})
        else:
            query = fluxql.measurements_in_bucket(bucket=bucket)
        measurements = self._values_from_query(query)
        if verbose:
            info(f"Measurements in bucket {bucket}:", verbose=self.verbose)
            for ix, m in enumerate(measurements, 1):
                detail(f"#{ix}  {bucket}  {m}", verbose=self.verbose)
            info(f"Found {len(measurements)} measurements in bucket {bucket}.", verbose=self.verbose)
        return measurements

    def show_data_versions_in_bucket(self, bucket: str, verbose: bool = True) -> list:
        """Show the distinct ``data_version`` tag values stored in *bucket*."""
        query = fluxql.tag_values(bucket=bucket, tag='data_version')
        versions = self._values_from_query(query)
        if verbose:
            info(f"Data versions in bucket {bucket}:", verbose=self.verbose)
            for ix, v in enumerate(versions, 1):
                detail(f"#{ix}  {bucket}  {v}", verbose=self.verbose)
            info(f"Found {len(versions)} data versions in bucket {bucket}.", verbose=self.verbose)
        return versions

    def show_units_in_field(self, bucket: str, measurement: str, field: str,
                            data_version: str | None = None, verbose: bool = True) -> list:
        """Show the distinct ``units`` tag values for *field* of *measurement*,
        optionally narrowed to one *data_version*."""
        conditions = {'_measurement': measurement, 'varname': field}
        if data_version:
            conditions['data_version'] = data_version
        query = fluxql.tag_values(bucket=bucket, tag='units', conditions=conditions)
        units = self._values_from_query(query)
        if verbose:
            info(f"Units for {field} in measurement {measurement} of bucket {bucket}: "
                 f"{units}", verbose=self.verbose)
        return units

    def show_field_overview(self, bucket: str, measurement: str, field: str,
                            data_version: str | None = None,
                            verbose: bool = True) -> dict:
        """Overview of a single *field*: all its tags (incl. units) plus the
        first and last record timestamps.

        Reduces the field's records to one point per series with ``first()`` /
        ``last()`` (see :func:`fluxql.field_records`), then collects, per tag, the
        distinct values across all series, and the overall first / last record
        time (in UTC, as the database stores them).

        Returns a dict ``{bucket, measurement, field, data_version, n_series,
        tags: {tag: [values]}, first: 'YYYY-MM-DD HH:MM:SS'|None, last: ...}``.
        Empty (no records) yields an empty ``tags`` and ``first``/``last`` None.
        """
        overview = {'bucket': bucket, 'measurement': measurement, 'field': field,
                    'data_version': data_version, 'n_series': 0, 'tags': {},
                    'first': None, 'last': None}

        last_df = self._query_one_df(fluxql.field_records(
            bucket=bucket, measurement=measurement, field=field,
            data_version=data_version, reducer='last'))
        if last_df.empty:
            if verbose:
                info(f"No records found for {field} in {measurement} of {bucket}.",
                     verbose=self.verbose)
            return overview

        overview['n_series'] = len(last_df)

        # Per tag, the distinct values seen across all of the field's series
        # (e.g. units that changed over time show up as more than one value).
        tags = {}
        for tag in TAGS:
            if tag in last_df.columns:
                vals = []
                for v in last_df[tag].tolist():
                    sv = '' if v is None else str(v)
                    if sv not in vals:
                        vals.append(sv)
                tags[tag] = vals
        overview['tags'] = tags

        if '_time' in last_df.columns:
            last_ts = pd.to_datetime(last_df['_time']).max()
            overview['last'] = (last_ts.strftime('%Y-%m-%d %H:%M:%S')
                                if pd.notna(last_ts) else None)

        first_df = self._query_one_df(fluxql.field_records(
            bucket=bucket, measurement=measurement, field=field,
            data_version=data_version, reducer='first'))
        if not first_df.empty and '_time' in first_df.columns:
            first_ts = pd.to_datetime(first_df['_time']).min()
            overview['first'] = (first_ts.strftime('%Y-%m-%d %H:%M:%S')
                                 if pd.notna(first_ts) else None)

        if verbose:
            info(f"Overview for {field} in {measurement} of {bucket}: "
                 f"{overview['n_series']} series, records {overview['first']} "
                 f"to {overview['last']} (UTC).", verbose=self.verbose)
        return overview

    def show_buckets(self) -> list:
        """Show all (non-system) buckets in the database"""
        query = fluxql.buckets()
        results = self._query_df(query)
        if results.empty or 'name' not in results.columns:
            return []
        results = results.drop(columns=[c for c in ('result', 'table') if c in results.columns])
        bucketlist = results['name'].tolist()
        bucketlist = [x for x in bucketlist if not x.startswith('_')]
        for ix, b in enumerate(bucketlist, 1):
            detail(f"#{ix}  {b}", verbose=self.verbose)
        info(f"Found {len(bucketlist)} buckets in database.", verbose=self.verbose)
        return bucketlist

    def _read_configs(self):
        """Read all YAML configuration files from *dirconf* and its secret sibling.

        See :mod:`diive.core.io.db.influx.config` for the expected directory
        layout. The database connection config is read from the sibling directory
        ``<dirconf>_secret/dbconf.yaml`` so secrets can be kept out of the (often
        version-controlled) main config directory.

        Only ``dbconf.yaml`` is required. A missing ``units.yaml``, ``dirs.yaml``
        or ``filegroups/`` gives an empty dict: download and upload do not use
        them.
        """
        # Config locations
        _dir_filegroups = self.dirconf / 'filegroups'
        _file_unitmapper = self.dirconf / 'units.yaml'
        _file_dirs = self.dirconf / 'dirs.yaml'
        _file_dbconf = Path(f"{self.dirconf}_secret") / 'dbconf.yaml'

        # Read configs
        conf_filetypes = get_conf_filetypes(folder=_dir_filegroups)
        conf_unitmapper = self._read_optional_configfile(_file_unitmapper)
        conf_dirs = self._read_optional_configfile(_file_dirs)
        conf_db = read_dbconf(config_file=_file_dbconf)
        info("Reading configuration files was successful.", verbose=self.verbose)
        return conf_filetypes, conf_unitmapper, conf_dirs, conf_db

    def _read_optional_configfile(self, config_file: Path) -> dict:
        """Read a passthrough config file, or return {} if it is missing or empty."""
        if not config_file.is_file():
            detail(f"Optional config file not found, skipped: {config_file}", verbose=self.verbose)
            return {}
        return read_configfile(config_file=config_file) or {}

    def _query_df(self, query: str):
        """Run a Flux query and return the result as a DataFrame (or list of them).

        Opens a client for the duration of the query and closes it afterwards,
        even if the query raises.
        """
        with get_client(self.conf_db) as client:
            query_api = get_query_api(client)
            return query_api.query_data_frame(query=query)

    def _query_one_df(self, query: str):
        """Run *query* and return a single DataFrame (concatenated if the client
        returns one frame per series; empty DataFrame if there are no rows)."""
        res = self._query_df(query)
        if isinstance(res, list):
            frames = [f for f in res if not f.empty]
            return pd.concat(frames, ignore_index=True) if frames else DataFrame()
        return res

    def _values_from_query(self, query: str) -> list:
        """Run *query* and return its ``_value`` column as a list ([] if empty).

        The schema helper queries (fields / measurements / tag values) all return
        a single ``_value`` column; an empty range yields an empty DataFrame
        without that column, which must not raise.
        """
        results = self._query_df(query)
        if isinstance(results, list):
            # Defensive: schema queries return a single frame, but guard anyway.
            values = []
            for frame in results:
                if not frame.empty and '_value' in frame.columns:
                    values.extend(frame['_value'].tolist())
            return values
        if results.empty or '_value' not in results.columns:
            return []
        return results['_value'].tolist()

    def test_connection(self) -> None:
        """Verify the database is reachable; raise :class:`ConnectionError` if not."""
        with get_client(self.conf_db) as client:
            if not client.ping():
                raise ConnectionError(
                    "Could not connect to the InfluxDB database (ping failed). "
                    "Check the connection settings in dbconf.yaml (url, org, token).")
        success("Connection to database works.", verbose=self.verbose)

    @staticmethod
    def _convert_datestr_to_iso8601(datestr: str, timezone_offset_to_utc_hours: int | float) -> str:
        """Convert date string to ISO 8601 format (needed for the InfluxDB query).

        InfluxDB stores data in UTC (same as GMT). The start/stop range is given
        in relation to the timezone the data should be in: e.g. to download data
        in CET, specify the range in CET.

        e.g. with ``timezone_offset_to_utc_hours=1`` the datestr
        ``'2022-05-27 00:00:00'`` becomes ``'2022-05-27T00:00:00+01:00'``, which
        corresponds to CET (winter time, without daylight savings).
        """
        _isostr = pd.Timestamp(datestr).isoformat()
        isostr_influx = f"{_isostr}{InfluxIO._format_utc_offset(timezone_offset_to_utc_hours)}"
        return isostr_influx

    def _detect_measurement_for_field(self, bucket: str, measurementslist: list, varnameslist: list) -> dict:
        """Detect measurement group of variable.

        Helper because the Flux query does not return the measurement group of
        the field. Used e.g. in diive meteoscreening, where the measurement group
        is important.

        Args:
            bucket: name of database bucket, e.g. "ch-dav_raw"
            measurementslist: list of measurements, e.g. ['TA', 'SW', 'LW']
            varnameslist: list of variable names, e.g. [TA_T1_35_1, SW_IN_T1_35_1]
        """
        assigned_measurements = {}
        for m in measurementslist:
            fieldslist = self.show_fields_in_measurement(bucket=bucket, measurement=m, verbose=False)
            for var in varnameslist:
                if var in fieldslist:
                    assigned_measurements[var] = m
        return assigned_measurements
