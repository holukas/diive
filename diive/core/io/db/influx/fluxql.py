"""
CORE.IO.DB.INFLUX.FLUXQL: FLUX QUERY STRING BUILDERS
====================================================

Helpers that build Flux query strings for InfluxDB v2.

Note:
    These helpers interpolate the given bucket, measurement, field, tag and
    data-version names directly into the query string without escaping. They
    assume *trusted* input (names originate from the local config files / the
    caller, not from untrusted external sources). Do not pass unsanitised user
    input.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def pivotstring() -> str:
    return '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'


def bucketstring(bucket: str) -> str:
    return f'from(bucket: "{bucket}")'


def rangestring(start: str, stop: str) -> str:
    return f'|> range(start: {start}, stop: {stop})'


def filterstring(queryfor: str, querylist: list, logic: str) -> str:
    """Build a Flux ``filter()`` that matches *queryfor* against any value in
    *querylist*, combined with the given *logic* operator (e.g. ``'or'``)."""
    filterstring = ''  # Query string
    for ix, var in enumerate(querylist):
        if ix == 0:
            filterstring += f'|> filter(fn: (r) => r["{queryfor}"] == "{var}"'
        else:
            filterstring += f' {logic} r["{queryfor}"] == "{var}"'
    filterstring = f"{filterstring})"  # Needs bracket at end
    return filterstring


def fields_in_measurement(bucket: str, measurement: str, days: int = 9999) -> str:
    """
    Show all available fields in measurement

    By default, the FluxQL function returns results from the
    last 30d so it is necessary to set the 'start' parameter
    to get ALL fields. Therefore, the start parameter is set
    to -9999d to get all fields available for the last 9999 days.

    Args:
        bucket: bucket name in InfluxDB
        measurement: name of the measurement, e.g. 'TA'
        days: show fields of the last *days* days

    Returns:
        query string for FluxQL
    """
    query = f'''
    import "influxdata/influxdb/schema"
    schema.measurementFieldKeys(
    bucket: "{bucket}",
    measurement: "{measurement}",
    start: -{days}d
    )
    '''
    return query


def fields_in_bucket(bucket: str) -> str:
    query = f'''
    import "influxdata/influxdb/schema"
    schema.fieldKeys(bucket: "{bucket}")
    '''
    return query


def measurements_in_bucket(bucket: str) -> str:
    query = f'''
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{bucket}")
    '''
    return query


def buckets() -> str:
    query = '''
    buckets()
    '''
    return query


def predicatestring(conditions: dict) -> str:
    """Build a Flux predicate function body from tag == value *conditions*.

    e.g. ``{'_measurement': 'TA', 'data_version': 'raw'}`` ->
    ``(r) => r["_measurement"] == "TA" and r["data_version"] == "raw"``.
    Returns the always-true predicate ``(r) => true`` when *conditions* is empty.
    """
    if not conditions:
        return '(r) => true'
    parts = [f'r["{tag}"] == "{value}"' for tag, value in conditions.items()]
    return f'(r) => {" and ".join(parts)}'


def field_records(bucket: str, measurement: str, field: str,
                  data_version: str | None = None, reducer: str = 'last',
                  days: int = 9999) -> str:
    """Reduce *field*'s records (raw, un-pivoted) to one point per series.

    Used to build a field overview: filtered to *measurement* / *field* (and
    optionally *data_version*), then reduced with *reducer* (``'last'`` or
    ``'first'``). Each returned row is one series (one unique tag-set), carrying
    all of its tag columns plus ``_time`` / ``_value`` -- so the tags that ever
    applied to the field, and the first / last record time, are recoverable.
    """
    conditions = {'_measurement': measurement, '_field': field}
    if data_version:
        conditions['data_version'] = data_version
    predicate = predicatestring(conditions)
    query = f'''
    from(bucket: "{bucket}")
    |> range(start: -{days}d)
    |> filter(fn: {predicate})
    |> {reducer}()
    '''
    return query


def tag_values(bucket: str, tag: str, conditions: dict | None = None,
               days: int = 9999) -> str:
    """Distinct values of *tag* in *bucket*, optionally narrowed by *conditions*.

    Backs the data-version / units browsing and the data-version-filtered
    measurement / field listings. As with :func:`fields_in_measurement`, the
    start parameter is pushed back ``-{days}d`` so values from the full history
    are returned (the schema functions default to the last 30 days).

    Args:
        bucket: bucket name in InfluxDB
        tag: tag key to list values for, e.g. ``'data_version'``, ``'units'``,
            ``'_measurement'``, ``'_field'``.
        conditions: optional ``{tag: value}`` filter applied via a predicate.
        days: look back this many days.
    """
    predicate = predicatestring(conditions or {})
    query = f'''
    import "influxdata/influxdb/schema"
    schema.tagValues(
    bucket: "{bucket}",
    tag: "{tag}",
    predicate: {predicate},
    start: -{days}d
    )
    '''
    return query
