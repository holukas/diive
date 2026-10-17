"""
METEO: METEOROLOGICAL DATA FORMAT CONVERSION
=============================================

Format meteorological data as EddyPro flux-processing input and for FLUXNET upload.

Part of the diive library: https://github.com/holukas/diive
"""

from pathlib import Path

import pandas as pd

from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import current_date_str_condensed
from diive.core.times.times import format_timestamp_to_fluxnet_format
from diive.core.times.times import insert_timestamp
from diive.core.utils.console import info


class FormatMeteoForEddyProFluxProcessing:
    """Reformat a 30-min meteo DataFrame into the EddyPro flux-processing input layout."""

    # Timestamp columns for EddyPro
    colname_timestamp1 = ('TIMESTAMP_1', 'yyyy-mm-dd')
    colname_timestamp2 = ('TIMESTAMP_2', 'HH:MM')

    def __init__(self, df: pd.DataFrame, cols: dict):
        """Store the data and the ``{old: (new, unit)}`` column-renaming map."""
        self._df = df.copy()
        self.cols = cols

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe reformatted for EddyPro."""
        if not isinstance(self._df, pd.DataFrame):
            raise Exception('No reformatted data available.')
        return self._df

    def get_results(self):
        """Return the reformatted DataFrame."""
        return self.df

    def run(self):
        """Sanitize timestamps, split date/time, fill gaps with -9999 and rename columns."""
        self._df = self._sanitize_timestamp()
        self._df = self._split_timestamp_date_time()

        info("Filling missing values with -9999 ...")
        self._df = self._df.fillna(-9999)

        self._df = self._rename_columns()

    def _rename_columns(self):
        info("Renaming columns ...")
        from diive.core.dfun.frames import rename_cols_to_multiindex
        df = self._df.copy()
        df = rename_cols_to_multiindex(df=df, renaming_dict=self.cols)
        return df

    def _split_timestamp_date_time(self):
        """Split timestamp into separate date and time columns.
        Timestamp column names are stored as tuples.
        """
        info(f"Splitting timestamp into two separate columns {self.colname_timestamp1} and {self.colname_timestamp2}")
        df = self._df.copy()
        df[self.colname_timestamp2] = df.index
        first_column = df.pop(self.colname_timestamp2)
        df.insert(0, self.colname_timestamp2, first_column)
        df[self.colname_timestamp2] = pd.to_datetime(df[self.colname_timestamp2])
        df[self.colname_timestamp2] = df[self.colname_timestamp2].dt.strftime('%H:%M')

        df[self.colname_timestamp1] = df.index.date
        first_column = df.pop(self.colname_timestamp1)
        df.insert(0, self.colname_timestamp1, first_column)
        df[self.colname_timestamp1] = pd.to_datetime(df[self.colname_timestamp1])
        df[self.colname_timestamp1] = df[self.colname_timestamp1].dt.strftime('%Y-%m-%d')
        return df

    def _sanitize_timestamp(self):
        tss = TimestampSanitizer(
            data=self._df,
            output_middle_timestamp=False,
            nominal_freq="30min",
            verbose=True
        )
        df = tss.get()
        return df


class FormatMeteoForFluxnetUpload:
    """Reformat a 30-min meteo DataFrame into the FLUXNET upload layout (START/END timestamps)."""

    # Timestamp columns for EddyPro
    colname_timestamp1 = ('TIMESTAMP_1', 'yyyy-mm-dd')
    colname_timestamp2 = ('TIMESTAMP_2', 'HH:MM')

    def __init__(self, df: pd.DataFrame, cols: dict):
        """Store the data and the ``{old: new}`` column-renaming map."""
        self._df = df.copy()
        self.cols = cols

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe reformatted for FLUXNET upload."""
        if not isinstance(self._df, pd.DataFrame):
            raise Exception('No reformatted data available.')
        return self._df

    def get_results(self):
        """Return the reformatted DataFrame."""
        return self.df

    def run(self):
        """Sanitize timestamps, insert START/END columns, fill gaps with -9999 and rename columns."""
        self._df = self._sanitize_timestamp()
        self._df = self._insert_timestamps_start_end()

        info("Filling missing values with -9999 ...")
        self._df = self._df.fillna(-9999)

        self._df = self._rename_columns()

    def export_yearly_files(self, site: str, outdir: str):
        """Create one file per year"""
        self._save_one_file_per_year(df=self.df, site=site, outdir=outdir)

    @staticmethod
    def _save_one_file_per_year(df: pd.DataFrame, site: str, outdir: str, add_runid: bool = True):
        """Save data to yearly files"""
        info(f"\nSaving yearly CSV files ...")
        uniq_years = list(df.index.year.unique())
        runid = f"_{current_date_str_condensed()}" if add_runid else ""
        for year in uniq_years:
            outname = f"{site}_{year}_fluxes_meteo{runid}.csv"
            outpath = Path(outdir) / outname
            yearlocs = df.index.year == year
            yeardata = df[yearlocs].copy()
            yeardata.to_csv(outpath, index=False)
            info(f"    --> Saved file {outpath}.")

    def _rename_columns(self):
        info("Renaming columns ...")
        df = self._df.copy()
        df = df.rename(columns=self.cols)
        return df

    def _insert_timestamps_start_end(self):
        """Insert two timestamp columns 'TIMESTAMP_START' and 'TIMESTAMP_END' as
        required for FLUXNET data submissions.
        """
        info(f"Inserting timestamp columns 'TIMESTAMP_START' and 'TIMESTAMP_END' ... ")
        df = self._df.copy()
        df = insert_timestamp(data=df, convention='end', insert_as_first_col=True, verbose=True)
        df = insert_timestamp(data=df, convention='start', insert_as_first_col=True, verbose=True)
        df['TIMESTAMP_END'] = format_timestamp_to_fluxnet_format(df=df, timestamp_col='TIMESTAMP_END')
        df['TIMESTAMP_START'] = format_timestamp_to_fluxnet_format(df=df, timestamp_col='TIMESTAMP_START')
        return df

    def _sanitize_timestamp(self):
        tss = TimestampSanitizer(
            data=self._df,
            output_middle_timestamp=True,
            nominal_freq="30min",
            verbose=True
        )
        df = tss.get()
        return df
