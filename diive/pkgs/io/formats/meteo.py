from pathlib import Path

import pandas as pd

from diive.core.dfun.frames import rename_cols_to_multiindex
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import current_date_str_condensed
from diive.core.times.times import format_timestamp_to_fluxnet_format
from diive.core.times.times import insert_timestamp


class FormatMeteoForEddyProFluxProcessing:
    # Timestamp columns for EddyPro
    colname_timestamp1 = ('TIMESTAMP_1', 'yyyy-mm-dd')
    colname_timestamp2 = ('TIMESTAMP_2', 'HH:MM')

    def __init__(self, df: pd.DataFrame, cols: dict):
        self._df = df.copy()
        self.cols = cols

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe reformatted for EddyPro."""
        if not isinstance(self._df, pd.DataFrame):
            raise Exception('No reformatted data available.')
        return self._df

    def get_results(self):
        return self.df

    def run(self):
        self._df = self._sanitize_timestamp()
        self._df = self._split_timestamp_date_time()

        print("Filling missing values with -9999 ...")
        self._df = self._df.fillna(-9999)

        self._df = self._rename_columns()

    def _rename_columns(self):
        print("Renaming columns ...")
        df = self._df.copy()
        df = rename_cols_to_multiindex(df=df, renaming_dict=self.cols)
        return df

    def _split_timestamp_date_time(self):
        """Split timestamp into separate date and time columns.
        Timestamp column names are stored as tuples.
        """
        print(f"Splitting timestamp into two separate columns {self.colname_timestamp1} and {self.colname_timestamp2}")
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
    # Timestamp columns for EddyPro
    colname_timestamp1 = ('TIMESTAMP_1', 'yyyy-mm-dd')
    colname_timestamp2 = ('TIMESTAMP_2', 'HH:MM')

    def __init__(self, df: pd.DataFrame, cols: dict):
        self._df = df.copy()
        self.cols = cols

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe reformatted for EddyPro."""
        if not isinstance(self._df, pd.DataFrame):
            raise Exception('No reformatted data available.')
        return self._df

    def get_results(self):
        return self.df

    def run(self):
        self._df = self._sanitize_timestamp()
        self._df = self._insert_timestamps_start_end()

        print("Filling missing values with -9999 ...")
        self._df = self._df.fillna(-9999)

        self._df = self._rename_columns()

    def export_yearly_files(self, site: str, outdir: str):
        """Create one file per year"""
        self._save_one_file_per_year(df=self.df, site=site, outdir=outdir)

    @staticmethod
    def _save_one_file_per_year(df: pd.DataFrame, site: str, outdir: str, add_runid: bool = True):
        """Save data to yearly files"""
        print(f"\nSaving yearly CSV files ...")
        uniq_years = list(df.index.year.unique())
        runid = f"_{current_date_str_condensed()}" if add_runid else ""
        for year in uniq_years:
            outname = f"{site}_{year}_fluxes_meteo{runid}.csv"
            outpath = Path(outdir) / outname
            yearlocs = df.index.year == year
            yeardata = df[yearlocs].copy()
            yeardata.to_csv(outpath, index=False)
            print(f"    --> Saved file {outpath}.")

    def _rename_columns(self):
        print("Renaming columns ...")
        df = self._df.copy()
        df = df.rename(columns=self.cols)
        return df

    def _insert_timestamps_start_end(self):
        """Insert two timestamp columns 'TIMESTAMP_START' and 'TIMESTAMP_END' as
        required for FLUXNET data submissions.
        """
        print(f"Inserting timestamp columns 'TIMESTAMP_START' and 'TIMESTAMP_END' ... ")
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

def _example_FormatMeteoForEddyProFluxProcessing_dataFromParquetFile():
    from diive.core.io.files import load_parquet, save_parquet

    # Name of the variables in the original data file
    SW_IN = 'SW_IN_T1_47_1_gfXG'
    RH = 'RH_T1_47_1'
    PPFD_IN = 'PPFD_IN_T1_47_1_gfXG'
    LW_IN = 'LW_IN_T1_47_1'
    TA = 'TA_T1_47_1_gfXG'
    PA = 'PA_T1_47_1'

    # Rename original variables for EddyPro, and add units
    rename_dict = {
        TA: ('Ta_1_1_1', 'C'),
        SW_IN: ('Rg_1_1_1', 'W+1m-2'),
        RH: ('RH_1_1_1', '%'),
        LW_IN: ('Lwin_1_1_1', 'W+1m-2'),
        PA: ('Pa_1_1_1', 'kPa'),
        PPFD_IN: ('PPFD_1_1_1', 'umol+1m-2s-1'),
    }

    # Load data
    SOURCEFILE = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\10_METEO\12.5_METEO7_GAPFILLED_2004-2024.parquet"
    df = load_parquet(filepath=SOURCEFILE)


    f = FormatMeteoForEddyProFluxProcessing(
        df=df,
        cols=rename_dict
    )
    f.run()

    # df.to_csv(r"F:\TMP\del.csv", index=False)
    # print(df)

def _example_FormatMeteoForEddyProFluxProcessing_dataFromDatabase():
    # Download example data from database
    from dbc_influxdb import dbcInflux  # Needed for communicating with the database
    SITE = 'ch-fru'  # Site name
    SW_IN = 'SW_IN_T1_1_1'
    RH = 'RH_T1_2_1'
    PPFD_IN = 'PPFD_IN_T1_2_1'
    LW_IN = 'LW_IN_T1_1_1'
    TA = 'TA_T1_2_1'
    PA = None
    START = '2024-01-01 00:01:00'  # Download data starting with this date
    STOP = '2024-02-01 00:01:00'  # Download data before this date (the stop date itself is not included)
    MEASUREMENTS = ['TA', 'RH', 'SW', 'PPFD', 'LW']
    FIELDS = [TA, RH, SW_IN, LW_IN, PPFD_IN, PA]
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
    data_version = "meteoscreening_diive"
    DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'
    dbc = dbcInflux(dirconf=DIRCONF)
    data_simple, data_detailed, assigned_measurements = \
        dbc.download(bucket=f'{SITE}_processed',
                     measurements=MEASUREMENTS,
                     fields=FIELDS,
                     start=START,
                     stop=STOP,
                     timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
                     data_version='meteoscreening_diive')
    # print(data_simple)

    rename_dict = {
        TA: ('Ta_1_1_1', 'C'),
        SW_IN: ('Rg_1_1_1', 'W+1m-2'),
        RH: ('RH_1_1_1', '%'),
        LW_IN: ('Lwin_1_1_1', 'W+1m-2'),
        # PA: ('Pa_1_1_1', 'kPa),
        PPFD_IN: ('PPFD_1_1_1', 'umol+1m-2s-1'),
    }

    f = FormatMeteoForEddyProFluxProcessing(
        df=data_simple,
        cols=rename_dict
    )
    f.run()

    # df.to_csv(r"F:\TMP\del.csv", index=False)
    # print(df)


def _example_fromDatabase_FormatMeteoForFluxnetUpload():
    # Download example data from database
    from dbc_influxdb import dbcInflux  # Needed for communicating with the database
    SITE = 'ch-fru'  # Site name
    START = '2024-01-01 00:01:00'  # Download data starting with this date
    STOP = '2025-01-01 00:01:00'  # Download data before this date (the stop date itself is not included)
    MEASUREMENTS = ['TA', 'RH', 'SW', 'PPFD', 'LW', 'TS', 'SWC', 'PREC', 'G']
    FIELDS = [
        'TA_T1_2_1',
        # 'RH_T1_2_1',
        # 'SW_IN_T1_1_1',
        # 'SW_OUT_T1_1_1',
        # 'LW_IN_T1_1_1',
        # 'LW_OUT_T1_1_1',
        # 'PPFD_IN_T1_2_1',
        # 'PPFD_OUT_T1_2_1',
        # 'TS_GF1_0.01_1',
        # 'TS_GF1_0.04_1',
        # 'TS_GF1_0.07_1',
        # 'TS_GF1_0.1_2',
        # 'TS_GF1_0.15_1',
        # 'TS_GF1_0.2_2',
        # 'TS_GF1_0.25_1',
        # 'TS_GF1_0.3_2',
        # 'TS_GF1_0.4_1',
        # 'TS_GF1_0.5_2',
        # 'TS_GF1_0.95_1',
        # 'TS_GF1_1_2',
        # 'SWC_GF1_0.05_1',
        # 'SWC_GF1_0.15_1',
        # 'SWC_GF1_0.25_1',
        # 'SWC_GF1_0.4_1',
        # 'SWC_GF1_0.95_1',
        # 'PREC_TOT_GF1_1_1',
        # 'G_GF1_0.06_1',
        # 'G_GF1_0.06_2',
    ]
    TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # Timezone, e.g. "1" is translated to timezone "UTC+01:00" (CET, winter time)
    data_version = "meteoscreening_diive"
    DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'
    dbc = dbcInflux(dirconf=DIRCONF)
    df, _, _ = \
        dbc.download(bucket=f'{SITE}_processed',
                     measurements=MEASUREMENTS,
                     fields=FIELDS,
                     start=START,
                     stop=STOP,
                     timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
                     data_version=data_version)
    # print(data_simple)

    rename_dict = {
        'TA_T1_2_1': 'TA_1_1_1',
        'RH_T1_2_1': 'RH_1_1_1',
        'SW_IN_T1_1_1': 'SW_IN_1_1_1',
        'SW_OUT_T1_1_1': 'SW_OUT_1_1_1',
        'LW_IN_T1_1_1': 'LW_IN_1_1_1',
        'LW_OUT_T1_1_1': 'LW_OUT_1_1_1',
        'PPFD_IN_T1_2_1': 'PPFD_IN_1_1_1',
        'PPFD_OUT_T1_2_1': 'PPFD_OUT_1_1_1',
        'TS_GF1_0.01_1': 'TS_1_1_1',
        'TS_GF1_0.04_1': 'TS_1_2_1',
        'TS_GF1_0.07_1': 'TS_1_3_1',
        'TS_GF1_0.1_2': 'TS_1_4_2',
        'TS_GF1_0.15_1': 'TS_1_5_1',
        'TS_GF1_0.2_2': 'TS_1_6_1',
        'TS_GF1_0.25_1': 'TS_1_7_1',
        'TS_GF1_0.3_2': 'TS_1_8_1',
        'TS_GF1_0.4_1': 'TS_1_9_1',
        'TS_GF1_0.5_2': 'TS_1_10_1',
        'TS_GF1_0.95_1': 'TS_1_11_1',
        'TS_GF1_1_2': 'TS_1_12_1',
        'SWC_GF1_0.05_1': 'SWC_1_1_1',
        'SWC_GF1_0.15_1': 'SWC_1_2_1',
        'SWC_GF1_0.25_1': 'SWC_1_3_1',
        'SWC_GF1_0.4_1': 'SWC_1_4_1',
        'SWC_GF1_0.95_1': 'SWC_1_5_1',
        'PREC_TOT_GF1_1_1': 'P_1_1_1',
        'G_GF1_0.06_1': 'G_1_1_1',
        'G_GF1_0.06_2': 'G_1_1_2'
    }

    f = FormatMeteoForFluxnetUpload(
        df=df,
        cols=rename_dict
    )
    f.run()

    results = f.get_results()

    f.export_yearly_files(site=SITE, outdir=r"F:\TMP")

    # results.to_csv(r"F:\TMP\del.csv", index=False)
    # print(results)


def _example_fromCsvFile_FormatMeteoForFluxnetUpload():
    filepath = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-fru_flux_product\dataset_ch-fru_flux_product\notebooks\10_METEO\13.1_CH-CHA_meteo_2005-2024.csv"
    df = pd.read_csv(filepath)
    df = df.set_index('TIMESTAMP_MIDDLE', inplace=False)
    # [print(c) for c in df.columns];

    rename_dict = {
        'G_GF1_0.03_1': 'G_1_1_1',
        'G_GF1_0.03_2': 'G_1_1_2',
        'G_GF1_0.03_3': 'G_1_1_3',
        'G_GF1_0.03_4': 'G_1_1_4',
        'G_GF1_0.06_1': 'G_1_2_1',
        'G_GF1_0.06_2': 'G_1_2_2',
        'LW_IN_T1_1_1': 'LW_IN_1_1_1',
        'LW_OUT_T1_1_1': 'LW_OUT_1_1_1',
        'PPFD_IN_T1_2_1': 'PPFD_IN_1_1_1',
        'PPFD_OUT_T1_2_1': 'PPFD_OUT_1_1_1',
        'PREC_TOT_GF1_1_1': 'P_1_1_1',
        'RH_T1_2_1': 'RH_1_1_1',
        'SWC_GF1_0.05_1': 'SWC_1_1_1',
        'SWC_GF1_0.05_2': 'SWC_1_1_2',
        'SWC_GF1_0.1_2': 'SWC_1_2_1',
        'SWC_GF1_0.15_1': 'SWC_1_3_1',
        'SWC_GF1_0.2_2': 'SWC_1_4_1',
        'SWC_GF1_0.25_1': 'SWC_1_5_1',
        'SWC_GF1_0.3_2': 'SWC_1_6_1',
        'SWC_GF1_0.4_1': 'SWC_1_7_1',
        'SWC_GF1_0.4_2': 'SWC_1_7_2',
        'SWC_GF1_0.5_2': 'SWC_1_8_1',
        'SWC_GF1_0.6_2': 'SWC_1_9_1',
        'SWC_GF1_0.75_1': 'SWC_1_10_1',
        'SWC_GF1_0.75_2': 'SWC_1_10_2',
        'SWC_GF1_0.95_1': 'SWC_1_11_1',
        'SWC_GF1_1_2': 'SWC_1_12_1',
        'SW_IN_T1_1_1': 'SW_IN_1_1_1',
        'SW_OUT_T1_1_1': 'SW_OUT_1_1_1',
        'TA_T1_2_1': 'TA_1_1_1',
        'TS_GF1_0.01_1': 'TS_1_1_1',
        'TS_GF1_0.02_1': 'TS_1_2_1',
        'TS_GF1_0.04_1': 'TS_1_3_1',
        'TS_GF1_0.05_2': 'TS_1_4_1',
        'TS_GF1_0.07_1': 'TS_1_5_1',
        'TS_GF1_0.1_1': 'TS_1_6_1',
        'TS_GF1_0.1_2': 'TS_1_6_2',
        'TS_GF1_0.15_1': 'TS_1_7_1',
        'TS_GF1_0.2_2': 'TS_1_8_1',
        'TS_GF1_0.25_1': 'TS_1_9_1',
        'TS_GF1_0.3_2': 'TS_1_10_1',
        'TS_GF1_0.4_1': 'TS_1_11_1',
        'TS_GF1_0.4_2': 'TS_1_11_2',
        'TS_GF1_0.5_2': 'TS_1_12_1',
        'TS_GF1_0.6_2': 'TS_1_13_1',
        'TS_GF1_0.75_2': 'TS_1_14_1',
        'TS_GF1_0.95_1': 'TS_1_15_1',
        'TS_GF1_1_2': 'TS_1_16_1',
    }

    f = FormatMeteoForFluxnetUpload(
        df=df,
        cols=rename_dict
    )
    f.run()

    results = f.get_results()

    import matplotlib.pyplot as plt
    import numpy as np
    results = results.replace(-9999, np.nan)
    results.plot(subplots=True, x_compat=True, figsize=(8, 23))
    plt.show()

    f.export_yearly_files(site="CH-Fru", outdir=r"F:\TMP")

    # results.to_csv(r"F:\TMP\del.csv", index=False)
    # print(results)


if __name__ == "__main__":
    _example_FormatMeteoForEddyProFluxProcessing_dataFromParquetFile()
    # _example_FormatMeteoForEddyProFluxProcessing_dataFromDatabase()
    # _example_fromCsvFile_FormatMeteoForFluxnetUpload()
    # _example_fromDatabase_FormatMeteoForFluxnetUpload()
