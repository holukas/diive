import pandas as pd

from diive.core.dfun.frames import rename_cols_to_multiindex
from diive.core.times.times import TimestampSanitizer


class FormatMeteoForEddyProProcessing:

    # Timestamp columns for EddyPro
    colname_timestamp1 = ('TIMESTAMP_1', 'yyyy-mm-dd')
    colname_timestamp2 = ('TIMSTAMP_2', 'HH:MM')

    def __init__(self, df: pd.DataFrame, cols: dict):
        self.df = df.copy()
        self.cols = cols

    def run(self):
        self.df = self._sanitize_timestamp()
        self.df = self._split_timestamp_date_time()
        self.df = self.df.fillna(-9999)
        self.df = self._rename_columns()

    def _rename_columns(self):
        df = self.df.copy()
        df = rename_cols_to_multiindex(df=df, renaming_dict=self.cols)
        return df


    def _split_timestamp_date_time(self):
        """Split timestamp into separate date and time columns.
        Timestamp column names are stored as tuples.
        """
        df = self.df.copy()
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
            data=self.df,
            output_middle_timestamp=False,
            nominal_freq="30min"
        )
        df = tss.get()
        return df


def example():
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
    
    f = FormatMeteoForEddyProProcessing(
        df=data_simple,
        cols=rename_dict
    )
    f.run()


    # df.to_csv(r"F:\TMP\del.csv", index=False)
    # print(df)


if __name__ == "__main__":
    example()
