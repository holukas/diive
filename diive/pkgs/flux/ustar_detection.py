from pandas import DataFrame

from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot
from diive.pkgs.createvar.potentialradiation import potrad


class UstarDetectionMPT:
    """

    USTAR THRESHOLD DETECTION
    =========================

    How thresholds are calculated
    -----------------------------
    - Data are divided into S seasons
    - In each season, data are divided into X air temperature (TA) classes
    - Each air temperature class is divided into Y ustar (USTAR) subclasses (data are binned per subclass)
    - Thresholds are calculated in each TA class
    - Season thresholds per bootstrapping run are calculated from all found TA class thresholds, e.g. the max of thresholds
    - Overall season thresholds are calculated from all season thresholds from all bootstrapping runs, e.g. the median
    - Yearly thresholds are calculated from overall season thresholds

    """

    def __init__(
            self,
            df: DataFrame,
            nee_col: str,
            ta_col: str,
            ustar_col: str,
            season_col: str = None,
            n_bootstraps: int = 100,
            swin_pot_col: str = None,
            nighttime_threshold: float = 20,
            utc_offset: int = None,
            lat: float = None,
            lon: float = None
    ):
        self.df = df.copy()
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.season_col = season_col
        self.n_bootstraps = n_bootstraps
        self.swin_pot_col = swin_pot_col
        self.nighttime_threshold = nighttime_threshold
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset

        # Setup
        if not self.swin_pot_col:
            self.swin_pot_col = self._calc_swin_pot()
        self.flag_dt_col, self.flag_nt_col = self._calc_nighttime_flag()
        self.season_col, self.season_type_col = self._add_season_info()
        self.workdf, self.workdf_dt, self.workdf_nt = self._assemble_work_dataframes()
        # TODO HIER WEITER

    def _add_season_info(self):
        """Add season and season type info to data in case none is available,
        i.e. 'None' was selected in the 'Seasons' drop-down menu.

        This additional info is needed to perform data calculations always the
        same way, regardless if already available seasons are used or no seasons
        are used.
        """

        if not self.season_col:
            # Create season column in case none is available. This basically defines all data
            # as one season and the data pool for threshold calcs is the whole dataset.
            # Doing it this way makes data handling of seasons more harmonized.
            # All data is basically one single season
            season_col = 'GRP_SEASON_0'
            season_type_col = 'GRP_SEASON_TYPE_0'
            self.df[season_col] = 0
            self.df[season_type_col] = 0
            return season_col, season_type_col
        else:
            # The selected column will be used for season grouping.
            pass

        # else:
        #     # Use already available season column.
        #     data_df.loc[:, self.season_type_col] = self.data_df[self.season_type_col].copy()
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_type_col].copy()

        # if self.season_data_pool == 'Season Type':
        #     # Calculate one threshold per season type, e.g. one for all summers. In this case,
        #     # all data from all e.g. summers are first pooled, and then one single threshold is
        #     # calculated based on the pooled data. The threshold is then valid for all summers across
        #     # all years.
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_grouping_col].copy()
        #
        # elif self.season_data_pool == 'Season':
        #     # Calculate threshold for each season, e.g. summer 2018, summer 2019, etc. To differentiate
        #     # between multiple summers, the year is added as additional information so the seasons
        #     # can be grouped later during calcs.
        #     data_df['year_aux'] = data_df.index.year.astype(str)
        #     data_df.loc[:, self.season_data_pool_col] = \
        #         data_df['year_aux'] + '_' + data_df.loc[:, self.season_data_pool_col].astype(str)
        #     data_df.drop('year_aux', axis=1, inplace=True)

        # return data_df

    def _assemble_work_dataframes(self):
        workdf = self.df[[self.nee_col, self.ta_col, self.ustar_col,
                          self.swin_pot_col, self.flag_dt_col, self.flag_nt_col,
                          self.season_col, self.season_type_col]].copy()
        workdf = workdf.dropna()
        is_daytime = workdf[self.flag_dt_col] == 1
        is_nighttime = workdf[self.flag_nt_col] == 1
        workdf_dt = workdf[is_daytime].copy()
        workdf_nt = workdf[is_nighttime].copy()
        return workdf, workdf_dt, workdf_nt

    def _calc_nighttime_flag(self):
        flag_daytime, flag_nighttime = daytime_nighttime_flag_from_swinpot(
            swinpot=self.df[self.swin_pot_col],
            nighttime_threshold=self.nighttime_threshold
        )
        self.df[flag_daytime.name] = flag_daytime
        self.df[flag_nighttime.name] = flag_nighttime
        return flag_daytime.name, flag_nighttime.name

    def _calc_swin_pot(self):
        """Calculate potential radiation or get directly from data"""
        if self.lat and self.lon:
            swin_pot = potrad(timestamp_index=self.df.index,
                              lat=self.lat,
                              lon=self.lon,
                              utc_offset=self.utc_offset)
            swin_pot_col = swin_pot.name
            self.df[swin_pot_col] = swin_pot
            return swin_pot_col
        else:
            raise Exception("Latitude and longitude are required "
                            "if potential radiation is not in data.")


def example():
    from pathlib import Path
    FOLDER = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\notebooks\Workbench\FLUXNET_CH4-N2O_Committee_WP2\data"

    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # filepaths = search_files(FOLDER, "*.csv")
    # filepaths = [fp for fp in filepaths if "_fluxnet_" in fp.stem and fp.stem.endswith("_adv")]
    # print(filepaths)
    # fr = MultiDataFileReader(filetype='EDDYPRO-FLUXNET-30MIN', filepaths=filepaths)
    # df = fr.data_df
    # from diive.core.io.files import save_parquet
    # save_parquet(outpath=FOLDER, filename="data", data=df)

    from diive.core.io.files import load_parquet
    filepath = Path(FOLDER) / 'data.parquet'
    df = load_parquet(filepath=filepath)

    # UstarDetectionMPT(
    #     df=df,
    #     nee_col='FC',
    #     ta_col='TA_1_1_1',
    #     ustar_col='USTAR',
    #     n_bootstraps=10,
    #     swin_pot_col='SW_IN_POT',
    #     nighttime_threshold=20,
    #     utc_offset=1,
    #     lat=47.115833,
    #     lon=8.537778
    # )


if __name__ == '__main__':
    example()
