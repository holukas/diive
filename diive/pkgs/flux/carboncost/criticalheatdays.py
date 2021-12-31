"""

FLUX: CARBON COST
=================

"""

from pathlib import Path

import numpy as np
import pandas as pd

from diive.common.dfun.fits import BinFitterCP
from diive.common.dfun.frames import splitdata_daynight
from diive.common.filereader.filereader import ConfigFileReader, DataFileReader
from diive.pkgs.flux.carboncost import figures

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)


class CriticalHeatDays:
    date_col = ('.DATE', '[aux]')
    grp_daynight_col = ('.GRP_DAYNIGHT', '[aux]')

    def __init__(
            self,
            df: pd.DataFrame,
            x_col: tuple,
            nee_col: tuple,
            gpp_col: tuple,
            reco_col: tuple,
            daytime_col: tuple,
            daytime_threshold: int = 20,
            set_daytime_if: str = 'Larger Than Threshold',
            usebins: int = 10,
            bootstrap_runs: int = 10
    ):
        self.df = df[[x_col, nee_col, gpp_col, reco_col, daytime_col]].copy()
        self.x_col = x_col
        self.nee_col = nee_col
        self.gpp_col = gpp_col
        self.reco_col = reco_col
        self.daytime_col = daytime_col
        self.daytime_threshold = daytime_threshold
        self.set_daytime_if = set_daytime_if
        self.usebins = usebins
        self.bootstrap_runs = bootstrap_runs

        self.predict_min_x = self.df[x_col].min()
        self.predict_max_x = self.df[x_col].max()

        self._chd_bts_results = {}  # Collects results for each bootstrap run
        self._nee_bts_results = {}

    @property
    def chd_results(self) -> dict:
        """Return bootstrap results"""
        if not self._chd_bts_results:
            raise Exception('Results for CHD threshold detection are empty')
        return self._chd_bts_results

    @property
    def nee_results(self) -> dict:
        """Return bootstrap results for daily NEE"""
        if not self._nee_bts_results:
            raise Exception('Results are empty')
        return self._nee_bts_results

    def analyze_nee(self):
        """Analyze NEE"""

        # NEE subset
        nee_df = self.df[[self.x_col, self.nee_col]].copy()

        # Daily maxima and sums
        nee_aggs_df = nee_df.groupby(nee_df.index.date).agg(['max', 'sum'])

        # Fit to bootstrapped data
        # Stored as bootstrap runs > 0 (bts>0)
        nee_bts_results = self._bootstrap_fits(df=nee_aggs_df, x_agg='max', y_agg='sum', fit_to_bins=0)

        # Collect results
        self._nee_bts_results['bts_results'] = nee_bts_results







    def _prepare_daytime_dataset(self):

        # Get daytime data from dataset
        df_daytime, \
        df_nighttime, \
        grp_daynight_col, \
        date_col, \
        flag_daynight_col = \
            self._get_daytime_data()

        # Aggregate daytime dataset
        df_daytime_aggs = \
            self._aggregate(df=df_daytime,
                            groupby_col=grp_daynight_col,
                            date_col=date_col,
                            min_vals=11,
                            aggs=['median', 'count', 'max'])
        return df_daytime_aggs


    def detect_chd_threshold(self):

        # Prepare dataset with daytime data only
        df_daytime_aggs = self._prepare_daytime_dataset()

        # Fit to bootstrapped data
        # Stored as bootstrap runs > 0 (bts>0)
        bts_results = self._bootstrap_fits(df=df_daytime_aggs, x_agg='max', y_agg='median', fit_to_bins=self.usebins)

        # Get flux equilibrium points (RECO = GPP) from bootstrap runs
        bts_linecrossings_df = self._bts_linecrossings_collect(bts_results=bts_results)

        # Calc flux equilibrium points aggregates from bootstrap runs
        bts_linecrossings_aggs = self._linecrossing_aggs(bts_linecrossings_df=bts_linecrossings_df)

        # Threshold for Critical Heat Days (CHDs)
        # defined as the minimum x (e.g. VPD) from bootstrap runs
        thres_chd = bts_linecrossings_aggs['x_min']

        # Collect days above CHD threshold
        df_daytime_aggs_chds = df_daytime_aggs.loc[df_daytime_aggs[self.x_col]['max'] > thres_chd, :].copy()

        # Number of days above CHD threshold
        num_chds = len(df_daytime_aggs_chds)

        # Collect results
        self._chd_bts_results['bts_results'] = bts_results
        self._chd_bts_results['bts_linecrossings_df'] = bts_linecrossings_df
        self._chd_bts_results['bts_linecrossings_aggs'] = bts_linecrossings_aggs
        self._chd_bts_results['thres_chd'] = thres_chd
        self._chd_bts_results['df_daytime_aggs_chds'] = df_daytime_aggs_chds
        self._chd_bts_results['num_chds'] = num_chds

    def plot_chd_detection(self, ax):
        figures.plot_gpp_reco_vs_vpd(ax=ax, results=self.chd_results)

    def plot_nee(self, ax):
        figures.plot_nee_vs_vpd(ax=ax, results=self.nee_results)

    def _get_daytime_data(self):
        """Get daytime data from dataset"""
        return splitdata_daynight(
            df=self.df,
            split_on_col=self.daytime_col,
            split_threshold=self.daytime_threshold,
            split_flagtrue=self.set_daytime_if
        )

    def _linecrossing_aggs(self, bts_linecrossings_df: pd.DataFrame) -> dict:
        """Aggregate linecrossing results from bootstrap runs"""
        # linecrossings_x = []
        # linecrossings_y_gpp = []
        # for b in range(1, self.bootstrap_runs + 1):
        #     linecrossings_x.append(self.bts_results[b]['linecrossing_vals']['x_col'])
        #     linecrossings_y_gpp.append(self.bts_results[b]['linecrossing_vals']['gpp_nom'])

        linecrossing_aggs = dict(
            x_median=bts_linecrossings_df['x_col'].median(),
            x_min=bts_linecrossings_df['x_col'].min(),
            x_max=bts_linecrossings_df['x_col'].max(),
            y_gpp_median=bts_linecrossings_df['gpp_nom'].median(),
            y_gpp_min=bts_linecrossings_df['gpp_nom'].min(),
            y_gpp_max=bts_linecrossings_df['gpp_nom'].max(),
        )

        return linecrossing_aggs

    def _bts_linecrossings_collect(self, bts_results):
        bts_linecrossings_df = pd.DataFrame()
        for bts in range(1, self.bootstrap_runs + 1):
            _series = bts_results[bts]['linecrossing_vals']
            _series.name = bts
            if bts == 1:
                bts_linecrossings_df = pd.DataFrame(bts_results[bts]['linecrossing_vals']).T
            else:
                bts_linecrossings_df = bts_linecrossings_df.append(_series)
        return bts_linecrossings_df

    def _bootstrap_fits(self, df, x_agg:str, y_agg:str, fit_to_bins:int) -> dict:
        """Bootstrap data and make fit for all fluxes"""
        bts_results = {}
        bts = 0
        while bts < self.bootstrap_runs + 1:
            print(f"Bootstrap run #{bts}")
            if bts > 0:
                # Bootstrap data
                bts_df = df.sample(n=int(len(df)), replace=True)
            else:
                # First run (bts=0) is with measured data (not bootstrapped)
                bts_df = df.copy()

            try:
                bts_results[bts] = self._fits(df=bts_df, x_agg=x_agg, y_agg=y_agg, fit_to_bins=fit_to_bins)
                bts += 1
            except ValueError:
                print(f"(!) WARNING Bootstrap run #{bts} was not successful")

        return bts_results

        # for bts in range(1, self.bootstrap_runs + 1):
        #     print(f"Bootstrap run #{bts}")
        #     bts_df = df.sample(n=int(len(df)), replace=True)
        #     try:
        #         self._fits(bts=bts, df=bts_df)
        #     except ValueError:
        #         print(f"(!) Repeating bootstrap run #{bts}")
        #         bts -= 1

    def _fits(self, df, x_agg:str, y_agg:str, fit_to_bins:int):
        """Make fit to GPP, RECO and NEE vs x"""

        # Check what is available
        gpp = True if self.gpp_col in df else False
        reco = True if self.reco_col in df else False
        nee = True if self.nee_col in df else False
        gpp_fit_results = None
        reco_fit_results = None
        nee_fit_results = None
        linecrossing_vals = None

        # GPP sums (class mean) vs classes of x (max)
        if gpp:
            print(f"    Fitting {self.gpp_col}")
            gpp_fit_results = \
                self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
                               y_col=self.gpp_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
            # fitplot(x=gpp_fit_results['x'], y=gpp_fit_results['y'], fit_df=gpp_fit_results['fit_df'])

        # RECO sums (class mean) vs classes of x (max)
        if reco:
            print(f"    Fitting {self.reco_col}")
            reco_fit_results = \
                self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
                               y_col=self.reco_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
            # fitplot(x=reco_fit_results['x'], y=reco_fit_results['y'], fit_df=reco_fit_results['fit_df'])

        # NEE sums (class mean) vs classes of x (max)
        if nee:
            print(f"    Fitting {self.nee_col}")
            nee_fit_results = \
                self._calc_fit(df=df, x_col=self.x_col, x_agg=x_agg,
                               y_col=self.nee_col, y_agg=y_agg, fit_to_bins=fit_to_bins)
            # fitplot(x=nee_fit_results['x'], y=nee_fit_results['y'], fit_df=nee_fit_results['fit_df'])

        # Line crossings
        if gpp and reco:
            linecrossing_vals = \
                self._detect_linecrossing(gpp_fit_results=gpp_fit_results,
                                          reco_fit_results=reco_fit_results)
            if isinstance(linecrossing_vals, pd.Series):
                pass
            else:
                raise ValueError

        # Store bootstrap results in dict
        bts_results = {'gpp': gpp_fit_results,
                       'reco': reco_fit_results,
                       'nee': nee_fit_results,
                       'linecrossing_vals': linecrossing_vals}

        return bts_results

    def _detect_linecrossing(self, gpp_fit_results, reco_fit_results):
        # Collect predicted vals in df
        linecrossings_df = pd.DataFrame()
        linecrossings_df['x_col'] = gpp_fit_results['fit_df']['fit_x']
        linecrossings_df['gpp_nom'] = gpp_fit_results['fit_df']['nom']
        linecrossings_df['reco_nom'] = reco_fit_results['fit_df']['nom']

        # https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
        linecrossings_idx = \
            np.argwhere(np.diff(np.sign(linecrossings_df['gpp_nom'] - linecrossings_df['reco_nom']))).flatten()

        num_linecrossings = len(linecrossings_idx)

        # There must be one single line crossing to accept result
        if num_linecrossings == 1:

            # Flux values at line crossing
            linecrossing_vals = linecrossings_df.iloc[linecrossings_idx[0] + 1]

            # GPP and RECO must be positive, also x value must be
            # above threshold, otherwise reject result
            if (linecrossing_vals['gpp_nom'] < 0) \
                    | (linecrossing_vals['reco_nom'] < 0) \
                    | (linecrossing_vals['x_col'] < 5):
                return None

            return linecrossing_vals

        else:
            # If there is more than one line crossing, reject result
            return None

    def _calc_fit(self, df, x_col, x_agg, y_col, y_agg, fit_to_bins):
        """Call BinFitterCP and fit to x and y"""

        # Names of x and y cols in aggregated df
        x_col = (x_col[0], x_col[1], x_agg)
        y_col = (y_col[0], y_col[1], y_agg)

        fitter = BinFitterCP(df=df,
                             x_col=x_col,
                             y_col=y_col,
                             num_predictions=1000,
                             predict_min_x=self.predict_min_x,
                             predict_max_x=self.predict_max_x,
                             bins_x_num=fit_to_bins,
                             bins_y_agg='median',
                             fit_type='quadratic')
        fitter.run()
        return fitter.get_results()

    def _aggregate(self, df, groupby_col, date_col, min_vals, aggs: list) -> pd.DataFrame:
        """Aggregate dataset by *day/night groups*"""

        # Aggregate values by day/night group membership, this drops the date col
        agg_df = \
            df.groupby(groupby_col) \
                .agg(aggs)
        # .agg(['median', q25, q75, 'count', 'max'])
        # .agg(['median', q25, q75, 'min', 'max', 'count', 'mean', 'std', 'sum'])

        # Add the date col back to data
        grp_daynight_col = \
            (groupby_col[0], groupby_col[1], groupby_col[1])
        agg_df[grp_daynight_col] = agg_df.index

        # For each day/night group, detect its start and end time

        ## Start date (w/ .idxmin)
        grp_starts = df.groupby(groupby_col).idxmin()[date_col].dt.date
        grp_starts = grp_starts.to_dict()
        grp_startdate_col = ('.GRP_STARTDATE', '[aux]', '[aux]')
        agg_df[grp_startdate_col] = agg_df[grp_daynight_col].map(grp_starts)

        ## End date (w/ .idxmax)
        grp_ends = df.groupby(groupby_col).idxmax()[date_col].dt.date
        grp_ends = grp_ends.to_dict()
        grp_enddate_col = ('.GRP_ENDDATE', '[aux]', '[aux]')
        agg_df[grp_enddate_col] = agg_df[grp_daynight_col].map(grp_ends)

        # Set start date as index
        agg_df = agg_df.set_index(grp_startdate_col)

        # Keep consecutive time periods with enough values (min. 11 half-hours)
        agg_df = agg_df.where(agg_df[self.x_col]['count'] >= min_vals).dropna()

        return agg_df


if __name__ == '__main__':
    configfilepath = Path('../../../../tests/testdata/DIIVE_CSV_30MIN.yml')
    filepath = Path('../../../../tests/testdata/testfile_ch-dav_2016-2020.diive.csv')

    # Read config file
    fileconfig = ConfigFileReader(configfilepath=configfilepath).read()

    # Read data file
    datafilereader = DataFileReader(
        filepath=filepath,
        data_skiprows=fileconfig['DATA']['SKIP_ROWS'],
        data_headerrows=fileconfig['DATA']['HEADER_ROWS'],
        data_headersection_rows=fileconfig['DATA']['HEADER_SECTION_ROWS'],
        data_na_vals=fileconfig['DATA']['NA_VALUES'],
        data_delimiter=fileconfig['DATA']['DELIMITER'],
        data_freq=fileconfig['DATA']['FREQUENCY'],
        timestamp_idx_col=fileconfig['TIMESTAMP']['INDEX_COLUMN'],
        timestamp_datetime_format=fileconfig['TIMESTAMP']['DATETIME_FORMAT'],
        timestamp_start_middle_end=fileconfig['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD']
    )
    df = datafilereader.get_data()
    df = df.loc[df.index.year > 2017, :]

    # Settings
    x_col = ('VPD_f', '-no-units-')
    nee_col = ('NEE_CUT_f', '-no-units-')
    gpp_col = ('GPP_DT_CUT', '-no-units-')
    reco_col = ('Reco_DT_CUT', '-no-units-')
    daytime_col = ('Rg_f', '-no-units-')

    # Use data from May to Sep only
    maysep_filter = (df.index.month >= 5) & (df.index.month <= 10)
    df = df.loc[maysep_filter].copy()

    # Carbon cost
    chd = CriticalHeatDays(
        df=df,
        x_col=x_col,
        nee_col=nee_col,
        gpp_col=gpp_col,
        reco_col=reco_col,
        daytime_col=daytime_col,
        daytime_threshold=50,
        set_daytime_if='Larger Than Threshold',
        usebins=5,
        bootstrap_runs=3
    )

    chd.detect_chd_threshold()

    # # Prepare figure
    # import matplotlib.gridspec as gridspec
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(9, 9))
    # gs = gridspec.GridSpec(1, 1)  # rows, cols
    # # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    # ax = fig.add_subplot(gs[0, 0])
    # chd.plot_chd_detection(ax=ax)
    # fig.show()

    chd.analyze_nee()

    # Prepare figure
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(1, 1)  # rows, cols
    # gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
    ax = fig.add_subplot(gs[0, 0])
    chd.plot_nee(ax=ax)
    fig.show()

    print("END")

    print(df)
