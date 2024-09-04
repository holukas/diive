"""

==================================
QCF - OVERALL QUALITY CONTROL FLAG
==================================

Combine multiple flags in one single QCF flag.

"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series



from diive.core.base.identify import identify_flagcols
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.createvar.daynightflag import daytime_nighttime_flag_from_swinpot


class FlagQCF:
    """Calculate overall quality flag QCF"""

    def __init__(self,
                 df: DataFrame,
                 series: Series,
                 outname: str = None,
                 swinpot: Series = None,
                 idstr: str = None,
                 nighttime_threshold: int = 50
                 ):
        self.df = df.copy()  # Original data
        self.series = series.copy()

        self.outname = outname if outname else series.name

        self.idstr = validate_id_string(idstr=idstr)

        # Identify FLAG columns
        flagcols = identify_flagcols(df=df, seriescol=str(series.name))
        self._flags_df = df[flagcols].copy()

        # Detect daytime and nighttime
        if isinstance(swinpot, Series):
            self.daytime, self.nighttime = \
                daytime_nighttime_flag_from_swinpot(swinpot=swinpot, nighttime_threshold=nighttime_threshold)
        else:
            self.daytime = None
            self.nighttime = None

        # Generate QCF column names
        self.filteredseriescol = f"{self.outname}{self.idstr}_QCF"  # Quality-controlled flux
        self.filteredseriescol_hq = f"{self.outname}{self.idstr}_QCF0"  # Quality-controlled flux, highest quality
        self.flagqcfcol = f"FLAG{self.idstr}_{self.outname}_QCF"  # Overall flag
        self.sumflagscol = f'SUM{self.idstr}_{self.outname}_FLAGS'
        self.sumhardflagscol = f'SUM{self.idstr}_{self.outname}_HARDFLAGS'
        self.sumsoftflagscol = f'SUM{self.idstr}_{self.outname}_SOFTFLAGS'

        self.daytime_accept_qcf_below = None
        self.nighttimetime_accept_qcf_below = None

    @property
    def flags(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._flags_df, DataFrame):
            raise Exception('Results for flags are empty')
        return self._flags_df

    @property
    def filteredseries(self) -> Series:
        """Return series with rejected values set to missing"""
        return self.flags[self.filteredseriescol]

    @property
    def filteredseries_hq(self) -> Series:
        """Return series with highest-quality fluxes only"""
        return self.flags[self.filteredseriescol_hq]

    @property
    def flagqcf(self) -> Series:
        """Return QCF flag for series"""
        return self.flags[self.flagqcfcol]

    def get(self) -> DataFrame:
        """Return original data with QCF flag"""
        returndf = self.df.copy()  # Main data
        newcols = [col for col in self.flags.columns if col not in returndf]
        newcolsdf = self.flags[newcols].copy()
        returndf = pd.concat([returndf, newcolsdf], axis=1)  # Add new columns to main data
        [print(f"++Added new column {c}.") for c in newcols]
        return returndf

    def calculate(self,
                  daytime_accept_qcf_below: int = 2,
                  nighttimetime_accept_qcf_below: int = 2):
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttimetime_accept_qcf_below = nighttimetime_accept_qcf_below
        self._flags_df = self._calculate_flagsums(df=self._flags_df)
        self._flags_df = self._calculate_flag_qcf(df=self._flags_df)
        self._add_series()
        self._calculate_series_qcf()



    def _add_series(self):
        self._flags_df[self.series.name] = self.series.copy()

    def _calculate_series_qcf(self):
        """Create quality-checked time series"""
        # Accepted-quality fluxes
        self._flags_df[self.filteredseriescol] = self._flags_df[self.series.name].copy()
        ix = self._flags_df[self.flagqcfcol] == 2
        self._flags_df.loc[ix, self.filteredseriescol] = np.nan

        # Highest-quality fluxes
        self._flags_df[self.filteredseriescol_hq] = self._flags_df[self.series.name].copy()
        ix = self._flags_df[self.flagqcfcol] > 0
        self._flags_df.loc[ix, self.filteredseriescol_hq] = np.nan

    def report_qcf_flags(self):

        flagcols = identify_flagcols(df=self.flags, seriescol=str(self.series.name))

        # Report for individual flags
        print(f"\n{'=' * 40}\nREPORT: FLAGS INCL. MISSING VALUES\n{'=' * 40}")
        print("Stats with missing values in the dataset")
        for col in flagcols:
            self._flagstats_dt_nt(col=col, df=self.flags)

        # Report for available values (missing values are ignored)
        # Flags
        print(f"\n{'=' * 40}\nREPORT: FLAGS FOR AVAILABLE RECORDS\n{'=' * 40}")
        print("Stats after removal of missing values")
        _df = self.flags.copy()
        ix_missing_vals = _df[self.series.name].isnull()
        _df = _df[~ix_missing_vals].copy()
        for col in flagcols:
            self._flagstats_dt_nt(col=col, df=_df)

    def _flagstats_dt_nt(self, col: str, df: DataFrame):
        print(f"{col}:")
        flag = df[col]
        self._flagstats(flag=flag, prefix="OVERALL")
        if isinstance(self.daytime, Series):
            flag = df[col].loc[self.daytime == 1]
            self._flagstats(flag=flag, prefix="DAYTIME")
        if isinstance(self.nighttime, Series):
            flag = df[col].loc[self.nighttime == 1]
            self._flagstats(flag=flag, prefix="NIGHTTIME")

    def report_qcf_evolution(self):
        """Apply multiple test flags sequentially"""
        # QCF flag evolution
        print(f"\n\n{'=' * 40}\nQCF FLAG EVOLUTION\n{'=' * 40}\n"
              f"This output shows the evolution of the QCF overall quality flag\n"
              f"when test flags are applied sequentially to the variable {self.series.name}.")

        flagcols = identify_flagcols(df=self.flags, seriescol=str(self.series.name))
        allflags_df = self.flags[flagcols].copy()
        ix_missing_vals = self.df[self.series.name].isnull()
        allflags_df = allflags_df[~ix_missing_vals].copy()  # Ignore missing values

        n_tests = len(allflags_df.columns) + 1  # +1 b/c for loop
        ix_first_test = 0
        n_vals_total_rejected = 0
        n_flag2 = 0
        perc_flag2 = 0
        n_vals = len(allflags_df)
        print(f"\nNumber of {self.series.name} records before QC: {n_vals}")
        for ix_last_test in range(1, n_tests):
            prog_testcols = flagcols[ix_first_test:ix_last_test]
            prog_df = allflags_df[prog_testcols].copy()

            # Calculate QCF (so far)
            prog_df = self._calculate_flagsums(df=prog_df)
            prog_df = self._calculate_flag_qcf(df=prog_df)

            # Count flag occurrences
            n_flag0 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 0].count()
            n_flag1 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 1].count()
            n_flag2 = prog_df[self.flagqcfcol].loc[prog_df[self.flagqcfcol] == 2].count()

            # Calculate some flag stats
            n_vals_test_rejected = n_flag2 - n_vals_total_rejected
            perc_vals_test_rejected = (n_vals_test_rejected / n_vals) * 100
            perc_flag0 = (n_flag0 / n_vals) * 100
            perc_flag1 = (n_flag1 / n_vals) * 100
            perc_flag2 = (n_flag2 / n_vals) * 100

            print(f"+++ {prog_testcols[ix_last_test - 1]} rejected {n_vals_test_rejected} values "
                  f"(+{perc_vals_test_rejected:.2f}%)      "
                  f"TOTALS: flag 0: {n_flag0} ({perc_flag0:.2f}%) / "
                  f"flag 1: {n_flag1} ({perc_flag1:.2f}%) / "
                  f"flag 2: {n_flag2} ({perc_flag2:.2f}%)")

            n_vals_total_rejected = n_flag2

        print(f"\nIn total, {n_flag2} ({perc_flag2:.2f}%) of the available records were rejected in this step.")
        print(f"INFO Rejected DAYTIME records where QCF flag >= {self.daytime_accept_qcf_below}")
        print(f"INFO Rejected NIGHTTIME records where QCF flag >= {self.nighttimetime_accept_qcf_below}")
        # print(f"\n| Note that this is not the final QC step. More values need to be \n"
        #       f"| rejected after storage correction (Level-3.1) during outlier\n"
        #       f"| removal (Level-3.2) and USTAR filtering (Level-3.3).")


    def report_highest_quality(self):
        from diive.core.dfun.stats import sstats
        hq_dt = self.flags.loc[self.daytime == 1, self.filteredseriescol_hq].copy()
        hq_nt = self.flags.loc[self.nighttime == 1, self.filteredseriescol_hq].copy()
        # stats_dt = sstats(s=hq_dt)
        # stats_nt = sstats(s=hq_nt)
        #
        # stats = pd.concat([stats_dt, stats_nt], axis=1, keys=['DAYTIME', 'NIGHTTIME'])
        #
        # winsize = int(hq_dt.count() / 10)
        # rmean_nt = hq_nt.rolling(window=winsize, center=True, min_periods=1).mean()
        # upper_nt = hq_nt.rolling(window=winsize, center=True, min_periods=1).std()
        # upper_nt = rmean_nt.add(upper_nt)
        # lower_nt = hq_nt.rolling(window=winsize, center=True, min_periods=1).std()
        # lower_nt = rmean_nt.sub(lower_nt)
        # upper_nt = hq_nt.rolling(window=winsize, center=True, min_periods=1).quantile(0.99)
        # lower_nt = hq_nt.rolling(window=winsize, center=True, min_periods=1).quantile(0.01)
        # hq_nt.plot()
        # rmean_nt.plot()
        # upper_nt.plot()
        # lower_nt.plot()
        # plt.show()

        # upper_nt.max()
        # lower_nt.min()


        from diive.core.plotting.histogram import HistogramPlot
        import matplotlib.pyplot as plt
        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax_series = fig.add_subplot(gs[0, 0])
        # ax_series_hist = fig.add_subplot(gs[0, 1])

        ax_series.plot(hq_nt.index, hq_nt,
                       label=f"{self.series.name}", color="#607D8B", linestyle='none', markeredgewidth=1,
                       marker='o', alpha=.5, markersize=6, markeredgecolor="#607D8B", fillstyle='none')
        ax_series.set_ylim(-10, 20)
        ax_series.axhline(hq_nt.quantile(0.03))

        hist_kwargs = dict(method='n_bins', n_bins=None, highlight_peak=True, show_zscores=True, show_info=False,
                           show_title=False, show_zscore_values=False, show_grid=False)
        # HistogramPlot(s=hq_nt, **hist_kwargs).plot(ax=ax_series_hist)

        fig.suptitle("Highest-quality nighttime flux after Level-2 (CH-CHA), zoomed in", fontsize=16)
        fig.tight_layout()
        fig.show()



        # todo hier weiter
        from diive.pkgs.outlierdetection.zscore import zScoreRolling
        zsc = zScoreRolling(series=hq_nt, thres_zscore=3, winsize=int(hq_nt.dropna().count() / 100),
                            showplot=True)
        zsc.calc(repeat=True)

        from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime
        zdn = zScoreDaytimeNighttime(
            series=hq_nt,
            lat=47.286417,
            lon=7.733750,
            utc_offset=1,
            thres_zscore=5.5,
            showplot=True,
            verbose=True)
        zdn.calc(repeat=True)



        from diive.pkgs.outlierdetection.lof import LocalOutlierFactorAllData
        n_neighbors = int(hq_nt.dropna().count() / 100)
        lofa = LocalOutlierFactorAllData(
            series=hq_nt,
            n_neighbors=n_neighbors,
            contamination='auto',
            showplot=True,
            verbose=True,
            n_jobs=-1
        )

        lofa.calc(repeat=True)
        print(f"--------------------------- n_neighbors: {n_neighbors}")




        # # todo isolation forest
        # from sklearn.ensemble import IsolationForest
        # # Create an Isolation Forest model
        # iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset
        # # Fit the model to your data
        # X = hq_dt.dropna().to_numpy().reshape(-1, 1)
        # iso_forest.fit(X)
        # # Predict outliers
        # outliers = iso_forest.predict(X)
        #
        # import matplotlib.pyplot as plt
        #
        # from sklearn.inspection import DecisionBoundaryDisplay
        #
        # disp = DecisionBoundaryDisplay.from_estimator(
        #     iso_forest,
        #     X,
        #     response_method="predict",
        #     alpha=0.5,
        # )
        # disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        # disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
        # plt.axis("square")
        # plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
        # plt.show()

        # # https://www.geeksforgeeks.org/anomaly-detection-using-isolation-forest/
        # from sklearn.ensemble import IsolationForest
        # from sklearn.model_selection import train_test_split
        # from sklearn.datasets import load_iris
        # import matplotlib.pyplot as plt
        # iris = load_iris()
        # X = iris.data
        # y = iris.target
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # # initialize and fit the model
        # clf = IsolationForest(contamination=0.1)
        # clf.fit(X_train)
        # # predict the anomalies in the data
        # y_pred_train = clf.predict(X_train)
        # y_pred_test = clf.predict(X_test)
        # print(y_pred_train)
        # print(y_pred_test)
        # def create_scatter_plots(X1, y1, title1, X2, y2, title2):
        #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #     # Scatter plot for the first set of data
        #     axes[0].scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], color='green', label='Normal')
        #     axes[0].scatter(X1[y1 == -1, 0], X1[y1 == -1, 1], color='red', label='Anomaly')
        #     axes[0].set_title(title1)
        #     axes[0].legend()
        #
        #     # Scatter plot for the second set of data
        #     axes[1].scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], color='green', label='Normal')
        #     axes[1].scatter(X2[y2 == -1, 0], X2[y2 == -1, 1], color='red', label='Anomaly')
        #     axes[1].set_title(title2)
        #     axes[1].legend()
        #
        #     plt.tight_layout()
        #     plt.show()
        #
        # # scatter plots
        # create_scatter_plots(X_train, y_pred_train, 'Training Data', X_test, y_pred_test, 'Test Data')
        #
        # range = np.arange(0, 1, 0.001)
        # test = pd.Series(index=range)
        # for x in range:
        #     q = hq_nt.quantile(x)
        #     test.loc[x] = q
        # test.plot()
        # plt.show()



        # self.flags.loc[self.nighttime == 1, self.filteredseriescol_hq].describe()
        # self.flags.loc[self.daytime == 1, self.filteredseriescol_hq].plot()
        # self.flags.loc[self.nighttime == 1, self.filteredseriescol_hq].plot()

    def _flagstats(self, flag: Series, prefix: str):
        n_values = len(flag)
        flagcounts = flag.groupby(flag).count()
        flagmissing = flag.isnull().sum()
        flagmissing_perc = (flagmissing / n_values) * 100
        for flagvalue in flagcounts.index:
            _counts = flagcounts[flagvalue]
            _counts_perc = (_counts / n_values) * 100
            print(f"    {prefix} flag {flagvalue}: {_counts} values ({_counts_perc:.2f}%)  ")
        print(f"    {prefix} flag missing: {flagmissing} values ({flagmissing_perc:.2f}%)  ")
        print("")

    def report_qcf_series(self):

        print(f"\n\n{'=' * 40}\nSUMMARY: {self.flagqcfcol}, QCF FLAG FOR {self.series.name}\n{'=' * 40}")

        series = self.flags[self.series.name]
        seriesqcf = self.flags[self.filteredseriescol]

        n_potential = len(series)
        n_measured = len(series.dropna())
        n_missed = n_potential - n_measured
        n_available = len(seriesqcf.dropna())  # Available after QC
        n_rejected = n_measured - n_available  # Rejected measured values

        perc_measured = (n_measured / n_potential) * 100
        perc_missed = (n_missed / n_potential) * 100
        perc_available = (n_available / n_measured) * 100
        perc_rejected = (n_rejected / n_measured) * 100

        start = series.index[0].strftime('%Y-%m-%d %H:%M')
        end = series.index[-1].strftime('%Y-%m-%d %H:%M')
        print(f"Between {start} and {end} ...\n"
              f"    Total flux records BEFORE quality checks: {n_measured} ({perc_measured:.2f}% of potential)\n"
              f"    Available flux records AFTER quality checks: {n_available} ({perc_available:.2f}% of total)\n"
              f"    Rejected flux records: {n_rejected} ({perc_rejected:.2f}% of total)\n"
              f"    Potential flux records: {n_potential}\n"
              f"    Potential flux records missed: {n_missed} ({perc_missed:.2f}% of potential)\n")

    def _calculate_flag_qcf(self, df: DataFrame) -> DataFrame:
        """Calculate overall QCF flag from flag sums"""

        # QCF is NaN if no flag is available
        df[self.flagqcfcol] = np.nan

        # QCF is 0 if all flags show zero
        ix = df[self.sumflagscol] == 0
        df.loc[ix, self.flagqcfcol] = 0

        # QCF is 2 if more than three soft flags were raised
        ix = df[self.sumsoftflagscol] > 3
        df.loc[ix, self.flagqcfcol] = 2

        # QCF is 2 if at least one hard flag was raised
        ix = df[self.sumhardflagscol] >= 2
        df.loc[ix, self.flagqcfcol] = 2

        # QCF is 1 if no hard flag and max. three soft flags and
        # min. one soft flag were raised
        ix = (df[self.sumsoftflagscol] <= 3) & (df[self.sumsoftflagscol] >= 1) & (df[self.sumhardflagscol] == 0)
        df.loc[ix, self.flagqcfcol] = 1

        # Flag daytime values based on param
        if isinstance(self.daytime, Series):
            ix = (df[self.flagqcfcol] >= self.daytime_accept_qcf_below) & (self.daytime == 1)
            df.loc[ix, self.flagqcfcol] = 2

        # Flag nighttime values based on param
        if isinstance(self.nighttime, Series):
            ix = (df[self.flagqcfcol] >= self.nighttimetime_accept_qcf_below) & (self.nighttime == 1)
            df.loc[ix, self.flagqcfcol] = 2

        # Daytime and nighttime flags are only calculated when swinpot is provided.
        # This means that if both do not exist, no separation into daytime and nighttime
        # was done. In that case, all records where QCF = 2 are rejected.
        if not isinstance(self.daytime, Series) and not isinstance(self.nighttime, Series):
            default_accept_qcf_below = 2
            ix = (df[self.flagqcfcol] >= default_accept_qcf_below)
            df.loc[ix, self.flagqcfcol] = 2

        return df

    def _calculate_flagsums(self, df: DataFrame) -> DataFrame:
        """Calculate sums of all individual flags"""
        sumhardflags = df[df == 2].sum(axis=1)  # The sum of all flags that show 2
        sumsoftflags = df[df == 1].sum(axis=1)  # The sum of all flags that show 1
        sumflags = sumhardflags.add(sumsoftflags)  # Sum of all flags
        df[self.sumhardflagscol] = sumhardflags
        df[self.sumsoftflagscol] = sumsoftflags
        df[self.sumflagscol] = sumflags
        return df

    def showplot_qcf_heatmaps(self, maxabsval: float = None, figsize: tuple = (18, 8)):

        fig = plt.figure(facecolor='white', figsize=figsize)
        gs = gridspec.GridSpec(1, 4)  # rows, cols
        gs.update(wspace=0.4, hspace=0, left=0.03, right=0.97, top=0.9, bottom=0.1)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1], sharey=ax_before)
        ax_flagsum = fig.add_subplot(gs[0, 2], sharey=ax_before)
        ax_flag = fig.add_subplot(gs[0, 3], sharey=ax_before)

        # Heatmaps
        vmin = -maxabsval if maxabsval else None
        vmax = maxabsval if maxabsval else None
        HeatmapDateTime(ax=ax_before, series=self.flags[self.series.name], vmin=vmin, vmax=vmax,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_after, series=self.flags[self.filteredseriescol], vmin=vmin, vmax=vmax,
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flagsum, series=self.flags[self.sumflagscol],
                        cb_digits_after_comma=0).plot()
        HeatmapDateTime(ax=ax_flag, series=self.flags[self.flagqcfcol],
                        cb_digits_after_comma=0).plot()
        plt.setp(ax_after.get_yticklabels(), visible=False)
        plt.setp(ax_flagsum.get_yticklabels(), visible=False)
        plt.setp(ax_flag.get_yticklabels(), visible=False)
        ax_after.axes.get_yaxis().get_label().set_visible(False)
        ax_flagsum.axes.get_yaxis().get_label().set_visible(False)
        ax_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()

    def showplot_qcf_timeseries(self, figsize=(16, 20)):
        self.flags.plot(subplots=True, figsize=figsize)
        plt.show()
