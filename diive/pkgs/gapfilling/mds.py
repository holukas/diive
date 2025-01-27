"""

MARGINAL DISTRIBUTION SAMPLING (MDS)
Gap-filling after Reichstein et al (2005)

Reference: https://doi.org/10.1111/j.1365-2486.2005.001002.x

"""
from collections import Counter
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, default_legend
from diive.core.plotting.styles.LightTheme import colorwheel_36, generate_plot_marker_list
from diive.pkgs.gapfilling.scores import prediction_scores


class FluxMDS:
    gfsuffix = '_gfMDS'

    def __init__(self,
                 df: DataFrame,
                 flux: str,
                 swin: str,
                 ta: str,
                 vpd: str,
                 swin_class: list = None,  # Default defined below: [20, 50]
                 ta_class: float = 2.5,
                 vpd_class: float = 0.5,
                 min_n_vals_nt: int = 0,
                 verbose: int = 1):
        """Gap-filling for ecosystem fluxes, based on marginal distribution sampling (MDS
        described in Reichstein et al. (2005).

        Missing values are replaced by the average *flux* value during
        similar meteorological conditions.

        The MDS method in diive was implemented following the description in
        Reichstein et al. (2005). One difference in the implementation is that
        diive introduces the parameter *min_n_vals_nt*, which allows to set a
        minimum of required values to calculate the average *flux* for the gap
        during nighttime conditions.

        Args:
            df: Dataframe that contains data for *flux*, *swin*, *ta* and *vpd*.
            flux: Name of flux variable in *df* that will be gap-filled.
            swin: Name of short-wave incoming radiation variable in *df*. (W m-2)
            ta: Name of air temperature variable in *df*. (°C)
            vpd: Name of vapor pressure deficit variable in *df*. (kPa)
            todo swin_class: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 50 W m-2 (default). (W m-2)
            ta_class: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 2.5 °C (default). (°C)
            vpd_class: Used for grouping *flux* data into groups of similar
                meteorological conditions. Data in the respective group must
                not deviate by more than +/- 0.5 kPa (default). (kPa)
            min_n_vals_nt: Minimum number of measured *flux* values required to
                calculate the average *flux* value for gaps during nighttime.
            verbose: Value 1 creates more text output.
        """
        self._gapfilling_df = df[[flux, swin, ta, vpd]].copy()
        self.flux = flux
        self.swin = swin
        self.ta = ta
        self.vpd = vpd
        if not swin_class:
            self.swin_class = [20, 50]
        else:
            if isinstance(swin_class, list):
                self.swin_class = swin_class
            else:
                raise TypeError('swin_class must be a list with two elements. (default: [20, 50])')
        self.ta_class = ta_class
        self.vpd_class = vpd_class
        self.min_n_vals_nt = min_n_vals_nt if min_n_vals_nt else 0
        self.verbose = verbose

        self._scores = dict()

        self.target_gapfilled = f"{self.flux}{self.gfsuffix}"
        self.target_gapfilled_flag = f"FLAG_{self.flux}{self.gfsuffix}_ISFILLED"

        self._gapfilling_df = self._add_newcols()

        self.workdf = DataFrame()

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled].copy()

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1+=gap-filled"""
        return self.gapfilling_df_[self.target_gapfilled_flag]

    @property
    def gapfilled_(self) -> pd.Series:
        """Gap-filled data."""
        series = self.get_gapfilled_target()
        if not isinstance(series, pd.Series):
            raise Exception('No gap-filled data available.')
        return series

    @property
    def target_col(self) -> str:
        """Gap-filled data."""
        if not isinstance(self.flux, str):
            raise Exception('No name for gap-filled variable available.')
        return self.flux

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Dataframe containing all data."""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception('No dataframe containing all data available.')
        return self._gapfilling_df

    @property
    def scores_(self) -> dict:
        """Return scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    def showplot(self):
        fig = plt.figure(facecolor='white', figsize=(16, 9), dpi=100, layout='constrained')
        gs = gridspec.GridSpec(2, 1, figure=fig)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        ax_flag = fig.add_subplot(gs[1, 0], sharex=ax)
        flag = self.gapfilling_df_[self.target_gapfilled_flag]
        uniqueflags = list(flag.unique())
        uniqueflags.sort()
        colors = colorwheel_36()
        maxcolors = len(colors)
        markers = generate_plot_marker_list()
        maxmarker = len(markers)
        for ix, uf in enumerate(uniqueflags):
            locs = flag == uf
            data = self.gapfilling_df_.loc[locs, :]
            label = f"measured ({self.flux})" if uf == 0 else f"gap-filled quality {uf}"
            n_vals = data[self.target_gapfilled].count()
            color = colors[35] if ix > (maxcolors - 1) else colors[ix]
            marker = markers[3] if ix > (maxmarker - 1) else markers[ix]
            ax.plot(data.index, data[self.target_gapfilled],
                    label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                    marker=marker, alpha=1, markersize=6, markeredgecolor=color, fillstyle='full')
            ax_flag.plot(data.index, data[self.target_gapfilled_flag],
                         label=f"{label} ({n_vals} values)", color=color, linestyle='none', markeredgewidth=1,
                         marker=marker, alpha=1, markersize=6, markeredgecolor=color, fillstyle='full')
        fig.suptitle(f"Variable {self.flux} gap-filled using MDS: {self.target_gapfilled}",
                     fontsize=theme.FIGHEADER_FONTSIZE)
        default_format(ax=ax)
        ax.tick_params(labelbottom=False)
        default_format(ax=ax_flag)
        default_legend(ax=ax_flag)
        # default_legend(ax=ax)
        fig.show()

    def report(self):

        potential_vals = len(self.gapfilling_df_.index)
        n_vals_before = self.gapfilling_df_[self.flux].count()
        n_vals_missing_before = self.gapfilling_df_[self.flux].isnull().sum()
        n_vals_after = self.gapfilling_df_[self.target_gapfilled].count()
        n_vals_missing_after = self.gapfilling_df_[self.target_gapfilled].isnull().sum()
        predictionsmeanquality = self.gapfilling_df_['.PREDICTIONS_QUALITY'].mean()
        flagcounts = Counter(self.gapfilling_df_[self.target_gapfilled_flag])

        print(f"{self.flux} before gap-filling:\n"
              f"    {potential_vals} potential values\n"
              f"    {n_vals_before} available values\n"
              f"    {n_vals_missing_before} missing values")

        print(f"\n{self.flux} after gap-filling ({self.target_gapfilled}):\n"
              f"    {potential_vals} potential values\n"
              f"    {n_vals_after} available values\n"
              f"    {n_vals_missing_after} missing values\n"
              f"    {predictionsmeanquality:.3f} predictions mean quality across all records (1=best)")

        print(f"\nGap-filling quality flags ({self.target_gapfilled_flag}):")
        for key, value in flagcounts.items():
            if key == 0:
                print(f"    Directly measured: {value} values (flag=0)")
            else:
                print(f"    Gap-filling quality {key}: {value} values (flag={key})")

        self.report_scores()

    def report_scores(self):
        print("\nMDS gap-filling scores:")
        for score, val in self.scores_.items():
            print(f'    {score}: {val:.3f}')

    def run(self):
        # https://www.geeksforgeeks.org/apply-function-to-every-row-in-a-pandas-dataframe/
        # https://labs.quansight.org/blog/unlocking-c-level-performance-in-df-apply

        self._gapfilling_df = self.gapfilling_df_.copy()
        locs_missing = self.gapfilling_df_['.PREDICTIONS'].isnull()
        self.workdf = self._gapfilling_df[locs_missing].copy()

        # A1: SWIN, TA, VPD, NEE available within 7 days (highest quality gap-filling).
        self.workdf, self._gapfilling_df = self._run_all_available(days=7, quality=1)

        # A2: SWIN, TA, VPD, NEE available within 14 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=14, quality=2)

        # A3: SWIN, NEE available within 7 days
        self.workdf, self._gapfilling_df = self._run_two_available(days=7, quality=3)

        # A4: NEE available within |dt| <= 1h on same day
        self.workdf, self._gapfilling_df = self._run_mdc(days=0, hours=1, quality=4)

        # B1: same hour NEE available within |dt| <= 1 day
        self.workdf, self._gapfilling_df = self._run_mdc(days=1, hours=1, quality=5)

        # B2: SWIN, TA, VPD, NEE available within 21 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=21, quality=6)

        # B3: SWIN, TA, VPD, NEE available within 28 days
        self.workdf, self._gapfilling_df = self._run_all_available(days=28, quality=7)

        # B4: SWIN, NEE available within 14 days
        self.workdf, self._gapfilling_df = self._run_two_available(days=14, quality=8)

        # C+: SWIN, TA, VPD, NEE available within 35-140 days
        quality = 8  # Quality from previous step B4
        for d in range(35, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_all_available(days=d, quality=quality)

        # C+: SWIN, NEE available within 21-140 days
        quality = 24  # Maximum possible quality from previous step C+
        for d in range(21, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_two_available(days=d, quality=quality)

        # C+: same hour NEE available within |dt| <= 7-X days
        quality = 42  # Maximum possible quality from previous step C+
        for d in range(21, 147, 7):
            quality += 1
            self.workdf, self._gapfilling_df = self._run_mdc(days=d, hours=1, quality=quality)

        # print(self.gapfilling_df_)

        # Gap-filled measurement time series
        self.gapfilling_df_[self.target_gapfilled] = self.gapfilling_df_[self.flux].fillna(
            self.gapfilling_df_['.PREDICTIONS'])

        # Gap-filling flag is 0 where measurement available
        locs_measured_missing = self.gapfilling_df_[self.flux].isnull()
        locs_measured_available = ~locs_measured_missing
        self.gapfilling_df_.loc[locs_measured_available, self.target_gapfilled_flag] = 0

        # Gap-filling flag is equal to prediction quality where measurement was missing
        self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag] = \
            self.gapfilling_df_.loc[locs_measured_missing, '.PREDICTIONS_QUALITY']
        # self.gapfilling_df_[self.target_gapfilled_flag] = \
        #     self.gapfilling_df_[self.target_gapfilled_flag].fillna(self.gapfilling_df_['.PREDICTIONS_QUALITY'])

        # # Flag
        # # Make flag column that indicates where predictions for
        # # missing targets are available, where 0=observed, 1=gapfilled
        # # todo Note that missing predicted gaps = 0. change?
        # _gapfilled_locs = self._gapfilling_df[self.pred_gaps_col].isnull()  # Non-gapfilled locations
        # _gapfilled_locs = ~_gapfilled_locs  # Inverse for gapfilled locations
        # self._gapfilling_df[self.target_gapfilled_flag_col] = _gapfilled_locs
        # self._gapfilling_df[self.target_gapfilled_flag_col] = self._gapfilling_df[
        #     self.target_gapfilled_flag_col].astype(
        #     int)

        # import matplotlib.pyplot as plt
        # # self.df[self.target_gapfilled_flag].plot(label="gapfilled", ls='none', markersize=4, marker="o")
        # # self.df[self.target_gapfilled].plot(label="gapfilled", ls='none', markersize=4, marker="o")
        # self.gapfilling_df_['.PREDICTIONS'].plot(label="predictions", ls='none', markersize=4, marker="o")
        # self.gapfilling_df_[self.flux].plot(ls='none', markersize=4, marker="o")
        # plt.legend()
        # plt.show()

        # Calculate scores
        scoredf = self.gapfilling_df_[['.PREDICTIONS', self.flux]].copy()
        scoredf = scoredf.dropna()
        self._scores = prediction_scores(predictions=scoredf['.PREDICTIONS'], targets=scoredf[self.flux])

        self._scores['mean_quality_flag_gap_predictions'] = \
            self.gapfilling_df_.loc[locs_measured_missing, self.target_gapfilled_flag].mean()

        print("MDS gap-filling done.")

    def _run_all_available(self, days: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"\nMDS gap-filling quality {quality} ...")

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        workdf['.PREDICTIONS'] = workdf.apply(self._a_1_2, axis=1)
        workdf['.PREDICTIONS_QUALITY'] = quality

        if quality == 1:
            _df['.PREDICTIONS'] = workdf['.PREDICTIONS'].copy()
            _df['.START'] = workdf['.START'].copy()
            _df['.END'] = workdf['.END'].copy()
            _df['.PREDICTIONS_QUALITY'] = workdf['.PREDICTIONS_QUALITY'].copy()
        else:
            _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _prepare_dataframes(self) -> tuple[DataFrame, DataFrame]:
        _df = self.gapfilling_df_.copy()
        workdf = self.workdf.copy()
        return _df, workdf

    def _run_two_available(self, days: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"MDS gap-filling quality {quality} ...")

        offset = pd.DateOffset(days=days)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        workdf['.PREDICTIONS'] = workdf.apply(self._a3, axis=1)
        workdf['.PREDICTIONS_QUALITY'] = quality

        _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _run_mdc(self, days: int, hours: int, quality: int):

        _df, workdf = self._prepare_dataframes()
        if workdf.empty:
            return workdf, _df

        print(f"MDS gap-filling quality {quality} ...")

        offset = pd.DateOffset(days=days, hours=hours)
        workdf['.START'] = pd.to_datetime(workdf.index) - offset
        workdf['.END'] = pd.to_datetime(workdf.index) + offset
        if days == 0:
            workdf['.PREDICTIONS'] = workdf.apply(self._a4, axis=1)
        else:
            workdf['.PREDICTIONS'] = workdf.apply(self._b1, axis=1)
        workdf['.PREDICTIONS_QUALITY'] = quality

        _df = self._fill_predictions(_df, workdf)

        locs_missing = workdf['.PREDICTIONS'].isnull()  # Still missing values after gap-filling
        workdf = workdf[locs_missing].copy()  # Prepare dataframe for next gap-filling
        return workdf, _df

    def _fill_predictions(self, _df, workdf) -> DataFrame:

        # Check where no new predictions are available
        locs_not_available_fills = workdf['.PREDICTIONS'].isnull()
        # The inverse are locations where predictions are available
        locs_available_fills = ~locs_not_available_fills
        # Check where predictions are still needed
        locs_need_fill = _df['.PREDICTIONS'].isnull()
        # Locations where new predictions are available and still needed (both locs_ must be True)
        locs = locs_available_fills & locs_need_fill

        _df.loc[locs, '.PREDICTIONS'] = workdf.loc[locs, '.PREDICTIONS']
        _df.loc[locs, '.PREDICTIONS_QUALITY'] = workdf.loc[locs, '.PREDICTIONS_QUALITY']
        _df.loc[locs, '.START'] = workdf.loc[locs, '.START']
        _df.loc[locs, '.END'] = workdf.loc[locs, '.END']

        return _df

    def _a_1_2(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_[f'.{self.ta}_UPPERLIM'] > row[self.ta])
                & (self.gapfilling_df_[f'.{self.ta}_LOWERLIM'] < row[self.ta])
                & (self.gapfilling_df_[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.gapfilling_df_[f'.{self.swin}_LOWERLIM'] < row[self.swin])
                & (self.gapfilling_df_[f'.{self.vpd}_UPPERLIM'] > row[self.vpd])
                & (self.gapfilling_df_[f'.{self.vpd}_LOWERLIM'] < row[self.vpd])
        )
        avg = self._calc_avg(locs=locs)
        return avg

    def _a3(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_[f'.{self.swin}_UPPERLIM'] > row[self.swin])
                & (self.gapfilling_df_[f'.{self.swin}_LOWERLIM'] < row[self.swin])
        )
        avg = self._calc_avg(locs=locs)
        return avg

    def _a4(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
        )
        avg = self._calc_avg(locs=locs)
        return avg

    def _b1(self, row):
        locs = (
                (self.gapfilling_df_.index >= row['.START'])
                & (self.gapfilling_df_.index <= row['.END'])
                & (self.gapfilling_df_.index.hour == row.name.hour)
        )
        avg = self._calc_avg(locs=locs)
        return avg

    def _calc_avg(self, locs: bool) -> float:
        _df = self.gapfilling_df_.loc[locs, [self.flux, self.swin]].copy()
        _array = _df[self.flux].to_numpy()
        # _array = self.gapfilling_df_.loc[locs, self.flux].to_numpy()
        n_vals = len(_array[~np.isnan(_array)])

        # Return NaN if no flux records available
        if n_vals == 0:
            avg = np.nan
            return avg

        # Check if this is nighttime data
        _swin = _df.copy()
        _swin = _df.dropna()  # Keep records where both flux and SW_IN are available
        _swin = _swin[self.swin].copy()
        _swin = _swin.to_numpy()
        _swin = np.nanmean(_swin)
        # Minimum number of values when building average for nighttime data
        min_n_vals = self.min_n_vals_nt if _swin < 100 else 0

        if n_vals >= min_n_vals:
            # print(n_vals)
            avg = np.nanmean(_array)
        else:
            avg = np.nan
        return avg

    def _add_newcols(self) -> pd.DataFrame:
        df = self.gapfilling_df_.copy()
        # Init new cols
        df['.TIMESTAMP'] = df.index
        df[self.target_gapfilled] = np.nan  # Gap-filling measurement
        df[self.target_gapfilled_flag] = np.nan  # Gap-filling flag
        df['.PREDICTIONS'] = np.nan
        df['.PREDICTIONS_SD'] = np.nan
        df['.PREDICTIONS_COUNTS'] = np.nan
        df['.PREDICTIONS_QUALITY'] = np.nan
        df['.START'] = np.nan
        df['.END'] = np.nan
        df[f'.{self.swin}_LOWERLIM'] = np.nan
        df[f'.{self.swin}_UPPERLIM'] = np.nan
        df[f'.{self.ta}_LOWERLIM'] = np.nan
        df[f'.{self.ta}_UPPERLIM'] = np.nan
        df[f'.{self.vpd}_LOWERLIM'] = np.nan
        df[f'.{self.vpd}_UPPERLIM'] = np.nan

        # Similarity limits for low radiation measurements
        lowrad = df[self.swin] <= 50
        df.loc[lowrad, f'.{self.swin}_LOWERLIM'] = df.loc[lowrad, self.swin].sub(self.swin_class[0])
        df.loc[lowrad, f'.{self.swin}_UPPERLIM'] = df.loc[lowrad, self.swin].add(self.swin_class[0])

        # Similarity limits for high radiation measurements
        highrad = df[self.swin] > 50
        df.loc[highrad, f'.{self.swin}_LOWERLIM'] = df.loc[highrad, self.swin].sub(self.swin_class[1])
        df.loc[highrad, f'.{self.swin}_UPPERLIM'] = df.loc[highrad, self.swin].add(self.swin_class[1])

        df[f'.{self.ta}_LOWERLIM'] = df[self.ta].sub(self.ta_class)
        df[f'.{self.ta}_UPPERLIM'] = df[self.ta].add(self.ta_class)
        df[f'.{self.vpd}_LOWERLIM'] = df[self.vpd].sub(self.vpd_class)
        df[f'.{self.vpd}_UPPERLIM'] = df[self.vpd].add(self.vpd_class)
        return df


def example():
    from diive.core.io.files import load_parquet

    SOURCEDIR = r"L:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2025_2005-2024\notebooks\30_MERGE_DATA"
    FILENAME = r"33.3_CH-CHA_IRGA+QCL+LGR+M10+MGMT_Level-1_eddypro_fluxnet_2005-2024.parquet"
    FILEPATH = Path(SOURCEDIR) / FILENAME
    df = load_parquet(filepath=FILEPATH)

    flux = 'FC'
    ta = 'TA_T1_2_1'
    swin = 'SW_IN_T1_2_1'
    vpd = 'VPD_T1_2_1'
    ustar = 'USTAR'
    ssitc = 'FC_SSITC_TEST'
    swin_class = [25, 25]  # W m-2
    ta_class = 2.5  # °C
    vpd_class = 0.5  # kPa; 5 hPa is default for reference

    locs = (
            (df.index.year >= 2023)
            & (df.index.year <= 2023)
            & (df.index.month >= 7)
            & (df.index.month <= 7)
    )
    subsetcols = [flux, swin, ta, vpd, ustar, ssitc]
    subsetdf = df.loc[locs, subsetcols].copy()
    good = (subsetdf[flux] > -50) & (subsetdf[flux] < 50) & (subsetdf[ustar] > 0.1) & (subsetdf[ssitc] < 1)
    subsetdf.loc[~good, flux] = np.nan
    # subsetdf.describe()

    import time
    a = time.perf_counter()
    mds = FluxMDS(
        df=subsetdf,
        flux=flux,
        ta=ta,
        swin=swin,
        vpd=vpd,
        swin_class=swin_class,
        ta_class=ta_class,
        vpd_class=vpd_class  # kPa; 5 hPa is default for reference
    )
    mds.run()
    mds.report()
    mds.showplot()
    b = time.perf_counter()
    print(f"Duration: {b - a}")


if __name__ == '__main__':
    example()
