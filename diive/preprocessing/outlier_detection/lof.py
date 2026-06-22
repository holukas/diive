"""
LOCAL OUTLIER FACTOR: DENSITY-BASED ANOMALY DETECTION
======================================================

Identify anomalies by measuring local density deviation relative to neighbors.

Part of the diive library: https://github.com/holukas/diive

Local outlier factor:
    "The anomaly score of each sample is called the Local Outlier Factor. It measures
    the local deviation of the density of a given sample with respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object is with
    respect to the surrounding neighborhood. More precisely, locality is given by
    k-nearest neighbors, whose distance is used to estimate the local density. By
    comparing the local density of a sample to the local densities of its neighbors,
    one can identify samples that have a substantially lower density than their neighbors.
    These are considered outliers." - scikit-learn documentation, 6 Jan 2024

Kudos:
    - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    - https://scikit-learn.org/stable/modules/outlier_detection.html
    - https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py
    - https://www.datatechnotes.com/2020/04/anomaly-detection-with-local-outlier-factor-in-python.html

References:
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

See examples/preprocessing/outlier_detection/lof.py for working examples.

"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex, DataFrame
from sklearn.neighbors import LocalOutlierFactor as SKLocalOutlierFactor

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


def lof(series: Series,
        n_neighbors: int = 20,
        contamination: float = 0.01,
        suffix: str = None,
        n_jobs: int = 1) -> DataFrame:
    """Unsupervised Outlier Detection using the Local Outlier Factor (LOF)."""

    if suffix is None:
        suffix = ""
    series = series.copy().dropna()
    ix = series.index
    vals = series.to_numpy().reshape(-1, 1)

    # Run analysis
    lof_detector = SKLocalOutlierFactor(
        n_neighbors=n_neighbors,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination=contamination,
        novelty=False,
        n_jobs=n_jobs
    )
    y_pred = lof_detector.fit_predict(vals)

    # Identify outliers
    outlier_mask = y_pred == -1
    outlier_ix = ix[outlier_mask]
    outlier_vals = series[outlier_mask]

    # Collect in dataframe
    series_clean = pd.Series(index=ix, data=series.to_numpy())
    series_outliers = pd.Series(index=outlier_ix, data=outlier_vals.to_numpy())
    frame = {f"SERIES_{suffix}": series_clean,
             f"OUTLIER_{suffix}": series_outliers}
    df = pd.DataFrame(frame)
    df[f"NOT_OUTLIER_{suffix}"] = df[f"SERIES_{suffix}"].copy()
    df.loc[outlier_ix, f"NOT_OUTLIER_{suffix}"] = np.nan

    return df


def suggest_lof_params(series: Series) -> dict:
    """Recommend a starting set of LocalOutlierFactor parameters for ``series``.

    LOF runs on the values themselves (a 1-D density estimate), so the only
    parameter with a defensible data link is ``n_neighbors``, which must stay below
    the sample count for a valid k-neighbors query. We keep scikit-learn's
    well-tested rule-of-thumb of 20 neighbors and only clamp it to the available
    data (relevant for short series). ``contamination`` is deliberately left at
    ``'auto'``: estimating the outlier fraction from the same data used to detect
    them is circular, which is exactly the case ``'auto'`` is built for.

    Args:
        series: Time series the parameters will be applied to.

    Returns:
        ``{'n_neighbors': int, 'contamination': 'auto'}`` — pass straight to the
        constructor (``LocalOutlierFactor(series=series, **suggest_lof_params(series))``).
    """
    n_valid = int(series.dropna().shape[0])
    n_neighbors = 20
    if n_valid > 1:
        n_neighbors = min(n_neighbors, n_valid - 1)
    return dict(n_neighbors=max(1, n_neighbors), contamination="auto")


@ConsoleOutputDecorator()
class LocalOutlierFactor(FlagBase):
    """Flag outliers using the scikit-learn Local Outlier Factor. See :meth:`__init__`."""

    flagid = "OUTLIER_LOF"

    def __init__(self,
                 series: Series,
                 idstr: str = None,
                 n_neighbors: int = 20,
                 contamination: float = 0.01,
                 separate_daytime_nighttime: bool = False,
                 lat: float = None,
                 lon: float = None,
                 utc_offset: int = None,
                 showplot: bool = False,
                 verbose: bool = False,
                 n_jobs: int = 1):
        """Identify outliers based on the local outlier factor.

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            n_neighbors: Number of neighbors to use by default for kneighbors queries.
                If n_neighbors is larger than the number of samples provided, all samples
                will be used (description taken from scikit, ref [1])
            contamination: The amount of contamination of the data set, i.e. the proportion
                of outliers in the data set. When fitting this is used to define the threshold
                on the scores of the samples.
                - if 'auto', the threshold is determined as in the original paper,
                - if a float, the contamination should be in the range (0, 0.5].
                (description taken from scikit, ref [1])
            separate_daytime_nighttime: If True, run outlier detection separately for daytime
                and nighttime periods. Requires lat, lon, and utc_offset parameters.
            lat: Latitude of location as float (e.g., 46.583056). Required if separate_daytime_nighttime=True.
            lon: Longitude of location as float (e.g., 9.790639). Required if separate_daytime_nighttime=True.
            utc_offset: UTC offset in hours for sun position calculations. Required if separate_daytime_nighttime=True.
            showplot: Show plot with results from the outlier detection.
            verbose: Print more text output.
            n_jobs: The number of parallel jobs to run for neighbors search. None means 1
                unless in a joblib.parallel_backend context. -1 means using all processors.
                (description taken from scikit, ref [1])

        Returns:
            Results dataframe via the @repeater wrapper function, dataframe contains
            the filtered time series and flags from all iterations.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if contamination != 'auto' and not (0 < contamination <= 0.5):
            raise ValueError(f"contamination must be 'auto' or in (0, 0.5], got {contamination}")

        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.showplot = showplot
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.separate_daytime_nighttime = separate_daytime_nighttime

        # Density-based detection has no single data-unit threshold band, so there
        # is nothing to overlay as upper/lower limit lines.
        self.last_upper_bound = None
        self.last_lower_bound = None
        self.is_daytime = None

        if self.separate_daytime_nighttime:
            if lat is None or lon is None or utc_offset is None:
                raise ValueError("lat, lon, and utc_offset are required when separate_daytime_nighttime=True")
            self.flag_daytime, flag_nighttime, self.is_daytime, self.is_nighttime = (
                create_daytime_nighttime_flags(timestamp_index=self.series.index, lat=lat, lon=lon,
                                               utc_offset=utc_offset))

    def calc(self, repeat: bool = True, progress_callback=None):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after each iteration (e.g. to drive a
                progress bar / live-update the cleaned series).

        """

        self._overall_flag, n_iterations = self.repeat(
            self.run_flagtests, repeat=repeat, progress_callback=progress_callback)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)
            if self.separate_daytime_nighttime:
                title = (f"Local outlier factor filter daytime/nighttime: {self.series.name}, "
                         f"n_iterations = {n_iterations}, "
                         f"n_outliers = {self.series[self.overall_flag == 2].count()}")
                self.plot_outlier_daytime_nighttime(series=self.series, flag_daytime=self.flag_daytime,
                                                    flag_quality=self.overall_flag, title=title)

    def _apply_lof_to_subset(self, series: Series, suffix: str = "") -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Apply LOF to a series subset and extract outlier indices.

        Args:
            series: Time series subset to analyze
            suffix: Column name suffix for LOF results

        Returns:
            (ok_indices, rejected_indices, n_outliers)
        """
        if len(series) == 0:
            return series.index, series.index[:0], 0

        lof_df = lof(series=series, n_neighbors=self.n_neighbors,
                     contamination=self.contamination, suffix=suffix, n_jobs=self.n_jobs)

        ok_col = f'NOT_OUTLIER_{suffix}' if suffix else 'NOT_OUTLIER_'
        outlier_col = f'OUTLIER_{suffix}' if suffix else 'OUTLIER_'

        ok_idx = lof_df[ok_col].dropna().index
        rejected_idx = lof_df[outlier_col].dropna().index
        n_outliers = len(rejected_idx)

        return ok_idx, rejected_idx, n_outliers

    def _finalize_flags(self, flag: pd.Series) -> tuple[pd.DataFrame, DatetimeIndex, DatetimeIndex]:
        """Create output dataframe and extract ok/rejected indices.

        Args:
            flag: Flag series with values 0 (ok) or 2 (outlier)

        Returns:
            (output_dataframe, ok_indices, rejected_indices)
        """
        df = pd.DataFrame(self.filteredseries)
        df['FLAG'] = flag
        df['CLEANED'] = df[self.filteredseries.name].copy()
        df.loc[df['FLAG'] > 0, 'CLEANED'] = np.nan

        ok_idx = flag[flag == 0].index
        rejected_idx = flag[flag == 2].index

        return df, ok_idx, rejected_idx

    def _flagtests(self, iteration: int) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag.

        Args:
            iteration: Current iteration number

        Returns:
            (ok_indices, rejected_indices, n_outliers) where:
                - ok_indices: DatetimeIndex of valid (non-outlier) records
                - rejected_indices: DatetimeIndex of outlier records
                - n_outliers: Total number of outliers found
        """
        flag = pd.Series(index=self.filteredseries.index, data=np.nan)

        if self.separate_daytime_nighttime:
            s_daytime = self.filteredseries[self.is_daytime]
            ok_dt, rejected_dt, n_out_dt = self._apply_lof_to_subset(s_daytime, suffix="DAYTIME")

            s_nighttime = self.filteredseries[self.is_nighttime]
            ok_nt, rejected_nt, n_out_nt = self._apply_lof_to_subset(s_nighttime, suffix="NIGHTTIME")

            flag.loc[ok_dt] = 0
            flag.loc[rejected_dt] = 2
            flag.loc[ok_nt] = 0
            flag.loc[rejected_nt] = 2

            if self.verbose:
                n_total = len(rejected_dt) + len(rejected_nt)
                detail(f"ITERATION#{iteration}", verbose=self.verbose)
                detail(f"Total found outliers: {len(rejected_dt)} values (daytime)", verbose=self.verbose)
                detail(f"Total found outliers: {len(rejected_nt)} values (nighttime)", verbose=self.verbose)
                detail(f"Total found outliers: {n_total} values (daytime+nighttime)", verbose=self.verbose)

        else:
            ok, rejected, n_out = self._apply_lof_to_subset(self.filteredseries, suffix="")
            flag.loc[ok] = 0
            flag.loc[rejected] = 2

            if self.verbose:
                detail(f"ITERATION#{iteration}: Total found outliers: {n_out} values (global)", verbose=self.verbose)

        _, ok, rejected = self._finalize_flags(flag)
        n_outliers = (flag == 2).sum()

        return ok, rejected, n_outliers


# Backward compatibility aliases
LocalOutlierFactorAllData = LocalOutlierFactor
LocalOutlierFactorDaytimeNighttime = LocalOutlierFactor
