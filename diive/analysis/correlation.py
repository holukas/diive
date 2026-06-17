"""
ANALYSIS: CORRELATION
======================

Daily and lagged correlation analysis for time series relationships.
Detect patterns, anomalies, and decoupling between variables.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from matplotlib import gridspec, pyplot as plt
from pandas import DataFrame, Series
from scipy import stats


def rank_drivers(
        df: DataFrame,
        target: str,
        features: list = None,
        method: str = 'pearson',
        max_lag: int = 0,
        min_records: int = 30,
) -> DataFrame:
    """Rank candidate variables by how strongly they correlate with a target.

    For each candidate column, computes its correlation with ``target``. When
    ``max_lag > 0``, the candidate is shifted by every lag in
    ``[-max_lag, +max_lag]`` (in records) and the lag with the strongest
    *absolute* correlation is reported — so lead/lag relationships (storage,
    advection, phenology) surface rather than only the contemporaneous one.

    Pure ranking utility: no plotting, no side effects. Intended for "what
    drives this variable?" exploration and for shortlisting gap-filling
    features.

    Args:
        df: DataFrame holding the target and candidate columns (DatetimeIndex
            when ``max_lag > 0``, so a shift equals a fixed time step).
        target: Target column name.
        features: Candidate columns to rank. Default: every numeric column
            except the target. Non-existent names and the target are dropped.
        method: ``'pearson'`` (linear) or ``'spearman'`` (rank/monotonic).
        max_lag: Maximum absolute lag in records to scan (0 = contemporaneous
            only). A positive ``BEST_LAG`` means the driver *leads* the target —
            its value that many records earlier aligns best with the target.
        min_records: Minimum overlapping non-NaN pairs required at a lag;
            candidates that never reach it are dropped.

    Returns:
        DataFrame (index 0..n-1) with columns ``DRIVER``, ``CORR`` (signed
        correlation at the best lag), ``ABS_CORR``, ``BEST_LAG`` (records),
        ``N`` (overlapping pairs), sorted by ``ABS_CORR`` descending. Empty when
        the target is constant/all-NaN or no candidate qualifies.

    Example:
        >>> ranked = rank_drivers(df, target='NEE', method='pearson', max_lag=4)
        >>> ranked.head()  # strongest drivers first
    """
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in df columns.")
    if method not in ('pearson', 'spearman'):
        raise ValueError("method must be 'pearson' or 'spearman'.")
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0.")

    y = pd.to_numeric(df[target], errors='coerce')

    if features is None:
        numeric = df.select_dtypes(include='number').columns.tolist()
        features = [c for c in numeric if c != target]
    else:
        features = [c for c in features if c != target and c in df.columns]

    lags = range(-max_lag, max_lag + 1)
    rows = []
    for col in features:
        x = pd.to_numeric(df[col], errors='coerce')
        best = None  # (abs_corr, corr, lag, n)
        for lag in lags:
            xs = x.shift(lag) if lag else x
            mask = y.notna() & xs.notna()
            n = int(mask.sum())
            if n < min_records:
                continue
            # A constant column over the overlap divides by a zero std inside
            # the correlation -> NaN (which we skip); silence that numpy warning.
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = y[mask].corr(xs[mask], method=method)
            if pd.isna(corr):
                continue
            abs_corr = abs(float(corr))
            if best is None or abs_corr > best[0]:
                best = (abs_corr, float(corr), lag, n)
        if best is not None:
            rows.append({'DRIVER': col, 'CORR': best[1], 'ABS_CORR': best[0],
                         'BEST_LAG': best[2], 'N': best[3]})

    result = pd.DataFrame(rows, columns=['DRIVER', 'CORR', 'ABS_CORR', 'BEST_LAG', 'N'])
    if not result.empty:
        result = result.sort_values('ABS_CORR', ascending=False).reset_index(drop=True)
    return result


class DailyCorrelation:
    """Calculate daily correlation between two time series.

    Computes correlation for each day and provides tools for analysis:
    - Access correlations via `.correlations` property or `.summary()`
    - Identify best/worst correlation days with `.get_days_by_correlation()`
    - Detect anomalous days with `.detect_anomalies()`
    - Visualize results with `.plot()`

    Args:
        s1: any time series, timestamp must overlap with *s2*
        s2: any time series, timestamp must overlap with *s1*
        mincorr: minimum absolute correlation for plot thresholds,
            must be between -1 and 1 (inclusive).
            Example: with *0.8*, correlations between -0.8 and +0.8 are low,
            correlations smaller than -0.8 and higher than +0.8 are high.

    Attributes:
        daycorrs_: Series with correlations for each day
        df_: Combined DataFrame with both input series and date column

    Properties:
        correlations: Daily correlations as pandas Series

    Methods:
        summary(): Comprehensive statistics (mean, median, skewness, kurtosis, normality)
        get_days_by_correlation(high): Days sorted by correlation strength
        detect_anomalies(method, threshold): Identify outlier correlation days
        plot(): Interactive visualization with correlation distribution and day examples

    Example:
        See `examples/analysis/analysis_daily_correlation.py` for comprehensive examples covering:
        quality checks (observed vs. potential radiation), physical relationships (temperature vs. radiation),
        biological processes (temperature vs. CO2 flux), and advanced methods like summary statistics
        and anomaly detection.
    """

    def __init__(self, s1: Series, s2: Series, mincorr: float = 0.8):
        if not (-1 <= mincorr <= 1):
            raise ValueError("mincorr must be between -1 and 1.")
        if s1.name is None or s2.name is None:
            raise ValueError("s1 and s2 must have names (set Series.name).")
        if s1.name == s2.name:
            raise ValueError(f"s1 and s2 must have different names, both are '{s1.name}'.")

        self.s1 = s1
        self.s2 = s2
        self.mincorr = abs(mincorr)

        df = pd.concat([s1, s2], axis=1)
        df['DATE'] = df.index.date.astype(str)

        daycorrs = (
            df.groupby('DATE')
            .apply(lambda g: g[s1.name].corr(g[s2.name]), include_groups=False)
            .rename('daycorrs')
        )
        daycorrs.index = pd.to_datetime(daycorrs.index)
        daycorrs = daycorrs.asfreq('1D')

        self.daycorrs_ = daycorrs
        self.df_ = df

    @property
    def result(self) -> Series:
        """Primary result: daily correlations as a pandas Series."""
        return self.daycorrs_

    @property
    def correlations(self) -> Series:
        """Daily correlations as a pandas Series."""
        return self.daycorrs_

    def summary(self) -> dict:
        """Get comprehensive summary statistics of correlations.

        Returns:
            dict with:
            - count: number of valid (non-NaN) days
            - median: median correlation
            - mean: mean correlation
            - std: standard deviation
            - min: minimum correlation
            - max: maximum correlation
            - p1: 1st percentile
            - p99: 99th percentile
            - skewness: distribution skewness (-1 to +1)
            - kurtosis: distribution kurtosis (heavy/light tails)
            - normality_statistic: Shapiro-Wilk test statistic
            - normality_pvalue: p-value (>0.05 suggests normal distribution)
            All float fields are NaN when no valid days exist.
        """
        daycorrs_clean = self.daycorrs_.dropna()
        n = len(daycorrs_clean)

        if n == 0:
            nan = float('nan')
            return {
                'count': 0, 'median': nan, 'mean': nan, 'std': nan,
                'min': nan, 'max': nan, 'p1': nan, 'p99': nan,
                'skewness': nan, 'kurtosis': nan,
                'normality_statistic': nan, 'normality_pvalue': nan,
            }

        stat, pvalue = stats.shapiro(daycorrs_clean) if n >= 3 else (float('nan'), float('nan'))
        desc = daycorrs_clean.describe(percentiles=[0.01, 0.5, 0.99])

        return {
            'count': int(desc['count']),
            'median': float(desc['50%']),
            'mean': float(desc['mean']),
            'std': float(desc['std']),
            'min': float(desc['min']),
            'max': float(desc['max']),
            'p1': float(desc['1%']),
            'p99': float(desc['99%']),
            'skewness': float(daycorrs_clean.skew()),
            'kurtosis': float(daycorrs_clean.kurt()),
            'normality_statistic': float(stat),
            'normality_pvalue': float(pvalue),
        }

    def get_days_by_correlation(self, high: bool = True) -> pd.DataFrame:
        """Get days sorted by correlation strength.

        Args:
            high: if True, return highest correlations first (descending).
                  if False, return lowest correlations first (ascending).

        Returns:
            DataFrame with columns ['date', 'correlation'] sorted by strength.
            Days with no valid correlation (NaN) are excluded.
        """
        sorted_corrs = self.daycorrs_.dropna().sort_values(key=abs, ascending=not high)
        return pd.DataFrame({
            'date': sorted_corrs.index,
            'correlation': sorted_corrs.to_numpy()
        }).reset_index(drop=True)

    def detect_anomalies(self, method: str = 'zscore', threshold: float = 2.0) -> pd.DataFrame:
        """Identify anomalous correlation days.

        Args:
            method: detection method:
                - 'zscore': z-score based (default, detects outliers)
                - 'iqr': interquartile range method (robust to extreme outliers)
            threshold: sensitivity threshold:
                - zscore: standard deviations from mean (default 2.0 = ~95% confidence)
                - iqr: multiplier for IQR (default 2.0 = moderate sensitivity)

        Returns:
            DataFrame with columns ['date', 'correlation', 'anomaly_score', 'is_anomaly']
            sorted by anomaly score (highest first).

        Example:
            Dates with z-score > 2.0 are flagged as anomalies (unusual correlation days).
        """
        clean = self.daycorrs_.dropna()

        if method == 'zscore':
            std = clean.std()
            if std == 0:
                scores = pd.Series(0.0, index=clean.index)
                is_anomaly = pd.Series(False, index=clean.index)
            else:
                scores = (clean - clean.mean()) / std
                is_anomaly = abs(scores) > threshold
        elif method == 'iqr':
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                scores = pd.Series(0.0, index=clean.index)
                is_anomaly = pd.Series(False, index=clean.index)
            else:
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                is_anomaly = (clean < lower) | (clean > upper)
                scores = (clean - clean.median()) / iqr
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'.")

        df = pd.DataFrame({
            'date': clean.index,
            'correlation': clean.to_numpy(),
            'anomaly_score': abs(scores).to_numpy(),
            'is_anomaly': is_anomaly.to_numpy()
        })
        return df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)

    def _plot_example_days(self, groups, axes: list, daycorrs: Series) -> None:
        """Plot time series for up to len(axes) example days; hides unused axes."""
        used = 0
        for day, day_df in groups:
            if used >= len(axes):
                break
            day_df = day_df.copy()
            day_df.index = day_df.index.time
            day_df[[self.s2.name, self.s1.name]].plot(ax=axes[used])
            axes[used].set_title(f"{day}, r = {daycorrs[day]:.3f}")
            used += 1
        for ax in axes[used:]:
            ax.set_visible(False)

    def plot(self, showplot: bool = True) -> plt.Figure:
        """Display daily correlation analysis plot.

        Args:
            showplot: if True, call plt.show() to display the figure interactively.
                Set to False when rendering in non-interactive environments (Sphinx Gallery,
                headless testing).

        Returns:
            matplotlib Figure.
        """
        daycorrs = self.daycorrs_
        df = self.df_
        s1 = self.s1
        s2 = self.s2
        mincorr = self.mincorr

        daycorrs_valid = daycorrs.dropna()
        _lowcorrs = daycorrs_valid.between(-mincorr, mincorr, inclusive='neither')
        lowcorrs_series = daycorrs_valid[_lowcorrs].sort_values(key=abs, ascending=True)
        highcorrs_series = daycorrs_valid[~_lowcorrs].sort_values(key=abs, ascending=False)

        lowcorr_dates = lowcorrs_series.index.astype(str).to_list()
        lowestcorr_dates = lowcorrs_series.head(3).index.astype(str).to_list()
        highestcorr_dates = highcorrs_series.head(3).index.astype(str).to_list()

        lowdays = df['DATE'].isin(lowcorr_dates)
        lowestdays = df['DATE'].isin(lowestcorr_dates)
        highestdays = df['DATE'].isin(highestcorr_dates)

        fig = plt.figure(facecolor='white', figsize=(8, 12), dpi=100)
        gs = gridspec.GridSpec(4, 3)
        gs.update(wspace=0.3, hspace=0.4, left=0.05, right=0.97, top=0.9, bottom=0.1)
        ax1 = fig.add_subplot(gs[0, 0:])
        ax2 = fig.add_subplot(gs[1, 0:])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        ax5 = fig.add_subplot(gs[2, 2])
        ax6 = fig.add_subplot(gs[3, 0])
        ax7 = fig.add_subplot(gs[3, 1])
        ax8 = fig.add_subplot(gs[3, 2])

        daycorrs.plot(
            ax=ax1,
            title=(
                f"Correlation between {s2.name} and {s1.name} per day "
                f"(n = {daycorrs.count()})\n"
                f"median = {daycorrs.median():.3f}, "
                f"99th percentile = {daycorrs.quantile(.99):.3f}, "
                f"1st percentile = {daycorrs.quantile(.01):.3f}, "
                f"min / max = {daycorrs.min():.3f} / {daycorrs.max():.3f}"
            )
        )
        ax1.axhline(-mincorr, c='#ff0051')
        ax1.axhline(mincorr, c='#ff0051')
        ax1.set_ylim(-1, 1)

        for day, day_df in df[lowdays].groupby(df[lowdays]['DATE']):
            day_df = day_df.copy()
            day_df.index = day_df.index.time
            day_df[[s2.name, s1.name]].plot(ax=ax2, legend=False, alpha=.3, color='grey')
        ax2.set_title(f"Found {len(lowcorr_dates)} low correlation days")

        self._plot_example_days(
            df[lowestdays].groupby(df[lowestdays]['DATE']),
            [ax3, ax4, ax5], daycorrs
        )
        self._plot_example_days(
            df[highestdays].groupby(df[highestdays]['DATE']),
            [ax6, ax7, ax8], daycorrs
        )

        fig.suptitle(f"Comparison between {s1.name} and {s2.name}")
        if showplot:
            plt.show()
        return fig
