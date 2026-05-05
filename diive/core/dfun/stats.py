# import diive.pkgs.dfun
# from stats.boxes import insert_statsboxes_txt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy import stats as scipy_stats

from diive.core.funcs.funcs import zscore


def q75(x):
    return x.quantile(0.75)


def q50(x):
    return x.quantile(0.5)


def q25(x):
    return x.quantile(0.25)


def q99(x):
    return x.quantile(0.99)


def q01(x):
    return x.quantile(0.01)


def q05(x):
    return x.quantile(0.05)


def q95(x):
    return x.quantile(0.95)


def series_start(series: pd.Series, dtformat: str = "%Y-%m-%d %H:%M"):
    """Return start datetime of series"""
    return series.index[0].strftime(dtformat)


def series_end(series: pd.Series, dtformat: str = "%Y-%m-%d %H:%M"):
    """Return end datetime of series"""
    return series.index[-1].strftime(dtformat)


def series_duration(series: pd.Series):
    """Return duration of series"""
    return series.index[-1] - series.index[0]


def series_numvals(series: pd.Series):
    """Return number of values in series"""
    return series.count()


def series_numvals_missing(series: pd.Series):
    """Return number of missing values in series"""
    return series.isnull().sum()


def series_perc_missing(series: pd.Series):
    """Return number of missing values in series as percentage"""
    return (series_numvals_missing(series) / len(series.index)) * 100


def series_sd_over_mean(series: pd.Series):
    """Return sd / mean"""
    return series.std() / series.mean()


# ============================================================
# TIME SERIES METRICS
# ============================================================

def coefficient_of_variation(s: Series) -> float:
    """Coefficient of variation (CV = SD / Mean).

    Relative variability measure standardized by the mean.
    Useful for comparing variability across different scales or units.
    """
    mean_val = s.mean()
    if mean_val == 0:
        return np.nan
    return s.std() / abs(mean_val)


def interquartile_range(s: Series) -> float:
    """Interquartile range (IQR = Q3 - Q1).

    Robust measure of spread that is resistant to outliers.
    Represents the range containing the middle 50% of values.
    """
    return s.quantile(0.75) - s.quantile(0.25)


def autocorrelation_lag1(s: Series) -> float:
    """Autocorrelation at lag-1.

    Measures correlation of series with itself shifted by one time step.
    Indicates how predictable the next value is from the current value.
    Range: [-1, 1] where 1 = perfect positive correlation, -1 = perfect negative.
    """
    s_clean = s.dropna()
    if len(s_clean) < 2:
        return np.nan
    return s_clean.autocorr(lag=1)


def series_skewness(s: Series) -> float:
    """Fisher-Pearson skewness (distribution asymmetry).

    Measures asymmetry of distribution:
    - > 0: right-skewed (tail on right, most values on left)
    - < 0: left-skewed (tail on left, most values on right)
    - ≈ 0: approximately symmetric
    """
    return scipy_stats.skew(s.dropna(), bias=False)


def series_kurtosis(s: Series) -> float:
    """Excess kurtosis (tail weight relative to normal distribution).

    Measures heaviness of distribution tails:
    - > 0: heavy tails (leptokurtic, more outliers than normal)
    - < 0: light tails (platykurtic, fewer outliers than normal)
    - ≈ 0: similar to normal distribution
    """
    return scipy_stats.kurtosis(s.dropna(), bias=False)


def mean_absolute_change(s: Series) -> float:
    """Mean absolute change between consecutive values.

    Measures average magnitude of changes over time.
    Higher values indicate more volatile/variable series.
    Lower values indicate smoother, more stable trends.
    """
    s_clean = s.dropna()
    if len(s_clean) < 2:
        return np.nan
    return s_clean.diff().abs().mean()


def mean_absolute_pct_change(s: Series) -> float:
    """Mean absolute percentage change between consecutive values.

    Normalized version of mean absolute change.
    Useful when the series has varying magnitude levels.
    """
    s_clean = s.dropna()
    if len(s_clean) < 2:
        return np.nan
    pct_change = s_clean.pct_change().abs()
    # Handle inf values from division by zero
    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
    return pct_change.mean()


def outlier_count(s: Series, threshold: float = 3, method: str = 'zscore') -> int:
    """Count statistical outliers.

    Args:
        s: Series to analyze
        threshold: Detection threshold
            - zscore: values > threshold * sigma (default 3 = 3-sigma)
            - iqr: values > Q3 + threshold * IQR (default 1.5)
            - mad: values > median + threshold * MAD (default 3)
        method: Detection method ('zscore', 'iqr', 'mad')

    Returns:
        Count of outlier values
    """
    s_clean = s.dropna()

    if method == 'zscore':
        z_scores = np.abs(scipy_stats.zscore(s_clean))
        return int((z_scores > threshold).sum())

    elif method == 'iqr':
        Q1 = s_clean.quantile(0.25)
        Q3 = s_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return int(((s_clean < lower) | (s_clean > upper)).sum())

    elif method == 'mad':
        median = s_clean.median()
        mad = np.median(np.abs(s_clean - median))
        if mad == 0:
            return 0
        return int((np.abs(s_clean - median) > threshold * mad).sum())

    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'iqr', or 'mad'")


def outlier_percentage(s: Series, threshold: float = 3, method: str = 'zscore') -> float:
    """Outlier count as percentage of total values."""
    return (outlier_count(s, threshold, method) / len(s.dropna())) * 100


def cumulative_sum(s: Series) -> float:
    """Total integrated value (cumulative sum).

    Useful for understanding total accumulation over the time period.
    For time series with positive and negative values, represents net change.
    """
    return s.sum()


def linear_trend_slope(s: Series) -> tuple:
    """Linear regression slope (trend direction and rate).

    Returns:
        (slope, intercept, r_value) tuple

    slope: direction and rate of change per time step
        > 0: upward trend
        < 0: downward trend
        ≈ 0: relatively stable
    r_value: correlation coefficient (goodness of fit) [-1, 1]
        > 0.7 or < -0.7: strong fit
        0.3 to 0.7: moderate fit
        < 0.3: weak fit
    """
    s_clean = s.dropna()
    if len(s_clean) < 2:
        return (np.nan, np.nan, np.nan)

    x = np.arange(len(s_clean))
    z = np.polyfit(x, s_clean.values, 1)
    p = np.poly1d(z)

    # Calculate R-value
    y_pred = p(x)
    ss_res = np.sum((s_clean.values - y_pred) ** 2)
    ss_tot = np.sum((s_clean.values - s_clean.mean()) ** 2)
    r_value = np.sqrt(1 - (ss_res / ss_tot)) if ss_tot != 0 else np.nan

    return (z[0], z[1], r_value)


def approximate_entropy(s: Series, m: int = 2, r: float = None) -> float:
    """Approximate entropy (regularity/randomness metric).

    Measures complexity and predictability of time series.
    Low entropy: predictable, regular patterns
    High entropy: random, irregular variations

    Args:
        s: Series to analyze
        m: Embedding dimension (default 2)
        r: Tolerance (default: 0.2 * std)

    Returns:
        Approximate entropy value
    """
    s_clean = s.dropna().values
    N = len(s_clean)

    if N < m + 1:
        return np.nan

    if r is None:
        r = 0.2 * s_clean.std()

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[s_clean[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    return abs(_phi(m) - _phi(m + 1))


def sstats(s: Series) -> DataFrame:
    """
    Calculate comprehensive time series statistics.

    Returns 30 metrics across 8 categories:

    TEMPORAL INFORMATION
    --------------------
    STARTDATE : str
        First timestamp in series (format: YYYY-MM-DD HH:MM)
    ENDDATE : str
        Last timestamp in series (format: YYYY-MM-DD HH:MM)
    PERIOD : timedelta
        Total time span covered by series

    VALUE COUNTS
    -----------
    NOV : int
        Number of valid (non-missing) observations
    MISSING : int
        Number of missing (NaN) values
    MISSING_PERC : float (%)
        Percentage of missing values

    CENTRAL TENDENCY (Location)
    ---------------------------
    MEAN : float
        Arithmetic average of all values
    MEDIAN : float
        Middle value when sorted (50th percentile)
    SUM : float
        Total sum of all values (integration over time period)

    DISPERSION (Spread) - Basic
    --------------------------
    SD : float
        Standard deviation - spread around mean
    VAR : float
        Variance - squared standard deviation
    SD/MEAN : float
        Ratio of standard deviation to mean (can be negative)

    RANGE & BOUNDS
    --------------
    MIN : float
        Minimum value in series
    MAX : float
        Maximum value in series

    DISPERSION (Spread) - Robust
    --------------------------
    CV : float
        Coefficient of variation (SD / |Mean|)
        Relative variability, independent of scale. Use for comparing different series.
    IQR : float
        Interquartile range (Q3 - Q1)
        Robust measure resistant to outliers. Represents middle 50% of data.

    DISTRIBUTION SHAPE
    ------------------
    SKEWNESS : float
        Fisher-Pearson skewness coefficient
        > 0: right-skewed (tail extends right, most values clustered left)
        < 0: left-skewed (tail extends left, most values clustered right)
        ≈ 0: symmetric distribution
    KURTOSIS : float
        Excess kurtosis (relative to normal distribution)
        > 0: heavy tails - more extreme values than normal
        < 0: light tails - fewer extreme values than normal
        ≈ 0: similar tail weight to normal distribution

    TEMPORAL DYNAMICS
    -----------------
    MEAN_ABS_CHANGE : float
        Average absolute change between consecutive time steps
        High values: volatile, rapidly changing series
        Low values: smooth, stable trends
    TREND_SLOPE : float
        Linear regression slope (change per time step)
        > 0: upward trend
        < 0: downward trend
        ≈ 0: stable (flat trend)
    ACF_LAG1 : float
        Autocorrelation at lag-1 (correlation with previous time step)
        Range: [-1, 1]
        > 0.7: strong temporal dependence (predictable)
        0.3-0.7: moderate dependence
        < 0.3: weak dependence (more random)

    QUALITY METRICS
    ---------------
    OUTLIER_COUNT : int
        Number of statistical outliers detected (>3-sigma by default)
        Helps identify measurement errors or extreme events
    OUTLIER_PERC : float (%)
        Percentage of values that are statistical outliers

    INTEGRATION
    -----------
    CUMSUM : float
        Total integrated value (sum of all values)
        For non-negative series: total accumulation
        For series with pos/neg: net change over period

    PERCENTILES (Distribution quantiles)
    ------------------------------------
    P01, P05, P25, P75, P95, P99 : float
        1st, 5th, 25th, 75th, 95th, 99th percentiles
        Define the distribution at key points.
        P25-P75 range captures middle 50% (similar to IQR).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with 30 statistics as rows

    Example
    -------
    >>> series = pd.Series([1.2, 1.5, 1.3, 1.8, 1.6, ...])
    >>> stats = sstats(series)
    >>> print(stats)
    """
    col = s.name
    df = pd.DataFrame(columns=[col])

    # === TEMPORAL INFORMATION ===
    df.loc['STARTDATE', col] = series_start(s)
    df.loc['ENDDATE', col] = series_end(s)
    df.loc['PERIOD', col] = series_duration(s)

    # === VALUE COUNTS ===
    df.loc['NOV', col] = series_numvals(s)
    df.loc['MISSING', col] = series_numvals_missing(s)
    df.loc['MISSING_PERC', col] = series_perc_missing(s)

    # === CENTRAL TENDENCY ===
    df.loc['MEAN', col] = s.mean()
    df.loc['MEDIAN', col] = s.quantile(q=0.50)

    # === DISPERSION (Basic) ===
    df.loc['SD', col] = s.std()
    df.loc['VAR', col] = s.var()
    df.loc['SD/MEAN', col] = series_sd_over_mean(s)

    # === RANGE & BOUNDS ===
    df.loc['SUM', col] = s.sum()
    df.loc['MIN', col] = s.min()
    df.loc['MAX', col] = s.max()

    # === DISPERSION (Robust) ===
    df.loc['CV', col] = coefficient_of_variation(s)
    df.loc['IQR', col] = interquartile_range(s)

    # === DISTRIBUTION SHAPE ===
    df.loc['SKEWNESS', col] = series_skewness(s)
    df.loc['KURTOSIS', col] = series_kurtosis(s)

    # === TEMPORAL DYNAMICS ===
    df.loc['MEAN_ABS_CHANGE', col] = mean_absolute_change(s)
    df.loc['TREND_SLOPE', col] = linear_trend_slope(s)[0]
    df.loc['ACF_LAG1', col] = autocorrelation_lag1(s)

    # === QUALITY METRICS ===
    df.loc['OUTLIER_COUNT', col] = outlier_count(s)
    df.loc['OUTLIER_PERC', col] = outlier_percentage(s)

    # === INTEGRATION ===
    df.loc['CUMSUM', col] = cumulative_sum(s)

    # === PERCENTILES ===
    df.loc['P01', col] = s.quantile(q=0.01)
    df.loc['P05', col] = s.quantile(q=0.05)
    df.loc['P25', col] = s.quantile(q=0.25)
    df.loc['P75', col] = s.quantile(q=0.75)
    df.loc['P95', col] = s.quantile(q=0.95)
    df.loc['P99', col] = s.quantile(q=0.99)

    return df


def sstats_doublediff_abs(s: Series) -> DataFrame:
    """Calculate stats for absolute double difference of series."""
    doublediff_abs, diff_to_prev_abs, diff_to_next_abs = double_diff_absolute(s=s)
    df = sstats(s=doublediff_abs)
    return df


def sstats_zscore(s: Series) -> DataFrame:
    """Calculate stats for z-scores of series."""
    z = zscore(series=s)
    df = sstats(s=z)
    return df


def double_diff_absolute(s: Series) -> tuple[Series, Series, Series]:
    """Calculate the absolute sum of differences between a data point and
    the respective preceding and next value."""
    shifted_prev = s.shift(1)
    diff_to_prev = s - shifted_prev
    diff_to_prev_abs = diff_to_prev.abs()
    shifted_next = s.shift(-1)
    diff_to_next = s - shifted_next
    diff_to_next_abs = diff_to_next.abs()
    doublediff_abs = diff_to_prev_abs + diff_to_next_abs
    # dd_abs = dd_abs ** 2
    doublediff_abs.name = 'DOUBLE_DIFF_ABS'
    return doublediff_abs, diff_to_prev_abs, diff_to_next_abs


def example():
    """Demonstrate comprehensive time series statistics."""
    from diive.configs.exampledata import load_exampledata_parquet

    print("=" * 80)
    print("TIME SERIES STATISTICS EXAMPLE")
    print("=" * 80)
    print()

    # Load example data
    df = load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()

    # Display statistics
    print(sstats(series))


if __name__ == '__main__':
    example()
