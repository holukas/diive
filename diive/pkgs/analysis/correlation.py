import pandas as pd
from matplotlib import pyplot as plt, gridspec as gridspec
from pandas import Series
from scipy import stats


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
        See `examples/analyses/correlation.py` for complete examples.
    """

    def __init__(self, s1: Series, s2: Series, mincorr: float = 0.8):
        if not (-1 <= mincorr <= 1):
            raise ValueError("mincorr must be between -1 and 1.")

        self.s1 = s1
        self.s2 = s2
        self.mincorr = abs(mincorr)

        # Calculate daily correlations
        df = pd.concat([s1, s2], axis=1)
        df['DATE'] = df.index.date.astype(str)

        groups = df.groupby('DATE')
        daycorrs_index = groups.count().index
        daycorrs = pd.Series(index=daycorrs_index, name='daycorrs')

        for day, day_df in groups:
            corr = day_df[s1.name].corr(day_df[s2.name])
            daycorrs.loc[day] = corr

        daycorrs.index = pd.to_datetime(daycorrs.index)
        daycorrs = daycorrs.asfreq('1d')

        self.daycorrs_ = daycorrs
        self.df_ = df

    @property
    def correlations(self) -> Series:
        """Daily correlations as a pandas Series."""
        return self.daycorrs_

    def summary(self) -> dict:
        """Get comprehensive summary statistics of correlations.

        Returns:
            dict with:
            - count: number of days
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
        """
        daycorrs_clean = self.daycorrs_.dropna()

        # Shapiro-Wilk normality test
        if len(daycorrs_clean) >= 3:
            stat, pvalue = stats.shapiro(daycorrs_clean)
        else:
            stat, pvalue = float('nan'), float('nan')

        return {
            'count': len(self.daycorrs_),
            'median': float(self.daycorrs_.median()),
            'mean': float(self.daycorrs_.mean()),
            'std': float(self.daycorrs_.std()),
            'min': float(self.daycorrs_.min()),
            'max': float(self.daycorrs_.max()),
            'p1': float(self.daycorrs_.quantile(0.01)),
            'p99': float(self.daycorrs_.quantile(0.99)),
            'skewness': float(stats.skew(daycorrs_clean)),
            'kurtosis': float(stats.kurtosis(daycorrs_clean)),
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
        """
        sorted_corrs = self.daycorrs_.sort_values(
            key=abs, ascending=not high
        )
        df = pd.DataFrame({
            'date': sorted_corrs.index,
            'correlation': sorted_corrs.values
        })
        return df.reset_index(drop=True)

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
        if method == 'zscore':
            scores = (self.daycorrs_ - self.daycorrs_.mean()) / self.daycorrs_.std()
            is_anomaly = abs(scores) > threshold
        elif method == 'iqr':
            q1 = self.daycorrs_.quantile(0.25)
            q3 = self.daycorrs_.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            is_anomaly = (self.daycorrs_ < lower) | (self.daycorrs_ > upper)
            scores = (self.daycorrs_ - q1) / iqr  # Normalized score
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'.")

        df = pd.DataFrame({
            'date': self.daycorrs_.index,
            'correlation': self.daycorrs_.values,
            'anomaly_score': abs(scores).values,
            'is_anomaly': is_anomaly.values
        })

        # Sort by anomaly score (highest first)
        df = df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)

        return df

    def plot(self):
        """Display daily correlation analysis plot."""
        daycorrs = self.daycorrs_
        df = self.df_
        s1 = self.s1
        s2 = self.s2
        mincorr = self.mincorr

        # Identify dates with low correlation
        _lowcorrs = daycorrs.between(-mincorr, mincorr, inclusive='neither')
        lowcorrs = daycorrs[_lowcorrs]
        lowcorrs = lowcorrs.sort_values(key=abs, ascending=True)
        lowestcorrs = lowcorrs.head(3)
        lowcorrs = lowcorrs.index.astype(str).to_list()
        lowdays = df['DATE'].isin(lowcorrs)

        # Identify dates with high correlation
        highcorrs = daycorrs[~_lowcorrs]
        highcorrs = highcorrs.sort_values(key=abs, ascending=False)
        highestcorrs = highcorrs.head(3)
        highestcorrs = highestcorrs.index.astype(str).to_list()
        highestdays = df['DATE'].isin(highestcorrs)

        # Identify dates with lowest correlation
        lowestcorrs = lowestcorrs.index.astype(str).to_list()
        lowestdays = df['DATE'].isin(lowestcorrs)

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
            ax=ax1, title=f"Correlation between {s2.name} and {s1.name} per day "
                          f"(n = {len(daycorrs)})\ncorrelation "
                          f"median = {daycorrs.median():.3f}, "
                          f"99th percentile = {daycorrs.quantile(.99):.3f} "
                          f"1st percentile = {daycorrs.quantile(.01):.3f}, "
                          f"min / max = {daycorrs.min():.3f} / {daycorrs.max():.3f} "
        )
        ax1.axhline(-mincorr, c='#ff0051')
        ax1.axhline(mincorr, c='#ff0051')
        ax1.set_ylim(-1, 1)

        # Get full resolution data for low-correlation days
        lowdays_fullres = df[lowdays].copy()
        groups2 = lowdays_fullres.groupby(lowdays_fullres['DATE'])
        for day, day_df in groups2:
            day_df.index = day_df.index.time
            day_df[[s2.name, s1.name]].plot(ax=ax2, legend=False, alpha=.3, color='grey')
        ax2.set_title(f"Found {len(lowcorrs)} low correlation days")

        # Get full resolution data for lowest-correlation days
        lowestdays_fullres = df[lowestdays].copy()
        groups3 = lowestdays_fullres.groupby(lowestdays_fullres['DATE'])
        axes = [ax3, ax4, ax5]
        counter = 0
        for day, day_df in groups3:
            day_df.index = day_df.index.time
            day_df[[s2.name, s1.name]].plot(ax=axes[counter])
            axes[counter].set_title(f"{day}, r = {daycorrs[day]:.3f}")
            counter += 1

        # Get full resolution data for highest-correlation days
        highestdays_fullres = df[highestdays].copy()
        groups4 = highestdays_fullres.groupby(highestdays_fullres['DATE'])
        axes = [ax6, ax7, ax8]
        counter = 0
        for day, day_df in groups4:
            day_df.index = day_df.index.time
            day_df[[s2.name, s1.name]].plot(ax=axes[counter])
            axes[counter].set_title(f"{day}, r = {daycorrs[day]:.3f}")
            counter += 1

        fig.suptitle(f"Comparison between {s1.name} and {s2.name}")
        fig.show()


def daily_correlation(s1: Series,
                      s2: Series,
                      mincorr: float = 0.8,
                      showplot: bool = False) -> Series:
    """Calculate daily correlation between two time series.

    Convenience function. See DailyCorrelation class for class-based API.

    Args:
        s1: any time series, timestamp must overlap with *s2*
        s2: any time series, timestamp must overlap with *s1*
        mincorr: minimum absolute correlation threshold
        showplot: if *True*, display plot of results

    Returns:
        series with correlations for each day

    Example:
        See `examples/analyses/correlation.py` for complete examples.
    """
    dc = DailyCorrelation(s1=s1, s2=s2, mincorr=mincorr)
    if showplot:
        dc.plot()
    return dc.daycorrs_
