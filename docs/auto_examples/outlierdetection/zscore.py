"""
Examples: Z-Score Outlier Detection

Z-score detects outliers as values that deviate significantly from the mean
(measured in standard deviations). Supports global, daytime/nighttime, and
rolling window approaches.
"""


def example_zscore_daytime_nighttime_separation():
    """Detect outliers using z-score with separate day/night thresholds.

    Calculates z-scores separately for daytime and nighttime records,
    useful when data characteristics vary by time of day.
    """
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()

    # Detect outliers with day/night separation
    detector = dv.zScoreDaytimeNighttime(
        series=series,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        thres_zscore=2.5,
        showplot=True,
        verbose=True
    )
    detector.calc(repeat=True)

    # Get flagged series (0=valid, 2=outlier)
    flag_series = detector.get_flag()


def example_zscore_global_threshold():
    """Detect outliers using z-score with global threshold.

    Single threshold applied to entire time series. Simpler and faster
    than day/night separation when time-of-day variation is not critical.
    """
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()

    # Detect outliers globally
    detector = dv.zScore(
        series=series,
        thres_zscore=2.0,
        showplot=True,
        verbose=True
    )
    detector.calc(repeat=False)

    # Get flagged series
    flag_series = detector.get_flag()


def example_zscore_rolling_window():
    """Detect outliers using rolling z-score (adaptive threshold).

    Calculates z-score from rolling mean and std dev, adapting threshold
    to local data characteristics. Useful for non-stationary time series.
    """
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()

    # Detect outliers with rolling threshold
    detector = dv.zScoreRolling(
        series=series,
        thres_zscore=2.5,
        winsize=48,  # ~24 hours for half-hourly data
        showplot=True,
        verbose=True
    )
    detector.calc(repeat=False)

    # Get flagged series
    flag_series = detector.get_flag()


if __name__ == '__main__':
    example_zscore_global_threshold()
    example_zscore_daytime_nighttime_separation()
    example_zscore_rolling_window()
