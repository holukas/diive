"""
Examples: Trimming Outliers (Trim Filter)

Trim filter removes values below a threshold, then removes an equal number
of values from the high end (trimmed mean approach).
"""


def example_trimlowoutliers_daytime_separation():
    """Detect and remove low outliers with day/night threshold separation.

    Demonstrates the TrimLow class on noisy temperature data with separate
    processing for daytime and nighttime records.
    """
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()

    # Extract 2018 temperature data
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()

    # Add impulse noise (spikes) for demonstration
    s_noise = dv.add_impulse_noise(
        series=s,
        factor_low=-10,
        factor_high=4,
        contamination=0.04,
        seed=42
    )
    s_noise.name = f"{s.name} + noise"

    # Detect and flag outliers
    trim = dv.TrimLow(
        series=s_noise,
        trim_daytime=False,  # Disable daytime filtering
        trim_nighttime=True,  # Enable nighttime filtering
        lower_limit=-75,      # Remove values below -75°C
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        showplot=True,
        verbose=True
    )
    trim.calc()

    # Get flagged series (0=valid, 2=outlier)
    flag_series = trim.overall_flag


def example_trimlowoutliers_daytime_only():
    """Detect and remove low outliers in daytime data only.

    Useful for filtering low outliers when nighttime data is stable
    (e.g., respiration measurements during day-only photosynthesis).
    """
    import diive as dv

    # Load example data
    df = dv.load_exampledata_parquet()

    # Extract 2018 temperature data
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()

    # Add impulse noise (spikes)
    s_noise = dv.add_impulse_noise(
        series=s,
        factor_low=-10,
        factor_high=4,
        contamination=0.04,
        seed=42
    )
    s_noise.name = f"{s.name} + noise"

    # Detect and flag outliers (daytime only)
    trim = dv.TrimLow(
        series=s_noise,
        trim_daytime=True,    # Enable daytime filtering
        trim_nighttime=False, # Disable nighttime filtering
        lower_limit=-75,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        showplot=True,
        verbose=True
    )
    trim.calc()

    # Get flagged series
    flag_series = trim.overall_flag


if __name__ == '__main__':
    example_trimlowoutliers_daytime_separation()
    example_trimlowoutliers_daytime_only()
