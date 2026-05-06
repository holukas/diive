"""
Local Outlier Factor (LOF) outlier detection examples.

This module demonstrates the LocalOutlierFactor class for identifying outliers
based on local density deviations. Two modes are available:
- Global mode: Single LOF threshold for entire series
- Day/night mode: Separate thresholds for daytime/nighttime periods
"""


def example_lof_with_impulse_noise():
    """LOF day/night mode with synthetic impulse noise.

    Demonstrates LocalOutlierFactor with separate contamination
    rates for daytime and nighttime. Uses synthetic noise to create realistic
    spike patterns in temperature data.
    """
    import diive as dv

    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()
    s_noise = dv.add_impulse_noise(series=s,
                                   factor_low=-10,
                                   factor_high=3,
                                   contamination=0.04)
    s_noise.name = f"{s.name}+noise"

    lofa = dv.LocalOutlierFactor(
        series=s_noise,
        n_neighbors=20,
        contamination=0.05,
        separate_daytime_nighttime=True,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        showplot=True,
        verbose=True,
        n_jobs=-1
    )

    lofa.calc(repeat=False)


def example_lof_global():
    """LOF global mode with single contamination rate.

    Demonstrates LocalOutlierFactor class for detecting outliers
    across the entire time series without day/night separation.
    """
    import diive as dv

    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()
    s_noise = dv.add_impulse_noise(series=s,
                                   factor_low=-10,
                                   factor_high=3,
                                   contamination=0.04)
    s_noise.name = f"{s.name}+noise"

    lof_global = dv.LocalOutlierFactor(
        series=s_noise,
        n_neighbors=20,
        contamination=0.05,
        separate_daytime_nighttime=False,
        showplot=True,
        verbose=True,
        n_jobs=-1
    )

    lof_global.calc(repeat=False)


if __name__ == '__main__':
    example_lof_with_impulse_noise()
    # example_lof_global()
