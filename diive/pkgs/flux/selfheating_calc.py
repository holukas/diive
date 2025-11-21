"""



    References:
        Burba et al. (2006). Correcting apparent off-season CO2 uptake due
            to surface heating of an open path gas analyzer: Progress report
            of an ongoing study. 13.
        Kittler et al. (2017). High-quality eddy-covariance CO2 budgets
            under cold climate conditions: Arctic Eddy-Covariance CO2 Budgets.
            Journal of Geophysical Research: Biogeosciences, 122(8), 2064–2084.
            https://doi.org/10.1002/2017JG003830

"""

import numpy as np
import pandas as pd


def corrected_flux(uncorrected_flux, fct_unsc_gf, sf_gf):
    fct = fct_unsc_gf.multiply(sf_gf)
    corrected_flux = uncorrected_flux + fct
    return corrected_flux, fct


def surface_temp_bur06(ta):
    """Calculate bulk instrument surface temperature (BUR06)

    :param ta: series, air temperature (°C)
    :return:
    series, surface temperature (BUR06) (°C)
    """
    ts_bur06 = 0.0025 * ta ** 2 + 0.9 * ta + 2.07
    print(f"Ts (BUR06), mean = {ts_bur06.mean():.2f}°C")
    return ts_bur06


def surface_temp_jar09(ta, daytime_filter, nighttime_filter):
    """Calculate bulk instrument surface temperature (JAR09)

    :param ta: series, air temperature (°C)
    :return:
    series, surface temperature (°C)
    """
    # Surface temperatures, separate for daytime and nighttime
    _ts_jar09_day = 0.93 * ta + 3.17
    _ts_jar09_night = 1.05 * ta + 1.52

    # Combine day and night temperatures
    ts_jar09_daynight = pd.Series(index=_ts_jar09_day.index)
    ts_jar09_daynight.loc[daytime_filter] = _ts_jar09_day  # Use daytime Ts in daytime data rows
    ts_jar09_daynight.loc[nighttime_filter] = _ts_jar09_night  # Use nighttime Ts in nighttime data rows

    # Stats
    print(f"Available daytime Ts (JAR09): {ts_jar09_daynight.loc[daytime_filter].count()} values")
    print(f"Available nighttime Ts (JAR09): {ts_jar09_daynight.loc[nighttime_filter].count()} values")
    print(f"Available Ts (JAR09): {ts_jar09_daynight.count()} total values")
    print(f"Ts (JAR09), mean = {ts_jar09_daynight.mean()}")
    return ts_jar09_daynight


def flux_correction_term_unscaled(ts, ta, qc_umol, ra, rho_v, rho_d):
    """Calculate unscaled flux correction term

    fct_unsc ... unscaled flux correction term

    Source:
        - Part of eq. (8) in Burba et al. (2006)
        - Similar to eq. (5) in Kittler et al. (2017)

    :param ts: series, bulk surface temperature (°C)
    :param ta: series, air temperature (°C)
    :param qc_umol: series, CO2 molar density (µmol m-3)
    :param ra: series, aerodynamic resistance (s m-1)
    :param rho_v: series, water vapor density (kg m-3)
    :param rho_d: series, dry air density (kg m-3)
    :return:
    pandas.Series
    """
    _a = (ts - ta) * qc_umol  # Uses BUR06 or JAR09 surface temperature
    _b = ra * (ta + 273.15)
    _c = 1 + 1.6077 * (rho_v / rho_d)
    fct_unsc = (_a / _b * _c)
    # flux_correction_term_unscaled = _a / _b * _c
    return fct_unsc


def gapfilling_lut(series):
    """Gap-fill time series using look-up table (LUT)

    The LUT contains hourly means of the series data for each month in the data

    :param series: series
    :return:
    """
    lutvals = pd.Series(index=series.index, data=np.nan)
    found_months = series.index.month.unique()
    found_hours = series.index.hour.unique()
    lut_df = series.groupby([series.index.month, series.index.hour]).mean().unstack()
    for found_month in found_months:
        for found_hour in found_hours:
            lutval = lut_df.loc[found_month, found_hour]  # Lookup value for this month and hour in LUT
            _filter = (series.index.month == found_month) & (
                    series.index.hour == found_hour)  # Indices of this month and hour in dataframe
            lutvals.loc[_filter] = lutval  # Fill in lookup value at index locations

    # Use the values from the LUT to gap-fill the calculated unscaled correction fluxes:
    series_gf = series.fillna(lutvals)  # Fill gaps

    return series_gf, lutvals


def remove_outliers(series, plot_title: str, n_sigmas: int = 5):
    """Remove outliers with Hampel filter, using running MAD (median absolute deviation)

    :param series: pandas.Series, data from which outliers are removed
    :param plot_title: Plot title
    :param n_sigmas:
    :return:
    pandas.Series with outliers removed
    """
    # n_sigmas = 4  # Number of sigmas for limits
    k = 1.4826  # Scale factor for Gaussian distribution
    window = 1440  # Rolling time window in number of records
    min_periods = 1  # Min number of records in window

    _series_rolling = series.rolling(window=window, min_periods=min_periods, center=True)
    _series_running_median = _series_rolling.median()
    _series_sub_winmed = series.sub(_series_running_median).abs()  # Data series minus median in time window
    _series_running_mad = _series_sub_winmed.rolling(window=window, min_periods=min_periods,
                                                     center=True).median().multiply(k)
    _diff_series_runmed = np.abs(series - _series_running_median)
    _diff_limit = _series_running_mad.multiply(n_sigmas)  # Limit

    _outliers_ix = _diff_series_runmed > _diff_limit
    series.loc[_outliers_ix] = np.nan  # Remove outliers from series (set to missing)
    return series

    # # Plot
    # figsize = (14, 5)
    # plt.figure()
    # series_orig.plot(figsize=figsize, title=f"{plot_title} BEFORE outlier removal");
    # plt.figure()
    # series.plot(title=f"{plot_title} AFTER outlier removal", figsize=figsize, label="series after outlier removal");
    # _diff_series_runmed.loc[~_outliers_ix].plot(
    #     label="absolute difference of: series - running median of series; for outlier detection")
    # _diff_limit.plot(label="limit from: series running MAD * number of sigmas")
    # _series_running_mad.plot(label="series running MAD (median absolute deviation)")
    # _series_running_median.plot(label="series running median")
    # plt.legend()
    # print(series.describe())

    # return series, series_orig
