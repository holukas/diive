"""
Examples for offset correction functions using offsetcorrection module.

Run this script to see offset correction results:
    python examples/corrections/offsetcorrection.py
"""
import numpy as np
import pandas as pd

import diive as dv


def example_remove_relativehumidity_offset():
    """Correct relative humidity values exceeding 100%.

    Demonstrates detecting and correcting RH measurements that exceed
    the physical maximum of 100%. Calculates daily mean of excess values
    and subtracts as offset, then caps maximum at 100%.
    """
    from diive.configs.exampledata import load_exampledata_parquet_meteo_rh_1MIN

    # Load actual RH data
    df = load_exampledata_parquet_meteo_rh_1MIN()
    series = df['RH_T1_47_1'].copy()
    # Scale to simulate high RH values exceeding 100% for demonstration
    series = series.multiply(1.4)

    print("Example 1: Remove relative humidity offset")
    print(f"Original RH range: {series.min():.2f}% to {series.max():.2f}%")
    print(f"Values exceeding 100%: {(series > 100).sum()}")

    # Correct RH offset
    series_corr = dv.remove_relativehumidity_offset(series=series, showplot=True)

    print(f"Corrected RH range: {series_corr.min():.2f}% to {series_corr.max():.2f}%")
    print(f"Values exceeding 100% after correction: {(series_corr > 100).sum()}\n")


def example_remove_radiation_zero_offset():
    """Correct radiation measurements with nighttime offset.

    Demonstrates detecting nighttime using solar geometry and correcting
    radiation data that has non-zero values during nighttime. Nighttime
    is set to zero after offset correction.
    """
    from diive.configs.exampledata import load_exampledata_parquet_meteo_swin_1MIN

    # Load actual solar radiation data
    df = load_exampledata_parquet_meteo_swin_1MIN()
    series = df['SW_IN_T1_47_1'].copy()

    # Site location (CH-LAE - Laegeren)
    SITE_LAT = 47.478333
    SITE_LON = 8.364389
    UTC_OFFSET = 1

    print("Example 2: Remove radiation zero offset")
    print(f"Original radiation range: {series.min():.2f} to {series.max():.2f} W/m2")

    # Correct radiation offset
    series_corr = dv.remove_radiation_zero_offset(
        series=series,
        lat=SITE_LAT,
        lon=SITE_LON,
        utc_offset=UTC_OFFSET,
        showplot=True
    )

    print(f"Corrected radiation range: {series_corr.min():.2f} to {series_corr.max():.2f} W/m2")
    nighttime_mean_corr = series_corr[series_corr.index.hour < 6].mean()


def example_measurement_offset_from_replicate():
    """Correct measurement offset using replicate data.

    Demonstrates detecting and correcting constant offset between a
    measurement and a reference replicate by finding the offset that
    minimizes the absolute difference between them.
    """
    # Create replicate (reference) data
    dates = pd.date_range('2024-01-01', periods=500, freq='30min')
    replicate = pd.Series(
        50 + 20 * np.sin(np.arange(500) * 2 * np.pi / 48),
        index=dates,
        name='replicate'
    )

    # Create measurement with constant offset of +4.2
    measurement = replicate.copy()
    measurement = measurement.add(4.2)
    measurement.name = 'measurement'

    print("Example 3: Detect and correct measurement offset")
    print(f"Replicate range: {replicate.min():.2f} to {replicate.max():.2f}")
    print(f"Measurement range: {measurement.min():.2f} to {measurement.max():.2f}")
    print(f"Expected offset: 4.2")

    # Detect and correct offset
    offset_corrector = dv.MeasurementOffsetFromReplicate(
        measurement=measurement,
        replicate=replicate,
        offset_start=-10,
        offset_end=10,
        offset_stepsize=0.1
    )

    measurement_corrected = offset_corrector.get_corrected_measurement()
    detected_offset = offset_corrector.get_offset()

    print(f"Detected offset: {detected_offset:.2f}")
    print(f"Corrected measurement range: {measurement_corrected.min():.2f} to {measurement_corrected.max():.2f}")
    print(f"Mean absolute difference (before): {(measurement - replicate).abs().mean():.4f}")
    print(f"Mean absolute difference (after): {(measurement_corrected - replicate).abs().mean():.4f}\n")


def example_wind_direction_offset():
    """Detect and correct wind direction offset using reference years.

    Demonstrates detecting and correcting wind direction measurement offset
    by comparing yearly wind direction histograms to a reference histogram
    built from known-correct reference years. Detects the offset that
    maximizes correlation with the reference distribution.
    """
    from diive.configs.exampledata import load_exampledata_winddir

    # Load wind direction data
    df = load_exampledata_winddir()
    winddir = df['wind_dir'].copy()

    # Filter to reference period and remove NaN values
    winddir = winddir.loc[winddir.index.year <= 2022]
    winddir = winddir.dropna()

    print("Example 4: Detect and correct wind direction offset")
    print(f"Wind direction range: {winddir.min():.1f} to {winddir.max():.1f} degrees")
    print(f"Years in data: {sorted(winddir.index.year.unique())}")

    # Detect offset using reference years (2021, 2022 known correct)
    offset_corrector = dv.WindDirOffset(
        winddir=winddir,
        hist_ref_years=[2021, 2022],
        offset_start=-50,
        offset_end=50,
        hist_n_bins=360
    )

    # Get results
    winddir_corrected = offset_corrector.get_corrected_wind_directions()
    yearly_offsets = offset_corrector.get_yearly_offsets()

    print("\nYearly wind direction offsets:")
    print(yearly_offsets.to_string(index=False))
    print(f"\nCorrected wind direction range: {winddir_corrected.min():.1f} to {winddir_corrected.max():.1f} degrees\n")


if __name__ == '__main__':
    example_remove_relativehumidity_offset()
    example_remove_radiation_zero_offset()
    example_measurement_offset_from_replicate()
    example_wind_direction_offset()
