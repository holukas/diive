"""
Examples for VPD (Vapor Pressure Deficit) calculations using calc_vpd_from_ta_rh.

VPD can be calculated from air temperature (TA) and relative humidity (RH),
which are widely available measurements in ecosystem monitoring networks.

Run this script to see VPD calculation examples:
    python examples/createvar/vpd.py

See Also
--------
diive.calc_vpd_from_ta_rh : VPD calculation function documentation.
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import diive as dv


def example_vpd_basic():
    """Calculate VPD from air temperature and relative humidity.

    Demonstrates direct VPD calculation from gap-filled TA and observed RH
    using the Magnus formula. VPD is calculated in kPa and can be used as
    a driver variable for ecosystem flux analysis.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Variables
    ta_col = 'Tair_f'  # Gap-filled air temperature
    rh_col = 'RH'      # Relative humidity
    vpd_col = 'VPD_calculated'

    # Subset data
    subset_df = df[[ta_col, rh_col]].copy()

    # Calculate VPD from TA and RH
    subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

    # Print statistics
    print("VPD Calculation Statistics:")
    print(f"Data points: {len(subset_df)}")
    print(f"Missing values (RH): {subset_df[rh_col].isnull().sum()}")
    print(f"VPD range: {subset_df[vpd_col].min():.3f} to {subset_df[vpd_col].max():.3f} kPa")
    print(f"Mean VPD: {subset_df[vpd_col].mean():.3f} kPa")
    print(f"\nFirst 10 rows:")
    print(subset_df.head(10))


def example_vpd_with_gapfilling():
    """Calculate VPD from TA and RH with gap statistics.

    Demonstrates VPD calculation and analysis of gaps in the data.
    Shows how RH gaps affect VPD calculations and the extent of missing
    values that would need gap-filling.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Variables
    ta_col = 'Tair_f'  # Gap-filled air temperature
    rh_col = 'RH'      # Relative humidity
    vpd_col = 'VPD_hPa'

    # Subset data - use 1 year to keep example quick
    df_subset = df.loc[(df.index.year == 2018)].copy()
    subset_df = df_subset[[ta_col, rh_col]].copy()

    # Calculate VPD
    subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

    print("\n" + "=" * 80)
    print("VPD Calculation Analysis (Year 2018)")
    print("=" * 80)
    print(f"Data points: {len(subset_df)}")
    print(f"Original gaps in RH: {subset_df[rh_col].isnull().sum()} records")
    print(f"Gaps propagated to VPD: {subset_df[vpd_col].isnull().sum()} records")

    print(f"\nVPD Statistics:")
    print(subset_df[vpd_col].describe())

    print(f"\nTA and RH Statistics:")
    print(f"TA: {subset_df[ta_col].min():.1f}°C to {subset_df[ta_col].max():.1f}°C")
    print(f"RH: {subset_df[rh_col].min():.1f}% to {subset_df[rh_col].max():.1f}%")


def example_vpd_heatmap_comparison():
    """Visualize VPD patterns using heatmap (daily and hourly breakdown).

    Shows daily and hourly patterns of measured VPD using heatmap
    visualization. Reveals diurnal and seasonal patterns in vapor
    pressure deficit.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Variables
    ta_col = 'Tair_f'
    rh_col = 'RH'
    vpd_col = 'VPD_hPa'

    # Subset and calculate VPD
    subset_df = df[[ta_col, rh_col]].copy()
    subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

    print("\nVPD Heatmap Visualization:")
    print(f"Data points: {len(subset_df)}")
    print(f"VPD statistics:")
    print(subset_df[vpd_col].describe())

    # Create heatmap visualization
    try:
        dv.plot_heatmap_datetime(
            series=subset_df[vpd_col],
            title='VPD - Daily & Hourly Patterns',
            zlabel='VPD (kPa)',
            cb_digits_after_comma=2,
            ax_orientation='horizontal',
            figsize=(14, 6)
        ).show()
    except Exception as e:
        print(f"\nVisualization display info: {type(e).__name__}")


if __name__ == '__main__':
    print("=" * 80)
    print("VPD Calculation Examples: Vapor Pressure Deficit from TA and RH")
    print("=" * 80)
    print()

    print("Example 1: Basic VPD calculation")
    print("-" * 80)
    example_vpd_basic()

    print("\n" + "=" * 80)
    print("Example 2: VPD calculation with gap-filling")
    print("-" * 80)
    example_vpd_with_gapfilling()

    print("\n" + "=" * 80)
    print("Example 3: Heatmap visualization")
    print("-" * 80)
    example_vpd_heatmap_comparison()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
