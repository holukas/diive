"""
Examples for EddyPro quality flags from raw data tests.

DIIVE uses a standard quality flag format where:
    - 0 = good quality (passes test)
    - 1 = soft warning (marginal, may indicate issues)
    - 2 = bad quality / hard fail (fails test)

EddyPro output files use different flag formats depending on the test type:
    - Some flags are integers (0=pass, 1=fail) that need conversion to DIIVE format
    - Some flags are multi-digit codes encoding multiple tests
    - Signal strength values are continuous and require threshold comparison

The functions in diive.pkgs.qaqc.eddyproflags extract information from EddyPro
output and either:
    - Calculate new quality flags by applying thresholds (e.g., signal strength)
    - Extract existing test flags and convert them to DIIVE standard format

This module demonstrates how to work with these functions and interpret their results.

Run this script to see EddyPro flag extraction:
    python examples/qaqc/eddyproflags.py
"""
from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN


def example_signal_strength_test():
    """Extract and analyze signal strength quality flags from EddyPro output.

    Demonstrates how to extract signal strength flags from EddyPro FluxNet data.
    Signal strength is a fundamental quality indicator for open-path IRGA and
    sonic anemometer measurements. High signal strength = good data quality.
    Low signal strength indicates instrumental issues (e.g., dust, condensation,
    optical drift) that degrade measurement reliability.

    This example:
    - Extracts the signal strength flag from EddyPro FluxNet output
    - Analyzes signal strength statistics for good vs. weak records
    - Reports data quality impact and retention rate
    - Provides quality assessment summary
    """
    from diive.pkgs.qaqc.eddyproflags import flag_signal_strength_eddypro_test

    # Load example EddyPro FluxNet data
    df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

    signal_col = 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN'
    var_col = 'FC'

    print(f"Loaded EddyPro FluxNet data: {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    # Apply signal strength test
    flag = flag_signal_strength_eddypro_test(
        df=df,
        signal_strength_col=signal_col,
        var_col=var_col,
        method='discard below',
        threshold=99,  # For demonstration purposes, assign bad flag for values below 99
        idstr='_L41'
    )

    # Calculate flag statistics
    n_retained = (flag == 0).sum()
    n_discarded = (flag == 2).sum()

    print("Signal Strength Quality Test Results")
    print("-" * 60)
    print(f"Retained records:           {n_retained:6d}")
    print(f"Discarded records:          {n_discarded:6d}")
    print()


def example_steadiness_horizontal_wind_test():
    """Extract and analyze wind steadiness quality flag from EddyPro output.

    Demonstrates how to extract the wind steadiness test flag from EddyPro FluxNet
    data and convert it to DIIVE format. The steadiness test evaluates whether
    horizontal wind components (u and v) show systematic changes throughout the
    measurement period, which indicates non-stationary conditions that degrade
    flux measurement quality.

    This example:
    - Extracts the wind steadiness flag from EddyPro FluxNet output
    - Converts EddyPro format (1=bad) to DIIVE format (2=bad)
    - Reports retained and discarded record counts
    """
    from diive.pkgs.qaqc.eddyproflags import flag_steadiness_horizontal_wind_eddypro_test

    # Load example EddyPro FluxNet data
    df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

    flux = 'FC'

    print(f"Loaded EddyPro FluxNet data: {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    # Extract steadiness flag
    flag = flag_steadiness_horizontal_wind_eddypro_test(
        df=df,
        flux=flux,
        idstr='_L41'
    )

    # Calculate flag statistics
    n_retained = (flag == 0).sum()
    n_discarded = (flag == 2).sum()

    print("Wind Steadiness Quality Test Results")
    print("-" * 60)
    print(f"Retained records:           {n_retained:6d}")
    print(f"Discarded records:          {n_discarded:6d}")
    print()


if __name__ == '__main__':
    print("=" * 80)
    print("EddyPro Quality Flag Examples")
    print("=" * 80)
    print()

    print("[EXAMPLE 1: Signal Strength Quality Test]")
    print("-" * 80)
    example_signal_strength_test()

    print("[EXAMPLE 2: Wind Steadiness Quality Test]")
    print("-" * 80)
    example_steadiness_horizontal_wind_test()

    print("\n" + "=" * 80)
    print("Note: These examples use real EddyPro FluxNet data (July 2021).")
    print("For your own EddyPro data, load your CSV and use the actual flag columns.")
    print("=" * 80)
