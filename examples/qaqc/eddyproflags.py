"""
Examples for EddyPro quality flags from raw data tests.

Demonstrates how to extract and work with quality test flags from EddyPro output files,
including signal strength, angle of attack, steadiness, and VM97 statistical tests.

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


if __name__ == '__main__':
    print("=" * 80)
    print("EddyPro Quality Flag Examples")
    print("=" * 80)
    print()

    print("[EXAMPLE: Signal Strength Quality Test]")
    print("-" * 80)
    example_signal_strength_test()
