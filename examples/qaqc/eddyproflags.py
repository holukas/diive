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


def example_angle_of_attack_test():
    """Extract and analyze angle of attack quality flag from EddyPro output.

    Demonstrates how to extract the angle of attack test flag from EddyPro FluxNet
    data and convert it to DIIVE format. The angle of attack test evaluates whether
    the wind vector relative to the sonic anemometer orientation is within acceptable
    limits. Large angles of attack indicate that the wind is approaching the
    anemometer at unfavorable angles, which can reduce measurement accuracy.

    This example:
    - Extracts the angle of attack flag from EddyPro FluxNet output
    - Converts EddyPro format (1=bad) to DIIVE format (2=bad)
    - Reports retained and discarded record counts
    """
    from diive.pkgs.qaqc.eddyproflags import flag_angle_of_attack_eddypro_test

    # Load example EddyPro FluxNet data
    df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

    flux = 'FC'

    print(f"Loaded EddyPro FluxNet data: {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    # Extract angle of attack flag
    flag = flag_angle_of_attack_eddypro_test(
        df=df,
        flux=flux,
        idstr='_L41'
    )

    # Calculate flag statistics
    n_retained = (flag == 0).sum()
    n_discarded = (flag == 2).sum()

    print("Angle of Attack Quality Test Results")
    print("-" * 60)
    print(f"Retained records:           {n_retained:6d}")
    print(f"Discarded records:          {n_discarded:6d}")
    print()


def example_vm97_quality_tests():
    """Extract and analyze VM97 (Vickers & Mahrt 1997) raw data quality test flags from EddyPro output.

    Demonstrates how to extract multiple raw data quality test flags from a single EddyPro VM97 code.
    VM97 tests are statistical quality assessments performed on the high-frequency raw eddy
    covariance data before flux calculation. The VM97 code is an 8-digit integer where each
    digit encodes results from a different quality test. This function extracts individual
    test results and converts them to DIIVE standard format.

    This example:
    - Extracts all 8 VM97 raw data quality tests from EddyPro FluxNet output
    - Converts EddyPro format to DIIVE format (hard flags: 2=bad, soft flags: 1=warning)
    - Reports pass/fail statistics for each individual test
    - Uses CO2 as the base variable (used to calculate FC flux)
    """
    from diive.pkgs.qaqc.eddyproflags import flags_vm97_eddypro_fluxnetfile_tests

    # Load example EddyPro FluxNet data
    df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

    flux = 'FC'
    fluxbasevar = 'CO2'

    print(f"Loaded EddyPro FluxNet data: {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    # Extract all 8 VM97 quality tests
    flags_df = flags_vm97_eddypro_fluxnetfile_tests(
        df=df,
        flux=flux,
        fluxbasevar=fluxbasevar,
        idstr='_L41',
        spikes=True,
        amplitude=True,
        dropout=True,
        abslim=True,
        skewkurt_hf=True,
        skewkurt_sf=True,
        discont_hf=True,
        discont_sf=True
    )

    print("VM97 Quality Test Results (All 8 Tests)")
    print("-" * 60)

    # Calculate and display statistics for each test
    test_info = [
        ('Spike detection', 'SPIKE_HF'),
        ('Amplitude resolution', 'AMPLITUDE_RESOLUTION_HF'),
        ('Dropout detection', 'DROPOUT_TEST'),
        ('Absolute limits', 'ABSOLUTE_LIMITS_HF'),
        ('Skewness/Kurtosis (hard)', 'SKEWKURT_HF'),
        ('Skewness/Kurtosis (soft)', 'SKEWKURT_SF'),
        ('Discontinuities (hard)', 'DISCONTINUITIES_HF'),
        ('Discontinuities (soft)', 'DISCONTINUITIES_SF'),
    ]

    for test_name, test_suffix in test_info:
        # Find the corresponding flag column
        flag_col = [col for col in flags_df.columns if test_suffix in col][0]
        flag_series = flags_df[flag_col]

        # Count results
        n_pass = (flag_series == 0).sum()
        n_fail = (flag_series == 2).sum()
        n_warn = (flag_series == 1).sum()

        print(f"{test_name:30s}  Pass: {n_pass:6d}  Fail: {n_fail:6d}  Warn: {n_warn:6d}")

    print()


def example_fluxbasevar_completeness_test():
    """Extract and analyze flux base variable completeness flag from EddyPro output.

    Demonstrates how to evaluate the completeness of the base variable used to calculate
    flux. For example, CO2 is the base variable used to calculate FC (carbon dioxide flux).
    High completeness indicates that the necessary measurements were available during
    the entire averaging period, which is essential for reliable flux calculation.

    This example:
    - Evaluates completeness of CO2 measurements (base variable for FC flux)
    - Converts completeness percentage to DIIVE quality flags
    - Reports completeness statistics (good/ok/bad quality)
    - Uses standard thresholds: 99% for good, 97% for ok
    """
    from diive.pkgs.qaqc.eddyproflags import flag_fluxbasevar_completeness_eddypro_test

    # Load example EddyPro FluxNet data
    df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

    flux = 'FC'
    fluxbasevar = 'CO2'

    print(f"Loaded EddyPro FluxNet data: {len(df)} records")
    print(f"Date range: {df.index[0]} to {df.index[-1]}\n")

    # Evaluate flux base variable completeness
    flag = flag_fluxbasevar_completeness_eddypro_test(
        df=df,
        flux=flux,
        fluxbasevar=fluxbasevar,
        thres_good=0.99,  # 99% completeness required for good flag
        thres_ok=0.97,    # 97% completeness required for ok flag
        idstr='_L41'
    )

    # Calculate flag statistics
    n_good = (flag == 0).sum()
    n_ok = (flag == 1).sum()
    n_bad = (flag == 2).sum()

    print(f"Base Variable Completeness Results ({fluxbasevar} for {flux} flux)")
    print("-" * 60)
    print(f"Good (>= 99% complete):     {n_good:6d} records")
    print(f"Ok (97-99% complete):       {n_ok:6d} records")
    print(f"Bad (< 97% complete):       {n_bad:6d} records")
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

    print("[EXAMPLE 3: Angle of Attack Quality Test]")
    print("-" * 80)
    example_angle_of_attack_test()

    print("[EXAMPLE 4: VM97 Quality Tests]")
    print("-" * 80)
    example_vm97_quality_tests()

    print("[EXAMPLE 5: Flux Base Variable Completeness]")
    print("-" * 80)
    example_fluxbasevar_completeness_test()
