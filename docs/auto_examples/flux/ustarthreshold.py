"""
USTAR threshold detection and filtering for low-turbulence flux data.

Demonstrates USTAR (friction velocity) threshold determination using multiple
temperature classes (Papale et al., 2006) and applying constant USTAR thresholds
to create multiple flux scenarios for uncertainty analysis.
"""

import time
import matplotlib.pyplot as plt
import diive as dv


def example_ustar_detection_mpt():
    """Detect USTAR thresholds using Multiple Temperature Classes (Papale et al., 2006).

    The UstarDetectionMPT class:
    - Stratifies data by temperature class to account for seasonal differences
    - Uses USTAR subclass analysis within each temperature class
    - Bootstraps the threshold detection for statistical robustness
    - Produces seasonal threshold estimates with uncertainty bounds
    """
    t_start = time.perf_counter()

    from diive.configs.exampledata import load_exampledata_parquet_lae

    # Load example data
    df = load_exampledata_parquet_lae()

    # Restrict to single year for demonstration
    locs = (df.index.year >= 2017) & (df.index.year <= 2017)
    df = df.loc[locs].copy()

    # Select required columns
    NEE_COL = "NEE_L3.1_L3.2_QCF_IRGA72"
    TA_COL = "TA_T1_47_1_gfXG_IRGA72"
    USTAR_COL = "USTAR_IRGA72"
    SW_IN = "SW_IN_T1_47_1_gfXG_IRGA72"

    df = df[[NEE_COL, TA_COL, USTAR_COL, SW_IN]].copy()
    df = df.dropna()

    # Initialize USTAR detection with bootstrapping
    print("Initializing UstarDetectionMPT with 10 bootstrap runs...")
    t_init_start = time.perf_counter()

    ust = dv.UstarDetectionMPT(
        df=df,
        nee_col=NEE_COL,
        ta_col=TA_COL,
        ustar_col=USTAR_COL,
        ta_n_classes=6,           # 6 temperature classes for seasonal variation
        ustar_n_classes=20,       # 20 USTAR classes within each temperature class
        n_bootstraps=5,          # 10 bootstrap iterations for threshold uncertainty
        swin_pot_col=SW_IN,
        nighttime_threshold=20,   # Minimum daytime solar radiation (W m-2)
        utc_offset=1,             # UTC+1 timezone
        lat=47.478333,            # CH-LAE latitude
        lon=8.364389              # CH-LAE longitude
    )
    t_init_end = time.perf_counter()
    print(f"[TIMER] Initialization: {t_init_end - t_init_start:.2f}s\n")

    # Run threshold detection
    print("Running USTAR threshold detection...")
    t_run_start = time.perf_counter()
    ust.run()
    t_run_end = time.perf_counter()
    print(f"[TIMER] run() execution: {t_run_end - t_run_start:.2f}s")

    t_total = time.perf_counter() - t_start
    print(f"[TIMER] Total time: {t_total:.2f}s\n")

    # Report results: seasonal USTAR thresholds with confidence intervals
    print("USTAR Thresholds by Season (16th, 50th, 84th percentiles from bootstrapping):")
    print("-" * 70)
    if hasattr(ust, 'seasonal_thresholds'):
        for season, threshold_data in ust.seasonal_thresholds.items():
            print(f"{season}: {threshold_data}")


def example_ustar_threshold_constant_scenarios():
    """Create multiple USTAR threshold scenarios for uncertainty propagation.

    The UstarThresholdConstantScenarios class:
    - Applies different fixed USTAR thresholds to create multiple flux datasets
    - Represents USTAR threshold uncertainty through scenario analysis
    - Common approach: use 16th, 50th, 84th percentiles from detection
    - Enables joint uncertainty quantification combining random + USTAR terms
    """
    # Load example data
    from diive.configs.exampledata import load_exampledata_parquet_cha

    df = load_exampledata_parquet_cha()

    print("Creating USTAR threshold scenarios...")
    print("Thresholds: 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 m/s\n")

    # Initialize with NEE flux and USTAR
    ust = dv.UstarThresholdConstantScenarios(
        series=df['FC'],                # Flux (NEE/FC)
        swinpot=df['SW_IN_T1_2_1'],    # Solar radiation
        ustar=df['USTAR']               # Friction velocity
    )

    # Calculate thresholds: removes NEE where USTAR < threshold
    ust.calc(
        ustarthresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        showplot=True,
        verbose=True
    )

    print("[OK] Scenarios created with 7 different USTAR thresholds")


def example_flag_multiple_constant_ustar_thresholds():
    """Apply multiple USTAR thresholds to create QA/QC flags for different scenarios.

    The FlagMultipleConstantUstarThresholds class:
    - Applies multiple USTAR thresholds simultaneously
    - Creates separate quality flags for each threshold (e.g., CUT_16, CUT_50, CUT_84)
    - Used in standard flux processing chains (FLUXNET, Swiss FluxNet)
    - Enables comparison of gap-filling results across different USTAR scenarios
    """
    # Load example data
    from diive.configs.exampledata import load_exampledata_parquet_cha

    df = load_exampledata_parquet_cha()

    print("Applying multiple USTAR thresholds...")
    print("Thresholds: CUT_16=0.053, CUT_50=0.071, CUT_84=0.095 m/s\n")

    # Select required columns
    NEE_COL = "NEE_L3.1_L3.2_QCF0"
    USTAR_COL = "USTAR"
    SERIES = df[NEE_COL].copy()
    USTAR = df[USTAR_COL].copy()

    # Initialize with multiple thresholds (16th, 50th, 84th percentiles)
    ust = dv.FlagMultipleConstantUstarThresholds(
        series=SERIES,
        ustar=USTAR,
        thresholds=[0.0532449, 0.0709217, 0.0949867],           # USTAR values in m/s
        threshold_labels=['CUT_16', 'CUT_50', 'CUT_84'],        # Scenario names
        showplot=True,
        verbose=True
    )

    # Apply flags (creates columns for each threshold)
    ust.calc()

    # Access results
    if hasattr(ust, 'results'):
        results_df = ust.get_results()
        print(f"\n[OK] Results shape: {results_df.shape}")
        print(f"[OK] Columns generated: {[c for c in results_df.columns if 'CUT_' in c]}")

    print("\nThese flags are used in flux processing chains to:")
    print("  - Quality control based on turbulence conditions")
    print("  - Create multiple gap-filling scenarios (rf/xgb per scenario)")
    print("  - Support joint uncertainty quantification (random + USTAR)")


if __name__ == '__main__':
    example_flag_multiple_constant_ustar_thresholds()
    example_ustar_detection_mpt()
    example_ustar_threshold_constant_scenarios()
