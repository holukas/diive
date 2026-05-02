"""
Examples for high-quality flux analysis with outlier detection.

Demonstrates robust outlier detection for CO2 flux (NEE) using Hampel filter
(Median Absolute Deviation) with automatic day/night separation based on
solar geometry. The Hampel method is ideal for removing measurement spikes
while preserving ecosystem signal.

Run this script to see flux quality analysis examples:
    python examples/flux/hqflux.py

See Also
--------
diive.pkgs.flux.hqflux : High-quality flux analysis and outlier detection functions.
"""
import diive as dv
from diive.pkgs.flux.hqflux import analyze_highest_quality_flux


def example_hqflux_hampel_co2():
    """Analyze high-quality CO2 flux (NEE) with Hampel filter.

    Demonstrates robust outlier detection for CO2 net ecosystem exchange (NEE) flux
    using the Hampel filter. The filter uses Median Absolute Deviation (MAD) which
    is robust to extreme values, and applies double-differencing (Papale et al. 2006)
    to remove biological trends and isolate measurement spikes.

    Automatically separates daytime and nighttime using solar elevation angle,
    with separate strictness thresholds for each period (high turbulence during day,
    stable conditions at night).
    """

    print("=" * 80)
    print("Example: High-Quality CO2 Flux (FC) Analysis with Hampel Filter")
    print("=" * 80)

    # Load example data
    from diive.configs.exampledata import load_exampledata_parquet_cha
    df = load_exampledata_parquet_cha()
    keeprows = df['FC_SSITC_TEST'] == 0
    df = df[keeprows].copy()
    flux_fc = df['FC'].copy()

    print(f"\nData: {flux_fc.name}")
    print(f"Period: {flux_fc.index.min().date()} to {flux_fc.index.max().date()}")
    print(f"Total records: {len(flux_fc)}")
    print(f"Valid records: {flux_fc.count()}")
    print(f"Missing: {flux_fc.isnull().sum()} ({flux_fc.isnull().sum() / len(flux_fc) * 100:.1f}%)")

    # Analyze with Hampel filter
    results = analyze_highest_quality_flux(
        flux=flux_fc,
        lat=47.286417,  # Swiss FluxNet site (CH-DAV)
        lon=8.010325,
        utc_offset=1,  # CET
        window_length=48 * 13,  # 13 days at 30-min frequency
        n_sigma_dt=5.5,
        n_sigma_nt=5.5,
        use_differencing=True,  # Papale method: isolate spikes from trends
        showplot=True
    )

    print(f"\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Filtered data shape: {results.shape}")
    print(f"Columns: {list(results.columns)}")
    print(f"\nRolling median ± 3 SD shows expected CO2 flux range")
    print(f"Points outside this range may indicate sensor issues or extreme events")
    print(f"\nHampel Filter Method:")
    print(f"  - Median Absolute Deviation (MAD) for robust statistics")
    print(f"  - Double-differencing to remove biological trends")
    print(f"  - Automatic day/night thresholds based on solar elevation")


if __name__ == '__main__':
    print("=" * 80)
    print("High-Quality CO2 Flux (NEE) Analysis Example")
    print("=" * 80)
    print()

    example_hqflux_hampel_co2()

    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
