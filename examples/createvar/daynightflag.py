"""
Examples for daytime/nighttime flag calculation using daynightflag module.

Run this script to see daytime/nighttime flag results:
    python examples/createvar/daynightflag.py
"""
import matplotlib.pyplot as plt

import diive as dv


def example_daytime_nighttime_flag():
    """Calculate daytime and nighttime flags based on potential radiation.

    Demonstrates identifying daytime and nighttime periods using solar geometry
    calculations. Flags are determined by comparing potential shortwave radiation
    to a threshold (default 50 W/m²). Useful for quality control and analysis
    of ecosystem processes that vary between day and night.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Site coordinates: CH-DAV (Davos, Switzerland)
    SITE_LAT = 47.286417
    SITE_LON = 7.733750
    UTC_OFFSET = 1
    NIGHTTIME_THRESHOLD = 50  # W/m²

    print("Example: Calculate daytime and nighttime flags")
    print(f"Site: CH-DAV (Davos)")
    print(f"Coordinates: {SITE_LAT}°N, {SITE_LON}°E")
    print(f"Nighttime threshold: {NIGHTTIME_THRESHOLD} W/m²")

    # Calculate daytime/nighttime flags
    dnf = dv.DaytimeNighttimeFlag(
        timestamp_index=df.index,
        nighttime_threshold=NIGHTTIME_THRESHOLD,
        lat=SITE_LAT,
        lon=SITE_LON,
        utc_offset=UTC_OFFSET
    )

    # Get results
    results = dnf.get_results()
    daytime_flag = dnf.get_daytime_flag()
    nighttime_flag = dnf.get_nighttime_flag()
    swinpot = dnf.get_swinpot()

    print(f"\nResults:")
    print(f"Total records: {len(results)}")
    print(f"Daytime records: {(daytime_flag == 1).sum()}")
    print(f"Nighttime records: {(nighttime_flag == 1).sum()}")
    print(f"Potential radiation range: {swinpot.min():.1f} to {swinpot.max():.1f} W/m²\n")

    # Visualize results with heatmaps
    print("Generating heatmap visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 9),
                             gridspec_kw={'wspace': 0.1},
                             constrained_layout=True)

    dv.plot_heatmap_datetime(ax=axes[0], series=daytime_flag,
                             zlabel="flag (1=daytime)", cb_digits_after_comma=0).plot()
    dv.plot_heatmap_datetime(ax=axes[1], series=nighttime_flag,
                             zlabel="flag (1=nighttime)", cb_digits_after_comma=0).plot()
    dv.plot_heatmap_datetime(ax=axes[2], series=swinpot,
                             zlabel="$\mathrm{W\ m^{-2}}$", cb_digits_after_comma=0).plot()

    axes[0].set_title("Daytime flag")
    axes[1].set_title("Nighttime flag")
    axes[2].set_title("Potential radiation")

    plt.show()


if __name__ == '__main__':
    example_daytime_nighttime_flag()
