"""
Examples for potential radiation calculations using potentialradiation module.

Run this script to see potential radiation examples:
    python examples/createvar/potentialradiation.py
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import diive as dv


def example_potrad_basic():
    """Calculate potential shortwave radiation using standard method.

    Demonstrates calculating theoretical solar radiation at Earth's surface
    for a specific location (Davos, Switzerland) using the Stull (1988) method.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Filter to 2018 for cleaner visualization
    f = df.index.year == 2018
    df = df[f].copy()

    # Calculate potential radiation (Stull method)
    sw_in_pot = dv.potrad(
        timestamp_index=df.index,
        lat=47.286417,  # Davos latitude
        lon=7.733750,  # Davos longitude
        utc_offset=1  # Central European Time
    )

    print("Potential radiation (Stull 1988 method):")
    print(f"Period: {sw_in_pot.index[0]} to {sw_in_pot.index[-1]}")
    print(f"Maximum: {sw_in_pot.max():.1f} W/m²")
    print(f"Mean: {sw_in_pot.mean():.1f} W/m²")


def example_potrad_eot_basic():
    """Calculate potential shortwave radiation using equation of time method.

    Demonstrates an alternative approach using the equation of time (EoT)
    for more accurate solar calculations at specific locations.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Filter to 2018
    f = df.index.year == 2018
    df = df[f].copy()

    # Calculate using equation of time method
    sw_in_pot_eot = dv.potrad_eot(
        timestamp_index=df.index,
        lat=47.286417,  # Davos latitude
        lon=7.733750,  # Davos longitude
        utc_offset=1,  # Central European Time
        use_atmospheric_transmission=False  # Top-of-Atmosphere
    )

    print("Potential radiation (Equation of Time method, TOA):")
    print(f"Period: {sw_in_pot_eot.index[0]} to {sw_in_pot_eot.index[-1]}")
    print(f"Maximum: {sw_in_pot_eot.max():.1f} W/m²")
    print(f"Mean: {sw_in_pot_eot.mean():.1f} W/m²")


def example_potrad_comparison():
    """Compare Stull method vs. Equation of Time method.

    Demonstrates the difference between two approaches for calculating
    potential radiation, including TOA vs. clear-sky surface approximation.
    """
    # Load example data
    df = dv.load_exampledata_parquet()

    # Filter to a single month for clarity
    f = (df.index.year == 2018) & (df.index.month == 7)
    df = df[f].copy()

    # Calculate using different methods
    # Method 1: Stull (1988)
    potrad_stull = dv.potrad(
        timestamp_index=df.index,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )

    # Method 2: Equation of Time - TOA
    potrad_eot_toa = dv.potrad_eot(
        timestamp_index=df.index,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        use_atmospheric_transmission=False
    )

    # Method 3: Equation of Time - Clear-sky surface
    potrad_eot_clearsky = dv.potrad_eot(
        timestamp_index=df.index,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1,
        use_atmospheric_transmission=True
    )

    print("Comparison of methods (July 2018):")
    print(f"Stull method max: {potrad_stull.max():.1f} W/m²")
    print(f"EoT (TOA) max: {potrad_eot_toa.max():.1f} W/m²")
    print(f"EoT (Clear-sky) max: {potrad_eot_clearsky.max():.1f} W/m²")
    print(f"\nMean difference (Stull - EoT TOA): {(potrad_stull - potrad_eot_toa).mean():.2f} W/m²")
    print(f"Mean difference (EoT TOA - Clear-sky): {(potrad_eot_toa - potrad_eot_clearsky).mean():.2f} W/m²")

    # Visualize comparison
    fig = plt.figure(facecolor='white', figsize=(16, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Plot 1: Stull method
    ax1.plot(potrad_stull.index, potrad_stull, color='#1565C0', linewidth=1.5)
    ax1.fill_between(potrad_stull.index, 0, potrad_stull, alpha=0.3, color='#1565C0')
    ax1.set_title('Stull (1988)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Radiation (W/m²)', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: EoT method (TOA)
    ax2.plot(potrad_eot_toa.index, potrad_eot_toa, color='#00BCD4', linewidth=1.5)
    ax2.fill_between(potrad_eot_toa.index, 0, potrad_eot_toa, alpha=0.3, color='#00BCD4')
    ax2.set_title('EoT (TOA)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Radiation (W/m²)', fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: EoT method (Clear-sky)
    ax3.plot(potrad_eot_clearsky.index, potrad_eot_clearsky, color='#D32F2F', linewidth=1.5)
    ax3.fill_between(potrad_eot_clearsky.index, 0, potrad_eot_clearsky, alpha=0.3, color='#D32F2F')
    ax3.set_title('EoT (Clear-sky)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Radiation (W/m²)', fontsize=9)
    ax3.set_xlabel('Date', fontsize=9)
    ax3.tick_params(labelsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison
    ax4.plot(potrad_stull.index, potrad_stull, color='#1565C0', linewidth=2, label='Stull', alpha=0.7)
    ax4.plot(potrad_eot_toa.index, potrad_eot_toa, color='#00BCD4', linewidth=2, label='EoT (TOA)', alpha=0.7)
    ax4.plot(potrad_eot_clearsky.index, potrad_eot_clearsky, color='#D32F2F', linewidth=2, label='EoT (Clear-sky)',
             alpha=0.7)
    ax4.set_title('Method Comparison', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Radiation (W/m²)', fontsize=9)
    ax4.set_xlabel('Date', fontsize=9)
    ax4.tick_params(labelsize=8)
    ax4.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, frameon=True, title='Methods', title_fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.show()


def example_potrad_heatmap():
    """Visualize potential radiation as heatmap.

    Shows daily and hourly patterns of potential radiation throughout
    the year using a heatmap visualization.
    """
    # Load full year of data
    df = dv.load_exampledata_parquet()
    f = df.index.year == 2018
    df = df[f].copy()

    # Calculate potential radiation
    sw_in_pot = dv.potrad(
        timestamp_index=df.index,
        lat=47.286417,
        lon=7.733750,
        utc_offset=1
    )

    # Create heatmap visualization
    dv.plot_heatmap_datetime(
        series=sw_in_pot,
        title='Potential Shortwave Radiation - Daily & Hourly Patterns',
        zlabel='W/m²',
        cb_digits_after_comma=0,
        ax_orientation='horizontal',
        figsize=(14, 6)
    ).show()


if __name__ == '__main__':
    print("Running potential radiation examples...\n")

    print("1. Basic potential radiation (Stull method)...")
    example_potrad_basic()

    print("\n2. Basic potential radiation (Equation of Time method)...")
    example_potrad_eot_basic()

    print("\n3. Comparison of methods...")
    example_potrad_comparison()

    print("\n4. Heatmap visualization...")
    example_potrad_heatmap()

    print("\nAll examples completed!")
