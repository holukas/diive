"""
Spectral analysis examples for NEE_CUT_REF_f (CO2 flux).

Variable: NEE_CUT_REF_f (Net Ecosystem Exchange)
Units: umol CO2 m-2 s-1

Run: python examples/timeseries/spectral_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import diive as dv


def example1_spectrogram_daily_pattern():
    """Example 1: Spectrogram showing how CO2 flux spectrum changes over time.

    Visualize power spectral density evolution across a 10-day period,
    revealing the persistent 24-hour photosynthesis cycle.
    """
    print("\n" + "="*70)
    print("Example 1: Spectrogram - CO2 Flux Spectrum Over Time")
    print("="*70)

    df = dv.load_exampledata_parquet()
    nee = df['NEE_CUT_REF_f'].loc['2015-06-01':'2015-06-10'].copy()
    nee = nee.interpolate(method='linear', limit_direction='both').ffill().bfill()

    print(f"Variable: NEE_CUT_REF_f (CO2 flux)")
    print(f"Period: {len(nee)} records (10 days)")
    print(f"Operation: Compute spectrogram using short-time Fourier transform")

    # Compute spectrogram
    nee_valid = nee.dropna().values
    f, t, Sxx = signal.spectrogram(nee_valid, nperseg=128, noverlap=64, scaling='spectrum')

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    # Spectrogram
    im = ax1.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frequency (cycles/record)', fontsize=11, fontweight='bold')
    ax1.set_title('Spectrogram: NEE_CUT_REF_f Power Over Time\n(How CO2 spectrum evolves through 10 days)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(f) * 0.3])  # Focus on low frequencies
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Power (dB)', fontsize=10, fontweight='bold')

    # Add interpretation
    ax1.text(0.02, 0.98, 'Key frequencies (bright bands):\n'
             '• 0.02 (24h): Primary photosynthesis/respiration cycle (predictable, use in gap-filling)\n'
             '• 0.04 (12h): Semi-diurnal secondary oscillation (temperature, respiration)\n'
             '• 0.06 (8h): Tertiary atmospheric circulation / measurement artifacts\n'
             '• 0.1 (5h): High-frequency weather/turbulence noise (ignore in gap-filling)',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='black'))

    # Time series below
    ax2.plot(nee.index, nee.values, linewidth=0.8, color='steelblue', alpha=0.8)
    ax2.fill_between(nee.index, nee.values, alpha=0.2, color='steelblue')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('NEE (umol CO2 m-2 s-1)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_title('Original CO2 Flux Time Series',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax2.text(0.02, 0.98, 'Positive = CO2 uptake (day) | Negative = CO2 release (night)\nDaily rhythm is clear and repeats every 24h',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.show()


def example2_annual_spectrogram_phenology():
    """Example 2: Spectrogram over full year showing seasonal changes in CO2 patterns.

    Visualize how the spectrum changes throughout the year as photosynthetic
    capacity varies with phenological stage.
    """
    print("\n" + "="*70)
    print("Example 2: Annual Spectrogram - Seasonal CO2 Phenology")
    print("="*70)

    df = dv.load_exampledata_parquet()
    nee = df['NEE_CUT_REF_f'].loc['2015-01-01':'2015-12-31'].copy()
    nee = nee.interpolate(method='linear', limit_direction='both').ffill().bfill()

    print(f"Variable: NEE_CUT_REF_f (CO2 flux)")
    print(f"Period: {len(nee)} records (full year 2015)")
    print(f"Operation: Compute spectrogram using sliding time windows")

    # Compute spectrogram with larger window for annual view
    nee_valid = nee.dropna().values
    f, t, Sxx = signal.spectrogram(nee_valid, nperseg=512, noverlap=256, scaling='spectrum')

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

    # Spectrogram
    im = ax1.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax1.set_ylabel('Frequency (cycles/record)', fontsize=11, fontweight='bold')
    ax1.set_title('Annual Spectrogram: NEE_CUT_REF_f Power Throughout Year\n(Seasonal changes in CO2 photosynthesis capacity)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(f) * 0.3])
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Power (dB)', fontsize=10, fontweight='bold')

    # Add interpretation
    ax1.text(0.02, 0.98,
             'Key frequencies throughout year:\n'
             '• 0.02 (24h): Strongest during growing season (Apr-Sep), weak in winter\n'
             '• 0.04 (12h) & 0.06 (8h): Secondary harmonics, follow 24h intensity\n'
             '• 0.1 (5h): Constant weather noise throughout year (ignore)\n'
             '• Seasonal cycle: Strong photosynthesis = bright harmonics, dormancy = faint',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='black'))

    # Time series with seasons
    ax2.plot(nee.index, nee.values, linewidth=0.5, color='steelblue', alpha=0.8)
    ax2.fill_between(nee.index, nee.values, alpha=0.15, color='steelblue')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('NEE (umol CO2 m-2 s-1)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_title('Annual CO2 Flux: Phenological Stages',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Mark seasons
    ax2.axvspan(pd.Timestamp('2015-01-01'), pd.Timestamp('2015-03-31'), alpha=0.1, color='blue', label='Winter')
    ax2.axvspan(pd.Timestamp('2015-04-01'), pd.Timestamp('2015-06-30'), alpha=0.1, color='green', label='Spring-Summer')
    ax2.axvspan(pd.Timestamp('2015-07-01'), pd.Timestamp('2015-09-30'), alpha=0.1, color='yellow', label='Summer-Fall')
    ax2.axvspan(pd.Timestamp('2015-10-01'), pd.Timestamp('2015-12-31'), alpha=0.1, color='orange', label='Fall-Winter')
    ax2.legend(loc='upper right', fontsize=9)

    ax2.text(0.02, 0.98,
             'Seasonal phenology:\n'
             '• Winter (Jan-Mar): Low uptake, near-zero daily cycle\n'
             '• Growing (Apr-Sep): Peak uptake, strong photosynthesis\n'
             '• Senescence (Oct-Dec): Declining uptake, weaker cycles',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.show()


if __name__ == '__main__':
    example1_spectrogram_daily_pattern()
    example2_annual_spectrogram_phenology()

    print("\n" + "="*70)
    print("Spectral analysis examples completed!")
    print("="*70 + "\n")
