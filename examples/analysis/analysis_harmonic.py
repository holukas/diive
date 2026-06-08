"""
==================================
Harmonic (Fourier) Analysis
==================================

Decompose a time series into harmonic components with ``dv.analysis.harmonic_analysis``.

Harmonic analysis runs an FFT and reports the **amplitude** and **phase** of the
signal at the fundamental frequency (``1 / period``) and its integer multiples
(2x, 3x, ...). It answers "which cyclic components make up this signal, and how
strong is each?" — for eddy-covariance data the obvious cases are the **24-hour
photosynthesis cycle** (and its overtones) and the **annual cycle**.

This example covers:

1. Diel harmonics of CO2 flux — the dominant 24 h cycle and its overtones.
2. Annual harmonics of air temperature — the yearly cycle from daily data.
3. The effect of the analysis **window** on the extracted amplitude.
4. A **spectrogram** (``dv.analysis.spectrogram``) — how the cycles evolve over
   time, e.g. the daily rhythm strengthening in the growing season.

Best for: identifying and quantifying the dominant cyclic components of a flux
or meteo time series.
"""

# %%
# Imports and data
# ^^^^^^^^^^^^^^^^^

import matplotlib.pyplot as plt

import diive as dv

df = dv.load_exampledata_parquet()
print(f"Loaded {df.shape[0]} records, {df.shape[1]} variables.")

# %%
# Example 1: Diel harmonics of CO2 flux
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The gap-filled CO2 flux (NEE) has a strong daily cycle: uptake during the day,
# release at night. With half-hourly data one day is 48 records, so the
# fundamental ``period`` is 48. The first harmonic is the 24 h cycle; higher
# harmonics describe how the daily shape departs from a pure sine wave.

print("\n" + "=" * 70)
print("Example 1: Diel harmonics of NEE (period = 48 records = 24 h)")
print("=" * 70)

# One summer month, where the diel cycle is most pronounced.
nee = df['NEE_CUT_REF_f'].loc['2015-06-01':'2015-06-30']
records_per_day = 48  # half-hourly data

result = dv.analysis.harmonic_analysis(
    series=nee,
    period=records_per_day,
    n_harmonics=6,
    window='hamming',
    verbose=True,
)

# Each harmonic carries an amplitude, a phase, and the frequency it sits at.
# Convert that frequency (cycles per record) into a period in hours for reading.
print(f"\nFundamental frequency: {result['fundamental_frequency']:.5f} cycles/record")
print(f"\n{'Harmonic':<10}{'Amplitude':>12}{'Phase (rad)':>14}{'Period (h)':>14}")
print("-" * 50)
for h in result['harmonics']:
    period_h = (1.0 / h['actual_frequency']) / 2.0  # records -> hours (2 rec/h)
    print(f"H{h['harmonic_number']:<9}{h['amplitude']:>12.3f}"
          f"{h['phase']:>14.2f}{period_h:>14.1f}")

dominant = max(result['harmonics'], key=lambda h: h['amplitude'])
print(f"\nDominant harmonic: H{dominant['harmonic_number']} "
      f"(~{(1.0 / dominant['actual_frequency']) / 2.0:.1f} h) -> the daily cycle.")

# %%
# Plot the amplitude spectrum and the per-harmonic amplitudes.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# Full amplitude spectrum (skip the DC bin at index 0).
freqs = result['frequencies'][1:]
amps = result['amplitudes'][1:]
ax1.plot(freqs, amps, color='#90A4AE', lw=1.0, zorder=1)
# Mark the extracted harmonics.
hx = [h['actual_frequency'] for h in result['harmonics']]
hy = [h['amplitude'] for h in result['harmonics']]
ax1.scatter(hx, hy, color='#F44336', zorder=3, s=40)
for h in result['harmonics']:
    ax1.annotate(f"H{h['harmonic_number']}",
                 (h['actual_frequency'], h['amplitude']),
                 textcoords='offset points', xytext=(4, 4), fontsize=9, color='#455A64')
ax1.set_xlim(0, hx[-1] * 1.3)
ax1.set_xlabel('Frequency (cycles per record)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Amplitude spectrum of NEE (June 2015)')
ax1.grid(True, alpha=0.2)

# Per-harmonic amplitudes as bars.
nums = [h['harmonic_number'] for h in result['harmonics']]
ax2.bar(nums, hy, color='#2196F3', edgecolor='white', width=0.7)
ax2.set_xlabel('Harmonic number  (1 = 24 h, 2 = 12 h, 3 = 8 h, ...)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Harmonic amplitudes — the 24 h cycle dominates')
ax2.set_xticks(nums)
ax2.grid(True, axis='y', alpha=0.2)

plt.show()

# %%
# Example 2: Annual harmonics of air temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# At a coarser timescale the dominant cycle is annual. Aggregate the half-hourly
# air temperature to daily means and analyse it with a period of 365 days; the
# first harmonic is the yearly cycle.

print("\n" + "=" * 70)
print("Example 2: Annual harmonics of air temperature (period = 365 days)")
print("=" * 70)

daily_ta = dv.times.resample_to_daily_agg(df['Tair_f'], agg='mean').loc['2014':'2018'].dropna()
print(f"Daily air temperature: {len(daily_ta)} days "
      f"({daily_ta.index[0].date()} to {daily_ta.index[-1].date()})")

result_ta = dv.analysis.harmonic_analysis(
    series=daily_ta,
    period=365,
    n_harmonics=4,
    window='hamming',
)

print(f"\n{'Harmonic':<10}{'Amplitude':>12}{'Period (days)':>16}")
print("-" * 38)
for h in result_ta['harmonics']:
    period_d = 1.0 / h['actual_frequency']
    print(f"H{h['harmonic_number']:<9}{h['amplitude']:>12.2f}{period_d:>16.0f}")

dominant_ta = max(result_ta['harmonics'], key=lambda h: h['amplitude'])
print(f"\nDominant harmonic: H{dominant_ta['harmonic_number']} "
      f"(~{1.0 / dominant_ta['actual_frequency']:.0f} days) -> the annual cycle.")

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
nums_ta = [h['harmonic_number'] for h in result_ta['harmonics']]
amps_ta = [h['amplitude'] for h in result_ta['harmonics']]
ax.bar(nums_ta, amps_ta, color='#43A047', edgecolor='white', width=0.7)
ax.set_xlabel('Harmonic number  (1 = annual, 2 = semi-annual, ...)')
ax.set_ylabel('Amplitude (deg C)')
ax.set_title('Air temperature harmonics — annual cycle dominates')
ax.set_xticks(nums_ta)
ax.grid(True, axis='y', alpha=0.2)
plt.show()

# %%
# Example 3: Effect of the analysis window
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``window`` tapers the series before the FFT to reduce spectral leakage
# (energy smearing into neighbouring frequencies). Different windows trade
# amplitude accuracy against leakage suppression, so the recovered amplitude of
# the fundamental shifts slightly between them.

print("\n" + "=" * 70)
print("Example 3: Window function effect on the diel fundamental (H1)")
print("=" * 70)

print(f"\n{'Window':<12}{'H1 amplitude':>14}")
print("-" * 26)
windows = ['hamming', 'hann', 'blackman']
h1_amps = []
for win in windows:
    res = dv.analysis.harmonic_analysis(series=nee, period=records_per_day,
                                        n_harmonics=1, window=win)
    amp = res['harmonics'][0]['amplitude']
    h1_amps.append(amp)
    print(f"{win:<12}{amp:>14.3f}")

fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
ax.bar(windows, h1_amps, color='#FFC107', edgecolor='white', width=0.6)
ax.set_ylabel('H1 amplitude')
ax.set_title('Window choice slightly changes the recovered amplitude')
ax.grid(True, axis='y', alpha=0.2)
for i, a in enumerate(h1_amps):
    ax.text(i, a, f"{a:.3f}", ha='center', va='bottom', fontsize=9, color='#455A64')
plt.show()

# %%
# Example 4: Spectrogram — how the cycles evolve over time
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``harmonic_analysis`` gives a single, static spectrum for the whole series. A
# spectrogram instead computes a spectrum in a sliding window, so it shows *when*
# each cycle is strong. Over a full year of NEE the 1-cycle-per-day band (the
# diel photosynthesis rhythm) is strong in the growing season and weak in winter.

print("\n" + "=" * 70)
print("Example 4: Spectrogram of NEE over a full year")
print("=" * 70)

nee_year = df['NEE_CUT_REF_f'].loc['2015-01-01':'2015-12-31']
spec = dv.analysis.spectrogram(nee_year, nperseg=512, noverlap=256, window='hann')

# Convert the record-based axes to physical units (half-hourly -> 48 rec/day).
cycles_per_day = spec['frequencies'] * 48.0   # cycles/record -> cycles/day
day_of_year = spec['times'] / 48.0            # record offset -> days from Jan 1
print(f"Spectrogram: {spec['power'].shape[0]} frequency bins "
      f"x {spec['power'].shape[1]} time segments")

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
mesh = ax.pcolormesh(day_of_year, cycles_per_day, spec['power_db'],
                     shading='gouraud', cmap='viridis')
ax.axhline(1.0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)  # 24 h cycle
ax.set_ylim(0, 4)  # focus on the diel band and its first overtones
ax.set_xlabel('Day of year (2015)')
ax.set_ylabel('Frequency (cycles per day)')
ax.set_title('Spectrogram of NEE — the 1/day cycle peaks in the growing season')
cb = fig.colorbar(mesh, ax=ax)
cb.set_label('Power (dB)')
plt.show()

# %%
# Summary
# ^^^^^^^
# - ``harmonic_analysis`` extracts the amplitude and phase at a chosen ``period``
#   and its overtones.
# - For NEE the **24 h** harmonic dominates; for air temperature the **annual**
#   harmonic dominates.
# - The ``window`` and ``n_harmonics`` arguments control spectral leakage and how
#   many overtones are reported.
# - ``spectrogram`` shows how those cycles strengthen and weaken **over time**.

print("\n" + "=" * 70)
print("Harmonic analysis example completed.")
print("=" * 70 + "\n")
