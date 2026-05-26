"""
=========================
Detect Timestamp Shifts
=========================

Detect clock/timestamp errors in meteorological time series by comparing
measured shortwave radiation against theoretical potential radiation.

Three detection methods are demonstrated:

- **FFT phase shift** — fast spectral comparison of the 24-hour diurnal cycle
- **Cross-correlation** — high-precision 1-minute lag search via scipy
- **Noon shift** — quick peak-time delta heuristic

All three methods share the same sign convention:
positive shift = measured peaks **earlier** than potential (leading clock),
negative shift = measured peaks **later** (lagging clock).
"""

# %%
# Load data and prepare the DataFrame
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The example dataset contains 30-minute ecosystem flux measurements.
# ``Rg_f`` is gap-filled global radiation in W/m².  We use a single year
# to keep runtime short.

import matplotlib.pyplot as plt

import diive as dv
from diive.preprocessing.qaqc.detect_timestamp_shifts import DetectTimestampShifts

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2022].copy()

# Keep only the measured radiation column; potential radiation is computed
# automatically from site coordinates inside the constructor.
df = df[['Rg_f']].copy()

print(f"Period  : {df.index.min().date()} to {df.index.max().date()}")
print(f"Records : {len(df)}")
print(f"Rg_f    : {df['Rg_f'].min():.1f} to {df['Rg_f'].max():.1f} W/m2")

# %%
# Construct DetectTimestampShifts
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``col_pot`` names the potential radiation column.  Because it is absent
# from ``df``, the constructor computes it automatically using ``potrad``
# with the supplied site coordinates.

dts = DetectTimestampShifts(
    df=df,
    col_meas='Rg_f',  # measured shortwave radiation column
    col_pot='SW_IN_POT',  # potential radiation column (auto-computed when absent)
    lat=47.286417,  # site latitude, decimal degrees
    lon=7.733750,  # site longitude, decimal degrees
    utc_offset=1,  # UTC offset in hours (CET = UTC+1)
)

# %%
# Alternative: supply a pre-computed potential radiation column
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If the potential radiation is already in the DataFrame, pass it directly.
# No lat/lon required.

from diive.features.variables.potentialradiation import potrad

df_with_pot = df.copy()
df_with_pot['SW_IN_POT'] = potrad(
    timestamp_index=df.index,
    lat=47.286417,
    lon=7.733750,
    utc_offset=1,
)

dts_precomputed = DetectTimestampShifts(
    df=df_with_pot,
    col_meas='Rg_f',
    col_pot='SW_IN_POT',  # already present; no lat/lon needed
)

# %%
# Method 1: FFT phase shift
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Projects each day's radiation onto the k=1 Fourier basis and compares
# phase angles.  Fast and reliable on clear days.

fft_results = dts.fft_phase_shift(
    min_clearness=0.6,  # skip days where measured/potential daily sum < 0.6
)

print(fft_results.dropna().describe().round(1))

# %%
# Plot FFT results
# ^^^^^^^^^^^^^^^^^
# Four panels: time series with rolling median, shift histogram, polar plot,
# monthly boxplot.

fig_fft, axes_fft = dts.plot_fft_results(
    amplitude_threshold=1000,  # ignore days where FFT amplitude < 1000 (weak signal)
    rolling_window=15,  # 15-day rolling median for trend line
    title='FFT Phase Shift — Example Dataset 2022',
)
plt.show()

# %%
# Method 2: Cross-correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Upsamples each day to 1-minute resolution, then uses scipy.signal.correlate
# to find the lag that maximises the Pearson correlation with potential
# radiation.  More precise than FFT; slower on very long datasets.

crosscorr_results = dts.crosscorr(
    max_shift_min=120,  # search window: +/- 2 hours
    upsample_freq='1min',  # upsample to 1-minute resolution
    min_clearness_index=0.5,  # skip heavy-overcast days
)

print(crosscorr_results.dropna().describe().round(2))

# %%
# Plot cross-correlation results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Two panels: scatter coloured by correlation strength, shift histogram.

fig_cc, axes_cc = dts.plot_crosscorr_results(
    min_corr=0.97,  # only plot days where max correlation > 0.97
    title='Cross-Correlation Shift — Example Dataset 2022',
)
plt.show()

# %%
# Method 3: Noon shift
# ^^^^^^^^^^^^^^^^^^^^^^
# Compares the timestamp of the daily radiation peak between measured and
# potential.  Very fast but sensitive to passing clouds near solar noon.

noon_results = dts.noon_shift(
    clearness_threshold=0.7,  # only include days where measured/potential sum > 0.7
)

print(noon_results.describe().round(1))

# %%
# Plot noon shift results
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Three panels: time series with rolling median, shift histogram, monthly
# boxplot.

fig_noon, axes_noon = dts.plot_noon_shift_results(
    rolling_window=15,  # 15-day rolling median for trend line
    title='Noon Shift (Peak Time) — Example Dataset 2022',
)
plt.show()

# %%
# Supporting plot: monthly diel cycles
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Overlays multi-year mean diel cycles per calendar month.  Systematic
# offsets between the measured curve and the potential radiation reference
# (dashed black) indicate a persistent timestamp shift.

fig_dc, axes_dc = dts.plot_monthly_dielcycles(
    years=[2022],  # list of years to overlay; defaults to all years in data
    colors=None,  # None uses Spectral_r colormap; pass an array to override
)
plt.show()

# %%
# Supporting plot: radiation fingerprint
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Heatmap with calendar days as rows and time-of-day as columns.  The
# parabolic upper boundary represents the potential radiation envelope;
# any horizontal offset between the measured peak and the envelope suggests
# a clock error.

fig_fp, ax_fp = dts.plot_radiation_fingerprint(
    year=2022,  # single year to visualise
    ax=None,  # None creates a new figure; pass an Axes to embed
    vmin=0,  # lower colour scale limit in W/m2
    vmax=None,  # None uses data maximum
)
plt.show()
