"""
==========================================
Comparing USTAR Threshold Detection Modes
==========================================

Run ``UstarThresholdDetection`` in all four modes on the same data and compare
the thresholds and how many nighttime NEE records each threshold removes.

- ``'oneflux'``: ONEFlux, moving point (MP) and change point (CP) thresholds
  pooled, as used for FLUXNET2015 and ICOS.
- ``'oneflux_mp'``: ONEFlux with the moving point thresholds only
  (Papale et al. 2006).
- ``'oneflux_cp'``: ONEFlux with the change point thresholds only
  (Barr et al. 2013).
- ``'reddyproc'``: REddyProc 1.3.4 (Wutzler et al. 2018), a moving point
  variant with its own seasons and bootstrap.

Each mode reproduces its reference software, including how it computes the
percentiles. The ONEFlux modes report p16, p50 and p84 of 100 draws per year
and method. REddyProc reports p5, p50 and p95 of 200 values per year.

The example uses two years of CH-LAE data (2016 and 2017).
"""

# %%
# Load the data
# ^^^^^^^^^^^^^

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import diive as dv

df = dv.load_exampledata_parquet_lae()

NEE = 'NEE_L3.1_L3.2_QCF_IRGA72'  # umol m-2 s-1
TA = 'TA_T1_47_1_gfXG_IRGA72'  # deg C
USTAR = 'USTAR_IRGA72'  # m s-1
SW_IN = 'SW_IN_T1_47_1_gfXG_IRGA72'  # W m-2

print(f"Data: {len(df)} records, {df.index.min().date()} to {df.index.max().date()}")

# %%
# Run all modes
# ^^^^^^^^^^^^^
#
# Each mode uses its reference settings: number of bootstrap draws and
# percentiles.

MODES = ['oneflux', 'oneflux_mp', 'oneflux_cp', 'reddyproc']

detectors = {}
for mode in MODES:
    det = dv.flux.UstarThresholdDetection(
        df, nee_col=NEE, ta_col=TA, ustar_col=USTAR, swin_col=SW_IN,
        mode=mode, random_state=42)
    det.run()
    detectors[mode] = det
    print(det.summary())
    print()

# %%
# Thresholds per year (VUT) and for all years (CUT)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the ONEFlux modes the VUT of a year pools the thresholds of the year and
# its neighbours, so with two years VUT and CUT are equal. REddyProc estimates
# each year on its own. Its CUT pools the values of all years, which is a diive
# extension: REddyProc itself reports per-year values only.

vut_p50 = pd.DataFrame({mode: det.get_vut_thresholds()['p50']
                        for mode, det in detectors.items()})

# Lowest percentile, p50 and highest percentile of each mode's CUT
cut = {}
for mode, det in detectors.items():
    values = det.get_cut_threshold()
    labels = list(values)
    cut[mode] = {'range': f"{labels[0]}-{labels[-1]}", 'low': values[labels[0]],
                 'p50': values['p50'], 'high': values[labels[-1]]}
cut = pd.DataFrame(cut).T
cut[['low', 'p50', 'high']] = cut[['low', 'p50', 'high']].astype(float)

print("VUT p50 (m s-1):")
print(vut_p50.round(4))
print("\nCUT (m s-1):")
print(cut.round(4))

# %%
# Share of nighttime NEE records removed
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# u* filtering keeps a record only where u* is at or above the threshold, so a
# record without u* is removed too. The share below counts nighttime records
# (SW_IN < 10 W m-2) with a valid NEE that the CUT p50 removes.

night = (df[SW_IN] < 10) & df[NEE].notna()
n_night = int(night.sum())

removed = {}
for mode in MODES:
    threshold = cut.loc[mode, 'p50']
    below = ~(df[USTAR] >= threshold)
    removed[mode] = 100 * int((night & below).sum()) / n_night

table = pd.DataFrame({
    'CUT p50 (m s-1)': cut['p50'],
    'night NEE removed (%)': pd.Series(removed),
})
print(f"Nighttime records with NEE: {n_night}")
print(table.round(3))

# %%
# Plot the comparison
# ^^^^^^^^^^^^^^^^^^^
#
# Left: CUT p50 with the lowest and highest reported percentile as error bars
# (p16 and p84 for ONEFlux, p5 and p95 for REddyProc). Right: share of
# nighttime NEE records removed at the CUT p50.

colors = ['#2196F3', '#03A9F4', '#F44336', '#FFC107']


def label_color(color):
    """White text on dark bars, black on light ones."""
    r, g, b = to_rgb(color)
    return 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'


x = np.arange(len(MODES))
p50 = cut.loc[MODES, 'p50'].to_numpy()
low = cut.loc[MODES, 'low'].to_numpy()
high = cut.loc[MODES, 'high'].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.bar(x, p50, color=colors, yerr=[p50 - low, high - p50], capsize=5)
for i, v in enumerate(p50):
    ax.text(i, v / 2, f"{v:.3f}", ha='center', va='center_baseline',
            color=label_color(colors[i]))
ax.set_xticks(x, MODES)
ax.set_ylabel('u* threshold (m s-1)')
ax.set_title('CUT p50 and percentile range')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
shares = [removed[m] for m in MODES]
ax.bar(x, shares, color=colors)
for i, v in enumerate(shares):
    ax.text(i, v / 2, f"{v:.0f} %", ha='center', va='center_baseline',
            color=label_color(colors[i]))
ax.set_xticks(x, MODES)
ax.set_ylabel('nighttime NEE records removed (%)')
ax.set_title('Removed at CUT p50')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('CH-LAE 2016-2017: u* threshold detection modes')
fig.tight_layout()
plt.show()

# %%
# How the modes differ
# ^^^^^^^^^^^^^^^^^^^^
#
# All modes use nighttime records only. The moving point methods (ONEFlux MP
# and REddyProc) split each season into 7 temperature classes and 20 u*
# classes and look for the u* class where the nighttime flux stops increasing
# with u*. The year gets the highest season threshold.
#
# - ONEFlux MP: calendar-quarter seasons, night is SW_IN < 10 W m-2, the flux
#   must reach 100 % of the mean of the next 10 classes. A bootstrap draw
#   resamples all records of the year.
# - REddyProc: seasons start in March, June, September and December (December
#   counts with January and February of the next year), night is
#   SW_IN < 10 W m-2, the flux must reach 95 % of the mean of the next 10
#   classes. A year with fewer than 3000 nighttime records is estimated from
#   all its records pooled. A bootstrap draw resamples the records within each
#   season.
# - ONEFlux CP: night is SW_IN < 5 W m-2, 4 seasons with equal numbers of
#   records, 4 to 8 temperature strata per season. In each stratum a two-part
#   line (sloped, then flat) is fitted, and its break point is the threshold if
#   it is significant. A year needs at least 4000 valid nighttime records, so
#   CH-LAE 2016 gives no CP thresholds.
#
# Both ONEFlux methods give higher thresholds than REddyProc here, and the
# change point method gives the highest.
