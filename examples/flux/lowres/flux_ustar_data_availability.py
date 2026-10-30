"""
=====================================
USTAR Threshold and Data Availability
=====================================

Show how much NEE data a u* threshold removes, by day and by night, with
``UstarThresholdConstantScenarios``.

u* filtering keeps a record only where u* is at or above the threshold. A record
without u* is removed too. A higher threshold removes more records, mostly at
night, when turbulence is weak. The removed records become gaps that the
gap-filling has to fill, so the choice of threshold decides how much of the
annual sum rests on measured data.

The example detects the threshold for one year of CH-LAE data (2017), then
applies a range of constant thresholds and counts the records that are left, the
share kept by day and by night, and the gaps that the 16th, 50th and 84th
percentile of the detected threshold leave.
"""

# %%
# Load the data
# ^^^^^^^^^^^^^
#
# NEE after outlier removal and before u* filtering, u*, air temperature and
# incoming shortwave radiation, for 2017. Day and night are told apart by the
# potential radiation of the site.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diive as dv

df = dv.load_exampledata_parquet_lae()
df = df.loc[df.index.year == 2017].copy()

NEE = 'NEE_L3.1_L3.2_QCF_IRGA72'  # umol m-2 s-1
TA = 'TA_T1_47_1_gfXG_IRGA72'  # deg C
USTAR = 'USTAR_IRGA72'  # m s-1
SW_IN = 'SW_IN_T1_47_1_gfXG_IRGA72'  # W m-2

sw_in_pot = dv.variables.potrad(
    timestamp_index=df.index,
    lat=47.478333,  # CH-LAE
    lon=8.364389,  # CH-LAE
    utc_offset=1,  # local standard time, CET (UTC+1)
)

print(f"Data: {len(df)} records, {df[NEE].notna().sum()} with NEE")

# %%
# Detect the threshold
# ^^^^^^^^^^^^^^^^^^^^
#
# The CUT (constant u* threshold) as ONEFlux detects it. Its 16th, 50th and
# 84th percentile are the usual USTAR scenarios of the flux post-processing
# chain (``CUT_16``, ``CUT_50``, ``CUT_84``). See the USTAR detection example
# for the method.

det = dv.flux.UstarThresholdDetection(
    df, nee_col=NEE, ta_col=TA, ustar_col=USTAR, swin_col=SW_IN,
    mode='oneflux', random_state=42,
)
det.run()
cut = det.get_cut_threshold()
for p, value in cut.items():
    print(f"CUT {p}: {value:.3f} m s-1")

# %%
# Records left after a few thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``UstarThresholdConstantScenarios`` builds one NEE series per threshold, with
# the records below that threshold removed. ``showplot=True`` draws the number
# of records left for all records, daytime and nighttime, and the share of the
# unfiltered series in percent.

scenarios = dv.flux.UstarThresholdConstantScenarios(
    series=df[NEE],
    ustar=df[USTAR],
    swinpot=sw_in_pot,
)
scenarios.calc(ustarthresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], showplot=True)

# %%
# Share kept, by day and by night
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same over a fine range of thresholds. ``scenariosdf`` holds the series of
# every threshold, and ``daytime`` / ``nighttime`` mark the records by
# potential radiation. The nighttime share drops much faster than the daytime
# share: at night u* is lower, and nighttime NEE is what the threshold is for.

thresholds = np.round(np.arange(0.0, 0.905, 0.025), 3)
scenarios.calc(ustarthresholds=list(thresholds))
sdf = scenarios.scenariosdf

measured = sdf[NEE].notna()
kept = pd.DataFrame(index=thresholds, columns=['all', 'day', 'night'], dtype=float)
for u in thresholds:
    col = sdf[f"{NEE}_CUT{u}"].notna()
    kept.loc[u, 'all'] = 100 * col.sum() / measured.sum()
    kept.loc[u, 'day'] = 100 * col[scenarios.daytime].sum() / measured[scenarios.daytime].sum()
    kept.loc[u, 'night'] = 100 * col[scenarios.nighttime].sum() / measured[scenarios.nighttime].sum()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(kept.index, kept['day'], color='#FFC107', lw=2.2, label='daytime')
ax.plot(kept.index, kept['night'], color='#2196F3', lw=2.2, label='nighttime')
ax.plot(kept.index, kept['all'], color='#455A64', lw=1.5, ls='--', label='all records')
for p, value in cut.items():
    ax.axvline(value, color='#F44336', lw=1.5 if p == 'p50' else 1,
               ls='-' if p == 'p50' else ':')
    ax.text(value, 3, f" CUT {p}", color='#F44336', rotation=90, va='bottom', ha='right')
ax.set_xlabel('u* threshold (m s-1)')
ax.set_ylabel('measured NEE records kept (%)')
ax.set_ylim(0, 102)
ax.set_title('CH-LAE 2017: NEE records kept after u* filtering')
ax.legend(loc='lower left')
fig.tight_layout()
plt.show()

print(kept.loc[[0.1, 0.3, 0.5, 0.7]].round(1))

# %%
# Gaps left by the USTAR scenarios
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The removed records join the existing gaps. ``GapStats`` counts the gaps of
# each scenario and how long they are. A gap of a day or more is long here
# (48 half-hours). Long gaps are harder to fill well, and the gap-filling
# methods differ most on them.

scenarios.calc(ustarthresholds=[round(v, 3) for v in cut.values()])
rows = {}
for p, value in cut.items():
    series = scenarios.scenariosdf[f"{NEE}_CUT{round(value, 3)}"]
    gs = dv.analysis.GapStats(series=series, long_gap_records=48)
    rows[f"CUT_{p[1:]} ({value:.3f})"] = {
        'missing (%)': round(100 * series.isna().mean(), 1),
        'gaps': len(gs.results),
        'long gaps (>= 1 day)': gs.summary['n_long_gaps'],
        'longest gap (days)': round(gs.results['GAP_LENGTH'].max() / 48, 1),
    }
print(pd.DataFrame.from_dict(rows, orient='index'))

# %%
# In the flux post-processing chain
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# After Level 3.3 the chain reports the same per USTAR scenario, also split
# into the quality classes the gap-filling uses. Not run here::
#
#     from diive.flux.postprocessingchain import (
#         plot_gapfilling_input, report_gapfilling_input, report_gaps)
#     report_gapfilling_input(data)  # records used, removed by the tests and by u*
#     report_gaps(data)  # gaps per scenario
#     plot_gapfilling_input(data)  # heatmaps of the records used per scenario
