"""
===================================
USTAR Threshold Detection (ONEFlux)
===================================

Detect the friction velocity (u*) threshold from the data with
``UstarThresholdDetection``.

At night, NEE is ecosystem respiration. When turbulence is weak (low u*), the
measured flux underestimates it. The u* threshold is the u* above which the
nighttime flux no longer depends on u*. Records below the threshold are removed
before gap-filling.

The default ``mode='oneflux'`` follows ONEFlux, as used for FLUXNET2015 and ICOS.
For each calendar year it runs the moving point method (MP, Papale et al. 2006)
and the change point method (CP, Barr et al. 2013), each with 100 bootstrap
draws. The thresholds of both methods are pooled. The result comes as a VUT
(variable u* threshold, one per year) and a CUT (constant u* threshold, one for
all years).

The example uses two years of CH-LAE data (2016 and 2017).
"""

# %%
# Load the data
# ^^^^^^^^^^^^^
#
# The detection needs four columns: NEE after outlier removal and before u*
# filtering, air temperature, u* and incoming shortwave radiation. The
# radiation defines the night.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

df = dv.load_exampledata_parquet_lae()

NEE = 'NEE_L3.1_L3.2_QCF_IRGA72'  # umol m-2 s-1
TA = 'TA_T1_47_1_gfXG_IRGA72'  # deg C
USTAR = 'USTAR_IRGA72'  # m s-1
SW_IN = 'SW_IN_T1_47_1_gfXG_IRGA72'  # W m-2

print(f"Data: {len(df)} records, {df.index.min().date()} to {df.index.max().date()}")

# %%
# Run the detection
# ^^^^^^^^^^^^^^^^^
#
# ``run()`` returns the VUT table: one row per year, one column per percentile
# of the bootstrap thresholds. p50 is the usual threshold, p16 and p84 give the
# spread.

det = dv.flux.UstarThresholdDetection(
    df,
    nee_col=NEE,
    ta_col=TA,
    ustar_col=USTAR,
    swin_col=SW_IN,
    mode='oneflux',  # 'oneflux', 'oneflux_mp', 'oneflux_cp' or 'reddyproc'
    n_iter=None,  # bootstrap draws per year and method, None = 100 (ONEFlux)
    percentiles=None,  # None = (16, 50, 84) for the ONEFlux modes
    random_state=42,  # seed, so a run can be repeated
    n_jobs=1,  # parallel processes over years and methods
    verbose=1,
)
vut = det.run()

print(det.summary())

# %%
# VUT and CUT
# ^^^^^^^^^^^
#
# As in ONEFlux, the VUT of year Y pools the thresholds of the years Y-1, Y and
# Y+1. The CUT pools the thresholds of all years. With only two years both
# pools hold the same values here, so VUT and CUT are equal. ONEFlux reports a
# CUT only for records of 3 or more years.

print("VUT (m s-1):")
print(vut.round(4))

cut = det.get_cut_threshold()
print("\nCUT (m s-1):")
for p, value in cut.items():
    print(f"  {p}: {value:.4f}")

# %%
# Years without thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# A method can fail in a year. The change point method needs at least 4000
# valid nighttime records in the year. In 2016 CH-LAE has fewer, so CP gives
# no thresholds for 2016, as in ONEFlux. ``failures_`` holds the reason per
# year and method.

for year, failed in det.failures_.items():
    for method, reason in failed.items():
        print(f"{year} {method.upper()}: {reason}")

# %%
# Bootstrap thresholds
# ^^^^^^^^^^^^^^^^^^^^
#
# ``samples_`` holds the bootstrap thresholds per year and method. The
# histogram shows the pooled values behind the CUT. The change point
# thresholds of 2017 are higher than the moving point thresholds.

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, 1, 41)
colors = {'mp': '#2196F3', 'cp': '#F44336'}
names = {'mp': 'moving point (MP)', 'cp': 'change point (CP)'}
for method in ('mp', 'cp'):
    values = [s[method] for s in det.samples_.values() if method in s]
    if values:
        values = np.concatenate(values)
        ax.hist(values[np.isfinite(values)], bins=bins, color=colors[method],
                alpha=0.6, label=names[method])
for p, value in cut.items():
    ax.axvline(value, color='#455A64', lw=1.5 if p == 'p50' else 1,
               ls='-' if p == 'p50' else '--')
    ax.text(value, ax.get_ylim()[1] * 0.95, f" {p}", color='#455A64', va='top')
ax.set_xlabel('u* threshold (m s-1)')
ax.set_ylabel('bootstrap draws')
ax.set_title('CH-LAE 2016-2017: bootstrap thresholds pooled for the CUT')
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()

# %%
# Use the threshold in the flux post-processing chain
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the chain, Level 3.3 runs the same detection and filters the flux with the
# result, one scenario per percentile (``CUT_16``, ``CUT_50``, ``CUT_84``). Not
# run here::
#
#     from diive.flux.postprocessingchain import run_level33_ustar_detection
#     data = run_level33_ustar_detection(
#         data, ta_col=TA, swin_col=SW_IN,
#         detection_mode='oneflux',
#         mode='cut',  # or 'vut' for one threshold per year
#         nighttime_only=False,  # True filters only nighttime records, as REddyProc
#     )
#
# ``run_chain`` does the same with ``FluxConfig(ustar_detection_mode='bootstrap',
# ustar_bootstrap_detection_mode='oneflux', ...)``.
