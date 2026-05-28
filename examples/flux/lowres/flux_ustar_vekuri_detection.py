"""
======================================
USTAR Vekuri Threshold Detection
======================================

Detect USTAR threshold using the quantile-based Vekuri approach.

Uses quantile binning for both temperature and USTAR classes, which gives
equal sample sizes per bin regardless of the underlying distribution.
A 3-year sliding window bootstrap returns per-year p16/p50/p84 thresholds
and a CUT (constant) threshold pooled across all years.
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^^

import diive as dv

data = dv.load_exampledata_parquet_lae()

print(f"Data: {len(data)} records, {data.index.min().date()} to {data.index.max().date()}")

# %%
# Detect thresholds
# ^^^^^^^^^^^^^^^^^^
#
# detect() returns seasonal thresholds; get_annual_thresholds() returns the
# conservative annual maximum across seasons.

detector = dv.flux.UstarVekuriThresholdDetection(data, verbose=1)
thresholds = detector.detect()

print("\n" + detector.summary())

annual = detector.get_annual_thresholds()
print(f"\nAnnual threshold (max across seasons): {annual['threshold']:.4f} m/s")

# %%
# Multi-year bootstrap uncertainty
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# UstarBootstrapThresholds runs N bootstrap iterations per calendar year,
# then extracts percentile thresholds from the resulting distribution.
# p50 is the recommended annual threshold; p16/p84 bound the uncertainty.
# get_cut_threshold() pools all years into a single CUT threshold.

boot = dv.flux.UstarBootstrapThresholds(
    df=data,
    detector_class=dv.flux.UstarVekuriThresholdDetection,
    n_iter=100,
    percentiles=(16, 50, 84),
    n_jobs=-1,
    verbose=1,
)

annual_boot = boot.run()
cut = boot.get_cut_threshold()

print("\n" + boot.summary())

# %%
# Annual per-year thresholds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The bootstrap DataFrame contains one row per year.
# p50 is the recommended threshold to use for USTAR filtering.

print("\nPer-year bootstrap thresholds (m/s):")
print(annual_boot.round(4))

# %%
# CUT threshold
# ^^^^^^^^^^^^^^
#
# The CUT (constant) threshold pools all years' bootstrap samples
# and returns a single threshold stable across the full record.

print(f"\nCUT (constant) threshold (m/s):")
for pct, val in cut.items():
    marker = "  <-- recommended" if pct == 'p50' else ""
    print(f"  {pct}: {val:.4f}{marker}")

# %%
# Seasonal thresholds overview
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The raw detect() output shows thresholds for each season.

import pandas as pd

print("\nSeasonal thresholds from detect():")
print(thresholds.round(4))

print("\n[OK] Vekuri USTAR detection complete.")
