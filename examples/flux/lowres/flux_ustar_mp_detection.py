"""
=====================================
Moving Point (MP) USTAR Detection
=====================================

Detect USTAR threshold using the Moving Point method (Papale et al., 2006).

Nighttime data is stratified by season, temperature class, and USTAR class.
The threshold is where NEE stops increasing with rising USTAR (forward mode).
A 3-year sliding window bootstrap returns per-year p16/p50/p84 thresholds
and a CUT (constant) threshold pooled across all years.
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet_lae()

print("=" * 70)
print("Moving Point (MP) USTAR Threshold Detection")
print("=" * 70)
print(f"\nData: {len(df)} records, {df.index.min().date()} to {df.index.max().date()}")

# %%
# Detect USTAR threshold
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Initialize detector with DataFrame. Column names are auto-detected.
# detect() returns seasonal thresholds; get_annual_thresholds() returns the
# conservative annual maximum across seasons.

detector = dv.flux.UstarMovingPointDetection(
    df=df,
    ta_classes_count=7,  # Temperature stratification classes (ONEFlux default)
    ustar_classes_count=20,  # USTAR stratification classes per temperature class
    verbose=1
)

detector.detect()

print("\n" + detector.summary())

annual = detector.get_annual_thresholds()
print(f"\nAnnual threshold (max across seasons): {annual['threshold']:.4f} m/s")

# %%
# Multi-year bootstrap uncertainty
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# UstarBootstrapThresholds runs N bootstrap iterations per calendar year,
# then extracts percentile thresholds from the resulting distribution.
# p50 is the recommended annual threshold; p16/p84 bound the uncertainty.
# get_cut_threshold() pools all years into a single CUT threshold.

boot = dv.flux.UstarBootstrapThresholds(
    df=df,
    detector_class=dv.flux.UstarMovingPointDetection,
    detector_kwargs=dict(ta_classes_count=7, ustar_classes_count=20),
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

print("\n[OK] Moving Point USTAR detection complete.")
