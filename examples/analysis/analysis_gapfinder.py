"""
========================
Gap Detection and Analysis
========================

Identify and characterize missing data periods in time series.

Scans a time series for consecutive NaN values and reports where gaps occur,
how long they are, and which gaps are longest. Useful for spotting data loss
events and assessing overall data quality.

Best for: Understanding missing data patterns and data availability
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

# Load the full 10-year dataset
data_df = dv.load_exampledata_parquet()

# Get the original NEE series (before any gap-filling)
series = data_df['NEE_CUT_REF_orig'].copy()

# %%
# Find all gaps
# ^^^^^^^^^^^^^

# Detect all consecutive missing periods
gf = dv.GapFinder(
    series=series,
    max_length=None,   # max gap size to include (records); None = no upper limit
    min_length=None,   # min gap size to include (records); None = no lower limit
    sort_results=True,  # sort by GAP_LENGTH descending
)

# print(gf) shows a formatted summary: series info, missing %, gap count,
# longest/median/mean gap, and active filters
print(gf)

# %%
# Top longest gaps
# ^^^^^^^^^^^^^^^^

# GAP_DURATION is auto-computed from the inferred time resolution
print("Top 5 longest gaps:")
print(gf.results[['GAP_START', 'GAP_END', 'GAP_LENGTH', 'GAP_DURATION']].head(5))

# %%
# Filter by gap size
# ^^^^^^^^^^^^^^^^^^

# Only gaps too long for simple interpolation (> 1 day = 48 records at 30min)
long_gaps = dv.GapFinder(
    series=series,
    max_length=None,   # no upper bound
    min_length=49,     # at least 49 records = just over 1 day at 30min
    sort_results=True,
)
print(long_gaps)

# Only short gaps suitable for linear interpolation (<= 4 records = 2 hours at 30min)
short_gaps = dv.GapFinder(
    series=series,
    max_length=4,      # at most 4 records = 2 hours at 30min
    min_length=None,   # no lower bound
    sort_results=True,
)
print(short_gaps)

# %%
# Visualization
# ^^^^^^^^^^^^^
# Two-panel figure:
# - top: daily data availability heatmap (day-of-year x year),
#   showing seasonal and annual patterns in data loss
# - bottom: gap length histogram (log scale) with duration reference
#   lines at 1h / 1 day / 1 week, inferred from time resolution

gf.showfig(
    title=f"Gap Analysis — {series.name}",  # figure suptitle
    saveplot=False,  # set True to save figure to disk
    path=None,       # output directory when saveplot=True
)
