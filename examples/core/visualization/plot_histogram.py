"""
====================
Histogram Plotting
====================

Distribution analysis with z-score overlay, peak highlighting, and statistics.

Best for: Analyzing data distributions, identifying outliers, comparing seasonal patterns
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
series = df['NEE_CUT_REF_f'].copy()

print(f"Loaded {len(series)} NEE flux records")
print(f"Statistics: mean={series.mean():.2f}, std={series.std():.2f}")

# %%
# Basic histogram with z-scores
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show distribution with z-score overlay and peak bin highlighting.

hist = dv.plot_histogram(
    s=series,
    method='n_bins',
    n_bins=20,
    xlabel='NEE flux',
    highlight_peak=True,
    show_zscores=True,
    show_zscore_values=True,
    show_info=True
)
hist.plot()

print("\nPlotted histogram with z-scores and peak highlighting")

# %%
# Yearly histograms
# ^^^^^^^^^^^^^^^^^
#
# Compare distributions across different years.

years = df.index.year.unique()[:3]

for year in years:
    year_series = df[df.index.year == year]['NEE_CUT_REF_f'].copy()
    if len(year_series) < 10:
        continue

    hist = dv.plot_histogram(
        s=year_series,
        method='n_bins',
        n_bins=15,
        xlabel='NEE flux',
        highlight_peak=True,
        show_zscores=True,
        title=f"Year {year}"
    )
    hist.plot()

print(f"Plotted histograms for years: {', '.join(map(str, years))}")
