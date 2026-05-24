"""
===========================
Yearly Comparison Histograms
===========================

Compare distributions across different years to identify temporal patterns.

Best for: Comparing seasonal patterns, analyzing year-to-year variations
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()

print(f"Loaded {len(df)} total records")

# %%
# Create histograms for multiple years
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Compare distributions across different years to identify temporal patterns.

years = df.index.year.unique()[:3]

for year in years:
    year_series = df[df.index.year == year]['NEE_CUT_REF_f'].copy()
    if len(year_series) < 10:
        continue

    hist = dv.plotting.HistogramPlot(
        s=year_series,
        method='n_bins',              # Binning method
        n_bins=15,                    # Number of bins
        ignore_fringe_bins=False      # Include edge bins
    )
    hist.plot(
        ax=None,                      # Create new figure
        xlabel='NEE flux',            # X-axis label
        title=f"Year {year}",         # Custom title
        highlight_peak=True,          # Highlight bin with most counts
        show_zscores=True,            # Show z-score overlay
        show_zscore_values=True,      # Show z-score values
        show_info=True,               # Show method and peak info
        show_counts=True,             # Label bar heights
        show_title=True,              # Display title
        show_grid=True                # Show gridlines
    )

    print(f"Year {year}: {len(year_series)} records, mean={year_series.mean():.2f}, std={year_series.std():.2f}")

print(f"\nPlotted histograms for years: {', '.join(map(str, years))}")
