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
# The chrome (x-label, grid) is identical for every year, so build one shared
# ``FormatStyle`` and reuse it across all the per-year plots. Only the title
# changes per year, passed as the legacy ``title=`` argument, which the plot
# folds onto the shared style.

style = dv.plotting.FormatStyle(
    xlabel='NEE flux',  # Same x-axis label for every year
    show_grid=True
)

years = df.index.year.unique()[:3]

for year in years:
    year_series = df[df.index.year == year]['NEE_CUT_REF_f'].copy()
    if len(year_series) < 10:
        continue

    hist = dv.plotting.HistogramPlot(
        series=year_series,
        method='n_bins',              # Binning method
        n_bins=15,                    # Number of bins
        ignore_fringe_bins=False      # Include edge bins
    )
    hist.plot(
        ax=None,                      # Create new figure
        # Shared chrome, with just the per-year title overridden via .merged()
        format_style=style.merged(title=f"Year {year}"),
        highlight_peak=True,          # Highlight bin with most counts
        show_zscores=True,            # Show z-score overlay
        show_zscore_values=True,      # Show z-score values
        show_info=True,               # Show method and peak info
        show_counts=True              # Label bar heights
    )

    print(f"Year {year}: {len(year_series)} records, mean={year_series.mean():.2f}, std={year_series.std():.2f}")

print(f"\nPlotted histograms for years: {', '.join(map(str, years))}")
