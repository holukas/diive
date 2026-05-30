"""
============================
Basic Histogram with Z-Scores
============================

Distribution analysis with z-score overlay, peak bin highlighting, and custom bin edges.

Best for: Analyzing data distributions, identifying outliers
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
# Create histogram with z-score overlay
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show distribution with z-score overlay and peak bin highlighting.

hist = dv.plotting.HistogramPlot(
    series=series,
    method='n_bins',  # Binning method
    n_bins=20,  # Number of bins
    ignore_fringe_bins=False  # Include edge bins
)
hist.plot(
    ax=None,  # Create new figure
    xlabel='NEE flux',  # X-axis label
    title=None,  # Auto-generated from series name
    highlight_peak=True,  # Highlight bin with most counts
    show_zscores=True,  # Show z-score overlay
    show_zscore_values=True,  # Show z-score values
    show_info=True,  # Show method and peak info
    show_counts=True,  # Label bar heights
    show_title=True,  # Display title
    show_grid=True  # Show gridlines
)

print("\nPlotted histogram with z-scores and peak highlighting")

# %%
# Custom bin edges
# ^^^^^^^^^^^^^^^^
#
# Pass a list of explicit bin edges instead of a count. Useful when you want
# unequal bin widths — e.g. finer resolution around zero for flux data.
# Follows Matplotlib convention: the last edge is the right boundary of the
# final bin (closed on the right).

custom_edges = [-30, -10, -5, -2, 0, 2, 5, 10, 30]  # unequal widths around zero

hist_custom = dv.plotting.HistogramPlot(
    series=series,
    method='n_bins',
    n_bins=custom_edges,  # list of edges instead of integer count
    ignore_fringe_bins=False
)
hist_custom.plot(
    ax=None,
    xlabel='NEE flux',
    highlight_peak=True,
    show_zscores=False,  # z-scores not meaningful with unequal bins
    show_info=True,
    show_counts=True,
    show_title=True,
    show_grid=True
)

print(f"Custom bins: {custom_edges}")
