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
# A shared FormatStyle for the chrome
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Chrome (title, axis labels, fonts, colours, grid) is described once with a
# ``FormatStyle`` and reused across both histograms below, so they share an
# identical look. Histogram-specific toggles (peak highlight, z-scores, info /
# count boxes) stay as direct ``plot()`` arguments — they are not chrome.

style = dv.plotting.FormatStyle(
    xlabel='NEE flux',  # title left at None -> auto-generated from series name
    show_grid=True,  # gridlines are chrome -> set on the style, not plot()
)

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
    format_style=style,  # Shared chrome (x-label, auto title, ...)
    highlight_peak=True,  # Highlight bin with most counts
    show_zscores=True,  # Show z-score overlay
    show_zscore_values=True,  # Show z-score values
    show_info=True,  # Show method and peak info
    show_counts=True,  # Label bar heights
    show_title=True,  # Display title
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
    format_style=style,  # same shared chrome as the first histogram
    highlight_peak=True,
    show_zscores=False,  # z-scores not meaningful with unequal bins
    show_info=True,
    show_counts=True,
    show_title=True
)

print(f"Custom bins: {custom_edges}")
