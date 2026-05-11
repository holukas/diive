"""
============================
Basic Histogram with Z-Scores
============================

Distribution analysis with z-score overlay and peak bin highlighting.

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

hist = dv.plot_histogram(
    s=series,
    method='n_bins',              # Binning method
    n_bins=20,                    # Number of bins
    ignore_fringe_bins=False      # Include edge bins
)
hist.plot(
    ax=None,                      # Create new figure
    xlabel='NEE flux',            # X-axis label
    title=None,                   # Auto-generated from series name
    highlight_peak=True,          # Highlight bin with most counts
    show_zscores=True,            # Show z-score overlay
    show_zscore_values=True,      # Show z-score values
    show_info=True,               # Show method and peak info
    show_counts=True,             # Label bar heights
    show_title=True,              # Display title
    show_grid=True                # Show gridlines
)

print("\nPlotted histogram with z-scores and peak highlighting")
