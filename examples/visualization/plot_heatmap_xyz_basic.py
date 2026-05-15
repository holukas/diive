"""
============================
HeatmapXYZ (Basic)
============================

Heatmaps for pre-aggregated data: binned by two dimensions (x, y), colored by a third (z).
Shows relationships between two variables through aggregated flux values.

Best for: Visualizing binned flux relationships (e.g., NEE binned by temperature and VPD)
"""

import diive as dv

# %%
# Load and aggregate data
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a GridAggregator to bin data by two variables (temperature and VPD)
# and compute mean NEE for each bin. HeatmapXYZ requires pre-aggregated data.

df = dv.load_exampledata_parquet()

# Select a time period and remove missing values
df_clean = df.loc[df.index.year == 2022].copy()
df_clean = df_clean[['Tair_f', 'VPD_f', 'NEE_CUT_REF_f']].dropna()

print(f"Loaded {len(df_clean)} records for aggregation")

# Aggregate data into 2D bins (temperature x VPD)
q = dv.GridAggregator(
    x=df_clean['Tair_f'],  # Temperature on x-axis
    y=df_clean['VPD_f'],  # VPD on y-axis
    z=df_clean['NEE_CUT_REF_f'],  # NEE as color values
    binning_type='quantiles',  # Create quantile-based bins
    n_bins=10,  # 10 bins per dimension
    min_n_vals_per_bin=5,  # Min 5 values per bin
    aggfunc='mean'  # Average within bins
)

print(f"Created {q.df_agg_long.shape[0]} bins from gridaggregator")

# %%
# HeatmapXYZ from GridAggregator
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create heatmap directly from GridAggregator output using from_gridaggregator().
# Automatically handles bin column naming and pre-aggregated data format.

hm = dv.plot_heatmap_xyz.from_gridaggregator(
    q,  # GridAggregator instance
    x_col='Tair_f',  # Original x series name
    y_col='VPD_f',  # Original y series name
    z_col='NEE_CUT_REF_f'  # Original z series name
)

hm.plot(
    ax=None,  # Create new figure
    title='NEE binned by Temperature and VPD',
    figsize=(10, 8),  # Figure size
    cmap='RdYlGn',  # Colormap (red=low uptake, green=high)
    vmin=-5,  # Minimum color value (umol/m2/s)
    vmax=5,  # Maximum color value
    cb_digits_after_comma=1,  # Colorbar precision
    show_values=True,  # Overlay bin values
    show_values_n_dec_places=1,  # 1 decimal place for values
    show_grid=True  # Show grid lines
)

print("Plotted HeatmapXYZ with temperature x VPD bins")

# %%
# HeatmapXYZ with custom styling
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create heatmap with custom colormap and suppress value overlays for cleaner appearance.

hm2 = dv.plot_heatmap_xyz.from_gridaggregator(
    q,
    x_col='Tair_f',
    y_col='VPD_f',
    z_col='NEE_CUT_REF_f'
)

hm2.plot(
    ax=None,
    title='NEE Response Surface (Continuous)',
    figsize=(10, 8),
    cmap='viridis',  # Alternative colormap
    vmin=-5,
    vmax=5,
    show_values=False,  # Don't show values
    show_grid=False,  # Cleaner without grid
    show_colormap=True  # Show colorbar
)

print("Plotted HeatmapXYZ with continuous colormap")

# %%
# HeatmapXYZ - variability (std) instead of mean
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Run GridAggregator with aggfunc='std' to see within-bin variability.
# Complements the mean plot: high std where the flux response is uncertain or noisy.

q_std = dv.GridAggregator(
    x=df_clean['Tair_f'],
    y=df_clean['VPD_f'],
    z=df_clean['NEE_CUT_REF_f'],
    binning_type='quantiles',
    n_bins=10,
    min_n_vals_per_bin=5,
    aggfunc='std'  # standard deviation within each bin
)

hm_std = dv.plot_heatmap_xyz.from_gridaggregator(
    q_std,
    x_col='Tair_f',
    y_col='VPD_f',
    z_col='NEE_CUT_REF_f'
)
hm_std.plot(
    ax=None,
    title='NEE within-bin variability (std)',
    figsize=(10, 8),
    cmap='YlOrRd',  # low std = yellow, high std = red
    cb_digits_after_comma=1,
    show_values=True,
    show_values_n_dec_places=1,
    show_grid=True
)

print("Plotted HeatmapXYZ with std aggregation")
