# Data Visualization Examples

Examples demonstrating visualization and plotting functions for time series and flux data.

20 examples across 10+ plot types.

## Examples by Plot Type

### Heatmaps

- **plot_heatmap_datetime_basic.py** — Datetime heatmaps (vertical/horizontal layouts, value overlay on 6-hourly grids)
- **plot_heatmap_advanced.py** — Year-month heatmaps: aggregation method comparison (mean/max/std), rank mode, multi-variable side-by-side
- **plot_heatmap_xyz_basic.py** — Pre-aggregated 2D heatmaps from GridAggregator (mean and std, flux binned by temperature and VPD)

### Scatter Plots

- **plot_scatter_xy_basic.py** — 2D scatter plots showing variable relationships
- **plot_scatter_xy_colored.py** — 3D scatter with color coding, binning, and trend visualization

### Time Series

- **plot_timeseries.py** — Line plots with matplotlib
- **plot_timeseries_interactive.py** — Interactive Bokeh plots with zoom, pan, and export
- **plot_timeseries_rangetool.py** — Interactive Bokeh plot with a RangeTool overview for navigating long series

### Diurnal & Cumulative Patterns

- **plot_dielcycle.py** — Diurnal cycles grouped by month or season
- **plot_cumulative_basic.py** — Cumulative flux over time with scenario comparison
- **plot_cumulative_year.py** — Yearly cumulative sums with reference bands

### Distributions

- **plot_histogram_basic.py** — Histograms with z-score overlay, peak detection, and custom bin edges
- **plot_histogram_yearly.py** — Year-over-year histograms showing temporal patterns

### Density & Binning

- **plot_hexbin_basic.py** — 2D hexagonal binning with percentile normalization and sparse-bin filtering
- **plot_hexbin_advanced.py** — Hexbin plots with absolute values and overlays
- **plot_ridgeline_basic.py** — Ridge plots grouped by week
- **plot_ridgeline_advanced.py** — Ridge plots grouped by month with styling options

### Circular / Spiral

- **plot_treering_temperature.py** — Tree-ring spiral: annual data as concentric rings, color = value,
  month labels around circumference, optional month lines and year separators, colorbar auto-extension
- **plot_treering_line_temperature.py** — Tree-ring radial line plot: each year as a line trace around
  a full circle; radial displacement encodes value; single-color and per-year colormap variants with
  optional fill between baseline and line

### Other Plots

- **plot_other_plots.py** — Long-term anomalies and trend visualization

## Two-Phase Design

All plotting classes separate data preparation from presentation:

**Phase 1: `__init__()`** takes data and computation parameters.  
**Phase 2: `plot()`** handles styling, axes, titles, labels, and colors.

This lets you reuse the same data across multiple plots with different visual styles:

```python
scatter = dv.plot_scatter_xy(x=df['A'], y=df['B'], z=df['C'], nbins=10)
scatter.plot(ax=axes[0], title='View 1', cmap='viridis')
scatter.plot(ax=axes[1], title='View 2', cmap='plasma')
```

## Available Plot Classes

See `diive.core.plotting` for the complete API:

- `HeatmapDateTime` — Datetime-aware heatmaps
- `HeatmapXYZ` — Pre-aggregated 2D heatmaps
- `HeatmapYearMonth` — Year-month aggregation
- `ScatterXY` — Customizable 2D/3D scatter
- `Cumulative` — Cumulative sum plots
- `CumulativeYear` — Yearly cumulative analysis
- `TimeSeries` — Line plots
- `DielCycle` — Diurnal cycle plots
- `HistogramPlot` — Distribution histograms
- `HexbinPlot` — 2D hexagonal binning
- `RidgeLinePlot` — Ridge line plots
- `TreeRingPlot` — Circular spiral: annual rings with color-coded values

## Running Examples

```bash
# Single example
uv run python examples/visualization/plot_heatmap_datetime_basic.py
uv run python examples/visualization/plot_scatter_xy_colored.py

# All visualization examples
uv run python examples/run_all_examples.py
```

All 20 examples follow the two-phase design pattern.
