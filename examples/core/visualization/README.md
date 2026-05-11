# Data Visualization Examples

Examples demonstrating visualization and plotting functions for time series and flux data.

## Contents

- **plot_heatmap_datetime_basic.py** — Datetime heatmaps with vertical/horizontal orientations
- **plot_heatmap_advanced.py** — Year-month aggregation and multi-variable comparison
- **plot_scatter_xy_basic.py** — Basic 2D scatter plots for variable relationships
- **plot_scatter_xy_colored.py** — 3D scatter plots with color coding and bin aggregation
- **plot_timeseries.py** — Line plots for time series data visualization
- **plot_timeseries_interactive.py** — Interactive Bokeh plots for data exploration
- **plot_dielcycle.py** — Diurnal (diel) cycle analysis and visualization
- **plot_histogram_basic.py** — Histogram with z-score overlay and peak highlighting
- **plot_histogram_yearly.py** — Yearly comparison histograms for temporal pattern analysis
- **plot_hexbin_basic.py** — Hexbin with percentile normalization for standardized comparison
- **plot_hexbin_advanced.py** — Advanced hexbin with absolute values and value overlays
- **plot_cumulative_basic.py** — Cumulative flux across all time with scenario comparison
- **plot_cumulative_year.py** — Yearly cumulative sums with reference band and highlighting
- **plot_ridgeline_basic.py** — Ridge line plots for weekly distribution analysis
- **plot_ridgeline_advanced.py** — Ridge line plots for monthly distribution analysis with multiple options
- **plot_other_plots.py** — Additional specialized plot types (anomalies)

## Related Documentation

See `diive.core.plotting` for available plot classes:
- `HeatmapDateTime` — Datetime-aware heatmaps
- `ScatterXY` — Customizable scatter plots
- `Cumulative` — Cumulative sum plots
- `TimeSeries` — Time series line plots

## Running Examples

```bash
uv run python examples/core/visualization/plot_heatmap_datetime.py
```

Or run all visualization examples:

```bash
uv run python examples/run_all_examples.py
```
