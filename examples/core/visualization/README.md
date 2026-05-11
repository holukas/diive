# Data Visualization Examples

Examples demonstrating visualization and plotting functions for time series and flux data.

## Contents

- **plot_heatmap_datetime.py** — Heatmap plots with datetime indices for time series data
- **plot_scatter_xy.py** — Scatter plots with customizable styling and annotations
- **plot_timeseries.py** — Line plots for time series data visualization
- **plot_timeseries_and_cumulative.py** — Combined time series and cumulative sum plots
- **plot_dielcycle.py** — Diurnal (diel) cycle analysis and visualization
- **plot_hexbin.py** — Hexbin plots for 2D density visualization
- **plot_histogram.py** — Histogram plots with distribution analysis
- **plot_ridgeline.py** — Ridge line plots for comparing distributions across categories
- **plot_other_plots.py** — Additional specialized plot types

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
