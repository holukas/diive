# Data Visualization Examples

Examples demonstrating visualization and plotting functions for time series and flux data.

## Contents

- **heatmap_datetime.py** — Heatmap plots with datetime indices for time series data
- **scatter_xy.py** — Scatter plots with customizable styling and annotations
- **timeseries.py** — Line plots for time series data visualization
- **timeseries_and_cumulative.py** — Combined time series and cumulative sum plots
- **dielcycle.py** — Diurnal (diel) cycle analysis and visualization
- **hexbin.py** — Hexbin plots for 2D density visualization
- **histogram.py** — Histogram plots with distribution analysis
- **ridgeline.py** — Ridge line plots for comparing distributions across categories
- **other_plots.py** — Additional specialized plot types

## Related Documentation

See `diive.core.plotting` for available plot classes:
- `HeatmapDateTime` — Datetime-aware heatmaps
- `ScatterXY` — Customizable scatter plots
- `Cumulative` — Cumulative sum plots
- `TimeSeries` — Time series line plots

## Running Examples

```bash
uv run python examples/core/visualization/heatmap_datetime.py
```

Or run all visualization examples:

```bash
uv run python examples/run_all_examples.py
```
