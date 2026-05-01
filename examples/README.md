# DIIVE Examples

Executable examples demonstrating how to use DIIVE for time series processing, gap-filling, and visualization.

## Structure

```
examples/
├── visualization/          # Plotting and visualization examples
│   ├── heatmap_datetime.py            # HeatmapDateTime and HeatmapYearMonth (5 examples)
│   ├── hexbin.py                      # HexbinPlot 2D hexagonal binning (3 examples)
│   ├── timeseries_and_cumulative.py   # Cumulative and CumulativeYear (3 examples)
│   ├── other_plots.py                 # LongtermAnomaliesYear (1 example)
│   ├── timeseries.py                  # TimeSeries interactive plots (1 example)
│   ├── dielcycle.py                   # DielCycle diurnal analysis (1 example)
│   ├── histogram.py                   # HistogramPlot distribution analysis (2 examples)
│   ├── ridgeline.py                   # RidgeLinePlot kernel density plots (2 examples)
│   ├── scatter_xy.py                  # ScatterXY scatter plots (3 examples)
│   └── heatmap_xyz.py                 # HeatmapXYZ 3D scatter heatmaps (TODO)
├── analyses/               # Time series analysis examples
│   ├── correlation.py                 # DailyCorrelation analysis with statistics and anomaly detection (1 example)
│   ├── decoupling.py                  # StratifiedAnalysis for hierarchical binning analysis (1 example)
│   ├── gapfinder.py                   # GapFinder for gap detection and analysis (1 example)
│   ├── gridaggregator.py              # GridAggregator for 2D grid-based aggregation (1 example)
│   └── histogram.py                   # Histogram for distribution analysis (1 example)
└── gap_filling/           # Gap-filling workflow examples (TODO)
    ├── quick_start.py             # Simple interpolation + quickfill
    ├── randomforest_ts.py         # RandomForestTS examples
    ├── xgboost_ts.py              # XGBoostTS examples
    ├── longterm_models.py         # Long-term multi-year models
    └── mds_filling.py             # MDS meteorological similarity
```

## Running Examples

**Run all examples at once:**
```bash
python examples/run_all_examples.py
```

Executes all 14 examples (9 visualization + 5 analysis) in parallel (4 concurrent workers) with execution time tracking.
- Shows individual timing for each example
- Detailed error messages if any fail
- ~2.7x faster than sequential execution

**Run individual examples:**
Each example file is standalone and executable:

```bash
python examples/visualization/heatmap_datetime.py
python examples/gap_filling/randomforest_ts.py
```

## Quick Start

**Visualization:**
```bash
python examples/visualization/heatmap_datetime.py
```

Displays heatmaps in vertical/horizontal orientations, year-month aggregations with ranks, and colormap previews.

**Gap-Filling (TODO - Phase 2):**
```bash
python examples/gap_filling/quick_start.py
```

Demonstrates simple interpolation and quick Random Forest gap-filling.

## Finding Help

- **API documentation:** See class docstrings (e.g., `help(HeatmapDateTime)`)
- **Examples:** Browse this folder for your use case
- **Architecture:** See `CLAUDE.md` for design decisions and workflows

## Contributing Examples

When adding a new example:
1. Create a function with a descriptive name: `example_<feature>_<variant>()`
2. Add docstring explaining what it demonstrates
3. Keep it runnable standalone: `python examples/<category>/<file>.py`
4. Use `dv.load_exampledata_parquet()` for consistent test data
5. Update this README with the new example

## Phases

- **Phase 1 (Complete):** Core visualization examples
  - HeatmapDateTime, HeatmapYearMonth (5 examples)
  - HexbinPlot (3 examples)
  - Cumulative, CumulativeYear (3 examples)
  - LongtermAnomaliesYear (1 example)
- **Phase 2 (Planned):** Additional visualization examples (dielcycle, heatmap_xyz)
- **Phase 3 (Planned):** Gap-filling workflow examples
- **Phase 4 (Future):** Feature engineering examples
