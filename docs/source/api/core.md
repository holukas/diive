# Core Modules API Reference

Core modules provide foundational utilities shared across diive.

## Base Module

Base classes for flags and data handling.

- `diive.core.base.FlagBase` - Base class for flag objects

## Data Frame Functions

Statistical utilities for DataFrames.

- `diive.core.dfun.sstats` - Statistical functions for DataFrames
- `diive.core.dfun.fit_to_bins_linreg` - Linear regression fitting to bins
- `diive.core.dfun.fit_to_bins_polyreg` - Polynomial regression fitting to bins

## File I/O

Reading and writing data files.

- `diive.core.io.DataFileReader` - Read single data files
- `diive.core.io.MultiDataFileReader` - Read multiple data files
- `diive.core.io.ReadFileType` - Determine file type and read accordingly
- `diive.core.io.FileSplitter` - Split data files by time period

## Time Series Handling

Timestamp validation, frequency detection, and time utilities.

- `diive.core.times.TimestampSanitizer` - Validate and clean timestamp data
- `diive.core.times.DetectFrequency` - Detect sampling frequency

## Plotting & Visualization

Publication-quality plotting classes.

### Heat Maps

- `diive.core.plotting.HeatmapBase` - Base class for heatmap plots
- `diive.core.plotting.HeatmapDateTime` - Heatmap with datetime axes
- `diive.core.plotting.HeatmapXYZ` - 3D heatmap visualization
- `diive.core.plotting.HeatmapYearMonth` - Year-month heatmap
- `diive.core.plotting.HexbinPlot` - Hexagonal bin plots

### Time Series & Scatter

- `diive.core.plotting.TimeSeries` - Time series line plots
- `diive.core.plotting.ScatterXY` - Scatter plots
- `diive.core.plotting.HistogramPlot` - Histogram plots

### Specialized Plots

- `diive.core.plotting.DielCycle` - Diurnal cycle plots
- `diive.core.plotting.RidgeLinePlot` - Ridge line plots
- `diive.core.plotting.Cumulative` - Cumulative distribution plots

## Machine Learning

ML-based utilities for gap-filling and analysis.

- `diive.core.ml.MlRegressorGapFillingBase` - Base class for ML-based gap-filling

## Utility Functions

General-purpose helper functions.

- `diive.core.funcs` - General utility functions
- `diive.core.utils` - Utility helper functions
