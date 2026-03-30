# Core Modules API Reference

Core modules provide foundational utilities shared across diive.

## Base Module

Base classes for flags and data handling.

```{autosummary}
:toctree: _autosummary

diive.core.base.FlagBase
```

## Data Frame Functions

Statistical utilities for DataFrames.

```{autosummary}
:toctree: _autosummary

diive.core.dfun.sstats
diive.core.dfun.fit_to_bins_linreg
diive.core.dfun.fit_to_bins_polyreg
```

## File I/O

Reading and writing data files.

```{autosummary}
:toctree: _autosummary

diive.core.io.DataFileReader
diive.core.io.MultiDataFileReader
diive.core.io.ReadFileType
diive.core.io.FileSplitter
```

## Time Series Handling

Timestamp validation, frequency detection, and time utilities.

```{autosummary}
:toctree: _autosummary

diive.core.times.TimestampSanitizer
diive.core.times.DetectFrequency
```

## Plotting & Visualization

Publication-quality plotting classes.

### Heat Maps

```{autosummary}
:toctree: _autosummary

diive.core.plotting.HeatmapBase
diive.core.plotting.HeatmapDateTime
diive.core.plotting.HeatmapXYZ
diive.core.plotting.HeatmapYearMonth
diive.core.plotting.HexbinPlot
```

### Time Series & Scatter

```{autosummary}
:toctree: _autosummary

diive.core.plotting.TimeSeries
diive.core.plotting.ScatterXY
diive.core.plotting.HistogramPlot
```

### Specialized Plots

```{autosummary}
:toctree: _autosummary

diive.core.plotting.DielCycle
diive.core.plotting.RidgeLinePlot
diive.core.plotting.Cumulative
```

## Machine Learning

ML-based utilities for gap-filling and analysis.

```{autosummary}
:toctree: _autosummary

diive.core.ml.MlRegressorGapFillingBase
```

## Utility Functions

General-purpose helper functions.

```{autosummary}
:toctree: _autosummary

diive.core.funcs
diive.core.utils
```
