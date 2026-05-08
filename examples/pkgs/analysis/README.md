# Time Series Analysis Examples

Examples demonstrating analysis functions for time series decomposition, correlation, and statistical analysis.

## Contents

- **correlation.py** — Cross-correlation and autocorrelation analysis
- **decoupling.py** — Flux decoupling and canopy-level analysis
- **gapfinder.py** — Detection and analysis of data gaps
- **gridaggregator.py** — Spatial gridding and aggregation of point measurements
- **histogram_distribution.py** — Distribution analysis and histograms
- **optimumrange.py** — Optimal range detection for data subsets
- **quantiles.py** — Quantile-based analysis and percentile calculations
- **seasonaltrend.py** — Seasonal decomposition and trend analysis
- **harmonic.py** — Harmonic analysis (Fourier decomposition)

## Related Documentation

See `diive.pkgs.analysis` for available analysis classes and functions.

## Running Examples

```bash
uv run python examples/pkgs/analysis/correlation.py
uv run python examples/pkgs/analysis/seasonaltrend.py
```

Or run all analysis examples:

```bash
uv run python examples/run_all_examples.py
```
