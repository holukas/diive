# Time Series Analysis Examples

Examples demonstrating analysis functions for time series decomposition, correlation, and statistical analysis.

## Contents

- **analysis_correlation.py** — Daily cross-correlation analysis between measured and potential radiation
- **analysis_decoupling.py** — Photosynthetic decoupling across temperature gradients
- **analysis_gapfinder.py** — Gap detection and characterization
- **analysis_gridaggregator.py** — Grid aggregation and multidimensional binning
- **analysis_histogram_distribution.py** — Distribution analysis via histograms
- **analysis_optimumrange.py** — Optimal range detection for ecosystem responses
- **analysis_quantiles.py** — Percentile and quantile analysis
- **analysis_seasonaltrend.py** — Seasonal-trend decomposition
- **analysis_harmonic.py** — Spectral analysis and Fourier decomposition

## Related Documentation

See `diive.pkgs.analysis` for available analysis classes and functions.

## Running Examples

```bash
uv run python examples/pkgs/analysis/analysis_correlation.py
uv run python examples/pkgs/analysis/analysis_seasonaltrend.py
```

Or run all analysis examples:

```bash
uv run python examples/run_all_examples.py
```
