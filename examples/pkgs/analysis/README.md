# Time Series Analysis Examples

Examples demonstrating statistical analysis, decomposition, and pattern detection for time series data.

9 examples covering correlation, spectral analysis, gap detection, and decomposition.

## Examples by Method

### Correlation & Covariance

- **analysis_correlation.py** — Cross-correlation, autocorrelation, lag detection, anomaly detection
- **analysis_decoupling.py** — Photosynthetic decoupling across temperature gradients

### Decomposition & Trends

- **analysis_seasonaltrend.py** — STL decomposition separating trend and seasonality
- **analysis_harmonic.py** — Spectral analysis and Fourier decomposition for frequency content

### Distribution & Ranges

- **analysis_histogram_distribution.py** — Distribution analysis via histograms and percentiles
- **analysis_quantiles.py** — Percentile and quantile calculations
- **analysis_optimumrange.py** — Find optimal ranges for ecosystem responses

### Data Characterization

- **analysis_gapfinder.py** — Detect and visualize gaps, assess data completeness
- **analysis_gridaggregator.py** — 2D spatial gridding and aggregation

## Common Patterns

**Decompose seasonal trends:**

```python
from diive.pkgs.analysis import SeasonalTrendDecomposition

std = SeasonalTrendDecomposition(series=df['NEE'], period=365)
trend = std.trend
seasonal = std.seasonal
```

**Find lagged correlations (e.g., radiation vs. photosynthesis):**

```python
from diive.pkgs.analysis import Correlation

corr = Correlation(
    series1=df['PAR'],
    series2=df['GPP'],
    lags=range(-24, 25)
)
max_correlation = corr.results[corr.max_lag]
```

**Analyze diurnal cycles:**

```python
from diive.pkgs.analysis import DielCycle

diel = DielCycle(series=df['NEE'])
mean_by_hour = diel.mean_by_hour
std_by_hour = diel.std_by_hour
```

## Running Examples

```bash
# Decomposition & trends
uv run python examples/pkgs/analysis/analysis_seasonaltrend.py

# Correlations & relationships
uv run python examples/pkgs/analysis/analysis_correlation.py

# Data characterization
uv run python examples/pkgs/analysis/analysis_gapfinder.py
uv run python examples/pkgs/analysis/analysis_quantiles.py

# Distribution & range analysis
uv run python examples/pkgs/analysis/analysis_histogram_distribution.py
uv run python examples/pkgs/analysis/analysis_optimumrange.py

# Spatial & spectral analysis
uv run python examples/pkgs/analysis/analysis_gridaggregator.py
uv run python examples/pkgs/analysis/analysis_harmonic.py

# Specialized analysis
uv run python examples/pkgs/analysis/analysis_decoupling.py

# All examples
uv run python examples/run_all_examples.py
```

## Related Classes

See `diive.pkgs.analysis` for full API documentation:

- `Correlation` — Cross-correlation and lag detection
- `SeasonalTrendDecomposition` — STL decomposition
- `Quantiles` — Percentile-based analysis
- `GapFinder` — Gap detection and reporting
- `GridAggregator` — Spatial binning and aggregation
- `Harmonic` — Spectral and Fourier analysis
