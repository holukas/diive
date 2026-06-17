# Time Series Analysis Examples

Examples demonstrating statistical analysis, decomposition, and pattern detection for time series data.

12 examples covering correlation, spectral analysis, gap detection, grid aggregation, and decomposition.

## Examples by Method

### Correlation & Covariance

- **analysis_daily_correlation.py** — Daily correlation coefficients for quality checks, relationship analysis, and statistical methods
- **analysis_driveranalysis.py** — _(experimental; `dv.analysis.experimental`)_ Evidence-triangulation driver attribution organized by epistemic level (association → temporal prediction → causation), with a convergence/divergence summary across SHAP, ALE, lagged/scale-resolved/stratified importance, and Granger
- **analysis_granger.py** — Granger causality testing to detect predictive relationships between time series
- **analysis_decoupling.py** — Stratified binning to reveal how ecosystem responses change across temperature ranges

### Decomposition & Trends

- **analysis_seasonaltrend.py** — STL decomposition separating trend and seasonality
- **analysis_harmonic.py** — `harmonic_analysis` + `spectrogram`: amplitude/phase of the diel and annual cycles, window effect, and a time-frequency map

### Distribution & Ranges

- **analysis_histogram_distribution.py** — Distribution analysis via histograms and percentiles
- **analysis_quantiles.py** — Percentile and quantile calculations
- **analysis_optimumrange.py** — Find optimal ranges for ecosystem responses

### Data Characterization

- **analysis_gapfinder.py** — Detect and characterize consecutive missing data periods; availability heatmap, gap-length histogram, size filters, summary statistics
- **analysis_gapstats.py** — Extended gap analysis: monthly/annual breakdown, long-gap listing, Rich console report, four-panel figure (availability heatmap, gap-spike timeline, monthly polar chart, gap-length histogram)
- **analysis_gridaggregator.py** — 2D grid aggregation with quantile, equal-width, and custom binning

## Common Patterns

**Decompose seasonal trends:**

```python
from diive.analysis import SeasonalTrendDecomposition

std = SeasonalTrendDecomposition(series=df['NEE'], period=365)
trend = std.trend
seasonal = std.seasonal
```

**Find lagged correlations (e.g., radiation vs. photosynthesis):**

```python
from diive.analysis import rank_drivers

# Ranks every other column against the target and scans lags.
# Columns: DRIVER, CORR, ABS_CORR, BEST_LAG, N (positive BEST_LAG = driver leads target).
ranked = rank_drivers(df, target='GPP', max_lag=24)
```

**Daily correlation between two series:**

```python
from diive.analysis import DailyCorrelation

dc = DailyCorrelation(series1=df['PAR'], series2=df['GPP'])
```

## Running Examples

```bash
# Decomposition & trends
uv run python examples/analysis/analysis_seasonaltrend.py

# Correlations & relationships
uv run python examples/analysis/analysis_daily_correlation.py
uv run python examples/analysis/analysis_granger.py

# Data characterization
uv run python examples/analysis/analysis_gapfinder.py
uv run python examples/analysis/analysis_gridaggregator.py
uv run python examples/analysis/analysis_quantiles.py

# Distribution & range analysis
uv run python examples/analysis/analysis_histogram_distribution.py
uv run python examples/analysis/analysis_optimumrange.py

# Spatial & spectral analysis
uv run python examples/analysis/analysis_gridaggregator.py
uv run python examples/analysis/analysis_harmonic.py

# Specialized analysis
uv run python examples/analysis/analysis_decoupling.py

# All examples
uv run python examples/run_all_examples.py
```

## Related Classes

See `dv.analysis` for full API documentation:

- `DailyCorrelation` — Daily correlation coefficients, summary statistics, anomaly detection
- `rank_drivers` — Rank drivers against a target with lag scanning
- `GrangerCausality` — Granger causality testing for predictive relationships
- `SeasonalTrendDecomposition` — STL decomposition
- `percentiles101` — Percentile-based analysis
- `GapFinder` — Gap detection and reporting
- `GapStats` — Extended gap analysis: monthly/annual breakdowns, long-gap listing, Rich report, multi-panel figure
- `GridAggregator` — 2D grid aggregation (quantile, equal-width, custom binning)
- `harmonic_analysis` / `spectrogram` — Spectral and time-frequency analysis
