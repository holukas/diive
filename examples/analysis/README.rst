Time Series Analysis Examples
=============================

Examples demonstrating statistical analysis, decomposition, and pattern detection for time series data.

14 examples covering correlation, spectral analysis, gap detection, grid aggregation, decomposition, and compound-extreme detection.

Examples by Method
------------------

Correlation & Covariance
~~~~~~~~~~~~~~~~~~~~~~~~

- **analysis_daily_correlation.py** — Daily correlation coefficients for quality checks, relationship analysis, and statistical methods
- **analysis_driveranalysis.py** — *(experimental;* ``dv.analysis.experimental``\ *)* Evidence-triangulation driver attribution organized by epistemic level (association → temporal prediction → causation), with a convergence/divergence summary across SHAP, ALE, lagged/scale-resolved/stratified importance, and Granger
- **analysis_granger.py** — Granger causality testing to detect predictive relationships between time series
- **analysis_decoupling.py** — Stratified binning to reveal how ecosystem responses change across temperature ranges

Decomposition & Trends
~~~~~~~~~~~~~~~~~~~~~~

- **analysis_seasonaltrend.py** — STL decomposition separating trend and seasonality
- **analysis_harmonic.py** — ``harmonic_analysis`` + ``spectrogram``: amplitude/phase of the diel and annual cycles, window effect, and a time-frequency map

Extremes
~~~~~~~~

- **analysis_compound_extremes.py** — ``CompoundExtremes`` + ``CompoundExtremesPlot``: classify months/days into none/air/soil/compound dry-hot extremes from VPD and SWC z-scores, with the quadrant scatter (after Wang et al., Fig. 2)

Distribution & Ranges
~~~~~~~~~~~~~~~~~~~~~

- **analysis_histogram_distribution.py** — Distribution analysis via histograms and percentiles
- **analysis_quantiles.py** — Percentile and quantile calculations
- **analysis_optimumrange.py** — Find optimal ranges for ecosystem responses
- **analysis_keep_records_where.py** — ``dv.keep_records_where``: keep records of a target where a condition variable falls in a [lower, upper] range; non-destructive, with masking vs dropping, one-sided ranges, and inverted selection

Data Characterization
~~~~~~~~~~~~~~~~~~~~~

- **analysis_gapfinder.py** — Detect and characterize consecutive missing data periods; availability heatmap, gap-length histogram, size filters, summary statistics
- **analysis_gapstats.py** — Extended gap analysis: monthly/annual breakdown, long-gap listing, Rich console report, four-panel figure (availability heatmap, gap-spike timeline, monthly polar chart, gap-length histogram)
- **analysis_gridaggregator.py** — 2D grid aggregation with quantile, equal-width, and custom binning

Common Patterns
---------------

**Decompose seasonal trends:**

.. code-block:: python

   import diive as dv

   std = dv.analysis.SeasonalTrendDecomposition(series=df['NEE'], seasonal_period=365)
   trend = std.trend
   seasonal = std.seasonal

**Find lagged correlations (e.g., radiation vs. photosynthesis):**

.. code-block:: python

   import diive as dv

   # Ranks every other column against the target and scans lags.
   # Columns: DRIVER, CORR, ABS_CORR, BEST_LAG, N (positive BEST_LAG = driver leads target).
   ranked = dv.analysis.rank_drivers(df, target='GPP', max_lag=24)

**Daily correlation between two series:**

.. code-block:: python

   import diive as dv

   dc = dv.analysis.DailyCorrelation(s1=df['PAR'], s2=df['GPP'])
   correlations = dc.correlations

Running Examples
----------------

.. code-block:: bash

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

   # Compound-extreme detection
   uv run python examples/analysis/analysis_compound_extremes.py

   # Specialized analysis
   uv run python examples/analysis/analysis_decoupling.py

   # All examples
   uv run python examples/run_all_examples.py

Related Classes
---------------

See ``dv.analysis`` for full API documentation:

- ``DailyCorrelation`` — Daily correlation coefficients, summary statistics, anomaly detection
- ``rank_drivers`` — Rank drivers against a target with lag scanning
- ``GrangerCausality`` — Granger causality testing for predictive relationships
- ``SeasonalTrendDecomposition`` — STL decomposition
- ``percentiles101`` — Percentile-based analysis
- ``GapFinder`` — Gap detection and reporting
- ``GapStats`` — Extended gap analysis: monthly/annual breakdowns, long-gap listing, Rich report, multi-panel figure
- ``GridAggregator`` — 2D grid aggregation (quantile, equal-width, custom binning)
- ``harmonic_analysis`` / ``spectrogram`` — Spectral and time-frequency analysis
