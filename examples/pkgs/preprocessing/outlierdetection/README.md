# Outlier Detection Methods Examples

Examples demonstrating 10 outlier detection methods for quality control, anomaly identification, and data cleaning.

## Methods by Complexity

**Beginner (simple, deterministic):**
- **outlier_absolutelimits.py** — Enforce physical min/max constraints
- **outlier_manualremoval.py** — Explicitly flag known problematic timestamps or periods
- **outlier_trim.py** — Symmetric trimmed mean approach (TrimLow)

**Intermediate (statistical, adaptive):**
- **outlier_hampel.py** — Median Absolute Deviation (MAD) in rolling windows
- **outlier_zscore.py** — Z-score thresholding (global, rolling, day/night, increments)
- **outlier_localsd.py** — Local standard deviation with adaptive windows
- **outlier_incremental.py** — Detect spikes via abrupt changes between records

**Advanced (machine learning, multi-method):**
- **outlier_lof.py** — Local Outlier Factor (density-based anomalies)
- **outlier_stepwise.py** — Chain multiple methods for progressive filtering

## Related Documentation

See `diive.pkgs.preprocessing.outlierdetection` for available detection classes:
- `AbsoluteLimits` — Threshold-based detection
- `Hampel` — MAD-based robust detection
- `LocalSD` — Adaptive local threshold
- `zScore` — Statistical threshold
- `zScoreRolling` — Adaptive rolling threshold
- `LocalOutlierFactor` — Density-based detection
- `StepwiseOutlierDetection` — Multi-stage orchestration

## Usage

```bash
uv run python examples/pkgs/preprocessing/outlierdetection/hampel.py
uv run python examples/pkgs/preprocessing/outlierdetection/stepwise.py
```

Or run all outlier detection examples:

```bash
uv run python examples/run_all_examples.py
```
