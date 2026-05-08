# Outlier Detection Methods Examples

Examples demonstrating various outlier detection methods for quality control and data cleaning.

## Contents

- **absolutelimits.py** — Min/max threshold-based outlier detection
- **hampel.py** — Robust Hampel filter (Median Absolute Deviation)
- **incremental.py** — Incremental/differencing-based outlier detection
- **localsd.py** — Local standard deviation adaptive threshold
- **lof.py** — Local Outlier Factor (density-based)
- **manualremoval.py** — Explicit manual data removal
- **stepwise.py** — Sequential multi-method outlier detection
- **trim.py** — Symmetric trimmed mean approach (TrimLow)
- **zscore.py** — Z-score and rolling z-score methods

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
