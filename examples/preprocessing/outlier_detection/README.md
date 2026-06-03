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

See `dv.outliers` for available detection classes:
- `AbsoluteLimits` — Threshold-based detection
- `Hampel` — MAD-based robust detection
- `LocalSD` — Adaptive local threshold
- `zScore` — Statistical threshold
- `zScoreRolling` — Adaptive rolling threshold
- `LocalOutlierFactor` — Density-based detection
- `StepwiseOutlierDetection` — Multi-stage orchestration

## Use Cases

**Quick spike detection (Hampel filter):**
```python
from diive.preprocessing.outlier_detection import Hampel

# Fast, robust detection using Median Absolute Deviation
detector = Hampel(
    dfin=df,
    col='NEE',
    n_sigma=5.5,
    win_size=48
)
flags = detector.flags_outliers  # 0=good, 2=outlier
cleaned = df[detector.flags_outliers != 2]
```

**Statistical thresholding (Z-score):**
```python
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection

# Global z-score
detector = StepwiseOutlierDetection(dfin=df, col='FCH4', site_lat=47.5, site_lon=8.4, utc_offset=1)
detector.flag_outliers_zscore_test(thres_zscore=4)

# Day/night separated z-score
detector.flag_outliers_zscore_test(
    thres_zscore=4,
    separate_daytime_nighttime=True
)
```

**Absolute physical limits:**
```python
from diive.preprocessing.outlier_detection import AbsoluteLimits

# Enforce known physical bounds
detector = AbsoluteLimits(
    dfin=df,
    col='RH',
    min_val=0,
    max_val=100
)
```

**Multi-stage filtering (sequential chain):**
```python
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection

# Progressive filtering: aggressive first, then refine
detector = StepwiseOutlierDetection(dfin=df, col='NEE', site_lat=46.8, site_lon=8.6)

detector.flag_outliers_hampel_dtnt_test(n_sigma=5.5)
detector.addflag()

detector.flag_outliers_localsd_test(n_sd=[3.5, 3.5], winsize=[24, 24])
detector.addflag()

detector.flag_outliers_zscore_test(thres_zscore=4)
detector.addflag()

# View final cleaned series
cleaned = detector.series_hires_cleaned
```

## Running Examples

```bash
# Beginner-friendly (simple, deterministic)
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_absolutelimits.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_manualremoval.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_trim.py

# Intermediate (statistical, adaptive)
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_hampel.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_zscore.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_localsd.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_incremental.py

# Advanced (machine learning, multi-method)
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_lof.py
uv run python examples/pkgs/preprocessing/outlier_detection/outlier_stepwise.py

# Run all outlier detection examples
uv run python examples/run_all_examples.py
```

## Choosing a Method

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| Quick QA/QC | Hampel | Fast, robust to skew, works with non-normal distributions |
| Known physical limits | AbsoluteLimits | Simple, deterministic, no parameters to tune |
| Normal distribution | zScore | Works well when data is normally distributed |
| Varying patterns | LocalSD | Adaptive to local variability |
| Abrupt changes | Incremental | Detects step changes, not global outliers |
| Unknown patterns | LOF | Density-based, finds anomalies without assumptions |
| Multiple tests | StepwiseOutlierDetection | Combine methods, each filters progressively |
