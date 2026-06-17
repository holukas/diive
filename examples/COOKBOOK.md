# DIIVE Cookbook

Six essential workflows, each in minimal working code.
Start here, then follow the links to full examples.

---

## 1. Load data and inspect

```python
import diive as dv

df = dv.load_exampledata_parquet()

print(df.shape)           # (rows, columns)
print(df.index[:3])       # timestamp index
print(df.dtypes)          # column types
print(dv.sstats(df))      # summary stats (mean, std, gaps, ...)
```

The built-in example dataset is a multi-year 30-min eddy covariance record (CH-DAV).
Swap in your own DataFrame from any source — diive only requires a `DatetimeIndex`.

Full examples: [io/io_load_save_parquet.py](io/io_load_save_parquet.py),
[io/io_read_single_file_with_readfiletype.py](io/io_read_single_file_with_readfiletype.py)

---

## 2. Clean a timestamp index

```python
import diive as dv

df = dv.load_exampledata_parquet()

sanitizer = dv.times.TimestampSanitizer(
    data=df,
    output_middle_timestamp=True,   # convert end-of-period to mid-period
    nominal_freq='30min',            # expected sampling frequency
    verbose=True
)
df_clean = sanitizer.get()
status   = sanitizer.get_status()   # rows added/removed, frequency confidence

print(status)
```

Handles: NaT values, duplicates, wrong order, irregular gaps, frequency detection.

Full example: [times/times_timestamp_sanitizer.py](times/times_timestamp_sanitizer.py)

---

## 3. Remove outliers from a time series

```python
import diive as dv
from diive.preprocessing.outlier_detection import StepwiseOutlierDetection

df = dv.load_exampledata_parquet()

detector = StepwiseOutlierDetection(
    dfin=df,
    col='NEE_CUT_REF_orig',
    site_lat=47.48,
    site_lon=8.37,
    utc_offset=1
)

# Chain methods — each operates on data already cleaned by the previous step
detector.flag_outliers_hampel_test(window_length=7 * 48, n_sigma_daytime=4.5,
                                   n_sigma_nighttime=4.5, showplot=False, verbose=False)
detector.addflag()

detector.flag_outliers_localsd_test(n_sd=[3.5, 3.5], winsize=[24, 24],
                                    separate_daytime_nighttime=True,
                                    showplot=False, verbose=False)
detector.addflag()

cleaned  = detector.series_hires_cleaned
original = detector.series_hires_orig
print(f"Removed {(original.notna() & cleaned.isna()).sum()} outliers")
```

Available methods on `StepwiseOutlierDetection`: `flag_outliers_hampel_test`,
`flag_outliers_localsd_test`, `flag_outliers_zscore_test`,
`flag_outliers_zscore_rolling_test`, `flag_outliers_lof_test`,
`flag_outliers_absolutelimits_test`, `flag_outliers_increments_zcore_test`.

Full example: [preprocessing/outlier_detection/outlier_stepwise.py](preprocessing/outlier_detection/outlier_stepwise.py)

---

## 4. Gap-fill a variable with Random Forest

```python
import diive as dv

TARGET_COL = 'NEE_CUT_REF_orig'

df = dv.load_exampledata_parquet()
df = df.loc['2020'].copy()                       # one year keeps the demo fast
df = df[[TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']]

# Step 1: build temporal features
engineer = dv.gapfilling.FeatureEngineer(
    target_col=TARGET_COL,
    features_rolling=[2, 4, 12, 24, 48],
    vectorize_timestamps=True,
    add_continuous_record_number=True
)
df_eng = engineer.fit_transform(df)

# Step 2: train, reduce features, gap-fill
rfts = dv.gapfilling.RandomForestTS(
    input_df=df_eng,
    target_col=TARGET_COL,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rfts.reduce_features(shap_threshold_factor=0.5)
rfts.trainmodel(showplot_scores=False, showplot_importance=False)
rfts.fillgaps(showplot_scores=False, showplot_importance=False)

# Step 3: access results
r = rfts.results                  # GapFillingResult dataclass
print(r.scores['r2'])             # gap-filling R²
gapfilled = r.gapfilled           # gap-filled Series (flag: 0=observed, 1=filled)
```

Swap `RandomForestTS` for `XGBoostTS` (same API) or `FluxMDS` for a no-training alternative.

Full examples: [gapfilling/gapfill_randomforest.py](gapfilling/gapfill_randomforest.py),
[gapfilling/gapfill_xgboost.py](gapfilling/gapfill_xgboost.py),
[gapfilling/gapfill_mds.py](gapfilling/gapfill_mds.py),
[gapfilling/gapfill_comparison.py](gapfilling/gapfill_comparison.py)

---

## 5. Run the full flux processing chain (L2–L4.1)

```python
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.flux.fluxprocessingchain import FluxConfig, init_flux_data, run_chain

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.loc['2024-06':'2024-06']           # one month for speed
df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                      if c in df.columns])  # reserved names; recomputed fresh

# Initialise the container: potential radiation, day/night flags, frozen meta.
data = init_flux_data(
    df=df,
    fluxcol='FC',
    site_lat=47.41887,
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,                 # W m-2
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)

# One FluxConfig drives the whole L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1 chain.
cfg = FluxConfig(
    fluxcol='FC',
    ustar_thresholds=[0.18],                # L3.3 constant USTAR threshold
    ustar_labels=['CUT_50'],
    mds_swin='SW_IN_T1_47_1_gfXG',          # L4.1 MDS drivers (must be in data.full_df)
    mds_ta='TA_T1_47_1_gfXG',
    mds_vpd='VPD_kPa',                       # MDS needs VPD in kPa, not hPa
    gapfill_mds=True,
    gapfill_rf=True,
    gapfilling_features=['TA_T1_47_1_gfXG', 'SW_IN_T1_47_1_gfXG', 'VPD_kPa'],
)

data = run_chain(data, cfg)
print(data.summary())
cols = data.gapfilled_cols()               # {method: {ustar_scenario: column_name}}
```

Returns a complete audit trail: one flag column per test, QCF composite flag, and
gap-filled flux at every USTAR scenario. For full control over every detector and
model knob, use the composable per-level API (`run_level2`, `run_level31`, ...).

Full example: [flux/fluxprocessingchain/fluxprocessingchain_runchain.py](flux/fluxprocessingchain/fluxprocessingchain_runchain.py) |
composable version: [flux/fluxprocessingchain/fluxprocessingchain_composable.py](flux/fluxprocessingchain/fluxprocessingchain_composable.py)

---

## 6. Visualize a time series as a heatmap

```python
import diive as dv

df = dv.load_exampledata_parquet()

# DateTime heatmap: hours on y-axis, days on x-axis
dv.plotting.HeatmapDateTime(series=df['NEE_CUT_REF_f']).plot()
```

Other common plots:

```python
# Year x month matrix (mean per cell)
dv.plotting.HeatmapXYZ(
    x=df.index.month,
    y=df.index.year,
    z=df['NEE_CUT_REF_f']
).plot()

# Scatter
dv.plotting.ScatterXY(x=df['Tair_f'], y=df['NEE_CUT_REF_f']).plot()

# Diel cycle by month
dv.plotting.DielCycle(series=df['NEE_CUT_REF_f']).plot()
```

All plotting classes follow the two-phase pattern: pass data to `__init__`, call
`.plot(ax=..., title=..., ...)` for styling. Calling `.plot()` a second time with a
different `ax` reuses the same computed data.

Full examples: [visualization/plot_heatmap_datetime_basic.py](visualization/plot_heatmap_datetime_basic.py),
[visualization/plot_scatter_xy_basic.py](visualization/plot_scatter_xy_basic.py),
[visualization/plot_dielcycle.py](visualization/plot_dielcycle.py)

---

## What next?

| Goal | Jump to |
|------|---------|
| All examples by topic | [CATALOG.md](CATALOG.md) |
| Complete example listing | [README.md](README.md) |
| Gap-filling method comparison | [gapfilling/gapfill_comparison.py](gapfilling/gapfill_comparison.py) |
| Outlier detection method comparison | [preprocessing/outlier_detection/outlier_stepwise.py](preprocessing/outlier_detection/outlier_stepwise.py) |
| High-resolution (10 Hz) analysis | [flux/hires/flux_windrotation.py](flux/hires/flux_windrotation.py) |
| Detect timestamp clock errors | [preprocessing/qaqc/qaqc_detect_timestamp_shifts.py](preprocessing/qaqc/qaqc_detect_timestamp_shifts.py) |
| Optimize gap-filling hyperparameters | [gapfilling/gapfill_optimize_randomforest.py](gapfilling/gapfill_optimize_randomforest.py) |
