# Eddy Covariance Flux Processing Examples

Examples demonstrating flux processing, quality control, and high-resolution analysis for eddy covariance data.

## Terminology

**Directory abbreviations** used throughout flux processing examples:
- **`lowres/`** — Low-resolution (e.g., 30-minute) flux data processing. Typically averaged or aggregated time series.
- **`hires/`** — High-resolution (e.g., 10 Hz or 20 Hz) raw sonic and gas analyzer data. Pre-averaging analysis before flux averaging.

## Contents

### Processing Chain
- **fluxprocessingchain/fluxprocessingchain_runchain.py** — Single-call `run_chain(data, FluxConfig)` example. Minimal config drives the full L2→L4.1 pipeline with sensible defaults. The easy path; use this when you want the chain to "just work".
- **fluxprocessingchain/fluxprocessingchain_composable.py** — Full L2→L4.1 pipeline using composable callables; RF, XGBoost, and MDS gap-filling from the same L3.3 state; on-demand `gap_stats()` after L3.3; `plot_gapfilled_heatmaps()` (side-by-side heatmap comparison) and `plot_cumulative_comparison()` (all methods on one axes) after L4.1. The full-control path — every detector class, model hyperparameter, MDS tolerance, and diagnostic flag is reachable here.

### Low-Resolution Flux Processing
- **lowres/flux_timelag_analysis.py** — Time lag detection and visualization for gas concentrations
- **lowres/flux_common.py** — Flux variable base detection and nomenclature
- **lowres/flux_hqflux.py** — Highest-quality flux filtering with Hampel outlier detection
- **lowres/flux_selfheating.py** — SCOP self-heating correction (quick demo)
- **lowres/flux_selfheating_production.py** — Complete production workflow: scaling factors from parallel measurements, applied to long-term data
- **lowres/flux_uncertainty.py** — Random uncertainty estimation (PAS20 method)
- **lowres/flux_ustar_mp_detection.py** — Moving Point (MP) USTAR detection (Papale et al. 2006) with multi-year bootstrap
- **lowres/flux_ustar_vekuri_detection.py** — Quantile-based USTAR detection (Vekuri method) with multi-year bootstrap
- **lowres/flux_ustar_method_comparison.py** — Side-by-side comparison of ONEFlux and Vekuri USTAR approaches

### High-Resolution (10 Hz) Flux Analysis
- **hires/flux_lag.py** — Time lag detection using MaxCovariance covariance analysis
- **hires/flux_lag_pwb.py** — PWB time lag detection: pre-whitening with block-bootstrap (Vitale et al. 2024), single averaging period, high-flux vs. low-flux comparison; demonstrates the 4-combination logic via `var_tsonic`
- **hires/flux_lag_pwbopt.py** — PWBOPT batch pipeline: multi-period PWB detection with S1/S2/S3 selection and standard vs. pre-filtered strategy comparison
- **hires/flux_lag_pwb_batch.py** — `PwbBatchDetection` Python API demo: distributes PWB detection across CPU cores with `ProcessPoolExecutor`, shows live Rich progress (growing results table + progress bar), applies PWBOPT post-processing (standard and pre-filtered strategies), and generates batch summary figures (3-panel per scalar + scatter/KDE via `PwboptLagPlot`)
- **hires/flux_lag_pwb_batch_cli.py** — CLI demo: generates synthetic EddyPro files and invokes `python -m diive.flux.hires.lag_pwb` as a subprocess; shows all available CLI flags
- **hires/flux_apply_tlag_cli.py** — `TlagApplier` CLI demo (`diive-tlag-apply-batch`): applies PWBOPT-detected lags from a `tlag_results.csv` to raw EC files, shifting each scalar by `round(tlag_s · hz)` rows; format-agnostic, with `--period-key-regex` / `--file-key-regex` bridging rotated-vs-raw filenames
- **hires/flux_detect_remove_tui_demo.py** — launches the `diive-tlag-pwb-detect-remove` **Textual TUI** in demo mode (no input data): a single-screen, soft-themed interface that animates the two-phase progress and streams the Rich-styled per-chunk output into a log panel. **Interactive** — not part of `run_all_examples.py`. Run directly: `uv run python examples/flux/hires/flux_detect_remove_tui_demo.py`
- **hires/flux_windrotation.py** — Double rotation tilt correction (`WindDoubleRotation`) followed by Reynolds decomposition (`reynolds_decomposition`) to extract turbulent fluctuations and compute eddy covariance fluxes
- **hires/flux_fluxdetectionlimit.py** — Flux detection limit and measurement sensitivity

## Related Documentation

Available classes and functions in `dv.flux`:
- **TimeLagAnalysis** — Time lag detection and visualization for gas concentrations
- **MaxCovariance** — Time lag detection via cross-covariance maximisation
- **PreWhiteningBootstrap** — PWB time lag detection (Vitale et al. 2024): pre-whitening + block-bootstrap, robust for low-magnitude fluxes (CH4, N2O). Provide `var_tsonic` to enable the full 4-combination RFlux v3.2.0 logic (strongly recommended for trace gases). **Requires wind-rotation-corrected input** (double rotation or planar-fit; e.g. EddyPro "Advanced" rotated output) — a non-zero mean W biases the cross-correlation.
- **PwbBatchDetection** — Parallel batch wrapper around `PreWhiteningBootstrap`. **Requires wind-rotation-corrected input** (same requirement as above).: distributes many EddyPro averaging-period files across CPU cores, collects results into a single DataFrame, writes a checkpoint CSV after every completed file (crash-safe), applies PWBOPT S1/S2/S3 selection and optional HDI pre-filter, and produces batch summary figures. Also callable as a CLI module: `python -m diive.flux.hires.lag_pwb --help` (alias: `diive-tlag-pwb-batch`).
- **TlagApplier** — Applies already-detected lags to raw EC files: reads a `tlag_results.csv` (from `PwbBatchDetection`) and shifts each scalar column backward by `round(tlag_s · hz)` rows, writing a parallel directory of lag-corrected files with the original header and column order preserved. Default lag column is `{prefix}_tlag_final_pf_s` (pre-filtered, gap-filled PWBOPT). Format-agnostic (`--sep` / `--skiprows` / `--extra-rows`); when detection ran on rotated files but the lag must be removed from differently-named raw files, `--period-key-regex` / `--file-key-regex` extract a common key per period. CLI: `diive-tlag-apply-batch`.
- **PerFilePipeline / `process_one_file`** — End-to-end **two-phase per-chunk** PWB pipeline. Splits each long raw file into fixed-length chunks (default 30 min); **phase 1 (detect)** rotates each chunk's wind in memory and runs PWB per scalar (no data written), PWBOPT then picks the *best* lag per chunk across the full sequence, and **phase 2 (remove)** shifts each scalar by that PWBOPT lag and writes one lag-corrected file per chunk. Unlike raw per-chunk removal, wide-HDI chunks get the neighbouring optimal lag. Parallel unit is one chunk (`ProcessPoolExecutor`); the live display stacks one row per worker plus an overall bar with ETA. Output is numbered by phase: `1_lag_detection/` (summary CSV, checkpoints, plots) and `2_lag_removed/` (the corrected chunk files — a clean input directory for the next flux step). CLI: `diive-tlag-pwb-detect-remove`. **Downstream flux software must run with EC time-lag maximization disabled.**
- **RandomUncertaintyPAS20** — Measurement uncertainty quantification
- **FlagMultipleConstantUstarThresholds** — USTAR filtering with multiple constant thresholds
- **FlagMultipleVariableUstarThresholds** — USTAR filtering with time-varying (per-record, e.g. per-year VUT) thresholds
- **UstarMovingPointDetection** — Moving-point USTAR detection (Papale et al. 2006)
- **UstarVekuriThresholdDetection** — Quantile-based USTAR detection (Vekuri method)
- **UstarBootstrapThresholds** — Multi-year bootstrap wrapper for any USTAR detector; 3-year sliding window. Returns **VUT** (variable, per-year p16/p50/p84 via `get_vut_thresholds()` / `run()`) and **CUT** (constant, pooled across years via `get_cut_threshold()`). diive's VUT is smoothed over the 3-year window (differs from strict single-year ONEFlux VUT)
- **ScopApplicator** — SCOP self-heating correction for open-path IRGA
- **`run_chain` / `FluxConfig`** — Single-call driver for the full L2→L4.1 flux processing pipeline; one `FluxConfig` per flux variable
- **Composable level callables** — `init_flux_data`, `run_level2`, `run_level31`, `make_level32_detector` + `run_level32`, `run_level33_constant_ustar` / `run_level33_variable_ustar` / `run_level33_ustar_detection` (mode `'cut'`/`'vut'`), `run_level41_mds` / `_rf` / `_xgb`; pure functions on a typed `FluxLevelData` container
- **`add_driver(data, series)`** — Add a computed driver column to `data.full_df` where L4.1 will read it
- **WindDoubleRotation** — Double rotation tilt correction (Wilczak et al. 2001): aligns coordinate system with mean wind direction; exposes `theta`, `phi`, `u2`, `v2`, `w2`
- **reynolds_decomposition** — Reynolds decomposition `x' = x - mean(x)`; applied after rotation to extract turbulent fluctuations of wind components and scalars before flux covariance calculation
- High-resolution analysis methods (lag detection, wind rotation)
- Flux variable detection and nomenclature

## Use Cases

**Process raw eddy covariance data — single-call driver:**
```python
from diive.flux.fluxprocessingchain import (
    FluxConfig, init_flux_data, run_chain,
)

cfg = FluxConfig(
    fluxcol='FC',
    ustar_thresholds=[0.18], ustar_labels=['CUT_50'],
    outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
    gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa_1_1_1'],
    level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
    mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa_1_1_1',
)
data = init_flux_data(df, fluxcol='FC', site_lat=47.48, site_lon=8.36, utc_offset=1)
data = run_chain(data, cfg)

results = data.fpc_df          # all per-level and gap-filled columns
cols = data.gapfilled_cols()   # {'rf': {'CUT_50': '...'}, 'xgb': ..., 'mds': ...}
```

**Composable per-level API** — for custom L3.2 outlier pipelines, custom feature engineering, or per-level inspection, call each `run_level*` directly. See `examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py`.

**Analyze time lag and measurement quality:**
```python
from diive import TimeLagAnalysis, RandomUncertaintyPAS20

# Detect optimal time lags for gas concentrations
analysis = TimeLagAnalysis(
    df=df,
    ignore_fringe_bins=[5, 10],
    lag_window_min=0.10,
    lag_window_max=1.00
)
co2_results = analysis.analyze_gas('CO2')
fig = analysis.plot_gas('CO2', outdir='output/')

# Quantify measurement uncertainty
unc = RandomUncertaintyPAS20(flux_series=df['FC'])
uncertainty = unc.get_uncertainty()
```

**High-resolution analysis:**
```python
import diive as dv

# Double rotation: align coordinate system with mean wind direction
wr = dv.WindDoubleRotation(u=df['u'], v=df['v'], w=df['w'])

# Reynolds decomposition: extract turbulent fluctuations
w_prime = dv.reynolds_decomposition(wr.w2)
c_prime = dv.reynolds_decomposition(df['CO2'])

# Eddy covariance flux: w'c'
flux = (w_prime * c_prime).mean()

# Time lag detection via cross-covariance maximisation
mc = dv.MaxCovariance(
    df=df,
    var_reference='w',
    var_lagged='CO2',
    lgs_winsize_from=-100,
    lgs_winsize_to=0,
    shift_stepsize=1,
    segment_name='CO2 lag'
)
mc.run()
```

## Running Examples

```bash
# Complete multi-level processing workflow (recommended starting point)
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py

# Low-resolution (30-min) processing
uv run python examples/flux/lowres/flux_timelag_analysis.py
uv run python examples/flux/lowres/flux_selfheating.py
uv run python examples/flux/lowres/flux_uncertainty.py
uv run python examples/flux/lowres/flux_ustar_mp_detection.py
uv run python examples/flux/lowres/flux_ustar_vekuri_detection.py
uv run python examples/flux/lowres/flux_ustar_method_comparison.py

# High-resolution (10 Hz) analysis
uv run python examples/flux/hires/flux_lag.py
uv run python examples/flux/hires/flux_lag_pwb.py
uv run python examples/flux/hires/flux_lag_pwbopt.py
uv run python examples/flux/hires/flux_lag_pwb_batch.py
uv run python examples/flux/hires/flux_lag_pwb_batch_cli.py
uv run python examples/flux/hires/flux_windrotation.py
uv run python examples/flux/hires/flux_fluxdetectionlimit.py

# PWB batch detection via CLI alias (real EddyPro files)
# diive-tlag-pwb-batch is a console-script shortcut for
# python -m diive.flux.hires.lag_pwb
uv run diive-tlag-pwb-batch \
    --input-dir /path/to/hires_files \
    --output-dir /path/to/results \
    --scalar CH4:ch4 --scalar N2O:n2o \
    --col-w w --col-tsonic ts \
    --usecols 0 1 2 3 6 7 \
    --col-names u v w ts ch4 n2o \
    --skiprows 9 --hz 20 --n-bootstrap 99 --n-workers 4 --save-plots

# Per-chunk detect + remove (two-phase): writes lag-corrected files ready for
# the next flux step. Splits long raw files into chunks, detects the lag per
# chunk, optimises it across chunks (PWBOPT), then removes the best lag.
# 10 Hz / 60-min chunks: just pass --hz 10 / --chunk-seconds 3600.
uv run diive-tlag-pwb-detect-remove \
    --input-dir /path/to/raw_6h_csv_files \
    --output-dir /path/to/output \
    --col-u u --col-v v --col-w w --col-tsonic ts \
    --scalar CH4:ch4 --scalar N2O:n2o \
    --skiprows 0 --extra-rows 2 --sep "," \
    --hz 20 --chunk-seconds 1800 --min-chunk-seconds 300 \
    --n-workers 8 --save-plots
# -> /path/to/output/2_lag_removed/  holds the corrected chunk files

# Apply already-detected lags from a tlag_results.csv to raw files (standalone)
uv run diive-tlag-apply-batch \
    --input-dir /path/to/raw_files \
    --output-dir /path/to/aligned \
    --results-csv /path/to/tlag_results.csv \
    --scalar CH4:ch4 --scalar N2O:n2o \
    --hz 20 --skiprows 9

# Run all flux examples
uv run python examples/run_all_examples.py
```

## Standards & Best Practices

- **FLUXNET conventions** — Data flows through 5 levels (L2→L3.1→L3.2→L3.3→L4.1)
- **Swiss FluxNet methodology** — Quality flags, storage correction, USTAR filtering
- **Unit consistency** — Always use SI units (W/m², K, hPa)
- **QC/QF flags** — Combine multiple quality tests into single QCF flag
- **Uncertainty propagation** — Random + systematic uncertainty estimation
