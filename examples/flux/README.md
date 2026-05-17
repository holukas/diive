# Eddy Covariance Flux Processing Examples

Examples demonstrating flux processing, quality control, and high-resolution analysis for eddy covariance data.

## Terminology

**Directory abbreviations** used throughout flux processing examples:
- **`lowres/`** — Low-resolution (e.g., 30-minute) flux data processing. Typically averaged or aggregated time series.
- **`hires/`** — High-resolution (e.g., 10 Hz or 20 Hz) raw sonic and gas analyzer data. Pre-averaging analysis before flux averaging.

## Contents

### Processing Chain
- **fluxprocessingchain/fluxprocessingchain.py** — Complete multi-level Swiss FluxNet workflow (L2-L4.1): Quality flags, storage correction, outlier detection, USTAR filtering, gap-filling

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
- **hires/flux_lag_pwb.py** — PWB time lag detection: pre-whitening with block-bootstrap (Vitale et al. 2024), single averaging period, high-flux vs. low-flux comparison
- **hires/flux_lag_pwbopt.py** — PWBOPT batch pipeline: multi-period PWB detection with S1/S2/S3 selection and standard vs. pre-filtered strategy comparison
- **hires/flux_windrotation.py** — Wind rotation and tilt correction for coordinate transformation
- **hires/flux_fluxdetectionlimit.py** — Flux detection limit and measurement sensitivity

## Related Documentation

Available classes and functions in `diive.pkgs.flux`:
- **TimeLagAnalysis** — Time lag detection and visualization for gas concentrations
- **MaxCovariance** — Time lag detection via cross-covariance maximisation
- **PreWhiteningBootstrap** — PWB time lag detection (Vitale et al. 2024): pre-whitening + block-bootstrap, robust for low-magnitude fluxes (CH4, N2O)
- **RandomUncertaintyPAS20** — Measurement uncertainty quantification
- **FlagMultipleConstantUstarThresholds** — USTAR filtering with multiple thresholds
- **UstarMovingPointDetection** — Moving-point USTAR detection (Papale et al. 2006)
- **UstarVekuriThresholdDetection** — Quantile-based USTAR detection (Vekuri method)
- **UstarBootstrapThresholds** — Multi-year bootstrap wrapper for any USTAR detector; 3-year sliding window, per-year p16/p50/p84, pooled CUT threshold
- **ScopApplicator** — SCOP self-heating correction for open-path IRGA
- **FluxProcessingChain** — Complete multi-level flux processing workflow
- High-resolution analysis methods (lag detection, wind rotation)
- Flux variable detection and nomenclature

## Use Cases

**Process raw eddy covariance data:**
```python
from diive.pkgs.flux.fluxprocessingchain import FluxProcessingChain

# Complete multi-level workflow
fpc = FluxProcessingChain(
    df=df,
    fluxcol='FC',
    site_lat=47.48,
    site_lon=8.36,
    utc_offset=1
)

# L2: Quality flag expansion
fpc.level2_quality_flag_expansion()
fpc.finalize_level2()

# L3.1-L3.3: Corrections, outlier removal, USTAR filtering
fpc.level31_storage_correction()
fpc.level32_stepwise_outlier_detection()
fpc.level33_constant_ustar(thresholds=[0.09])

# L4.1: Gap-fill with machine learning
fpc.level41_longterm_xgboost(
    features=['TA', 'VPD', 'SW_IN'],
    n_estimators=500
)

results = fpc.get_data()
```

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
from diive.pkgs.flux.hires import FluxLag, WindRotation

# Detect time lag via covariance
lag = FluxLag(
    w_10hz=df['w'],
    c_10hz=df['CO2'],
    h_10hz=df['H2O']
)
optimal_lag = lag.get_lag()

# Rotate wind to streamline coordinates
rotated = WindRotation(
    u=df['u'], v=df['v'], w=df['w'],
    canopy_height=20
)
```

## Running Examples

```bash
# Complete multi-level processing workflow (recommended starting point)
uv run python examples/flux/fluxprocessingchain/fluxprocessingchain.py

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
uv run python examples/flux/hires/flux_windrotation.py
uv run python examples/flux/hires/flux_fluxdetectionlimit.py

# Run all flux examples
uv run python examples/run_all_examples.py
```

## Standards & Best Practices

- **FLUXNET conventions** — Data flows through 5 levels (L2→L3.1→L3.2→L3.3→L4.1)
- **Swiss FluxNet methodology** — Quality flags, storage correction, USTAR filtering
- **Unit consistency** — Always use SI units (W/m², K, hPa)
- **QC/QF flags** — Combine multiple quality tests into single QCF flag
- **Uncertainty propagation** — Random + systematic uncertainty estimation
