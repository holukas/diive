# Eddy Covariance Flux Processing Examples

Examples demonstrating flux processing, quality control, and high-resolution analysis for eddy covariance data.

## Contents

### Processing Chain
- **fluxprocessingchain/fluxprocessingchain.py** — Complete multi-level Swiss FluxNet workflow (L2-L4.1): Quality flags, storage correction, outlier detection, USTAR filtering, gap-filling

### Low-Resolution Flux Processing
- **lowres/flux_common.py** — Flux variable base detection and nomenclature
- **lowres/flux_hqflux.py** — Highest-quality flux filtering with Hampel outlier detection
- **lowres/flux_selfheating.py** — Oxygen sensor self-heating correction (SCOP methodology)
- **lowres/flux_uncertainty.py** — Random uncertainty estimation (PAS20 method)
- **lowres/flux_ustarthreshold.py** — USTAR threshold detection and filtering
- **lowres/flux_ustar_mp_detection.py** — Moving Point (MP) USTAR detection method

### High-Resolution (10 Hz) Flux Analysis
- **hires/flux_lag.py** — Time lag detection using MaxCovariance covariance analysis
- **hires/flux_windrotation.py** — Wind rotation and tilt correction for coordinate transformation
- **hires/flux_fluxdetectionlimit.py** — Flux detection limit and measurement sensitivity

## Related Documentation

See `diive.pkgs.flux` for available processing classes and functions:
- Flux quality control and filtering
- High-resolution analysis methods
- Sensor corrections
- USTAR filtering for low-turbulence periods
- Gap-filling methods

## Running Examples

```bash
# Processing chain (complete workflow)
uv run python examples/pkgs/flux/fluxprocessingchain/fluxprocessingchain.py

# Low-resolution processing
uv run python examples/pkgs/flux/lowres/flux_common.py
uv run python examples/pkgs/flux/lowres/flux_hqflux.py
uv run python examples/pkgs/flux/lowres/flux_selfheating.py
uv run python examples/pkgs/flux/lowres/flux_uncertainty.py
uv run python examples/pkgs/flux/lowres/flux_ustarthreshold.py
uv run python examples/pkgs/flux/lowres/flux_ustar_mp_detection.py

# High-resolution analysis
uv run python examples/pkgs/flux/hires/flux_lag.py
uv run python examples/pkgs/flux/hires/flux_windrotation.py
uv run python examples/pkgs/flux/hires/flux_fluxdetectionlimit.py
```

Or run all flux examples:

```bash
uv run python examples/run_all_examples.py
```
