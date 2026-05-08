# Eddy Covariance Flux Processing Examples

Examples demonstrating flux processing, quality control, and high-resolution analysis for eddy covariance data.

## Contents

- **common.py** — Common utilities and helper functions for flux processing
- **hqflux/** — Highest-quality flux filtering and analysis
- **hires/** — High-resolution flux analysis (MaxCovariance, WindRotation, FluxDetectionLimit)
- **selfheating/** — Oxygen sensor self-heating correction
- **uncertainty/** — Random uncertainty estimation (PAS20 method)
- **ustarthreshold/** — USTAR threshold detection and filtering
- **ustar_mp_detection/** — Moving Point (MP) USTAR detection method

## Related Documentation

See `diive.pkgs.flux` for available processing classes and functions:
- Flux quality control and filtering
- High-resolution analysis methods
- Sensor corrections
- USTAR filtering for low-turbulence periods

## Running Examples

```bash
uv run python examples/pkgs/flux/common.py
uv run python examples/pkgs/flux/hqflux/hqflux.py
uv run python examples/pkgs/flux/hires/lag.py
```

Or run all flux examples:

```bash
uv run python examples/run_all_examples.py
```
