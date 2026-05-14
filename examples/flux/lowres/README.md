# Low-Resolution Flux Processing Examples

Low-resolution (30-minute) eddy-covariance flux processing at the Swiss FluxNet site (CH-LAE).
Covers time lag detection, quality filtering, self-heating correction, uncertainty estimation, and turbulence thresholds.

## Examples

### Time Lag Detection
- **flux_timelag_analysis.py** — Time lag detection and visualization for gas concentrations using covariance analysis

### Base Flux Detection & Quality
- **flux_common.py** — Detect flux variable nomenclature and base gas measurements
- **flux_hqflux.py** — Extract highest-quality flux records using Hampel outlier filtering

### Self-Heating Correction (SCOP)
- **flux_selfheating.py** — Quick demonstration of SCOP workflow (5 USTAR classes, 5 bootstrap runs)
- **flux_selfheating_production.py** — Complete production workflow: create scaling factors table from parallel measurements (20 classes, 100 bootstrap runs) and apply to long-term data

### Measurement Uncertainty & Turbulence
- **flux_uncertainty.py** — Random uncertainty estimation using the Pilegaard et al. (2020) method
- **flux_ustar_mp_detection.py** — Moving Point (MP) friction velocity detection for nighttime turbulence thresholds

## Running Examples

```bash
# Run one example
uv run python examples/flux/lowres/flux_timelag_analysis.py

# Run all low-res flux examples
uv run python examples/flux/lowres/flux_selfheating.py
uv run python examples/flux/lowres/flux_selfheating_production.py
uv run python examples/flux/lowres/flux_uncertainty.py
```

## Key Concepts

**Self-heating correction (SCOP method):**
- Sun-induced heating of open-path IRGA sensors creates spurious negative CO2 flux
- Correction uses physics-based unscaled term (FCT_UNSC) scaled by factors from parallel measurements
- Scaling factors binned by USTAR and time-of-day, reusable across seasons

**Quality filtering:**
- Hampel filter for identifying robust flux estimates
- USTAR-based selection for stable atmospheric conditions

**Uncertainty:** Random measurement uncertainty following Pilegaard et al. (2020) with PAS20 method

## Related Documentation

- [Flux Processing Chain](../README.md) — Multi-level L2-L4.1 workflow
- [High-Resolution Analysis](../hires/README.md) — 10 Hz eddy covariance processing
- Source: `diive/pkgs/flux/lowres/`
