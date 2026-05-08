# High-Resolution Eddy Covariance Analysis Examples

Examples demonstrating high-resolution time lag detection, wind rotation correction, and flux detection limit analysis.

## Contents

- **lag.py** — Time lag detection using maximum covariance (MaxCovariance)
- **windrotation.py** — Coordinate rotation correction for wind measurements
- **fluxdetectionlimit.py** — Flux detection limit and signal-to-noise analysis

## Related Documentation

See `diive.pkgs.flux.hires` for:
- `MaxCovariance` — Lag detection class
- `WindRotation2D` — Wind coordinate rotation
- `FluxDetectionLimit` — Detection limit analysis

## Usage

```bash
uv run python examples/pkgs/flux/hires/lag.py
uv run python examples/pkgs/flux/hires/windrotation.py
```
