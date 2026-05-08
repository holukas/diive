# Feature Engineering & Variable Creation Examples

Examples demonstrating creation and engineering of variables for time series analysis and gap-filling.

## Contents

- **air.py** — Air properties (temperature, humidity, density)
- **conversions.py** — Unit conversions and coordinate transformations
- **daynightflag.py** — Daytime/nighttime classification based on solar geometry
- **laggedvariants.py** — Lagged and shifted variable creation
- **noise.py** — Synthetic noise generation and manipulation
- **potentialradiation.py** — Potential (clear-sky) radiation calculation
- **timesince.py** — Time-since-event features
- **vpd.py** — Vapor Pressure Deficit (VPD) calculation

## Related Documentation

See `diive.pkgs.features` for available functions and classes.

## Running Examples

```bash
uv run python examples/pkgs/features/vpd.py
uv run python examples/pkgs/features/daynightflag.py
```

Or run all feature examples:

```bash
uv run python examples/run_all_examples.py
```
