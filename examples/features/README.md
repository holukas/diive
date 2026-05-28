# Feature Engineering & Variable Creation Examples

Examples demonstrating creation and engineering of variables for time series analysis and modeling.

11 examples across ML feature engineering, sensor corrections, energy conversions, air properties, derived variables, and temporal encoding.

## Examples by Category

### ML Feature Engineering

- **feature_engineer.py** — 8-stage composable pipeline: lag, rolling stats, differencing, EMA, polynomial, temporal encoding, STL, record numbering

### Sensor Corrections

- **feature_sonic_temp_conversion.py** — Air temperature from sonic temperature and water vapor

### Energy Conversions

- **feature_latent_heat.py** — Latent heat of vaporization from air temperature
- **feature_evapotranspiration.py** — Evapotranspiration from latent heat flux

### Physical Properties

- **feature_air.py** — Air density, resistance, heat capacity, viscosity
- **feature_vpd.py** — Vapor Pressure Deficit from temperature and humidity

### Radiation & Solar Geometry

- **feature_potentialradiation.py** — Clear-sky radiation calculation
- **feature_daynightflag.py** — Daytime/nighttime classification

### Time-Based Features

- **feature_laggedvariants.py** — Lagged and shifted variables for modeling
- **feature_timesince.py** — Time-since-event features (days, hours)

### Testing & Synthetic Data

- **feature_noise.py** — Generate synthetic noise for quality testing

## Quick Start

Calculate environmental properties:

```python
from diive.features import feature_vpd, feature_air_density, feature_potentialradiation

vpd = feature_vpd(T_celsius=df['TA'], RH_percent=df['RH'])
rho = feature_air_density(T_celsius=df['TA'], p_kpa=df['PA'])
sw_pot = feature_potentialradiation(df.index, lat=47.5, lon=8.4)
```

Create modeling features:

```python
from diive.features import feature_laggedvariants, feature_timesince

df_lagged = feature_laggedvariants(df, 'NEE', lags=[-2, -1, 1, 2])
df['Days_Since_Fire'] = feature_timesince(fire_date)
```

## Running Examples

```bash
# ML feature engineering pipeline
uv run python examples/features/feature_engineer.py

# Sensor corrections
uv run python examples/features/feature_sonic_temp_conversion.py

# Physical properties
uv run python examples/features/feature_air.py
uv run python examples/features/feature_vpd.py

# Solar geometry & radiation
uv run python examples/features/feature_potentialradiation.py
uv run python examples/features/feature_daynightflag.py

# Energy conversions
uv run python examples/features/feature_latent_heat.py
uv run python examples/features/feature_evapotranspiration.py

# Time series transformations
uv run python examples/features/feature_laggedvariants.py
uv run python examples/features/feature_timesince.py

# Synthetic data
uv run python examples/features/feature_noise.py

# All examples
uv run python examples/run_all_examples.py
```

## Available Functions

See `diive.pkgs.features` for the complete API including:

- Air: density, resistance, heat capacity, viscosity
- Radiation: potential radiation, clear-sky models
- Conversions: temperature, energy, water units
- Derived: VPD, relative humidity from dew point
- Temporal: day/night flags, diel cycles
