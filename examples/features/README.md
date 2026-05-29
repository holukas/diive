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
import diive as dv

# Vapor pressure deficit (kPa) from air temperature and relative humidity
vpd = dv.variables.calc_vpd_from_ta_rh(df=df, ta_col='TA', rh_col='RH')

# Aerodynamic resistance and dry air density
ra = dv.variables.aerodynamic_resistance(u_ms=df['WS'], ustar_ms=df['USTAR'])
rho_d = dv.variables.dry_air_density(rho_a=rho_a, rho_v=rho_v)

# Potential (clear-sky) shortwave radiation
sw_pot = dv.variables.potrad(timestamp_index=df.index, lat=47.5, lon=8.4, utc_offset=1)
```

Create modeling features:

```python
import diive as dv

# Lagged / shifted variables (lag range -2 to +1, excluding the target)
df_lagged = dv.variables.lagged_variants(df=df, lag=[-2, 1], stepsize=1,
                                         exclude_cols=['NEE'])

# Records since the last time a condition held (e.g. since last rain)
ts = dv.variables.TimeSince(df['PREC'], lower_lim=0, include_lim=False)
ts.calc()
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

See `dv.variables` for the complete API including:

- Air: density, resistance, heat capacity, viscosity
- Radiation: potential radiation, clear-sky models
- Conversions: temperature, energy, water units
- Derived: VPD, relative humidity from dew point
- Temporal: day/night flags, diel cycles
