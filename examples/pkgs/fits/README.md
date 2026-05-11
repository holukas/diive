# Data Fitting Examples

Examples demonstrating curve fitting and regression for time series analysis and modeling.

**1 example covering polynomial fitting and custom function fitting.**

## Contents

- **fit_fitter.py** — Polynomial fitting, light-response curves, custom function fitting

## Use Cases

**Fit polynomial to light-response relationship:**
```python
from diive.pkgs.fits import Fitter

# Fit polynomial (e.g., CO2 uptake vs. light)
fitter = Fitter(
    x=df['PAR'],
    y=df['GPP'],
    fit_type='polynomial',
    degree=2  # Quadratic fit
)
coeffs = fitter.get_coefficients()
curve = fitter.get_fitted_curve()
```

**Fit rectangular hyperbola (light saturation curve):**
```python
from diive.pkgs.fits import Fitter

# Hyperbolic function: y = (a*x) / (b + x)
fitter = Fitter(
    x=df['PAR'],
    y=df['GPP'],
    fit_type='hyperbolic'
)
```

**Fit temperature response curves:**
```python
from diive.pkgs.fits import Fitter

# Lloyd-Taylor or exponential temperature response
fitter = Fitter(
    x=df['Temperature'],
    y=df['Respiration'],
    fit_type='exponential'
)
# y = a * exp(b * x)
```

## Related Documentation

See `diive.pkgs.fits` for available fitting classes:
- Polynomial fitting (linear, quadratic, cubic, etc.)
- Custom function fitting
- Parameter uncertainty estimation
- Goodness-of-fit statistics

## Running Examples

```bash
# Curve fitting demonstration
uv run python examples/pkgs/fits/fit_fitter.py

# Run all fitting examples
uv run python examples/run_all_examples.py
```

## Common Applications

- **Light-response curves** — CO2 uptake or fluorescence vs. PAR
- **Temperature dependencies** — Respiration vs. temperature
- **Saturation curves** — Enzyme kinetics, gas exchange responses
- **Calibration curves** — Sensor response linearity
- **Ecosystem response functions** — Evapotranspiration vs. VPD or radiation
