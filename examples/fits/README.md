# Data Fitting Examples

Examples demonstrating curve fitting and regression for time series analysis and modeling.

**2 examples covering binned polynomial fitting and ecosystem driver-response analysis.**

## Contents

### Binned Fitting (BinFitterCP)
- **fit_binfittercp.py** — Binned curve fitting with confidence/prediction intervals, result interpretation
- **fit_fitter.py** — NEE response to vapor pressure deficit, uncertainty quantification

## Use Cases

**Fit a polynomial to a driver-response relationship:**
```python
import diive as dv

# Fit a quadratic to binned data (e.g., CO2 uptake vs. light)
bf = dv.analysis.BinFitterCP(
    df=df,
    xcol='PAR',                   # X variable (predictor)
    ycol='GPP',                   # Y variable (response)
    n_bins_x=10,                  # Divide X into 10 equal-width bins
    bins_y_agg='mean',            # Aggregate Y per bin ('mean' or 'median')
    fit_type='quadratic_offset'   # y = ax^2 + bx + c
)
bf.run()

results = bf.get_results()
print(results['fit_equation_str'])  # Fitted equation
print(results['fit_params_opt'])    # Optimal parameters
print(results['fit_r2'])            # R^2
```

**Choose the fit type:**
```python
import diive as dv

# Four fit types are available:
#   'linear'            y = ax + b
#   'quadratic_offset'  y = ax^2 + bx + c  (default)
#   'quadratic'         y = ax^2 + bx
#   'cubic'             y = ax^3 + bx^2 + cx + d
bf = dv.analysis.BinFitterCP(df=df, xcol='Tair_f', ycol='VPD_f', fit_type='cubic')
bf.run()
```

**Inspect the fitted curve and its uncertainty:**
```python
import diive as dv

bf = dv.analysis.BinFitterCP(df=df, xcol='Tair_f', ycol='VPD_f', n_predictions=1000)
bf.run()

# fit_df holds the smooth curve plus the uncertainty bands:
#   nom_lower_ci95 / nom_upper_ci95     95% confidence interval (mean fit)
#   lower_predband / upper_predband     95% prediction interval (individual values)
fit_df = bf.get_results()['fit_df']

bf.showplot(show_unbinned_data=True, show_prediction_interval=True)
```

## Related Documentation

See `dv.analysis` (e.g. `BinFitterCP`) for available fitting classes:
- Polynomial fitting of binned data (linear, quadratic, cubic)
- Binning and per-bin aggregation of the response variable
- Confidence and prediction intervals
- Goodness-of-fit statistics (R²)

## Running Examples

```bash
# Binned curve fitting with temperature-VPD relationship
uv run python examples/fits/fit_binfittercp.py

# Ecosystem driver-response (NEE-VPD) fitting
uv run python examples/fits/fit_fitter.py

# Run all fitting examples
uv run python examples/run_all_examples.py
```

## Common Applications

- **Light-response curves** — CO2 uptake or fluorescence vs. PAR
- **Temperature dependencies** — Respiration vs. temperature
- **Calibration curves** — Sensor response linearity
- **Ecosystem response functions** — Evapotranspiration vs. VPD or radiation
