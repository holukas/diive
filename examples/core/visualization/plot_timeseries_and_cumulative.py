"""
============================
Cumulative Sum Visualization
============================

Cumulative time series plots showing running totals and annual summaries.
Useful for flux budget analysis and trend detection over time.

Best for: Analyzing cumulative fluxes, comparing annual totals, budget analysis
"""

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df_orig = dv.load_exampledata_parquet()

# Convert units: umol CO2 m-2 s-1 --> g C m-2 30min-1
conversion_factor = 0.02161926
series_units = r'($\mathrm{gC\ m^{-2}}$)'

print(f"Loaded {len(df_orig)} records")
print(f"Unit conversion factor: {conversion_factor}")

# %%
# Cumulative flux across all years
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Show cumulative totals for multiple USTAR scenarios.

df_cum = df_orig[['NEE_CUT_16_f', 'NEE_CUT_REF_f', 'NEE_CUT_84_f']].copy()
df_cum = df_cum.multiply(conversion_factor)

dv.plot_cumulative(
    df=df_cum,
    units=series_units,
    start_year=2015,
    end_year=2019
).plot()

print("Plotted cumulative flux for USTAR scenarios")

# %%
# Yearly cumulative with reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Annual totals with mean and standard deviation band from reference period.

series = df_orig['NEE_CUT_REF_f'].copy()
series = series.multiply(conversion_factor)

dv.plot_cumulative_year(
    series=series,
    series_units=series_units,
    yearly_end_date=None,
    start_year=2015,
    end_year=2019,
    show_reference=True,
    excl_years_from_reference=None,
    highlight_year_color='#F44336'
).plot()

print("Plotted yearly cumulative sums with reference band")

# %%
# Yearly cumulative with highlighted year
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Emphasize a specific year for comparison.

dv.plot_cumulative_year(
    series=series,
    series_units=series_units,
    yearly_end_date=None,
    start_year=2015,
    end_year=2019,
    show_reference=True,
    excl_years_from_reference=None,
    highlight_year=2017,
    highlight_year_color='#F44336'
).plot()

print("Plotted yearly cumulative sums with 2017 highlighted")
