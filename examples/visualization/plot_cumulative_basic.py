"""
============================
Cumulative Flux Visualization
============================

Cumulative time series plots showing running totals across all time periods.
Useful for tracking cumulative fluxes and total mass balance.

Best for: Cumulative flux budgets, total mass balance analysis
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
# Plot cumulative totals for multiple USTAR filtering scenarios.
# Shows how cumulative flux diverges between conservative and generous thresholds.
#
# Chrome (title and font sizes) is described once with a ``FormatStyle`` and
# passed via ``format_style=``. The data units stay on the ``Cumulative``
# constructor (they belong to the series, not to the shared style), and the
# colorbar-free line styling stays on the constructor/plot call.

df_cum = df_orig[['NEE_CUT_16_f', 'NEE_CUT_REF_f', 'NEE_CUT_84_f']].copy()
df_cum = df_cum.multiply(conversion_factor)

style = dv.plotting.FormatStyle(
    title='Cumulative NEE by USTAR scenario (2015-2019)',
    axlabel_fontsize=12,
)

dv.plotting.Cumulative(
    df=df_cum,
    units=series_units,
    start_year=2015,
    end_year=2019
).plot(format_style=style)

print("\nPlotted cumulative flux comparing USTAR filtering scenarios")
print("USTAR scenarios: CUT_16 (conservative), CUT_REF (reference), CUT_84 (generous)")
