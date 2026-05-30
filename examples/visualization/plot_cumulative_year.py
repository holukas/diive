"""
=============================
Yearly Cumulative Summaries
=============================

Annual cumulative sum visualization with reference band and highlighted years.
Compare individual years against climatological mean and variability.

Best for: Annual budget analysis, comparing specific years to baseline, trend detection
"""

import diive as dv

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^

df_orig = dv.load_exampledata_parquet()

# Convert units: umol CO2 m-2 s-1 --> g C m-2 30min-1
conversion_factor = 0.02161926
series_units = r'($\mathrm{gC\ m^{-2}}$)'

series = df_orig['NEE_CUT_REF_f'].copy()
series = series.multiply(conversion_factor)

print(f"Loaded {len(df_orig)} records")
print(f"Unit conversion factor: {conversion_factor}")

# %%
# Yearly cumulative with reference band
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Plot annual cumulative sums with mean and standard deviation from reference period.
# Gray band shows climatological range, helping identify anomalous years.

dv.plotting.CumulativeYear(
    series=series,
    series_units=series_units,
    yearly_end_date=None,
    start_year=2015,
    end_year=2019,
    show_reference=True,
    excl_years_from_reference=None,
).plot(highlight_year_color='#F44336')

print("\nPlotted yearly cumulative sums with reference band (2015-2017 baseline)")

# %%
# Highlight specific year for comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Emphasize a single year for detailed comparison against reference period.

dv.plotting.CumulativeYear(
    series=series,
    series_units=series_units,
    yearly_end_date=None,
    start_year=2015,
    end_year=2019,
    show_reference=True,
    excl_years_from_reference=None,
    highlight_year=2017,
).plot(highlight_year_color='#F44336')

print("Plotted yearly cumulative sums with year 2017 highlighted")
print("Highlighting enables easy identification of outlier years and trends")
