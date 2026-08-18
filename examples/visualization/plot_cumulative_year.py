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
# A shared FormatStyle for the chrome
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Build the plot chrome (font sizes, grid) once as a ``FormatStyle`` and reuse
# it across both cumulative plots below, so they share one look. The y-axis
# units are folded in from the ``series_units`` constructor argument, so they
# are not repeated here. (CumulativeYear owns its own dynamic per-year title,
# so a ``title`` set on the style does not apply here.) Data-rendering choices
# like ``highlight_year_color`` stay direct on ``plot()``.

style = dv.plotting.FormatStyle(
    axlabel_fontsize=12,
    ticks_fontsize=10,
    show_grid=False,
)

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
).plot(format_style=style, highlight_year_color='#F44336')

print("\nPlotted yearly cumulative sums with reference band (2015-2017 baseline)")

# %%
# Highlight specific year for comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Emphasize a single year for detailed comparison against reference period.
# The same ``style`` is reused, so both figures share identical chrome.

dv.plotting.CumulativeYear(
    series=series,
    series_units=series_units,
    yearly_end_date=None,
    start_year=2015,
    end_year=2019,
    show_reference=True,
    excl_years_from_reference=None,
    highlight_year=2017,
).plot(format_style=style, highlight_year_color='#F44336')

print("Plotted yearly cumulative sums with year 2017 highlighted")
print("Highlighting enables easy identification of outlier years and trends")
