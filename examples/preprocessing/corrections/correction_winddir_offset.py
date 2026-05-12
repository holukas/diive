"""
==================================
Correct Wind Direction Offset
==================================

Detect and correct wind direction measurement offset by comparing yearly
wind direction histograms to a reference histogram. Finds the offset that
maximizes correlation with reference distribution.
"""

# %%
# Load multi-year wind direction data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Wind direction measurements can drift over time due to sensor
# misalignment or calibration issues. We load multi-year wind direction
# data where some years are known to be correct (reference years).

import diive as dv
from diive.configs.exampledata import load_exampledata_winddir

df = load_exampledata_winddir()
winddir = df['wind_dir'].copy()

# Filter to years with data
winddir = winddir.loc[winddir.index.year <= 2022].dropna()

print("Input wind direction data:")
print(f"  Range: {winddir.min():.1f} to {winddir.max():.1f} degrees")
print(f"  Years available: {sorted(winddir.index.year.unique())}")

# %%
# Detect offset using reference years
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Years 2021 and 2022 are known to be correctly calibrated. We build
# a reference wind direction histogram from these years, then search
# for the yearly offset in other years that maximizes correlation.

offset_corrector = dv.WindDirOffset(
    winddir=winddir,
    hist_ref_years=[2021, 2022],
    offset_start=-50,
    offset_end=50,
    hist_n_bins=360
)

print("\nOffset detection:")
print(f"  Reference years: [2021, 2022]")
print(f"  Search range: -50 to +50 degrees")
print(f"  Histogram bins: 360 (1 degree resolution)")

# %%
# Extract yearly offsets and corrected values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Get the detected offset for each year and apply correction.

winddir_corrected = offset_corrector.get_corrected_wind_directions()
yearly_offsets = offset_corrector.get_yearly_offsets()

print("\nYearly wind direction offsets (degrees):")
print(yearly_offsets.to_string(index=False))

# %%
# Verify correction
# ^^^^^^^^^^^^^^^^^
#
# Check the corrected wind direction range and compare to original.

print(f"\nCorrected wind direction data:")
print(f"  Range: {winddir_corrected.min():.1f} to {winddir_corrected.max():.1f} degrees")
print(f"  Original range: {winddir.min():.1f} to {winddir.max():.1f} degrees")

# Show data availability by year
print(f"\nData availability by year:")
for year in sorted(winddir.index.year.unique()):
    n_records = len(winddir[winddir.index.year == year])
    offset = yearly_offsets[yearly_offsets['YEAR'] == year]['OFFSET'].values[0]
    print(f"  {year}: {n_records:5d} records, offset: {offset:+6.1f}°")
