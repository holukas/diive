"""
============================================
Vapor Pressure Deficit (VPD) Calculations
============================================

Demonstrates calculating VPD from air temperature and relative humidity using
the Magnus formula. VPD is a key driver variable in ecosystem flux analysis,
representing the drying power of the atmosphere.

Best for: Understanding atmospheric water stress and its drivers.
"""

# %%
# Basic VPD Calculation
# ^^^^^^^^^^^^^^^^^^^^^
#
# Calculate VPD from air temperature (TA) and relative humidity (RH).
# Demonstrates direct VPD calculation using the Magnus formula.

import diive as dv

# Load example data
df = dv.load_exampledata_parquet()

# Variables
ta_col = 'Tair_f'  # Gap-filled air temperature
rh_col = 'RH'  # Relative humidity
vpd_col = 'VPD_calculated'

# Subset data
subset_df = df[[ta_col, rh_col]].copy()

# Calculate VPD from TA and RH
subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

# Print statistics
print("VPD Calculation")
print("=" * 50)
print(f"Data points: {len(subset_df)}")
print(f"Missing values (RH): {subset_df[rh_col].isnull().sum()}")
print(f"VPD range: {subset_df[vpd_col].min():.3f} to {subset_df[vpd_col].max():.3f} kPa")
print(f"Mean VPD: {subset_df[vpd_col].mean():.3f} kPa")
print(f"\nFirst 10 rows:")
print(subset_df.head(10))

# %%
# VPD Calculation with Gap Analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Calculate VPD and analyze gaps in the data.
# Shows how RH gaps affect VPD calculations and the extent of missing values.

# Load example data
df = dv.load_exampledata_parquet()

# Variables
ta_col = 'Tair_f'  # Gap-filled air temperature
rh_col = 'RH'  # Relative humidity
vpd_col = 'VPD_hPa'

# Subset data - use 1 year to keep example quick
df_subset = df.loc[(df.index.year == 2018)].copy()
subset_df = df_subset[[ta_col, rh_col]].copy()

# Calculate VPD
subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

print("\nVPD Calculation Analysis (Year 2018)")
print("=" * 50)
print(f"Data points: {len(subset_df)}")
print(f"Original gaps in RH: {subset_df[rh_col].isnull().sum()} records")
print(f"Gaps propagated to VPD: {subset_df[vpd_col].isnull().sum()} records")

print(f"\nVPD Statistics:")
print(subset_df[vpd_col].describe())

print(f"\nTA and RH Statistics:")
print(f"TA: {subset_df[ta_col].min():.1f}°C to {subset_df[ta_col].max():.1f}°C")
print(f"RH: {subset_df[rh_col].min():.1f}% to {subset_df[rh_col].max():.1f}%")

# %%
# VPD Pattern Visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Visualize VPD patterns using heatmap showing daily and hourly breakdown.
# Reveals diurnal and seasonal patterns in vapor pressure deficit.

# Load example data
df = dv.load_exampledata_parquet()

# Variables
ta_col = 'Tair_f'
rh_col = 'RH'
vpd_col = 'VPD_hPa'

# Subset and calculate VPD
subset_df = df[[ta_col, rh_col]].copy()
subset_df[vpd_col] = dv.calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

print("\nVPD Pattern Visualization")
print("=" * 50)
print(f"Data points: {len(subset_df)}")
print(f"VPD statistics:")
print(subset_df[vpd_col].describe())

# Create heatmap visualization
try:
    dv.plot_heatmap_datetime(
        series=subset_df[vpd_col],
        title='VPD - Daily & Hourly Patterns',
        zlabel='VPD (kPa)',
        cb_digits_after_comma=2,
        ax_orientation='horizontal',
        figsize=(14, 6)
    ).show()
except Exception as e:
    print(f"\nVisualization display info: {type(e).__name__}")
