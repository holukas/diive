"""
======================
Percentile/Quantile Analysis
======================

Calculate and visualize percentiles 0-100 for time series data.

Demonstrates percentile analysis across the full 0-100 range for air
temperature data, showing the distribution from minimum to maximum values
with detailed statistics.

Best for: Understanding data distribution quantiles and percentile values
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()

# %%
# Calculate percentiles 0-100
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Calculate percentiles 0-100 for air temperature
percentiles_df = dv.percentiles101(series=df['Tair_f'], showplot=True, verbose=True)

# %%
# Display full results
# ^^^^^^^^^^^^^^^^^^^

print("\nAll percentiles (0-100):")
print(percentiles_df)

# %%
# Summary statistics
# ^^^^^^^^^^^^^^^^^^

print(f"\nSummary of Tair_f:")
print(f"  0th percentile (min):   {percentiles_df.iloc[0]['VALUE']:.2f} C")
print(f"  25th percentile:        {percentiles_df[percentiles_df['PERCENTILE'] == 25]['VALUE'].values[0]:.2f} C")
print(f"  50th percentile (median): {percentiles_df[percentiles_df['PERCENTILE'] == 50]['VALUE'].values[0]:.2f} C")
print(f"  75th percentile:        {percentiles_df[percentiles_df['PERCENTILE'] == 75]['VALUE'].values[0]:.2f} C")
print(f"  100th percentile (max): {percentiles_df.iloc[-1]['VALUE']:.2f} C")
