"""
TEMPORAL MATRICES: YEAR × MONTH REPRESENTATIONS
===============================================

Convert time series to matrix format for heatmap visualization and long-term pattern analysis.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^
#
# Use temperature data spanning multiple years.

import pandas as pd
import matplotlib.pyplot as plt
import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Years: {series.index[0].year} to {series.index[-1].year}")
print(f"  Time span: {series.index[-1] - series.index[0]}")

# %%
# Create monthly aggregation matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Resample time series to monthly mean and convert to year × month matrix.

monthly_mean = dv.resample_to_monthly_agg_matrix(
    series=series,
    agg='mean'
)

print("\nMonthly mean matrix (year × month):")
print(f"  Shape: {monthly_mean.shape} (years × months)")
print(f"  Years: {monthly_mean.index.min()} to {monthly_mean.index.max()}")
print(f"\nFirst 5 years:")
print(monthly_mean.head())

# %%
# Matrix statistics
# ^^^^^^^^^^^^^^^^^^
#
# Analyze the temporal matrix to identify patterns.

print("\n" + "="*60)
print("Temporal matrix statistics")
print("="*60)

# Find warmest/coldest months
overall_mean = monthly_mean.values.mean()
overall_std = monthly_mean.values.std()

print(f"Overall mean: {overall_mean:.2f}°C")
print(f"Overall std: {overall_std:.2f}°C")

# By month (seasonal pattern)
print("\nMonthly averages (seasonal cycle):")
monthly_avg = monthly_mean.mean(axis=0)
for month, temp in enumerate(monthly_avg, 1):
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"  {month_names[month-1]}: {temp:.2f}°C")

# By year (long-term trend)
print("\nYearly averages (long-term trend):")
yearly_avg = monthly_mean.mean(axis=1)
for year, temp in yearly_avg.items():
    print(f"  {year}: {temp:.2f}°C")

# %%
# Use case: Identify warmest/coldest periods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Find the warmest and coldest month in the dataset.

flat_matrix = monthly_mean.values.flatten()
max_idx = flat_matrix.argmax()
min_idx = flat_matrix.argmin()

max_year, max_month = divmod(max_idx, 12)
min_year, min_month = divmod(min_idx, 12)

max_year_val = monthly_mean.index[max_year]
min_year_val = monthly_mean.index[min_year]

print("\n" + "="*60)
print("Extremes in the dataset")
print("="*60)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"Warmest month: {max_year_val} {month_names[max_month]} ({monthly_mean.iloc[max_year, max_month]:.2f}°C)")
print(f"Coldest month: {min_year_val} {month_names[min_month]} ({monthly_mean.iloc[min_year, min_month]:.2f}°C)")
print(f"Difference: {monthly_mean.iloc[max_year, max_month] - monthly_mean.iloc[min_year, min_month]:.2f}°C")

# %%
# Matrix with ranking
# ^^^^^^^^^^^^^^^^^^^
#
# Show ranking of each month within its category (hottest, coldest, etc.).

monthly_rank = dv.resample_to_monthly_agg_matrix(
    series=series,
    agg='mean',
    ranks=True
)

print("\n" + "="*60)
print("Monthly ranking (percentile within each month across years)")
print("="*60)

print(f"Ranking matrix shape: {monthly_rank.shape}")
print("\nExample: January rankings (position in Jan distribution):")
jan_ranks = monthly_rank.iloc[:, 0]  # January column
print(jan_ranks)

print("\nInterpretation: Value of 90 means hottest January in the record")
print("Value of 10 means coldest January")

# %%
# Visualize temporal matrix with heatmap
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use heatmap to visualize long-term temperature patterns.

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: Mean temperature heatmap
ax = axes[0]
im = ax.imshow(monthly_mean.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Month', fontsize=11)
ax.set_title('Annual Temperature Cycle (Heatmap)', fontsize=12, fontweight='bold')
ax.set_yticks(range(12))
ax.set_yticklabels(month_names)
ax.set_xticks(range(0, len(monthly_mean), max(1, len(monthly_mean)//10)))
ax.set_xticklabels(monthly_mean.index[::max(1, len(monthly_mean)//10)], rotation=45)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Temperature (°C)', fontsize=10)

# Bottom: Ranking heatmap
ax = axes[1]
im = ax.imshow(monthly_rank.T, aspect='auto', cmap='YlGnBu', interpolation='nearest')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Month', fontsize=11)
ax.set_title('Temperature Ranking by Month (100=hottest, 0=coldest)', fontsize=12, fontweight='bold')
ax.set_yticks(range(12))
ax.set_yticklabels(month_names)
ax.set_xticks(range(0, len(monthly_rank), max(1, len(monthly_rank)//10)))
ax.set_xticklabels(monthly_rank.index[::max(1, len(monthly_rank)//10)], rotation=45)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Percentile', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Use case: Detect trends across years
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Analyze whether specific months are getting warmer/colder over time.

print("\n" + "="*60)
print("Long-term temperature trends by month")
print("="*60)

for month in [0, 3, 6, 9]:  # Jan, Apr, Jul, Oct
    month_name = month_names[month]
    month_temps = monthly_mean.iloc[:, month]

    # Simple linear trend (rise per decade)
    years_elapsed = (month_temps.index - month_temps.index[0]).astype('timedelta64[D]') / 365.25
    trend = (month_temps.iloc[-1] - month_temps.iloc[0]) / (years_elapsed[-1] - years_elapsed[0]) * 10

    print(f"{month_name:>3}: {month_temps.mean():.2f}°C avg, trend: {trend:+.3f}°C/decade")

# %%
# Alternative aggregations
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create matrices with different aggregation methods.

monthly_sum = dv.resample_to_monthly_agg_matrix(series=series, agg='sum')
monthly_max = dv.resample_to_monthly_agg_matrix(series=series, agg='max')
monthly_min = dv.resample_to_monthly_agg_matrix(series=series, agg='min')

print("\n" + "="*60)
print("Alternative aggregations available")
print("="*60)

aggregations = {
    'mean': monthly_mean,
    'sum': monthly_sum,
    'max': monthly_max,
    'min': monthly_min,
}

print("Aggregation methods available:")
for method, matrix in aggregations.items():
    sample_value = matrix.iloc[0, 0]
    print(f"  {method:>6}: {sample_value:.2f} (Jan, year 1)")

print("\nNote: sum/max/min are useful for other variables:")
print("  - Precipitation: sum gives total rainfall")
print("  - Radiation: max/min shows daily extremes")
print("  - Flux: sum gives cumulative gas exchange")
