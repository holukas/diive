"""
TEMPORAL MATRICES: YEAR × MONTH REPRESENTATIONS
===============================================

Convert time series to year × month matrix for pattern analysis and visualization.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()
series = df['Tair_f'].copy()

print("Example data loaded:")
print(f"  Records: {len(series)}")
print(f"  Years: {series.index[0].year} to {series.index[-1].year}")

# %%
# Create monthly aggregation matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

monthly_mean = dv.times.resample_to_monthly_agg_matrix(series=series, agg='mean')

print("\nMonthly mean matrix:")
print(f"  Shape: {monthly_mean.shape} (years × months)")
print(f"  Years: {monthly_mean.index.min()} to {monthly_mean.index.max()}")
print(f"\nFirst 5 years:")
print(monthly_mean.head())

# %%
# Analyze patterns
# ^^^^^^^^^^^^^^^^

print("\n" + "=" * 60)
print("Seasonal and yearly patterns")
print("=" * 60)

# Seasonal cycle
monthly_avg = monthly_mean.mean(axis=0)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\nMonthly averages:")
for month, temp in enumerate(monthly_avg, 1):
    print(f"  {month_names[month - 1]}: {temp:.2f}°C")

# Extremes
flat_matrix = monthly_mean.values.flatten()
max_idx = flat_matrix.argmax()
min_idx = flat_matrix.argmin()
max_year, max_month = divmod(max_idx, 12)
min_year, min_month = divmod(min_idx, 12)

print(f"\nWarmest: {monthly_mean.index[max_year]} {month_names[max_month]} ({monthly_mean.iloc[max_year, max_month]:.2f}°C)")
print(f"Coldest: {monthly_mean.index[min_year]} {month_names[min_month]} ({monthly_mean.iloc[min_year, min_month]:.2f}°C)")

# %%
# Matrix with ranking
# ^^^^^^^^^^^^^^^^^^^
#
# Percentile ranking within each month (100=warmest, 0=coldest).

monthly_rank = dv.times.resample_to_monthly_agg_matrix(series=series, agg='mean', ranks=True)

print("\n" + "=" * 60)
print("Monthly ranking")
print("=" * 60)
print(f"\nJanuary rankings (example):")
print(monthly_rank.iloc[:, 0])

# %%
# Alternative aggregations
# ^^^^^^^^^^^^^^^^^^^^^^^^

print("\n" + "=" * 60)
print("Other aggregation methods")
print("=" * 60)

monthly_sum = dv.times.resample_to_monthly_agg_matrix(series=series, agg='sum')
monthly_max = dv.times.resample_to_monthly_agg_matrix(series=series, agg='max')
monthly_min = dv.times.resample_to_monthly_agg_matrix(series=series, agg='min')

methods = {
    'mean': monthly_mean,
    'sum': monthly_sum,
    'max': monthly_max,
    'min': monthly_min,
}

print("Available aggregations:")
for method, matrix in methods.items():
    val = matrix.iloc[0, 0]
    print(f"  {method:>6}: {val:7.2f} (Jan, year 1)")
