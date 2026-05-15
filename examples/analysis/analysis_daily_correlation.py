"""
=============================
Daily Correlation Analysis
=============================

Calculate daily correlation coefficients between pairs of time series.
This demonstrates how to identify relationships and detect anomalies
on a per-day basis.

Daily correlation is useful for:
- Quality assurance (checking sensor performance or timestamp shifts)
- Identifying days with unusual relationships
- Comparing observed vs. modeled variables
- Understanding temporal variability in ecosystem processes

Best for: Understanding daily-scale relationships between variables,
detecting measurement quality issues, and identifying anomalous days.
"""

# %%
# Overview: Why Daily Correlation?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Calculating correlation on a per-day basis reveals temporal patterns
# that would be hidden by single overall correlation values.
#
# **Use cases:**
# 1. Quality check: Observed radiation should be highly correlated with
#    potential radiation (physics-based). Low daily correlations suggest
#    sensor issues or timestamp problems.
# 2. Physical relationships: Temperature often correlates with radiation
#    on daily timescales.
# 3. Ecosystem responses: Biological variables (e.g., photosynthesis)
#    show variable relationships with drivers across different days.

# %%
# Load Data and Prepare Variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import diive as dv
from diive.pkgs.features.variables.potentialradiation import potrad

# Load example data (use full year 2022)
df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2022].copy()

# Extract key variables
rg_series = df['Rg_f'].copy()         # Observed global radiation
ta_series = df['Tair_f'].copy()       # Air temperature
nee_series = df['NEE_CUT_REF_f'].copy()  # Net ecosystem exchange (CO2 flux)

print(f"Data Summary:")
print(f"  Period: {df.index.min().date()} to {df.index.max().date()}")
print(f"  Records: {len(df)}")
print(f"\nVariables:")
print(f"  Rg_f (observed radiation): {rg_series.min():.1f} to {rg_series.max():.1f} W/m2")
print(f"  Tair_f (air temperature): {ta_series.min():.1f} to {ta_series.max():.1f} C")
print(f"  NEE_CUT_REF_f (CO2 flux): {nee_series.min():.2f} to {nee_series.max():.2f} umol m-2 s-1")

# Calculate potential solar radiation (clear-sky reference)
sw_pot = potrad(
    timestamp_index=rg_series.index,
    lat=47.286417,   # Site latitude
    lon=7.733750,    # Site longitude
    utc_offset=1     # UTC+1
)

print(f"\nCalculated potential radiation (SW_IN_POT): {sw_pot.min():.1f} to {sw_pot.max():.1f} W/m²")

# %%
# Example 1: Quality Check - Observed vs Potential Radiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Observed radiation should be highly correlated with potential radiation
# (clear-sky model). Low daily correlations indicate sensor issues, clouds,
# or timestamp problems.

print(f"\n{'='*60}")
print(f"EXAMPLE 1: Observed Radiation vs Potential Radiation")
print(f"{'='*60}")
print(f"\nExpectation: High correlations (>0.8) indicate good measurement quality")
print(f"and clear-sky conditions. Low correlations suggest clouds or issues.")

corr1 = dv.daily_correlation(
    s1=rg_series,
    s2=sw_pot,
    mincorr=0.8
)

print(f"\nDaily correlations (first 10 days):")
print(corr1.correlations.head(10))

print(f"\nSummary statistics:")
print(f"  Count: {len(corr1.correlations)}")
print(f"  Mean: {corr1.correlations.mean():.4f}")
print(f"  Median: {corr1.correlations.median():.4f}")
print(f"  Min: {corr1.correlations.min():.4f}")
print(f"  Max: {corr1.correlations.max():.4f}")
print(f"  Std: {corr1.correlations.std():.4f}")

print(f"\nDays with HIGHEST correlations (best match):")
highest = corr1.correlations.nlargest(5)
for date, corr_val in highest.items():
    print(f"  {date.date()}: {corr_val:.4f}")

print(f"\nDays with LOWEST correlations (worst match - likely cloudy):")
lowest = corr1.correlations.nsmallest(5)
for date, corr_val in lowest.items():
    print(f"  {date.date()}: {corr_val:.4f}")

# %%
# Example 2: Physical Relationship - Temperature vs Radiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Air temperature typically correlates positively with solar radiation
# on clear days, but relationships vary with season, cloud cover, and
# advection of air masses.

print(f"\n{'='*60}")
print(f"EXAMPLE 2: Air Temperature vs Observed Radiation")
print(f"{'='*60}")
print(f"\nExpectation: Positive correlations on clear days, variable on cloudy days.")
print(f"Low/negative correlations indicate advection or unusual conditions.")

corr2 = dv.daily_correlation(
    s1=ta_series,
    s2=rg_series,
    mincorr=0.5
)

print(f"\nDaily correlations (first 10 days):")
print(corr2.correlations.head(10))

print(f"\nSummary statistics:")
print(f"  Count: {len(corr2.correlations)}")
print(f"  Mean: {corr2.correlations.mean():.4f}")
print(f"  Median: {corr2.correlations.median():.4f}")
print(f"  Min: {corr2.correlations.min():.4f}")
print(f"  Max: {corr2.correlations.max():.4f}")
print(f"  Std: {corr2.correlations.std():.4f}")

print(f"\nDays with strong positive correlation (T increases with radiation):")
positive = corr2.correlations.nlargest(5)
for date, corr_val in positive.items():
    print(f"  {date.date()}: {corr_val:.4f}")

print(f"\nDays with weak/negative correlation (unusual patterns):")
negative = corr2.correlations.nsmallest(5)
for date, corr_val in negative.items():
    print(f"  {date.date()}: {corr_val:.4f}")

# %%
# Example 3: Ecosystem Driver-Response - Temperature vs NEE
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Net ecosystem exchange (NEE) of CO2 represents photosynthesis and respiration.
# The relationship with temperature varies with season: weak in summer
# (light-limited), stronger in spring/fall (temperature-controlled respiration).

print(f"\n{'='*60}")
print(f"EXAMPLE 3: Air Temperature vs NEE (CO2 Flux)")
print(f"{'='*60}")
print(f"\nExpectation: Variable correlations across seasons. Winter/spring")
print(f"often show positive correlations (warmer = more respiration).")
print(f"Summer shows weak/negative correlations (light-limited photosynthesis).")

corr3 = dv.daily_correlation(
    s1=ta_series,
    s2=nee_series
)

print(f"\nDaily correlations (first 10 days):")
print(corr3.correlations.head(10))

print(f"\nSummary statistics:")
print(f"  Count: {len(corr3.correlations)}")
print(f"  Mean: {corr3.correlations.mean():.4f}")
print(f"  Median: {corr3.correlations.median():.4f}")
print(f"  Min: {corr3.correlations.min():.4f}")
print(f"  Max: {corr3.correlations.max():.4f}")
print(f"  Std: {corr3.correlations.std():.4f}")

print(f"\nDays with strong positive correlation (warmth increases CO2 release):")
positive_nee = corr3.correlations.nlargest(5)
for date, corr_val in positive_nee.items():
    print(f"  {date.date()}: {corr_val:.4f}")

print(f"\nDays with negative correlation (warmth increases CO2 uptake):")
negative_nee = corr3.correlations.nsmallest(5)
for date, corr_val in negative_nee.items():
    print(f"  {date.date()}: {corr_val:.4f}")

# %%
# Comparing All Three Relationships
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Summary of what daily correlation reveals across different types
# of variable relationships.

print(f"\n{'='*60}")
print(f"SUMMARY: Correlation Patterns Across Relationships")
print(f"{'='*60}")

print(f"\n1. Observed vs Potential Radiation (physical law):")
print(f"   Mean correlation: {corr1.correlations.mean():.4f}")
print(f"   Range: {corr1.correlations.min():.4f} to {corr1.correlations.max():.4f}")
print(f"   (High & consistent - expected)")

print(f"\n2. Temperature vs Radiation (physical relationship):")
print(f"   Mean correlation: {corr2.correlations.mean():.4f}")
print(f"   Range: {corr2.correlations.min():.4f} to {corr2.correlations.max():.4f}")
print(f"   (Variable - influenced by season, cloud cover, advection)")

print(f"\n3. Temperature vs NEE (biological process):")
print(f"   Mean correlation: {corr3.correlations.mean():.4f}")
print(f"   Range: {corr3.correlations.min():.4f} to {corr3.correlations.max():.4f}")
print(f"   (Highly variable - controlled by multiple factors)")

print(f"\n{'='*60}")
print(f"Key Insights:")
print(f"{'='*60}")
print(f"• Physical relationships (1) are stable and high")
print(f"• Environmental relationships (2) vary with conditions")
print(f"• Biological relationships (3) are complex and variable")
print(f"• Daily correlation reveals patterns masked by overall statistics")
print(f"• Anomalous days can indicate measurement issues or unusual conditions")

# %%
# Advanced Class Methods
# ^^^^^^^^^^^^^^^^^^^^^^
# The DailyCorrelation class provides additional analysis methods beyond
# simple statistics. Demonstrate comprehensive summary and sorting.

print(f"\n{'='*60}")
print(f"ADVANCED: Using Class Methods for Deep Analysis")
print(f"{'='*60}")

# Use Example 1 (Observed vs Potential Radiation) for demonstration
print(f"\nExample 1: Comprehensive Summary Statistics")
print(f"-" * 60)
summary_stats = corr1.summary()
print(f"  Count:          {summary_stats['count']}")
print(f"  Mean:           {summary_stats['mean']:.4f}")
print(f"  Median:         {summary_stats['median']:.4f}")
print(f"  Std Dev:        {summary_stats['std']:.4f}")
print(f"  Range:          {summary_stats['min']:.4f} to {summary_stats['max']:.4f}")
print(f"  1st percentile: {summary_stats['p1']:.4f}")
print(f"  99th percentile:{summary_stats['p99']:.4f}")
print(f"  Skewness:       {summary_stats['skewness']:.4f}")
print(f"  Kurtosis:       {summary_stats['kurtosis']:.4f}")
print(f"  Normality test: p-value = {summary_stats['normality_pvalue']:.6f}")

# Sort by correlation strength
print(f"\nDays Sorted by Correlation Strength (Top 5):")
best_days = corr1.get_days_by_correlation(high=True)
print(best_days.head(5).to_string())

print(f"\nDays Sorted by Correlation Strength (Bottom 5):")
worst_days = corr1.get_days_by_correlation(high=False)
print(worst_days.head(5).to_string())
