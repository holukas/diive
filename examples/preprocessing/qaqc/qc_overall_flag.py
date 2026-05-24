"""
================================
Overall Quality Control Flag (QCF)
================================

Combine multiple quality test flags into a single overall quality indicator.
QCF values: 0=good, 1=marginal, 2=poor/rejected.
"""

# %%
# Create synthetic test data with multiple quality flags
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate hourly flux measurements with individual quality test results.
# Each test produces flags: 0=pass, 1=soft warning, 2=hard fail.

import numpy as np
import pandas as pd

import diive as dv

# Create hourly test data
dates = pd.date_range('2023-06-01', periods=240, freq='h')
dates.name = 'TIMESTAMP_START'
df = pd.DataFrame(index=dates)

# Measured flux
np.random.seed(42)
df['NEE'] = np.random.normal(loc=-2.0, scale=1.5, size=len(df))

# Create individual quality test flags
# Pattern: FLAG_TEST{n}_{idstr}_{variable}_TEST
df['FLAG_TEST1_L41_NEE_TEST'] = 0
df.iloc[[2, 3, 28, 30, 54, 55, 80, 82, 106, 108, 132, 135, 158, 160, 186, 188, 212, 214],
df.columns.get_loc('FLAG_TEST1_L41_NEE_TEST')] = 2

df['FLAG_TEST2_L41_NEE_TEST'] = 0
df.iloc[[18, 20, 45, 47, 70, 72, 95, 98, 120, 122, 147, 150, 172, 174, 200, 202, 227, 230],
df.columns.get_loc('FLAG_TEST2_L41_NEE_TEST')] = 2

df['FLAG_TEST3_L41_NEE_TEST'] = 0
df.iloc[5:12, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1
df.iloc[42:52, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1
df.iloc[92:102, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1

df['FLAG_TEST4_L41_NEE_TEST'] = 0
df.iloc[[11, 35, 62, 88, 115, 141, 178, 205, 232],
df.columns.get_loc('FLAG_TEST4_L41_NEE_TEST')] = 2

df['FLAG_TEST5_L41_NEE_TEST'] = 0
df.iloc[25:38, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1
df.iloc[75:88, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1

df['FLAG_TEST6_L41_NEE_TEST'] = 0
df.iloc[15:25, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1
df.iloc[55:70, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1

print("Test data created:")
print(f"  Records: {len(df)}")
print(f"  Individual test flags: 6 (with overlaps)")

# %%
# Calculate overall QCF
# ^^^^^^^^^^^^^^^^^^^^
#
# Combine all test flags into single quality indicator.
# Hard flags (2) weighted more heavily than soft flags (1).

qcf = dv.qaqc.FlagQCF(
    df=df,
    target_col='NEE',
    outname='NEE',
    idstr='_L41'
)

# Calculate QCF from flags
qcf.calculate(daytime_accept_qcf_below=2)

# Results
qcf_good = (qcf.flagqcf == 0).sum()
qcf_marginal = (qcf.flagqcf == 1).sum()
qcf_poor = (qcf.flagqcf == 2).sum()

print("\nQCF Results:")
print(f"  QCF=0 (good): {qcf_good} records ({100 * qcf_good / len(df):.1f}%)")
print(f"  QCF=1 (marginal): {qcf_marginal} records ({100 * qcf_marginal / len(df):.1f}%)")
print(f"  QCF=2 (rejected): {qcf_poor} records ({100 * qcf_poor / len(df):.1f}%)")

# %%
# Access quality-controlled series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Multiple filtering options based on QCF thresholds.

# Series with QCF=2 set to NaN
qcf_series = qcf.filteredseries
# Series with QCF>0 set to NaN (only highest-quality data)
qcf_hq = qcf.filteredseries_hq

print("\nData availability:")
print(f"  Original: {df['NEE'].notna().sum()} records")
print(
    f"  After QCF filter (QCF<2): {qcf_series.notna().sum()} records ({100 * qcf_series.notna().sum() / df['NEE'].notna().sum():.1f}%)")
print(
    f"  HQ only (QCF=0): {qcf_hq.notna().sum()} records ({100 * qcf_hq.notna().sum() / df['NEE'].notna().sum():.1f}%)")

# %%
# Handle missing values in quality assessment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Missing/NaN values are automatically considered poor quality (QCF=2).
# This section demonstrates how missing data propagates through QC flagging.

import numpy as np

# Add some missing values to flux
df['NEE'].iloc[[10, 50, 100, 150, 200]] = np.nan

print("\nMissing Data Impact:")
print(f"  Total records: {len(df)}")
print(f"  Records with data: {df['NEE'].notna().sum()}")
print(f"  Missing records: {df['NEE'].isna().sum()}")

# Missing values are automatically flagged as poor quality
qcf_with_missing = dv.qaqc.FlagQCF(
    df=df,
    target_col='NEE',
    outname='NEE',
    idstr='_L41_missing'
)
qcf_with_missing.calculate(daytime_accept_qcf_below=2)

# Count quality flags including missing values
missing_as_poor = (qcf_with_missing.flagqcf == 2).sum()
valid_only = df['NEE'].notna().sum()

print(f"\nQCF with missing values:")
print(f"  QCF=2 (rejected/missing): {missing_as_poor} records ({100 * missing_as_poor / len(df):.1f}%)")
print(f"  Valid data remaining: {valid_only} records ({100 * valid_only / len(df):.1f}%)")

# Flagging records with excessive missing data in a window
print("\nDetecting excessive missing data:")
window = 24  # 24-hour window for daily data
missing_count = df['NEE'].rolling(window=window, min_periods=1).apply(lambda x: x.isna().sum())
excessive_missing = missing_count > (window * 0.5)  # More than 50% missing in window
print(f"  Windows with >50% missing: {excessive_missing.sum()}")

# %%
# Generate quality reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Detailed analysis of quality control impact.

print("\n" + "=" * 60)
print("Quality Control Reports")
print("=" * 60)

# Overall summary
print("\nOverall QC Summary:")
qcf_with_missing.report_qcf_series()

# Individual test statistics
print("\nIndividual Test Flag Statistics:")
qcf_with_missing.report_qcf_flags()

# Test evolution (impact of each test)
print("\nTest Impact Analysis:")
qcf_with_missing.report_qcf_evolution()
