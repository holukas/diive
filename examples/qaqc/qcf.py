"""
Example for overall quality control flag (QCF) calculation using FlagQCF.

Demonstrates how to combine multiple individual test flags into a single
quality indicator (QCF) and generate quality-controlled data series.

Run this script to see QCF calculation:
    python examples/qaqc/qcf.py
"""
import numpy as np
import pandas as pd
import diive as dv


def example_qcf_basic():
    """Combine multiple quality test flags into overall QCF flag.

    Demonstrates the basic QCF workflow: initialize with test flags,
    calculate QCF from hard/soft flag counts, and access quality-controlled
    series. Shows how QCF values (0=good, 1=marginal, 2=poor) drive data
    rejection logic.
    """
    # Create test data: hourly flux measurements
    dates = pd.date_range('2023-06-01', periods=240, freq='h')
    dates.name = 'TIMESTAMP_START'
    df = pd.DataFrame(index=dates)

    # Original measured flux
    np.random.seed(42)
    df['NEE'] = np.random.normal(loc=-2.0, scale=1.5, size=len(df))

    # Create individual quality test flags (0=pass, 1=soft warning, 2=hard fail)
    # Pattern: FLAG_TEST{n}_{idstr}_{variable}_TEST
    # Realistic temporal patterns reflecting typical data quality issues across 10-day period:
    # - TEST1: Range/threshold violations (early morning issues: sunrise/instrument warmup)
    # - TEST2: Variability issues (concentrated around midday peak, scattered evening)
    # - TEST3: Soft warnings (broad distributed coverage, overlaps with multiple tests)
    # - TEST4: Outliers/spikes (isolated incidents throughout)
    # - TEST5: Soft warnings (distributed, specific problematic hours)
    # - TEST6: Soft warnings (concentrated in afternoon/evening)

    df['FLAG_TEST1_L41_NEE_TEST'] = 0
    # Early morning issues (sunrise transitions) - spread across days
    df.iloc[[2, 3, 28, 30, 54, 55, 80, 82, 106, 108, 132, 135, 158, 160, 186, 188, 212, 214],
            df.columns.get_loc('FLAG_TEST1_L41_NEE_TEST')] = 2  # 18 hard fails scattered

    df['FLAG_TEST2_L41_NEE_TEST'] = 0
    # Midday variability + evening issues - different pattern than TEST1
    df.iloc[[18, 20, 45, 47, 70, 72, 95, 98, 120, 122, 147, 150, 172, 174, 200, 202, 227, 230],
            df.columns.get_loc('FLAG_TEST2_L41_NEE_TEST')] = 2  # 18 hard fails, different locations

    df['FLAG_TEST3_L41_NEE_TEST'] = 0
    # Soft warnings: broader ranges with gradual quality degradation
    df.iloc[5:12, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1    # Range 5-11 (7 records)
    df.iloc[42:52, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1   # Range 42-51 (10 records)
    df.iloc[92:102, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1  # Range 92-101 (10 records)
    df.iloc[165:178, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1 # Range 165-177 (13 records)
    df.iloc[210:224, df.columns.get_loc('FLAG_TEST3_L41_NEE_TEST')] = 1 # Range 210-223 (14 records)

    df['FLAG_TEST4_L41_NEE_TEST'] = 0
    # Isolated outliers/spikes throughout entire period (strategic locations)
    df.iloc[[11, 35, 62, 88, 115, 141, 178, 205, 232],
            df.columns.get_loc('FLAG_TEST4_L41_NEE_TEST')] = 2  # 9 isolated hard fails

    df['FLAG_TEST5_L41_NEE_TEST'] = 0
    # Soft warnings: afternoon/evening hours with specific problem areas
    df.iloc[25:38, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1   # Range 25-37 (13 records)
    df.iloc[75:88, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1   # Range 75-87 (13 records)
    df.iloc[128:142, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1 # Range 128-141 (14 records)
    df.iloc[190:203, df.columns.get_loc('FLAG_TEST5_L41_NEE_TEST')] = 1 # Range 190-202 (13 records)

    df['FLAG_TEST6_L41_NEE_TEST'] = 0
    # Soft warnings: concentrated in specific periods (afternoon/night variability)
    df.iloc[15:25, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1   # Range 15-24 (10 records)
    df.iloc[55:70, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1   # Range 55-69 (15 records)
    df.iloc[110:128, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1 # Range 110-127 (18 records)
    df.iloc[155:170, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1 # Range 155-169 (15 records)
    df.iloc[220:238, df.columns.get_loc('FLAG_TEST6_L41_NEE_TEST')] = 1 # Range 220-237 (18 records)

    print(f"Test data: {len(df)} hourly records")
    print(f"Individual test flags created (with overlaps):")
    print(f"  - TEST1: {(df['FLAG_TEST1_L41_NEE_TEST'] == 2).sum()} hard flags")
    print(f"  - TEST4: {(df['FLAG_TEST4_L41_NEE_TEST'] == 2).sum()} hard flags (overlaps TEST1)")
    print(f"  - TEST2: {(df['FLAG_TEST2_L41_NEE_TEST'] == 1).sum()} soft flags (overlaps TEST1 and TEST4)")
    print(f"  - TEST3: {(df['FLAG_TEST3_L41_NEE_TEST'] == 1).sum()} soft flags (overlaps TEST2)")
    print(f"  - TEST5: {(df['FLAG_TEST5_L41_NEE_TEST'] == 1).sum()} soft flags (overlaps TEST3)")
    print(f"  - TEST6: {(df['FLAG_TEST6_L41_NEE_TEST'] == 1).sum()} soft flags (overlaps TEST3 and TEST5)")

    # Initialize QCF calculator
    qcf = dv.FlagQCF(
        df=df,
        target_col='NEE',  # Target variable column name
        outname='NEE',
        idstr='_L41'
    )

    # Calculate QCF from flags
    qcf.calculate(daytime_accept_qcf_below=2)

    # Access results
    print(f"\nQCF Results:")
    print(f"  QCF=0 (good): {(qcf.flagqcf == 0).sum()} records")
    print(f"  QCF=1 (marginal): {(qcf.flagqcf == 1).sum()} records")
    print(f"  QCF=2 (rejected): {(qcf.flagqcf == 2).sum()} records")

    # Quality-controlled series
    qcf_series = qcf.filteredseries
    qcf_hq = qcf.filteredseries_hq
    print(f"\nData availability after QC:")
    print(f"  Original records: {df['NEE'].notna().sum()}")
    print(f"  After QCF (NaN where QCF>=2): {qcf_series.notna().sum()}")
    print(f"  Highest-quality only (NaN where QCF>0): {qcf_hq.notna().sum()}")
    print(f"  Data loss: {100 - (qcf_series.notna().sum() / df['NEE'].notna().sum() * 100):.1f}%")

    # === REPORTING METHODS ===
    print("\n" + "=" * 80)
    print("DETAILED QUALITY CONTROL REPORTS")
    print("=" * 80)

    # 1. Overall QC summary
    print("\n[REPORT 1: OVERALL QUALITY CONTROL SUMMARY]")
    qcf.report_qcf_series()

    # 2. Individual test flag statistics
    print("\n[REPORT 2: INDIVIDUAL TEST FLAG STATISTICS]")
    qcf.report_qcf_flags()

    # 3. Test evolution (impact of each test)
    print("\n[REPORT 3: TEST IMPACT ANALYSIS - Sequential Application]")
    qcf.report_qcf_evolution()

    # === VISUALIZATION ===
    print("\nGenerating QCF heatmaps (4 panels)...")
    print("  Panels: (1) Original series  (2) QC series  (3) Flag sums  (4) QCF flag")
    qcf.showplot_qcf_heatmaps(figsize=(16, 6))


if __name__ == '__main__':
    example_qcf_basic()
