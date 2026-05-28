"""
Load EddyPro _FLUXNET_ output file, limit time range, and save as subset parquet.

This script loads a CSV file, filters it to a specified time range, and optionally:
- Removes intermediate processing columns (starting with dot)
- Excludes columns matching specific patterns
- Keeps only columns matching specific patterns (inclusion filter)
Saves the subset as a new parquet file with a simpler filename.

Usage:
    python scripts/parquet_timerange_subset_ch-cha.py

    Then modify the configuration variables below to customize for your data:
    - PARQUET_FILE: Source parquet file path
    - START_YEAR, END_YEAR, START_MONTH, END_MONTH: Time range to keep
    - REMOVE_DOT_COLUMNS: Remove columns starting with '.' (default True)
    - EXCLUDE_PATTERNS: Exclude columns containing patterns (e.g., "FLAG", None)
    - COLUMN_PATTERN: Keep only columns containing patterns (e.g., "FC", ["FC", "LE"], None)
    - OUTPUT_DIR, OUTPUT_FILENAME: Output location and filename
"""
from pathlib import Path

import diive as dv
from diive.core.io.files import save_parquet

# ===========================
# Configuration
# ===========================

# Source file (full path)
CSV_FILE = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-hon_flux_product\data\OPENLAG-IRGA-Level-0_fluxnet_2024-2026.03\eddypro_CH-HON_2025_FR-20260414-093114_fluxnet_2026-04-14T104905_adv.csv"

# Time range to keep
START_YEAR = 2025
END_YEAR = 2025
START_MONTH = 1  # June
END_MONTH = 12  # August

# Column filtering (optional)
# Keep only columns containing these patterns. Set to None to keep all columns.
# Examples: "FC" for CO2 flux, ["FC", "LE"] for CO2 and latent heat, None for all
COLUMN_PATTERN = [
    # 'FC',
    # 'FC_CORRDIFF',
    # 'FC_NR',
    # 'FC_NSR',
    # 'FC_RANDUNC_HF',
    # 'FC_SCF',
    # 'FC_SSITC_TEST',
    # 'SC_SINGLE',
    # 'SW_IN_POT',
    # 'SW_IN_T1_47_1_gfXG',
    # 'TA_T1_47_1_gfXG',
    # 'VPD_T1_47_1_gfXG',
    # 'TIMESTAMP_START',
    # 'USTAR',
    # 'VM97',
    # 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
    # 'EXPECT_NR',
    # 'CO2_NR',
    'TLAG_ACTUAL'
]

# Exclude columns (optional)
# Remove columns starting with a dot (intermediate processing columns)
REMOVE_DOT_COLUMNS = True
# Exclude columns containing these patterns. Set to None to keep all columns.
# Examples: "FLAG" to exclude quality flags, ["FLAG", "SUM"] for flags and sums, None for none
EXCLUDE_PATTERNS = ["BADM"]

# Output directory and filename
OUTPUT_DIR = Path(__file__).parent  # Save in scripts/ folder
OUTPUT_FILENAME = "exampledata_PARQUET_CH-HON_2025_LEVEL-0_TLAG-VARS_IRGA75_SUBSET"


# ===========================
# Processing
# ===========================

def main():
    """Load EddyPro CSV, subset by time and columns, save as parquet."""

    print("=" * 80)
    print("EddyPro FLUXNET CSV to Parquet Subsetter")
    print("=" * 80)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load EddyPro CSV file
    print(f"\nLoading EddyPro FLUXNET CSV file...")
    print(f"Source: {CSV_FILE}")
    loaddatafile = dv.read_file_type(
        filetype='EDDYPRO-FLUXNET-CSV-30MIN',
        filepath=CSV_FILE,
        data_nrows=None
    )
    df, _ = loaddatafile.get_filedata()

    print(f"\n" + "=" * 80)
    print(f"Original Data")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Period: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {sorted(df.columns.tolist())}")

    # Filter by year range
    df_subset = df.loc[(df.index.year >= START_YEAR) & (df.index.year <= END_YEAR)].copy()

    # Filter by month range (if within same year)
    if START_YEAR == END_YEAR:
        df_subset = df_subset.loc[(df_subset.index.month >= START_MONTH) &
                                  (df_subset.index.month <= END_MONTH)].copy()

    # Remove columns starting with a dot (if enabled)
    cols_before_dot_filter = len(df_subset.columns)
    if REMOVE_DOT_COLUMNS:
        cols_to_keep = [col for col in df_subset.columns if not col.startswith('.')]
        df_subset = df_subset[cols_to_keep].copy()
    cols_after_dot_filter = len(df_subset.columns)
    cols_removed_dot = cols_before_dot_filter - cols_after_dot_filter

    # Exclude columns by pattern (if specified)
    cols_before_exclude_filter = len(df_subset.columns)
    if EXCLUDE_PATTERNS:
        # Handle both single string and list of patterns
        exclude_patterns = EXCLUDE_PATTERNS if isinstance(EXCLUDE_PATTERNS, list) else [EXCLUDE_PATTERNS]
        cols_to_keep = [col for col in df_subset.columns if not any(pat in col for pat in exclude_patterns)]
        df_subset = df_subset[cols_to_keep].copy()
    cols_after_exclude_filter = len(df_subset.columns)
    cols_removed_exclude = cols_before_exclude_filter - cols_after_exclude_filter

    # Include columns by pattern (if specified)
    cols_after_include_filter = len(df_subset.columns)
    if COLUMN_PATTERN:
        # Handle both single string and list of patterns
        patterns = COLUMN_PATTERN if isinstance(COLUMN_PATTERN, list) else [COLUMN_PATTERN]
        cols_to_keep = [col for col in df_subset.columns if any(pat in col for pat in patterns)]
        df_subset = df_subset[cols_to_keep].copy()
    cols_removed_include = cols_after_include_filter - len(df_subset.columns)

    # Remove completely empty columns (all NaN)
    cols_before_empty_filter = len(df_subset.columns)
    cols_to_keep = [col for col in df_subset.columns if df_subset[col].notna().any()]
    df_subset = df_subset[cols_to_keep].copy()
    cols_removed_empty = cols_before_empty_filter - len(df_subset.columns)

    print(f"\n" + "=" * 80)
    print(f"Subset Data (Years {START_YEAR}-{END_YEAR}, Months {START_MONTH}-{END_MONTH})")
    print("=" * 80)
    print(f"Shape: {df_subset.shape}")
    print(f"Period: {df_subset.index.min()} to {df_subset.index.max()}")
    print(f"Records kept: {len(df_subset)} / {len(df)} ({len(df_subset) / len(df) * 100:.1f}%)")
    if REMOVE_DOT_COLUMNS and cols_removed_dot > 0:
        print(f"Removed {cols_removed_dot} columns starting with '.'")
    if EXCLUDE_PATTERNS and cols_removed_exclude > 0:
        exclude_str = EXCLUDE_PATTERNS if isinstance(EXCLUDE_PATTERNS, str) else ", ".join(EXCLUDE_PATTERNS)
        print(f"Excluded {cols_removed_exclude} columns matching patterns: {exclude_str}")
    if COLUMN_PATTERN and cols_removed_include > 0:
        patterns_str = COLUMN_PATTERN if isinstance(COLUMN_PATTERN, str) else ", ".join(COLUMN_PATTERN)
        print(f"Kept only {len(df_subset.columns)} columns matching patterns: {patterns_str}")
    if cols_removed_empty > 0:
        print(f"Removed {cols_removed_empty} completely empty columns (all NaN)")
    print(f"Columns: {sorted(df_subset.columns.tolist())}")

    # Save subset
    print(f"\n" + "=" * 80)
    print(f"Saving Subset")
    print("=" * 80)
    output_path = save_parquet(
        filename=OUTPUT_FILENAME,
        data=df_subset,
        outpath=str(OUTPUT_DIR)
    )

    print(f"\n" + "=" * 80)
    print(f"Complete!")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Records: {len(df)} -> {len(df_subset)}")
    print(f"Columns: {cols_before_dot_filter}", end="")
    if REMOVE_DOT_COLUMNS and cols_removed_dot > 0:
        print(f" -> {cols_after_dot_filter}", end="")
    if EXCLUDE_PATTERNS and cols_removed_exclude > 0:
        print(f" -> {cols_after_exclude_filter}", end="")
    if COLUMN_PATTERN and cols_removed_include > 0:
        print(f" -> {cols_after_include_filter}", end="")
    if cols_removed_empty > 0:
        print(f" -> {len(df_subset.columns)}", end="")
    print()
    if REMOVE_DOT_COLUMNS and cols_removed_dot > 0:
        print(f"  - Removed {cols_removed_dot} starting with '.'")
    if EXCLUDE_PATTERNS and cols_removed_exclude > 0:
        exclude_str = EXCLUDE_PATTERNS if isinstance(EXCLUDE_PATTERNS, str) else ", ".join(EXCLUDE_PATTERNS)
        print(f"  - Excluded {cols_removed_exclude} matching patterns: {exclude_str}")
    if COLUMN_PATTERN and cols_removed_include > 0:
        patterns_str = COLUMN_PATTERN if isinstance(COLUMN_PATTERN, str) else ", ".join(COLUMN_PATTERN)
        print(f"  - Included {cols_after_include_filter - cols_removed_include} columns matching patterns")
    if cols_removed_empty > 0:
        print(f"  - Removed {cols_removed_empty} completely empty columns (all NaN)")

    print(df_subset)


if __name__ == '__main__':
    main()
