"""
Load parquet file, limit time range, and save as subset parquet.

This script loads a large parquet file, filters it to a specified time range, and optionally:
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

from diive.core.io.files import load_parquet, save_parquet

# ===========================
# Configuration
# ===========================

# Source parquet file (full path)
PARQUET_FILE = r"F:\Sync\luhk_work\dev-data\diive-data\dev_scripts\22.4_FLUXES_L1_IRGA72+METEO7_2016-2024.parquet"

# Time range to keep
START_YEAR = 2024
END_YEAR = 2024
START_MONTH = 1  # June
END_MONTH = 12  # August

# Column filtering (optional)
# Keep only columns containing these patterns. Set to None to keep all columns.
# Examples: "FC" for CO2 flux, ["FC", "LE"] for CO2 and latent heat, None for all
COLUMN_PATTERN = [
    'FC',
    'FC_CORRDIFF',
    'FC_NR',
    'FC_NSR',
    'FC_RANDUNC_HF',
    'FC_SCF',
    'FC_SSITC_TEST',
    'SC_SINGLE',
    'SW_IN_POT',
    'SW_IN_T1_47_1_gfXG',
    'TA_T1_47_1_gfXG',
    'VPD_T1_47_1_gfXG',
    'TIMESTAMP_START',
    'USTAR',
    'VM97',
    'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
    'EXPECT_NR',
    'CO2_NR'
]

# Exclude columns (optional)
# Remove columns starting with a dot (intermediate processing columns)
REMOVE_DOT_COLUMNS = True
# Exclude columns containing these patterns. Set to None to keep all columns.
# Examples: "FLAG" to exclude quality flags, ["FLAG", "SUM"] for flags and sums, None for none
EXCLUDE_PATTERNS = ["BADM"]

# Output directory and filename
OUTPUT_DIR = Path(__file__).parent  # Save in scripts/ folder
OUTPUT_FILENAME = "exampledata_PARQUET_CH-LAE_22.4_FLUXES_LEVEL-1_IRGA72+METEO7_2024_SUBSET"


# ===========================
# Processing
# ===========================

def main():
    """Load, subset, and save parquet file."""

    print("=" * 80)
    print("Parquet Time Range Subsetter")
    print("=" * 80)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load parquet file
    print(f"\nLoading parquet file...")
    print(f"Source: {PARQUET_FILE}")
    df = load_parquet(PARQUET_FILE)

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
        print(f" -> {len(df_subset.columns)}", end="")
    print()
    if REMOVE_DOT_COLUMNS and cols_removed_dot > 0:
        print(f"  - Removed {cols_removed_dot} starting with '.'")
    if EXCLUDE_PATTERNS and cols_removed_exclude > 0:
        exclude_str = EXCLUDE_PATTERNS if isinstance(EXCLUDE_PATTERNS, str) else ", ".join(EXCLUDE_PATTERNS)
        print(f"  - Excluded {cols_removed_exclude} matching patterns: {exclude_str}")
    if COLUMN_PATTERN and cols_removed_include > 0:
        patterns_str = COLUMN_PATTERN if isinstance(COLUMN_PATTERN, str) else ", ".join(COLUMN_PATTERN)
        print(f"  - Kept only columns matching patterns: {patterns_str}")


if __name__ == '__main__':
    main()
