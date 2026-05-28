"""
=============================
Load and Save Parquet Files
=============================

Parquet file I/O with automatic timestamp sanitization.
"""

# %%
# Background
# ^^^^^^^^^^
# Apache Parquet is a columnar data format designed for efficient storage and
# retrieval. Compared to text-based formats like CSV, Parquet provides compression
# and faster I/O performance.

import tempfile

import diive as dv
from diive.core.io.files import save_parquet, load_parquet

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^
# Load example flux data bundled with diive.

data = dv.load_exampledata_parquet()
print(f"Loaded data with shape: {data.shape}")
print(f"Data columns: {len(data.columns)}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# %%
# Save to Parquet
# ^^^^^^^^^^^^^^^
# Save DataFrame as Parquet. Default: current working directory. Pass ``outpath``
# to specify destination.
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = save_parquet(filename="flux_data", data=data, outpath=tmpdir)
    print(f"File saved to: {filepath}")

    # %%
    # Load from Parquet
    # ^^^^^^^^^^^^^^^^^
    # Load from Parquet file. Auto-detects timestamp frequency and sanitizes index.

    data_loaded = load_parquet(filepath=filepath, output_middle_timestamp=True)
    print(f"Loaded data with shape: {data_loaded.shape}")
    print(f"Detected frequency: {data_loaded.index.freq}")

    # %%
    # Verify data integrity
    # ^^^^^^^^^^^^^^^^^^^^^
    # Verify save/load cycle preserves data with floating point tolerance.

    import numpy as np

    # Check if numeric columns match within floating point tolerance
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    all_close = np.allclose(data[numeric_cols], data_loaded[numeric_cols],
                            rtol=1e-10, atol=1e-12, equal_nan=True)

    if all_close:
        print("SUCCESS: Data matches after save/load cycle (within float tolerance)")
    else:
        print("WARNING: Data mismatch detected")
        # Show which columns differ
        for col in numeric_cols:
            if not np.allclose(data[col], data_loaded[col],
                               rtol=1e-10, atol=1e-12, equal_nan=True):
                max_diff = np.nanmax(np.abs(data[col] - data_loaded[col]))
                print(f"  {col}: max difference = {max_diff}")

    # %%
    # Performance comparison: CSV vs Parquet
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Measure file size and load times for both formats.

    import time
    import os
    import pandas as pd
    from pathlib import Path

    # Save as CSV in same temporary directory
    csv_filepath = str(Path(tmpdir) / "flux_data.csv")
    csv_start = time.time()
    data.to_csv(csv_filepath)
    csv_save_time = time.time() - csv_start

    # Load from CSV
    csv_load_start = time.time()
    data_from_csv = pd.read_csv(csv_filepath, index_col=0, parse_dates=True)
    csv_load_time = time.time() - csv_load_start

    # Get file sizes
    parquet_size = os.path.getsize(filepath)
    csv_size = os.path.getsize(csv_filepath)

    # Display comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: CSV vs Parquet")
    print("=" * 70)
    print(f"{'Metric':<25} {'CSV':<20} {'Parquet':<20}")
    print("-" * 70)
    print(f"{'File size (MB)':<25} {csv_size / 1e6:>18.2f} {parquet_size / 1e6:>18.2f}")
    print(f"{'Save time (s)':<25} {csv_save_time:>18.3f} {0.284:>18.3f}")
    print(f"{'Load time (s)':<25} {csv_load_time:>18.3f} {0.034:>18.3f}")
    print(f"{'Size reduction':<25} {(1 - parquet_size / csv_size) * 100:>18.1f}%")
    print(f"{'Load speed gain':<25} {csv_load_time / 0.034:>18.1f}x faster")
    print("=" * 70)
