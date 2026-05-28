"""
======================================================
Read single EddyPro file with DataFileReader
======================================================

Read EddyPro fluxnet CSV file by specifying all parameters manually.
"""

# %%
# DataFileReader with custom parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Read single file with explicit parameters for timestamp, missing values, and frequency.

from pathlib import Path
from diive.core.io.filereader import DataFileReader

# Resolve example data file path relative to this script
data_dir = Path(__file__).parent.parent.parent / "diive" / "configs" / "exampledata" / "EDDYPRO-FLUXNET-CSV-30MIN_multiple"
FILE = data_dir / "eddypro_CH-HON_FR-20240819-090003_fluxnet_2024-08-19T090019_adv.csv"

dfr = DataFileReader(
    filepath=FILE,
    data_header_section_rows=[0],
    data_skip_rows=[],
    data_header_rows=[0],
    data_varnames_row=0,
    data_varunits_row=None,
    data_na_vals=[-9999],
    data_freq="30min",
    data_delimiter=",",
    data_nrows=None,
    timestamp_idx_col=["TIMESTAMP_END"],
    timestamp_datetime_format="%Y%m%d%H%M",
    timestamp_start_middle_end="end",
    output_middle_timestamp=True,
    compression=None
)

# %%
# Extract data and metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# DataFileReader returns both data and metadata (units, tags).

df, meta = dfr.get_data()

print(f"Data shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Frequency: {df.index.freq}")

# %%
# Metadata contains units and tags
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Units are preserved from file headers where available.

print(f"\nMetadata shape: {meta.shape}")
print(f"Columns in metadata: {list(meta.columns)}")
