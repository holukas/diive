"""
========================================================
Read multiple EddyPro files with MultiDataFileReader
========================================================

Load and merge multiple EddyPro CSV files using pre-defined filetype.
"""

# %%
# Search and list files
# ^^^^^^^^^^^^^^^^^^^^^
# Locate multiple files matching a pattern in a directory.

from pathlib import Path
from diive.core.io.filereader import MultiDataFileReader, search_files

# Resolve example data directory relative to this script
data_dir = Path(__file__).parent.parent.parent / "diive" / "configs" / "exampledata" / "EDDYPRO-FLUXNET-CSV-30MIN_multiple"
filepaths = search_files(
    searchdirs=str(data_dir),
    pattern='eddypro_CH-HON_FR-*_fluxnet_*_adv.csv'
)

print(f"Found {len(filepaths)} files:")
for fp in filepaths:
    print(f"  {fp.name}")

# %%
# Load multiple files with pre-defined filetype
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MultiDataFileReader merges data from multiple files automatically.

mdf = MultiDataFileReader(
    filetype='EDDYPRO-FLUXNET-CSV-30MIN',
    filepaths=filepaths,
    output_middle_timestamp=True
)

df = mdf.data_df
meta = mdf.metadata_df

# %%
# Check merged data
# ^^^^^^^^^^^^^^^^^
# Files are concatenated along the time axis.

print(f"\nMerged data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Frequency: {df.index.freq}")
print(f"Number of variables: {len(df.columns)}")

# %%
# Metadata from all files combined
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print(f"\nMetadata rows: {len(meta)}")
print(f"Variables with units: {(meta['UNITS'] != '-no-units-').sum()}")
