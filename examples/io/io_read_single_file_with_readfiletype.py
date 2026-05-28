"""
================================================
Read single EddyPro file with ReadFileType
================================================

Load EddyPro CSV file using pre-defined filetype configuration.
"""

# %%
# ReadFileType with pre-defined filetype
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Simplest approach: use a known filetype to read single file automatically.

from pathlib import Path
from diive.core.io.filereader import ReadFileType

# Resolve example data file path relative to this script
data_dir = Path(__file__).parent.parent.parent / "diive" / "configs" / "exampledata" / "EDDYPRO-FLUXNET-CSV-30MIN_multiple"
filepath = data_dir / "eddypro_CH-HON_FR-20240819-090003_fluxnet_2024-08-19T090019_adv.csv"

rft = ReadFileType(
    filetype='EDDYPRO-FLUXNET-CSV-30MIN',
    filepath=filepath
)

# %%
# Extract data and metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# ReadFileType automatically applies all pre-configured parameters.

df, meta = rft.get_filedata()

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Frequency: {df.index.freq}")

# %%
# Available variables
# ^^^^^^^^^^^^^^^^^^^

print(f"\nTotal columns: {len(df.columns)}")
print(f"\nFirst 10 columns:")
print(df.columns[:10].tolist())

print(f"\nVariable groups:")
co2_cols = [c for c in df.columns if 'CO2' in c]
h2o_cols = [c for c in df.columns if 'H2O' in c]
print(f"  CO2 variables: {len(co2_cols)}")
print(f"  H2O variables: {len(h2o_cols)}")
