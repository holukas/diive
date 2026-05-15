"""
=============================
QuickFill: Rapid Prototyping
=============================

Rapid gap-filling using RandomForestTS with minimal features.

QuickFill enables exploratory and preliminary gap-filling with speed. It uses
single lag feature only (immediate past value) and minimal tree complexity
for very fast execution (~3 seconds for typical dataset).

Quality: Low (exploratory only - not for production)
Speed: Very fast (~3 seconds)
Use case: Quick prototyping, testing, parameter exploration
"""

# %%
# QuickFill example
# =================
#
# For exploratory analysis, use QuickFillRFTS for rapid gap-filling with
# minimal features and fast execution. Demonstrates complete workflow from
# data loading through results reporting.

import pandas as pd
import importlib.metadata
from datetime import datetime

import diive as dv

# %%
# Session information
# ^^^^^^^^^^^^^^^^^^^
#
# Display DIIVE version and current timestamp.

dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Session timestamp: {dt_string}")
version_diive = importlib.metadata.version("diive")
print(f"DIIVE version: v{version_diive}")

# %%
# QuickFillRFTS overview
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Show the QuickFillRFTS class docstring explaining the rapid prototyping approach.

print("\n" + dv.QuickFillRFTS.__name__)
print(dv.QuickFillRFTS.__doc__)

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^
#
# Load example ecosystem flux data and filter to high-quality observations.

TARGET_COL = 'NEE_CUT_REF_orig'
subsetcols = [TARGET_COL, 'Tair_f', 'VPD_f', 'Rg_f']

df = dv.load_exampledata_parquet()
keep = (df.index.year >= 2020) & (df.index.year <= 2020)
df = df[keep].copy()

# Only High-quality (QCF=0) measured NEE used for model training in this example
lowquality = df["QCF_NEE"] > 0
df.loc[lowquality, TARGET_COL] = pd.NA
df = df[subsetcols].copy()

# %%
# Data summary statistics
# ^^^^^^^^^^^^^^^^^^^^^^^

print("\nData summary:")
df.describe()
statsdf = dv.sstats(df[TARGET_COL])
print(statsdf)

dv.TimeSeries(series=df[TARGET_COL]).plot()

# %%
# Run QuickFill
# ^^^^^^^^^^^^^
#
# Execute rapid gap-filling with minimal configuration.

qf = dv.QuickFillRFTS(df=df, target_col=TARGET_COL)
qf.fill()
qf.report()
gapfilled = qf.get_gapfilled_target()

# %%
# Visualize results
# ^^^^^^^^^^^^^^^^^
#
# Display heatmap comparisons of observed vs. gap-filled data.

dv.HeatmapDateTime(series=df[TARGET_COL]).show()
dv.HeatmapDateTime(series=gapfilled).show()

print("✓ QuickFill gap-filling complete.")
