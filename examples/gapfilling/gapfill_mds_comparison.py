"""
==================================
FluxMDS reproducibility / determinism
==================================

Show that MDS gap-filling is deterministic: running ``FluxMDS`` twice on the
same data produces bit-identical gap-filled values and quality flags.

Unlike the machine-learning gap-fillers (Random Forest / XGBoost), MDS does not
train a model and uses no random sampling - it fills each gap from the average
of meteorologically similar measured records (the faithful ONEFlux marginal-
distribution-sampling cascade). So repeated runs always return exactly the same
result, which makes MDS reproducible without a random seed.
"""

# %%
# Imports
# =======
import numpy as np
import matplotlib.pyplot as plt

import diive as dv
from diive.gapfilling.mds import FluxMDS

# %%
# Load one year of example data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2022].copy()

flux = 'NEE_CUT_REF_orig'
swin, ta, vpd = 'Rg_f', 'Tair_f', 'VPD_f'

print(f"Records: {len(df):,d}")
print(f"Missing {flux}: {df[flux].isnull().sum():,d} "
      f"({100.0 * df[flux].isnull().sum() / len(df):.1f}%)")

# %%
# Run FluxMDS twice
# ^^^^^^^^^^^^^^^^^^
runs = []
for _ in range(2):
    model = FluxMDS(df=df, flux=flux, swin=swin, ta=ta, vpd=vpd, verbose=0)
    model.run()
    runs.append(model)

gapfilled_a = runs[0].get_gapfilled_target().to_numpy()
gapfilled_b = runs[1].get_gapfilled_target().to_numpy()
flag_a = runs[0].get_flag().to_numpy()
flag_b = runs[1].get_flag().to_numpy()

# %%
# Verify the two runs are identical
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
values_identical = np.array_equal(gapfilled_a, gapfilled_b, equal_nan=True)
flags_identical = np.array_equal(flag_a, flag_b, equal_nan=True)

print(f"\nGap-filled values identical: {values_identical}")
print(f"Quality flags identical:     {flags_identical}")
assert values_identical and flags_identical, "MDS gap-filling is not deterministic"
print("OK - MDS gap-filling is deterministic (no random seed needed).")

# %%
# Plot the gap-filled series
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
model = runs[0]
fig, ax = plt.subplots(figsize=(14, 4), dpi=100, layout='constrained')
flag = model.get_flag()
measured = flag == 0
ax.plot(df.index[measured], gapfilled_a[measured], ls='none', marker='o',
        ms=2, color='#455A64', label='measured')
ax.plot(df.index[~measured], gapfilled_a[~measured], ls='none', marker='o',
        ms=2, color='#F44336', label='gap-filled (MDS)')
ax.set_ylabel(flux)
ax.legend(loc='upper right')
ax.set_title('FluxMDS gap-filling (2022)')
# fig.show()  # disabled for the example gallery
