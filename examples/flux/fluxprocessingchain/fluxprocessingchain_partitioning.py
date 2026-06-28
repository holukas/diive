"""
==================================================
Flux Processing Chain - NEE Partitioning (Level 4.2)
==================================================

After gap-filling (Level 4.1) the chain can split the gap-filled net ecosystem
exchange (NEE) into its gross components, gross primary production (GPP) and
ecosystem respiration (RECO), at **Level 4.2**.

Four faithful partitioning ports are wired in, each its own composable callable
(mirroring the four L4.1 gap-fillers):

- ``run_level42_nighttime_oneflux``   -> ``*_NT_OF`` (Reichstein 2005, ONEFlux)
- ``run_level42_nighttime_reddyproc`` -> ``*_NT_RP`` (Reichstein 2005, REddyProc)
- ``run_level42_daytime_reddyproc``   -> ``*_DT_RP`` (Lasslop 2010, REddyProc)
- ``run_level42_daytime_oneflux``     -> ``*_DT_OF`` (Lasslop 2010, ONEFlux)

Partitioning runs once per USTAR scenario; each variant's output columns get
the scenario label appended (``RECO_NT_OF_CUT_50``) so all scenarios coexist.
The **nighttime** variants need a gap-filled NEE for the GPP residual and read
it from the L4.1 method named by ``partition_gapfill_method`` (default
``'mds'``). The **daytime** variants use measured NEE only.

This example uses the single-call ``run_chain`` path; the same four callables
are available on the composable per-level API (see
``fluxprocessingchain_composable.py``).
"""

# %%
# Imports
# ^^^^^^^

from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.flux.fluxprocessingchain import (
    FluxConfig,
    add_driver,
    init_flux_data,
    run_chain,
)

# %%
# Load data
# ^^^^^^^^^
#
# The full CH-LAE 2024 year — partitioning fits the temperature / light
# response over time, so it needs a real seasonal cycle (a single month would
# leave most records unpartitioned). We drop the reserved ``SW_IN_POT`` /
# ``DAYTIME`` / ``NIGHTTIME`` columns so ``init_flux_data`` can recompute them.

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                      if c in df.columns])

# %%
# Register VPD in kPa
# ^^^^^^^^^^^^^^^^^^^
#
# Both MDS (L4.1) and the daytime partitioning variants want VPD in **kPa**;
# the input VPD here is in hPa. Convert once and register via ``add_driver`` so the
# column lands in ``data.full_df`` where the levels read their drivers.

data = init_flux_data(
    df=df,
    fluxcol="FC",
    site_lat=47.41887,  # CH-LAE
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)
vpd_kpa = (df['VPD_T1_47_1_gfXG'] / 10.0).rename('VPD_kPa')
data = add_driver(data, vpd_kpa)

# %%
# Build the FluxConfig
# ^^^^^^^^^^^^^^^^^^^^^
#
# We gap-fill with MDS only (fast) and enable all four partitioning variants.
# The ``partition_*`` driver fields point at columns in ``data.full_df``:
#
# - ``partition_ta`` / ``partition_sw_in`` — *measured* TA / SW_IN (used for the
#   day/night split and the fits in three of the four variants). This LAE subset
#   ships only the gap-filled meteo, so we reuse the gap-filled columns here; a
#   real workflow would point these at the raw measured series.
# - ``partition_ta_f`` / ``partition_sw_in_f`` / ``partition_vpd_f`` — gap-filled
#   meteo drivers.
#
# The nighttime variants additionally consume the L4.1 gap-filled NEE selected
# by ``partition_gapfill_method`` (here MDS).

cfg = FluxConfig(
    fluxcol='FC',
    ustar_thresholds=[0.07],
    ustar_labels=['CUT_50'],
    level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
    # L4.1 — MDS only (gap-filled NEE for the nighttime GPP residual).
    gapfill_mds=True,
    gapfill_rf=False,
    gapfill_xgb=False,
    mds_swin='SW_IN_T1_47_1_gfXG',
    mds_ta='TA_T1_47_1_gfXG',
    mds_vpd='VPD_kPa',
    # L4.2 — all four partitioning variants.
    partition_nighttime_oneflux=True,
    partition_nighttime_reddyproc=True,
    partition_daytime_reddyproc=True,
    partition_daytime_oneflux=True,
    partition_gapfill_method='mds',          # which L4.1 NEE feeds the nighttime variants
    partition_ta='TA_T1_47_1_gfXG',          # measured TA (stand-in: gap-filled here)
    partition_sw_in='SW_IN_T1_47_1_gfXG',    # measured SW_IN (stand-in: gap-filled here)
    partition_ta_f='TA_T1_47_1_gfXG',        # gap-filled TA
    partition_sw_in_f='SW_IN_T1_47_1_gfXG',  # gap-filled SW_IN
    partition_vpd_f='VPD_kPa',               # gap-filled VPD (kPa)
)

# %%
# Run the chain
# ^^^^^^^^^^^^^
#
# One call drives L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1 -> L4.2. The summary
# includes a partitioning section (RECO / GPP valid-record counts per variant
# and USTAR scenario).

data = run_chain(data, cfg)
print(f"\nlevels run: {data.level_ids}")
print(data.summary())

# %%
# Discover the partitioning columns
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``partitioned_cols()`` returns ``{variant: {ustar_scenario: [column, ...]}}``
# — the canonical way to find the RECO / GPP columns in ``fpc_df`` without
# digging through the instances. ``data.levels.level42_<variant>[scenario]``
# holds each fitted partitioning instance for inspection.

cols = data.partitioned_cols()
for variant, scen_cols in cols.items():
    for scen, names in scen_cols.items():
        print(f"{variant} / {scen}: {names}")

# %%
# Compare RECO across the four methods
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The four ports do not produce identical numbers (different reference
# implementations and day/night splits), so comparing their daily-mean RECO is
# a useful sanity check. Headless here (``showplot`` left to the caller).

import matplotlib.pyplot as plt

reco_cols = {
    'Nighttime ONEFlux':   'RECO_NT_OF_CUT_50',
    'Nighttime REddyProc': 'RECO_NT_RP_CUT_50',
    'Daytime REddyProc':   'RECO_DT_RP_CUT_50',
    'Daytime ONEFlux':     'RECO_DT_OF_CUT_50',
}
fig, ax = plt.subplots(figsize=(13, 5))
for label, col in reco_cols.items():
    if col in data.fpc_df.columns:
        ax.plot(data.fpc_df[col].resample('D').mean(), lw=1.2, label=label)
ax.axhline(0, color='#455A64', lw=0.8)
ax.set_ylabel('RECO (umol m-2 s-1)')
ax.set_title('L4.2 RECO by partitioning method - CH-LAE 2024 daily means (CUT_50)')
ax.legend(ncol=2, frameon=False)
# plt.show()  # disabled for headless example runs
