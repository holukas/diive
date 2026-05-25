"""
====================================================
Flux Processing Chain - Composable Functions
====================================================

The flux processing chain is also exposed as standalone pure functions, one per
level.  Each function takes a ``FluxLevelData`` container and returns a new
one — no shared state, no orchestrator class required.

This example runs **L2 + L3.1 + L3.2** to produce a quality-controlled,
storage-corrected, outlier-cleaned flux without going through USTAR filtering
or gap-filling.  Compare with ``fluxprocessingchain.py`` for the full chain
via the ``FluxProcessingChain`` orchestrator.
"""

# %%
# Imports
# ^^^^^^^

import diive as dv
from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
from diive.pkgs.flux.fluxprocessingchain import (
    init_flux_data,
    make_level32_detector,
    run_level2,
    run_level31,
    run_level32,
)

# %%
# Load data
# ^^^^^^^^^

df = load_exampledata_parquet_lae_level1_30MIN()
df = df.loc['2024-06':'2024-06']  # one month for speed

# %%
# Step 1: build the initial FluxLevelData container
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``init_flux_data`` adds potential radiation and day/night flags, assembles
# the frozen site-metadata record, and returns a container ready for the
# first level.

data = init_flux_data(
    df=df,
    fluxcol="FC",
    site_lat=47.41887,        # CH-HON
    site_lon=8.491318,
    utc_offset=1,
    nighttime_threshold=20,
    daytime_accept_qcf_below=2,
    nighttimetime_accept_qcf_below=2,
)
print(data)   # FluxLevelData has a useful __repr__

# %%
# Step 2: run Level-2 (quality flag expansion)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each test is enabled by passing a config dict containing at least
# ``{'apply': True}``.  Pass ``None`` (or omit) to skip a test.

data = run_level2(
    data,
    ssitc={'apply': True, 'setflag_timeperiod': None},
    gas_completeness={'apply': True},
    spectral_correction_factor={'apply': True},
    signal_strength={
        'apply': True,
        'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
        'method': 'discard below',
        'threshold': 60,
    },
    raw_data_screening_vm97={
        'apply': True,
        'spikes': True, 'amplitude': False, 'dropout': True,
        'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
        'discont_hf': False, 'discont_sf': False,
    },
)
print(f"After L2: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# Step 3: run Level-3.1 (storage correction)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

data = run_level31(data, gapfill_storage_term=True, set_storage_to_zero=False)
print(f"After L3.1: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# Step 4: run Level-3.2 (outlier removal)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Level-3.2 uses a stateful ``StepwiseOutlierDetection`` instance.  The
# ``make_level32_detector`` factory wires it to the right ``dfin`` / ``col``
# / site coordinates so you don't have to.  Call any number of
# ``flag_outliers_*`` / ``addflag`` methods on it, then hand it to
# ``run_level32``.

sod = make_level32_detector(data)
sod.flag_outliers_hampel_test(
    window_length=48 * 13,
    n_sigma_daytime=5.5, n_sigma_nighttime=5.5,
    use_differencing=True, separate_daytime_nighttime=True,
    showplot=False, verbose=True, repeat=True,
)
sod.addflag()

data = run_level32(data, outlier_detector=sod)
print(f"After L3.2: filteredseries = {data.filteredseries.name}, "
      f"{data.filteredseries.dropna().count()} valid records")

# %%
# Step 5: stop here — inspect typed per-level results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``data.levels`` is a typed ``LevelResults`` dataclass.  Every per-level
# object lives behind a named attribute (no magic-string dict lookups).

print(f"Levels run: {data.level_ids}")
print(f"L2 instance:        {type(data.levels.level2).__name__}")
print(f"L2 QCF:             {type(data.levels.level2_qcf).__name__}")
print(f"L3.1 instance:      {type(data.levels.level31).__name__}")
print(f"Storage-corr. col:  {data.levels.flux_corrected_col}")
print(f"L3.2 instance:      {type(data.levels.level32).__name__}")
print(f"L3.2 QCF:           {type(data.levels.level32_qcf).__name__}")

final_df = data.fpc_df
print(f"Final dataframe: {final_df.shape[0]} rows x {final_df.shape[1]} cols")

# %%
# Why use the composable API?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - **Partial pipelines**: stop after any level — no chain overhead.
# - **Custom steps**: skip ``run_level32`` and write your own outlier logic on
#   ``data.fpc_df``, then continue with ``run_level33_constant_ustar``.
# - **Branching**: run two L4.1 methods (e.g. MDS and XGBoost) from the same
#   L3.3 state by reusing the same ``FluxLevelData``.
# - **Pure functions**: each call returns a new container, never mutates the
#   input — easy to unit-test, easy to reason about.
#
# For the full L2-to-L4.1 chain (RF + XGBoost gap-filling), the
# ``FluxProcessingChain`` orchestrator is more concise — see
# ``fluxprocessingchain.py``.
