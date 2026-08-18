"""
====================================================
Flux Processing Chain - Level 2 (Quality Flags)
====================================================

Level 2 is the first real step of the flux processing chain: it expands the
quality flags an EddyPro **FLUXNET** output file already carries into individual
diive test flags, then aggregates them into one overall quality-control flag
(QCF) for the flux.

This example runs **only Level 2** on a genuine EddyPro FLUXNET output file
(Level-0 output), so it stays focused on what L2 does and reads:

  load EddyPro FLUXNET CSV  ->  ``init_flux_data``  ->  ``run_level2``

Each L2 test reads a fixed EddyPro-FLUXNET input column (see
:func:`level2_test_inputs`). The result is a QCF per record:
0 = good, 1 = marginal, 2 = bad — the filtered flux keeps only QCF below the
accept threshold configured at ``init_flux_data``.

For the full L2 -> L4.2 pipeline see
``fluxprocessingchain_composable.py`` (composable callables) and
``fluxprocessingchain_runchain.py`` (single-call driver).
"""

# %%
# Imports
# ^^^^^^^

from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
from diive.flux.fluxprocessingchain import (
    init_flux_data,
    level2_test_inputs,
    run_level2,
)

# Set True to display the QCF heatmaps interactively.
SHOWPLOT = True

# %%
# Load the EddyPro FLUXNET output file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN`` reads a real EddyPro FLUXNET
# 30-minute output file (CH-AWS, July 2022) and returns ``(data, metadata)``.
# This is the kind of file L2 is designed to consume — its columns
# (``FC_SSITC_TEST``, ``CO2_VM97_TEST``, ``CO2_NR``, ...) are the EddyPro
# quality diagnostics L2 expands into flags.

df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

# ``init_flux_data`` reserves ``SW_IN_POT`` / ``DAYTIME`` / ``NIGHTTIME`` (it
# computes its own from potential radiation) and raises if any pre-exists. Drop
# them so the chain can populate fresh values.
df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                      if c in df.columns])

# %%
# Which columns does each L2 test read?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``level2_test_inputs`` reports the EddyPro-FLUXNET input column(s) each test
# depends on, templated on the flux column and its base variable. Handy to
# check upfront whether a loaded dataset actually provides them.

for test_key, info in level2_test_inputs(fluxcol='FC', fluxbasevar='CO2').items():
    present = all(c in df.columns for c in info['inputs']) if info['inputs'] else None
    note = 'user-chosen column' if info['user_col'] else f"reads {info['inputs']} (available: {present})"
    print(f"  {test_key:<32} {note}")

# %%
# Step 1: initialise the FluxLevelData container
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``init_flux_data`` calculates potential radiation, derives daytime/nighttime
# flags, and assembles a frozen site-metadata record. ``daytime_accept_qcf_below=2``
# keeps QCF=0 (all tests pass) and QCF=1 (soft warnings) for daytime records;
# set to 1 to accept only the strictest quality.

data = init_flux_data(
    df=df,
    fluxcol='FC',
    site_lat=46.583056,  # CH-AWS (Alp Weissenstein)
    site_lon=9.790639,
    utc_offset=1,
    nighttime_threshold=20,  # W m-2; SW_IN below this = nighttime
    daytime_accept_qcf_below=2,
    nighttime_accept_qcf_below=2,
)
print(data)  # FluxLevelData has a useful __repr__

# %%
# Step 2: Level-2 — quality flag expansion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Each test is enabled by passing a config dict with ``{'apply': True, ...}``.
# Omit or pass ``None`` to skip. Recognised test keys: ``ssitc``,
# ``gas_completeness``, ``spectral_correction_factor``, ``signal_strength``,
# ``raw_data_screening_vm97``, ``angle_of_attack``,
# ``steadiness_of_horizontal_wind``.

data = run_level2(
    data,
    ssitc={'apply': True, 'setflag_timeperiod': None},
    gas_completeness={'apply': True},
    spectral_correction_factor={'apply': True},
    signal_strength={
        'apply': True,
        'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
        'method': 'discard below',  # low signal = problem on LI-7200
        'threshold': 60,
    },
    raw_data_screening_vm97={
        'apply': True,
        'spikes': True, 'amplitude': False, 'dropout': True,
        'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
        'discont_hf': False, 'discont_sf': False,
    },
)

# %%
# Step 3: inspect the Level-2 result
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# After L2, ``data.filteredseries`` is the QCF-filtered flux (``FC_L2_QCF``),
# and ``data.levels.filteredseries_hq`` keeps only the strictest QCF=0 records.

filtered = data.filteredseries
hq = data.levels.filteredseries_hq

print(f"L2 level ids:           {data.level_ids}")
print(f"Filtered flux column:   {filtered.name}")
print(f"Accepted records (QCF): {filtered.dropna().count()}")
print(f"High-quality (QCF=0):   {hq.dropna().count()}")

# The QCF object carries the per-test screening report and heatmaps. The full
# console reports (``qcf.report_qcf_flags()`` / ``qcf.report_qcf_evolution()``)
# stream the retained/rejected breakdown to a Rich-capable terminal.
qcf = data.levels.level2_qcf

# %%
# QCF heatmaps (L2 quality overview)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

if SHOWPLOT:
    qcf.showplot_qcf_heatmaps()
