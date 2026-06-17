"""
FLUX PROCESSING CHAIN: MULTI-LEVEL WORKFLOW
=============================================

Swiss FluxNet multi-level post-processing: quality flag expansion (L2), storage correction (L3.1),
outlier removal (L3.2), USTAR filtering (L3.3), gap-filling (L4.1).

Two entry points
----------------

**Composable per-level API** — one callable per level, each returning a new
``FluxLevelData`` container::

    from diive.flux.fluxprocessingchain import (
        init_flux_data, run_level2, run_level31,
        make_level32_detector, run_level32,
        run_level33_constant_ustar,
        run_level41_mds, run_level41_rf, run_level41_xgb,
    )

    data = init_flux_data(df, fluxcol='FC', site_lat=..., site_lon=..., utc_offset=1)
    data = run_level2(data, ssitc={'apply': True, ...}, ...)
    data = run_level31(data)
    data, sod = make_level32_detector(data); sod.flag_outliers_hampel_test(...); sod.addflag()
    data = run_level32(data, outlier_detector=sod)
    data = run_level33_constant_ustar(data, thresholds=[0.18], threshold_labels=['CUT_50'])
    data = run_level41_mds(data, swin='SW_IN', ta='TA', vpd='VPD_kPa')

**Single-call convenience API** — one ``FluxConfig``, one ``run_chain`` call::

    from diive.flux.fluxprocessingchain import init_flux_data, run_chain, FluxConfig

    cfg = FluxConfig(fluxcol='FC', ustar_thresholds=[0.18], ustar_labels=['CUT_50'], ...)
    data = init_flux_data(df, fluxcol='FC', site_lat=..., site_lon=..., utc_offset=1)
    data = run_chain(data, cfg)

``run_chain`` is intentionally simple and **not** a parity superset of the
composable API — it picks fixed defaults for every per-level knob beyond what
``FluxConfig`` exposes (see the "What ``run_chain`` does *not* expose" section
in :func:`run_chain` for the exact list). The composable per-level callables
are the path to **full control** — every detector class, every model
hyperparameter, every diagnostic flag is reachable there and only there. Reach
for ``run_chain`` when you want a clean, low-friction default; reach for the
composable API when you need to override anything beyond ``FluxConfig``.

Why the per-level signatures look different
-------------------------------------------

Each level's signature matches the *shape* of what it controls.  The same shape
forced onto every level would harm at least one of them, so the API embraces five
small patterns rather than one bad uniform one:

============================== ============================= ===============================================
Level                          Shape                         Why this shape
============================== ============================= ===============================================
``run_level2``                 per-test config dicts         N independent EddyPro tests, each with its own
                                                             ``apply`` / threshold / time-period settings.
                                                             Flat kwargs would explode the namespace.
``run_level31``                flat booleans                 Two binary decisions only — a config object
                                                             would be overkill.
``run_level32``                pre-built detector object     Outlier detection is inherently sequential and
                                                             stateful: each ``flag_*`` + ``addflag()`` pair
                                                             filters the survivors of the previous step.
                                                             Not expressible as kwargs without a DSL.
``run_level33_constant_ustar`` parallel lists                Multiple USTAR scenarios are positional pairs
                                                             (threshold, label).
``run_level33_variable_ustar`` ``{label: threshold Series}`` Time-varying (e.g. per-year VUT) thresholds —
                                                             one full per-record Series per scenario.
``run_level41_*``              built object + ad-hoc kwargs  Feature engineering is itself an 8-stage
                                                             configuration; passing it as a built object
                                                             is correct.
============================== ============================= ===============================================

If you find the per-level variation distracting, use ``run_chain(data, config)``
as the **high-level entry point**: one ``FluxConfig`` carries the
typical-user decisions (which L2 tests, which USTAR mode, which gap-filling
methods, which driver columns), and ``run_chain`` picks fixed defaults for
everything else. It is **not** a parity superset of the composable API —
the per-detector and per-model knobs are reachable only by calling the
per-level functions directly. See :func:`run_chain` for the explicit list
of what it does and does not expose.

Column naming convention
------------------------

Filtered flux columns in ``data.fpc_df`` carry one of two QCF suffixes that
encode *which* quality threshold was applied:

- ``..._QCF`` — the flux filtered at the **user-defined** acceptance threshold
  (``daytime_accept_qcf_below`` / ``nighttime_accept_qcf_below``, set in
  ``init_flux_data``).  This is the "accepted-quality" series that drives
  gap-filling and most downstream analysis.
- ``..._QCF0`` — the flux filtered at **strictly QCF=0** (all quality tests
  passed; no soft warnings tolerated).  This is the "highest-quality"
  reference series, used as the training target for ML gap-filling and
  anywhere you want to be conservative.

When ``accept_qcf_below=1`` the two contain identical values (both keep only
QCF=0), but the column names still differ to preserve the *intent* — re-run
with ``accept_qcf_below=2`` and they diverge without renaming.  The same
distinction shows up on ``LevelResults`` fields: ``filteredseries_*_qcf``
(user-accepted) vs ``filteredseries_hq`` / ``filteredseries_level33_hq``
(strictly QCF=0, "hq" = high-quality).

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.fluxprocessingchain.container import (
    DEFAULT_LEVEL2_TEST_SETTINGS,
    FluxConfig,
    FluxLevelData,
    FluxMeta,
    LevelResults,
    add_driver,
)
from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
from diive.flux.fluxprocessingchain.levels import (
    VM97_SUBTESTS,
    init_flux_data,
    level2_test_inputs,
    level31_storage_col,
    make_level32_detector,
    make_level41_engineer,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level33_ustar_detection,
    run_level33_variable_ustar,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
    run_level42_daytime_oneflux,
    run_level42_daytime_reddyproc,
    run_level42_nighttime_oneflux,
    run_level42_nighttime_reddyproc,
)
from diive.flux.fluxprocessingchain.run_chain import run_chain
from diive.flux.fluxprocessingchain.codegen import (
    chain_to_code, level2_to_code, level31_to_code, level32_to_code, level33_to_code,
    level41_to_code, level42_to_code,
)

__all__ = [
    # Containers / config
    'DEFAULT_LEVEL2_TEST_SETTINGS',
    'FluxConfig',
    'FluxLevelData',
    'FluxMeta',
    'LevelResults',
    # Single-call convenience driver
    'run_chain',
    # Reproducible-script codegen
    'chain_to_code',
    'level2_to_code',
    'level31_to_code',
    'level32_to_code',
    'level33_to_code',
    'level41_to_code',
    'level42_to_code',
    # Composable callables
    'init_flux_data',
    'add_driver',
    'run_level2',
    'VM97_SUBTESTS',
    'level2_test_inputs',
    'run_level31',
    'level31_storage_col',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level33_ustar_detection',
    'run_level33_variable_ustar',
    'make_level41_engineer',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
    'run_level42_nighttime_oneflux',
    'run_level42_nighttime_reddyproc',
    'run_level42_daytime_reddyproc',
    'run_level42_daytime_oneflux',
    # Level classes (for type-checking downstream)
    'FluxQualityFlagsEddyPro',
    'FluxStorageCorrectionSinglePointEddyPro',
]
