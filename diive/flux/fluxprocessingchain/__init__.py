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
    sod = make_level32_detector(data); sod.flag_outliers_hampel_test(...); sod.addflag()
    data = run_level32(data, outlier_detector=sod)
    data = run_level33_constant_ustar(data, thresholds=[0.18], threshold_labels=['CUT_50'])
    data = run_level41_mds(data, swin='SW_IN', ta='TA', vpd='VPD_kPa')

**Single-call convenience API** — one ``FluxConfig``, one ``run_chain`` call::

    from diive.flux.fluxprocessingchain import init_flux_data, run_chain, FluxConfig

    cfg = FluxConfig(fluxcol='FC', ustar_thresholds=[0.18], ustar_labels=['CUT_50'], ...)
    data = init_flux_data(df, fluxcol='FC', site_lat=..., site_lon=..., utc_offset=1)
    data = run_chain(data, cfg)

Use ``run_chain`` for standard FLUXNET-style processing; drop down to the
composable API when you need custom L3.2 outlier logic, custom feature
engineering, or fine control over per-level diagnostics.

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
``run_level41_*``              built object + ad-hoc kwargs  Feature engineering is itself an 8-stage
                                                             configuration; passing it as a built object
                                                             is correct.
============================== ============================= ===============================================

If you find the per-level variation distracting, use ``run_chain(data, config)``
— one ``FluxConfig`` covers all of them.

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
    FluxConfig,
    FluxLevelData,
    FluxMeta,
    LevelResults,
    add_driver,
)
from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
from diive.flux.fluxprocessingchain.levels import (
    init_flux_data,
    make_level32_detector,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level33_ustar_detection,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)
from diive.flux.fluxprocessingchain.run_chain import run_chain

__all__ = [
    # Containers / config
    'FluxConfig',
    'FluxLevelData',
    'FluxMeta',
    'LevelResults',
    # Single-call convenience driver
    'run_chain',
    # Composable callables
    'init_flux_data',
    'add_driver',
    'run_level2',
    'run_level31',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level33_ustar_detection',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
    # Level classes (for type-checking downstream)
    'FluxQualityFlagsEddyPro',
    'FluxStorageCorrectionSinglePointEddyPro',
]
