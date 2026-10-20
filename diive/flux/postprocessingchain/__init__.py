"""
FLUX POST-PROCESSING CHAIN: MULTI-LEVEL WORKFLOW
================================================

Swiss FluxNet multi-level post-processing: quality flag expansion (L2), storage correction (L3.1),
outlier removal (L3.2), USTAR filtering (L3.3), gap-filling (L4.1) and optional NEE partitioning (L4.2).

Two entry points
----------------

**Composable per-level API** — one callable per level, each returning a new
``FluxLevelData`` container::

    from diive.flux.postprocessingchain import (
        init_flux_data, run_level2, run_level31,
        make_level32_detector, run_level32,
        run_level33_constant_ustar,
        run_level41_mds, run_level41_rf, run_level41_xgb,
        run_level42_partitioning,
    )

    data = init_flux_data(df, fluxcol='FC', site_lat=..., site_lon=..., utc_offset=1)
    data = run_level2(data, ssitc={'apply': True, ...}, ...)
    data = run_level31(data)
    data, sod = make_level32_detector(data); sod.flag_outliers_hampel_test(...); sod.addflag()
    data = run_level32(data, outlier_detector=sod)
    data = run_level33_constant_ustar(data, thresholds=[0.18], threshold_labels=['CUT_50'])
    data = run_level41_mds(data, swin='SW_IN', ta='TA', vpd='VPD_kPa')
    data = run_level42_partitioning(data, method='nighttime', ta='TA', sw_in='SW_IN',
                                    ta_f='TA_F')

**Single-call convenience API** — one ``FluxConfig``, one ``run_chain`` call::

    from diive.flux.postprocessingchain import init_flux_data, run_chain, FluxConfig

    cfg = FluxConfig(fluxcol='FC', ustar_thresholds=[0.18], ustar_labels=['CUT_50'], ...)
    data = init_flux_data(df, fluxcol='FC', site_lat=..., site_lon=..., utc_offset=1)
    data = run_chain(data, cfg)

``run_chain`` covers the standard pipeline and uses fixed defaults for every
per-level setting that ``FluxConfig`` does not expose (see "What ``run_chain``
does *not* expose" in :func:`run_chain`). Every detector class, model
hyperparameter and diagnostic flag is reachable only through the composable
per-level callables.

Why the per-level signatures look different
-------------------------------------------

Each level's signature matches the *shape* of what it controls. One shape
forced onto every level would fit at least one of them badly, so the API uses
a few small patterns:

============================== ============================= ===============================================
Level                          Shape                         Why this shape
============================== ============================= ===============================================
``run_level2``                 per-test config dicts         N independent EddyPro tests, each with its own
                                                             ``apply`` / threshold / time-period settings.
                                                             Flat kwargs would explode the namespace.
``run_level31``                flat keyword arguments        Two binary decisions and an optional storage
                                                             column name. A config object would be
                                                             overkill.
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
``run_level42_*``              driver column names           Each variant names the measured and gap-filled
                                                             meteo columns it reads from ``full_df``.
                                                             ``run_level42_partitioning`` adds the
                                                             ``method`` / ``mode`` switches.
============================== ============================= ===============================================

``run_chain(data, config)`` hides these differences: one ``FluxConfig`` holds
the usual decisions (which L2 tests, which USTAR mode, which gap-filling
methods, which partitioning, which driver columns).

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
  reference series, for analyses that need the strictest filter. Gap-filling
  (L4.1) uses the ``..._QCF`` series.

With ``daytime_accept_qcf_below=1`` and ``nighttime_accept_qcf_below=1``
(the defaults) the two contain identical values (both keep only QCF=0), but
the column names still differ to preserve the *intent*. Re-run with ``2`` and
they diverge without renaming.  The same
distinction shows up on ``LevelResults`` fields: ``filteredseries_*_qcf``
(user-accepted) vs ``filteredseries_hq`` / ``filteredseries_level33_hq``
(strictly QCF=0, "hq" = high-quality).

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.postprocessingchain.container import (
    DEFAULT_LEVEL2_TEST_SETTINGS,
    FluxConfig,
    FluxLevelData,
    FluxMeta,
    LevelResults,
    add_driver,
)
from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
from diive.flux.postprocessingchain.levels import (
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
    run_level42_partitioning,
)
from diive.flux.postprocessingchain.run_chain import run_chain
from diive.flux.postprocessingchain.reports import (
    gapfilled_variables,
    plot_feature_ranks_per_year,
    plot_gapfilled_cumulative,
    plot_mds_gapfilling_qualities,
    report_gapfilling_feature_importances,
    report_gapfilling_model_scores,
    report_gapfilling_poolyears,
    report_gapfilling_variables,
    report_traintest_details,
    report_traintest_model_scores,
)
from diive.flux.postprocessingchain.codegen import (
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
    'run_level42_partitioning',
    # Reporting over a finished chain
    'report_gapfilling_variables',
    'report_gapfilling_model_scores',
    'report_traintest_model_scores',
    'report_traintest_details',
    'report_gapfilling_feature_importances',
    'report_gapfilling_poolyears',
    'gapfilled_variables',
    'plot_gapfilled_cumulative',
    'plot_feature_ranks_per_year',
    'plot_mds_gapfilling_qualities',
    # Level classes (for type-checking downstream)
    'FluxQualityFlagsEddyPro',
    'FluxStorageCorrectionSinglePointEddyPro',
]
