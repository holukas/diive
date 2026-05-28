"""
RUN CHAIN: SINGLE-CALL FLUX PROCESSING DRIVER
==============================================

Convenience function that drives the full Swiss FluxNet processing chain
(L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1) from one ``FluxConfig``.

For custom L3.2 outlier pipelines or custom feature engineering, drop down to
the composable per-level API documented in the package docstring.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from diive.core.utils.console import rule
from diive.flux.fluxprocessingchain.container import FluxConfig, FluxLevelData
from diive.flux.fluxprocessingchain.levels import (
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


def run_chain(data: FluxLevelData, config: FluxConfig) -> FluxLevelData:
    """Drive the full processing chain (L2 -> L3.1 -> L3.2 -> L3.3 -> L4.1).

    A single-call convenience wrapper for the standard FLUXNET-style workflow.
    Each step is the same composable callable you would otherwise call by hand;
    this function only routes ``FluxConfig`` fields to their corresponding
    per-level arguments.

    L3.2 (Hampel outlier detection) runs when ``config.run_l32=True`` (the
    default) using ``config.outlier_sigma_daytime`` / ``..._nighttime``. Set
    ``run_l32=False`` for fluxes / sites where Hampel-style outlier removal
    is not appropriate (e.g. manually screened inputs). For multi-step
    outlier pipelines (Hampel + z-score rolling, manual removal, abslim, ...)
    use the composable API directly.

    L3.3 (USTAR filtering) supports two modes via ``config.ustar_detection_mode``:

    - ``'constant'`` (default) — apply pre-computed thresholds from
      ``config.ustar_thresholds`` (e.g. produced by REddyProc externally).
      Fastest.
    - ``'bootstrap'`` — detect thresholds from the data via multi-year
      bootstrap (the FLUXNET-standard workflow, Papale et al. 2006).
      Requires ``config.ustar_bootstrap_ta_col`` and
      ``config.ustar_bootstrap_swin_col``; thresholds and ``CUT_<p>``
      labels are generated from ``config.ustar_bootstrap_percentiles``
      (default ``(16, 50, 84)``). The fitted
      :class:`UstarBootstrapThresholds` is attached to
      ``data.levels.ustar_detection`` for post-hoc inspection.

    L4.1 runs only the methods whose ``config.gapfill_*`` flag is ``True``.
    The ML methods (RF / XGBoost) are built with a minimal default
    ``FeatureEngineer`` (lag, rolling, timestamps).  For custom feature
    engineering, use ``run_level41_rf`` / ``run_level41_xgb`` directly.

    **Contextual field validation:** ``FluxConfig`` makes most fields
    optional. ``run_chain`` validates that each enabled feature has the
    fields it needs (``outlier_sigma_*`` when ``run_l32=True``,
    ``mds_swin`` / ``mds_ta`` / ``mds_vpd`` when ``gapfill_mds=True``,
    ``gapfilling_features`` when ``gapfill_rf`` or ``gapfill_xgb`` is
    ``True``) and raises a single ``ValueError`` listing every missing field.

    Args:
        data: Initial container from ``init_flux_data()``.
        config: Per-flux configuration object.

    Returns:
        New ``FluxLevelData`` with all requested levels populated.

    Raises:
        ValueError: If ``config.fluxcol`` does not match
            ``data.meta.fluxcol``, or if any enabled feature is missing the
            fields it needs (see contextual-validation note above).

    Example::

        from diive.flux.fluxprocessingchain import (
            FluxConfig, init_flux_data, run_chain,
        )

        cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.18],
            ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5,
            outlier_sigma_nighttime=5.5,
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa_1_1_1'],
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa_1_1_1',
        )
        data = init_flux_data(df, fluxcol='FC',
                              site_lat=46.6, site_lon=9.8, utc_offset=1)
        data = run_chain(data, cfg)
    """
    if data.meta.fluxcol != config.fluxcol:
        raise ValueError(
            f"data.meta.fluxcol={data.meta.fluxcol!r} does not match "
            f"config.fluxcol={config.fluxcol!r}. Build the FluxLevelData with "
            f"the same fluxcol as the FluxConfig."
        )

    # Contextual validation: each conditional field is required only when its
    # enabling flag is on. Catch all missing fields up front so the caller
    # gets a single clear error rather than a half-completed pipeline.
    if config.ustar_detection_mode not in ('constant', 'bootstrap'):
        raise ValueError(
            f"ustar_detection_mode must be 'constant' or 'bootstrap'; "
            f"got {config.ustar_detection_mode!r}."
        )
    _missing: list[str] = []
    if config.ustar_detection_mode == 'constant':
        if not config.ustar_thresholds:
            _missing.append(
                "ustar_thresholds (ustar_detection_mode='constant')"
            )
    else:  # 'bootstrap'
        if not config.ustar_bootstrap_ta_col:
            _missing.append(
                "ustar_bootstrap_ta_col (ustar_detection_mode='bootstrap')"
            )
        if not config.ustar_bootstrap_swin_col:
            _missing.append(
                "ustar_bootstrap_swin_col (ustar_detection_mode='bootstrap')"
            )
    if config.run_l32:
        if config.outlier_sigma_daytime is None:
            _missing.append("outlier_sigma_daytime (run_l32=True)")
        if config.outlier_sigma_nighttime is None:
            _missing.append("outlier_sigma_nighttime (run_l32=True)")
    if config.gapfill_mds:
        if not (config.mds_swin and config.mds_ta and config.mds_vpd):
            _missing.append("mds_swin / mds_ta / mds_vpd (gapfill_mds=True)")
    if config.gapfill_rf or config.gapfill_xgb:
        if not config.gapfilling_features:
            _missing.append(
                "gapfilling_features (gapfill_rf=True or gapfill_xgb=True)"
            )
    if _missing:
        raise ValueError(
            "FluxConfig is missing field(s) required by the enabled features:\n"
            + "\n".join(f"  - {m}" for m in _missing)
            + "\nEither set them on the FluxConfig, or disable the matching "
              "flag (run_l32=False / gapfill_mds=False / "
              "gapfill_rf=False / gapfill_xgb=False)."
        )

    rule(f"run_chain: full pipeline for {config.fluxcol}")

    # ---------------------------------------------------------------- Level-2
    data = run_level2(data, **(config.level2_test_settings or {}))

    # -------------------------------------------------------------- Level-3.1
    data = run_level31(
        data,
        gapfill_storage_term=config.gapfill_storage_term,
        set_storage_to_zero=config.set_storage_to_zero,
    )

    # -------------------------------------------------------------- Level-3.2
    # Default detector: Hampel with separate day/night sigmas. Skipped when
    # run_l32=False — useful for fluxes where Hampel-style outlier removal
    # isn't appropriate (e.g. manually screened inputs).
    if config.run_l32:
        sod = make_level32_detector(data)
        sod.flag_outliers_hampel_test(
            window_length=config.outlier_window_length,
            n_sigma_daytime=config.outlier_sigma_daytime,
            n_sigma_nighttime=config.outlier_sigma_nighttime,
            separate_daytime_nighttime=True,
            showplot=False,
            verbose=False,
        )
        sod.addflag()
        data = run_level32(data, outlier_detector=sod)

    # -------------------------------------------------------------- Level-3.3
    if config.ustar_detection_mode == 'constant':
        data = run_level33_constant_ustar(
            data,
            thresholds=list(config.ustar_thresholds),
            threshold_labels=(list(config.ustar_labels)
                              if config.ustar_labels is not None else None),
            showplot=False,
            verbose=False,
        )
    else:  # 'bootstrap' — detect thresholds from the data via multi-year
        # bootstrap. Slower than constant mode but follows the FLUXNET-standard
        # workflow. The fitted UstarBootstrapThresholds is attached to
        # data.levels.ustar_detection for post-hoc inspection.
        data = run_level33_ustar_detection(
            data,
            ta_col=config.ustar_bootstrap_ta_col,
            swin_col=config.ustar_bootstrap_swin_col,
            n_iter=config.ustar_bootstrap_n_iter,
            n_jobs=config.ustar_bootstrap_n_jobs,
            percentiles=tuple(config.ustar_bootstrap_percentiles),
            showplot=False,
            verbose=False,
        )

    # -------------------------------------------------------------- Level-4.1
    if config.gapfill_mds:
        data = run_level41_mds(
            data,
            swin=config.mds_swin,
            ta=config.mds_ta,
            vpd=config.mds_vpd,
        )

    if config.gapfill_rf or config.gapfill_xgb:
        engineer = _default_engineer(config.gapfilling_features)
        if config.gapfill_rf:
            data = run_level41_rf(
                data,
                features=list(config.gapfilling_features),
                engineer=engineer,
            )
        if config.gapfill_xgb:
            data = run_level41_xgb(
                data,
                features=list(config.gapfilling_features),
                engineer=engineer,
            )

    return data


def _default_engineer(features: list[str]):
    """Build a minimal FeatureEngineer for run_chain's ML gap-filling.

    Stages enabled: lag (-1..-1), rolling (4, 12, 48), vectorized timestamps.
    All other 8-stage knobs are off. For richer feature engineering use the
    composable run_level41_rf / run_level41_xgb directly.
    """
    from diive.core.ml.feature_engineer import FeatureEngineer
    return FeatureEngineer(
        target_col='_target_',  # placeholder; ignored by run_level41_*
        features_lag=[-1, -1],
        features_lag_stepsize=1,
        features_rolling=[4, 12, 48],
        features_rolling_stats=['median', 'std'],
        vectorize_timestamps=True,
    )
