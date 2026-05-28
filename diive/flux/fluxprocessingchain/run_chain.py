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

    Single-call convenience wrapper for the standard FLUXNET-style workflow.
    Each step is the same composable callable you would otherwise call by hand;
    this function only routes ``FluxConfig`` fields to the per-level arguments
    it considers high-level enough to expose.

    **``run_chain`` is intentionally simple.** It picks fixed defaults for
    every per-level knob that is not on ``FluxConfig``. For full control over
    detectors, model hyperparameters, MDS tolerances, etc., call the
    per-level functions directly — they accept the full set of arguments that
    ``run_chain`` hides.

    Per-level behaviour
    -------------------

    *Level-2 (quality flags)* — runs the tests listed in
    ``config.level2_test_settings``; the always-on missing-values test runs
    regardless.

    *Level-3.1 (storage correction)* — controlled by
    ``config.set_storage_to_zero`` and ``config.gapfill_storage_term``.

    *Level-3.2 (outlier detection)* — **unconditional in ``run_chain``**
    because L3.3 USTAR filtering depends on outlier-screened data. Uses a
    single Hampel filter with separate day / night sigmas
    (``config.outlier_sigma_daytime`` / ``..._nighttime``) and the rolling
    window ``config.outlier_window_length``. All other Hampel knobs
    (``use_differencing``, ``repeat``, ``k``, ...) are fixed at the underlying
    function's defaults. If you have screened outliers upstream and genuinely
    want to skip L3.2, use the composable per-level API instead of
    ``run_chain``.

    *Level-3.3 (USTAR filtering)* — dispatches on
    ``config.ustar_detection_mode``:

    - ``'constant'`` (default) — apply pre-computed thresholds from
      ``config.ustar_thresholds`` (e.g. produced by REddyProc externally).
    - ``'bootstrap'`` — detect thresholds from the data via multi-year
      bootstrap (FLUXNET standard, Papale et al. 2006). Requires
      ``config.ustar_bootstrap_ta_col`` / ``..._swin_col``; thresholds and
      ``CUT_<p>`` labels are generated from
      ``config.ustar_bootstrap_percentiles`` (default ``(16, 50, 84)``). The
      fitted :class:`UstarBootstrapThresholds` is attached to
      ``data.levels.ustar_detection``.

    *Level-4.1 (gap-filling)* — runs only the methods whose
    ``config.gapfill_*`` flag is ``True``. MDS uses the underlying
    :class:`FluxMDS` defaults for tolerances and ``avg_min_n_vals``. RF and
    XGBoost are built with a minimal ``FeatureEngineer`` (lag, rolling,
    vectorized timestamps) and the model class's default hyperparameters;
    SHAP feature reduction is **off**.

    What ``run_chain`` does *not* expose
    ------------------------------------

    The following knobs are reachable only via the composable per-level API.
    If you need any of them, build the chain step by step with
    ``run_level2`` / ``run_level31`` / ``make_level32_detector`` +
    ``run_level32`` / ``run_level33_constant_ustar`` (or
    ``run_level33_ustar_detection``) / ``run_level41_mds`` /
    ``run_level41_rf`` / ``run_level41_xgb`` instead.

    - **L3.2** — non-Hampel detectors (z-score rolling, abslim, LocalSD,
      manual removal), multi-step pipelines (Hampel + something else with
      sequential ``addflag()``), and Hampel sub-options
      (``use_differencing``, ``repeat``, ``k``, ``window_length`` per call).
    - **L3.3** — diagnostic ``showplot=True`` and ``verbose=True`` (both
      forced to ``False`` here for non-interactive use); custom
      ``detector_class`` / ``detector_kwargs`` in bootstrap mode (defaults
      to ONEFlux moving-point).
    - **L4.1 MDS** — ``swin_tol`` / ``ta_tol`` / ``vpd_tol`` /
      ``avg_min_n_vals`` (fixed at :class:`FluxMDS` defaults).
    - **L4.1 RF / XGBoost** — model hyperparameters (``n_estimators``,
      ``max_depth``, ``learning_rate``, ...), the ``FeatureEngineer``
      itself (use ``engineer=`` on the per-level function for the full
      8-stage configuration), and SHAP feature reduction
      (``reduce_features=True``).

    Contextual field validation
    ---------------------------

    ``FluxConfig`` makes most fields optional and ``run_chain`` validates
    that each enabled feature has the fields it needs
    (``outlier_sigma_*`` when ``run_l32=True``, ``mds_swin`` / ``mds_ta`` /
    ``mds_vpd`` when ``gapfill_mds=True``, ``gapfilling_features`` when
    ``gapfill_rf`` or ``gapfill_xgb`` is ``True``,
    ``ustar_thresholds`` in ``'constant'`` mode,
    ``ustar_bootstrap_ta_col`` / ``..._swin_col`` in ``'bootstrap'`` mode).
    A single ``ValueError`` lists every missing field.

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
        # When the user supplies more than one threshold, requiring explicit
        # labels prevents the auto-generated CUT_0/CUT_1/... fallback from
        # silently labelling percentile-based thresholds with non-percentile
        # names (a real footgun for FLUXNET-style 16/50/84 workflows).
        elif len(config.ustar_thresholds) > 1 and not config.ustar_labels:
            _missing.append(
                "ustar_labels (required when ustar_thresholds has more than "
                "one entry, to avoid silently mislabelling percentile-based "
                "thresholds; e.g. ['CUT_16', 'CUT_50', 'CUT_84'])"
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
    # L3.2 is unconditional in run_chain because L3.3 USTAR filtering depends
    # on outlier-screened data — running USTAR detection on outlier-contaminated
    # records biases the threshold's effect and can spuriously reject good
    # nighttime flux. Users who genuinely want to skip L3.2 must drop to the
    # composable per-level API.
    if config.outlier_sigma_daytime is None:
        _missing.append("outlier_sigma_daytime (L3.2 is mandatory in run_chain)")
    if config.outlier_sigma_nighttime is None:
        _missing.append("outlier_sigma_nighttime (L3.2 is mandatory in run_chain)")
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
              "flag (gapfill_mds=False / gapfill_rf=False / gapfill_xgb=False). "
              "L3.2 cannot be disabled from run_chain — use the composable "
              "per-level API if you genuinely need to skip outlier removal."
        )

    # Validate that every driver / feature column the chain will read from
    # data.full_df actually exists there. The per-level functions perform the
    # same check, but failing here lets the user fix all column-name typos
    # in one round instead of after each level runs.
    _column_misses: list[str] = []
    if config.gapfill_mds:
        for _fld, _val in (('mds_swin', config.mds_swin),
                           ('mds_ta', config.mds_ta),
                           ('mds_vpd', config.mds_vpd)):
            if _val not in data.full_df.columns:
                _column_misses.append(f"{_fld}={_val!r}")
    if config.ustar_detection_mode == 'bootstrap':
        for _fld, _val in (('ustar_bootstrap_ta_col', config.ustar_bootstrap_ta_col),
                           ('ustar_bootstrap_swin_col', config.ustar_bootstrap_swin_col)):
            if _val not in data.full_df.columns:
                _column_misses.append(f"{_fld}={_val!r}")
    if config.gapfill_rf or config.gapfill_xgb:
        for _val in (config.gapfilling_features or ()):
            if _val not in data.full_df.columns:
                _column_misses.append(f"gapfilling_features entry {_val!r}")
    if _column_misses:
        raise KeyError(
            "FluxConfig references column(s) not present in data.full_df:\n"
            + "\n".join(f"  - {m}" for m in _column_misses)
            + f"\nAvailable columns: {list(data.full_df.columns)}"
        )

    # Warn (don't fail) on a pipeline that runs L2 with nothing but the
    # always-on missing-values test — easy to do accidentally, and produces
    # an L2 QCF that flags only fully-missing records.
    if not config.level2_test_settings:
        import warnings
        warnings.warn(
            "FluxConfig.level2_test_settings is empty — Level-2 will run only "
            "the always-on missing-values test, so the L2 QCF will accept "
            "essentially every record where the flux is non-NaN. This is "
            "almost certainly not what you want for a production chain; "
            "supply at least an SSITC test (e.g. "
            "level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}}).",
            UserWarning,
            stacklevel=2,
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
    # Single Hampel filter with separate day/night sigmas. Unconditional —
    # L3.3 USTAR filtering depends on outlier-screened data, so the chain
    # cannot skip this step. Users who need a non-Hampel detector, a
    # multi-step pipeline, or want to skip L3.2 entirely must call the
    # composable per-level API directly.
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
        # [-1, -1] is intentional, not a typo: FeatureEngineer treats the
        # list as [min_lag, max_lag] (inclusive on both ends), so this
        # produces a single lag feature at lag = -1 (one record into the
        # past). For multi-lag windows, use the composable API and pass a
        # built FeatureEngineer with e.g. features_lag=[-2, 2].
        features_lag=[-1, -1],
        features_lag_stepsize=1,
        features_rolling=[4, 12, 48],
        features_rolling_stats=['median', 'std'],
        vectorize_timestamps=True,
    )
