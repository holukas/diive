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

    The L3.2 step builds a default Hampel detector with separate day/night
    sigmas (``config.outlier_sigma_daytime`` / ``..._nighttime``) and runs it
    as a single test.  For multi-step outlier pipelines (Hampel + z-score
    rolling, manual removal, abslim, ...) use the composable API directly.

    L4.1 runs only the methods whose ``config.gapfill_*`` flag is ``True``.
    The ML methods (RF / XGBoost) are built with a minimal default
    ``FeatureEngineer`` (lag, rolling, timestamps).  For custom feature
    engineering, use ``run_level41_rf`` / ``run_level41_xgb`` directly.

    Args:
        data: Initial container from ``init_flux_data()``.
        config: Per-flux configuration object.

    Returns:
        New ``FluxLevelData`` with all requested levels populated.

    Raises:
        ValueError: If MDS is enabled but ``mds_swin`` / ``mds_ta`` /
            ``mds_vpd`` are not set, or if RF/XGBoost is enabled but
            ``gapfilling_features`` is empty.

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
            level2_tests={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
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

    rule(f"run_chain: full pipeline for {config.fluxcol}")

    # ---------------------------------------------------------------- Level-2
    data = run_level2(data, **(config.level2_tests or {}))

    # -------------------------------------------------------------- Level-3.1
    data = run_level31(data, set_storage_to_zero=config.set_storage_to_zero)

    # -------------------------------------------------------------- Level-3.2
    # Default detector: Hampel with separate day/night sigmas. Users who want a
    # different outlier pipeline should not use run_chain.
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
    data = run_level33_constant_ustar(
        data,
        thresholds=list(config.ustar_thresholds),
        threshold_labels=list(config.ustar_labels),
        showplot=False,
        verbose=False,
    )

    # -------------------------------------------------------------- Level-4.1
    if config.gapfill_mds:
        if not (config.mds_swin and config.mds_ta and config.mds_vpd):
            raise ValueError(
                "gapfill_mds=True requires mds_swin, mds_ta, and mds_vpd to be "
                "set on the FluxConfig."
            )
        data = run_level41_mds(
            data,
            swin=config.mds_swin,
            ta=config.mds_ta,
            vpd=config.mds_vpd,
        )

    if config.gapfill_rf or config.gapfill_xgb:
        if not config.gapfilling_features:
            raise ValueError(
                "gapfill_rf/gapfill_xgb=True requires gapfilling_features to "
                "be non-empty on the FluxConfig."
            )
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
