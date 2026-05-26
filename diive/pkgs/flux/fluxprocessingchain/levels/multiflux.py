"""
MULTI-FLUX CHAIN
================

Convenience wrapper that runs the full L2→L4.1 pipeline for a single flux
from a :class:`~diive.pkgs.flux.fluxprocessingchain.container.FluxConfig`.

Use this when processing multiple fluxes (FC, H, LE, N2O, CH4, …) in a loop
so that site-level parameters are written once and per-flux settings are
captured in typed :class:`FluxConfig` objects.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from diive.pkgs.flux.fluxprocessingchain.container import FluxConfig, FluxLevelData
from diive.pkgs.flux.fluxprocessingchain.levels._init import init_flux_data
from diive.pkgs.flux.fluxprocessingchain.levels.level2 import run_level2
from diive.pkgs.flux.fluxprocessingchain.levels.level31 import run_level31
from diive.pkgs.flux.fluxprocessingchain.levels.level32 import make_level32_detector, run_level32
from diive.pkgs.flux.fluxprocessingchain.levels.level33 import run_level33_constant_ustar
from diive.pkgs.flux.fluxprocessingchain.levels.level41 import (
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)

if TYPE_CHECKING:
    import pandas as pd
    from diive.core.ml.feature_engineer import FeatureEngineer

# Production-ready defaults — override via rf_kwargs / xgb_kwargs / mds_kwargs
_RF_DEFAULTS: dict = dict(
    n_estimators=350, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
)
_XGB_DEFAULTS: dict = dict(
    n_estimators=350, max_depth=6, learning_rate=0.05,
    early_stopping_rounds=30, min_child_weight=5,
    random_state=42, n_jobs=-1,
)
_MDS_DEFAULTS: dict = dict(ta_tol=2.5, vpd_tol=0.5)


def run_flux_chain(
        df: 'pd.DataFrame',
        config: FluxConfig,
        *,
        site_lat: float,
        site_lon: float,
        utc_offset: int,
        nighttime_threshold: float = 20.0,
        daytime_accept_qcf_below: int = 2,
        nighttime_accept_qcf_below: int = 2,
        engineer: 'FeatureEngineer | None' = None,
        rf_kwargs: dict | None = None,
        xgb_kwargs: dict | None = None,
        mds_kwargs: dict | None = None,
        showplot: bool = False,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Run the full L2→L4.1 flux processing chain for one flux variable.

    Site-level parameters (coordinates, QCF thresholds) are passed as keyword
    arguments so they can be written once and shared across all
    :class:`FluxConfig` objects in a multi-flux loop::

        SITE = dict(site_lat=47.42, site_lon=9.84, utc_offset=1,
                    nighttime_threshold=20,
                    daytime_accept_qcf_below=2,
                    nighttime_accept_qcf_below=2)

        results: dict[str, FluxLevelData] = {}
        for cfg in [fc_cfg, h_cfg, n2o_cfg]:
            results[cfg.fluxcol] = run_flux_chain(df, cfg, **SITE,
                                                   engineer=engineer)

    The Hampel outlier filter is the only L3.2 method wired into this helper.
    If you need a different outlier method or a chain of methods, call the
    individual level functions (:func:`make_level32_detector`, :func:`run_level32`)
    directly instead of using ``run_flux_chain``.

    Args:
        df: Full input DataFrame (EddyPro output + meteorological variables).
        config: Per-flux settings (:class:`FluxConfig`).
        site_lat: Site latitude (decimal degrees, positive = N).
        site_lon: Site longitude (decimal degrees, positive = E).
        utc_offset: UTC offset in whole hours (e.g. 1 for CET).
        nighttime_threshold: SW_IN threshold below which it is nighttime (W m-2).
        daytime_accept_qcf_below: Accept daytime records with QCF < this value.
            2 = keep QCF 0 and 1 (FLUXNET/Swiss FluxNet convention).
        nighttime_accept_qcf_below: Accept nighttime records with QCF < this value.
        engineer: Pre-built :class:`~diive.core.ml.feature_engineer.FeatureEngineer`
            instance.  Required when ``config.gapfill_rf`` or
            ``config.gapfill_xgb`` is ``True``.  The same instance is reused
            for both ML methods — feature engineering runs only once.
        rf_kwargs: Override Random Forest hyperparameters (merged into defaults:
            ``n_estimators=350``, ``max_depth=15``).
        xgb_kwargs: Override XGBoost hyperparameters (merged into defaults:
            ``n_estimators=350``, ``max_depth=6``).
        mds_kwargs: Override MDS parameters (merged into defaults:
            ``ta_tol=2.5``, ``vpd_tol=0.5``).
        showplot: Show diagnostic plots at each level.
        verbose: Print progress messages.

    Returns:
        :class:`FluxLevelData` after all configured levels have run.
    """
    # ------------------------------------------------------------------ validate
    if (config.gapfill_rf or config.gapfill_xgb) and engineer is None:
        raise ValueError(
            f"engineer must be provided for flux '{config.fluxcol}' because "
            f"gapfill_rf={config.gapfill_rf} and/or gapfill_xgb={config.gapfill_xgb}. "
            f"Build a FeatureEngineer with your feature configuration and pass it here."
        )
    if config.gapfill_mds:
        missing = [k for k, v in
                   [('mds_swin', config.mds_swin), ('mds_ta', config.mds_ta),
                    ('mds_vpd', config.mds_vpd)]
                   if v is None]
        if missing:
            raise ValueError(
                f"FluxConfig for '{config.fluxcol}' has gapfill_mds=True but "
                f"{missing} are None.  Set them to the column names in your DataFrame."
            )

    # ------------------------------------------------------------------ L1: init
    data = init_flux_data(
        df=df,
        fluxcol=config.fluxcol,
        site_lat=site_lat,
        site_lon=site_lon,
        utc_offset=utc_offset,
        nighttime_threshold=nighttime_threshold,
        daytime_accept_qcf_below=daytime_accept_qcf_below,
        nighttime_accept_qcf_below=nighttime_accept_qcf_below,
    )

    # ------------------------------------------------------------------ L2
    data = run_level2(data, **config.level2_tests)

    # ------------------------------------------------------------------ L3.1
    data = run_level31(
        data,
        gapfill_storage_term=True,
        set_storage_to_zero=config.set_storage_to_zero,
    )

    # ------------------------------------------------------------------ L3.2
    sod = make_level32_detector(data)
    sod.flag_outliers_hampel_test(
        window_length=config.outlier_window_length,
        n_sigma_daytime=config.outlier_sigma_daytime,
        n_sigma_nighttime=config.outlier_sigma_nighttime,
        use_differencing=True,
        separate_daytime_nighttime=True,
        showplot=showplot,
        verbose=verbose,
        repeat=True,
    )
    sod.addflag()
    data = run_level32(data, outlier_detector=sod)

    # ------------------------------------------------------------------ L3.3
    data = run_level33_constant_ustar(
        data,
        thresholds=config.ustar_thresholds,
        threshold_labels=config.ustar_labels,
        showplot=showplot,
        verbose=verbose,
    )

    # ------------------------------------------------------------------ L4.1
    _rf_kw = {**_RF_DEFAULTS, **(rf_kwargs or {})}
    _xgb_kw = {**_XGB_DEFAULTS, **(xgb_kwargs or {})}
    _mds_kw = {**_MDS_DEFAULTS, **(mds_kwargs or {})}

    if config.gapfill_rf:
        data = run_level41_rf(
            data, features=config.gapfilling_features, engineer=engineer,
            reduce_features=True, verbose=1 if verbose else 0, **_rf_kw,
        )

    if config.gapfill_xgb:
        data = run_level41_xgb(
            data, features=config.gapfilling_features, engineer=engineer,
            reduce_features=True, verbose=1 if verbose else 0, **_xgb_kw,
        )

    if config.gapfill_mds:
        data = run_level41_mds(
            data,
            swin=config.mds_swin, ta=config.mds_ta, vpd=config.mds_vpd,
            **_mds_kw,
        )

    return data
