"""
LEVEL 4.1: GAP-FILLING
=======================

Composable callables for the three gap-filling methods supported by the
chain: MDS, long-term Random Forest, and long-term XGBoost.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import pandas as pd

from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData

if TYPE_CHECKING:
    from diive.core.ml.feature_engineer import FeatureEngineer


_LEVEL41_IDSTR = 'L4.1'


def _append_level_id(level_ids: list[str]) -> list[str]:
    new = list(level_ids)
    if _LEVEL41_IDSTR not in new:
        new.append(_LEVEL41_IDSTR)
    return new


def _extract_gapfilling_flag_col(gapfilling_df: pd.DataFrame) -> str | None:
    """Find the single ``FLAG_*_ISFILLED`` column in a gap-filling DataFrame."""
    candidates = [c for c in gapfilling_df.columns
                  if str(c).startswith("FLAG_") and str(c).endswith("_ISFILLED")]
    return candidates[0] if len(candidates) == 1 else None


def _require_level33(data: FluxLevelData) -> dict[str, pd.Series]:
    if not data.levels.filteredseries_level33_qcf:
        raise RuntimeError(
            "run_level33_constant_ustar() must be called before any "
            "run_level41_* function."
        )
    return data.levels.filteredseries_level33_qcf


def run_level41_mds(
        data: FluxLevelData,
        *,
        swin: str,
        ta: str,
        vpd: str,
        swin_tol: list | None = None,
        ta_tol: float = 2.5,
        vpd_tol: float = 0.5,
        avg_min_n_vals: int = 5,
) -> FluxLevelData:
    """
    Level-4.1: Gap-fill using Marginal Data Substitution (MDS).

    Runs one MDS instance per USTAR scenario found in
    ``data.levels.filteredseries_level33_qcf``.

    Args:
        data: FluxLevelData after ``run_level33_constant_ustar()``.
        swin: Column name for shortwave incoming radiation (W m-2).
        ta: Column name for air temperature (deg C).
        vpd: Column name for vapour pressure deficit (kPa).
        swin_tol: Tolerance window for radiation matching [absolute, relative].
        ta_tol: Temperature tolerance (deg C).
        vpd_tol: VPD tolerance (kPa).
        avg_min_n_vals: Minimum number of values for averaging.

    Returns:
        Updated FluxLevelData; MDS instances accessible via
        ``data.levels.level41_mds[ustar_scenario]``.
    """
    from diive.pkgs.gapfilling.mds import FluxMDS

    filteredseries_l33 = _require_level33(data)
    fpc_df = data.fpc_df.copy()
    mds_results: dict = {}

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        scen_df = data.full_df[[swin, ta, vpd]].copy()
        scen_df = pd.concat([scen_df, fpc_df[ustar_flux.name]], axis=1)

        instance = FluxMDS(
            df=scen_df,
            flux=ustar_flux.name,
            swin=swin, ta=ta, vpd=vpd,
            swin_tol=swin_tol, ta_tol=ta_tol, vpd_tol=vpd_tol,
            avg_min_n_vals=avg_min_n_vals,
        )
        instance.run()

        fpc_df = pd.concat([fpc_df, instance.get_gapfilled_target(), instance.get_flag()], axis=1)
        mds_results[ustar_scen] = instance

    new_levels = replace(data.levels, level41_mds={**data.levels.level41_mds, **mds_results})
    return replace(
        data,
        fpc_df=fpc_df,
        levels=new_levels,
        level_ids=_append_level_id(data.level_ids),
    )


def _run_level41_ml(
        data: FluxLevelData,
        *,
        model_factory: Callable,
        features: list[str],
        engineer: 'FeatureEngineer',
        reduce_features: bool,
        verbose: int,
        model_kwargs: dict,
        results_attr: str,
) -> FluxLevelData:
    """Internal: shared workflow for Random Forest and XGBoost L4.1 gap-filling."""
    filteredseries_l33 = _require_level33(data)
    fpc_df = data.fpc_df.copy()
    ml_results: dict = {}

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        scen_features = data.full_df[features].copy()
        engineered = engineer.fit_transform(scen_features)
        scen_df = pd.concat([engineered, fpc_df[ustar_flux.name]], axis=1).copy()

        instance = model_factory(
            input_df=scen_df,
            target_col=ustar_flux.name,
            verbose=verbose,
            **model_kwargs,
        )
        instance.create_yearpools()
        instance.initialize_yearly_models()
        if reduce_features:
            instance.reduce_features_across_years()
        instance.fillgaps()

        flag_col = _extract_gapfilling_flag_col(instance.gapfilling_df_)
        fpc_df = pd.concat(
            [fpc_df, instance.gapfilled_.copy(), instance.gapfilling_df_[flag_col]],
            axis=1,
        )
        ml_results[ustar_scen] = instance

    existing = getattr(data.levels, results_attr)
    new_levels = replace(data.levels, **{results_attr: {**existing, **ml_results}})
    return replace(
        data,
        fpc_df=fpc_df,
        levels=new_levels,
        level_ids=_append_level_id(data.level_ids),
    )


def run_level41_rf(
        data: FluxLevelData,
        *,
        features: list[str],
        engineer: 'FeatureEngineer',
        reduce_features: bool = False,
        verbose: int = 0,
        **rf_kwargs,
) -> FluxLevelData:
    """
    Level-4.1: Gap-fill using long-term Random Forest with feature engineering.

    Runs one model per USTAR scenario.  Feature engineering is performed once
    (via the supplied ``engineer``) and reused across all scenarios.

    Args:
        data: FluxLevelData after ``run_level33_constant_ustar()``.
        features: Input feature column names (resolved from ``data.full_df``).
        engineer: Pre-configured ``FeatureEngineer`` instance.  See
            ``diive.core.ml.feature_engineer.FeatureEngineer`` for the full
            list of feature-engineering parameters.
        reduce_features: Apply SHAP-based feature selection across all years.
        verbose: Verbosity level (0=silent, 1+=progress).
        **rf_kwargs: sklearn ``RandomForestRegressor`` hyperparameters
            (``n_estimators``, ``max_depth``, ``min_samples_split``, ...).

    Returns:
        Updated FluxLevelData; RF instances accessible via
        ``data.levels.level41_rf[ustar_scenario]``.
    """
    from diive.pkgs.gapfilling.longterm import LongTermGapFillingRandomForestTS
    return _run_level41_ml(
        data,
        model_factory=LongTermGapFillingRandomForestTS,
        features=features,
        engineer=engineer,
        reduce_features=reduce_features,
        verbose=verbose,
        model_kwargs=rf_kwargs,
        results_attr='level41_rf',
    )


def run_level41_xgb(
        data: FluxLevelData,
        *,
        features: list[str],
        engineer: 'FeatureEngineer',
        reduce_features: bool = False,
        verbose: int = 0,
        **xgb_kwargs,
) -> FluxLevelData:
    """
    Level-4.1: Gap-fill using long-term XGBoost with feature engineering.

    Runs one model per USTAR scenario.  Feature engineering is performed once
    (via the supplied ``engineer``) and reused across all scenarios.  XGBoost
    often outperforms Random Forest on non-linear patterns.

    Args:
        data: FluxLevelData after ``run_level33_constant_ustar()``.
        features: Input feature column names (resolved from ``data.full_df``).
        engineer: Pre-configured ``FeatureEngineer`` instance.
        reduce_features: Apply SHAP-based feature selection across all years.
        verbose: Verbosity level (0=silent, 1+=progress).
        **xgb_kwargs: ``XGBRegressor`` hyperparameters (``n_estimators``,
            ``max_depth``, ``learning_rate``, ``early_stopping_rounds``, ...).

    Returns:
        Updated FluxLevelData; XGBoost instances accessible via
        ``data.levels.level41_xgb[ustar_scenario]``.
    """
    from diive.pkgs.gapfilling.longterm import LongTermGapFillingXGBoostTS
    return _run_level41_ml(
        data,
        model_factory=LongTermGapFillingXGBoostTS,
        features=features,
        engineer=engineer,
        reduce_features=reduce_features,
        verbose=verbose,
        model_kwargs=xgb_kwargs,
        results_attr='level41_xgb',
    )
