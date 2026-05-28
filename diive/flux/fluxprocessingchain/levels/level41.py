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

from diive.core.utils.console import rule
from diive.flux.fluxprocessingchain.container import FluxLevelData

if TYPE_CHECKING:
    from diive.core.ml.feature_engineer import FeatureEngineer


_LEVEL41_IDSTR = 'L4.1'


def _append_level_id(level_ids: list[str]) -> list[str]:
    new = list(level_ids)
    if _LEVEL41_IDSTR not in new:
        new.append(_LEVEL41_IDSTR)
    return new


def _warn_scenario_overwrite(existing: dict, new: dict, method_label: str) -> None:
    """Warn when a re-run would silently replace previously-stored scenario results.

    Each ``run_level41_*`` call stores its per-scenario instance in
    ``data.levels.level41_<method>[scen]``. Calling the same method twice with
    overlapping scenario labels (e.g. hyperparameter sweeps) silently drops the
    prior instance via the standard ``{**existing, **new}`` dict-merge pattern,
    making earlier runs unrecoverable. Warn so the user knows it happened.
    """
    import warnings
    overlap = sorted(set(existing).intersection(new))
    if overlap:
        warnings.warn(
            f"run_level41_{method_label}() is replacing previously-stored "
            f"scenario result(s) for: {overlap}. The earlier instance(s) for "
            f"these scenarios will be lost. To keep both runs, give the "
            f"USTAR scenarios distinct labels (e.g. add a suffix at L3.3) or "
            f"copy data.levels.level41_{method_label} before re-running.",
            UserWarning,
            stacklevel=3,
        )


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
        swin: Column name for shortwave incoming radiation (**W m-2**).
            Must exist in ``data.full_df`` (the original EddyPro input).
        ta: Column name for air temperature (**deg C**).
            Must exist in ``data.full_df``.
        vpd: Column name for vapour pressure deficit (**kPa**).
            Must exist in ``data.full_df``.
            EddyPro outputs VPD in hPa — divide by 10 before passing here.
            A ``UserWarning`` is raised automatically if the median looks like hPa.
        swin_tol: Tolerance window for radiation matching [absolute, relative].
        ta_tol: Temperature tolerance (deg C).
        vpd_tol: VPD tolerance (kPa).
        avg_min_n_vals: Minimum number of values for averaging.

    Returns:
        Updated FluxLevelData; MDS instances accessible via
        ``data.levels.level41_mds[ustar_scenario]``.
    """
    from diive.gapfilling.mds import FluxMDS
    import warnings

    # Validate that all three driver columns exist in full_df before the loop.
    missing_drivers = [c for c in (swin, ta, vpd) if c not in data.full_df.columns]
    if missing_drivers:
        raise KeyError(
            f"MDS driver column(s) not found in data.full_df: {missing_drivers}. "
            f"Driver columns must exist in the original input DataFrame (data.full_df), "
            f"not only in data.fpc_df. "
            f"Available columns: {list(data.full_df.columns)}"
        )

    # Sanity-check driver units. MDS uses absolute tolerances on raw values,
    # so wrong units silently break similarity matching without raising —
    # warn loudly when medians fall outside the physical range expected for
    # the documented units.

    # VPD must be kPa (typical 0.05–3). EddyPro outputs hPa (10x larger).
    vpd_median = data.full_df[vpd].median()
    if vpd_median > 10:
        warnings.warn(
            f"VPD column '{vpd}' has a median of {vpd_median:.1f} — this looks "
            f"like hPa, but MDS requires kPa (typical daytime values: 0.5-3 kPa). "
            f"Divide your VPD column by 10 before calling run_level41_mds().",
            UserWarning,
            stacklevel=2,
        )

    # TA must be degrees Celsius (typical -30..+40). A median > 100 is almost
    # certainly Kelvin (typical median ~283); a median > 50 is implausible.
    ta_median = data.full_df[ta].median()
    if ta_median > 100:
        warnings.warn(
            f"TA column '{ta}' has a median of {ta_median:.1f} - this looks "
            f"like Kelvin, but MDS requires degrees Celsius (typical values "
            f"-30..+40 deg C). Subtract 273.15 before calling run_level41_mds().",
            UserWarning,
            stacklevel=2,
        )
    elif ta_median > 50:
        warnings.warn(
            f"TA column '{ta}' has a median of {ta_median:.1f} deg C - this is "
            f"outside the plausible range for air temperature. Check the unit "
            f"and column choice before calling run_level41_mds().",
            UserWarning,
            stacklevel=2,
        )

    # SW_IN must be W m-2 (overall median typically 50-300; nighttime zeros pull
    # it down). A median > 2000 is implausible for half-hourly SW_IN in W m-2.
    swin_median = data.full_df[swin].median()
    if swin_median > 2000:
        warnings.warn(
            f"SW_IN column '{swin}' has a median of {swin_median:.0f} - this is "
            f"implausibly high for shortwave-in (W m-2). Check unit and column "
            f"choice before calling run_level41_mds().",
            UserWarning,
            stacklevel=2,
        )

    rule("Level 4.1: Gap-Filling (MDS)")

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

    _warn_scenario_overwrite(data.levels.level41_mds, mds_results, 'mds')
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
    # Validate feature columns exist in full_df before doing any work.
    missing_features = [c for c in features if c not in data.full_df.columns]
    if missing_features:
        raise KeyError(
            f"Feature column(s) not found in data.full_df: {missing_features}. "
            f"Feature columns must exist in the original input DataFrame (data.full_df), "
            f"not only in data.fpc_df. "
            f"Available columns: {list(data.full_df.columns)}"
        )

    filteredseries_l33 = _require_level33(data)
    fpc_df = data.fpc_df.copy()
    ml_results: dict = {}

    # Feature engineering does not depend on the target (USTAR scenario flux),
    # so run it once and reuse the result across all scenarios.
    engineered = engineer.fit_transform(data.full_df[features].copy())

    for ustar_scen, ustar_flux in filteredseries_l33.items():
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
        if flag_col is None:
            candidates = [c for c in instance.gapfilling_df_.columns
                          if str(c).startswith("FLAG_")]
            raise RuntimeError(
                f"Could not identify a unique FLAG_*_ISFILLED column for USTAR "
                f"scenario {ustar_scen!r}. Found FLAG_ columns: {candidates}. "
                f"Expected exactly one column matching the pattern "
                f"FLAG_*_ISFILLED in the gap-filling model's gapfilling_df_. "
                f"If you supplied a custom model_factory, check that it produces "
                f"this column; otherwise please open an issue."
            )
        fpc_df = pd.concat(
            [fpc_df, instance.gapfilled_.copy(), instance.gapfilling_df_[flag_col]],
            axis=1,
        )
        ml_results[ustar_scen] = instance

    existing = getattr(data.levels, results_attr)
    # results_attr is 'level41_rf' or 'level41_xgb'; method_label is the suffix.
    _warn_scenario_overwrite(existing, ml_results, results_attr.removeprefix('level41_'))
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
        features: Column names to use as predictor variables, resolved from
            ``data.full_df`` (the original EddyPro input DataFrame).  Columns
            added only to ``data.fpc_df`` are not available here.  Typical
            NEE drivers: air temperature (``TA_1_1_1``), shortwave radiation
            (``SW_IN_1_1_1``), VPD, relative humidity, soil temperature,
            and PPFD.  The more complete and gap-free these columns are, the
            better the gap-filling coverage.
        engineer: Pre-configured ``FeatureEngineer`` instance.  See
            ``diive.core.ml.feature_engineer.FeatureEngineer`` for the full
            list of feature-engineering parameters.  Feature engineering is
            applied once and reused for all USTAR scenarios.

            ``FeatureEngineer`` requires a ``target_col`` argument, but for
            L4.1 gap-filling the value does not matter — pass any string that
            is not in your feature list (e.g. ``'_target_'``).  The engineered
            features are computed from the predictor columns only::

                from diive.core.ml.feature_engineer import FeatureEngineer
                engineer = FeatureEngineer(
                    target_col='_target_',   # placeholder; value irrelevant here
                    features_lag=True,
                    features_rolling=True,
                )

        reduce_features: Apply SHAP-based feature selection across all years.
        verbose: Verbosity level (0=silent, 1+=progress).
        **rf_kwargs: sklearn ``RandomForestRegressor`` hyperparameters
            (``n_estimators``, ``max_depth``, ``min_samples_split``, ...).

    Returns:
        Updated FluxLevelData; RF instances accessible via
        ``data.levels.level41_rf[ustar_scenario]``.
        Use ``data.gapfilled_cols()`` to retrieve gap-filled column names.
    """
    rule("Level 4.1: Gap-Filling (Random Forest)")
    from diive.gapfilling.longterm import LongTermGapFillingRandomForestTS
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
        features: Column names to use as predictor variables, resolved from
            ``data.full_df`` (the original EddyPro input DataFrame).  Columns
            added only to ``data.fpc_df`` are not available here.  Typical
            NEE drivers: air temperature (``TA_1_1_1``), shortwave radiation
            (``SW_IN_1_1_1``), VPD, relative humidity, soil temperature,
            and PPFD.  Feature engineering is applied once and reused for all
            USTAR scenarios.
        engineer: Pre-configured ``FeatureEngineer`` instance.  Pass
            ``target_col='_target_'`` (any placeholder); the value is
            irrelevant for L4.1 feature engineering (see ``run_level41_rf``
            for a usage example).
        reduce_features: Apply SHAP-based feature selection across all years.
        verbose: Verbosity level (0=silent, 1+=progress).
        **xgb_kwargs: ``XGBRegressor`` hyperparameters (``n_estimators``,
            ``max_depth``, ``learning_rate``, ``early_stopping_rounds``, ...).

    Returns:
        Updated FluxLevelData; XGBoost instances accessible via
        ``data.levels.level41_xgb[ustar_scenario]``.
        Use ``data.gapfilled_cols()`` to retrieve gap-filled column names.
    """
    rule("Level 4.1: Gap-Filling (XGBoost)")
    from diive.gapfilling.longterm import LongTermGapFillingXGBoostTS
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
