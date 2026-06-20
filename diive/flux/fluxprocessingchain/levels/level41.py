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

from diive.core.utils.console import detail, rule, warn
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._rerun import (
    drop_columns_for_key,
    record_added_columns,
)
from diive.flux.fluxprocessingchain.levels._shared import (
    append_level_id,
    assert_aligned_index,
    require_level33,
    warn_scenario_overwrite,
)

if TYPE_CHECKING:
    from diive.core.ml.feature_engineer import FeatureEngineer


_LEVEL41_IDSTR = 'L4.1'


def make_level41_engineer(
        data: FluxLevelData,
        features: list[str],
        **engineer_kwargs,
) -> 'FeatureEngineer':
    """Factory for a Level-4.1 :class:`FeatureEngineer` with sensible defaults.

    Symmetric with :func:`make_level32_detector`: validates that every
    requested feature column exists in ``data.full_df``, then builds a
    :class:`FeatureEngineer` pre-configured with the same defaults
    ``run_chain`` uses (symmetric ``[-2, 2]`` lag window, first- and
    second-order differencing, 4 / 12 / 48-record rolling median + std,
    vectorized timestamps). Any keyword argument supported by
    :class:`FeatureEngineer` can be passed in ``**engineer_kwargs`` to
    override the defaults — supply ``features_stl=True`` for STL
    decomposition, override ``features_lag``/``features_rolling``, disable
    ``vectorize_timestamps``, and so on.

    Usage::

        from diive.flux.fluxprocessingchain import (
            make_level41_engineer, run_level41_rf,
        )

        engineer = make_level41_engineer(
            data,
            features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
            # Optional overrides:
            features_lag=[-4, 4],          # wider symmetric window
            features_ema=[6, 12, 24, 48],  # add EMA stage
        )
        data = run_level41_rf(
            data,
            features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
            engineer=engineer,
            reduce_features=True,
        )

    Args:
        data: Current FluxLevelData. Used only to validate that ``features``
            are present in ``data.full_df``; the engineer itself is built
            from defaults + user overrides.
        features: Predictor column names. Every entry must exist in
            ``data.full_df``.
        **engineer_kwargs: Forwarded verbatim to
            :class:`FeatureEngineer`. Overrides the defaults documented
            above; see ``FeatureEngineer`` for the full 8-stage option set.

    Returns:
        Pre-configured ``FeatureEngineer`` ready to pass as the ``engineer=``
        argument of ``run_level41_rf`` or ``run_level41_xgb``.

    Raises:
        KeyError: If any feature is not in ``data.full_df``.
    """
    from diive.core.ml.feature_engineer import FeatureEngineer

    missing = [c for c in features if c not in data.full_df.columns]
    if missing:
        raise KeyError(
            f"Feature column(s) not found in data.full_df: {missing}. "
            f"Use add_driver() to register a computed driver, or check "
            f"the spelling. Available columns: {list(data.full_df.columns)}"
        )

    # Defaults match _default_engineer in run_chain (kept in sync). User
    # overrides via **engineer_kwargs win on any conflict.
    defaults = {
        'target_col': '_target_',
        'features_lag': [-2, 2],
        'features_lag_stepsize': 1,
        'features_diff': [1, 2],
        'features_rolling': [4, 12, 48],
        'features_rolling_stats': ['median', 'std'],
        'vectorize_timestamps': True,
    }
    defaults.update(engineer_kwargs)
    return FeatureEngineer(**defaults)


def _extract_gapfilling_flag_col(gapfilling_df: pd.DataFrame) -> str | None:
    """Find the single ``FLAG_*_ISFILLED`` column in a gap-filling DataFrame."""
    candidates = [c for c in gapfilling_df.columns
                  if str(c).startswith("FLAG_") and str(c).endswith("_ISFILLED")]
    return candidates[0] if len(candidates) == 1 else None


def run_level41_mds(
        data: FluxLevelData,
        *,
        swin: str,
        ta: str,
        vpd: str,
        swin_tol: list | None = None,
        ta_tol: float = 2.5,
        vpd_tol: float = 0.5,
        avg_min_n_vals: int = 2,
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

    # Driver coverage check: MDS fills a target record by averaging neighbours
    # whose driver values fall within tolerance. Gaps in TA / SW_IN / VPD
    # themselves therefore directly cap the achievable fill rate, but the
    # final gap-filled column gives no hint that the driver coverage is the
    # bottleneck. Log coverage upfront so a low fill rate downstream is
    # explainable rather than mysterious; warn when any driver is below 80%
    # because that is usually the dominant cause of poor MDS performance.
    n_total = len(data.full_df)
    for _drv_label, _drv_col in (('SW_IN', swin), ('TA', ta), ('VPD', vpd)):
        _n_valid = int(data.full_df[_drv_col].notna().sum())
        _coverage = _n_valid / n_total if n_total else 0.0
        _msg = (f"MDS driver coverage: {_drv_label}={_drv_col!r}  "
                f"{_n_valid}/{n_total} valid ({_coverage:.1%})")
        if _coverage < 0.80:
            warn(f"{_msg} - low driver coverage will cap the gap-fill rate "
                 f"regardless of model quality")
        else:
            detail(_msg)

    # Require L3.3 *before* mutating state. If we dropped columns first and
    # then raised, the caller's data would be left in a half-cleaned state
    # if they tried to recover from the exception. Check ordering matters.
    filteredseries_l33 = require_level33(data)

    # Re-run cleanup: drop the previous run's MDS columns from fpc_df so
    # re-running this method doesn't accumulate duplicates. L4.1 is additive
    # *across methods* (mds / rf / xgb live in independent buckets), so this
    # cleanup is per-method; rf and xgb columns are untouched.
    data = drop_columns_for_key(data, 'L4.1_mds')
    pre_columns = list(data.fpc_df.columns)
    fpc_df = data.fpc_df.copy()
    mds_results: dict = {}

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        scen_df = data.full_df[[swin, ta, vpd]].copy()
        assert_aligned_index(scen_df, fpc_df[ustar_flux.name],
                             context=f"run_level41_mds[{ustar_scen!r}] scen_df build")
        scen_df = pd.concat([scen_df, fpc_df[ustar_flux.name]], axis=1)

        instance = FluxMDS(
            df=scen_df,
            flux=ustar_flux.name,
            swin=swin, ta=ta, vpd=vpd,
            swin_tol=swin_tol, ta_tol=ta_tol, vpd_tol=vpd_tol,
            avg_min_n_vals=avg_min_n_vals,
        )
        instance.run()

        gapfilled = instance.get_gapfilled_target()
        flag = instance.get_flag()
        assert_aligned_index(fpc_df, gapfilled, flag,
                             context=f"run_level41_mds[{ustar_scen!r}] merge into fpc_df")
        fpc_df = pd.concat([fpc_df, gapfilled, flag], axis=1)
        mds_results[ustar_scen] = instance

    warn_scenario_overwrite(data.levels.level41_mds, mds_results,
                            fn_label='run_level41_mds', results_attr='level41_mds')
    new_levels = replace(data.levels, level41_mds={**data.levels.level41_mds, **mds_results})
    final = replace(
        data,
        fpc_df=fpc_df,
        levels=new_levels,
        level_ids=append_level_id(data.level_ids, _LEVEL41_IDSTR),
    )
    return replace(final, added_columns=record_added_columns(final, 'L4.1_mds', pre_columns))


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

    # Require L3.3 *before* mutating state — see the same guard in
    # run_level41_mds. Dropping columns first and then raising would leave
    # the caller's data in a half-cleaned state on a recovery path.
    filteredseries_l33 = require_level33(data)

    # Re-run cleanup: drop this method's previous columns from fpc_df. The
    # tracking key is e.g. 'L4.1_rf' for results_attr='level41_rf'.
    method_label = results_attr.removeprefix('level41_')
    tracking_key = f'L4.1_{method_label}'
    data = drop_columns_for_key(data, tracking_key)
    pre_columns = list(data.fpc_df.columns)
    fpc_df = data.fpc_df.copy()
    ml_results: dict = {}

    # Feature engineering does not depend on the target (USTAR scenario flux),
    # so run it once and reuse the result across all scenarios.
    engineered = engineer.fit_transform(data.full_df[features].copy())

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        assert_aligned_index(engineered, fpc_df[ustar_flux.name],
                             context=f"_run_level41_ml[{ustar_scen!r}] scen_df build")
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
        gapfilled = instance.gapfilled_.copy()
        flag = instance.gapfilling_df_[flag_col]
        assert_aligned_index(fpc_df, gapfilled, flag,
                             context=f"_run_level41_ml[{ustar_scen!r}] merge into fpc_df")
        fpc_df = pd.concat([fpc_df, gapfilled, flag], axis=1)
        ml_results[ustar_scen] = instance

    existing = getattr(data.levels, results_attr)
    warn_scenario_overwrite(existing, ml_results,
                            fn_label=f'run_level41_{method_label}',
                            results_attr=results_attr)
    new_levels = replace(data.levels, **{results_attr: {**existing, **ml_results}})
    final = replace(
        data,
        fpc_df=fpc_df,
        levels=new_levels,
        level_ids=append_level_id(data.level_ids, _LEVEL41_IDSTR),
    )
    return replace(final, added_columns=record_added_columns(final, tracking_key, pre_columns))


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
            features are computed from the predictor columns only.  Each
            ``features_*`` parameter takes a list (or ``None`` to disable
            the stage); see :class:`FeatureEngineer` for the full 8-stage
            option set, or use :func:`make_level41_engineer` for a factory
            pre-configured with sensible 30-min defaults that you can
            override piecewise::

                from diive.core.ml.feature_engineer import FeatureEngineer
                engineer = FeatureEngineer(
                    target_col='_target_',   # placeholder; value irrelevant here
                    features_lag=[-2, 2],            # symmetric ±2-record lag window
                    features_rolling=[4, 12, 48],    # rolling windows (records)
                    features_rolling_stats=['median', 'std'],
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
