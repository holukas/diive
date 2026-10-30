"""
LEVEL 4.1: GAP-FILLING
=======================

Composable callables for the three gap-filling methods supported by the
chain: MDS, long-term Random Forest, and long-term XGBoost. After MDS,
``run_level41_ustar_reference`` selects the ONEFlux REF u* scenario.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import pandas as pd

from diive.core.utils.console import detail, info, rule, warn
from diive.flux.postprocessingchain.container import FluxLevelData
from diive.flux.postprocessingchain.levels._rerun import (
    clear_ustar_reference,
    drop_columns_for_key,
    record_added_columns,
)
from diive.flux.postprocessingchain.levels._shared import (
    append_level_id,
    assert_aligned_index,
    middle_scenario,
    require_level33,
    select_scenarios,
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

        from diive.flux.postprocessingchain import (
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
        mode: str = 'diive',
        fill_long_gaps: bool | None = None,
        swin_qc: str | None = None,
        ta_qc: str | None = None,
        vpd_qc: str | None = None,
) -> FluxLevelData:
    """
    Level-4.1: Gap-fill using Marginal Distribution Sampling (MDS).

    Runs one MDS instance per USTAR scenario found in
    ``data.levels.filteredseries_level33_qcf``.

    With ``fill_long_gaps=False`` (the default of ``mode='oneflux'``) the
    gap-filled columns in ``fpc_df`` lose the long gaps, as in the FLUXNET
    product of ONEFlux (see :class:`diive.gapfilling.mds.FluxMDS`). As in
    ONEFlux, the later steps work on the filled gaps: the REF selection
    (``run_level41_ustar_reference``) and the partitioning (Level 4.2) take
    the gap-filled NEE with the long gaps filled from the MDS instances
    (``get_gapfilled_target(full=True)``), and only their NEE, GPP and RECO
    output columns lose the long gaps of the matching scenario.

    Args:
        data: FluxLevelData after ``run_level33_constant_ustar()``.
        swin: Column name for shortwave incoming radiation (**W m-2**).
            Must exist in ``data.full_df`` (the original EddyPro input).
        ta: Column name for air temperature (**deg C**).
            Must exist in ``data.full_df``.
        vpd: Column name for vapour pressure deficit (**kPa**).
            Must exist in ``data.full_df``.
        swin_tol: SWIN tolerance clamp ``[min, max]`` in W m-2 (default
            ``[20, 50]``): a record's tolerance is its own SWIN clamped into
            this range.
        ta_tol: Temperature tolerance (deg C).
        vpd_tol: VPD tolerance (kPa).
        avg_min_n_vals: Minimum number of values for averaging.
        mode: ``'diive'`` (default, Vekuri et al. 2023 symmetric mean in
            daytime), ``'oneflux'`` (bit for bit as ONEFlux) or
            ``'reddyproc'`` (as REddyProc).
            See :class:`diive.gapfilling.mds.FluxMDS`.
        fill_long_gaps: Keep the long gaps filled (True) or set them missing
            as ONEFlux does in its FLUXNET product since 2025 (False). None
            (default) means False for ``mode='oneflux'`` and True for the
            other modes. See :class:`diive.gapfilling.mds.FluxMDS`.
        swin_qc, ta_qc, vpd_qc: Optional QC flag column of each driver
            (0 = measured, above 0 = gap-filled, e.g. ``FLAG_..._ISFILLED``
            or FLUXNET ``*_F_QC``), in ``data.full_df``. With gap-filled
            drivers this is how ONEFlux gap-fills NEE: only measured driver
            values serve as similar samples. See
            :class:`diive.gapfilling.mds.FluxMDS`.

    Returns:
        Updated FluxLevelData; MDS instances accessible via
        ``data.levels.level41_mds[ustar_scenario]``.
    """
    from diive.gapfilling.mds import FluxMDS

    # Validate that the driver columns (and any QC columns) exist in full_df before the loop.
    qc_cols = [c for c in (swin_qc, ta_qc, vpd_qc) if c is not None]
    missing_drivers = [c for c in (swin, ta, vpd, *qc_cols) if c not in data.full_df.columns]
    if missing_drivers:
        raise KeyError(
            f"MDS driver column(s) not found in data.full_df: {missing_drivers}. "
            f"Driver columns must exist in the original input DataFrame (data.full_df), "
            f"not only in data.fpc_df. "
            f"Available columns: {list(data.full_df.columns)}"
        )

    rule("Level 4.1: Gap-Filling (MDS)")

    # Driver coverage check: MDS fills a target record by averaging neighbours
    # whose driver values fall within tolerance. Where TA / SW_IN / VPD are
    # missing, it falls back to the mean diurnal cycle, so the record is still
    # filled but with a lower-quality method. The gap-filled column gives no
    # hint that driver coverage is the cause. Log coverage upfront; warn when
    # any driver is below 80%, because that is usually the main reason for
    # poor MDS fill quality.
    n_total = len(data.full_df)
    for _drv_label, _drv_col in (('SW_IN', swin), ('TA', ta), ('VPD', vpd)):
        _n_valid = int(data.full_df[_drv_col].notna().sum())
        _coverage = _n_valid / n_total if n_total else 0.0
        _msg = (f"MDS driver coverage: {_drv_label}={_drv_col!r}  "
                f"{_n_valid}/{n_total} valid ({_coverage:.1%})")
        if _coverage < 0.80:
            warn(f"{_msg} - where drivers are missing, MDS falls back to the "
                 f"mean diurnal cycle (lower fill quality)")
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
    # The REF was selected from the previous MDS versions: drop it as well.
    data = clear_ustar_reference(data)
    pre_columns = list(data.fpc_df.columns)
    fpc_df = data.fpc_df.copy()
    mds_results: dict = {}

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        scen_df = data.full_df[list(dict.fromkeys([swin, ta, vpd, *qc_cols]))].copy()
        assert_aligned_index(scen_df, fpc_df[ustar_flux.name],
                             context=f"run_level41_mds[{ustar_scen!r}] scen_df build")
        scen_df = pd.concat([scen_df, fpc_df[ustar_flux.name]], axis=1)

        instance = FluxMDS(
            df=scen_df,
            flux=ustar_flux.name,
            swin=swin, ta=ta, vpd=vpd,
            swin_tol=swin_tol, ta_tol=ta_tol, vpd_tol=vpd_tol,
            avg_min_n_vals=avg_min_n_vals, mode=mode, fill_long_gaps=fill_long_gaps,
            swin_qc=swin_qc, ta_qc=ta_qc, vpd_qc=vpd_qc,
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
        scenarios=None,
        shared_features: bool = True,
        n_estimators_cv: bool | dict = False,
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
    filteredseries_l33 = select_scenarios(data, scenarios)

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

    # Every model of every scenario holds out the same timestamps as test
    # records: the first instance draws them, the others take them over. No
    # model, feature selection or tree selection then trains on a record that
    # another model tests on.
    instances = {}
    test_records = None
    for ustar_scen, ustar_flux in filteredseries_l33.items():
        assert_aligned_index(engineered, fpc_df[ustar_flux.name],
                             context=f"_run_level41_ml[{ustar_scen!r}] scen_df build")
        scen_df = pd.concat([engineered, fpc_df[ustar_flux.name]], axis=1).copy()

        instance = model_factory(
            input_df=scen_df,
            target_col=ustar_flux.name,
            verbose=verbose,
            test_records=test_records,
            **model_kwargs,
        )
        test_records = instance.test_records_
        instance.create_yearpools()
        instance.initialize_yearly_models()
        if reduce_features:
            instance.reduce_features_across_years()
        instances[ustar_scen] = instance

    # One feature set for all scenarios: the features any scenario selected.
    # With a set of its own per scenario, the spread between the scenarios
    # would mix the effect of the USTAR threshold with that of the selection.
    if reduce_features and shared_features and len(instances) > 1:
        shared = sorted(set().union(*(i.features_reduced_across_years for i in instances.values())))
        counts = ", ".join(f"{s} {len(i.features_reduced_across_years)}" for s, i in instances.items())
        info(f"Features selected per scenario: {counts}. All scenarios use the "
             f"{len(shared)} features any of them selected.", verbose=verbose)
        for instance in instances.values():
            instance.use_features(shared)

    if n_estimators_cv:
        _set_n_estimators_from_blocked_cv(instances, n_estimators_cv, verbose,
                                          middle=middle_scenario(data, instances))

    for ustar_scen, ustar_flux in filteredseries_l33.items():
        instance = instances[ustar_scen]
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


def _set_n_estimators_from_blocked_cv(instances: dict, settings: bool | dict,
                                      verbose: int, middle: str) -> None:
    """Choose one number of trees by blocked cross-validation and give it to every model.

    Runs on the training records of each yearly pool of one scenario (``middle``, the one
    nearest the 50th percentile, unless ``settings['scenario']`` names it). All yearly models of all scenarios hold out the same
    timestamps (``test_records_``), so these pools contain no test record of any model. Every
    model of every scenario then gets that number of trees and no early stopping. Setting the
    models up again keeps their train/test split.
    """
    from diive.gapfilling.tree_selection import select_n_estimators_blocked_cv
    cv = {} if settings is True else dict(settings)
    labels = list(instances)
    scenario = cv.pop('scenario', None) or middle
    if scenario not in instances:
        raise ValueError(f"Scenario {scenario!r} for the tree selection not found. Available: {labels}")
    ref = instances[scenario]
    pools = {}
    for year, model in ref.results_yearly_.items():
        years = ref.yearpools[year]['poolyears']
        label = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
        pools.setdefault(label, model.train_df)  # neighbouring years can share one pool
    info(f"Choosing the number of trees by blocked cross-validation on {scenario} "
         f"({len(pools)} pools) ...", verbose=verbose)
    # The model's own settings: they include the default random_state of the long-term
    # models, which then also seeds the folds.
    selection = select_n_estimators_blocked_cv(pools, ref.target_col, ref.kwargs,
                                               verbose=verbose, **cv)
    selection.scenario = scenario
    for instance in instances.values():
        instance.kwargs = {**instance.kwargs, 'n_estimators': selection.n_estimators,
                           'early_stopping_rounds': None}
        instance.initialize_yearly_models()
        instance.tree_selection_ = selection


def run_level41_rf(
        data: FluxLevelData,
        *,
        features: list[str],
        engineer: 'FeatureEngineer',
        reduce_features: bool = False,
        shared_features: bool = True,
        scenarios: list[str] | None = None,
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
        shared_features: With ``reduce_features``, fit every USTAR scenario on
            the same features: those selected for any scenario (default). With
            False each scenario keeps its own selection, and the spread between
            the scenarios then also reflects the selection, not only the USTAR
            threshold.
        scenarios: USTAR scenario labels to gap-fill. None (default) takes all
            Level 3.3 scenarios except the 40 REF versions of
            ``reference=True`` (see ``select_scenarios``), so with REF on it
            runs on e.g. CUT_16, CUT_50, CUT_84 and the REF scenario.
        verbose: Verbosity level (0=silent, 1+=progress).
        **rf_kwargs: sklearn ``RandomForestRegressor`` hyperparameters
            (``n_estimators``, ``max_depth``, ``min_samples_split``, ...).
            Without ``random_state`` the models use ``random_state=42``, so
            identical runs give identical results. All yearly models of all
            scenarios hold out the same random test records (see
            :class:`~diive.gapfilling.longterm.LongTermGapFillingBase`).

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
        scenarios=scenarios,
        shared_features=shared_features,
    )


def run_level41_xgb(
        data: FluxLevelData,
        *,
        features: list[str],
        engineer: 'FeatureEngineer',
        reduce_features: bool = False,
        shared_features: bool = True,
        n_estimators_cv: bool | dict = False,
        scenarios: list[str] | None = None,
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
        shared_features: As in :func:`run_level41_rf`.
        n_estimators_cv: Choose the number of trees by blocked cross-validation
            (:func:`~diive.gapfilling.tree_selection.select_n_estimators_blocked_cv`)
            and use it as a fixed ``n_estimators`` for every year and scenario, with no
            early stopping. It runs after the feature selection, on the training records
            of each yearly pool of one scenario, holding out whole blocks of days. All
            yearly models of all scenarios hold out the same random test records, so
            neither the feature selection nor the tree selection sees a record that a
            model is tested on, and the test scores stay out-of-sample.
            ``n_estimators`` sets the largest number tested. True uses the defaults
            (the scenario nearest the 50th percentile, 14-day blocks, 5 folds, the
            minimum of the median error curve). A dict changes them, e.g.
            ``{'block_days': 7, 'scenario': 'CUT_50'}``.
            The result is ``tree_selection_`` of each scenario's model, shown by
            ``report_tree_selection`` and ``plot_tree_selection``. Default False.
        scenarios: As in :func:`run_level41_rf`.
        verbose: Verbosity level (0=silent, 1+=progress).
        **xgb_kwargs: ``XGBRegressor`` hyperparameters (``n_estimators``,
            ``max_depth``, ``learning_rate``, ``early_stopping_rounds``, ...).
            Without ``random_state`` the models use ``random_state=42``, so
            identical runs give identical results (see
            :class:`~diive.gapfilling.longterm.LongTermGapFillingBase`).

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
        scenarios=scenarios,
        shared_features=shared_features,
        n_estimators_cv=n_estimators_cv,
    )


def run_level41_ustar_reference(
        data: FluxLevelData,
        *,
        mef_qc: int | None = None,
        mef_min_gap: int | None = None,
        mef_max_gap: int | None = None,
        verbose: bool = True,
) -> FluxLevelData:
    """
    Level-4.1: Select the REF u* scenario as ONEFlux does (``NEE_CUT_REF`` / ``NEE_VUT_REF``).

    ONEFlux ``nee_proc`` gap-fills NEE filtered at 40 u* percentiles (1.25 to
    98.75) and takes as REF the version most similar to all others, by model
    efficiency. See :func:`diive.flux.lowres.ustar_reference.select_ustar_reference`
    for the method and the ONEFlux details it reproduces.

    Needs ``run_level33_ustar_detection(..., reference=True)`` (CUT or VUT,
    detected or given) and ``run_level41_mds`` afterwards, so that each of the
    40 scenarios has an MDS gap-filled series. ONEFlux gap-fills with its own
    MDS: run ``run_level41_mds(mode='oneflux')`` for the REF ONEFlux would
    select. With another MDS mode the REF is selected among those series.

    .. important::

        The u* filter of Level 3.3 keeps the first record above the threshold
        after a period below it, ONEFlux removes it (finding L74). The
        gap-filled versions then differ from ONEFlux's, and in the checks
        against ONEFlux this changed the REF at some time resolutions in every
        case, even with ``mode='oneflux'`` in MDS.

    The half-hourly REF is added to ``fpc_df`` as its own series: the
    gap-filled NEE and its flag of the selected scenario, with the scenario
    label replaced by ``CUT_REF`` (or ``VUT_REF``), e.g.
    ``NEE_L3.1_L3.3_CUT_REF_QCF_gfMDS``. The other scenarios are not changed.
    The selection uses the gap-filled versions with their long gaps filled,
    as ONEFlux ``nee_proc`` does, also when MDS removed them from the output
    (``fill_long_gaps=False``). The REF series then lacks the long gaps of the
    selected scenario, as ``NEE_CUT_REF`` in the FLUXNET product.
    The REF of the daily, weekly, monthly and yearly values are in
    ``data.levels.ustar_reference.scenarios``.

    Args:
        data: FluxLevelData after ``run_level41_mds``.
        mef_qc: Highest MDS quality kept for the selection (``nee_proc``
            ``-mef_qc``, default 2).
        mef_min_gap: Smallest block of records left out (default 240).
        mef_max_gap: Distance in records from a good record beyond which
            low-quality records are left out (default 720).
        verbose: Print the selected REF per time resolution.

    Returns:
        Updated FluxLevelData with the REF columns in ``fpc_df`` and
        ``data.levels.ustar_reference`` (:class:`UstarReference`): REF label,
        the scenario and percentile it maps to, its threshold (per year for
        VUT), the REF scenario per time resolution and the full selection
        result (scores, model efficiency, records left out).

    Raises:
        RuntimeError: If L3.3 did not add the REF scenarios, MDS has not
            gap-filled all of them, or ONEFlux finds no half-hourly REF.
    """
    from diive.core.utils.console import console as _console
    from diive.flux.lowres.ustar_reference import (
        ONEFLUX_MEF_MAX_GAP, ONEFLUX_MEF_MIN_GAP, ONEFLUX_MEF_QC, REF_AGGREGATIONS,
        select_ustar_reference,
    )
    from diive.flux.postprocessingchain.container import UstarReference

    scen = data.levels.ustar_scenarios
    if scen is None or not scen.reference_labels:
        raise RuntimeError(
            "run_level41_ustar_reference() needs the 40 ONEFlux REF scenarios: call "
            "run_level33_ustar_detection(..., reference=True) first.")
    labels = list(scen.reference_labels)
    missing = [lab for lab in labels if lab not in data.levels.level41_mds]
    if missing:
        raise RuntimeError(
            f"run_level41_ustar_reference() needs the MDS gap-filled series of every REF "
            f"scenario: call run_level41_mds() first. Missing: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}")

    rule("Level 4.1: REF u* scenario (ONEFlux model efficiency)", verbose=verbose)
    data = clear_ustar_reference(data)
    pre_columns = list(data.fpc_df.columns)

    nee, qc = {}, {}
    for lab in labels:
        inst = data.levels.level41_mds[lab]
        # nee_proc: QC 0 where the record is measured, else the MDS quality
        # (dataset.c:5014); a record MDS could not fill has none. nee_proc
        # selects on the filled long gaps, the product removes them later.
        qc[lab] = inst.get_qc(full=True)
        nee[lab] = inst.get_gapfilled_target(full=True)
    percentiles = [scen.percentiles[lab] for lab in labels]
    result = select_ustar_reference(
        pd.DataFrame(nee), pd.DataFrame(qc), percentiles,
        mef_qc=ONEFLUX_MEF_QC if mef_qc is None else mef_qc,
        mef_min_gap=ONEFLUX_MEF_MIN_GAP if mef_min_gap is None else mef_min_gap,
        mef_max_gap=ONEFLUX_MEF_MAX_GAP if mef_max_gap is None else mef_max_gap)
    if result.index['HH'] < 0:
        raise RuntimeError("ONEFlux finds no half-hourly REF for these data "
                           "(a version without any value).")

    source = labels[result.index['HH']]
    ref_label = f"{scen.threshold_type.upper()}_REF"
    inst = data.levels.level41_mds[source]
    gf_col, flag_col = inst.target_gapfilled, inst.target_gapfilled_flag
    new_gf, new_flag = gf_col.replace(source, ref_label), flag_col.replace(source, ref_label)
    fpc_df = data.fpc_df.copy()
    assert_aligned_index(fpc_df, data.fpc_df[gf_col],
                         context="run_level41_ustar_reference merge into fpc_df")
    fpc_df[new_gf] = data.fpc_df[gf_col]
    fpc_df[new_flag] = data.fpc_df[flag_col]

    record = UstarReference(
        label=ref_label,
        scenario=source,
        percentile=scen.percentiles[source],
        threshold=scen.thresholds[source],
        gapfilled_col=new_gf,
        flag_col=new_flag,
        scenarios={agg: (labels[result.index[agg]] if result.index[agg] >= 0 else None)
                   for agg in REF_AGGREGATIONS},
        result=result,
    )
    if verbose:
        _console.print(result.summary())
        _console.print(f"{ref_label} = {source}")
    final = replace(data, fpc_df=fpc_df,
                    levels=replace(data.levels, ustar_reference=record))
    return replace(final, added_columns=record_added_columns(final, 'L4.1_ref', pre_columns))
