"""
GAP-FILLING: NUMBER OF TREES FROM BLOCKED CROSS-VALIDATION
==========================================================

Choose one number of trees for XGBoost gap-filling models by cross-validation on
whole blocks of days instead of single records.

Part of the diive library: https://github.com/holukas/diive
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame

from diive.core.utils.console import detail, info

#: Settings of the model that are not XGBoost booster parameters.
_NOT_BOOSTER = ('n_estimators', 'early_stopping_rounds', 'shap_max_rows', 'test_size', 'below_zero')


@dataclass
class TreeSelection:
    """Result of :func:`select_n_estimators_blocked_cv`.

    Attributes:
        n_estimators: The chosen number of trees.
        curves: Blocked cross-validation RMSE after each number of trees (rows, from 1),
            one column per pool.
        median_curve: Median of the curves, relative to its minimum (1.0 = minimum).
        scores: Per pool: the blocked RMSE and MAE at ``n_estimators``, the R² derived from
            the RMSE and the variance of the target, and the number of records.
        block_days: Length of the held-out blocks in days.
        n_folds: Number of folds.
        tolerance: Relative tolerance of the selection.
        max_trees: The largest number of trees tested.
        scenario: Label of the data the selection ran on, set by the caller (e.g. the USTAR
            scenario of the flux chain).
    """
    n_estimators: int
    curves: DataFrame
    median_curve: pd.Series
    scores: DataFrame
    block_days: float
    n_folds: int
    tolerance: float
    max_trees: int
    scenario: str = ''


def blocked_folds(index: pd.DatetimeIndex, block_days: float = 14, n_folds: int = 5,
                  random_state: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds of whole blocks of days, for cross-validation of a gap-filling model.

    The record is cut into consecutive blocks of ``block_days``; the blocks are shuffled
    and dealt to ``n_folds`` folds in turn. Each fold holds out its blocks as a whole, so
    the held-out records look like real gaps of that length. Records next to a held-out
    record are mostly held out too, unlike with a random split of single records, where
    the direct neighbours of almost every held-out record are in the training data.

    Args:
        index: Timestamps of the records (need not be regular).
        block_days: Length of a block in days.
        n_folds: Number of folds.
        random_state: Seed for the shuffling of the blocks.

    Returns:
        List of ``(train_positions, test_positions)`` arrays, one per fold.
    """
    t = (index - index[0]) / pd.Timedelta(days=block_days)
    block = np.floor(np.asarray(t, dtype=float)).astype(int)
    blocks = np.unique(block)
    if len(blocks) < n_folds:
        raise ValueError(f"{len(blocks)} block(s) of {block_days} days cannot make {n_folds} folds.")
    rng = np.random.default_rng(random_state)
    fold_of_block = dict(zip(rng.permutation(blocks), np.arange(len(blocks)) % n_folds, strict=True))
    fold = np.array([fold_of_block[b] for b in block])
    positions = np.arange(len(index))
    return [(positions[fold != k], positions[fold == k]) for k in range(n_folds)]


def select_n_estimators_blocked_cv(pools: dict, target_col: str, model_kwargs: dict,
                                   max_trees: int | None = None, block_days: float = 14,
                                   n_folds: int = 5, tolerance: float = 0.0,
                                   random_state: int | None = None,
                                   verbose: int = 0) -> TreeSelection:
    """Choose one number of trees for XGBoost by blocked cross-validation.

    For each pool (the training records of one model, e.g. one year and its neighbours),
    XGBoost is cross-validated on folds of whole ``block_days`` blocks (:func:`blocked_folds`),
    and the RMSE on the held-out blocks is recorded after every tree. The curves of all
    pools are scaled to their minimum, and the number of trees is the minimum of their median
    curve, rounded to ten. With fewer trees the model misses structure; with more it fits
    patterns that help with single missing records but not with gaps of days, and the error
    on the held-out blocks rises again. A ``tolerance`` above zero takes instead the smallest
    number at which the median curve comes within that fraction of its minimum.

    Use the result as a fixed ``n_estimators`` for every model, without early stopping. All
    models then have the same size, so differences between them come from their data, and
    the choice does not rest on one small random validation set per model.

    Args:
        pools: ``{label: DataFrame}`` with the target and the features, complete records.
        target_col: Name of the target column.
        model_kwargs: XGBoost settings of the gap-filling models. ``n_estimators`` gives
            the largest number of trees tested unless ``max_trees`` is set;
            ``early_stopping_rounds``, ``shap_max_rows``, ``test_size`` and
            ``below_zero`` are ignored. Its ``random_state`` seeds the folds unless
            ``random_state`` is given.
        max_trees: Largest number of trees tested.
        block_days: Length of the held-out blocks in days.
        n_folds: Number of folds.
        tolerance: 0 (default) takes the minimum of the median curve; e.g. 0.01 takes the
            smallest number of trees with a median RMSE up to 1 % above that minimum.
        random_state: Seed for the folds.
        verbose: Verbosity level.

    Returns:
        :class:`TreeSelection`.
    """
    import xgboost as xgb
    from xgboost import XGBRegressor

    max_trees = int(max_trees or model_kwargs.get('n_estimators') or 1000)
    kwargs = {k: v for k, v in model_kwargs.items() if k not in _NOT_BOOSTER}
    params = {k: v for k, v in XGBRegressor(**kwargs).get_xgb_params().items() if v is not None}
    seed = random_state if random_state is not None else model_kwargs.get('random_state')

    curves, mae, rows = {}, {}, {}
    for label, df in pools.items():
        df = df.dropna()
        info(f"Blocked cross-validation for {label}: {len(df):,} records, {n_folds} folds of "
             f"{block_days:g}-day blocks, up to {max_trees} trees ...", verbose=verbose)
        dtrain = xgb.DMatrix(df.drop(columns=target_col).to_numpy(), label=df[target_col].to_numpy())
        folds = blocked_folds(df.index, block_days=block_days, n_folds=n_folds, random_state=seed)
        res = xgb.cv(params, dtrain, num_boost_round=max_trees, folds=folds,
                     metrics=['rmse', 'mae'], as_pandas=True, verbose_eval=False)
        curves[label] = res['test-rmse-mean'].to_numpy()
        mae[label] = res['test-mae-mean'].to_numpy()
        rows[label] = (len(df), float(df[target_col].var()))
        detail(f"{label}: minimum RMSE {curves[label].min():.3f} after "
               f"{int(curves[label].argmin()) + 1} trees", verbose=verbose)

    curves = DataFrame(curves, index=pd.RangeIndex(1, max_trees + 1, name='trees'))
    relative = curves / curves.min()
    median_curve = relative.median(axis=1)
    values = median_curve.to_numpy()
    first = int(np.argmax(values <= values.min() * (1 + tolerance))) + 1
    n_estimators = min(max_trees, max(10, int(round(first / 10)) * 10))

    scores = DataFrame({
        'records': {k: v[0] for k, v in rows.items()},
        'rmse': curves.loc[n_estimators],
        'mae': {k: v[n_estimators - 1] for k, v in mae.items()},
        'r2': {k: 1 - curves.loc[n_estimators, k] ** 2 / v[1] for k, v in rows.items()},
    })
    rule = (f"median RMSE within {100 * tolerance:g} % of its minimum" if tolerance
            else "minimum of the median RMSE")
    info(f"Number of trees from blocked cross-validation: {n_estimators} ({rule})", verbose=verbose)
    return TreeSelection(n_estimators=n_estimators, curves=curves, median_curve=median_curve,
                         scores=scores, block_days=block_days, n_folds=n_folds,
                         tolerance=tolerance, max_trees=max_trees)
