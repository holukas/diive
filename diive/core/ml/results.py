"""
GAP-FILLING RESULT: STRUCTURED OUTPUT CONTAINER
================================================

Dataclass returned by the `.results` property on all gap-filling classes.
Bundles every post-run output into a single, inspectable object.

Part of the diive library: https://github.com/holukas/diive
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class GapFillingResult:
    """Structured output from a gap-filling run.

    Returned by the ``.results`` property on :class:`RandomForestTS`,
    :class:`XGBoostTS`, :class:`FluxMDS`, and their variants.

    Attributes:
        gapfilled: Gap-filled target time series (observed + filled values).
        flag: Quality flag — 0 = observed, 1 = gap-filled, 2 = fallback (ML only).
        scores: In-sample performance metrics — the final model predicting on ALL
            complete rows, including the rows it was trained on, so these are
            optimistically biased. Dict with keys mae, medae, mse, rmse, mape, maxe,
            r2. Use *scores_traintest* for an honest generalization estimate.
        gapfilling_df: Full results DataFrame (all auxiliary columns included).
        scores_traintest: Held-out (out-of-sample) metrics from the train/test split —
            the generalization estimate; same keys as *scores*. None for MDS.
        feature_importances: SHAP importances from the gap-filling model. None for MDS.
        feature_importances_traintest: SHAP importances from the train/test model. None for MDS.
        model: Trained sklearn/XGBoost regressor. None for MDS.
        accepted_features: Feature names kept after SHAP-based reduction. None if not run.
        rejected_features: Feature names removed after SHAP-based reduction. None if not run.

    Example::

        rf = dv.gapfilling.RandomForestTS(input_df=engineered, target_col='NEE_f')
        rf.run()
        r = rf.results

        r.gapfilled          # gap-filled Series
        r.flag               # flag Series
        r.scores['r2']       # gap-filling R²
        r.scores_traintest['rmse']
        r.feature_importances
        r.model.feature_importances_
    """

    gapfilled: pd.Series
    flag: pd.Series
    scores: dict
    gapfilling_df: pd.DataFrame
    scores_traintest: Optional[dict] = None
    feature_importances: Optional[pd.DataFrame] = None
    feature_importances_traintest: Optional[pd.DataFrame] = None
    model: Optional[object] = None
    accepted_features: Optional[list] = None
    rejected_features: Optional[list] = None
