"""
ANALYSIS: COMPOUND EXTREMES
============================

Classify time periods (months or days) into compound-extreme categories from the
standardized anomalies (z-scores) of two driver variables.

The canonical use case is compound dry-hot detection (after Wang et al.): a month
or day is flagged as an atmospheric-dryness extreme when VPD is anomalously high,
a soil-dryness extreme when soil water content is anomalously low, and a *compound*
extreme when both occur together. The method is generic, though: any two variables
with a defined extreme direction (high or low) can be combined.

Part of the diive library: https://github.com/holukas/diive
"""

from typing import Literal

import pandas as pd
from pandas import DataFrame, Series

# Category codes stored in the CATEGORY column. The human-readable labels are
# configurable (var1_label / var2_label / compound_label / none_label); these
# codes stay fixed so downstream code can branch on them regardless of labels.
CAT_NONE = 'none'
CAT_VAR1 = 'var1'
CAT_VAR2 = 'var2'
CAT_COMPOUND = 'compound'

CATEGORY_ORDER = [CAT_NONE, CAT_VAR1, CAT_VAR2, CAT_COMPOUND]


class CompoundExtremes:
    """Detect compound extremes from two standardized drivers. See :meth:`__init__`."""

    def __init__(
            self,
            var1: Series,
            var2: Series,
            agg: Literal['monthly', 'daily'] = 'monthly',
            agg_func: Literal['mean', 'median', 'min', 'max', 'sum'] = 'mean',
            var1_extreme: Literal['high', 'low'] = 'high',
            var2_extreme: Literal['high', 'low'] = 'low',
            threshold: float = 2.0,
            var1_threshold: float = None,
            var2_threshold: float = None,
            standardize_by: Literal['season', 'record'] = 'season',
            var1_label: str = None,
            var2_label: str = None,
            compound_label: str = 'Compound',
            none_label: str = 'None',
    ):
        """Classify months or days into compound-extreme categories.

        Each period is aggregated to the chosen resolution, both variables are
        standardized to z-scores, and a period is flagged extreme for a variable
        when its z-score crosses the threshold in the variable's extreme direction.
        The four categories are: none, var1-only, var2-only, and compound (both).

        Args:
            var1: First driver, e.g. VPD. Any sampling frequency (aggregated internally).
            var2: Second driver, e.g. soil water content.
            agg: Temporal resolution to analyze. 'monthly' -> month-start periods,
                'daily' -> calendar days.
            agg_func: Aggregation applied when resampling to *agg* resolution.
            var1_extreme: Which tail of *var1* counts as extreme. 'high' flags
                z >= threshold (e.g. high VPD), 'low' flags z <= -threshold.
            var2_extreme: Which tail of *var2* counts as extreme (e.g. 'low' for
                soil water content -> drought).
            threshold: z-score magnitude (always positive) used for both variables
                unless overridden per variable.
            var1_threshold: Optional per-variable threshold magnitude for *var1*
                (falls back to *threshold*).
            var2_threshold: Optional per-variable threshold magnitude for *var2*.
            standardize_by: How z-scores are computed.
                - 'season': deseasonalized — each period is standardized against the
                  same position in the seasonal cycle (calendar month for monthly,
                  day-of-year for daily). Removes the seasonal cycle so genuinely
                  anomalous periods stand out. This is the default and the standard
                  choice for compound-extreme detection.
                - 'record': a single mean/std over the whole aggregated record.
                  Simpler, but the seasonal cycle of variables such as VPD dominates
                  the z-score.
            var1_label: Human-readable label for the var1-only category (default:
                the *var1* series name).
            var2_label: Human-readable label for the var2-only category (default:
                the *var2* series name).
            compound_label: Label for the compound (both-extreme) category.
            none_label: Label for the no-extreme category.

        Properties:
            .results: DataFrame indexed by period with aggregated values, z-scores,
                per-variable extreme flags, the CATEGORY code, and a human LABEL.
            .counts: Per-category record counts (Series, indexed by label).
            .labels: Period-name Series (e.g. '2022-08') for annotating plots.

        Example:
            See `examples/analysis/analysis_compound_extremes.py`.

        See Also:
            diive.plotting.CompoundExtremesPlot : the quadrant scatter for these results.
        """
        if var1.name is None:
            var1 = var1.rename('VAR1')
        if var2.name is None:
            var2 = var2.rename('VAR2')
        if var1.name == var2.name:
            raise ValueError("CompoundExtremes: var1 and var2 must have different names.")

        self.var1 = var1
        self.var2 = var2
        self.var1name = str(var1.name)
        self.var2name = str(var2.name)
        self.agg = agg
        self.agg_func = agg_func
        self.var1_extreme = var1_extreme
        self.var2_extreme = var2_extreme
        self.var1_threshold = var1_threshold if var1_threshold is not None else threshold
        self.var2_threshold = var2_threshold if var2_threshold is not None else threshold
        self.standardize_by = standardize_by

        self.var1_label = var1_label if var1_label is not None else self.var1name
        self.var2_label = var2_label if var2_label is not None else self.var2name
        self.compound_label = compound_label
        self.none_label = none_label

        self.var1_z_col = f"{self.var1name}_Z"
        self.var2_z_col = f"{self.var2name}_Z"

        self._results = None
        self._calc()

    @property
    def results(self) -> DataFrame:
        """Classification results, one row per analyzed period."""
        if self._results is None:
            raise Exception("CompoundExtremes results are empty.")
        return self._results

    @property
    def counts(self) -> Series:
        """Record count per category label (ordered none/var1/var2/compound)."""
        order = [self.none_label, self.var1_label, self.var2_label, self.compound_label]
        c = self.results['LABEL'].value_counts()
        return c.reindex(order).fillna(0).astype(int)

    @property
    def labels(self) -> Series:
        """Period-name labels (e.g. '2022-08') aligned to the results index."""
        return self.results['PERIOD']

    @property
    def label_map(self) -> dict:
        """Category code -> human label mapping."""
        return {CAT_NONE: self.none_label, CAT_VAR1: self.var1_label,
                CAT_VAR2: self.var2_label, CAT_COMPOUND: self.compound_label}

    def _aggregate(self, series: Series) -> Series:
        rule = 'MS' if self.agg == 'monthly' else 'D'
        return series.resample(rule).agg(self.agg_func)

    def _zscore(self, series: Series) -> Series:
        if self.standardize_by == 'record':
            std = series.std()
            return (series - series.mean()) / std if std else series * 0.0

        # 'season': standardize within the matching position in the seasonal cycle.
        group = series.index.month if self.agg == 'monthly' else series.index.dayofyear
        grouped = series.groupby(group)
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        z = (series - mean) / std
        # A season group with a single member (or zero variance) has std=NaN/0 and
        # cannot yield a meaningful anomaly; leave those periods as NaN (not extreme).
        return z.where(std.ne(0))

    def _extreme_flag(self, z: Series, direction: str, thr: float) -> Series:
        if direction == 'high':
            return z >= thr
        return z <= -thr

    def _calc(self):
        agg1 = self._aggregate(self.var1)
        agg2 = self._aggregate(self.var2)
        df = pd.concat([agg1, agg2], axis=1)
        df.columns = [self.var1name, self.var2name]

        df[self.var1_z_col] = self._zscore(df[self.var1name])
        df[self.var2_z_col] = self._zscore(df[self.var2name])

        # Periods missing either z-score cannot be classified.
        df = df.dropna(subset=[self.var1_z_col, self.var2_z_col])

        flag1 = self._extreme_flag(df[self.var1_z_col], self.var1_extreme, self.var1_threshold)
        flag2 = self._extreme_flag(df[self.var2_z_col], self.var2_extreme, self.var2_threshold)
        df['VAR1_EXTREME'] = flag1
        df['VAR2_EXTREME'] = flag2

        category = pd.Series(CAT_NONE, index=df.index, dtype=object)
        category[flag1 & ~flag2] = CAT_VAR1
        category[~flag1 & flag2] = CAT_VAR2
        category[flag1 & flag2] = CAT_COMPOUND
        df['CATEGORY'] = category
        df['LABEL'] = category.map(self.label_map)

        fmt = '%Y-%m' if self.agg == 'monthly' else '%Y-%m-%d'
        df['PERIOD'] = df.index.strftime(fmt)

        self._results = df
