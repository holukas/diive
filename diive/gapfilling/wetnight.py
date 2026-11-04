"""
GAP-FILLING: LONG-TERM XGBOOST WITH A LLOYD-TAYLOR FILL OF WET NIGHTS
=====================================================================

Long-term XGBoost gap-filling in which the night gaps marked as wet are filled from a
Lloyd-Taylor respiration model instead.

An open-path gas analyser cannot measure in rain and fog. A machine-learning filler learns
these nights only from the few humid nights the analyser did measure, and from drivers such
as VPD near zero that it shares with calm foggy nights of low flux. It then fills the wet
night gaps too low. Night-time NEE is ecosystem respiration, which a temperature response
fitted to the dry measured nights describes without these drivers.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.optimize import curve_fit

from diive.core.utils.console import detail, info, warn
from diive.flux.partitioning.nighttime_oneflux import lloyd_taylor
from diive.gapfilling.longterm import LongTermGapFillingXGBoostTS

#: Suffix of the gap-filled column, next to ``_gfXG`` of the plain XGBoost fill.
GFSUFFIX = '_gfXGWN'

#: Flag value of a wet night gap filled from the Lloyd-Taylor model.
FLAG_WETNIGHT = 3


def _as_bool(values, index: pd.Index, name: str) -> Series:
    """``values`` as a boolean series on ``index``; records not given are False."""
    if not isinstance(values, Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")
    return values.reindex(index).astype('boolean').fillna(False).astype(bool)


class LongTermGapFillingXGBoostWetNightTS(LongTermGapFillingXGBoostTS):
    """Long-term XGBoost gap-filling with a Lloyd-Taylor fill of the wet night gaps.

    See :meth:`__init__`."""

    def __init__(self,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 wet: Series,
                 ta: Series,
                 night: Series,
                 dry: Series = None,
                 min_records: int = 30,
                 e0_bounds: tuple = (50.0, 450.0),
                 verbose: int = 0,
                 test_size: float = 0.25,
                 below_zero: str = None,
                 test_records: pd.Index = None,
                 **kwargs):
        """Long-term XGBoost gap-filling with a Lloyd-Taylor fill of the wet night gaps.

        All gaps are first filled by :class:`LongTermGapFillingXGBoostTS`, unchanged: the
        same features, training records and yearly models. The night gaps marked as wet are
        then filled from a Lloyd-Taylor model (Lloyd and Taylor 1994) fitted to the measured
        dry nights of the same year: the temperature sensitivity E0 per year, the reference
        respiration at 15 degC per month, or per year where a month has fewer than
        ``min_records`` dry nights. A year with fewer than ``min_records`` dry nights keeps
        the XGBoost fill.

        The gap-filled column ends in ``_gfXGWN`` instead of ``_gfXG``, so the two fills can
        exist side by side. Its flag is 0 for measured, 1 for XGBoost, 2 for the XGBoost
        fallback and 3 for a wet night gap filled from the Lloyd-Taylor model.

        Args:
            input_df: DataFrame with the target and the pre-engineered feature columns.
            target_col: Column of the flux to gap-fill, night-time values being respiration.
            wet: Boolean series, True where a record counts as wet (for example rain in the
                previous hours, a wet-window signal of the analyser, an overcast sky). Which
                conditions count is the caller's choice.
            ta: Air temperature (degC) for the Lloyd-Taylor model.
            night: Boolean series, True at night.
            dry: Boolean series, True for the measured nights the model is fitted to.
                Default: the records that are not wet.
            min_records: Fewest dry measured nights for a reference respiration of its own
                (per month) and for a fit at all (per year).
            e0_bounds: Bounds of E0 (degC) in the fit.
            Other args: see :class:`LongTermGapFillingXGBoostTS`.

        Attributes:
            wetnight_estimate_: Lloyd-Taylor estimate for every record of the fitted years.
            wetnight_params_: Per year: E0, the yearly reference respiration and the number
                of dry nights fitted.
            n_wetnight_filled_: Number of wet night gaps filled from the Lloyd-Taylor model.
        """
        super().__init__(input_df=input_df, target_col=target_col, verbose=verbose,
                         test_size=test_size, below_zero=below_zero, test_records=test_records,
                         **kwargs)
        idx = input_df.index
        self.wet = _as_bool(wet, idx, 'wet')
        self.night = _as_bool(night, idx, 'night')
        self.dry = ~self.wet if dry is None else _as_bool(dry, idx, 'dry')
        if not isinstance(ta, Series):
            raise TypeError(f"ta must be a pandas Series, got {type(ta).__name__}")
        self.ta = ta.reindex(idx).astype(float)
        self.min_records = min_records
        self.e0_bounds = e0_bounds
        self._wetnight_estimate = pd.Series(np.nan, index=idx, name=f'{target_col}_LT_WETNIGHT')
        self._wetnight_params = pd.DataFrame()
        self._n_wetnight_filled = 0
        self._flag_col = None

    @property
    def wetnight_estimate_(self) -> Series:
        """Lloyd-Taylor estimate for every record of the fitted years."""
        return self._wetnight_estimate

    @property
    def wetnight_params_(self) -> DataFrame:
        """Per year: E0 (degC), yearly reference respiration at 15 degC and dry nights fitted."""
        return self._wetnight_params

    @property
    def n_wetnight_filled_(self) -> int:
        """Number of wet night gaps filled from the Lloyd-Taylor model."""
        return self._n_wetnight_filled

    def get_flag(self) -> Series:
        """Gap-filling flag across all years (0 measured, 1 XGBoost, 2 fallback, 3 wet night)."""
        if self._flag_col is None:
            return super().get_flag()
        return self.gapfilling_df_[self._flag_col]

    def fillgaps(self):
        """Fill all gaps with XGBoost, then the wet night gaps from the Lloyd-Taylor model."""
        super().fillgaps()
        self._fill_wet_nights()

    def _fit_year(self, observed: Series, ta: Series) -> tuple:
        """E0 and the yearly reference respiration from the dry nights of one year."""
        (rref, e0), _ = curve_fit(lambda t, r, e: lloyd_taylor(t, r, e), ta.to_numpy(), observed.to_numpy(),
                                  p0=(3.0, 150.0), bounds=([0.0, self.e0_bounds[0]], [np.inf, self.e0_bounds[1]]))
        return rref, e0

    def _fill_wet_nights(self):
        model = next(iter(self.results_yearly_.values()))
        gf_col, flag_col = model.target_gapfilled_col, model.target_gapfilled_flag_col
        df = self._gapfilling_df
        observed = df[self.target_col]
        flag = df[flag_col]
        fit_records = self.night & self.dry & observed.notna() & self.ta.notna()

        params = {}
        for year in sorted(set(df.index.year)):
            in_year = df.index.year == year
            fit = fit_records & in_year
            if fit.sum() < self.min_records:
                warn(f"{self.target_col}, {year}: {int(fit.sum())} dry measured nights, fewer than "
                     f"{self.min_records}; its wet night gaps keep the XGBoost fill.", verbose=self.verbose)
                continue
            rref_year, e0 = self._fit_year(observed[fit], self.ta[fit])
            # Reference respiration per month with E0 of the year: least squares of rref * f(TA)
            unit = pd.Series(lloyd_taylor(self.ta[in_year].to_numpy(), 1.0, e0), index=df.index[in_year])
            rref = pd.Series(rref_year, index=unit.index)
            for month in range(1, 13):
                fm = fit & (df.index.month == month)
                if fm.sum() >= self.min_records:
                    f = unit[fm[in_year].to_numpy()]
                    rref[unit.index.month == month] = (f * observed[fm]).sum() / (f * f).sum()
            self._wetnight_estimate.loc[unit.index] = unit * rref
            params[year] = {'E0': e0, 'RREF_YEAR': rref_year, 'N_DRY_NIGHTS': int(fit.sum())}
        self._wetnight_params = pd.DataFrame(params).T

        fill = (flag != 0) & self.night & self.wet & self._wetnight_estimate.notna()
        new_gf = gf_col.replace('_gfXG', GFSUFFIX) if '_gfXG' in gf_col else f'{gf_col}{GFSUFFIX}'
        new_flag = flag_col.replace(gf_col, new_gf)
        df.loc[fill, gf_col] = self._wetnight_estimate[fill]
        df.loc[fill, flag_col] = FLAG_WETNIGHT
        df.rename(columns={gf_col: new_gf, flag_col: new_flag}, inplace=True)
        self._gapfilled = df[new_gf].copy()
        self._flag_col = new_flag
        self._n_wetnight_filled = int(fill.sum())
        info(f"{self.target_col}: {self._n_wetnight_filled:,} wet night gaps filled from the Lloyd-Taylor "
             f"model ({len(params)} years fitted).", verbose=self.verbose)
        detail(f"Lloyd-Taylor parameters per year:\n{self._wetnight_params.round(3)}", verbose=self.verbose)
