"""
GAP-FILLING: SHORTWAVE INCOMING RADIATION (SW_IN)
=================================================

Physics-aware gap-filling for shortwave incoming radiation.
Uses potential radiation to partition daytime and nighttime:
nighttime gaps are set to zero (physically correct),
daytime gaps are filled with XGBoost trained on daytime data only.

Part of the diive library: https://github.com/holukas/diive
"""

import pandas as pd
from pandas import DataFrame, Series

from diive.core.ml.feature_engineer import FeatureEngineer
from diive.core.ml.results import GapFillingResult
from diive.core.utils.console import info, rule, success, warn
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.variables.radiation import potrad


class SWINGapFillerXGBoost:
    """Physics-aware gap-filling for shortwave incoming radiation using XGBoost.

    Partitions the time series into daytime and nighttime using potential radiation:
    - Nighttime gaps are set to zero (no solar radiation after sunset, physically correct)
    - Daytime gaps are filled with XGBoost trained on daytime observations only

    By default, only potential radiation (SW_IN_POT) and timestamp features are used,
    together with lag and rolling features of SW_IN itself.  No additional driver
    variables are required.  SW_IN_POT, calculated from site latitude and longitude,
    encodes solar angle, day length, and seasonal amplitude — the dominant drivers of
    SW_IN variability — and is the single most important predictor.

    Feature engineering is applied to the full time series before subsetting to daytime,
    so lag and rolling features correctly span day/night boundaries.

    Args:
        series: SW_IN time series to gap-fill (W m-2). NaN values are gaps.
        lat: Site latitude in degrees North (-90 to 90).
        lon: Site longitude in degrees East (-180 to 180).
        utc_offset: UTC offset of the timestamp index, e.g. 1 for UTC+01:00.
        context_df: Optional DataFrame of additional driver variables (e.g. TA, VPD).
            Must share the same DatetimeIndex as *series*. When provided, these columns
            are included in feature engineering alongside SW_IN_POT. Default: None
            (only SW_IN_POT and timestamp features are used).
        nighttime_threshold: Potential radiation threshold below which a record is
            classified as nighttime (W m-2). Default: 20.
        correct_nighttime_offset: If True, apply remove_radiation_zero_offset() to the
            series before gap-filling. This corrects sensors that measure small non-zero
            (often negative) values at night by subtracting the daily mean nighttime
            value as an offset from the whole series, then setting nighttime to zero.
            The corrected series is used for all subsequent gap-filling steps and is
            stored in gapfilling_df as '{target_col}_offset_corrected'. Default: False.
        reduce_features: Apply SHAP-based feature reduction after initial training.
            Removes features whose importance is at or below the random-noise baseline.
            Increases training time but can improve generalisation. Default: False.
        verbose: Verbosity level: 0=silent, 1=progress, 2+=detailed. Default: 0.
        **kwargs: XGBoost hyperparameters forwarded to XGBRegressor (n_estimators,
            max_depth, learning_rate, subsample, colsample_bytree, random_state, etc.).

    Methods:
        run(): Execute the full gap-filling workflow. Returns self for chaining.

    Attributes:
        results: GapFillingResult populated after run(). Contains gapfilled series,
            flags, scores, SHAP importances, and the trained XGBoost model.

    Result flags:
        0 = observed (any period)
        1 = gap-filled by XGBoost (daytime ML or daytime fallback)
        2 = gap-filled by physics (nighttime set to zero)

    Example:
        See examples/gapfilling/gapfill_swin.py for a complete worked example.
    """

    SWINPOT_COL = 'SW_IN_POT'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 context_df: DataFrame = None,
                 nighttime_threshold: float = 20,
                 correct_nighttime_offset: bool = False,
                 reduce_features: bool = False,
                 verbose: int = 0,
                 **kwargs):
        self.series = series.copy()
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.context_df = context_df
        self.nighttime_threshold = nighttime_threshold
        self.correct_nighttime_offset = correct_nighttime_offset
        self.reduce_features = reduce_features
        self.verbose = verbose
        self.kwargs = kwargs

        self._results = None

    @property
    def results(self) -> GapFillingResult:
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results")
        return self._results

    def run(self) -> 'SWINGapFillerXGBoost':
        """Execute gap-filling: optional offset correction, nighttime zeros, daytime XGBoost.

        Returns:
            self — for method chaining.
        """
        target_col = self.series.name or 'SW_IN'

        if self.verbose >= 1:
            rule(f"SW_IN Gap-Filling ({target_col})")

        # Optional: correct nighttime sensor offset before gap-filling.
        # Imported here to avoid loading the corrections module unless needed.
        series_corrected = None
        if self.correct_nighttime_offset:
            from diive.preprocessing.corrections.offsetcorrection import (
                remove_radiation_zero_offset,
            )
            if self.verbose >= 1:
                info("Applying nighttime offset correction ...")
            series_corrected = remove_radiation_zero_offset(
                series=self.series,
                lat=self.lat,
                lon=self.lon,
                utc_offset=self.utc_offset,
                showplot=False,
            )

        # The working series is the corrected one (if requested) or the original.
        working_series = series_corrected if series_corrected is not None else self.series.copy()

        # Potential radiation drives the daytime/nighttime split and is the
        # primary feature for daytime prediction.
        swinpot = potrad(
            timestamp_index=working_series.index,
            lat=self.lat,
            lon=self.lon,
            utc_offset=self.utc_offset,
        )
        daytime_mask = swinpot >= self.nighttime_threshold

        gaps = working_series.isna()
        nighttime_gaps = gaps & ~daytime_mask
        daytime_gaps = gaps & daytime_mask

        if self.verbose >= 1:
            info(f"Records: {daytime_mask.sum()} daytime | {(~daytime_mask).sum()} nighttime")
            info(f"Gaps: {daytime_gaps.sum()} daytime | {nighttime_gaps.sum()} nighttime")

        # Nighttime: zero is the physically correct value (no solar radiation).
        filled = working_series.copy()
        filled.loc[nighttime_gaps] = 0.0

        # Daytime: XGBoost trained on observed daytime values.
        daytime_results = None
        if daytime_gaps.sum() > 0:
            daytime_results = self._fill_daytime(
                series=filled,
                swinpot=swinpot,
                daytime_mask=daytime_mask,
                target_col=target_col,
            )
            # Overwrite daytime rows with the model's gapfilled output.
            # gapfilled preserves observed values and fills only gaps.
            filled.loc[daytime_results.gapfilled.index] = daytime_results.gapfilled
        elif self.verbose >= 1:
            info("No daytime gaps found — XGBoost step skipped.")

        # Flags: 0=observed, 1=gap-filled by XGBoost, 2=nighttime set to zero.
        flag = pd.Series(index=working_series.index, data=0, dtype=int, name='flag')
        flag.loc[nighttime_gaps] = 2
        if daytime_results is not None:
            # Daytime model returns 0=observed, 1=gap-filled, 2=fallback.
            # Re-encode fallback (2) as 1 to keep a clean three-level scheme
            # where 2 means nighttime physics fill.
            daytime_flag = daytime_results.flag.copy()
            daytime_flag[daytime_flag == 2] = 1
            flag.loc[daytime_flag.index] = daytime_flag.values

        # Build the results DataFrame.  Include the offset-corrected series
        # when the correction was applied so the user can inspect the before/after.
        gf_dict = {target_col: self.series}
        if series_corrected is not None:
            gf_dict[f'{target_col}_offset_corrected'] = series_corrected
        gf_dict[f'{target_col}_gapfilled'] = filled
        gf_dict[self.SWINPOT_COL] = swinpot
        gf_dict['flag'] = flag
        gapfilling_df = pd.DataFrame(gf_dict)

        self._results = GapFillingResult(
            gapfilled=filled,
            flag=flag,
            scores=daytime_results.scores if daytime_results else {},
            gapfilling_df=gapfilling_df,
            scores_traintest=daytime_results.scores_traintest if daytime_results else None,
            feature_importances=daytime_results.feature_importances if daytime_results else None,
            feature_importances_traintest=(
                daytime_results.feature_importances_traintest if daytime_results else None
            ),
            model=daytime_results.model if daytime_results else None,
            accepted_features=daytime_results.accepted_features if daytime_results else None,
            rejected_features=daytime_results.rejected_features if daytime_results else None,
        )

        if self.verbose >= 1:
            total_filled = (flag > 0).sum()
            success(f"Done — {total_filled} records filled ({gaps.sum()} gaps total)")

        return self

    def _fill_daytime(self,
                      series: Series,
                      swinpot: Series,
                      daytime_mask: Series,
                      target_col: str) -> GapFillingResult:
        """Build feature matrix and run XGBoost on the daytime subset."""

        # Assemble input: target + SW_IN_POT + optional context drivers.
        # Using the nighttime-zero-filled series for rolling/lag features
        # is intentional: it gives physically correct context (zero at night)
        # for features that span the day/night boundary.
        input_df = pd.DataFrame({target_col: series, self.SWINPOT_COL: swinpot})
        if self.context_df is not None:
            for col in self.context_df.columns:
                input_df[col] = self.context_df[col]

        # Feature engineering on the FULL index so lag/rolling windows are
        # correct at the dawn/dusk boundaries when we later subset to daytime.
        engineer = FeatureEngineer(
            target_col=target_col,
            features_lag=[-2, 2],
            features_rolling=[4, 8, 24, 48],
            features_rolling_stats=['median'],
            features_ema=[6, 24],
            vectorize_timestamps=True,
            verbose=self.verbose,
        )
        full_features_df = engineer.fit_transform(input_df)

        # Subset to daytime rows only.
        daytime_df = full_features_df.loc[daytime_mask].copy()

        n_complete = int(daytime_df[target_col].notna().sum())
        if n_complete < 20:
            warn(f"Only {n_complete} complete daytime records available for XGBoost training.")

        model = XGBoostTS(
            input_df=daytime_df,
            target_col=target_col,
            verbose=self.verbose,
            below_zero='zero',
            **self.kwargs,
        )

        model.trainmodel(showplot_scores=False, showplot_importance=False)

        if self.reduce_features:
            model.reduce_features()

        model.fillgaps(showplot_scores=False, showplot_importance=False)

        return model.results
