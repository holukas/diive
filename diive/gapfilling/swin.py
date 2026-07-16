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
from diive.core.utils.console import console as _console, detail, info, rule, success, warn
from diive.gapfilling.xgboost_ts import XGBoostTS
from diive.variables.radiation import potrad


class SWINGapFillerXGBoost:
    """Physics-aware gap-filling for shortwave incoming radiation using XGBoost.

    Partitions the time series into daytime and nighttime using potential radiation:
    - Nighttime gaps are set to zero (no solar radiation after sunset, physically correct)
    - Daytime gaps are filled with XGBoost trained on daytime observations only

    By default, only potential radiation (SW_IN_POT) and timestamp features are used.
    No additional driver variables are required.  SW_IN_POT, calculated from site latitude and longitude,
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
            are included in feature engineering alongside SW_IN_POT. Column names must
            not collide with the target column name or with ``'SW_IN_POT'``. Default:
            None (only SW_IN_POT and timestamp features are used).
        nighttime_threshold: Potential-radiation cutoff (W m-2). Records with
            ``SW_IN_POT < nighttime_threshold`` are classified as nighttime; records
            with ``SW_IN_POT >= nighttime_threshold`` are daytime. Default: 0.001,
            which matches the threshold used internally by
            ``remove_nighttime_zero_offset`` so that the day/night split is
            consistent whether or not offset correction is enabled.
        correct_nighttime_offset: If True, apply remove_nighttime_zero_offset() to the
            series before gap-filling. This corrects sensors that measure small non-zero
            (often negative) values at night by subtracting the daily mean nighttime
            value as an offset from the whole series, then setting nighttime to zero.
            The corrected series is used for all subsequent gap-filling steps and is
            stored in gapfilling_df as '{target_col}_offset_corrected'. Default: False.
        interpolate_short_gaps: Maximum gap length, in records, to fill by
            interpolating the clearness index (SW_IN / SW_IN_POT) instead of the
            model. Interpolation never bridges a night. None (default) disables it,
            leaving every daytime gap to XGBoost.

            Worth enabling: the model cannot see the target's own neighbours, since
            the feature engineer excludes the target from every feature, so a short
            gap is exactly the case it is blind to. ``2`` (1 h on 30-min data) is the
            recommended limit. Measured on CH-DAV 30-min data with 15% scattered gaps,
            daytime gap RMSE against a model-only fill: 125 -> 88 at ``1``, 125 -> 77
            at ``2``, and no further gain above that — a scattered-gap record holds
            almost no longer runs. Raising the limit has little upside: interpolation
            still wins on gaps as long as 8 h (189 vs 219 W m-2) but collapses once a
            gap outlasts the observations bracketing it (1 day: 337 vs 172).
        reduce_features: Apply SHAP-based feature reduction after initial training.
            Removes features whose importance is at or below the random-noise baseline.
            Increases training time but can improve generalisation. Default: False.
        features_lag: ``[min_lag, max_lag]`` range for lag features of non-target
            columns (i.e. SW_IN_POT and any context drivers). Default: ``[-2, 2]``,
            which on 30-min data creates lags of -1h, -30min, +30min, +1h.
        features_rolling: Window sizes (in records) for rolling statistics. Default:
            ``[4, 8, 24, 48]`` — on 30-min data: 2h, 4h, 12h, 24h windows. Adjust to
            match your sampling frequency.
        features_rolling_stats: Extra rolling statistics beyond the default mean+std.
            Default: ``['median']``.
        features_ema: EMA spans (in records). Default: ``[6, 24]`` — short and
            day-scale memory on 30-min data.
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
        1 = daytime gap, filled by the XGBoost model
        2 = daytime gap, filled by the timestamp-only fallback model, i.e. a
            driver was missing here so the full model could not predict
        3 = nighttime gap, set to zero by physics
        4 = daytime gap, filled by clearness-index interpolation (only when
            interpolate_short_gaps is set)

        Values 0/1/2 carry the same meaning as everywhere else in diive (see
        GapFillingResult); 3 is specific to this class, which is the only
        gap-filler with a physics branch. Flag 2 matters most when *context_df*
        is used: it marks records where the extra drivers were unavailable and
        the fill therefore fell back to timestamps alone.

    Example:
        See examples/gapfilling/gapfill_swin.py for a complete worked example.
    """

    SWINPOT_COL = 'SW_IN_POT'
    FLAG_COL = 'flag'
    _DEFAULT_TARGET_NAME = 'SW_IN'

    # Below this potential radiation the clearness index SW_IN/SW_IN_POT divides by
    # a near-zero denominator and explodes, so it is not interpolated there. The
    # excluded band carries negligible energy.
    KT_MIN_SWINPOT = 50.0

    FLAG_OBSERVED = 0
    FLAG_MODEL = 1
    FLAG_FALLBACK = 2
    FLAG_NIGHTTIME_ZERO = 3
    FLAG_INTERPOLATED = 4

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 context_df: DataFrame = None,
                 nighttime_threshold: float = 0.001,
                 correct_nighttime_offset: bool = False,
                 interpolate_short_gaps: int = None,
                 reduce_features: bool = False,
                 features_lag: list = None,
                 features_rolling: list = None,
                 features_rolling_stats: list = None,
                 features_ema: list = None,
                 verbose: int = 0,
                 **kwargs):
        """Construct the gap-filler. See the class docstring for the full parameter list."""
        if series is None or series.empty:
            raise ValueError("series is empty — nothing to gap-fill.")
        if series.notna().sum() == 0:
            raise ValueError("series has no observed values — cannot train a model.")

        self.series = series.copy()
        self.target_col = self._resolve_target_col(self.series)

        # Reject target names that collide with reserved output columns.
        reserved = {self.SWINPOT_COL, self.FLAG_COL}
        if self.target_col in reserved:
            raise ValueError(
                f"series name '{self.target_col}' collides with a reserved output "
                f"column. Reserved names: {sorted(reserved)}. Please rename the series."
            )

        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset

        # Defensive copy of context_df (mirrors series copy above).
        if context_df is not None:
            if not isinstance(context_df, DataFrame):
                raise TypeError("context_df must be a pandas DataFrame.")
            self.context_df = context_df.copy()
        else:
            self.context_df = None

        self.nighttime_threshold = nighttime_threshold
        self.correct_nighttime_offset = correct_nighttime_offset
        if interpolate_short_gaps is not None and interpolate_short_gaps < 1:
            raise ValueError(
                f"interpolate_short_gaps must be >= 1 record or None, "
                f"got {interpolate_short_gaps}."
            )
        self.interpolate_short_gaps = interpolate_short_gaps
        self.reduce_features = reduce_features

        # Feature-engineering windows (configurable; defaults assume 30-min data).
        self.features_lag = [-2, 2] if features_lag is None else features_lag
        self.features_rolling = [4, 8, 24, 48] if features_rolling is None else features_rolling
        self.features_rolling_stats = (
            ['median'] if features_rolling_stats is None else features_rolling_stats
        )
        self.features_ema = [6, 24] if features_ema is None else features_ema

        self.verbose = verbose
        self.kwargs = kwargs

        self._results = None

    @staticmethod
    def _resolve_target_col(series: Series) -> str:
        """Pick the target column name from the series name, falling back to 'SW_IN'.

        Refuses tuple / non-string names because the gap-filler constructs derived
        column names via f-strings (e.g. ``f'{name}_gapfilled'``), and stringified
        tuples produce surprising, unstable column names.
        """
        name = series.name
        if name is None:
            return SWINGapFillerXGBoost._DEFAULT_TARGET_NAME
        if not isinstance(name, str):
            raise TypeError(
                f"series.name must be a string or None, got {type(name).__name__} "
                f"({name!r}). Rename the series with `series.rename('SW_IN')`."
            )
        return name

    @property
    def results(self) -> GapFillingResult:
        """GapFillingResult produced by :meth:`run` (raises if called before run)."""
        if self._results is None:
            raise RuntimeError("Call .run() before accessing .results")
        return self._results

    def report(self):
        """Formatted post-run summary: parameters, data & performance, flags, scores."""
        if self._results is None:
            raise RuntimeError("Call .run() before .report().")

        from rich.table import Table

        target_col = self.target_col
        df = self._results.gapfilling_df
        flag = self._results.flag
        swinpot = df[self.SWINPOT_COL]
        daytime_mask = swinpot >= self.nighttime_threshold

        n_total = len(df.index)
        n_day = int(daytime_mask.sum())
        n_night = n_total - n_day

        observed_before = self.series.notna()
        n_obs_before = int(observed_before.sum())
        n_obs_day = int((observed_before & daytime_mask).sum())
        n_obs_night = int((observed_before & ~daytime_mask).sum())
        n_gaps_before = n_total - n_obs_before
        n_gaps_day = int((~observed_before & daytime_mask).sum())
        n_gaps_night = int((~observed_before & ~daytime_mask).sum())

        n_filled_model = int((flag == self.FLAG_MODEL).sum())
        n_filled_fallback = int((flag == self.FLAG_FALLBACK).sum())
        n_filled_phys = int((flag == self.FLAG_NIGHTTIME_ZERO).sum())
        n_filled_interp = int((flag == self.FLAG_INTERPOLATED).sum())
        n_after = int(self._results.gapfilled.notna().sum())
        n_missing_after = n_total - n_after

        def pct(n, total):
            return 100.0 * n / total if total else 0.0

        rule(f"SW_IN Gap-Filling Report: {target_col}")
        _console.print(
            "  [bold]Algorithm:[/bold] physics-aware partitioning + XGBoost\n"
            "    Nighttime gaps  -> set to 0 W m-2 (no incoming solar radiation)\n"
            "    Daytime gaps    -> XGBoost trained on daytime observations\n"
            "    Day/night split -> SW_IN_POT (potential radiation) vs threshold"
        )

        rule("Parameters", min_level=2)
        _console.print(
            f"  Site latitude              {self.lat}\n"
            f"  Site longitude             {self.lon}\n"
            f"  UTC offset                 {self.utc_offset}\n"
            f"  Nighttime threshold        {self.nighttime_threshold} W m-2  "
            f"(SW_IN_POT < threshold -> night)\n"
            f"  Correct nighttime offset   {self.correct_nighttime_offset}\n"
            f"  Reduce features (SHAP)     {self.reduce_features}\n"
            f"  features_lag               {self.features_lag}\n"
            f"  features_rolling           {self.features_rolling}\n"
            f"  features_rolling_stats     {self.features_rolling_stats}\n"
            f"  features_ema               {self.features_ema}"
        )

        rule("Data & Performance", min_level=2)
        _console.print(
            f"  Total records              {n_total:>10,d}\n"
            f"  Daytime records            {n_day:>10,d}  ({pct(n_day, n_total):.1f}%)\n"
            f"  Nighttime records          {n_night:>10,d}  ({pct(n_night, n_total):.1f}%)\n"
            f"\n"
            f"  Observed before            {n_obs_before:>10,d}  "
            f"({pct(n_obs_before, n_total):.1f}%)\n"
            f"    of which daytime         {n_obs_day:>10,d}\n"
            f"    of which nighttime       {n_obs_night:>10,d}\n"
            f"  Gaps before                {n_gaps_before:>10,d}  "
            f"({pct(n_gaps_before, n_total):.1f}%)\n"
            f"    daytime gaps             {n_gaps_day:>10,d}\n"
            f"    nighttime gaps           {n_gaps_night:>10,d}\n"
            f"\n"
            f"  Filled by XGBoost          {n_filled_model:>10,d}  "
            f"({pct(n_filled_model, max(n_gaps_before, 1)):.1f}% of gaps)\n"
            f"  Filled by fallback         {n_filled_fallback:>10,d}  "
            f"({pct(n_filled_fallback, max(n_gaps_before, 1)):.1f}% of gaps)\n"
            f"  Filled by physics (=0)     {n_filled_phys:>10,d}  "
            f"({pct(n_filled_phys, max(n_gaps_before, 1)):.1f}% of gaps)\n"
            f"  Filled by interpolation    {n_filled_interp:>10,d}  "
            f"({pct(n_filled_interp, max(n_gaps_before, 1)):.1f}% of gaps)\n"
            f"  Remaining missing          {n_missing_after:>10,d}\n"
            f"  Final coverage             {n_after:>10,d}  "
            f"({pct(n_after, n_total):.1f}%)"
        )

        rule("Flag Distribution", min_level=2)
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        table.add_column("Flag", style="dim", no_wrap=True)
        table.add_column("Count", justify="right")
        table.add_column("  %", justify="right")
        table.add_column("Meaning")
        flag_meanings = {
            self.FLAG_OBSERVED: "observed",
            self.FLAG_MODEL: "gap-filled by XGBoost (daytime)",
            self.FLAG_FALLBACK: "gap-filled by fallback, driver missing (daytime)",
            self.FLAG_NIGHTTIME_ZERO: "gap-filled by physics (nighttime = 0)",
            self.FLAG_INTERPOLATED: "gap-filled by clearness-index interpolation",
        }
        for f_val in flag_meanings:
            count = int((flag == f_val).sum())
            table.add_row(
                str(f_val),
                f"{count:,d}",
                f"{pct(count, n_total):.1f}%",
                flag_meanings[f_val],
            )
        _console.print(table)

        rule("Daytime Model Scores", min_level=2)
        scores = self._results.scores or {}
        if scores:
            for score, val in scores.items():
                score_display = score.replace('_', ' ').upper()
                _console.print(f"  {score_display:<8} {val:.4f}")
        else:
            _console.print("  No XGBoost scores — no daytime gaps to fill.")

    def run(self) -> 'SWINGapFillerXGBoost':
        """Execute gap-filling: optional offset correction, nighttime zeros, daytime XGBoost.

        Returns:
            self — for method chaining.
        """
        target_col = self.target_col

        if self.verbose >= 1:
            rule(f"SW_IN Gap-Filling ({target_col})")

        # Capture the original gap mask BEFORE any correction is applied.
        # remove_nighttime_zero_offset() zeros ALL nighttime positions (including NaN),
        # so using working_series.isna() afterwards would miss those nighttime gaps
        # and assign them flag=0 (observed) instead of flag=2 (physics fill).
        original_gaps = self.series.isna()

        # Optional: correct nighttime sensor offset before gap-filling.
        # Imported here to avoid loading the corrections module unless needed.
        series_corrected = None
        if self.correct_nighttime_offset:
            from diive.preprocessing.corrections.offsetcorrection import (
                remove_nighttime_zero_offset,
            )
            if self.verbose >= 1:
                info("Applying nighttime offset correction ...")
            series_corrected = remove_nighttime_zero_offset(
                series=self.series.copy(),  # copy to prevent mutation of self.series.name
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

        gaps = original_gaps
        nighttime_gaps = gaps & ~daytime_mask
        daytime_gaps = gaps & daytime_mask

        if self.verbose >= 1:
            info(f"Records: {daytime_mask.sum()} daytime | {(~daytime_mask).sum()} nighttime")
            info(f"Gaps: {daytime_gaps.sum()} daytime | {nighttime_gaps.sum()} nighttime")

        # Nighttime: zero is the physically correct value (no solar radiation).
        filled = working_series.copy()
        filled.loc[nighttime_gaps] = 0.0

        # Short daytime gaps: interpolate the clearness index. Computed here but
        # deliberately NOT written into `filled` yet — the model must train on
        # observations only, never on this interpolation.
        interpolated = None
        if self.interpolate_short_gaps:
            interpolated = self._interpolate_short_gaps(working_series, swinpot)
            if self.verbose >= 1:
                info(f"Interpolated {int(interpolated.notna().sum())} short daytime "
                     f"gap records (<= {self.interpolate_short_gaps} records).")

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
        else:
            if self.verbose >= 1:
                info("No daytime gaps found — XGBoost step skipped.")
            if self.reduce_features and self.verbose >= 1:
                info("reduce_features=True has no effect when there are no daytime gaps.")

        # Interpolation wins over the model on the gaps it covers, so it is applied
        # last. The model has no access to the target's own neighbours (the feature
        # engineer excludes the target), which is exactly the information a short
        # gap turns on.
        if interpolated is not None:
            interp_locs = interpolated.notna()
            filled.loc[interp_locs] = interpolated[interp_locs]

        # Make sure the published gap-filled series carries the public name,
        # not the XGBoost-internal '_gfXG' suffix that XGBoostTS attaches.
        filled.name = target_col

        # Flags: see the class docstring. The daytime model already emits 0/1/2
        # on the shared GapFillingResult scale, so its flags carry over as-is and
        # the model-vs-fallback distinction survives into the published flag.
        flag = pd.Series(index=working_series.index, data=self.FLAG_OBSERVED,
                         dtype=int, name=self.FLAG_COL)
        flag.loc[nighttime_gaps] = self.FLAG_NIGHTTIME_ZERO
        if daytime_results is not None:
            # Index-aligned assignment (consistent with the `filled` assignment above).
            flag.loc[daytime_results.flag.index] = daytime_results.flag
        if interpolated is not None:
            # After the model flags, mirroring the value assignment order above.
            flag.loc[interpolated.notna()] = self.FLAG_INTERPOLATED

        # Build the results DataFrame.  Include the offset-corrected series
        # when the correction was applied so the user can inspect the before/after.
        gf_dict = {target_col: self.series}
        if series_corrected is not None:
            gf_dict[f'{target_col}_offset_corrected'] = series_corrected
        gf_dict[f'{target_col}_gapfilled'] = filled
        gf_dict[self.SWINPOT_COL] = swinpot
        gf_dict[self.FLAG_COL] = flag
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

    def _interpolate_short_gaps(self, series: Series, swinpot: Series) -> Series:
        """Interpolate short daytime gaps in clearness-index space.

        SW_IN is solar geometry times sky state. Interpolating W m-2 directly fights
        the diurnal ramp — a straight line across a morning gap undershoots a steeply
        climbing curve — so this divides by SW_IN_POT first, interpolates the sky
        state (the slowly varying part), and multiplies the geometry back in.

        Interpolation runs per calendar day, which is what stops it bridging a night:
        midnight is dark at the latitudes this targets, so a night always falls on a
        day boundary, and the sky state on the far side of one is not recoverable
        from the near side.

        Returns:
            Series carrying interpolated values at accepted gaps, NaN everywhere
            else — including gaps that were too long or could not be anchored.
        """
        limit = self.interpolate_short_gaps
        kt = series / swinpot.where(swinpot >= self.KT_MIN_SWINPOT)

        interp = kt.groupby(kt.index.normalize()).transform(
            lambda day: day.interpolate(method='time', limit_area='inside')
        )

        # Accept whole gaps only. pandas' interpolate(limit=) fills the first `limit`
        # records of a longer gap and leaves a ragged tail, so select runs explicitly.
        missing = kt.isna()
        run_length = missing.groupby((~missing).cumsum()).transform('sum')

        # `series.isna()` guards observed records: kt is also NaN wherever SW_IN_POT
        # is below the floor, and those must never be overwritten.
        accepted = missing & (run_length <= limit) & interp.notna() & series.isna()
        return (interp * swinpot).where(accepted)

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
            if not self.context_df.index.equals(series.index):
                raise ValueError(
                    "context_df index does not match series index — "
                    "both must share the same DatetimeIndex."
                )
            reserved = {target_col, self.SWINPOT_COL}
            collisions = sorted(reserved.intersection(self.context_df.columns))
            if collisions:
                raise ValueError(
                    f"context_df contains column(s) {collisions} that collide with "
                    f"the target or SW_IN_POT. Rename or drop them before passing."
                )
            for col in self.context_df.columns:
                input_df[col] = self.context_df[col]

        # Feature engineering on the FULL index so lag/rolling windows are
        # correct at the dawn/dusk boundaries when we later subset to daytime.
        engineer = FeatureEngineer(
            target_col=target_col,
            features_lag=self.features_lag,
            features_rolling=self.features_rolling,
            features_rolling_stats=self.features_rolling_stats,
            features_ema=self.features_ema,
            vectorize_timestamps=True,
            verbose=self.verbose,
        )
        full_features_df = engineer.fit_transform(input_df)
        detail(
            f"FeatureEngineer produced {full_features_df.shape[1] - 1} feature columns "
            f"over {len(full_features_df)} rows.",
            verbose=self.verbose,
        )

        # Subset to daytime rows only.
        daytime_df = full_features_df.loc[daytime_mask].copy()
        n_complete = int(daytime_df[target_col].notna().sum())
        detail(
            f"Daytime subset: {len(daytime_df)} rows, {n_complete} with observed target.",
            verbose=self.verbose,
        )
        if n_complete < 20:
            warn(f"Only {n_complete} complete daytime records available for XGBoost training.",
                 verbose=self.verbose)

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
            model.trainmodel(showplot_scores=False, showplot_importance=False)

        model.fillgaps(showplot_scores=False, showplot_importance=False)

        return model.results
