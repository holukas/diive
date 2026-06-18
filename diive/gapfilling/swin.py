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
        1 = gap-filled by XGBoost (daytime ML or daytime fallback)
        2 = gap-filled by physics (nighttime set to zero)

    Example:
        See examples/gapfilling/gapfill_swin.py for a complete worked example.
    """

    SWINPOT_COL = 'SW_IN_POT'
    FLAG_COL = 'flag'
    _DEFAULT_TARGET_NAME = 'SW_IN'

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 context_df: DataFrame = None,
                 nighttime_threshold: float = 0.001,
                 correct_nighttime_offset: bool = False,
                 reduce_features: bool = False,
                 features_lag: list = None,
                 features_rolling: list = None,
                 features_rolling_stats: list = None,
                 features_ema: list = None,
                 verbose: int = 0,
                 **kwargs):
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

        n_filled_xgb = int((flag == 1).sum())
        n_filled_phys = int((flag == 2).sum())
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
            f"  Filled by XGBoost          {n_filled_xgb:>10,d}  "
            f"({pct(n_filled_xgb, max(n_gaps_before, 1)):.1f}% of gaps)\n"
            f"  Filled by physics (=0)     {n_filled_phys:>10,d}  "
            f"({pct(n_filled_phys, max(n_gaps_before, 1)):.1f}% of gaps)\n"
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
            0: "observed",
            1: "gap-filled by XGBoost (daytime)",
            2: "gap-filled by physics (nighttime = 0)",
        }
        for f_val in (0, 1, 2):
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

        # Make sure the published gap-filled series carries the public name,
        # not the XGBoost-internal '_gfXG' suffix that XGBoostTS attaches.
        filled.name = target_col

        # Flags: 0=observed, 1=gap-filled by XGBoost, 2=nighttime set to zero.
        flag = pd.Series(index=working_series.index, data=0, dtype=int, name=self.FLAG_COL)
        flag.loc[nighttime_gaps] = 2
        if daytime_results is not None:
            # Daytime model returns 0=observed, 1=gap-filled, 2=fallback.
            # Re-encode fallback (2) as 1 to keep a clean three-level scheme
            # where 2 means nighttime physics fill.
            daytime_flag = daytime_results.flag.copy()
            daytime_flag[daytime_flag == 2] = 1
            # Index-aligned assignment (consistent with the `filled` assignment above).
            flag.loc[daytime_flag.index] = daytime_flag

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
            model.trainmodel(showplot_scores=False, showplot_importance=False)

        model.fillgaps(showplot_scores=False, showplot_importance=False)

        return model.results
