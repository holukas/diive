"""
GAP-FILLING: SHORTWAVE INCOMING RADIATION (SW_IN)
=================================================

Physics-aware gap-filling for shortwave incoming radiation.
Uses potential radiation to partition daytime and nighttime:
nighttime gaps are set to zero (physically correct),
daytime gaps are filled with XGBoost trained on daytime data only.

With no context drivers every feature is a deterministic function of the
timestamp, so the daytime model reproduces a climatology and cannot know
whether a gap was overcast or clear. A second radiation measurement is the
one routinely available input that breaks that ceiling; see the
SWINGapFillerXGBoost docstring.

Part of the diive library: https://github.com/holukas/diive
"""

import inspect
from contextlib import contextmanager
from time import perf_counter

import pandas as pd
from pandas import DataFrame, Series

from diive.core.ml.feature_engineer import FeatureEngineer
from diive.core.ml.results import GapFillingResult

# Read off the real signature so this stays correct as FeatureEngineer grows.
_FEATURE_ENGINEER_PARAMS = frozenset(
    inspect.signature(FeatureEngineer.__init__).parameters) - {'self'}
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
    so rolling features correctly span day/night boundaries.

    What the default configuration can and cannot do:
        With no *context_df*, every feature is a deterministic function of the
        timestamp — SW_IN_POT, the timestamp features and the record number are all
        fixed once the time is known. The model therefore reproduces a climatology, the
        expected SW_IN for that time of day and year, and has no way to know whether a
        given gap was overcast or clear. That is the accuracy ceiling, and no further
        feature derived from the timestamp can raise it: on CH-DAV 30-min data,
        SW_IN_POT plus timestamps (15 features) scores the same daytime-gap RMSE as
        the same set expanded with rolling and EMA variants of SW_IN_POT (29 features)
        — 138 W m-2 either way.

        Two things do raise it, and both are worth more than any feature tuning:

        - A second radiation measurement passed through *context_df* — a co-located
          pyranometer, a PPFD sensor, or a nearby station. It is the only routinely
          available source of sky-state information. On CH-DAV, adding PPFD moved
          daytime-gap RMSE from 138 to 26 W m-2 wherever that second sensor was
          available.
        - *interpolate_short_gaps* for gaps of an hour or two, which uses the target's
          own neighbours — information the model never sees, since the feature engineer
          excludes the target from every feature. It is off by default: it helps under
          the ceiling but overwrites better model fills once a strong *context_df*
          sensor is present, so enable it only in the no-context case.

    Args:
        series: SW_IN time series to gap-fill (W m-2). NaN values are gaps.
        lat: Site latitude in degrees North (-90 to 90).
        lon: Site longitude in degrees East (-180 to 180).
        utc_offset: UTC offset of the timestamp index, e.g. 1 for UTC+01:00.
        context_df: Optional DataFrame of additional driver variables. Must share the
            same DatetimeIndex as *series*. When provided, these columns are included
            in feature engineering alongside SW_IN_POT. Column names must not collide
            with the target column name or with ``'SW_IN_POT'``. Default: None (only
            SW_IN_POT and timestamp features are used).

            The most valuable thing to put here by far is a **second radiation
            measurement** — a co-located pyranometer, a PPFD sensor, or a nearby
            station. Unlike TA or VPD it measures the sky state directly, which is
            exactly what the timestamp cannot supply (see above). It does not need to
            be in W m-2 or gap-free: the model learns the relationship from whatever
            overlap exists, and records where it is missing simply fall back to the
            climatology the default configuration would have produced anyway.

            Preference order for a context_df radiation source, best first: a
            co-located second sensor (pyranometer or PPFD) that sees the same sky;
            a nearby station's radiation if the site is climatically similar; and,
            where no local or neighbouring sensor exists, satellite or reanalysis
            SW_IN such as ERA5-Land. All wire in the same way through context_df;
            each carries synoptic sky state the timestamp cannot, so even a coarse
            reanalysis product beats the timestamp-only climatology.
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
            stored in gapfilling_df as '{target_col}_offset_corrected'. Default: True.
            On an already quality-controlled series this is close to a no-op; on a raw
            pyranometer record it removes the thermal-offset bias. Set to False if the
            input is already offset-corrected.
        interpolate_short_gaps: Maximum gap length, in records, to fill by
            interpolating the clearness index (SW_IN / SW_IN_POT) instead of the
            model. Interpolation never bridges a night.

            Default: ``'auto'`` — enabled at a 2-record limit when *context_df* is
            None, disabled when a context driver is given, which is the branch that
            measured better in each case (see below). Pass an integer for an explicit
            record limit, or None to disable it outright.

            Worth enabling when the model is climatology-bound (no context_df): the
            model cannot see the target's own neighbours — the feature engineer
            excludes the target from every feature — so a short gap is exactly the case
            it is blind to. ``2`` (1 h on 30-min data) is the recommended limit.
            Measured on CH-DAV 30-min data with 15% scattered gaps, daytime gap RMSE
            against a model-only fill: 125 -> 88 at ``1``, 125 -> 77 at ``2``, and no
            further gain above that — a scattered-gap record holds almost no longer
            runs. Raising the limit has little upside: interpolation still wins on gaps
            as long as 8 h (189 vs 219 W m-2) but collapses once a gap outlasts the
            observations bracketing it (1 day: 337 vs 172).

            And the reason ``'auto'`` turns it off once a context driver is present:
            interpolation overwrites the model on the gaps it covers. That is a gain
            under the ceiling, but a strong, near-complete second radiation sensor in
            context_df resolves short gaps better than clearness-index interpolation
            does, so enabling it there *raises* RMSE (CH-DAV: context-only 13.5 vs
            context+interp 66 W m-2). Note that ``'auto'`` keys off whether context_df
            was passed at all, not off how good or complete that driver is — with a
            weak or very gappy context sensor, an explicit integer may beat it.
        reduce_features: Apply SHAP-based feature reduction after initial training.
            Removes features whose importance is at or below the random-noise baseline.
            Increases training time but can improve generalisation. Default: False.
        feature_kwargs: FeatureEngineer arguments, overriding the SW_IN defaults in
            ``_FE_DEFAULTS`` (see that constant for the values and the reasoning).
            Every FeatureEngineer parameter is reachable this way, including the
            diff, polynomial and STL stages, which are off by default. ``target_col``
            and ``verbose`` are set by this class and rejected here. Window sizes are
            in records and assume 30-min data — scale them to your frequency.

            Three of the defaults are worth understanding before overriding them:

            ``features_lag=[]``. Lag features are a bad trade here. The model
            predicts only records where every feature is present and sends the rest
            to a timestamp-only fallback, so a lag converts "this record's neighbour
            is missing" into "this record cannot be filled properly". On CH-DAV with
            a PPFD reference, ``[-2, 2]`` pushed 1157 of 2556 otherwise-fillable
            records to the fallback and raised their RMSE from 26 to 97 W m-2, on top
            of costing 20% of the training rows. On SW_IN_POT lags are harmless (it
            never has gaps) but useless, being the same function of the timestamp.
            Set them only for a gap-free driver where they earn their keep.

            ``features_rolling``/``features_ema`` exclude SW_IN_POT. Rolling and EMA
            variants of a deterministic timestamp curve carry nothing beyond its raw
            value and the timestamp features. Both stages therefore act on context
            drivers only, and with no *context_df* they have no effect at all.

            ``add_continuous_record_number=True``. Lets the model isolate periods —
            a sensor swap, progressive soiling, a calibration change. Safe for
            gap-filling, which only interpolates within the record, so the trees
            never extrapolate it. Measured neutral on CH-DAV's quality-controlled
            Rg_f, which has no drift to find; cheap insurance on long raw records.

            Example::

                feature_kwargs={'features_rolling': [2, 4, 12],  # 15-min data
                                'features_diff': [1]}            # enable diff stage
        verbose: Verbosity level: 0=silent, 1=progress, 2+=detailed. Default: 0.
        **kwargs: XGBoost hyperparameters forwarded to XGBRegressor (n_estimators,
            max_depth, learning_rate, subsample, colsample_bytree, random_state, etc.),
            overriding the SW_IN defaults in ``_XGB_DEFAULTS``: n_estimators=3000,
            max_depth=6, early_stopping_rounds=20. If you raise n_estimators, keep
            early stopping on — see the note at ``_XGB_DEFAULTS`` for why.

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

    # XGBoost defaults for SW_IN, overridable through **kwargs.
    #
    # depth 6 with a large tree budget and early stopping beats a small fixed
    # budget by a wide margin on daytime-gap RMSE (CH-DAV, 10 years, 20% gaps:
    # 133 -> 110 W/m2 with no context, 25 -> 23 with a PPFD context sensor).
    #
    # early_stopping_rounds is NOT optional here. Without it all 3000 trees are
    # built, which barely improves RMSE (110.1 -> 110.6) but makes the SHAP pass
    # 3-7x slower, since TreeSHAP cost is linear in tree count. 20 rounds lands
    # at ~600-1000 trees and captures the full accuracy of the 3000-tree model.
    # random_state is pinned so a gap-fill is reproducible: without it XGBoost
    # seeds itself and every run returns different fills, scores and SHAP
    # importances. Pass random_state=None explicitly to opt back out.
    _XGB_DEFAULTS = {
        'n_estimators': 3000,
        'max_depth': 6,
        'early_stopping_rounds': 20,
        'random_state': 42,
    }

    # SHAP over the whole record is the slowest step once trees number in the
    # hundreds. Mean |SHAP| converges well before that: at 10k rows the feature
    # ranking was identical to the full record (Kendall tau 1.000, importances
    # within 2%). Raise or set to None for importances over every row.
    _SHAP_MAX_ROWS = 10_000

    # Gap length that interpolate_short_gaps='auto' fills, in records.
    _INTERP_AUTO_RECORDS = 2

    # FeatureEngineer defaults for SW_IN, overridable through feature_kwargs.
    # Window sizes are record counts and assume 30-min data (2h, 4h, 12h, 24h
    # rolling; 3h and 12h EMA); scale them yourself for another frequency.
    #
    # Two of these are load-bearing rather than merely tuned:
    #   - features_lag is empty because the model only predicts rows where every
    #     feature is present. A lag turns "my neighbour is missing" into "I cannot
    #     be filled", demoting records to the timestamp-only fallback.
    #   - SW_IN_POT is excluded from the rolling and EMA stages: rolling variants
    #     of a deterministic timestamp curve carry nothing its raw value and the
    #     timestamp features do not already have.
    # The diff, polynomial and STL stages stay off; reach them via feature_kwargs.
    _FE_DEFAULTS = {
        'features_lag': [],
        'features_rolling': [4, 8, 24, 48],
        'features_rolling_stats': ['median'],
        'features_rolling_exclude_cols': [SWINPOT_COL],
        'features_ema': [6, 24],
        'features_ema_exclude_cols': [SWINPOT_COL],
        'vectorize_timestamps': True,
        'add_continuous_record_number': True,
    }

    # Set by this class per run, so a caller cannot pass them through feature_kwargs.
    _FE_RESERVED = ('target_col', 'verbose')

    def __init__(self,
                 series: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 context_df: DataFrame = None,
                 nighttime_threshold: float = 0.001,
                 correct_nighttime_offset: bool = True,
                 interpolate_short_gaps: int | str | None = 'auto',
                 reduce_features: bool = False,
                 feature_kwargs: dict = None,
                 verbose: int = 0,
                 **kwargs):
        """Construct the gap-filler. See the class docstring for the full parameter list."""
        if series is None or series.empty:
            raise ValueError("series is empty — nothing to gap-fill.")
        if series.notna().sum() == 0:
            raise ValueError("series has no observed values — cannot train a model.")

        self.verbose = verbose
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
        if (interpolate_short_gaps is not None
                and interpolate_short_gaps != 'auto'
                and interpolate_short_gaps < 1):
            raise ValueError(
                f"interpolate_short_gaps must be >= 1 record, None or 'auto', "
                f"got {interpolate_short_gaps}."
            )
        self._interpolate_short_gaps_arg = interpolate_short_gaps
        self.reduce_features = reduce_features

        # Feature engineering: same merge pattern as the XGBoost kwargs below.
        # Reject the two the class sets itself rather than let them collide as an
        # opaque "multiple values for keyword argument" TypeError at call time.
        feature_kwargs = feature_kwargs or {}
        clashes = sorted(set(feature_kwargs) & set(self._FE_RESERVED))
        if clashes:
            raise ValueError(
                f"feature_kwargs may not set {clashes}: this class sets "
                f"{list(self._FE_RESERVED)} itself. Pass the rest of the "
                f"FeatureEngineer arguments instead."
            )
        merged_fe = {**self._FE_DEFAULTS, **feature_kwargs}
        # Copy list values so one instance cannot mutate the shared class defaults.
        self.feature_kwargs = {k: (list(v) if isinstance(v, list) else v)
                               for k, v in merged_fe.items()}

        # A FeatureEngineer argument passed at the top level would otherwise land in
        # **kwargs and go to XGBRegressor, which ignores unknown parameters with only
        # a warning — a silently ineffective setting. Point the caller at the dict.
        misplaced = sorted(set(kwargs) & _FEATURE_ENGINEER_PARAMS)
        if misplaced:
            raise TypeError(
                f"{misplaced} are FeatureEngineer arguments and must be passed in "
                f"feature_kwargs, not as top-level keywords (top-level keywords go "
                f"to XGBRegressor). Use feature_kwargs={{{misplaced[0]!r}: ...}}."
            )

        # 'auto' picks the branch that measured better for the data at hand: short-gap
        # interpolation beats a climatology-bound model on the gaps it covers, but
        # loses to a real context sensor, which resolves those same gaps better.
        if self._interpolate_short_gaps_arg == 'auto':
            self.interpolate_short_gaps = (
                None if self.context_df is not None
                else self._INTERP_AUTO_RECORDS
            )
        else:
            self.interpolate_short_gaps = self._interpolate_short_gaps_arg

        self.kwargs = {**self._XGB_DEFAULTS, **kwargs}

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
            f"  context drivers            "
            f"{list(self.context_df.columns) if self.context_df is not None else 'none'}"
        )

        # Both parameter blocks are echoed in full, so a report shows exactly what
        # ran — including the defaults the caller never touched.
        rule("Feature engineering", min_level=2)
        _console.print("\n".join(
            f"  {key:<26} {value}"
            + ("  (context drivers only)" if key in ('features_rolling', 'features_ema')
               else "")
            for key, value in self.feature_kwargs.items()
        ))

        rule("XGBoost", min_level=2)
        _console.print("\n".join(
            f"  {key:<26} {value}" for key, value in self.kwargs.items()
        ))

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

    @property
    def _live_ok(self) -> bool:
        """True when the console can render a live spinner/bar (terminal or Jupyter).

        In a piped/redirected run neither is true, so we fall back to plain lines.
        """
        return bool(getattr(_console, "is_jupyter", False)
                    or getattr(_console, "is_terminal", False))

    @contextmanager
    def _stage(self, label: str, *, spinner: bool = True):
        """Run one numbered progress stage: time it and print a single result line.

        At verbose==1 a transient spinner shows the stage is running (so a long step
        is not a silent pause) and then vanishes, leaving one dense result line per
        stage. At verbose>=2 it prints a start header instead, so the sub-component
        detail that level emits has room. Yields a dict whose ``'msg'`` the caller may
        set to append a short summary to the result line.
        """
        self._step_no += 1
        holder = {"msg": ""}
        t0 = perf_counter()
        if self.verbose == 1 and spinner and self._live_ok:
            with _console.status(f"Step {self._step_no}: {label} ...", spinner="dots"):
                yield holder
        else:
            if self.verbose >= 2:
                info(f"Step {self._step_no}: {label} ...")
            yield holder
        if self.verbose >= 1:
            extra = f", {holder['msg']}" if holder['msg'] else ""
            info(f"Step {self._step_no}: {label} ({perf_counter() - t0:.1f}s{extra})")

    def run(self) -> 'SWINGapFillerXGBoost':
        """Execute gap-filling: optional offset correction, nighttime zeros, daytime XGBoost.

        Progress: at ``verbose>=1`` each stage prints one dense result line with its
        elapsed time; on an interactive terminal or in Jupyter a transient spinner
        (and, for the SHAP fill, a progress bar) shows the current stage while it runs,
        so a long step is not a silent pause. Sub-component chatter is suppressed at
        this level and shown only at ``verbose>=2``. Set ``verbose=0`` to silence.

        Returns:
            self — for method chaining.
        """
        target_col = self.target_col
        self._step_no = 0
        run_t0 = perf_counter()

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
            with self._stage("Correcting nighttime sensor offset"):
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
        with self._stage("Computing potential radiation and day/night split") as st:
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
            st["msg"] = (f"{int(daytime_mask.sum())} daytime / "
                         f"{int((~daytime_mask).sum())} nighttime records")

        if self.verbose >= 1:
            info(f"Gaps to fill: {int(daytime_gaps.sum())} daytime, "
                 f"{int(nighttime_gaps.sum())} nighttime")

        # Nighttime: zero is the physically correct value (no solar radiation).
        with self._stage("Filling nighttime gaps with 0 W/m2 (no incoming solar radiation)") as st:
            filled = working_series.copy()
            filled.loc[nighttime_gaps] = 0.0
            st["msg"] = f"{int(nighttime_gaps.sum())} records set to 0"

        # Short daytime gaps: interpolate the clearness index. Computed here but
        # deliberately NOT written into `filled` yet — the model must train on
        # observations only, never on this interpolation.
        interpolated = None
        if self.interpolate_short_gaps:
            with self._stage(f"Interpolating short daytime gaps "
                             f"(<= {self.interpolate_short_gaps} records)") as st:
                interpolated = self._interpolate_short_gaps(working_series, swinpot)
                st["msg"] = f"{int(interpolated.notna().sum())} records interpolated"

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
            feature_importances_reduction=(
                daytime_results.feature_importances_reduction if daytime_results else None
            ),
            model=daytime_results.model if daytime_results else None,
            accepted_features=daytime_results.accepted_features if daytime_results else None,
            rejected_features=daytime_results.rejected_features if daytime_results else None,
        )

        if self.verbose >= 1:
            total_filled = (flag > 0).sum()
            success(f"Done in {perf_counter() - run_t0:.1f}s - {total_filled} records "
                    f"filled ({gaps.sum()} gaps total)")

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
        # Using the nighttime-zero-filled series is intentional: it gives physically
        # correct context (zero at night) for features that span the day/night
        # boundary.
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

        # Feature engineering on the FULL index so rolling windows are correct at the
        # dawn/dusk boundaries when we later subset to daytime.
        #
        # SW_IN_POT is excluded from the rolling and EMA stages: it is a deterministic
        # function of the timestamp, so its derived variants are the same function
        # again and measure identically to leaving them out. Only measured context
        # drivers carry sky state worth smoothing.
        # Sub-components get a quieter verbosity than swin's own progress layer:
        # at swin verbose==1 their internal line dumps (engineered-column lists,
        # per-model reports) are suppressed so only the numbered stage lines show;
        # at swin verbose>=2 they stream their full detail.
        sub_v = self.verbose if self.verbose >= 2 else 0

        with self._stage("Engineering features") as st:
            engineer = FeatureEngineer(
                target_col=target_col,
                verbose=sub_v,
                **self.feature_kwargs,
            )
            full_features_df = engineer.fit_transform(input_df)
            st["msg"] = f"{full_features_df.shape[1] - 1} features over {len(full_features_df)} rows"

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
            verbose=sub_v,
            below_zero='zero',
            shap_max_rows=self._SHAP_MAX_ROWS,
            **self.kwargs,
        )

        with self._stage(f"Training XGBoost on {n_complete} daytime records "
                         f"(incl. SHAP importances)"):
            model.trainmodel(showplot_scores=False, showplot_importance=False)

        if self.reduce_features:
            with self._stage("Reducing features (SHAP vs random benchmark) and retraining"):
                model.reduce_features()
                model.trainmodel(showplot_scores=False, showplot_importance=False)

        # Filling gaps: the SHAP importance over the whole record is the slowest
        # step, so drive a progress bar off fillgaps' progress_callback when the
        # console can render one; otherwise fall back to a plain timed line.
        self._step_no += 1
        t0 = perf_counter()
        label = "Filling daytime gaps (SHAP importances over full record)"
        if self.verbose == 1 and self._live_ok:
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
            with Progress(
                TextColumn(f"  Step {self._step_no}: {label}"),
                BarColumn(),
                TaskProgressColumn(),
                console=_console,
                transient=True,
            ) as prog:
                task = prog.add_task("fill", total=1.0)
                model.fillgaps(showplot_scores=False, showplot_importance=False,
                               progress_callback=lambda f: prog.update(task, completed=f))
        else:
            if self.verbose >= 2:
                info(f"Step {self._step_no}: {label} ...")
            model.fillgaps(showplot_scores=False, showplot_importance=False)
        if self.verbose >= 1:
            info(f"Step {self._step_no}: {label} ({perf_counter() - t0:.1f}s)")

        return model.results
