"""
Feature Engineering for Time Series Gap-Filling

This module provides standalone feature engineering for time series data,
enabling pre-computation of features before gap-filling model training.

Supports 8-stage feature engineering pipeline:
1. Lag features (temporal context)
2. Rolling statistics (mean, std, median, min, max, percentiles)
3. Temporal differencing (rate of change, acceleration)
4. Exponential moving average (weighted historical context)
5. Polynomial expansion (non-linear relationships)
6. STL decomposition (trend, seasonal, residual)
7. Timestamp features (diurnal, seasonal cycles)
8. Sequential record numbering (temporal ordering)

Example:
    engineer = FeatureEngineer(
        target_col='NEE',
        features_lag=[-2, 2],
        features_rolling=[12, 48],
        features_rolling_stats=['median', 'min', 'max'],
        features_poly_degree=2,
        features_stl=True,
        vectorize_timestamps=True,
    )

    df_engineered = engineer.fit_transform(df)
    # Now df_engineered has all features computed and is ready for models
"""

import pandas as pd
import numpy as np
from pandas import DataFrame

import diive.core.dfun.frames as fr
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import vectorize_timestamps
from diive.pkgs.createvar.laggedvariants import lagged_variants


class FeatureEngineer:
    """
    Standalone feature engineering for time series data.

    Separates feature engineering from gap-filling models, enabling:
    - Pre-computation of features once, reuse across multiple models
    - Independent testing and debugging
    - Composition-based API: engineer features → pass to any model
    - Feature reuse with non-gap-filling models

    Usage:
        engineer = FeatureEngineer(
            target_col='target_column_name',
            features_lag=[-1, 1],
            features_rolling=[12],
            # ... other parameters
        )
        df_engineered = engineer.fit_transform(input_df)
        # Pass df_engineered to gap-filling models
    """

    def __init__(self,
                 target_col: str or tuple,
                 verbose: int = 0,
                 features_lag: list = None,
                 features_lag_stepsize: int = 1,
                 features_lag_exclude_cols: list = None,
                 features_rolling: list = None,
                 features_rolling_exclude_cols: list = None,
                 features_rolling_stats: list = None,
                 features_diff: list = None,
                 features_diff_exclude_cols: list = None,
                 features_ema: list = None,
                 features_ema_exclude_cols: list = None,
                 features_poly_degree: int = None,
                 features_poly_exclude_cols: list = None,
                 features_stl: bool = False,
                 features_stl_method: str = 'stl',
                 features_stl_seasonal_period: int = None,
                 features_stl_exclude_cols: list = None,
                 features_stl_components: list = None,
                 vectorize_timestamps: bool = False,
                 add_continuous_record_number: bool = False,
                 sanitize_timestamp: bool = False):
        """
        Initialize feature engineer with 8-stage composable pipeline.

        The FeatureEngineer creates temporal and statistical features from time series data.
        Set unused parameters to None/False to skip those stages. All features use naming
        convention: `.{col}_{STAGE}_{detail}` (e.g., `.Tair_f-1` for lag, `.Tair_f_MEAN12`
        for rolling mean with window 12).

        === STAGE 1: LAG FEATURES (Temporal Context) ===
        Args:
            features_lag (list): [min_lag, max_lag] range for creating lag features.
                Default: None (disabled)
                Typical values:
                  - None: Disable (very fast, but no temporal context)
                  - [-1, -1]: Only past value (minimal, good for speed tests)
                  - [-1, 1]: Past and future lags (captures local patterns)
                  - [-2, 2]: Extended range for slower-moving signals
                Effect on time series: Creates past/future context. Negative lag=past,
                    positive lag=future. Example: [-1, 1] with stepsize=1 creates lags
                    [-1, 0, 1] but 0 is excluded, resulting in [-1, 1].

            features_lag_stepsize (int): Step between lags when range is large.
                Default: 1 (every lag value)
                Typical values:
                  - 1: Include all lags in range ([-2,2,stepsize=1] → [-2,-1,1,2])
                  - 2: Skip every other ([-4,4,stepsize=2] → [-4,-2,2,4], faster)
                Effect: Reduces number of features for large lag ranges. Trades feature
                    detail for speed.

            features_lag_exclude_cols (list): Column names to exclude from lagging.
                Default: None (lag all columns except target)
                Typical use: Skip invariant columns like site ID, quality flags
                Effect: None means "lag everything". Empty list [] means same (lag all).

        === STAGE 2: ROLLING STATISTICS (Local Context Windows) ===
        Args:
            features_rolling (list): Window sizes for rolling aggregation.
                Default: None (disabled)
                Typical values:
                  - None: Disable (very fast)
                  - [12, 24]: 12 and 24 hour windows for 30-min flux data
                  - [7, 30]: Weekly and monthly windows for daily data
                Effect on time series: Creates local averages. For 30-min data, window=12
                    means ~6-hour rolling window. Captures short-term trends and noise.

            features_rolling_stats (list): Statistics to compute (beyond default mean/std).
                Default: None (use mean + std only)
                Typical values:
                  - None: Just mean and std deviation (fast)
                  - ['median', 'min', 'max']: Robust to outliers
                  - ['median', 'min', 'max', 'q25', 'q75']: Full distribution
                Effect: None means only mean+std computed. Other stats add robustness to
                    outliers and capture asymmetric distributions.

            features_rolling_exclude_cols (list): Exclude columns from rolling.
                Default: None (roll all columns except target)
                Typical use: Skip categorical or invariant columns
                Effect: None means "roll everything".

        === STAGE 3: TEMPORAL DIFFERENCING (Rate of Change) ===
        Args:
            features_diff (list): Difference orders [1, 2, ...].
                Default: None (disabled)
                Typical values:
                  - None: Disable (no rate-of-change features)
                  - [1]: First-order differences (rate of change)
                  - [1, 2]: First and second differences (rate + acceleration)
                Effect on time series: Order-1 differencing removes trends, reveals
                    variability. Order-2 reveals acceleration. Essential for detecting
                    rapidly changing signals (e.g., convection events, sensor failures).

            features_diff_exclude_cols (list): Exclude from differencing.
                Default: None (diff all columns except target)

        === STAGE 4: EXPONENTIAL MOVING AVERAGE (Weighted Historical Context) ===
        Args:
            features_ema (list): EMA spans [6, 24, 48, ...].
                Default: None (disabled)
                Typical values:
                  - None: Disable (no long-term memory)
                  - [6, 24]: 6-hr and 24-hr exponential moving averages (for 30-min data)
                  - [7, 30]: Weekly and monthly for daily data
                Effect on time series: EMA gives more weight to recent values while
                    remembering history. Span=6 means ~6 timesteps influence (20% decaying
                    weight structure). Essential for multi-timescale patterns.

            features_ema_exclude_cols (list): Exclude from EMA.
                Default: None (apply EMA to all columns except target)

        === STAGE 5: POLYNOMIAL EXPANSION (Non-linear Relationships) ===
        Args:
            features_poly_degree (int): Polynomial degree.
                Default: None (disabled)
                Typical values:
                  - None: Disable (linear relationships only)
                  - 2: Squared terms (quadratic relationships, most common)
                  - 3: Cubic terms (rare, very non-linear)
                Effect on time series: Degree=2 creates X² features for each column.
                    Essential for capturing saturation effects (e.g., photosynthesis
                    plateaus at high light). Quadratic relationships common in ecology.

            features_poly_exclude_cols (list): Exclude from polynomial.
                Default: None (apply to all columns except target)

        === STAGE 6: STL DECOMPOSITION (Trend/Seasonal Separation) ===
        Args:
            features_stl (bool): Enable STL decomposition.
                Default: False (disabled)
                Set to: True or False
                Effect if False: No decomposition, skip this stage entirely (fast).
                Effect if True: Separates time series into trend, seasonal, residual
                    components. Requires specifying seasonal_period and method.

            features_stl_method (str): Decomposition algorithm.
                Default: 'stl' (Seasonal-Trend Loess)
                Options:
                  - 'stl': Robust, handles gaps, works with non-stationary data (best)
                  - 'classical': Moving average method, assumes stationarity
                  - 'harmonic': FFT-based, no series length constraints, very fast
                Effect: 'stl' recommended for flux data (has gaps, strong seasonality).
                    'harmonic' for very long series or when speed critical.

            features_stl_seasonal_period (int): Seasonal period in timesteps.
                Default: None (auto-detect via periodogram if features_stl=True)
                Typical values for 30-min flux data:
                  - None: Auto-detect (recommended, ~48 for daily cycle)
                  - 48: Daily cycle (30-min × 48 = 24 hours)
                  - 336: Weekly cycle (30-min × 336 = 7 days)
                  - 17520: Annual cycle (30-min × 17520 = 365 days)
                Effect: None means auto-detect from data (slower but robust). Explicit
                    value is faster. Wrong period produces useless seasonal component.

            features_stl_exclude_cols (list): Exclude from STL.
                Default: None (apply to all columns except target)
                Typical use: Skip columns without strong seasonality

            features_stl_components (list): Components to extract.
                Default: None (extract all: trend, seasonal, residual)
                Options:
                  - None: All three components
                  - ['trend']: Trend only (removes noise and seasonality)
                  - ['trend', 'seasonal']: Trend and seasonal (removes residual noise)
                  - ['trend', 'seasonal', 'residual']: All components (default)
                Effect: None extracts all. Selecting subset reduces features but keeps
                    what's needed. Trend alone useful for detrending, removes diurnal cycles.

        === STAGE 7: TIMESTAMP FEATURES (Diurnal/Seasonal Cycles) ===
        Args:
            vectorize_timestamps (bool): Add timestamp-derived features.
                Default: False (disabled, fast)
                Set to: True or False
                Effect if False: No timestamp features (skip stage, 2-3% speedup).
                Effect if True: Creates 19 features: year, season, month, week, DOY
                    (day-of-year), hour + sin/cos versions for circular encoding. Captures
                    diurnal cycles (photosynthesis) and seasonal patterns (dormancy).
                Typical use:
                  - False: For fast testing or when temporal patterns already captured
                  - True: Essential for gap-filling (captures daily/seasonal cycles)
                Effect on time series: Sine/cosine encoding treats day-of-year as
                    circular (Dec→Jan continuous). Creates cyclic features that models
                    can learn (e.g., morning vs evening photosynthesis).

        === STAGE 8: SEQUENTIAL RECORD NUMBERING (Temporal Ordering) ===
        Args:
            add_continuous_record_number (bool): Add sequential 1,2,3,... numbering.
                Default: False (disabled)
                Set to: True or False
                Effect if False: No record number (skip stage).
                Effect if True: Creates column .RECORDNUMBER = 1, 2, 3, ... Captures
                    long-term drift (instrument drift, ecosystem changes).
                Typical use:
                  - False: Most cases (minimal benefit, adds collinearity)
                  - True: When long-term drift is important (>1 year data)
                Effect on time series: Linear feature captures systematic drift over
                    time (e.g., accumulation, senescence).

        === DATA QUALITY ===
        Args:
            sanitize_timestamp (bool): Validate and prepare timestamps.
                Default: False (skip validation, assume clean timestamps)
                Set to: True or False
                Effect if False: Trust input timestamps (fast, assumes clean).
                Effect if True: Validate and fix timestamp issues (duplicates, gaps,
                    non-monotonic). Slightly slower but catches data problems.
                Typical use:
                  - False: When data already validated (production)
                  - True: First-time data processing (safety check)

        === METADATA ===
        Args:
            target_col (str): Column name of target variable.
                Effect: Excluded from all engineered features. All features created from
                    other columns, target preserved unchanged in output.

            verbose (int): Progress reporting.
                Default: 0 (silent)
                Options:
                  - 0: No output
                  - 1: Progress updates (recommended)
                  - 2+: Detailed logging (debugging)

        === EXAMPLE SCENARIOS ===

        Quick/Minimal (testing, fastest):
            FeatureEngineer(
                target_col='NEE',
                features_lag=[-1, -1],  # Only past
                # All others: None/False (disabled)
            )  # ~5 features total

        Fast/Standard (production, balanced):
            FeatureEngineer(
                target_col='NEE',
                features_lag=[-1, 1],
                features_rolling=[12, 24],
                features_diff=[1],
                vectorize_timestamps=True,
            )  # ~15-20 features total

        Comprehensive (research, most detailed):
            FeatureEngineer(
                target_col='NEE',
                features_lag=[-2, 2],
                features_rolling=[6, 12, 24],
                features_rolling_stats=['median', 'min', 'max'],
                features_diff=[1, 2],
                features_ema=[6, 24],
                features_poly_degree=2,
                features_stl=True,
                features_stl_seasonal_period=48,
                vectorize_timestamps=True,
            )  # ~50-100 features total

        Returns:
            None. Configure with __init__, call fit_transform(df) to engineer features.
        """
        # Store configuration (no data processing in __init__)
        self.target_col = target_col
        self.verbose = verbose

        # Lag parameters
        self.features_lag = features_lag
        self.features_lag_stepsize = features_lag_stepsize
        self.features_lag_exclude_cols = features_lag_exclude_cols

        # Rolling parameters
        self.features_rolling = features_rolling
        self.features_rolling_exclude_cols = features_rolling_exclude_cols
        self.features_rolling_stats = features_rolling_stats

        # Differencing parameters
        self.features_diff = features_diff
        self.features_diff_exclude_cols = features_diff_exclude_cols

        # EMA parameters
        self.features_ema = features_ema
        self.features_ema_exclude_cols = features_ema_exclude_cols

        # Polynomial parameters
        self.features_poly_degree = features_poly_degree
        self.features_poly_exclude_cols = features_poly_exclude_cols

        # STL parameters
        self.features_stl = features_stl
        self.features_stl_method = features_stl_method
        self.features_stl_seasonal_period = features_stl_seasonal_period
        self.features_stl_exclude_cols = features_stl_exclude_cols
        self.features_stl_components = features_stl_components

        # Timestamp and record number parameters
        self.vectorize_timestamps = vectorize_timestamps
        self.add_continuous_record_number = add_continuous_record_number
        self.sanitize_timestamp = sanitize_timestamp

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform data (fit on all data, stateless operations).

        Args:
            df: Input DataFrame with target column and feature columns

        Returns:
            DataFrame with all engineered features added
        """
        return self._create_features(df)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform using fitted configuration.

        For stateless feature engineering operations (lag, rolling, etc.),
        this is identical to fit_transform().

        Args:
            df: Input DataFrame to transform

        Returns:
            DataFrame with engineered features
        """
        return self._create_features(df)

    def _create_features(self, df: DataFrame) -> DataFrame:
        """
        Main orchestration method: apply all feature engineering stages.

        Implements the 8-stage feature engineering pipeline:
        1. Lag features
        2. Rolling statistics (basic + advanced)
        3. Temporal differencing
        4. Exponential moving average
        5. Polynomial expansion
        6. STL decomposition
        7. Timestamp features
        8. Sequential record number

        Args:
            df: Input DataFrame with target and feature columns

        Returns:
            DataFrame with all engineered features
        """
        # Store original input features (exclude target)
        self.original_input_features = [c for c in df.columns if c != self.target_col]

        # DataFrame that contains the target and all original features
        expanded_df = df.copy()

        # Add lagged features
        if self.features_lag:
            expanded_df = self._create_lagged_variants(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add rolling statistics (basic)
        if self.features_rolling:
            expanded_df = self._create_rolling_features(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add rolling statistics (advanced)
        if self.features_rolling and self.features_rolling_stats:
            expanded_df = self._create_rolling_features_advanced(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add temporal differencing
        if self.features_diff:
            expanded_df = self._create_differencing_features(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add exponential moving average
        if self.features_ema:
            expanded_df = self._create_ema_features(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add polynomial features
        if self.features_poly_degree:
            expanded_df = self._create_polynomial_features(work_df=expanded_df)

        # Add STL decomposition features
        if self.features_stl:
            expanded_df = self._create_stl_features(
                work_df=df[self.original_input_features].copy(),
                expanded_df=expanded_df
            )

        # Add timestamp features
        if self.vectorize_timestamps:
            expanded_df = vectorize_timestamps(df=expanded_df, txt="")
            # Keep only sine/cosine variants, drop linear versions
            expanded_df = expanded_df.drop(
                columns=['.HOUR', '.SEASON', '.MONTH', '.WEEK', '.DOY'],
                errors='ignore'
            )

        # Add continuous record number
        if self.add_continuous_record_number:
            expanded_df = fr.add_continuous_record_number(df=expanded_df)

        # Sanitize timestamp
        if self.sanitize_timestamp:
            verbose = True if self.verbose > 0 else False
            tss = TimestampSanitizer(
                data=expanded_df,
                output_middle_timestamp=True,
                verbose=verbose
            )
            expanded_df = tss.get()

        return expanded_df

    # ========== Wrapper methods (orchestrate actual feature engineering) ==========

    def _create_lagged_variants(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for lag feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add lagged features because there are no original features.")

        _out_df = lagged_variants(
            df=work_df,
            stepsize=self.features_lag_stepsize,
            lag=self.features_lag,
            exclude_cols=self.features_lag_exclude_cols,
            verbose=self.verbose
        )
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_rolling_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for basic rolling feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add rolling features because there are no original features.")

        _out_df = self._rolling_features(
            df=work_df,
            windows=self.features_rolling,
            exclude_cols=self.features_rolling_exclude_cols
        )
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_rolling_features_advanced(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for advanced rolling feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add advanced rolling features because there are no original features.")

        _out_df = self._rolling_features_advanced(
            df=work_df,
            windows=self.features_rolling,
            stats=self.features_rolling_stats,
            exclude_cols=self.features_rolling_exclude_cols
        )
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_differencing_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for differencing feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add differencing features because there are no original features.")

        _out_df = self._differencing_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_ema_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for EMA feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add EMA features because there are no original features.")

        _out_df = self._ema_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_polynomial_features(self, work_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for polynomial feature creation."""
        _out_df = self._polynomial_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in work_df.columns]
        work_df = work_df.join(_out_df[newcols])
        return work_df

    def _create_stl_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for STL feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add STL features because there are no original features.")

        _out_df = self._stl_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    # ========== Core feature engineering methods ==========

    def _rolling_features(self, df: pd.DataFrame, windows: list, exclude_cols: list = None) -> pd.DataFrame:
        """Add rolling mean and std of feature columns at multiple window sizes."""
        exclude = [self.target_col] + (exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude]
        newcols = []

        for w in windows:
            rolled = df[feature_cols].rolling(window=w, min_periods=1)
            mean_df = rolled.mean()
            std_df = rolled.std(ddof=0)  # population std to avoid NaN for window=1

            mean_df.columns = [f'.{c}_MEAN{w}' for c in feature_cols]
            std_df.columns = [f'.{c}_SD{w}' for c in feature_cols]

            df = pd.concat([df, mean_df, std_df], axis=1)
            newcols += mean_df.columns.tolist() + std_df.columns.tolist()

        if self.verbose:
            print(f"++ Added rolling features (windows={windows}) for {len(feature_cols)} columns: {newcols}")
        return df

    def _rolling_features_advanced(self, df: pd.DataFrame, windows: list, stats: list,
                                    exclude_cols: list = None) -> pd.DataFrame:
        """Add advanced rolling statistics (median, min, max, percentiles)."""
        exclude = [self.target_col] + (exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude]
        newcols = []

        stat_name_map = {
            'median': ('MEDIAN', lambda x: x.median()),
            'min': ('MIN', lambda x: x.min()),
            'max': ('MAX', lambda x: x.max()),
            'std': ('SD', lambda x: x.std(ddof=0)),
            'q25': ('Q25', lambda x: x.quantile(0.25)),
            'q75': ('Q75', lambda x: x.quantile(0.75)),
        }

        for w in windows:
            rolled = df[feature_cols].rolling(window=w, min_periods=1)

            for stat in stats:
                if stat not in stat_name_map:
                    if self.verbose:
                        print(f"Warning: unknown rolling statistic '{stat}', skipping")
                    continue

                stat_display_name, stat_func = stat_name_map[stat]
                stat_df = stat_func(rolled)
                stat_df.columns = [f'.{c}_ROLL{stat_display_name}{w}' for c in feature_cols]

                df = pd.concat([df, stat_df], axis=1)
                newcols += stat_df.columns.tolist()

        if self.verbose and newcols:
            print(f"++ Added advanced rolling statistics (stats={stats}, windows={windows}) "
                  f"for {len(feature_cols)} columns: {newcols}")
        return df

    def _differencing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 1st and 2nd order differences for capturing temporal momentum."""
        if not self.features_diff:
            return df

        exclude = [self.target_col] + (self.features_diff_exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('.')]
        newcols = []

        for order in self.features_diff:
            if order < 1:
                continue
            diff_df = df[feature_cols].copy()
            for _ in range(order):
                diff_df = diff_df.diff()
            diff_df.columns = [f'.{c}_DIFF{order}' for c in feature_cols]
            df = pd.concat([df, diff_df], axis=1)
            newcols += diff_df.columns.tolist()

        if self.verbose:
            print(f"++ Added differencing features (orders={self.features_diff}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

    def _ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving average (EMA) of feature columns."""
        if not self.features_ema:
            return df

        exclude = [self.target_col] + (self.features_ema_exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('.')]
        newcols = []

        for span in self.features_ema:
            if span < 1:
                continue
            ema_df = df[feature_cols].ewm(span=span, min_periods=1, adjust=False).mean()
            ema_df.columns = [f'.{c}_EMA{span}' for c in feature_cols]
            df = pd.concat([df, ema_df], axis=1)
            newcols += ema_df.columns.tolist()

        if self.verbose:
            print(f"++ Added EMA features (spans={self.features_ema}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

    def _polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features by expanding each feature to specified degree."""
        if not self.features_poly_degree or self.features_poly_degree < 2:
            return df

        exclude = [self.target_col] + (self.features_poly_exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('.')]
        newcols = []

        for degree in range(2, self.features_poly_degree + 1):
            poly_df = df[feature_cols].copy() ** degree
            poly_df.columns = [f'.{c}_POL{degree}' for c in feature_cols]
            df = pd.concat([df, poly_df], axis=1)
            newcols += poly_df.columns.tolist()

        if self.verbose:
            print(f"++ Added polynomial features (degree={self.features_poly_degree}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

    def _stl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add STL (Seasonal-Trend Loess) decomposition components as features."""
        if not self.features_stl:
            return df

        from diive.pkgs.analyses.seasonaltrend import SeasonalTrendDecomposition

        exclude = [self.target_col] + (self.features_stl_exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('.')]
        newcols = []

        # Determine which components to extract
        components_to_extract = self.features_stl_components or ['trend', 'seasonal', 'residual']
        if not isinstance(components_to_extract, list):
            components_to_extract = [components_to_extract]

        # Filter to valid components
        valid_components = {'trend', 'seasonal', 'residual'}
        components_to_extract = [c for c in components_to_extract if c in valid_components]

        if not components_to_extract:
            if self.verbose:
                print(f"Warning: No valid STL components specified. Valid options: {valid_components}")
            return df

        for col in feature_cols:
            # Check if column is complete (no gaps)
            if df[col].isna().sum() > 0:
                if self.verbose:
                    print(f"Skipping STL decomposition for {col} (contains {df[col].isna().sum()} gaps)")
                continue

            try:
                # Apply STL decomposition
                decomp = SeasonalTrendDecomposition(
                    series=df[col],
                    method=self.features_stl_method,
                    seasonal_period=self.features_stl_seasonal_period,
                    verbose=False
                )

                # Extract selected components
                for component in components_to_extract:
                    if component == 'trend':
                        stl_df = decomp.trend.to_frame(name=f'.{col}_STL_TREND')
                    elif component == 'seasonal':
                        stl_df = decomp.seasonal.to_frame(name=f'.{col}_STL_SEASONAL')
                    elif component == 'residual':
                        stl_df = decomp.residual.to_frame(name=f'.{col}_STL_RESIDUAL')
                    else:
                        continue

                    df = pd.concat([df, stl_df], axis=1)
                    newcols += stl_df.columns.tolist()

            except Exception as e:
                if self.verbose:
                    print(f"Warning: STL decomposition failed for {col}: {str(e)}")
                continue

        if self.verbose and newcols:
            print(f"++ Added STL features (method={self.features_stl_method}, "
                  f"components={components_to_extract}) for {len([c for c in feature_cols if any(nc.startswith(f'.{c}_STL') for nc in newcols)])} "
                  f"complete columns: {newcols}")
        return df
