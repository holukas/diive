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
        Initialize feature engineer with configuration parameters.

        Args:
            target_col: Column name of target variable (excluded from engineered features)
            verbose: Verbosity level (0=silent, 1+=progress updates)
            features_lag: List [min_lag, max_lag] for lag feature range
            features_lag_stepsize: Step size for lag generation
            features_lag_exclude_cols: Columns to exclude from lagging
            features_rolling: List of window sizes for rolling statistics
            features_rolling_exclude_cols: Columns to exclude from rolling
            features_rolling_stats: Advanced stats: 'median', 'min', 'max', 'std', 'q25', 'q75'
            features_diff: List of difference orders [1, 2, ...]
            features_diff_exclude_cols: Columns to exclude from differencing
            features_ema: List of EMA spans [6, 24, ...]
            features_ema_exclude_cols: Columns to exclude from EMA
            features_poly_degree: Polynomial degree (2, 3, ...)
            features_poly_exclude_cols: Columns to exclude from polynomial
            features_stl: Enable STL decomposition
            features_stl_method: STL method ('stl', 'classical', 'harmonic')
            features_stl_seasonal_period: Period for seasonal component
            features_stl_exclude_cols: Columns to exclude from STL
            features_stl_components: Components to extract (['trend', 'seasonal', 'residual'])
            vectorize_timestamps: Add timestamp features (year, season, month, etc.)
            add_continuous_record_number: Add sequential record numbering
            sanitize_timestamp: Validate and prepare timestamps
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
