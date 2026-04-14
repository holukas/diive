"""
kudos: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from yellowbrick.regressor import PredictionError, ResidualsPlot

import diive.core.dfun.frames as fr
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import vectorize_timestamps
from diive.pkgs.createvar.laggedvariants import lagged_variants
from diive.pkgs.gapfilling.scores import prediction_scores

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


class MlRegressorGapFillingBase:

    def __init__(self,
                 regressor,
                 input_df: DataFrame,
                 target_col: str or tuple,
                 verbose: int = 0,
                 features_lag: list[int, int] = None,
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
                 sanitize_timestamp: bool = False,
                 test_size: float = 0.25,
                 **kwargs):
        """
        Base class for machine-learning gap-filling using Random Forest or XGBoost.

        Trains a predictive model on complete (non-gap) data to fill missing values in a target
        time series. Includes comprehensive feature engineering pipeline with lagged variants,
        rolling statistics, temporal differencing, exponential moving averages, and polynomial
        expansion. Features are extracted from driver variables (temperature, radiation, VPD, etc.)
        to provide the model with temporal context and non-linear patterns.

        Core Philosophy: Feature engineering is parameterizable to enable task-specific customization
        while maintaining consistent methodology across implementations (RandomForest, XGBoost,
        long-term gap-filling). Each parameter offers selective control over what temporal and
        statistical contexts the model sees, balancing model complexity, interpretability, and accuracy.

        Args:
            regressor:
                Sklearn-compatible regressor class (RandomForestRegressor or XGBRegressor).

            input_df:
                Input DataFrame with time series data. Must contain 1 target column and 1+ feature
                columns. Timestamps should be in DataFrame index (DatetimeIndex).

            target_col:
                Column name of the variable to gap-fill (string or tuple for multi-level columns).

            verbose:
                Verbosity level: 0=silent, 1=progress updates, 2+=detailed output.

            test_size:
                Proportion of complete (non-gap) data to reserve for testing, between 0.0-1.0.
                Default: 0.25 (75% train, 25% test on non-gap data only).

                Why: Separate train/test evaluation on complete data assesses real-world predictive
                skill without contamination from gap-filled values.
                Effect: Smaller test_size provides tighter train set but more noise in test metrics.
                Trade-off: 0.25 is standard for balanced estimate stability vs training data size.

            features_lag:
                List [min_lag, max_lag] specifying range of lagged variants to create.
                Creates lagged copies at all integers between min_lag and max_lag (exclusive of 0).
                Default: None (no lagging).

                Why Implemented: Time series observations depend on recent history. Lagged features
                capture autoregressive structure—how the target depends on itself at prior times.
                Effect on Data: Encodes temporal autocorrelation; extends decision boundary by
                allowing model to learn patterns from {t-1, t-2, t+1, t+2} alongside current {t}.
                Advantages: Captures persistence/inertia in data (e.g., energy availability today
                relates to yesterday's weather). Essential for accurate flux predictions.
                Disadvantages: Reduces available training samples near gaps (lags create NaN);
                can inflate feature count rapidly. Bidirectional lags (negative values) require
                lookback capacity unavailable at gap boundaries.
                Example: features_lag=[-2, 2] creates lags [-2, -1, +1, +2]:
                    TA    = [  5,   6,   7, 8  ]
                    TA-2  = [NaN, NaN,   5, 6  ]
                    TA-1  = [NaN,   5,   6, 7  ]  (each TA paired with preceding TA-1)
                    TA+1  = [  6,   7,   8, NaN]  (each TA paired with next TA+1)
                    TA+2  = [  7,   8, NaN, NaN]
                Column naming: '{col}{sign}{lag}' (e.g., 'Tair_f-1', 'Tair_f+1')

            features_lag_stepsize:
                Step size for creating lags within the specified range. Default: 1 (every record).
                Why: Reduces feature dimensionality for large lag ranges without losing structure.
                Example: features_lag=[-4, 4], features_lag_stepsize=2 creates lags [-4, -2, 2, 4].
                Trade-off: Larger stepsize reduces overfitting risk but misses intermediate patterns.

            features_lag_exclude_cols:
                List of column names to exclude from lagging.
                Default: None (all feature columns are lagged).
                Why: Some features (e.g., record numbers) are nonsensical lagged; time-invariant
                attributes should not have temporal variants.
                Example: ['RECORD_NUMBER'] skips lagging for continuous record numbers.

            features_rolling:
                List of rolling window sizes (in records) for rolling statistics.
                For each window, rolling mean and std are computed for every feature column
                (except target and excluded columns). Default: None (no rolling statistics).

                Why Implemented: Time series data exhibit local context—current conditions depend
                on recent envelope of variability. Rolling windows capture "local stationarity."
                Effect on Data: Mean represents recent baseline level; std represents recent
                volatility/uncertainty. A narrow band (low std) vs wide band (high std) convey
                different information about the observation environment.
                Advantages: Computationally cheap; never introduces NaN (min_periods=1);
                resistant to outliers in mean/std form; two window sizes (e.g., 3h and 24h)
                capture both diurnal and daily cycles cheaply.
                Disadvantages: Reduces time series stationarity further (derivatives of derivatives);
                high correlation with trend-based features; may overweight short-term noise.
                Example: features_rolling=[6, 48] with 30-min data creates 3-hour and 24-hour
                rolling mean/std for each driver variable.
                Column naming: '.{col}_mean{w}', '.{col}_std{w}' (e.g., '.Tair_f_mean6').

            features_rolling_exclude_cols:
                List of column names excluded from rolling statistics computation.
                Default: None.
                Example: ['Rg_f'] skips rolling features for Rg_f.

            features_rolling_stats:
                List of advanced rolling statistics beyond mean and std.
                Options: ['median', 'min', 'max', 'std', 'q25', 'q75'].
                Default: None (only mean and std computed if features_rolling specified).

                Why Implemented: Mean/std assume Gaussian distribution; real data may be
                skewed or bimodal. Percentiles and extremes capture distribution shape.
                Effect on Data: Median is robust to outliers; min/max bound the plausible range;
                quartiles reveal asymmetry (e.g., q75-median vs median-q25).
                Advantages: Provides fine-grained information about local data structure.
                Disadvantages: Multiplies feature count (6-8 stats per window per variable);
                increased computation; more features = higher overfitting risk; diminishing
                returns after median/min/max.
                Example: features_rolling_stats=['median', 'min', 'max', 'q25', 'q75'].
                Column naming: '.{col}_ROLLMEDIAN{w}', '.{col}_ROLLMIN{w}', '.{col}_ROLLMAX{w}',
                '.{col}_ROLLQ25{w}', '.{col}_ROLLQ75{w}', '.{col}_ROLLSD{w}'.

            features_diff:
                List of difference orders for temporal momentum feature engineering.
                Computes 1st-order differences (rate of change), 2nd-order (acceleration), etc.
                Default: None (no differencing).

                Why Implemented: Rate of change encodes temporal direction and speed—crucial for
                transient events. Ecosystem fluxes ramp up/down steeply at sunrise/sunset; a
                high DIFF captures "TA is rising fast" without requiring past values.
                Effect on Data: De-trends data (removes level); DIFF1 makes I(1) series I(0);
                DIFF2 captures acceleration (useful for curvature). Emphasizes transitions.
                Advantages: Captures change velocity essential for forecasting turning points;
                computationally trivial; no NaN inflation.
                Disadvantages: Loses level information (absolute value); amplifies noise
                (high-frequency components); DIFF2 extremely noisy unless data is smooth.
                Example: features_diff=[1, 2] creates 1st and 2nd order differences.
                Column naming: '.{col}_DIFF{order}' (e.g., '.Tair_f_DIFF1', '.Tair_f_DIFF2').

            features_ema:
                List of span values for exponential moving average (EMA) feature engineering.
                For each span, creates '.{col}_EMA{span}' columns for all feature columns.
                EMA applies exponential decay weighting where recent values are weighted more.
                Default: None (no EMA features).

                Why Implemented: For non-stationary time series (seasonally evolving baselines),
                EMA tracks the moving target better than fixed windows. Recent = more relevant.
                Effect on Data: EMA smooths noise while respecting trends; acts as adaptive
                baseline that responds to gradual regime shifts. Complements rolling stats by
                emphasizing recent behavior over uniform weighting.
                Advantages: Excellent for trending data; captures adaptive baseline without lag;
                memory-efficient (only needs current observation and prior EMA, not full window);
                span interpretation intuitive ("equivalent to N-period window").
                Disadvantages: Requires parameter tuning (which spans?); produces correlated
                features if overlapping spans; less robust to outliers than median-based stats.
                Example: features_ema=[6, 24, 48] with 30-min data adds 3h, 12h, and 24h EMAs.
                Column naming: '.{col}_EMA{span}' (e.g., '.Tair_f_EMA6', '.Tair_f_EMA24').

            features_ema_exclude_cols:
                List of column names excluded from EMA computation.
                Default: None.
                Example: ['RECORD_NUMBER'] skips EMA for continuous record numbers.

            features_poly_degree:
                Polynomial degree for non-linear relationship modeling (2 for squared, 3 for cubed, etc.).
                Default: None (no polynomial expansion).

                Why Implemented: Many physical processes are non-linear. Radiation drives flux via
                square-law absorption; temperature effects are often cubic. Tree models (RF/XGB)
                can approximate polynomials via splits, but explicit polynomial features help.
                Effect on Data: Degree-2 expands feature space quadratically; enables the model
                to learn u-shaped, inverse, or accelerating relationships directly.
                Advantages: Captures obvious non-linearities (e.g., TA² for temperature stress);
                tree models still benefit (shorter tree paths to optimal split); interpretable
                (can compute d(flux)/d(TA) as linear combination of TA and TA²).
                Disadvantages: Explodes feature count (d features become d + d² at degree-2);
                high intercorrelation with original features; degree-3+ prone to overfitting;
                less effective for tree models than for linear models (trees naturally approximate
                polynomials), but still useful for complex relationships.
                Example: features_poly_degree=2 creates squared terms for each driver variable.
                Column naming: '.{col}_POL{degree}' (e.g., '.Tair_f_POL2' for squared).

            features_poly_exclude_cols:
                List of column names excluded from polynomial expansion.
                Default: None.
                Example: ['RECORD_NUMBER'] skips polynomial features for record numbers.

            features_stl:
                Enable STL (Seasonal-Trend Loess) decomposition feature engineering.
                Default: False (disabled).

                Why Implemented: Ecosystem fluxes exhibit multi-scale temporal structure:
                long-term trends (phenology, instrument drift), recurring seasonal patterns
                (diurnal/annual cycles), and residual variability (noise, anomalies). STL
                separates these components, allowing the model to learn how each scale
                contributes to the target flux.
                Effect on Data: For each complete (gap-free) feature column, extracts trend
                (slow changes), seasonal (periodic patterns), and residual (noise). Creates
                3 new features per column per component: .{col}_STL_TREND, .{col}_STL_SEASONAL,
                .{col}_STL_RESIDUAL.
                Advantages: Reveals structure hidden in raw data; handles non-stationary
                baseline (trend component); robust to outliers; no parameter tuning required
                for seasonal period (auto-detected). Particularly effective for flux data
                with strong seasonality.
                Disadvantages: Only applies to complete columns (skips gap-filled data);
                computationally more expensive than rolling statistics; extracted components
                are smoothed (less granular). Can explode feature count if many features.
                Example: features_stl=True extracts trend/seasonal/residual for all complete
                driver variables.

            features_stl_method:
                Decomposition method for STL. Options: 'stl', 'classical', 'harmonic'.
                Default: 'stl' (Seasonal-Trend Loess, most robust).

                Why Options Exist:
                - 'stl': Robust, handles gaps and non-stationary data (default, recommended)
                - 'classical': Simple moving-average based, assumes stationarity
                - 'harmonic': Fourier-based, reveals frequency-domain structure

            features_stl_seasonal_period:
                Seasonal period in observations for STL decomposition.
                Default: None (auto-detect via periodogram).
                Example: 365 for daily data (annual cycle), 48 for 30-min data (daily cycle).

                Why Auto-detection: Many datasets don't have known periods. Auto-detection
                finds the dominant frequency via spectral analysis.
                Manual Specification: Provide if you know the period (e.g., 24h cycle for
                hourly data = period of 24).

            features_stl_exclude_cols:
                List of column names excluded from STL decomposition.
                Default: None.
                Example: ['RECORD_NUMBER'] skips STL for record numbers (nonsensical to decompose).

            features_stl_components:
                List of STL components to extract. Options: 'trend', 'seasonal', 'residual'.
                Default: None (extracts all three: ['trend', 'seasonal', 'residual']).
                Example: ['trend', 'seasonal'] extracts only trend and seasonal, skipping residual.

                Why Selective: Reduces feature dimensionality if residual/seasonal are noisy.
                All components: Maximum information, risk of overfitting.
                Trend only: Captures long-term baseline changes (useful for drifting data).
                Seasonal only: Captures periodic patterns (useful if trend is stable).

            vectorize_timestamps:
                Include timestamp attributes as numeric features: year, season, month, week, doy, hour.
                Provides the model with annual and diurnal cycles. Default: False.

                Why Implemented: Ecosystem processes are strongly seasonal and diurnal. Winter flux
                ≠ summer flux; nighttime NEE ≠ daytime NEE. The model needs to know "which season?"
                and "which hour?" to make context-aware predictions.
                Effect on Data: Adds ~6 features capturing temporal phase. Diurnal cycle (hour)
                often most predictive; day-of-year captures phenology. Season is binned month.
                Advantages: Captures known periodic behavior without explicit formulas (let model
                learn the phase relationships). Essential for multi-year datasets.
                Disadvantages: Creates circular correlation (hour 23 is near hour 0, but as numbers
                23≠0); may overfit if dataset spans <1 full seasonal cycle.

            add_continuous_record_number:
                Add continuous record numbering as a feature (1, 2, 3, ..., n).
                Captures long-term trends and drift. Default: False.

                Why Implemented: Data quality, measurement stability, or environmental state may
                drift over months/years. Record number as a feature allows the model to capture
                linear or monotonic temporal trends without explicit detrending.
                Effect on Data: Acts as a "time axis" feature; tree splits can partition into
                "early dataset" vs "late dataset" regimes.
                Advantages: Simple; captures instrument drift, aging, or seasonal ecosystem shifts.
                Disadvantages: Only captures trends monotonic in time; not useful for cyclic patterns.

            sanitize_timestamp:
                Validate and prepare timestamps for further processing (check continuity, format, etc.).
                Default: False.

                Why Implemented: Missing time steps, duplicates, or format mismatches break feature
                engineering (lags, rolling windows assume regular frequency). Sanitization detects
                and flags such issues before processing.
                Effect on Data: May resample to regular frequency, warn about missing periods, or
                reject data with gaps.
                Advantages: Prevents silent failures; ensures consistent feature engineering behavior.
                Disadvantages: May alter data; expensive if large dataset.

            **kwargs:
                Regressor-specific hyperparameters passed to the sklearn regressor.
                For RandomForestRegressor: n_estimators, max_depth, min_samples_split, etc.
                For XGBRegressor: n_estimators, max_depth, learning_rate, early_stopping_rounds, etc.

        Attributes:
            model_: Trained regressor instance.
            gapfilling_df_: DataFrame with gap-filled target and auxiliary variables.
            feature_importances_: SHAP feature importance from gap-filling model.
            feature_importances_traintest_: SHAP feature importance from train/test model.
            scores_: Model performance metrics (MAE, RMSE, R²) for gap-filling.
            scores_traintest_: Model performance metrics from train/test split.

        Feature Engineering Pipeline Order:
            1. Lag features: temporal past/future context (captures autoregression)
            2. Rolling statistics: short-term local variability (captures envelope/volatility)
            3. Differencing: temporal momentum and rate of change (captures transitions)
            4. EMA: adaptive baseline with recent-value emphasis (tracks non-stationary level)
            5. Polynomial: non-linear relationships (captures acceleration/saturation)
            6. STL decomposition: trend/seasonal/residual components (captures multi-scale patterns)
            7. Timestamp features: annual/diurnal cycles (captures seasonality)
            8. Record number: long-term drift (captures monotonic trends)

        Design Philosophy:
        All feature engineering parameters are optional (default None) to allow users to customize
        based on their data characteristics:
        - Highly periodic data: add vectorize_timestamps + rolling stats
        - Trending data: add add_continuous_record_number + EMA
        - Non-stationary flux data: add features_lag + features_diff + features_ema
        - Nonlinear relationships: add features_poly_degree
        Parameterization ensures each feature engineering type can be toggled without code changes,
        enabling systematic comparison of engineering strategies (ablation studies).
        """

        # Args
        self.regressor = regressor
        input_df = input_df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.features_lag = features_lag
        self.features_lag_stepsize = features_lag_stepsize
        self.features_lag_exclude_cols = features_lag_exclude_cols
        self.features_rolling = features_rolling
        self.features_rolling_exclude_cols = features_rolling_exclude_cols
        self.features_rolling_stats = features_rolling_stats
        self.features_diff = features_diff
        self.features_diff_exclude_cols = features_diff_exclude_cols
        self.features_ema = features_ema
        self.features_ema_exclude_cols = features_ema_exclude_cols
        self.features_poly_degree = features_poly_degree
        self.features_poly_exclude_cols = features_poly_exclude_cols
        self.features_stl = features_stl
        self.features_stl_method = features_stl_method
        self.features_stl_seasonal_period = features_stl_seasonal_period
        self.features_stl_exclude_cols = features_stl_exclude_cols
        self.features_stl_components = features_stl_components
        self.verbose = verbose
        self.vectorize_timestamps = vectorize_timestamps
        self.add_continuous_record_number = add_continuous_record_number
        self.sanitize_timestamp = sanitize_timestamp
        self.kwargs = kwargs

        self._random_state = self.kwargs['random_state'] if 'random_state' in self.kwargs else None

        if self.regressor == RandomForestRegressor:
            self.gfsuffix = '_gfRF'
        elif self.regressor == XGBRegressor:
            self.gfsuffix = '_gfXG'
        else:
            self.gfsuffix = '_gf'

        if verbose:
            print(f"\n\n{'=' * 60}\nStarting gap-filling for\n{self.target_col}\nusing {self.regressor}\n{'=' * 60}")

        # Create model dataframe and Add additional data columns
        self.model_df = input_df.copy()

        # Original input features (all features except target)
        self.original_input_features = self.model_df.drop(columns=self.target_col).columns.tolist()

        # Create additional data columns
        self.model_df = self._create_additional_datacols()

        self._check_n_cols()

        # Check if features complete
        n_vals_index = len(self.model_df.index)
        fstats = self.model_df[self.original_input_features].describe()
        not_complete = fstats.loc['count'] < n_vals_index
        not_complete = not_complete[not_complete].index.tolist()
        if len(not_complete) > 0:
            print(f"(!)Some features are incomplete and have less than {n_vals_index} values:")
            for nc in not_complete:
                print(f"    --> {nc} ({fstats[nc]['count'].astype(int)} values)")
            print(f"This means that not all target values can be predicted based on the full model.")

        # Create training (75%) and testing dataset (25%)
        # Sort index to keep original order
        _temp_df = self.model_df.copy().dropna()

        self.train_df, self.test_df = train_test_split(_temp_df, test_size=self.test_size,
                                                       random_state=self._random_state, shuffle=True)

        self.train_df = self.train_df.sort_index()
        self.test_df = self.test_df.sort_index()

        self.random_col = None

        # Instantiate model with params
        self._model = self.regressor(**self.kwargs)

        # Attributes
        self._gapfilling_df = None  # Will contain gapfilled target and auxiliary variables
        # self._model = None
        self._traintest_details = dict()
        self._feature_importances = dict()
        self._feature_importances_traintest = pd.DataFrame()
        self._feature_importances_reduction = pd.DataFrame()
        self._scores = dict()
        self._scores_traintest = dict()
        self._accepted_features = []
        self._rejected_features = []

        self.n_timeseriessplits = None

    def get_gapfilled_target(self):
        """Gap-filled target time series"""
        return self.gapfilling_df_[self.target_gapfilled_col]

    def get_flag(self):
        """Gap-filling flag, where 0=observed, 1=gap-filled, 2=gap-filled with fallback"""
        return self.gapfilling_df_[self.target_gapfilled_flag_col]

    @property
    def model_(self):
        """Return model, trained on test data"""
        if not self._model:
            raise Exception(f'Not available: model.')
        return self._model

    @property
    def feature_importances_(self) -> DataFrame:
        """Return feature importance for model used in gap-filling"""
        if not isinstance(self._feature_importances, DataFrame):
            raise Exception(f'Not available: feature importances for gap-filling.')
        return self._feature_importances

    @property
    def feature_importances_traintest_(self) -> DataFrame:
        """Return feature importance from model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_traintest, DataFrame):
            raise Exception(f'Not available: feature importances from training & testing.')
        return self._feature_importances_traintest

    @property
    def feature_importances_reduction_(self) -> pd.DataFrame:
        """Return feature importance from feature reduction, model training on training data,
        with importances calculated using test data (holdout set)"""
        if not isinstance(self._feature_importances_reduction, pd.DataFrame):
            raise Exception(f'Not available: feature importances from feature reduction.')
        return self._feature_importances_reduction

    @property
    def scores_(self) -> dict:
        """Return model scores for model used in gap-filling"""
        if not self._scores:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores

    @property
    def scores_traintest_(self) -> dict:
        """Return model scores for model trained on training data,
        with scores calculated using test data (holdout set)"""
        if not self._scores_traintest:
            raise Exception(f'Not available: model scores for gap-filling.')
        return self._scores_traintest

    @property
    def gapfilling_df_(self) -> DataFrame:
        """Return gapfilled data and auxiliary variables"""
        if not isinstance(self._gapfilling_df, DataFrame):
            raise Exception(f'Gapfilled data not available.')
        return self._gapfilling_df

    @property
    def traintest_details_(self) -> dict:
        """Return details from train/test splits"""
        if not self._traintest_details:
            raise Exception(f'Not available: details about training & testing.')
        return self._traintest_details

    @property
    def accepted_features_(self) -> list:
        """Return list of accepted features from feature reduction"""
        if not self._accepted_features:
            raise Exception(f'Not available: accepted features from feature reduction.')
        return self._accepted_features

    @property
    def rejected_features_(self) -> list:
        """Return list of rejected features from feature reduction"""
        if self._rejected_features is None:
            raise Exception(f'Not available: rejected features from feature reduction.')
        return self._rejected_features

    @staticmethod
    def _fi_across_splits(feature_importances_splits) -> DataFrame:
        """Calculate overall feature importance as mean across splits."""
        fi_columns = [c for c in feature_importances_splits.columns if str(c).endswith("_IMPORTANCE")]
        fi_df = feature_importances_splits[fi_columns].copy()
        fi_df = fi_df.mean(axis=1)
        return fi_df

    def _remove_rejected_features(self, shap_threshold_factor: float = 1.0, infotxt="[ FEATURE REDUCTION ]") -> list:
        """Remove features that are below importance threshold or that are below
        zero from model dataframe. The updated model dataframe will then be used
        for the next (final) model.
        """

        fi_df = self.feature_importances_reduction_.copy()
        series = fi_df['SHAP_IMPORTANCE'].copy()

        # Threshold for feature reduction: random_importance + k * random_sd
        random_importance = series.loc[self.random_col]
        random_sd = fi_df.loc[self.random_col, 'SHAP_SD'] if 'SHAP_SD' in fi_df.columns else 0
        threshold = random_importance + shap_threshold_factor * random_sd
        print(f"{infotxt} >>> Setting threshold for feature rejection to {threshold}.")
        print(f"{infotxt} >>> Random variable importance: {random_importance:.6f}, SD: {random_sd:.6f}")

        # Get accepted features
        accepted_locs = ((series > threshold) & (series > 0))
        accepted_df = pd.DataFrame(series[accepted_locs])
        accepted_features = accepted_df.index.tolist()
        print(f"\n{infotxt} >>> Accepted features and their importance:\n{accepted_df}")

        # Get rejected features (everything not accepted, incl. boundary and random col)
        rejected_locs = ~accepted_locs
        rejected_df = pd.DataFrame(series[rejected_locs])
        rejected_features = rejected_df.index.tolist()
        print(f"\n{infotxt} >>> Rejected features and their importance:\n{rejected_df}")

        # Update dataframe, keep accepted columns
        accepted_cols = [self.target_col]
        accepted_cols = accepted_cols + accepted_features

        return accepted_cols

    @staticmethod
    def _fitmodel(model, X_train, y_train, X_test, y_test):
        """Fit model."""
        if isinstance(model, RandomForestRegressor):
            model.fit(X=X_train, y=y_train)
        elif isinstance(model, XGBRegressor):
            model.fit(X=X_train, y=y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
        return model

    def trainmodel(self,
                   showplot_scores: bool = True,
                   showplot_importance: bool = True):
        """
        Train random forest model for gap-filling

        No gap-filling is done here, only the model is trained.

        Args:
            showplot_scores: shows plot of predicted vs observed
            showplot_importance: shows plot of SHAP feature importances

        """

        print("\nTraining final model ...")
        idtxt = f"TRAIN & TEST "

        # Set training and testing data
        train_df = self.train_df.copy()
        y_train = np.array(train_df[self.target_col])
        X_train = np.array(self.train_df.drop(self.target_col, axis=1))
        X_test = np.array(self.test_df.drop(self.target_col, axis=1))
        y_test = np.array(self.test_df[self.target_col])
        X_names = self.train_df.drop(self.target_col, axis=1).columns.tolist()

        # Info
        print(f">>> Training model {self.regressor} based on data between "
              f"{self.train_df.index[0]} and {self.train_df.index[-1]} ...")

        # Train the model on training data
        print(f">>> Fitting model to training data ...")
        self._model = self._fitmodel(model=self._model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # Predict targets in test data
        print(f">>> Using model to predict target {self.target_col} in unseen test data ...")
        pred_y_test = self.model_.predict(X=X_test)

        # Calculate SHAP-based feature importance on test data and store in dataframe
        print(f">>> Using model to calculate SHAP feature importance based on unseen test data ...")
        self._feature_importances_traintest = self._shap_importance(
            model=self.model_, X=X_test, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (SHAP) ...")
            plot_feature_importance(feature_importances=self.feature_importances_traintest_)

        # Scores
        print(f">>> Calculating prediction scores based on predicting unseen test data of {self.target_col} ...")
        self._scores_traintest = prediction_scores(predictions=pred_y_test, targets=y_test)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values ...")
            plot_observed_predicted(predictions=pred_y_test,
                                    targets=y_test,
                                    scores=self.scores_traintest_,
                                    infotxt=f"{idtxt} trained on training set, tested on test set",
                                    random_state=self._random_state)

            print(f">>> Plotting residuals and prediction error ...")
            plot_prediction_residuals_error_regr(
                model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                infotxt=f"{idtxt} trained on training set, tested on test set")

        # Collect results
        print(
            f">>> Collecting results, details about training and testing can be accessed by calling .report_traintest().")
        self._traintest_details = dict(
            train_df=self.train_df,
            test_df=self.test_df,
            test_size=self.test_size,
            X_names=X_names,
            model=self.model_,
        )

        print(f">>> Done.")

    def report_traintest(self):
        """Results from model training on test data"""

        idtxt = "MODEL TRAINING & TESTING RESULTS"
        results = self.traintest_details_
        fi = self.feature_importances_traintest_
        test_size_perc = self.test_size * 100
        training_size_perc = 100 - test_size_perc

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"## DATA\n"
            f"  > target: {self.target_col}\n"
            f"  > features: {len(results['X_names'])} {results['X_names']}\n"
            f"  > {len(self.model_df)} records (with missing)\n"
            f"  > {len(self.model_df.dropna())} available records for target and all features (no missing values)\n"
            f"  > training on {len(self.train_df)} records ({training_size_perc:.1f}%) "
            f"of {len(self.train_df)} features between {self.train_df.index[0]} and {self.train_df.index[-1]}\n"
            f"  > testing on {len(self.test_df)} unseen records ({test_size_perc:.1f}%) "
            f"of {self.target_col} between {self.test_df.index[0]} and {self.test_df.index[-1]}\n"
            f"\n"
            f"## MODEL\n"
            f"  > the model was trained on training data ({len(self.train_df)} records)\n"
            f"  > the model was tested on test data ({len(self.test_df)} values)\n"
            f"  > estimator:  {self.model_}\n"
            f"  > parameters:  {self.model_.get_params()}\n"
            f"  > number of features used in model:  {len(results['X_names'])}\n"
            f"  > names of features used in model:  {results['X_names']}\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"  > feature importances were calculated based on unseen test data of {self.target_col} "
            f"({len(self.test_df[self.target_col])} records).\n"
            f"  > feature importances show mean absolute SHAP values.\n"
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"\n"
            f"## MODEL SCORES\n"
            f"  All scores were calculated based on unseen test data ({len(self.test_df[self.target_col])} records).\n"
            f"  > MAE:  {self.scores_traintest_['mae']} (mean absolute error)\n"
            f"  > MedAE:  {self.scores_traintest_['medae']} (median absolute error)\n"
            f"  > MSE:  {self.scores_traintest_['mse']} (mean squared error)\n"
            f"  > RMSE:  {self.scores_traintest_['rmse']} (root mean squared error)\n"
            f"  > MAXE:  {self.scores_traintest_['maxe']} (max error)\n"
            f"  > MAPE:  {self.scores_traintest_['mape']:.3f} (mean absolute percentage error)\n"
            f"  > R2:  {self.scores_traintest_['r2']}\n"
        )

    def fillgaps(self,
                 showplot_scores: bool = True,
                 showplot_importance: bool = True):
        """
        Gap-fill data with previously built model

        No new model is built here, instead the last model built in
        the preceding step .trainmodel() is used.

        y = target
        X = features

        """
        self._fillgaps_fullmodel(showplot_scores, showplot_importance)
        self._fillgaps_fallback()
        self._fillgaps_combinepredictions()

    def reduce_features(self, shap_threshold_factor: float = 0.5):
        """Reduce number of features using SHAP importance

        A random variable is added to features and SHAP importances
        are calculated. The SHAP importance of the random variable is the
        benchmark to determine whether a feature is relevant. Features where
        SHAP importance is smaller or equal to (random_importance + k * random_sd)
        are rejected, where k is shap_threshold_factor.

        Args:
            shap_threshold_factor:
                Factor k for SHAP-based feature reduction threshold.
                Threshold is calculated as: random_importance + k * random_sd
                Default 0.5 uses 0.5-sigma confidence (lenient). Higher values are more conservative
                (reject more features). Lower values are more lenient.
        """

        infotxt = "[ FEATURE REDUCTION ]"

        # Info
        print(f"\n{infotxt} Feature reduction based on SHAP importance ...")

        df = self.train_df.copy()
        df = df.dropna()

        # Add random variable as feature
        df, self.random_col = self._add_random_variable(df=df)

        X = np.array(df.drop(self.target_col, axis=1))
        y = np.array(df[self.target_col])

        # Instantiate model with params
        model = self.regressor(**self.kwargs)

        model.get_params()

        # Fit model to training data
        model = self._fitmodel(model=model, X_train=X, y_train=y, X_test=X, y_test=y)

        # https://mljar.com/blog/visualize-tree-from-random-forest/
        # todo from dtreeviz.trees import dtreeviz  # will be used for tree visualization
        # _ = tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)

        # Calculate SHAP importance for all data
        print(f"{infotxt} >>> Calculating feature importances (SHAP) ...")
        X_names = df.drop(self.target_col, axis=1).columns.tolist()
        feature_importances = self._shap_importance(model=model, X=X, X_names=X_names)
        self._feature_importances_reduction = feature_importances.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        # Remove variables where mean feature importance across all splits is smaller
        # than or equal to random variable
        # Update dataframe for model building
        accepted_cols = self._remove_rejected_features(shap_threshold_factor=shap_threshold_factor)

        # Update model data, keep accepted features
        print(f"{infotxt} >>> Removing rejected features from model data ...")
        self.train_df = self.train_df[accepted_cols].copy()
        self.test_df = self.test_df[accepted_cols].copy()
        self.model_df = self.model_df[accepted_cols].copy()

        self._accepted_features = self.train_df.drop(columns=self.target_col).columns.tolist()
        self._rejected_features = [x for x in X_names if x not in self.accepted_features_ and x != self.random_col]

    def report_feature_reduction(self):
        """Results from feature reduction"""

        idtxt = "FEATURE REDUCTION"

        # # Original features without random variable
        # _X_names = [x for x in fi.index if x != self.random_col]

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"- Target variable: {self.target_col}\n"
            f"\n"
            f"- The random variable {self.random_col} was added to the original features, "
            f"used as benchmark for detecting relevant feature importances.\n"
            f"\n"
            f"SHAP IMPORTANCE (mean absolute SHAP values):\n"
            f"\n"
            f"{self.feature_importances_reduction_}"
            f"\n"
            f"\n"
            f"- These results are from feature reduction. Note that feature importances for "
            f"the final model are calculated during gap-filling.\n"
            f"\n"
            f"--> {len(self.original_input_features)} original input features (before feature reduction): "
            f"{self.original_input_features}\n"
            f"--> {len(self.rejected_features_)} rejected features (during feature reduction): "
            f"{self.rejected_features_ if self.rejected_features_ else 'None.'}\n"
            f"--> {len(self.accepted_features_)} accepted features (after feature reduction): "
            f"{self.accepted_features_}\n"
        )

    def report_gapfilling(self):
        """Results from gap-filling"""
        # Setup
        idtxt = "GAP-FILLING RESULTS"

        df = self.gapfilling_df_
        model = self.model_
        scores = self.scores_
        fi = self.feature_importances_

        feature_names = fi.index.to_list()
        n_features = len(feature_names)

        locs_observed = df[self.target_gapfilled_flag_col] == 0
        locs_hq = df[self.target_gapfilled_flag_col] == 1
        locs_observed_missing_fromflag = df[self.target_gapfilled_flag_col] > 0
        locs_fallback = df[self.target_gapfilled_flag_col] == 2

        n_observed = locs_observed.sum()
        n_hq = locs_hq.sum()
        n_observed_missing_fromflag = locs_observed_missing_fromflag.sum()
        n_available = len(df[self.target_gapfilled_col].dropna())
        n_potential = len(df.index)
        n_fallback = locs_fallback.sum()
        test_size_perc = self.test_size * 100

        print(
            f"\n"
            f"{'=' * len(idtxt)}\n"
            f"{idtxt}\n"
            f"{'=' * len(idtxt)}\n"
            f"\n"
            f"Model scores and feature importances were calculated from high-quality "
            f"predicted targets ({n_hq} values, {self.target_gapfilled_col} where flag=1) "
            f"in comparison to observed targets ({n_observed} values, {self.target_col}).\n"
            f"\n"
            f"## TARGET\n"
            f"- first timestamp:  {df.index[0]}\n"
            f"- last timestamp:  {df.index[-1]}\n"
            f"- potential number of values: {n_potential} values)\n"
            f"- target column (observed):  {self.target_col}\n"
            f"- missing records (observed):  {df[self.target_col].isnull().sum()} "
            f"(cross-check from flag: {n_observed_missing_fromflag})\n"
            f"- target column (gap-filled):  {self.target_gapfilled_col}  ({n_available} values)\n"
            f"- missing records (gap-filled):  {df[self.target_gapfilled_col].isnull().sum()}\n"
            f"- gap-filling flag: {self.target_gapfilled_flag_col}\n"
            f"  > flag 0 ... observed targets ({n_observed} values)\n"
            f"  > flag 1 ... targets gap-filled with high-quality, all features available ({n_hq} values)\n"
            f"  > flag 2 ... targets gap-filled with fallback ({n_fallback} values)\n"
            f"\n"
            f"## FEATURE IMPORTANCES\n"
            f"- names of features used in model:  {feature_names}\n"
            f"- number of features used in model:  {n_features}\n"
            f"- feature importances calculated using SHAP (TreeExplainer).\n"
            f"\n"
            f"{fi}"
            f"\n"
            f"\n"
            f"## MODEL\n"
            f"The model was trained on a training set with test size {test_size_perc:.2f}%.\n"
            f"- estimator:  {model}\n"
            f"- parameters:  {model.get_params()}\n"
            f"\n"
            f"## MODEL SCORES\n"
            f"- MAE:  {scores['mae']} (mean absolute error)\n"
            f"- MedAE:  {scores['medae']} (median absolute error)\n"
            f"- MSE:  {scores['mse']} (mean squared error)\n"
            f"- RMSE:  {scores['rmse']} (root mean squared error)\n"
            f"- MAXE:  {scores['maxe']} (max error)\n"
            f"- MAPE:  {scores['mape']:.3f} (mean absolute percentage error)\n"
            f"- R2:  {scores['r2']}\n"
        )

    def _create_lagged_variants(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add lagged features because there are no original features.")
        _out_df = lagged_variants(df=work_df,
                                  stepsize=self.features_lag_stepsize,
                                  lag=self.features_lag,
                                  exclude_cols=self.features_lag_exclude_cols,
                                  verbose=self.verbose)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_rolling_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add rolling features because there are no original features.")
        _out_df = self._rolling_features(df=work_df,
                                         windows=self.features_rolling,
                                         exclude_cols=self.features_rolling_exclude_cols)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_rolling_features_advanced(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add advanced rolling features because there are no original features.")
        _out_df = self._rolling_features_advanced(df=work_df,
                                                  windows=self.features_rolling,
                                                  stats=self.features_rolling_stats,
                                                  exclude_cols=self.features_rolling_exclude_cols)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_differencing_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add differencing features because there are no original features.")
        _out_df = self._differencing_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_ema_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add EMA features because there are no original features.")
        _out_df = self._ema_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _create_additional_datacols(self) -> pd.DataFrame:

        # Dataframe that contains the target and all original features
        expanded_df = self.model_df.copy()

        # Add lagged cols
        if self.features_lag:
            expanded_df = self._create_lagged_variants(work_df=self.model_df[self.original_input_features].copy(),
                                                       expanded_df=expanded_df)

        if self.features_rolling:
            expanded_df = self._create_rolling_features(work_df=self.model_df[self.original_input_features].copy(),
                                                        expanded_df=expanded_df)

        if self.features_rolling and self.features_rolling_stats:
            expanded_df = self._create_rolling_features_advanced(work_df=self.model_df[self.original_input_features].copy(),
                                                                 expanded_df=expanded_df)

        if self.features_diff:
            expanded_df = self._create_differencing_features(work_df=self.model_df[self.original_input_features].copy(),
                                                             expanded_df=expanded_df)

        if self.features_ema:
            expanded_df = self._create_ema_features(work_df=self.model_df[self.original_input_features].copy(),
                                                    expanded_df=expanded_df)

        if self.features_poly_degree:
            expanded_df = self._create_polynomial_features(work_df=expanded_df)

        if self.features_stl:
            expanded_df = self._create_stl_features(work_df=self.model_df[self.original_input_features].copy(),
                                                   expanded_df=expanded_df)

        if self.vectorize_timestamps:
            expanded_df = vectorize_timestamps(df=expanded_df, txt="")
            # For cyclical variables, keep only the sine/cosine variants, drop linear versions
            expanded_df = expanded_df.drop(columns=['.HOUR', '.SEASON', '.MONTH', '.WEEK', '.DOY'])

        if self.add_continuous_record_number:
            expanded_df = fr.add_continuous_record_number(df=expanded_df)

        # Timestamp sanitizer
        if self.sanitize_timestamp:
            verbose = True if self.verbose > 0 else False
            tss = TimestampSanitizer(data=expanded_df, output_middle_timestamp=True, verbose=verbose)
            expanded_df = tss.get()

        return expanded_df

    def _shap_importance(self, model, X, X_names) -> DataFrame:
        """
        Calculate SHAP-based feature importance.

        Uses TreeExplainer for tree-based models (XGBoost, RandomForest).
        Returns mean absolute SHAP values as feature importance.
        """

        # Create explainer and calculate SHAP values
        # Handle XGBoost base_score parameter format issue with monkey-patch
        # Some XGBoost/environment combinations return base_score as '[-4.121306E0]' which
        # float() cannot parse. We monkey-patch float() to handle this.
        _builtin_float = float

        def _patched_float(x):
            """float() that handles bracket-enclosed scientific notation like '[-4.121306E0]'"""
            if isinstance(x, str):
                x_stripped = x.strip('[]')
                if x_stripped != x:  # Only use patched version if brackets were removed
                    return _builtin_float(x_stripped)
            return _builtin_float(x)

        # Temporarily replace float in builtins
        import builtins
        original_float = builtins.float
        builtins.float = _patched_float

        try:
            explainer = shap.TreeExplainer(model)
        finally:
            # Always restore original float
            builtins.float = original_float

        shap_values = explainer.shap_values(X)

        # Handle case where shap_values is a list (for some model types)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP values as importance
        importance_values = np.abs(shap_values).mean(axis=0)

        # Calculate standard deviation for reference
        importance_std = np.abs(shap_values).std(axis=0)

        # Create DataFrame with feature importances
        fidf = pd.DataFrame({
            'SHAP_IMPORTANCE': importance_values,
            'SHAP_SD': importance_std
        }, index=X_names)

        fidf = fidf.sort_values(by='SHAP_IMPORTANCE', ascending=False)

        return fidf

    def _add_random_variable(self, df: DataFrame) -> tuple[DataFrame, str]:
        # Add random variable as benchmark for relevant feature importances
        random_col = '.RANDOM'  # Random variable as benchmark for relevant importances
        df[random_col] = np.random.RandomState(self._random_state).randn(df.shape[0])
        return df, random_col

    # def _lag_features(self, features_lag_exclude_cols):
    #     """Add lagged variants of variables as new features"""
    #     exclude_cols = [self.target_col]
    #     if features_lag_exclude_cols:
    #         exclude_cols += features_lag_exclude_cols
    #     return diive.pkgs.createvar.laggedvariants.lagged_variants(df=self.model_df,
    #                                                                stepsize=self.features_lag_stepsize,
    #                                                                lag=self.features_lag,
    #                                                                exclude_cols=exclude_cols,
    #                                                                verbose=self.verbose)

    def _rolling_features(self, df: pd.DataFrame, windows: list, exclude_cols: list = None) -> pd.DataFrame:
        """Add rolling mean and std of feature columns at multiple window sizes.

        For each window size w and each feature column col (excluding target and
        any cols in exclude_cols), two new columns are added:
            '.{col}_mean{w}' — rolling mean over the previous w records
            '.{col}_std{w}'  — rolling std over the previous w records

        Rolling statistics use min_periods=1 so no new NaN values are introduced
        at the start of the series.

        Args:
            df: DataFrame with feature columns and DatetimeIndex.
            windows: List of window sizes in records (e.g. [6, 48] for 3h and 24h
                     at 30-min resolution).
            exclude_cols: Column names to skip. Target column is always excluded.

        Returns:
            DataFrame with additional rolling feature columns appended.
        """
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
            print(f"++ Added rolling features (windows={windows}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

    def _rolling_features_advanced(self, df: pd.DataFrame, windows: list, stats: list,
                                    exclude_cols: list = None) -> pd.DataFrame:
        """Add advanced rolling statistics (median, min, max, percentiles) to features.

        For each window size w, feature column col, and statistic stat, creates:
            '.{col}_ROLL{STAT}{w}' — rolling statistic over the previous w records

        Rolling statistics use min_periods=1 so no new NaN values are introduced
        at the start of the series.

        Args:
            df: DataFrame with feature columns and DatetimeIndex.
            windows: List of window sizes in records (e.g. [6, 48] for 3h and 24h
                     at 30-min resolution).
            stats: List of statistics to compute. Options: 'median', 'min', 'max',
                   'std', 'q25', 'q75'
            exclude_cols: Column names to skip. Target column is always excluded.

        Returns:
            DataFrame with additional rolling feature columns appended.
        """
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
        """Add 1st and 2nd order differences for capturing temporal momentum.

        For each difference order and each feature column, creates a new column:
            '.{col}_DIFF{order}' — difference at the specified order

        Higher order differences capture acceleration/curvature in feature changes.
        Useful for flux data where momentum and rate-of-change are predictive.

        Returns:
            DataFrame with additional differencing feature columns appended.
        """
        if not self.features_diff:
            return df

        exclude = [self.target_col] + (self.features_diff_exclude_cols or [])
        # Only difference original features, not engineered ones starting with '.'
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
        """Add exponential moving average (EMA) of feature columns.

        For each span value and each feature column, creates:
            '.{col}_EMA{span}' — EMA with the specified span

        EMA applies exponential decay weighting where recent values are weighted more.
        Uses min_periods=1 to avoid introducing new NaN values.

        Returns:
            DataFrame with additional EMA feature columns appended.
        """
        if not self.features_ema:
            return df

        exclude = [self.target_col] + (self.features_ema_exclude_cols or [])
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('.')]
        newcols = []

        for span in self.features_ema:
            if span < 1:
                continue
            # adjust=False uses expanding window behavior (more standard for time series)
            ema_df = df[feature_cols].ewm(span=span, min_periods=1, adjust=False).mean()
            ema_df.columns = [f'.{c}_EMA{span}' for c in feature_cols]
            df = pd.concat([df, ema_df], axis=1)
            newcols += ema_df.columns.tolist()

        if self.verbose:
            print(f"++ Added EMA features (spans={self.features_ema}) for {len(feature_cols)} columns: "
                  f"{newcols}")
        return df

    def _create_polynomial_features(self, work_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for polynomial feature creation."""
        _out_df = self._polynomial_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in work_df.columns]
        work_df = work_df.join(_out_df[newcols])
        return work_df

    def _create_stl_features(self, work_df: pd.DataFrame, expanded_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper for STL decomposition feature creation."""
        if len(work_df.columns) == 0:
            raise ValueError("Cannot add STL features because there are no original features.")
        _out_df = self._stl_features(df=work_df)
        newcols = [c for c in _out_df.columns if c not in expanded_df.columns]
        expanded_df = expanded_df.join(_out_df[newcols])
        return expanded_df

    def _stl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add STL (Seasonal-Trend Loess) decomposition components as features.

        For each feature column (excluding target and specified exclude_cols), applies
        Seasonal-Trend decomposition to extract trend, seasonal, and residual components.
        Only applies to complete columns (no gaps) to avoid circular dependencies.

        For each selected component, creates new columns:
            '.{col}_STL_TREND' — trend component (long-term direction)
            '.{col}_STL_SEASONAL' — seasonal component (recurring patterns)
            '.{col}_STL_RESIDUAL' — residual component (noise/anomalies)

        STL features capture:
            - Trend: Slow, monotonic changes (instrument drift, seasonal baseline shifts)
            - Seasonal: Recurring periodic patterns (diurnal, weekly, annual cycles)
            - Residual: High-frequency noise and anomalies

        Advantages:
            - Robust to non-stationary data and gaps (uses only complete columns)
            - Reveals structure without assumptions of stationarity
            - Captures multi-scale temporal patterns (trend, seasonal, residual)
            - Quality-weighted: can incorporate data quality flags

        Disadvantages:
            - Only applies to complete columns (skips columns with gaps)
            - Requires seasonal period specification or auto-detection
            - Higher computational cost than rolling statistics
            - Extracted components are smoothed (less granular than original data)

        Returns:
            DataFrame with additional STL feature columns appended.
        """
        if not self.features_stl:
            return df

        from diive.pkgs.analyses.seasonaltrend import SeasonalTrendDecomposition

        exclude = [self.target_col] + (self.features_stl_exclude_cols or [])
        # Only decompose original features, not engineered ones starting with '.'
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

    def _polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features by expanding each feature to specified degree.

        For each polynomial degree and each feature column, creates new columns:
            '.{col}_POL{degree}' — feature raised to the specified degree

        Polynomial features capture non-linear relationships.
        Useful for modeling phenomena with squared or cubic relationships (e.g., radiation effects).

        Returns:
            DataFrame with additional polynomial feature columns appended.
        """
        if not self.features_poly_degree or self.features_poly_degree < 2:
            return df

        exclude = [self.target_col] + (self.features_poly_exclude_cols or [])
        # Only create polynomial features for original and lagged features, not engineered ones starting with '.'
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

    def _check_n_cols(self):
        """Check number of columns"""
        if len(self.model_df.columns) == 1:
            raise Exception(f"(!) Stopping execution because dataset comprises "
                            f"only one single column : {self.model_df.columns}")

    def _fillgaps_fullmodel(self, showplot_scores, showplot_importance):
        """Apply model to fill missing targets for records where all features are available
        (high-quality gap-filling)"""

        print("\nGap-filling using final model ...")

        # Original input data, contains target and features
        # This dataframe has the full timestamp
        df = self.model_df.copy()

        # Test how the model performs with all y data
        # Since the model was previously trained on test data,
        # here it is checked how well the model performs when
        # predicting all available y data.
        # This is needed to calculate feature importance and scores.
        y, X, X_names, timestamp = fr.convert_to_arrays(
            df=df, target_col=self.target_col, complete_rows=True)

        # Predict all targets (no test split)
        print(f">>> Using final model on all data to predict target {self.target_col} ...")
        pred_y = self.model_.predict(X=X)

        # Calculate SHAP-based feature importance and store in dataframe
        print(f">>> Using final model on all data to calculate SHAP feature importance ...")
        self._feature_importances = self._shap_importance(
            model=self._model, X=X, X_names=X_names)

        if showplot_importance:
            print(">>> Plotting feature importances (SHAP) ...")
            plot_feature_importance(feature_importances=self.feature_importances_)

        # Scores, using all targets
        print(f">>> Calculating prediction scores based on all data predicting {self.target_col} ...")
        self._scores = prediction_scores(predictions=pred_y, targets=y)

        if showplot_scores:
            print(f">>> Plotting observed and predicted values based on all data ...")
            plot_observed_predicted(predictions=pred_y,
                                    targets=y,
                                    scores=self.scores_,
                                    infotxt=f"trained on training set, tested on FULL set",
                                    random_state=self._random_state)

            # print(f">>> Plotting residuals and prediction error based on all data ...")
            # plot_prediction_residuals_error_regr(
            #     model=self.model_, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            #     infotxt=f"trained on training set, tested on full set")

        # In the next step, all available features are used to
        # predict the target for records where all features are available.
        # Feature data for records where all features are available:
        print(f">>> Predicting target {self.target_col} where all features are available ...", end=" ")
        features_df = df.drop(self.target_col, axis=1)  # Remove target data
        features_df = features_df.dropna()  # Keep rows where all features available
        X = features_df.to_numpy()  # Features are needed as numpy array
        feature_names = features_df.columns.tolist()

        # Predict targets for all records where all features are available
        pred_y = self.model_.predict(X=X)
        print(f"predicted {len(pred_y)} records.")

        # Collect gapfilling results in df
        # Define column names for gapfilled_df
        print(">>> Collecting results for final model ...")
        self._define_cols()

        # Collect predictions in dataframe
        self._gapfilling_df = pd.DataFrame(data={self.pred_fullmodel_col: pred_y}, index=features_df.index)

        # Add target to dataframe
        self._gapfilling_df[self.target_col] = df[self.target_col].copy()

        # Gap locations
        # Make column that contains predicted values
        # for rows where target is missing
        _gap_locs = self._gapfilling_df[self.target_col].isnull()  # Locations where target is missing
        self._gapfilling_df[self.pred_gaps_col] = self._gapfilling_df.loc[
            _gap_locs, self.pred_fullmodel_col]

        # Flag
        # Make flag column that indicates where predictions for
        # missing targets are available, where 0=observed, 1=gapfilled
        # todo Note that missing predicted gaps = 0. change?
        _gapfilled_locs = self._gapfilling_df[self.pred_gaps_col].isnull()  # Non-gapfilled locations
        _gapfilled_locs = ~_gapfilled_locs  # Inverse for gapfilled locations
        self._gapfilling_df[self.target_gapfilled_flag_col] = _gapfilled_locs
        self._gapfilling_df[self.target_gapfilled_flag_col] = self._gapfilling_df[
            self.target_gapfilled_flag_col].astype(
            int)

        # Gap-filled time series
        # Fill missing records in target with predicions
        n_missing = self._gapfilling_df[self.target_col].isnull().sum()
        print(f">>> Filling {n_missing} missing records in target with predictions from final model ...")
        print(f">>> Storing gap-filled time series in variable {self.target_gapfilled_col} ...")
        self._gapfilling_df[self.target_gapfilled_col] = \
            self._gapfilling_df[self.target_col].fillna(self._gapfilling_df[self.pred_fullmodel_col])

        # Restore original full timestamp
        print(">>> Restoring original timestamp in results ...")
        self._gapfilling_df = self._gapfilling_df.reindex(df.index)

        # SHAP values
        # https://pypi.org/project/shap/
        # https://mljar.com/blog/feature-importance-in-random-forest/

    def _fillgaps_fallback(self):

        # Fallback gapfilling
        # Fill still existing gaps in full timestamp data
        # Build fallback model exclusively from timestamp features.
        # Here, the model is trained on the already gapfilled time series,
        # using info from the timestamp, e.g. DOY
        _still_missing_locs = self._gapfilling_df[self.target_gapfilled_col].isnull()
        _num_still_missing = _still_missing_locs.sum()  # Count number of still-missing values

        print(f"\nGap-filling {_num_still_missing} remaining missing records in "
              f"{self.target_gapfilled_col} using fallback model ...")

        if _num_still_missing > 0:

            print(f">>> Fallback model is trained on {self.target_gapfilled_col} and timestamp info ...")

            fallback_predictions, \
                fallback_timestamp = \
                self._predict_fallback(series=self._gapfilling_df[self.target_gapfilled_col])

            fallback_series = pd.Series(data=fallback_predictions, index=fallback_timestamp)
            self._gapfilling_df[self.pred_fallback_col] = fallback_series
            self._gapfilling_df[self.target_gapfilled_col] = \
                self._gapfilling_df[self.target_gapfilled_col].fillna(fallback_series)

            self._gapfilling_df.loc[_still_missing_locs, self.target_gapfilled_flag_col] = 2  # Adjust flag, 2=fallback
        else:
            print(f">>> Fallback model not necessary, all gaps were already filled.")
            self._gapfilling_df[self.pred_fallback_col] = None

        # Cumulative
        self._gapfilling_df[self.target_gapfilled_cumu_col] = \
            self._gapfilling_df[self.target_gapfilled_col].cumsum()

    def _fillgaps_combinepredictions(self):
        """Combine predictions of full model with fallback predictions"""
        print(">>> Combining predictions from full model and fallback model ...")
        # First add predictions from full model
        self._gapfilling_df[self.pred_col] = self._gapfilling_df[self.pred_fullmodel_col].copy()
        # Then fill remaining gaps with predictions from fallback model
        self._gapfilling_df[self.pred_col] = (
            self._gapfilling_df[self.pred_col].fillna(self._gapfilling_df[self.pred_fallback_col]))

    def _predict_fallback(self, series: pd.Series):
        """Fill data gaps using timestamp features only, fallback for still existing gaps"""
        gf_fallback_df = pd.DataFrame(series)
        gf_fallback_df = vectorize_timestamps(df=gf_fallback_df, txt="(ONLY FALLBACK)")

        # Build model for target predictions *from timestamp*
        y_fallback, X_fallback, _, _ = \
            fr.convert_to_arrays(df=gf_fallback_df,
                                 target_col=self.target_gapfilled_col,
                                 complete_rows=True)

        # Instantiate new model with same params as before
        model_fallback = self.regressor(**self.kwargs)

        # Train the model on all available records ...
        model_fallback = self._fitmodel(model=model_fallback, X_train=X_fallback, y_train=y_fallback, X_test=X_fallback,
                                        y_test=y_fallback)
        # model_fallback.fit(X=X_fallback, y=y_fallback)

        # ... and use it to predict all records for full timestamp
        full_timestamp_df = gf_fallback_df.drop(self.target_gapfilled_col, axis=1)  # Remove target data
        X_fallback_full = full_timestamp_df.to_numpy()  # Features are needed as numpy array

        print(f">>> Predicting target {self.target_gapfilled_col} using fallback model ...")
        pred_y_fallback = model_fallback.predict(X=X_fallback_full)  # Predict targets in test data
        full_timestamp = full_timestamp_df.index

        return pred_y_fallback, full_timestamp

    def _results(self, gapfilled_df, most_important_df, model_r2, still_missing_locs):
        """Summarize gap-filling results"""

        _vals_max = len(gapfilled_df.index)
        _vals_before = len(gapfilled_df[self.target_col].dropna())
        _vals_after = len(gapfilled_df[self.target_gapfilled_col].dropna())
        _vals_fallback_filled = still_missing_locs.sum()
        _perc_fallback_filled = (_vals_fallback_filled / _vals_max) * 100

        print(f"Gap-filling results for {self.target_col}\n"
              f"max possible: {_vals_max} values\n"
              f"before gap-filling: {_vals_before} values\n"
              f"after gap-filling: {_vals_after} values\n"
              f"gap-filled with fallback: {_vals_fallback_filled} values / {_perc_fallback_filled:.1f}%\n"
              f"used features:\n{most_important_df}\n"
              f"predictions vs targets, R2 = {model_r2:.3f}")

    def _define_cols(self):
        self.pred_col = ".PREDICTIONS"
        self.pred_fullmodel_col = ".PREDICTIONS_FULLMODEL"
        self.pred_fallback_col = ".PREDICTIONS_FALLBACK"
        self.pred_gaps_col = ".GAP_PREDICTIONS"
        self.target_gapfilled_col = f"{self.target_col}{self.gfsuffix}"
        self.target_gapfilled_flag_col = f"FLAG_{self.target_gapfilled_col}_ISFILLED"  # "[0=measured]"
        self.target_gapfilled_cumu_col = ".GAPFILLED_CUMULATIVE"


def plot_feature_importance(feature_importances: pd.DataFrame):
    """
    Plot SHAP feature importance as horizontal bar chart with error bars.

    Visualizes relative importance of features in the model with standard deviation
    as error bars. Features are sorted by importance for easy interpretation.
    """
    # Scientific color palette
    COLOR_BAR = '#003A70'  # Deep Blue
    COLOR_ERROR = '#C41E3A'  # Crimson Red (error bars)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, ax = plt.subplots(figsize=(10, max(8, len(feature_importances) * 0.35)), dpi=100)

    # Prepare data
    _fidf = feature_importances.copy().sort_values(by='SHAP_IMPORTANCE', ascending=True)
    importances = _fidf['SHAP_IMPORTANCE']
    errors = _fidf['SHAP_SD']
    labels = _fidf.index

    # Create horizontal bar chart
    bars = ax.barh(range(len(importances)), importances,
                   color=COLOR_BAR, alpha=0.85, edgecolor=COLOR_TEXT, linewidth=0.8)

    # Add error bars
    ax.errorbar(importances, range(len(importances)), xerr=errors,
                fmt='none', ecolor=COLOR_ERROR, elinewidth=2, capsize=4,
                capthick=2, alpha=0.8, zorder=3)

    # Styling
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color=COLOR_TEXT)
    ax.set_xlabel('Feature Importance (mean |SHAP value|)', fontsize=14,
                  color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Feature', fontsize=14, color=COLOR_TEXT, fontweight='600')
    ax.set_title('SHAP Feature Importance', fontsize=16, fontweight='bold',
                 color='black', pad=15)

    # Grid
    ax.grid(True, axis='x', alpha=0.4, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11,
                   length=4, width=1, color=COLOR_TEXT)
    ax.tick_params(axis='x', labelsize=10)

    # Add value labels on bars
    for i, (imp, err) in enumerate(zip(importances, errors)):
        ax.text(imp + err * 0.5, i, f'{imp:.3f}', va='center', fontsize=9.5,
                color=COLOR_TEXT, fontweight='500')

    fig.tight_layout()
    fig.show()


def plot_observed_predicted(targets: np.ndarray,
                            predictions: np.ndarray,
                            scores: dict,
                            infotxt: str = "",
                            random_state: int = None):
    """
    Plot observed vs. predicted values with enhanced visual styling.

    Creates a 2-panel figure showing:
    - Left: Actual vs. Predicted scatter with accuracy bands and perfect prediction line
    - Right: Residuals vs. Predicted with zero line and error regions

    Visual styling follows diive's Material Design theme with color-coded accuracy zones.
    """
    # Scientific color palette - high contrast, publication-ready
    COLOR_SCATTER = '#003A70'  # Deep Blue
    COLOR_RESIDUAL = '#C41E3A'  # Crimson Red
    COLOR_PERFECT = '#2C3E50'  # Dark Slate Blue-Gray
    COLOR_GOOD = '#F4A300'  # Golden Yellow (±10% error)
    COLOR_WARN = '#E67F0D'  # Deep Orange (±20% error)
    COLOR_ERROR = '#C41E3A'  # Crimson Red (>20% error)
    COLOR_GRID = '#BDC3C7'  # Cool Gray
    COLOR_ZERO = '#000000'  # Black
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5.5), dpi=100)

    # ==================== PANEL 1: Actual vs. Predicted ====================
    ax = axs[0]

    # Calculate data ranges for reference lines and zones
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    margin = (max_val - min_val) * 0.05
    plot_min, plot_max = min_val - margin, max_val + margin

    # Add accuracy zones (filled regions)
    # Perfect prediction line (y=x) with ±10% and ±20% bands
    x_ref = np.array([plot_min, plot_max])

    # ±20% error zone (strong orange, widest)
    ax.fill_between(x_ref, x_ref * 0.80, x_ref * 1.20,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±20% error band')

    # ±10% error zone (strong green, narrower)
    ax.fill_between(x_ref, x_ref * 0.90, x_ref * 1.10,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±10% error band')

    # Perfect prediction line (diagonal)
    ax.plot(x_ref, x_ref, '--', color=COLOR_PERFECT, lw=2, alpha=0.9,
            label='Perfect prediction', zorder=2)

    # Scatter plot with custom styling
    ax.scatter(targets, predictions,
               c=COLOR_SCATTER, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Predictions')

    # Formatting
    ax.set_xlabel('Observed values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Actual vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])
    ax.set_ylim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== PANEL 2: Residuals vs. Predicted ====================
    ax = axs[1]
    residuals = targets - predictions

    # Calculate residual statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    # Add reference bands (±1σ, ±2σ)
    zero_line_y = [plot_min, plot_max]
    ax.fill_between([plot_min, plot_max], -2 * std_residual, 2 * std_residual,
                    color=COLOR_WARN, alpha=0.18, zorder=0, label='±2σ region')
    ax.fill_between([plot_min, plot_max], -1 * std_residual, 1 * std_residual,
                    color=COLOR_GOOD, alpha=0.22, zorder=1, label='±1σ region')

    # Zero line (perfect predictions have zero residuals)
    ax.axhline(y=0, color=COLOR_ZERO, linestyle='-', linewidth=1.5, alpha=0.85, zorder=2)

    # Scatter plot for residuals
    ax.scatter(predictions, residuals,
               c=COLOR_RESIDUAL, edgecolors=COLOR_PERFECT, s=50, alpha=0.75,
               linewidth=0.8, zorder=3, label='Residuals')

    # Formatting
    ax.set_xlabel('Predicted values', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Residuals (Observed − Predicted)', fontsize=16, color=COLOR_TEXT, fontweight='600')
    ax.set_title('Residuals vs. Predicted', fontsize=15, fontweight='bold', color='black', pad=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim([plot_min, plot_max])

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13,
                   length=5, width=1, color=COLOR_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)

    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor=COLOR_GRID)

    # ==================== Figure Title and Info Box ====================
    n_vals = len(predictions)
    mae = scores['mae']
    rmse = scores['rmse']
    r2 = scores['r2']

    # Main title
    title_text = f"Model Predictions: {infotxt}" if infotxt else "Model Predictions"
    fig.suptitle(title_text, fontsize=18, fontweight='bold', color='black', y=0.99)

    # Info box with metrics (positioned in figure space)
    info_lines = [
        f"n = {n_vals:,} samples",
        f"MAE = {mae:.4f}",
        f"RMSE = {rmse:.4f}",
        f"R² = {r2:.4f}"
    ]
    info_text = '\n'.join(info_lines)

    fig.text(0.99, 0.01, info_text,
             fontsize=11, color=COLOR_TEXT, verticalalignment='bottom',
             horizontalalignment='right', family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5',
                       edgecolor=COLOR_GRID, linewidth=1.2, alpha=0.95))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


def plot_prediction_residuals_error_regr(model,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         infotxt: str):
    """
    Plot residuals and prediction error diagnostics.

    Creates two diagnostic plots using yellowbrick visualizers:
    - Left: Q-Q plot of residuals to assess normality assumption
    - Right: Actual vs. Predicted with prediction error visualization

    Args:
        model: Fitted regression model with predict() method
        X_train: predictors in training data (n_samples, n_features)
        y_train: targets in training data (n_samples,)
        X_test: predictors in test data (n_samples, n_features)
        y_test: targets in test data (n_samples,)
        infotxt: text displayed in figure header for context

    Notes:
        - Q-Q plot: If residuals are normally distributed, points should fall on
          the diagonal line. Deviations suggest non-normality.
        - Prediction Error: Shows scatter of predictions vs. actual values with
          perfect prediction line (y=x) and prediction confidence region.

    References:
        - https://www.scikit-yb.org/en/latest/api/regressor/residuals.html
        - https://www.scikit-yb.org/en/latest/api/regressor/peplot.html
    """

    # Scientific color palette for consistency
    COLOR_POINTS = '#003A70'  # Deep Blue
    COLOR_LINE = '#F4A300'  # Golden Yellow
    COLOR_ERROR = '#C41E3A'  # Crimson Red
    COLOR_TEXT = '#2C3E50'  # Dark Slate Gray

    # ==================== PLOT 1: Q-Q Plot (Residuals Normality) ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Residuals Analysis: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    # Q-Q Plot detects normality violations
    vis = ResidualsPlot(model, hist=False, qqplot=True, ax=ax)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Q-Q Plot (Normality Assessment)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Sample Quantiles', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        else:
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)

    plt.tight_layout()
    vis.show()

    # ==================== PLOT 2: Prediction Error Plot ====================
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    fig.suptitle(f"Prediction Error: {infotxt}", fontsize=15, fontweight='bold', y=0.98)

    vis = PredictionError(model)
    vis.fit(X_train, y_train)
    vis.score(X_test, y_test)

    # Enhance styling
    ax.set_title('Actual vs. Predicted (with Confidence Region)', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Actual Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.set_ylabel('Predicted Values', fontsize=12, color=COLOR_TEXT, fontweight='600')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Update scatter point colors
    for collection in ax.collections:
        collection.set_edgecolor(COLOR_TEXT)
        collection.set_facecolor(COLOR_POINTS)
        collection.set_alpha(0.7)
        collection.set_linewidth(0.5)

    # Update line colors
    for line in ax.get_lines():
        if line.get_marker() in ['.', 'o', 's']:
            line.set_color(COLOR_POINTS)
            line.set_alpha(0.7)
        elif line.get_linestyle() == '--':
            line.set_color(COLOR_LINE)
            line.set_linewidth(2)
            line.set_alpha(0.85)
        else:
            line.set_color(COLOR_ERROR)
            line.set_linewidth(1.5)
            line.set_alpha(0.7)

    plt.tight_layout()
    vis.show()
