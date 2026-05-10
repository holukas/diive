"""
U* Moving Point (MP) Detection - Papale et al. (2006)

This module implements the ONEFlux USTAR moving point threshold detection algorithm
as described in Papale et al. (2006). The algorithm identifies the friction velocity (u*)
threshold below which eddy covariance flux measurements are unreliable due to insufficient
turbulent mixing.

Algorithm Overview:
1. Filter data to NIGHTTIME ONLY (SW_IN < 10 W/m²) where respiration is pure and stable
2. Stratify data by SEASON (4 groups), then TEMPERATURE CLASS (7 classes), then USTAR CLASS (20 classes)
3. For each temperature class:
   - Validate temperature-USTAR independence (correlation check)
   - Create quantile-based USTAR classes
   - Compute mean respiration (NEE) per USTAR class
   - Find stability threshold using forward/back mode detection
4. Aggregate thresholds across temperature classes using median
5. Bootstrap resampling provides uncertainty estimates

The forward mode searches from low to high USTAR for where respiration stabilizes.
The back mode works backwards from high USTAR for additional validation.

References:
    Papale, D., et al. (2006). Towards a standardized processing of net ecosystem
    productivity and flux measurements of seasonal and interannual variability.
    Biogeosciences. https://doi.org/10.5194/bg-3-571-2006
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings


class UstarMovingPointDetection:
    """
    Detect USTAR threshold using ONEFlux moving point method (Papale et al., 2006).

    This class implements the complete USTAR threshold detection workflow for eddy covariance
    flux measurements. It identifies the friction velocity (u*) above which flux measurements
    are reliable by analyzing the stability of nighttime respiration across USTAR classes.

    The key insight is that during nighttime, without photosynthesis, NEE represents pure
    ecosystem respiration. At low u* (poor turbulence), measured respiration is systematically
    underestimated due to storage and vertical mixing limitations. At high u* (good turbulence),
    respiration stabilizes. The threshold is where respiration stops changing with increasing u*.

    Algorithm steps:
    1. FILTER DATA: Select nighttime records only (SW_IN < 10 W/m²) for pure respiration signal
    2. STRATIFY BY SEASON: Divide year into 4 seasons (winter, spring, summer, fall)
    3. STRATIFY BY TEMPERATURE: Within each season, divide into 7 temperature classes
    4. STRATIFY BY USTAR: Within each temperature class, divide into 20 USTAR classes
    5. COMPUTE STATISTICS: Calculate mean respiration per USTAR class
    6. VALIDATE: Check temperature-USTAR independence and first USTAR validity
    7. DETECT THRESHOLD: Use forward/back mode detection to find stability point
    8. AGGREGATE: Median across temperature classes gives season threshold
    9. BOOTSTRAP: Repeat with resampled data for uncertainty estimation

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index and required columns: NEE, TA, USTAR, SW_IN
    nee_col : str, optional
        Net ecosystem exchange column name (auto-detected if None)
        Looks for columns containing 'NEE' and 'QCF' preferentially
    ta_col : str, optional
        Air temperature column name (auto-detected if None)
        Looks for columns containing 'TA_' or similar
    ustar_col : str, optional
        Friction velocity column name (auto-detected if None)
        Looks for columns containing 'USTAR' or 'U_STAR'
    swin_col : str, optional
        Shortwave radiation column name (auto-detected if None)
        Used to identify nighttime (SW_IN < 10 W/m²)
    ta_classes_count : int, default=7
        Number of temperature stratification classes per season (matches ONEFlux default)
    ustar_classes_count : int, default=20
        Number of USTAR stratification classes per temperature class (matches ONEFlux default)
    season_groups : List[List[int]], optional
        Custom season grouping by month numbers [1-12]
        Default: [[12,1,2], [3,4,5], [6,7,8], [9,10,11]] for DJF, MAM, JJA, SON
    bootstrapping_times : int, default=100
        Number of bootstrap resampling iterations for uncertainty estimation
    verbose : int, default=0
        Verbosity level: 0=silent, 1=progress, 2=detailed debug output

    Attributes
    ----------
    results_ : pd.DataFrame
        Detected thresholds with columns: 'forward_mode', 'back_mode'
        Index contains season labels (Season 1, 2, 3, 4)
    bootstrap_stats_ : pd.DataFrame
        Bootstrap statistics including mean, std, percentiles for each mode

    Examples
    --------
    >>> import diive as dv
    >>> df = dv.load_exampledata_parquet_lae()
    >>> detector = dv.UstarMovingPointDetection(df)
    >>> thresholds = detector.detect()
    >>> print(detector.summary())
    >>> stats = detector.bootstrap()

    Notes
    -----
    The algorithm requires:
    - At least 3000 total records for valid detection
    - At least 160 records per season
    - At least 100 records per temperature class

    Nighttime filtering is critical: SW_IN < 10 W/m² selects purely dark conditions
    where respiration is the only flux component, free from photosynthetic variability.

    See Also
    --------
    FlagMultipleConstantUstarThresholds : Apply detected USTAR thresholds to filter data.
    FluxProcessingChain : Complete multi-level flux processing workflow.

    Example
    -------
    See `examples/pkgs/flux/lowres/flux_ustar_mp_detection.py` for complete examples
    of USTAR threshold detection using the moving point method with bootstrap uncertainty estimation.
    """

    # Constants from ONEFlux (types.h) - DO NOT MODIFY without consulting Papale et al. (2006)
    NIGHT_THRESHOLD = 10.0  # W/m² - threshold for identifying nighttime (SWIN_FOR_NIGHT in C code)
    MIN_SAMPLES_PERIOD = 3000  # Minimum total records required (MIN_VALUE_PERIOD)
    MIN_SAMPLES_SEASON = 160  # Minimum records per season (MIN_VALUE_SEASON)
    MIN_SAMPLES_TA_CLASS = 100  # Minimum records per temperature class (TA_CLASS_MIN_SAMPLE)
    CORRELATION_CHECK = 0.5  # Maximum |correlation(TA, USTAR)| allowed for valid TA class
    FIRST_USTAR_MEAN_CHECK = 0.2  # Maximum first USTAR class mean in m/s (validation threshold)
    PERCENTILE_CHECK = 90  # Percentile of USTAR distribution for back mode start
    THRESHOLD_NOT_FOUND = 10.0  # Marker value indicating threshold could not be detected (USTAR_THRESHOLD_NOT_FOUND)

    def __init__(
        self,
        df: pd.DataFrame,
        nee_col: Optional[str] = None,
        ta_col: Optional[str] = None,
        ustar_col: Optional[str] = None,
        swin_col: Optional[str] = None,
        ta_classes_count: int = 7,
        ustar_classes_count: int = 20,
        season_groups: Optional[List[List[int]]] = None,
        bootstrapping_times: int = 100,
        verbose: int = 0,
    ):
        # Validate input
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")

        # Auto-detect columns if not specified
        if nee_col is None:
            candidates = [col for col in df.columns if 'NEE' in col and 'QCF' in col]
            if not candidates:
                candidates = [col for col in df.columns if 'NEE' in col]
            if candidates:
                nee_col = candidates[0]
            else:
                nee_col = self._find_column(df, ['NEE', 'NEE_CUT_REF', 'NEE_L3'])

        if ta_col is None:
            candidates = [col for col in df.columns if 'TA_' in col]
            if candidates:
                ta_col = candidates[0]
            else:
                ta_col = self._find_column(df, ['TA', 'TSYS', 'TAIR'])

        if ustar_col is None:
            candidates = [col for col in df.columns if 'USTAR' in col]
            if candidates:
                ustar_col = candidates[0]
            else:
                ustar_col = self._find_column(df, ['U_STAR'])

        if swin_col is None:
            candidates = [col for col in df.columns if 'SW_IN' in col]
            if candidates:
                swin_col = candidates[0]
            else:
                swin_col = self._find_column(df, ['SWIN', 'RAD'])

        if nee_col is None:
            raise ValueError("Could not find NEE column. Specify with nee_col parameter.")
        if ta_col is None:
            raise ValueError("Could not find TA (temperature) column. Specify with ta_col parameter.")
        if ustar_col is None:
            raise ValueError("Could not find USTAR column. Specify with ustar_col parameter.")
        if swin_col is None:
            raise ValueError("Could not find SW_IN (shortwave radiation) column. Specify with swin_col parameter.")

        self.df = df.copy()
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col

        self.ta_classes_count = ta_classes_count
        self.ustar_classes_count = ustar_classes_count
        self.bootstrapping_times = bootstrapping_times
        self.verbose = verbose

        # Default season groups: 4 seasons
        if season_groups is None:
            season_groups = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        self.season_groups = season_groups
        self.seasons_count = len(season_groups)

        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'TIMESTAMP' in self.df.columns:
                self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])
                self.df.set_index('TIMESTAMP', inplace=True)
            else:
                raise ValueError("DataFrame must have datetime index or TIMESTAMP column")

        # Extract month
        self.df['month'] = self.df.index.month

        # Results storage
        self.results_ = {}
        self.bootstrap_results_ = {}

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def detect(self) -> pd.DataFrame:
        """
        Main entry point: Detect USTAR thresholds for all seasons (Papale et al., 2006).

        This method orchestrates the complete USTAR threshold detection workflow:
        1. Validates data sufficiency
        2. Filters to valid, nighttime records only
        3. Processes each season independently
        4. Aggregates seasonal thresholds across temperature classes (median)
        5. Calculates annual threshold as maximum of seasonal values (conservative approach)

        Returns
        -------
        pd.DataFrame
            Seasonal thresholds with columns: 'forward_mode', 'back_mode'
            Rows: Season 1, 2, 3, 4 (winter, spring, summer, fall)
            Values: USTAR threshold in m/s, or NaN if detection failed

        Notes
        -----
        Annual thresholds (maximum across seasons) stored in annual_thresholds_ attribute.
        Use get_annual_thresholds() to retrieve the annual values for data filtering.
        """
        # STEP 1: Data validation
        # ============================================================================
        if self.verbose >= 1:
            print(f"Detecting USTAR thresholds (Papale et al., 2006)...")
            print(f"  Data: {len(self.df)} records, {self.seasons_count} seasons, "
                  f"{self.ta_classes_count} TA classes, {self.ustar_classes_count} USTAR classes")

        # Check minimum data volume (ONEFlux requirement: MIN_VALUE_PERIOD = 3000 records)
        if len(self.df) < self.MIN_SAMPLES_PERIOD:
            raise ValueError(
                f"Insufficient data: {len(self.df)} records, need at least {self.MIN_SAMPLES_PERIOD}"
            )

        # STEP 2: Remove records with missing values
        # ============================================================================
        # Only process records where all 4 required variables are valid (non-NaN)
        # This corresponds to the ALL_VALID flag check in ONEFlux C code
        valid_mask = (
            self.df[self.nee_col].notna() &  # NEE (net ecosystem exchange)
            self.df[self.ta_col].notna() &   # TA (air temperature)
            self.df[self.ustar_col].notna() &  # USTAR (friction velocity)
            self.df[self.swin_col].notna()   # SW_IN (shortwave radiation)
        )
        df_valid = self.df[valid_mask].copy()

        if len(df_valid) < self.MIN_SAMPLES_PERIOD:
            raise ValueError(
                f"Insufficient valid data: {len(df_valid)} records, "
                f"need at least {self.MIN_SAMPLES_PERIOD}"
            )

        # STEP 3: Identify nighttime records (SW_IN < 10 W/m²)
        # ============================================================================
        # CRITICAL: Algorithm ONLY uses nighttime data because:
        #   - No photosynthesis at night → pure respiration signal
        #   - Respiration is stable, independent of light-driven physiological responses
        #   - USTAR effect on measured NEE is consistent and measurable
        #   - Daytime photosynthesis confounds the analysis (highly variable with light/clouds/stomata)
        # This is the key difference from qualitycontrol: USTAR thresholds determined from night data
        # are then APPLIED to quality-control daytime measurements
        # Matches ONEFlux: "if ( !IS_INVALID_VALUE(ut->rows_full_details[i].night) )"
        df_valid['is_night'] = df_valid[self.swin_col] < self.NIGHT_THRESHOLD

        # STEP 4: Process each season independently
        # ============================================================================
        # Papale et al. (2006) recommends seasonal thresholds because USTAR requirements
        # vary with atmospheric stability, which changes seasonally
        thresholds_list = []
        for season_idx, months in enumerate(self.season_groups):
            if self.verbose >= 1:
                print(f"  Season {season_idx + 1}: months {months}")

            # Filter to season AND nighttime (respiration only, no photosynthesis noise)
            season_mask = (df_valid['month'].isin(months)) & (df_valid['is_night'])
            df_season = df_valid[season_mask].copy()

            # Check minimum season data (ONEFlux requirement: MIN_VALUE_SEASON = 160 records)
            if len(df_season) < self.MIN_SAMPLES_SEASON:
                if self.verbose >= 2:
                    print(f"    Insufficient data: {len(df_season)} samples (need {self.MIN_SAMPLES_SEASON})")
                thresholds_list.append([np.nan, np.nan])
                continue

            # Detect threshold for this season using temperature stratification
            # Returns both forward mode (ascending USTAR) and back mode (descending USTAR)
            forward_th, back_th = self._detect_season(df_season)
            thresholds_list.append([forward_th, back_th])

            if self.verbose >= 2:
                print(f"    Forward mode: {forward_th:.4f} m/s")
                print(f"    Back mode:    {back_th:.4f} m/s")

        # STEP 5: Store seasonal results
        # ============================================================================
        self.results_ = pd.DataFrame(
            thresholds_list,
            columns=['forward_mode', 'back_mode'],
            index=[f'Season {i+1}' for i in range(len(thresholds_list))]
        )

        # STEP 6: Calculate annual thresholds (ONEFlux conservative approach)
        # ============================================================================
        # Annual threshold = MAXIMUM seasonal threshold (conservative filtering)
        # Matches ONEFlux: "The whole data set is filtered according to the highest threshold found"
        # This ensures we use the most restrictive (highest) USTAR threshold across all seasons
        forward_thresholds = [t[0] for t in thresholds_list if not np.isnan(t[0]) and t[0] != self.THRESHOLD_NOT_FOUND]
        back_thresholds = [t[1] for t in thresholds_list if not np.isnan(t[1]) and t[1] != self.THRESHOLD_NOT_FOUND]

        self.annual_thresholds_ = {
            'forward_mode': np.max(forward_thresholds) if forward_thresholds else self.THRESHOLD_NOT_FOUND,
            'back_mode': np.max(back_thresholds) if back_thresholds else self.THRESHOLD_NOT_FOUND
        }

        if self.verbose >= 1:
            print(f"\nAnnual thresholds (conservative approach):")
            print(f"  Forward Mode: {self.annual_thresholds_['forward_mode']:.4f} m/s" if self.annual_thresholds_['forward_mode'] != self.THRESHOLD_NOT_FOUND else "  Forward Mode: Not found")
            print(f"  Back Mode:    {self.annual_thresholds_['back_mode']:.4f} m/s" if self.annual_thresholds_['back_mode'] != self.THRESHOLD_NOT_FOUND else "  Back Mode: Not found")

        return self.results_

    def _detect_season(self, df_season: pd.DataFrame) -> Tuple[float, float]:
        """
        Detect USTAR threshold for a single season using temperature stratification.

        Temperature stratification is critical because USTAR requirements depend on atmospheric
        stability, which varies strongly with temperature. Stable conditions (cold nights) require
        higher USTAR to mix CO2 to the sensor height, while unstable conditions (warm nights) have
        lower USTAR requirements.

        Algorithm flow (matches ONEFlux ustar.c):
        1. Sort data by temperature (TA)
        2. Create 7 equal-sized temperature classes using quantile method
        3. For each temperature class:
           a. Validate temperature-USTAR independence (correlation < 0.5)
           b. Sort data by USTAR within the temperature class
           c. Create 20 equal-sized USTAR quantile classes
           d. Compute mean NEE (respiration) per USTAR class
           e. Validate first USTAR class (mean < 0.2 m/s)
           f. Detect forward mode threshold (ascending USTAR search)
           g. Detect back mode threshold (descending USTAR search)
        4. Aggregate thresholds across temperature classes using MEDIAN
           (robust to outlier TA classes, matches ONEFlux approach)
        5. Return (forward_threshold, back_threshold) for the season

        Parameters
        ----------
        df_season : pd.DataFrame
            Nighttime data filtered to one season

        Returns
        -------
        Tuple[float, float]
            (forward_mode_threshold, back_mode_threshold) in m/s
            Returns (10.0, 10.0) if threshold cannot be detected (THRESHOLD_NOT_FOUND marker)
        """
        # Store thresholds detected in each temperature class
        # Later aggregated using median (robust aggregation method, matches ONEFlux)
        thresholds_forward = []
        thresholds_back = []

        # STEP 1: Sort data by temperature for class creation
        # ============================================================================
        # Create equal-sized quantile bins across the temperature range
        # This ensures each TA class has similar sample sizes
        df_sorted = df_season.sort_values(self.ta_col).reset_index(drop=True)

        # STEP 2: Create 7 equal-sized temperature classes
        # ============================================================================
        # Each class gets ~1/7 of the season's data, ensuring statistical independence
        # between classes and sufficient samples for stable respiration means
        n_per_ta = len(df_sorted) // self.ta_classes_count
        if n_per_ta < self.MIN_SAMPLES_TA_CLASS:
            # Insufficient data to create valid temperature classes
            return self.THRESHOLD_NOT_FOUND, self.THRESHOLD_NOT_FOUND

        # STEP 3: Process each temperature class independently
        # ============================================================================
        # For each temperature range, detect its own USTAR threshold
        # Thresholds will later be median-aggregated across all TA classes
        for ta_class_idx in range(self.ta_classes_count):
            # Slice data for this temperature class (quantile-based)
            start_idx = ta_class_idx * n_per_ta
            if ta_class_idx == self.ta_classes_count - 1:
                end_idx = len(df_sorted)  # Include remainder in last class
            else:
                end_idx = (ta_class_idx + 1) * n_per_ta

            df_ta_class = df_sorted.iloc[start_idx:end_idx]

            # Skip if too few samples for this TA class (minimum 100 records per ONEFlux)
            if len(df_ta_class) < self.MIN_SAMPLES_TA_CLASS:
                continue

            # VALIDATION STEP 1: Temperature-USTAR correlation check
            # ============================================================================
            # Correlation between TA and USTAR should be low (< 0.5) to ensure:
            #   - Temperature class stratification is meaningful (independent from USTAR)
            #   - USTAR variation within the TA class is not a proxy for temperature
            #   - The USTAR effect on respiration can be isolated
            # If |correlation| > 0.5, the TA and USTAR are confounded → invalid TA class
            corr = df_ta_class[[self.ta_col, self.ustar_col]].corr().iloc[0, 1]
            if np.isnan(corr) or abs(corr) > self.CORRELATION_CHECK:
                # This TA class is invalid (too much TA-USTAR correlation), skip it
                continue

            # Detect USTAR threshold for this temperature class
            # Calls _detect_ta_class which stratifies by USTAR and finds stability point
            forward_th, back_th = self._detect_ta_class(df_ta_class)

            # Collect valid thresholds for later aggregation
            if not np.isnan(forward_th) and forward_th != self.THRESHOLD_NOT_FOUND:
                thresholds_forward.append(forward_th)
            if not np.isnan(back_th) and back_th != self.THRESHOLD_NOT_FOUND:
                thresholds_back.append(back_th)

        # STEP 4: Aggregate thresholds across temperature classes
        # ============================================================================
        # Use MEDIAN aggregation (robust to outliers, matches ONEFlux approach)
        # If no valid TA classes found, return THRESHOLD_NOT_FOUND (10.0) marker
        forward_result = np.median(thresholds_forward) if thresholds_forward else self.THRESHOLD_NOT_FOUND
        back_result = np.median(thresholds_back) if thresholds_back else self.THRESHOLD_NOT_FOUND

        return forward_result, back_result

    def _detect_ta_class(self, df_ta_class: pd.DataFrame) -> Tuple[float, float]:
        """
        Detect USTAR threshold for one temperature class via USTAR stratification.

        Within a temperature class, this method stratifies by USTAR to find the threshold
        where respiration stabilizes. The key insight is that as USTAR increases:
          1. Low USTAR: respiration is underestimated (mixing limitations)
          2. Mid USTAR: respiration increases as more flux is captured (partial recovery)
          3. High USTAR: respiration stabilizes (full recovery, no further change)

        The threshold is the USTAR value where further increases don't change respiration.

        Algorithm (matches ONEFlux ustar.c):
        1. Create 20 equal-sized USTAR quantile classes within this TA class
        2. Calculate mean respiration (NEE) per USTAR class
        3. Validate: first USTAR class must be low (< 0.2 m/s) to detect the threshold
        4. Forward mode: ascending search for stabilization point
        5. Back mode: descending search for additional validation

        Parameters
        ----------
        df_ta_class : pd.DataFrame
            Nighttime data filtered to one temperature class

        Returns
        -------
        Tuple[float, float]
            (forward_mode_threshold, back_mode_threshold) in m/s
        """
        # STEP 1: Sort by USTAR for quantile-based class creation
        # ============================================================================
        # Create equal-sized bins across the USTAR range within this temperature class
        df_sorted = df_ta_class.sort_values(self.ustar_col).reset_index(drop=True)

        # Create 20 USTAR classes (equal-sized quantile bins)
        n_per_ustar = len(df_sorted) // self.ustar_classes_count
        if n_per_ustar < 1:
            # Not enough data to create valid USTAR classes
            return self.THRESHOLD_NOT_FOUND, self.THRESHOLD_NOT_FOUND

        # STEP 2: Compute statistics for each USTAR class
        # ============================================================================
        # Calculate mean USTAR and mean respiration (NEE) per class
        # These are the values used in the moving point detection algorithm
        ustar_means = []  # Mean friction velocity for each USTAR class
        nee_means = []    # Mean respiration for each USTAR class

        for ustar_class_idx in range(self.ustar_classes_count):
            # Slice data for this USTAR class (quantile-based)
            start_idx = ustar_class_idx * n_per_ustar
            if ustar_class_idx == self.ustar_classes_count - 1:
                end_idx = len(df_sorted)  # Include remainder in last class
            else:
                end_idx = (ustar_class_idx + 1) * n_per_ustar

            if start_idx >= end_idx:
                break

            # Calculate class statistics
            df_ustar_class = df_sorted.iloc[start_idx:end_idx]
            ustar_means.append(df_ustar_class[self.ustar_col].mean())
            nee_means.append(df_ustar_class[self.nee_col].mean())

        # Need at least 2 USTAR classes to detect a threshold
        if len(ustar_means) < 2:
            return self.THRESHOLD_NOT_FOUND, self.THRESHOLD_NOT_FOUND

        ustar_means = np.array(ustar_means)
        nee_means = np.array(nee_means)

        # VALIDATION STEP 2: First USTAR class validity check
        # ============================================================================
        # The lowest USTAR class mean must be < 0.2 m/s to ensure:
        #   - We capture the low-turbulence regime where respiration is underestimated
        #   - The threshold can be properly detected (need data from below the threshold)
        # If the lowest class is already >= 0.2 m/s, the data doesn't span the problematic range
        if ustar_means[0] > self.FIRST_USTAR_MEAN_CHECK:
            # Data doesn't include sufficiently low USTAR values
            return self.THRESHOLD_NOT_FOUND, self.THRESHOLD_NOT_FOUND

        # STEP 3: Detect forward mode threshold
        # ============================================================================
        # Search ascending through USTAR classes for where respiration stabilizes
        # Window size = 10 classes (ONEFlux default WINDOWS_SIZE_FOR_FORWARD_MODE)
        # n = 1 (check next 1 class for stability)
        forward_th = self._forward_mode(ustar_means, nee_means, n=1, window_size=10)

        # STEP 4: Detect back mode threshold
        # ============================================================================
        # Search descending from high USTAR for additional stability validation
        # Window size = 6 classes (ONEFlux default WINDOWS_SIZE_FOR_BACK_MODE)
        # n = 1 (check previous 1 class for stability)
        back_th = self._back_mode(ustar_means, nee_means, n=1, window_size=6)

        return forward_th, back_th

    def _forward_mode(
        self,
        ustar_means: np.ndarray,
        nee_means: np.ndarray,
        n: int = 1,
        threshold_check: float = 1.0,
        window_size: int = 10
    ) -> float:
        """
        Forward mode: Find first USTAR where next n classes satisfy stabilization.

        For each USTAR class i, compute forward moving window mean of next n classes,
        then check if current class NEE >= window_mean * threshold_check.
        Window size defaults to 10 (ONEFlux default WINDOWS_SIZE_FOR_FORWARD_MODE).
        """
        for i in range(len(ustar_means) - n):
            if np.isnan(nee_means[i]):
                continue

            # Calculate window means for next n positions
            window_means = []
            skip = False
            for y in range(n):
                start_idx = i + 1 + y
                end_idx = min(start_idx + window_size, len(nee_means))
                window = nee_means[start_idx:end_idx]

                if len(window) == 0 or np.any(np.isnan(window)):
                    skip = True
                    break

                window_means.append(np.nanmean(window))

            if skip:
                continue

            # Check if current class meets all n threshold conditions
            if np.all(nee_means[i:i+n] >= np.array(window_means) * threshold_check):
                return ustar_means[i]

        return self.THRESHOLD_NOT_FOUND

    def _back_mode(
        self,
        ustar_means: np.ndarray,
        nee_means: np.ndarray,
        n: int = 1,
        threshold_check: float = 1.0,
        percentile: float = 90.0,
        window_size: int = 6
    ) -> float:
        """
        Back mode: Start from high USTAR, work backwards finding stabilization point.

        For each position i starting from percentile, compute window mean from position i
        forward for window_size steps, then check if previous n values satisfy condition.
        Window size defaults to 6 (ONEFlux default WINDOWS_SIZE_FOR_BACK_MODE).
        """
        # Start from percentile
        start_idx = int(len(ustar_means) * percentile / 100.0)
        if start_idx >= len(ustar_means):
            start_idx = len(ustar_means) - 1

        # Work backwards
        for i in range(start_idx, n, -1):
            # Calculate window mean from i forward for window_size
            end_idx = min(i + window_size, len(nee_means))
            window = nee_means[i:end_idx]

            if len(window) == 0 or np.any(np.isnan(window)):
                continue

            window_mean = np.nanmean(window)
            threshold = window_mean * threshold_check

            # Check previous n values
            prev_nee = nee_means[max(0, i - n):i]
            if len(prev_nee) == 0 or np.any(np.isnan(prev_nee)):
                continue

            # All previous values must be <= threshold
            if np.all(prev_nee <= threshold):
                return ustar_means[i - 1]

        # Fallback: use percentile value
        if start_idx < len(ustar_means):
            return ustar_means[start_idx]

        return self.THRESHOLD_NOT_FOUND

    def bootstrap(self, n_iter: Optional[int] = None) -> pd.DataFrame:
        """
        Run bootstrap resampling to estimate USTAR threshold uncertainty.

        Parameters
        ----------
        n_iter : int, optional
            Number of bootstrap iterations (uses self.bootstrapping_times if None)

        Returns
        -------
        pd.DataFrame
            Bootstrap statistics (mean, std, percentiles)
        """
        if n_iter is None:
            n_iter = self.bootstrapping_times

        if self.verbose >= 1:
            print(f"Running {n_iter} bootstrap iterations...")

        boot_results = {i: [] for i in range(self.seasons_count)}

        for boot_idx in range(n_iter):
            if self.verbose >= 2 and boot_idx % 10 == 0:
                print(f"  Iteration {boot_idx + 1}/{n_iter}")

            # Resample with replacement
            df_boot = self.df.sample(n=len(self.df), replace=True)

            # Detect on bootstrap sample
            try:
                detector_boot = UstarMovingPointDetection(
                    df_boot,
                    nee_col=self.nee_col,
                    ta_col=self.ta_col,
                    ustar_col=self.ustar_col,
                    swin_col=self.swin_col,
                    ta_classes_count=self.ta_classes_count,
                    ustar_classes_count=self.ustar_classes_count,
                    season_groups=self.season_groups,
                    verbose=0,
                )
                results = detector_boot.detect()

                for season_idx in range(self.seasons_count):
                    if season_idx < len(results):
                        boot_results[season_idx].append(results.iloc[season_idx].values)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"    Bootstrap {boot_idx} failed: {str(e)}")
                continue

        # Compute statistics
        return self._compute_bootstrap_statistics(boot_results)

    def _compute_bootstrap_statistics(self, boot_results: Dict) -> pd.DataFrame:
        """Compute statistics from bootstrap results."""
        stats_list = []

        for season_idx in range(self.seasons_count):
            boot_vals = np.array(boot_results[season_idx])

            if len(boot_vals) == 0:
                stats_list.append({
                    'forward_mean': np.nan, 'forward_std': np.nan,
                    'forward_p05': np.nan, 'forward_p95': np.nan,
                    'back_mean': np.nan, 'back_std': np.nan,
                    'back_p05': np.nan, 'back_p95': np.nan,
                })
                continue

            stats_list.append({
                'forward_mean': np.nanmean(boot_vals[:, 0]),
                'forward_std': np.nanstd(boot_vals[:, 0]),
                'forward_p05': np.nanpercentile(boot_vals[:, 0], 5),
                'forward_p95': np.nanpercentile(boot_vals[:, 0], 95),
                'back_mean': np.nanmean(boot_vals[:, 1]),
                'back_std': np.nanstd(boot_vals[:, 1]),
                'back_p05': np.nanpercentile(boot_vals[:, 1], 5),
                'back_p95': np.nanpercentile(boot_vals[:, 1], 95),
            })

        return pd.DataFrame(
            stats_list,
            index=[f'Season {i+1}' for i in range(self.seasons_count)]
        )

    def summary(self) -> str:
        """Return formatted summary of detection results."""
        if self.results_.empty:
            return "No detection results. Run detect() first."

        lines = ["USTAR Threshold Detection Results (Papale et al., 2006)"]
        lines.append("=" * 75)
        lines.append(f"{'Season':<15} {'Forward Mode (m/s)':<25} {'Back Mode (m/s)':<25}")
        lines.append("-" * 75)

        for idx, row in self.results_.iterrows():
            forward = f"{row['forward_mode']:.4f}" if not np.isnan(row['forward_mode']) and row['forward_mode'] != self.THRESHOLD_NOT_FOUND else "Not found"
            back = f"{row['back_mode']:.4f}" if not np.isnan(row['back_mode']) and row['back_mode'] != self.THRESHOLD_NOT_FOUND else "Not found"
            lines.append(f"{idx:<15} {forward:<25} {back:<25}")

        return "\n".join(lines)

    def get_annual_thresholds(self) -> Dict[str, float]:
        """
        Get annual USTAR thresholds (maximum across all seasons).

        Returns
        -------
        Dict[str, float]
            Annual thresholds with keys: 'forward_mode', 'back_mode'
            Values: USTAR threshold in m/s (10.0 if not found)

        Notes
        -----
        Annual threshold uses conservative approach: maximum of seasonal thresholds.
        Matches ONEFlux: "The whole data set is filtered according to the highest
        threshold found (conservative approach)."

        Example
        -------
        >>> detector = UstarMovingPointDetection(df)
        >>> detector.detect()
        >>> annual = detector.get_annual_thresholds()
        >>> print(f"Forward mode: {annual['forward_mode']:.4f} m/s")
        """
        if not hasattr(self, 'annual_thresholds_'):
            raise RuntimeError("Detection not yet performed. Call detect() first.")
        return self.annual_thresholds_.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UstarMovingPointDetection("
            f"n_records={len(self.df)}, "
            f"ta_classes={self.ta_classes_count}, "
            f"ustar_classes={self.ustar_classes_count}, "
            f"seasons={self.seasons_count})"
        )
