"""
USTAR VEKURI THRESHOLD DETECTION: QUANTILE-BASED APPROACH
==========================================================

Detect friction velocity (u*) threshold using quantile-based stratification.

Part of the diive library: https://github.com/holukas/diive
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings

from diive.core.utils.console import info, detail, warn


class UstarVekuriThresholdDetection:
    """
    Detect USTAR threshold using quantile-based stratification approach.

    This class implements a simplified USTAR threshold detection workflow that is
    more computationally efficient than the full ONEFlux method while maintaining
    robust stratification.

    The key insight is stratifying by quantiles rather than equal-sized bins,
    ensuring each class has similar sample sizes regardless of the distribution
    of underlying variables.

    Algorithm steps:
    1. FILTER DATA: Select nighttime records only (SW_IN < 10 W/m² by default)
    2. STRATIFY BY SEASON: Divide year into 4 seasons
    3. STRATIFY BY TEMPERATURE: Within each season, divide into 6 quantile-based classes
    4. STRATIFY BY USTAR: Within each temperature class, divide into 20 quantile-based classes
    5. COMPUTE STATISTICS: Calculate mean respiration per USTAR class
    6. VALIDATE: Check temperature-USTAR independence (correlation < 0.4)
    7. DETECT THRESHOLD: Ascending search for flux stability point
    8. AGGREGATE: Median across temperature classes gives season threshold
    9. BOOTSTRAP: Repeat with resampled data for uncertainty estimation

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with datetime index and required columns: NEE, TA, USTAR, SW_IN
    nee_col : str, optional
        Net ecosystem exchange column name (auto-detected if None)
    ta_col : str, optional
        Air temperature column name (auto-detected if None)
    ustar_col : str, optional
        Friction velocity column name (auto-detected if None)
    swin_col : str, optional
        Shortwave radiation column name (auto-detected if None)
    par_col : str, optional
        Photosynthetically active radiation column (alternative nighttime filter)
    n_temperature_classes : int, default=6
        Number of temperature quantile classes per season
    n_ustar_classes : int, default=20
        Number of USTAR quantile classes per temperature class
    season_groups : List[List[int]], optional
        Custom season grouping by month numbers [1-12]
        Default: [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]
    bootstrapping_times : int, default=100
        Number of bootstrap resampling iterations for uncertainty estimation
    verbose : int, default=0
        Verbosity level: 0=silent, 1=progress, 2=detailed debug output

    Attributes
    ----------
    results_ : pd.DataFrame
        Detected thresholds with column: 'threshold'
        Index contains season labels (Season 1, 2, 3, 4)
    bootstrap_stats_ : pd.DataFrame
        Bootstrap statistics including mean, std, percentiles

    Examples
    --------
    >>> import diive as dv
    >>> df = dv.load_exampledata_parquet_lae()
    >>> # Filter to nighttime
    >>> df_night = df[df['SW_IN'] < 10].copy()
    >>> detector = dv.UstarVekuriThresholdDetection(df_night)
    >>> thresholds = detector.detect()
    >>> print(detector.summary())
    >>> stats = detector.bootstrap()

    Notes
    -----
    The algorithm expects:
    - Pre-filtered or nighttime data (SW_IN < 10 W/m² or PAR < 20)
    - At least 100-200 records per season for valid detection

    See Also
    --------
    UstarMovingPointDetection : Full ONEFlux implementation with forward/back modes.
    FlagMultipleConstantUstarThresholds : Apply detected USTAR thresholds to filter data.
    """

    CORRELATION_CHECK = 0.4
    STABILITY_THRESHOLD = 0.95
    WINDOW_SIZE = 10

    def __init__(
        self,
        df: pd.DataFrame,
        nee_col: Optional[str] = None,
        ta_col: Optional[str] = None,
        ustar_col: Optional[str] = None,
        swin_col: Optional[str] = None,
        par_col: Optional[str] = None,
        n_temperature_classes: int = 6,
        n_ustar_classes: int = 20,
        season_groups: Optional[List[List[int]]] = None,
        bootstrapping_times: int = 100,
        verbose: int = 0,
    ):
        if df is None or df.empty:
            raise ValueError("Input DataFrame cannot be None or empty")

        # Auto-detect columns
        if nee_col is None:
            candidates = [col for col in df.columns if 'NEE' in col and 'QCF' in col]
            if not candidates:
                candidates = [col for col in df.columns if 'NEE' in col]
            if candidates:
                nee_col = candidates[0]
            else:
                nee_col = self._find_column(df, ['NEE', 'flux', 'NEE_CUT_REF', 'NEE_L3'])

        if ta_col is None:
            candidates = [col for col in df.columns if 'TA_' in col]
            if candidates:
                ta_col = candidates[0]
            else:
                ta_col = self._find_column(df, ['TA', 'T', 'TSYS', 'TAIR'])

        if ustar_col is None:
            candidates = [col for col in df.columns if 'USTAR' in col]
            if candidates:
                ustar_col = candidates[0]
            else:
                ustar_col = self._find_column(df, ['U_STAR', 'ustar'])

        if swin_col is None:
            candidates = [col for col in df.columns if 'SW_IN' in col]
            if candidates:
                swin_col = candidates[0]
            else:
                swin_col = self._find_column(df, ['SWIN', 'RAD', 'SW_RAD'])

        if par_col is None:
            candidates = [col for col in df.columns if 'PAR' in col]
            if candidates:
                par_col = candidates[0]

        if nee_col is None:
            raise ValueError("Could not find NEE column. Specify with nee_col parameter.")
        if ta_col is None:
            raise ValueError("Could not find TA (temperature) column. Specify with ta_col parameter.")
        if ustar_col is None:
            raise ValueError("Could not find USTAR column. Specify with ustar_col parameter.")

        self.df = df.copy()
        self.nee_col = nee_col
        self.ta_col = ta_col
        self.ustar_col = ustar_col
        self.swin_col = swin_col
        self.par_col = par_col

        self.n_temperature_classes = n_temperature_classes
        self.n_ustar_classes = n_ustar_classes
        self.bootstrapping_times = bootstrapping_times
        self.verbose = verbose

        if season_groups is None:
            season_groups = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        self.season_groups = season_groups
        self.seasons_count = len(season_groups)

        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'TIMESTAMP' in self.df.columns:
                self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])
                self.df.set_index('TIMESTAMP', inplace=True)
            else:
                raise ValueError("DataFrame must have datetime index or TIMESTAMP column")

        self.df['month'] = self.df.index.month
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
        Main entry point: Detect USTAR thresholds for all seasons.

        Returns
        -------
        pd.DataFrame
            Seasonal thresholds with column: 'threshold'
            Rows: Season 1, 2, 3, 4
            Values: USTAR threshold in m/s, or NaN if detection failed
        """
        if self.verbose >= 1:
            info(f"Detecting USTAR thresholds (Vekuri quantile-based approach) | "
                 f"{len(self.df)} records, {self.seasons_count} seasons, "
                 f"{self.n_temperature_classes} TA classes, {self.n_ustar_classes} USTAR classes")

        # Remove records with missing values
        valid_mask = (
            self.df[self.nee_col].notna() &
            self.df[self.ta_col].notna() &
            self.df[self.ustar_col].notna()
        )

        # Add nighttime filter if SW_IN column exists
        if self.swin_col and self.swin_col in self.df.columns:
            valid_mask = valid_mask & (self.df[self.swin_col] < 10)
        elif self.par_col and self.par_col in self.df.columns:
            valid_mask = valid_mask & (self.df[self.par_col] < 20)

        df_valid = self.df[valid_mask].copy()

        if len(df_valid) < 100:
            raise ValueError(f"Insufficient valid data: {len(df_valid)} records")

        # Detect threshold for each season
        thresholds_list = []
        for season_idx, months in enumerate(self.season_groups):
            if self.verbose >= 1:
                info(f"  Season {season_idx + 1}: months {months}")

            season_mask = df_valid['month'].isin(months)
            df_season = df_valid[season_mask].copy()

            if len(df_season) < 50:
                if self.verbose >= 2:
                    detail(f"    Insufficient data: {len(df_season)} samples")
                thresholds_list.append([np.nan])
                continue

            threshold = self._detect_season(df_season)
            thresholds_list.append([threshold])

            if self.verbose >= 2 and not np.isnan(threshold):
                detail(f"    Threshold: {threshold:.4f} m/s")

        # Store results
        self.results_ = pd.DataFrame(
            thresholds_list,
            columns=['threshold'],
            index=[f'Season {i+1}' for i in range(len(thresholds_list))]
        )

        # Calculate annual threshold (maximum across seasons)
        thresholds = [t[0] for t in thresholds_list if not np.isnan(t[0])]
        self.annual_thresholds_ = {
            'threshold': np.max(thresholds) if thresholds else np.nan
        }

        if self.verbose >= 1:
            annual = self.annual_thresholds_['threshold']
            if not np.isnan(annual):
                info(f"Annual threshold: {annual:.4f} m/s")

        return self.results_

    def _detect_season(self, df_season: pd.DataFrame) -> float:
        """Detect USTAR threshold for a single season."""
        thresholds_forward = []

        try:
            # Create temperature classes using quantiles
            temperature_classes = pd.qcut(
                df_season[self.ta_col], self.n_temperature_classes, duplicates='drop'
            )

            for t_class, temp_class_data in df_season.groupby(temperature_classes, observed=True):
                # Validate temperature-USTAR independence
                corr = temp_class_data[[self.ta_col, self.ustar_col]].corr().iloc[0, 1]
                if np.isnan(corr) or abs(corr) >= self.CORRELATION_CHECK:
                    continue

                # Create USTAR classes using quantiles
                ustar_classes = pd.qcut(
                    temp_class_data[self.ustar_col], self.n_ustar_classes, duplicates='drop'
                )

                # Calculate statistics per USTAR class
                temp_class_data_copy = temp_class_data.copy()
                temp_class_data_copy['ustar_class'] = ustar_classes

                ustar_means = []
                flux_means = []

                for _, group in temp_class_data_copy.groupby('ustar_class', observed=True):
                    ustar_means.append(group[self.ustar_col].mean())
                    flux_means.append(group[self.nee_col].mean())

                if len(ustar_means) < 3:
                    continue

                # Detect threshold using ascending search
                threshold = self._detect_threshold(np.array(ustar_means), np.array(flux_means))
                if not np.isnan(threshold):
                    thresholds_forward.append(threshold)

        except Exception as e:
            if self.verbose >= 2:
                warn(f"Error in season detection: {str(e)}")
            return np.nan

        # Aggregate using median
        return np.median(thresholds_forward) if len(thresholds_forward) >= 1 else np.nan

    def _detect_threshold(self, ustar_means: np.ndarray, flux_means: np.ndarray) -> float:
        """
        Detect threshold using ascending search.

        Finds first USTAR class where flux >= 0.95 * mean(flux[i+1:i+WINDOW_SIZE])
        """
        for i in range(len(ustar_means) - 1):
            if np.isnan(flux_means[i]):
                continue

            # Calculate window mean of next classes
            end_idx = min(i + 1 + self.WINDOW_SIZE, len(flux_means))
            if end_idx <= i + 1:
                continue

            window_mean = np.nanmean(flux_means[i + 1:end_idx])
            if np.isnan(window_mean):
                continue

            # Check stability condition
            if flux_means[i] >= self.STABILITY_THRESHOLD * window_mean:
                return ustar_means[i]

        return np.nan

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
            info(f"Running {n_iter} bootstrap iterations...")

        boot_results = {i: [] for i in range(self.seasons_count)}

        for boot_idx in range(n_iter):
            if self.verbose >= 2 and boot_idx % 10 == 0:
                detail(f"  Iteration {boot_idx + 1}/{n_iter}")

            df_boot = self.df.sample(n=len(self.df), replace=True)

            try:
                detector_boot = UstarVekuriThresholdDetection(
                    df_boot,
                    nee_col=self.nee_col,
                    ta_col=self.ta_col,
                    ustar_col=self.ustar_col,
                    swin_col=self.swin_col,
                    par_col=self.par_col,
                    n_temperature_classes=self.n_temperature_classes,
                    n_ustar_classes=self.n_ustar_classes,
                    season_groups=self.season_groups,
                    verbose=0,
                )
                results = detector_boot.detect()

                for season_idx in range(self.seasons_count):
                    if season_idx < len(results):
                        val = results.iloc[season_idx].values[0]
                        if not np.isnan(val):
                            boot_results[season_idx].append(val)
            except Exception:
                if self.verbose >= 2:
                    warn(f"Bootstrap {boot_idx} failed")
                continue

        return self._compute_bootstrap_statistics(boot_results)

    def _compute_bootstrap_statistics(self, boot_results: Dict) -> pd.DataFrame:
        """Compute statistics from bootstrap results."""
        stats_list = []

        for season_idx in range(self.seasons_count):
            boot_vals = boot_results[season_idx]

            if len(boot_vals) == 0:
                stats_list.append({
                    'mean': np.nan,
                    'std': np.nan,
                    'p05': np.nan,
                    'p50': np.nan,
                    'p95': np.nan,
                })
            else:
                stats_list.append({
                    'mean': np.mean(boot_vals),
                    'std': np.std(boot_vals),
                    'p05': np.percentile(boot_vals, 5),
                    'p50': np.percentile(boot_vals, 50),
                    'p95': np.percentile(boot_vals, 95),
                })

        return pd.DataFrame(
            stats_list,
            index=[f'Season {i+1}' for i in range(self.seasons_count)]
        )

    def summary(self) -> str:
        """Return formatted summary of detection results."""
        if self.results_.empty:
            return "No detection results. Run detect() first."

        lines = ["USTAR Threshold Detection Results (Vekuri Quantile Method)"]
        lines.append("=" * 60)
        lines.append(f"{'Season':<15} {'Threshold (m/s)':<20}")
        lines.append("-" * 60)

        for idx in self.results_.index:
            val = self.results_.loc[idx, 'threshold']
            threshold_str = f"{val:.4f}" if not np.isnan(val) else "Not found"
            lines.append(f"{idx:<15} {threshold_str:<20}")

        return "\n".join(lines)

    def get_annual_thresholds(self) -> Dict[str, float]:
        """
        Get annual USTAR threshold (maximum across all seasons).

        Returns
        -------
        Dict[str, float]
            Annual threshold with key: 'threshold'
        """
        if not hasattr(self, 'annual_thresholds_'):
            raise RuntimeError("Detection not yet performed. Call detect() first.")
        return self.annual_thresholds_.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UstarVekuriThresholdDetection("
            f"n_records={len(self.df)}, "
            f"ta_classes={self.n_temperature_classes}, "
            f"ustar_classes={self.n_ustar_classes}, "
            f"seasons={self.seasons_count})"
        )
