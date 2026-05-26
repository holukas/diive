"""
FLUX PROCESSING CHAIN: MULTI-LEVEL EDDY COVARIANCE PROCESSING
============================================================

Swiss FluxNet-compliant post-processing: quality control, storage correction, outlier
detection, USTAR filtering, and gap-filling.

Two ways to use the chain:

1. **Composable functions** (recommended for custom pipelines)::

       from diive.pkgs.flux.fluxprocessingchain import (
           init_flux_data, run_level2, run_level31,
           make_level32_detector, run_level32,
           run_level33_constant_ustar,
           run_level41_mds, run_level41_rf, run_level41_xgb,
       )

       data = init_flux_data(df, fluxcol='FC', site_lat=46.6, site_lon=9.8, utc_offset=1)
       data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None}, ...)
       data = run_level31(data, gapfill_storage_term=True)
       # stop here for L2+L3.1 only
       final_df = data.fpc_df

2. **FluxProcessingChain class** (convenience wrapper for the full chain)::

       fpc = FluxProcessingChain(df=df, fluxcol='FC', site_lat=46.6, ...)
       fpc.level2_quality_flag_expansion(**LEVEL2_SETTINGS)
       fpc.level31_storage_correction()
       ...

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.core.utils.console import console as _console, info
from diive.core.plotting.cumulative import Cumulative, CumulativeYear
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData, LevelResults
from diive.pkgs.flux.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.flux.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.flux.fluxprocessingchain.levels import (
    init_flux_data,
    make_level32_detector,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level33_ustar_detection,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)
from diive.pkgs.flux.lowres.hqflux import analyze_highest_quality_flux
from diive.pkgs.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds
from diive.pkgs.preprocessing.outlier_detection import StepwiseOutlierDetection
from diive.pkgs.preprocessing.qaqc import FlagQCF


def _build_feature_engineer(
        sanitize_timestamp: bool, vectorize_timestamps: bool, add_continuous_record_number: bool,
        features_lag, features_lag_stepsize, features_lag_exclude_cols,
        features_rolling, features_rolling_exclude_cols, features_rolling_stats,
        features_diff, features_diff_exclude_cols,
        features_ema, features_ema_exclude_cols,
        features_poly_degree, features_poly_exclude_cols,
        features_stl, features_stl_method, features_stl_seasonal_period,
        features_stl_exclude_cols, features_stl_components,
) -> FeatureEngineer:
    """Construct a FeatureEngineer from the legacy `features_*` keyword set."""
    return FeatureEngineer(
        target_col='_temp_target_placeholder_',
        features_lag=features_lag,
        features_lag_stepsize=features_lag_stepsize,
        features_lag_exclude_cols=features_lag_exclude_cols,
        features_rolling=features_rolling,
        features_rolling_exclude_cols=features_rolling_exclude_cols,
        features_rolling_stats=features_rolling_stats,
        features_diff=features_diff,
        features_diff_exclude_cols=features_diff_exclude_cols,
        features_ema=features_ema,
        features_ema_exclude_cols=features_ema_exclude_cols,
        features_poly_degree=features_poly_degree,
        features_poly_exclude_cols=features_poly_exclude_cols,
        features_stl=features_stl,
        features_stl_method=features_stl_method,
        features_stl_seasonal_period=features_stl_seasonal_period,
        features_stl_exclude_cols=features_stl_exclude_cols,
        features_stl_components=features_stl_components,
        vectorize_timestamps=vectorize_timestamps,
        add_continuous_record_number=add_continuous_record_number,
        sanitize_timestamp=sanitize_timestamp,
    )


class FluxProcessingChain:
    """
    Convenience orchestrator for the Swiss FluxNet multi-level flux processing chain.

    Wraps the composable level callables so users who want to run all five
    levels in sequence can do so with method-call style.  The underlying
    container is accessible as ``fpc.data`` and typed results live in
    ``fpc.data.levels`` (see ``LevelResults``).

    For custom pipelines (partial runs, custom L3.2, branching at L4.1),
    use the standalone callables in
    ``diive.pkgs.flux.fluxprocessingchain.levels`` directly.

    See Also:
        init_flux_data, run_level2, run_level31, make_level32_detector,
        run_level32, run_level33_constant_ustar, run_level41_mds,
        run_level41_rf, run_level41_xgb
    """

    def __init__(
            self,
            df: DataFrame,
            fluxcol: str,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            nighttime_threshold: float = 20,
            daytime_accept_qcf_below: int = 1,
            nighttime_accept_qcf_below: int = 1,
    ):
        """
        Initialise the processing chain.

        Builds the initial FluxLevelData container (adds potential radiation
        and day/night flags, assembles frozen site metadata).

        Args:
            df: Input DataFrame containing flux and meteorological data.
            fluxcol: Raw flux column name (e.g. 'FC', 'LE', 'H').
            site_lat: Site latitude (decimal degrees).
            site_lon: Site longitude (decimal degrees).
            utc_offset: UTC offset (hours).
            nighttime_threshold: Potential radiation threshold (W m-2)
                below which records are treated as nighttime. Defaults to 20.
            daytime_accept_qcf_below: Daytime QCF acceptance threshold
                (0, 1, or 2). Defaults to 1.
            nighttime_accept_qcf_below: Nighttime QCF acceptance
                threshold (0, 1, or 2). Defaults to 1.
        """
        self._data: FluxLevelData = init_flux_data(
            df=df,
            fluxcol=fluxcol,
            site_lat=site_lat,
            site_lon=site_lon,
            utc_offset=utc_offset,
            nighttime_threshold=nighttime_threshold,
            daytime_accept_qcf_below=daytime_accept_qcf_below,
            nighttime_accept_qcf_below=nighttime_accept_qcf_below,
        )
        # Pending L3.2 detector — configured across multiple method calls
        self._pending_level32: StepwiseOutlierDetection | None = None

    # ------------------------------------------------------------------
    # Primary access points for the composable container
    # ------------------------------------------------------------------

    @property
    def data(self) -> FluxLevelData:
        """The underlying FluxLevelData container."""
        return self._data

    @property
    def levels(self) -> LevelResults:
        """Typed per-level results (shortcut for ``fpc.data.levels``)."""
        return self._data.levels

    # ------------------------------------------------------------------
    # Convenience shortcuts (delegate into data / data.levels)
    # ------------------------------------------------------------------

    @property
    def fpc_df(self) -> DataFrame:
        """Working dataframe holding flux + all flag/QCF columns produced so far."""
        return self._data.fpc_df

    @property
    def df(self) -> DataFrame:
        """Full input dataframe (with potential radiation + day/night flags added)."""
        return self._data.full_df

    @property
    def filteredseries(self) -> Series:
        """QCF-filtered flux series from the most recently completed level."""
        if self._data.filteredseries is None:
            raise RuntimeError('No filtered series yet; run level2_quality_flag_expansion() first.')
        return self._data.filteredseries

    @property
    def filteredseries_hq(self) -> Series:
        """Highest-quality (QCF=0) flux series. Set after L2, updated by L3.1."""
        s = self._data.levels.filteredseries_hq
        if s is None:
            raise RuntimeError('No high-quality filtered series yet; run level2_quality_flag_expansion() first.')
        return s

    @property
    def filteredseries_level2_qcf(self) -> Series:
        return self._data.levels.filteredseries_level2_qcf

    @property
    def filteredseries_level31_qcf(self) -> Series:
        return self._data.levels.filteredseries_level31_qcf

    @property
    def filteredseries_level32_qcf(self) -> Series:
        return self._data.levels.filteredseries_level32_qcf

    @property
    def filteredseries_level33_qcf(self) -> dict:
        return self._data.levels.filteredseries_level33_qcf

    @property
    def level2(self) -> FluxQualityFlagsEddyPro:
        inst = self._data.levels.level2
        if inst is None:
            raise RuntimeError('Run level2_quality_flag_expansion() first.')
        return inst

    @property
    def level2_qcf(self) -> FlagQCF:
        return self._data.levels.level2_qcf

    @property
    def level31(self) -> FluxStorageCorrectionSinglePointEddyPro:
        inst = self._data.levels.level31
        if inst is None:
            raise RuntimeError('Run level31_storage_correction() first.')
        return inst

    @property
    def level32(self) -> StepwiseOutlierDetection:
        inst = self._data.levels.level32
        if inst is None:
            raise RuntimeError('Run level32_stepwise_outlier_detection() + finalize_level32() first.')
        return inst

    @property
    def level32_qcf(self) -> FlagQCF:
        return self._data.levels.level32_qcf

    @property
    def level33(self) -> FlagMultipleConstantUstarThresholds:
        inst = self._data.levels.level33
        if inst is None:
            raise RuntimeError('Run level33_constant_ustar() first.')
        return inst

    @property
    def level33_qcf(self) -> dict:
        return self._data.levels.level33_qcf

    @property
    def ustar_detection(self):
        """``UstarBootstrapThresholds`` instance from ``level33_ustar_detection()``, or None."""
        return self._data.levels.ustar_detection

    @property
    def level41(self) -> dict:
        """Nested dict ``{method: {ustar_scenario: gap_filling_instance}}`` for L4.1.

        Built on the fly from the typed ``levels.level41_*`` fields.
        """
        return self._data.levels.level41_methods()

    @property
    def levelidstr(self) -> list:
        return self._data.level_ids

    # Meta shortcuts
    @property
    def fluxcol(self) -> str: return self._data.meta.fluxcol
    @property
    def fluxbasevar(self) -> str: return self._data.meta.fluxbasevar
    @property
    def outname(self) -> str: return self._data.meta.outname
    @property
    def site_lat(self) -> float: return self._data.meta.site_lat
    @property
    def site_lon(self) -> float: return self._data.meta.site_lon
    @property
    def utc_offset(self) -> int: return self._data.meta.utc_offset
    @property
    def swinpot_col(self) -> str: return self._data.meta.swinpot_col
    @property
    def ustarcol(self) -> str: return self._data.meta.ustarcol

    # ------------------------------------------------------------------
    # Level methods — thin delegations
    # ------------------------------------------------------------------

    def level2_quality_flag_expansion(
            self,
            signal_strength: dict | bool = False,
            raw_data_screening_vm97: dict | bool = False,
            ssitc: dict | bool = False,
            gas_completeness: dict | bool = False,
            spectral_correction_factor: dict | bool = False,
            angle_of_attack: dict | bool = False,
            steadiness_of_horizontal_wind: dict | bool = False,
    ):
        """Expand flux quality flags from EddyPro output and compute Level-2 QCF.

        ``finalize_level2()`` afterwards is a no-op kept for API compatibility.
        """
        # Coerce False → None for compatibility with old call sites that
        # passed `signal_strength=False` to mean "skip".
        def _norm(x):
            return x if isinstance(x, dict) else None

        self._data = run_level2(
            self._data,
            signal_strength=_norm(signal_strength),
            raw_data_screening_vm97=_norm(raw_data_screening_vm97),
            ssitc=_norm(ssitc),
            gas_completeness=_norm(gas_completeness),
            spectral_correction_factor=_norm(spectral_correction_factor),
            angle_of_attack=_norm(angle_of_attack),
            steadiness_of_horizontal_wind=_norm(steadiness_of_horizontal_wind),
        )

    def finalize_level2(self):
        """Deprecated no-op; ``level2_quality_flag_expansion()`` now finalizes inline."""
        warnings.warn(
            "finalize_level2() is a no-op since v0.91.0; "
            "level2_quality_flag_expansion() now computes the QCF inline.",
            DeprecationWarning, stacklevel=2,
        )

    def level31_storage_correction(self, gapfill_storage_term: bool = True,
                                   set_storage_to_zero: bool = False):
        """Apply single-point storage correction and re-apply L2 QCF.

        ``finalize_level31()`` afterwards is a no-op kept for API compatibility.
        """
        self._data = run_level31(
            self._data,
            gapfill_storage_term=gapfill_storage_term,
            set_storage_to_zero=set_storage_to_zero,
        )

    def finalize_level31(self):
        """Deprecated no-op; ``level31_storage_correction()`` now finalizes inline."""
        warnings.warn(
            "finalize_level31() is a no-op since v0.91.0; "
            "level31_storage_correction() now computes the QCF inline.",
            DeprecationWarning, stacklevel=2,
        )

    # --- Level-3.2 (multi-call configuration) ---

    def level32_stepwise_outlier_detection(self):
        """Initialise the Level-3.2 outlier detector for stepwise configuration."""
        self._pending_level32 = make_level32_detector(self._data)

    def level32_flag_outliers_abslim_test(self, minval=None, maxval=None,
                                          separate_daytime_nighttime=False,
                                          daytime_minmax=None, nighttime_minmax=None,
                                          showplot=False, verbose=False):
        self._pending_level32.flag_outliers_abslim_test(
            minval=minval, maxval=maxval,
            separate_daytime_nighttime=separate_daytime_nighttime,
            daytime_minmax=daytime_minmax, nighttime_minmax=nighttime_minmax,
            showplot=showplot, verbose=verbose)

    def level32_flag_outliers_localsd_test(self, n_sd=7, winsize=None, showplot=False,
                                           constant_sd=False, separate_daytime_nighttime=False,
                                           verbose=False, repeat=True):
        self._pending_level32.flag_outliers_localsd_test(
            n_sd=n_sd, winsize=winsize,
            separate_daytime_nighttime=separate_daytime_nighttime,
            constant_sd=constant_sd, showplot=showplot,
            verbose=verbose, repeat=repeat)

    def level32_flag_manualremoval_test(self, remove_dates, showplot=False, verbose=False):
        self._pending_level32.flag_manualremoval_test(
            remove_dates=remove_dates, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_increments_zcore_test(self, thres_zscore=30, showplot=False,
                                                    verbose=False, repeat=True):
        self._pending_level32.flag_outliers_increments_zcore_test(
            thres_zscore=thres_zscore, showplot=showplot, verbose=verbose, repeat=repeat)

    def level32_flag_outliers_trim_low_test(self, trim_daytime=False, trim_nighttime=False,
                                            lower_limit=None, showplot=False, verbose=False):
        self._pending_level32.flag_outliers_trim_low_test(
            trim_daytime=trim_daytime, trim_nighttime=trim_nighttime,
            lower_limit=lower_limit, showplot=showplot, verbose=verbose)

    def level32_flag_outliers_hampel_test(self, window_length=13, n_sigma=5.5,
                                          n_sigma_daytime=None, n_sigma_nighttime=None,
                                          k=1.4826, use_differencing=True,
                                          separate_daytime_nighttime=False, showplot=False,
                                          verbose=False, repeat=True):
        self._pending_level32.flag_outliers_hampel_test(
            window_length=window_length, n_sigma=n_sigma,
            n_sigma_daytime=n_sigma_daytime, n_sigma_nighttime=n_sigma_nighttime,
            k=k, use_differencing=use_differencing,
            separate_daytime_nighttime=separate_daytime_nighttime,
            showplot=showplot, verbose=verbose, repeat=repeat)

    def level32_flag_outliers_zscore_rolling_test(self, thres_zscore=4, showplot=False,
                                                  verbose=False, plottitle=None,
                                                  repeat=True, winsize=None):
        self._pending_level32.flag_outliers_zscore_rolling_test(
            thres_zscore=thres_zscore, showplot=showplot, verbose=verbose,
            plottitle=plottitle, winsize=winsize, repeat=repeat)

    def level32_flag_outliers_zscore_test(self, thres_zscore=4, separate_daytime_nighttime=False,
                                          lat=None, lon=None, utc_offset=None,
                                          showplot=False, plottitle=None, verbose=False,
                                          repeat=True, idstr=None):
        self._pending_level32.flag_outliers_zscore_test(
            thres_zscore=thres_zscore,
            separate_daytime_nighttime=separate_daytime_nighttime,
            lat=lat, lon=lon, utc_offset=utc_offset,
            showplot=showplot, plottitle=plottitle,
            verbose=verbose, repeat=repeat, idstr=idstr)

    def level32_flag_outliers_lof_test(self, n_neighbors=None, contamination='auto',
                                       separate_daytime_nighttime=False,
                                       showplot=False, verbose=False, repeat=True, n_jobs=1):
        self._pending_level32.flag_outliers_lof_test(
            n_neighbors=n_neighbors, contamination=contamination,
            separate_daytime_nighttime=separate_daytime_nighttime,
            showplot=showplot, verbose=verbose, repeat=repeat, n_jobs=n_jobs)

    def level32_addflag(self):
        self._pending_level32.addflag()

    def finalize_level32(self):
        """Compute the overall QCF for Level-3.2 and merge results into ``fpc_df``."""
        self._data = run_level32(self._data, outlier_detector=self._pending_level32)
        self._pending_level32 = None

    # --- Level-3.3 ---

    def level33_constant_ustar(self, thresholds, threshold_labels,
                               showplot=True, verbose=True):
        """Flag low-turbulence periods using constant USTAR thresholds.

        ``finalize_level33()`` afterwards is a no-op kept for API compatibility.
        """
        self._data = run_level33_constant_ustar(
            self._data,
            thresholds=thresholds,
            threshold_labels=threshold_labels,
            showplot=showplot, verbose=verbose,
        )

    def level33_ustar_detection(
            self,
            ta_col: str,
            swin_col: str,
            detector_class=None,
            detector_kwargs: dict | None = None,
            n_iter: int = 100,
            n_jobs: int = 1,
            percentiles: tuple = (16, 50, 84),
            showplot: bool = True,
            verbose: bool = True,
    ):
        """Auto-detect USTAR threshold via bootstrap, then apply it as constant scenario(s).

        Runs a multi-year bootstrap USTAR threshold detection and uses the CUT
        (constant upper threshold) percentile results as one filtering scenario each.
        The detection results are stored in ``fpc.ustar_detection`` for inspection.

        Use this instead of ``level33_constant_ustar()`` when you do not yet have a
        site-specific threshold — for example on the first run or for a new site.
        Once you are happy with the detected threshold, you can switch to
        ``level33_constant_ustar()`` in subsequent runs using the p50 value from
        ``fpc.ustar_detection.get_cut_threshold()['p50']``.

        Args:
            ta_col: Air temperature column name (deg C) in the input DataFrame.
            swin_col: Incoming shortwave radiation column (W m-2) in the input DataFrame.
                Used to identify nighttime (SW_IN < 10 W m-2).
            detector_class: USTAR detection class. Defaults to
                ``UstarMovingPointDetection`` (ONEFlux algorithm, Papale et al. 2006).
                Can also be ``UstarVekuriThresholdDetection`` for the quantile-based method.
            detector_kwargs: Extra kwargs forwarded to the detector constructor
                (e.g. ``ta_classes_count``, ``ustar_classes_count``). Column-name
                arguments are set automatically.
            n_iter: Bootstrap iterations per year window. Defaults to 100.
            n_jobs: Parallel workers (1 = sequential, -1 = all CPUs). Defaults to 1.
            percentiles: Bootstrap percentiles to compute and use as separate USTAR
                scenarios. Each value ``p`` becomes a scenario labelled ``CUT_p``.
                Defaults to ``(16, 50, 84)``.
            showplot: Show diagnostic filtering plots. Defaults to True.
            verbose: Print detection progress and summary. Defaults to True.

        Example::

            fpc.level33_ustar_detection(
                ta_col='TA_T1_47_1',
                swin_col='SW_IN_T1_47_1',
                n_iter=100,
                n_jobs=-1,
            )
            print(fpc.ustar_detection.summary())
            cut_p50 = fpc.ustar_detection.get_cut_threshold()['p50']
        """
        self._data = run_level33_ustar_detection(
            self._data,
            ta_col=ta_col,
            swin_col=swin_col,
            detector_class=detector_class,
            detector_kwargs=detector_kwargs,
            n_iter=n_iter,
            n_jobs=n_jobs,
            percentiles=percentiles,
            showplot=showplot,
            verbose=verbose,
        )

    def finalize_level33(self):
        """Deprecated no-op; ``level33_constant_ustar()`` now finalizes inline."""
        warnings.warn(
            "finalize_level33() is a no-op since v0.91.0; "
            "level33_constant_ustar() now computes the QCF inline.",
            DeprecationWarning, stacklevel=2,
        )

    # --- Level-4.1 ---

    def level41_mds(self, swin=None, ta=None, vpd=None,
                    swin_tol=None, ta_tol=2.5, vpd_tol=0.5, avg_min_n_vals=5):
        self._data = run_level41_mds(
            self._data, swin=swin, ta=ta, vpd=vpd,
            swin_tol=swin_tol, ta_tol=ta_tol, vpd_tol=vpd_tol,
            avg_min_n_vals=avg_min_n_vals)

    def level41_longterm_random_forest(self, features=None, reduce_features=False, verbose=0,
                                       features_lag=None, features_lag_stepsize=1,
                                       features_lag_exclude_cols=None,
                                       features_rolling=None, features_rolling_exclude_cols=None,
                                       features_rolling_stats=None,
                                       features_diff=None, features_diff_exclude_cols=None,
                                       features_ema=None, features_ema_exclude_cols=None,
                                       features_poly_degree=None, features_poly_exclude_cols=None,
                                       features_stl=False, features_stl_method='stl',
                                       features_stl_seasonal_period=None,
                                       features_stl_exclude_cols=None, features_stl_components=None,
                                       vectorize_timestamps=False,
                                       add_continuous_record_number=False,
                                       sanitize_timestamp=False,
                                       **rf_kwargs):
        """Gap-fill with long-term Random Forest. See ``run_level41_rf`` for details."""
        engineer = _build_feature_engineer(
            sanitize_timestamp=sanitize_timestamp,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            features_lag=features_lag, features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff, features_diff_exclude_cols=features_diff_exclude_cols,
            features_ema=features_ema, features_ema_exclude_cols=features_ema_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            features_stl=features_stl, features_stl_method=features_stl_method,
            features_stl_seasonal_period=features_stl_seasonal_period,
            features_stl_exclude_cols=features_stl_exclude_cols,
            features_stl_components=features_stl_components,
        )
        self._data = run_level41_rf(
            self._data, features=features, engineer=engineer,
            reduce_features=reduce_features, verbose=verbose, **rf_kwargs,
        )

    def level41_longterm_xgboost(self, features=None, reduce_features=False, verbose=0,
                                 features_lag=None, features_lag_stepsize=1,
                                 features_lag_exclude_cols=None,
                                 features_rolling=None, features_rolling_exclude_cols=None,
                                 features_rolling_stats=None,
                                 features_diff=None, features_diff_exclude_cols=None,
                                 features_ema=None, features_ema_exclude_cols=None,
                                 features_poly_degree=None, features_poly_exclude_cols=None,
                                 features_stl=False, features_stl_method='stl',
                                 features_stl_seasonal_period=None,
                                 features_stl_exclude_cols=None, features_stl_components=None,
                                 vectorize_timestamps=False,
                                 add_continuous_record_number=False,
                                 sanitize_timestamp=False,
                                 **xgb_kwargs):
        """Gap-fill with long-term XGBoost. See ``run_level41_xgb`` for details."""
        engineer = _build_feature_engineer(
            sanitize_timestamp=sanitize_timestamp,
            vectorize_timestamps=vectorize_timestamps,
            add_continuous_record_number=add_continuous_record_number,
            features_lag=features_lag, features_lag_stepsize=features_lag_stepsize,
            features_lag_exclude_cols=features_lag_exclude_cols,
            features_rolling=features_rolling,
            features_rolling_exclude_cols=features_rolling_exclude_cols,
            features_rolling_stats=features_rolling_stats,
            features_diff=features_diff, features_diff_exclude_cols=features_diff_exclude_cols,
            features_ema=features_ema, features_ema_exclude_cols=features_ema_exclude_cols,
            features_poly_degree=features_poly_degree,
            features_poly_exclude_cols=features_poly_exclude_cols,
            features_stl=features_stl, features_stl_method=features_stl_method,
            features_stl_seasonal_period=features_stl_seasonal_period,
            features_stl_exclude_cols=features_stl_exclude_cols,
            features_stl_components=features_stl_components,
        )
        self._data = run_level41_xgb(
            self._data, features=features, engineer=engineer,
            reduce_features=reduce_features, verbose=verbose, **xgb_kwargs,
        )

    # ------------------------------------------------------------------
    # Reporting and plotting
    # ------------------------------------------------------------------

    def get_data(self, verbose: int = 1) -> DataFrame:
        """Return the full input DataFrame combined with all new results/flags."""
        full_df = self._data.full_df
        new_cols = [c for c in self.fpc_df.columns if c not in full_df.columns]
        if verbose:
            info("New variables from flux processing chain:")
            for c in new_cols:
                info(f"  {c}")
            info("No variables in input data were overwritten, only new variables added.")
        return pd.concat([full_df, self.fpc_df[new_cols]], axis=1)

    def get_gapfilled_names(self) -> list:
        return [self.level41[m][s].gapfilled_.name
                for m, scenarios in self.level41.items()
                for s in scenarios]

    def get_nongapfilled_names(self) -> list:
        return [self.level41[m][s].target_col
                for m, scenarios in self.level41.items()
                for s in scenarios]

    def report_gapfilling_variables(self):
        for gfmethod, ustar_scenarios in self.level41.items():
            for s in ustar_scenarios:
                inst = self.level41[gfmethod][s]
                info(f"{gfmethod} ({s}): {inst.target_col} -> {inst.gapfilled_.name}")

    def get_gapfilled_variables(self) -> DataFrame:
        cols = self.get_gapfilled_names() + self.get_nongapfilled_names()
        return self.fpc_df[cols].copy()

    def _report_scores(self, attr: str, label: str, outfile_prefix: str, outpath: str | None):
        for gfmethod, ustar_scenarios in self.level41.items():
            for s in ustar_scenarios:
                inst = self.level41[gfmethod][s]
                if not hasattr(inst, attr):
                    info(f"{gfmethod} {s} does not have {label}.")
                    continue
                info(f"{label.upper()} ({gfmethod}): {s}")
                scores = getattr(inst, attr)
                try:
                    df = pd.DataFrame.from_dict(scores, orient='columns')
                except ValueError:
                    df = pd.DataFrame.from_dict(scores, orient='index')
                _console.print(df.to_string())
                if outpath:
                    df.to_csv(Path(outpath) / f"{outfile_prefix}_{s}_{gfmethod}.csv")

    def report_traintest_model_scores(self, outpath: str | None = None):
        self._report_scores('scores_traintest_', 'train/test model scores',
                            'traintest_model_scores', outpath)

    def report_traintest_details(self, outpath: str | None = None):
        self._report_scores('traintest_details_', 'train/test details',
                            'traintest_model_details', outpath)

    def report_gapfilling_model_scores(self, outpath: str | None = None):
        self._report_scores('scores_', 'model scores',
                            'gapfilling_model_scores', outpath)

    def report_gapfilling_feature_importances(self, outpath: str | None = None):
        for gfmethod, ustar_scenarios in self.level41.items():
            for s in ustar_scenarios:
                inst = self.level41[gfmethod][s]
                if not hasattr(inst, 'feature_importance_per_year'):
                    info(f"{gfmethod} {s} does not have feature importances.")
                    continue
                info(f"FEATURE IMPORTANCES ({gfmethod}): {s}")
                df = inst.feature_importance_per_year
                _console.print(df.to_string())
                if outpath:
                    df.to_csv(Path(outpath) / f"gapfilling_model_feature_importances_{s}_{gfmethod}.csv")

    def report_gapfilling_poolyears(self):
        info("Data pools used for machine-learning gap-filling:")
        for gfmethod, ustar_scenarios in self.level41.items():
            for s in ustar_scenarios:
                inst = self.level41[gfmethod][s]
                if not hasattr(inst, 'yearpools'):
                    info(f"{gfmethod} {s} did not use poolyears.")
                    continue
                for yr, pool in inst.yearpools.items():
                    info(f"{yr}: {gfmethod} used data from {pool['poolyears']} "
                         f"for gap-filling {inst.target_col} -> {inst.gapfilled_.name}")

    def showplot_feature_ranks_per_year(self):
        for gfmethod, ustar_scenarios in self.level41.items():
            for s in ustar_scenarios:
                inst = self.level41[gfmethod][s]
                if not hasattr(inst, 'results_yearly_'):
                    info(f"{gfmethod} {s} does not have feature ranks.")
                    continue
                title = f"{inst.gapfilled_.name} ({s})"
                first_key = next(iter(inst.results_yearly_))
                model_params = inst.results_yearly_[first_key].model_.get_params()
                inst.showplot_feature_ranks_per_year(
                    title=title, subtitle=f"MODEL: {gfmethod} / PARAMS: {model_params}")

    def showplot_mds_gapfilling_qualities(self):
        for s, inst in self.level41.get('mds', {}).items():
            inst.showplot()

    def showplot_gapfilled_heatmap(self, vmin: float = None, vmax: float = None):
        gapfilled_vars = self.get_gapfilled_names()
        nongapfilled_vars = self.get_nongapfilled_names()
        gfvars = self.get_gapfilled_variables()
        for ix, g in enumerate(gapfilled_vars):
            fig = plt.figure(figsize=(12, 12), dpi=100)
            gs = gridspec.GridSpec(1, 2)
            ax_ngf = fig.add_subplot(gs[0, 0])
            ax_gf = fig.add_subplot(gs[0, 1])
            series_ngf = gfvars[nongapfilled_vars[ix]].copy()
            if isinstance(series_ngf, pd.DataFrame):
                series_ngf = series_ngf.loc[:, ~series_ngf.columns.duplicated()].copy().squeeze()
            HeatmapDateTime(series=series_ngf).plot(ax=ax_ngf, vmin=vmin, vmax=vmax)
            HeatmapDateTime(series=gfvars[g]).plot(ax=ax_gf, vmin=vmin, vmax=vmax)
            fig.tight_layout()
            fig.show()

    def showplot_gapfilled_cumulative(self, gain: float = 1, units: str = "", per_year: bool = True,
                                      start_year: int = None, end_year: int = None,
                                      show_reference: bool = True,
                                      excl_years_from_reference: list = None):
        gapfilled_vars = self.get_gapfilled_names()
        gfvars = self.get_gapfilled_variables()[gapfilled_vars]
        if per_year:
            for g in gfvars:
                CumulativeYear(
                    series=gfvars[g].multiply(gain),
                    series_units=units,
                    yearly_end_date=None,
                    start_year=start_year, end_year=end_year,
                    show_reference=show_reference,
                    excl_years_from_reference=excl_years_from_reference,
                    highlight_year=None,
                    highlight_year_color='#F44336').plot()
        else:
            df = gfvars[gapfilled_vars].copy().multiply(gain)
            Cumulative(df=df, units=units, start_year=start_year, end_year=end_year).plot()

    def analyze_highest_quality_flux(self, showplot: bool = True):
        analyze_highest_quality_flux(
            flux=self.fpc_df[self.filteredseries_hq.name],
            lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset,
            showplot=showplot)


class LoadEddyProOutputFiles:

    def __init__(self, sourcedir,
                 filetype: Literal['EDDYPRO-FLUXNET-CSV-30MIN', 'EDDYPRO-FULL-OUTPUT-CSV-30MIN']):
        self.sourcedir = sourcedir
        self.filetype = filetype
        self._maindf = None
        self._filepaths = None
        self._metadata = None

    @property
    def maindf(self) -> DataFrame:
        if not isinstance(self._maindf, DataFrame):
            raise RuntimeError('No data available, please run .loadfiles() first.')
        return self._maindf

    @property
    def filepaths(self) -> list:
        if not isinstance(self._filepaths, list):
            raise RuntimeError('Filepaths not available, please run .searchfiles() first.')
        return self._filepaths

    @property
    def metadata(self) -> DataFrame:
        if not isinstance(self._metadata, DataFrame):
            raise RuntimeError('No metadata available, please run .loadfiles() first.')
        return self._metadata

    def searchfiles(self, extension: str = '*.csv'):
        from diive.core.io.filereader import search_files
        fileids = self._init_filetype()
        self._filepaths = search_files(self.sourcedir, extension)
        self._filepaths = filter_strings_by_elements(list1=self.filepaths, list2=fileids)
        info(f"Found {len(self.filepaths)} files with extension {extension} and file IDs {fileids}:")
        for ix, f in enumerate(self.filepaths):
            info(f"  File #{ix + 1}: {f}")

    def loadfiles(self):
        from diive.core.io.filereader import MultiDataFileReader
        loaddatafile = MultiDataFileReader(filetype=self.filetype, filepaths=self.filepaths)
        self._maindf = loaddatafile.data_df
        self._metadata = loaddatafile.metadata_df

    def _init_filetype(self):
        if self.filetype == 'EDDYPRO-FLUXNET-CSV-30MIN':
            return ['eddypro_', '_fluxnet_']
        if self.filetype == 'EDDYPRO-FULL-OUTPUT-CSV-30MIN':
            return ['eddypro_', '_full_output_']
        raise ValueError(f"Unknown filetype: {self.filetype}")


# Flux variables that must NOT receive USTAR turbulence filtering.
# For these, run_level33 is called with thresholds=[0] (filter nothing).
# Extend this set if your site uses non-standard column names for energy fluxes.
_ENERGY_FLUX_VARS = frozenset({'H', 'LE', 'G', 'SH', 'SLE', 'FH2O'})


class QuickFluxProcessingChain:

    def __init__(
            self,
            fluxvars: list,
            sourcedirs: list,
            site_lat: float,
            site_lon: float,
            utc_offset: int,
            nighttime_threshold: int = 20,
            daytime_accept_qcf_below: int = 2,
            nighttime_accept_qcf_below: int = 2,
            test_signal_strength=False,
            test_signal_strength_col='',
            test_signal_strength_method='discard above',
            test_signal_strength_threshold=999,
    ):
        self.fluxvars = fluxvars
        self.sourcedirs = sourcedirs
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.utc_offset = utc_offset
        self.nighttime_threshold = nighttime_threshold
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttime_accept_qcf_below = nighttime_accept_qcf_below
        self.test_signal_strength = test_signal_strength
        self.test_signal_strength_col = test_signal_strength_col
        self.test_signal_strength_method = test_signal_strength_method
        self.test_signal_strength_threshold = test_signal_strength_threshold
        self.fpc = None
        self._run()

    def _run(self):
        self.maindf, self.metadata = self._load_data()
        for fluxcol in self.fluxvars:
            self.fpc = self._start_fpc(fluxcol=fluxcol)
            self._run_level2()
            self._run_level31()
            self._run_level32()
            self._run_level33(fluxcol=fluxcol)

    def _run_level33(self, fluxcol):
        if fluxcol in _ENERGY_FLUX_VARS:
            thresholds, ustar_scenarios = [0], ['CUT_NONE']
        else:
            thresholds, ustar_scenarios = [0.08], ['CUT_PRELIM']
        self.fpc.level33_constant_ustar(thresholds=thresholds,
                                        threshold_labels=ustar_scenarios, showplot=False)
        for s in ustar_scenarios:
            self.fpc.level33_qcf[s].showplot_qcf_heatmaps()
            self.fpc.level33_qcf[s].report_qcf_evolution()

    def _run_level32(self):
        self.fpc.level32_stepwise_outlier_detection()
        self.fpc.level32_flag_outliers_zscore_test(
            thres_zscore=4, separate_daytime_nighttime=True,
            lat=self.site_lat, lon=self.site_lon, utc_offset=self.utc_offset,
            showplot=False, verbose=True, repeat=True)
        self.fpc.level32_addflag()
        self.fpc.finalize_level32()

    def _run_level31(self):
        self.fpc.level31_storage_correction(gapfill_storage_term=True)

    def _run_level2(self):
        self.fpc.level2_quality_flag_expansion(
            signal_strength={
                'apply': self.test_signal_strength,
                'signal_strength_col': self.test_signal_strength_col,
                'method': self.test_signal_strength_method,
                'threshold': self.test_signal_strength_threshold},
            raw_data_screening_vm97={
                'apply': True, 'spikes': True, 'amplitude': False, 'dropout': True,
                'abslim': False, 'skewkurt_hf': False, 'skewkurt_sf': False,
                'discont_hf': False, 'discont_sf': False},
            ssitc={'apply': True, 'setflag_timeperiod': None},
            gas_completeness={'apply': True},
            spectral_correction_factor={'apply': True},
            angle_of_attack={'apply': False, 'application_dates': False},
            steadiness_of_horizontal_wind={'apply': False},
        )

    def _start_fpc(self, fluxcol: str):
        return FluxProcessingChain(
            df=self.maindf, fluxcol=fluxcol,
            site_lat=self.site_lat, site_lon=self.site_lon,
            utc_offset=self.utc_offset,
            daytime_accept_qcf_below=self.daytime_accept_qcf_below,
            nighttime_accept_qcf_below=self.nighttime_accept_qcf_below,
        )

    def _load_data(self):
        ep = LoadEddyProOutputFiles(sourcedir=self.sourcedirs, filetype='EDDYPRO-FLUXNET-CSV-30MIN')
        ep.searchfiles()
        ep.loadfiles()
        return ep.maindf, ep.metadata
