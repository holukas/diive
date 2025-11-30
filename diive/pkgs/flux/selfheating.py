"""
SELF-HEATING CORRECTION FOR OPEN-PATH IRGAS (SCOP)

    Calculation of the flux correction term (FCT) for open-path infrared gas analyzers (IRGAs)
    based on scaling factors from parallel measurements with an (en)closed-path IRGA.

    The correction can be applied to CO2 fluxes (NEE, µmol m-2 s-1).
    It can be used for H2O fluxes (LE, W m-2), but the code has only been implemented for testing.
    When LE from the open-path shows higher fluxes than LE from the closed-path, then this
    correction is most likely the wrong approach. The self-heating correction always increases the
    flux (it assumes heating lowers density, so it adds density back).

    This correction is designed to remove spurious CO2 flux measurements caused by the sun-induced
    heating of the open-path (OP) instrument surfaces. This self-heating warms the air passing the
    sensor head, creating a low-density thermal plume around the sampling volume. Since the OP-IRGA
    measures molar density, this plume artificially lowers the measured CO2 concentration, resulting
    in a systematic, non-biological uptake (negative flux) bias, especially prevalent during high
    solar radiation and low wind conditions.

Core Correction Principle

    The method first calculates an unscaled flux correction term (FCT_UNSC) based on boundary-layer
    dynamics and instrument-specific thermal properties (e.g., sensible heat flux S or instrument
    surface temperature TS). FCT_UNSC is proportional to the heat transfer from the instrument surfaces
    to the air, which directly drives the artificial dilution.

Scaling and Final Application

    To apply the correction, FCT_UNSC is scaled using a scaling factor SF derived from a parallel
    measurement campaign with a (en)closed path IRGA.

1.  Reference: A co-located (en)closed-path (CP) IRGA, e.g. the LI-7200, which is largely
    immune to self-heating effects, serves as the "true" flux reference.
2.  Optimization: The scaling factor SF is determined by minimizing the difference between the CP
    reference flux and the OP flux corrected by FCT_UNSC * SF. In this implementation, the optimization
    is binned by a key environmental variable, such as USTAR friction velocity, and done separately for
    daytime and nighttime conditions.
3.  Final corrected OP flux: The final flux correction term (FCT) is calculated as FCT_UNSC * SF. This
    term is then added to the raw OP flux to produce the corrected measurement:

    Flux_OP_corr = Flux_OP_raw + FCT

The result is a corrected flux measurement.

References:
    (BUR06) Burba et al. (2006). Correcting apparent off-season CO2 uptake due
        to surface heating of an open path gas analyzer: Progress report
        of an ongoing study. 13.
    (JAR09) Järvi, L., Mammarella, I., Eugster, W., Ibrom, A., Siivola, E., Dellwik,
        E., Keronen, P., Burba, G., & Vesala, T. (2009). Comparison of net CO2
        fluxes measured with open- and closed-path infrared gas analyzers in
        an urban complex environment. 14, 16.
    (KIT17) Kittler et al. (2017). High-quality eddy-covariance CO2 budgets
        under cold climate conditions: Arctic Eddy-Covariance CO2 Budgets.
        Journal of Geophysical Research: Biogeosciences, 122(8), 2064–2084.
        https://doi.org/10.1002/2017JG003830

Abbreviations:

    Required variables and units:
        swin ... shortwave-incoming radiation (W m-2)
        ta ... ambient air temperature (°C)
        u ... horizontal wind speed (m s-1)
        ustar ... USTAR friction velocity (m s-1)
        qc ... CO2 molar density (µmol m-3)
        rho_a ... air density (kg m-3)
        rho_v ... water vapor density (kg m-3)
        c_p ... specific heat at constant pressure, air heat capacity (J K-1 kg-1)

    Newly calculated variables:
        ra ... aerodynamic resistance (s m-1)
        rho_d ... dry air density (kg m-3)
        k_air ... thermal conductivity of air (W m-1 K-1)
        ts ... bulk instrument surface temperature (°C), from BUR06/JAR09
        s ... sensible heat from all key instrument surfaces (W m-2), from BUR08
        fct_unsc ... unscaled flux correction term (flux units)
        fct_unsc_gf ... gap-filled unscaled flux correction term (flux units)
        fct ... flux correction term (flux units)
        sf ... scaling factor (unitless)
        lv ... latent heat of vaporization (J µmol-1), in this units can be used for LE

    Other:
        OP ... open-path
        CP ... enclosed-path
        wpl ... WPL-correction

"""
import time
from dataclasses import dataclass
from typing import Literal, Optional

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from diive.pkgs.createvar.air import dry_air_density, aerodynamic_resistance
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag
from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
from diive.pkgs.outlierdetection.hampel import HampelDaytimeNighttime

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 14)
pd.set_option('display.max_rows', 30)

FluxType = Literal["CO2", "H2O"]


@dataclass(frozen=True)
class ColumnConfig:
    """
    Configuration for column definitions related to flux and thermal properties.

    This class defines constant configurations for column names used in flux and
    thermal properties-related calculations. It organizes these columns by grouping
    similar properties and suffixes or prefixes they may require during computation.
    """
    class_var_group: str = 'GROUP_CLASSVAR'
    daytime: str = 'DAYTIME'
    air_thermal_conductivity: str = 'AIR_THERMAL_CONDUCTIVITY'
    aerodynamic_resistance: str = 'AERODYNAMIC_RESISTANCE'
    dry_air_density: str = 'DRY_AIR_DENSITY'
    t_instr_surface: str = 'TS'

    # Dynamic base names (will be prefixed/suffixed based on flux type)
    flux_corr_suffix: str = '_OP_CORR'
    fct_unsc: str = 'FCT_UNSC'
    fct_unsc_gf: str = 'FCT_UNSC_gfRF'
    fct: str = 'FCT'
    sf: str = 'SF'
    s: str = 'S'  # Sensible heat from all key instrument surfaces (W m-2) (BUR08)


class ScopPhysics:

    def __init__(self,
                 flux_type: FluxType,
                 ta: pd.Series,
                 gas_density: pd.Series,
                 rho_a: pd.Series,
                 rho_v: pd.Series,
                 u: pd.Series,
                 c_p: pd.Series,
                 ustar: pd.Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 remove_outliers_method: Literal["fast", "separate"] = "fast"):
        """
        Args:
            flux_type: "CO2" or "H2O"
            ta: series, air temperature (°C)
            gas_density: series, molar density of the gas (µmol m-3)
            rho_a: series, air density (kg m-3)
            rho_v: series, water vapor density (kg m-3)          
            u: series, wind speed (m s-1)            
            c_p: series, air heat capacity, specific heat at constant pressure of ambient air (J K-1 kg-1)
            ustar: series, friction velocity (m s-1)
            lat: float, latitude of the site
            lon: float, longitude of the site
            utc_offset: int, UTC offset of the timestamp (hours)
            remove_outliers_method: str, method to remove outliers from the data
                Used for removing outliers in 'ra' and 'fct_unsc'.
        """
        self.flux_type = flux_type
        self.ta = ta
        self.gas_density = gas_density
        self.rho_a = rho_a
        self.rho_v = rho_v
        self.u = u
        self.c_p = c_p
        self.ustar = ustar
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset
        self.remove_outliers_method = remove_outliers_method

        self.cols = ColumnConfig()

        # Detect daytime/nighttime from potential radiation (daytime=1, nighttime=0)
        dnf = DaytimeNighttimeFlag(timestamp_index=self.ta.index, nighttime_threshold=20,
                                   lat=self.lat, lon=self.lon, utc_offset=self.utc_offset)
        self.daytime = dnf.get_daytime_flag()
        self.swin_pot = dnf.get_swinpot()

        # Calculate aerodynamic resistance (ra) (s m-1)
        self.ra = self._calc_aerodynamic_resistance(u=self.u, ustar=self.ustar, rem_outliers=True)

        # Calculate dry air density (rho_d) (kg m-3)
        self.rho_d = dry_air_density(rho_v=rho_v, rho_a=rho_a)

        # Calculate thermal conductivity of air (required for BUR08) (k_air) (W m-1 K-1)
        self.k_air = self._calc_air_thermal_conductivity(ta=self.ta)

        # Calculate latent heat of vaporization (required for LE only) (J µmol-1)
        self.lv = self._calc_latent_heat_vaporization_j_umol(ta=self.ta)

        # Calculated in .run()
        self.ts = pd.Series(name=self.cols.t_instr_surface)  # Bulk instrument surface temperature (BUR06, JAR09)
        self.S = pd.Series(name=self.cols.s)  # Sensible heat from all key instrument surfaces  (BUR08)
        self.fct_unsc = pd.Series(name=self.cols.fct_unsc)  # Unscaled flux correction term, produced by all methods
        self.fct_unsc_gf = pd.Series(name=self.cols.fct_unsc_gf)  # Unscaled flux correction term, gap-filled

        # Initialize BUR08 specific surfaces
        self.ts_top = None
        self.ts_bottom = None
        self.ts_spar = None

    def get_results(self) -> pd.DataFrame:
        frame = {
            self.cols.fct_unsc_gf: self.fct_unsc_gf,
            self.cols.fct_unsc: self.fct_unsc,
            self.cols.s: self.S,
            self.cols.t_instr_surface: self.ts,
            self.ta.name: self.ta,
            self.c_p.name: self.c_p,
            self.ra.name: self.ra,
            self.rho_a.name: self.rho_a,
            self.rho_d.name: self.rho_d,
            self.rho_v.name: self.rho_v,
            self.u.name: self.u,
            self.ustar.name: self.ustar,
            self.gas_density.name: self.gas_density,
            self.swin_pot.name: self.swin_pot,
            self.k_air.name: self.k_air,
            self.daytime.name: self.daytime,
            self.lv.name: self.lv
        }
        return pd.DataFrame.from_dict(frame)

    def run(self, correction_method_base: Literal["JAR09", "BUR06", "BUR08"] = "JAR09",
            gapfill: bool = True):
        if correction_method_base == "BUR06":
            self.ts = self._estimate_surface_temp_bur06()
            self.fct_unsc = self._flux_correction_term_unscaled_jar09_bur06(ts=self.ts)
        elif correction_method_base == "JAR09":
            self.ts = self._estimate_surface_temp_jar09()
            self.fct_unsc = self._flux_correction_term_unscaled_jar09_bur06(ts=self.ts)
        elif correction_method_base == "BUR08":
            self.fct_unsc, self.S = self._flux_correction_term_unscaled_bur08()

        self.fct_unsc.name = self.cols.fct_unsc

        # Remove outliers from unscaled flux correction term
        if self.remove_outliers_method == "fast":
            self.fct_unsc = self._remove_outliers_fast(series=self.fct_unsc)
        elif self.remove_outliers_method == "separate":
            self.fct_unsc = self._remove_outliers(series=self.fct_unsc)

        if gapfill:
            self.fct_unsc_gf = self._gapfill()

    def stats(self):
        """
        Prints a diagnostic summary of the physics calculation.
        Focuses on Data Coverage (Hybrid Gap-Filling), Instrument Heating,
        and Correction Magnitude.
        """
        print(f"\n{'=' * 65}")
        print(f"SCOP PHYSICS DIAGNOSTICS ({self.flux_type})")
        print(f"{'=' * 65}")

        # --- 1. DATA COVERAGE (Hybrid RF + MDV) ---
        n_total = len(self.ta)
        n_raw = self.fct_unsc.count()
        n_final = self.fct_unsc_gf.count()
        n_filled = n_final - n_raw

        print(f"1. DATA COVERAGE & GAP-FILLING")
        print(f"{'-' * 30}")
        print(f"   Total Timestamps       : {n_total:,}")
        print(f"   Raw Physics Calculated : {n_raw:,}  ({n_raw / n_total:>6.1%})")
        print(f"   Final Gap-Filled (GF)  : {n_final:,}  ({n_final / n_total:>6.1%})")
        print(f"   -> Imputed (RF + MDV)  : {n_filled:,}  ({n_filled / n_total:>6.1%})")

        # --- 2. HEATING EFFECT (Delta T) ---
        # The core physical assumption is that Ts > Ta due to radiation.
        if self.ts is not None and not self.ts.dropna().empty:
            delta_t = self.ts - self.ta

            # Filter for Daytime (when heating is relevant)
            day_mask = (self.daytime == 1)

            # Stats
            d_mean = delta_t.loc[day_mask].mean()
            d_max = delta_t.loc[day_mask].max()
            n_mean = delta_t.loc[~day_mask].mean()

            print(f"\n2. INSTRUMENT SELF-HEATING (Ts - Ta)")
            print(f"{'-' * 30}")
            print(f"   Avg Daytime Heating    : {d_mean:+.2f} °C")
            print(f"   Max Daytime Heating    : {d_max:+.2f} °C")
            print(f"   Avg Nighttime Offset   : {n_mean:+.2f} °C")

        # --- 3. SENSIBLE HEAT FLUX (BUR08 Only) ---
        if not self.S.dropna().empty:
            s_day = self.S.loc[self.daytime == 1].mean()
            print(f"\n   Modeled Sensible Heat (S) : {s_day:.1f} W m-2 (Daytime Avg)")

        # --- 4. CORRECTION MAGNITUDE (FCT_UNSC) ---
        # Units are always [µmol m-2 s-1] regardless of flux type in this class.
        print(f"\n3. UNSCALED CORRECTION TERM (FCT_unsc)")
        print(f"   (Units: µmol m-2 s-1)")
        print(f"{'-' * 30}")

        # Group stats by Day/Night
        df_stat = pd.DataFrame({'FCT': self.fct_unsc_gf, 'Day': self.daytime})
        g = df_stat.groupby('Day')['FCT'].agg(['mean', 'std', 'min', 'max'])

        # Safely get rows (handle if missing day or night data)
        row_day = g.loc[1] if 1 in g.index else pd.Series(0, index=g.columns)
        row_night = g.loc[0] if 0 in g.index else pd.Series(0, index=g.columns)

        print(
            f"   DAYTIME   : {row_day['mean']:+.4f} ± {row_day['std']:.4f}  [Range: {row_day['min']:.2f} to {row_day['max']:.2f}]")
        print(
            f"   NIGHTTIME : {row_night['mean']:+.4f} ± {row_night['std']:.4f}  [Range: {row_night['min']:.2f} to {row_night['max']:.2f}]")

        # --- 5. DRIVERS OVERVIEW ---
        day_idx = self.daytime == 1
        swin_avg = self.swin_pot[day_idx].mean() if not self.swin_pot.isna().all() else 0
        u_avg = self.u[day_idx].mean() if not self.u.isna().all() else 0
        ra_avg = self.ra[day_idx].mean() if not self.ra.isna().all() else 0

        print(f"\n4. KEY DRIVERS (Daytime Avg)")
        print(f"{'-' * 30}")
        print(f"   Radiation (SW_IN)      : {swin_avg:.0f} W m-2")
        print(f"   Wind Speed (U)         : {u_avg:.2f} m s-1")
        print(f"   Aero Resistance (Ra)   : {ra_avg:.1f} s m-1")

        print(f"{'=' * 65}\n")

    def _gapfill(self):
        """
        Gap-fill unscaled flux correction term using a hybrid approach:
        1. RandomForestTS (Machine Learning)
        2. Mean Diurnal Variation (MDV) Lookup Table (Fallback)
        """
        # 1. RANDOM FOREST GAP-FILLING
        frame = {
            self.cols.fct_unsc: self.fct_unsc,
            self.ta.name: self.ta,
            self.c_p.name: self.c_p,
            self.rho_a.name: self.rho_a,
            self.u.name: self.u,
            self.ustar.name: self.ustar
        }
        gfdf = pd.DataFrame.from_dict(frame).dropna()

        # Run random forest
        rfts = RandomForestTS(
            input_df=gfdf, target_col=self.cols.fct_unsc, verbose=True,
            features_lag=[-1, -1], vectorize_timestamps=True,
            add_continuous_record_number=True, sanitize_timestamp=True,
            perm_n_repeats=1, n_estimators=100, random_state=42,
            max_depth=None, min_samples_split=10, min_samples_leaf=5,
            criterion='squared_error', n_jobs=-1
        )
        rfts.trainmodel(showplot_scores=False, showplot_importance=False)
        rfts.fillgaps(showplot_scores=False, showplot_importance=False)

        # Initial result from RF
        fct_rf = rfts.gapfilling_df_[self.cols.fct_unsc_gf].copy()

        # 2. MDV LOOKUP TABLE (FILL REMAINING GAPS)
        # Align RF result to original index (fill missing spots with NaN)
        fct_hybrid = fct_rf.reindex(self.ta.index)

        # Identify gaps that RF missed (e.g., if drivers were missing)
        missing_mask = fct_hybrid.isna()

        if missing_mask.any():
            print(f"    (i) RF left {missing_mask.sum()} gaps. Filling with MDV Lookup Table...")

            # Create a temporary DataFrame for grouping
            df_lut = pd.DataFrame({'Target': fct_hybrid}, index=self.ta.index)

            # Ensure DatetimeIndex
            if not isinstance(df_lut.index, pd.DatetimeIndex):
                df_lut.index = pd.to_datetime(df_lut.index)

            # Add temporal features
            df_lut['__MONTH'] = df_lut.index.month
            df_lut['__HOUR'] = df_lut.index.hour
            df_lut['__MINUTE'] = df_lut.index.minute

            # Use pre-calculated daytime flag if available, else derive it
            if self.daytime is not None:
                # Ensure alignment
                df_lut['__DAYTIME'] = self.daytime.reindex(df_lut.index).fillna(-1)
            else:
                # Fallback if daytime flag isn't ready (unlikely in this class flow)
                df_lut['__DAYTIME'] = 0

            # Define groupers
            grouper_fine = ['__MONTH', '__DAYTIME', '__HOUR', '__MINUTE']
            grouper_coarse = ['__MONTH', '__DAYTIME']

            # Calculate medians (transform broadcasts back to original shape)
            mdv_fine = df_lut.groupby(grouper_fine)['Target'].transform('median')
            mdv_coarse = df_lut.groupby(grouper_coarse)['Target'].transform('median')
            global_median = df_lut['Target'].median()

            # Apply waterfall filling
            # 1. Fill with fine-grained MDV
            fct_hybrid = fct_hybrid.fillna(mdv_fine)

            # 2. Fill remaining with coarse MDV (Month-Daytime)
            fct_hybrid = fct_hybrid.fillna(mdv_coarse)

            # 3. Fill any final edge cases with global median
            fct_hybrid = fct_hybrid.fillna(global_median)

            print(f"    > Gaps remaining after MDV: {fct_hybrid.isna().sum()}")

        return fct_hybrid

    @staticmethod
    def _calc_air_thermal_conductivity(ta: pd.Series) -> pd.Series:
        """
        Calculates the thermal conductivity of air using a linear approximation.

        This method uses a linear approximation to estimate the thermal conductivity of air 
        for temperatures ranging from -50°C to 100°C. The calculation is based on the 
        temperature-dependent relationship where the thermal conductivity at 0°C is approximately 
        0.02425 W/m·K and increases by approximately 0.00007 W/m·K per degree Celsius.

        Args:
            ta (pd.Series): A pandas Series representing the air temperature in degrees Celsius.

        Returns:
            pd.Series: A pandas Series containing the calculated thermal conductivity of air (W m-1 K-1)
                corresponding to the input temperatures. 
        """
        # Linear approximation suitable for atmospheric range (-50 to 100 C)
        # k ~ 0.02425 at 0C, increasing by ~0.00007 per degree C
        k_air = 0.02425 + (0.00007 * ta)
        k_air.name = "AIR_THERMAL_CONDUCTIVITY"
        return k_air

    import pandas as pd

    @staticmethod
    def _calc_latent_heat_vaporization_j_umol(ta: pd.Series) -> pd.Series:
        """
        Calculates Latent Heat of Vaporization in [J µmol-1].

        Needed for the correction of the latent heat flux LE.

        Formula derivation:
        1. Lv [J/kg]  = (2.501 - 0.00237 * Ta) * 10^6
        2. Mw [kg/mol] = 0.018015
        3. Lv [J/µmol] = Lv [J/kg] * Mw * 10^-6

        (The 10^6 and 10^-6 cancel out).
        """
        # Molar mass of water (kg/mol)
        MW_WATER = 0.01801528

        # Calculate directly in J / µmol
        lv_j_umol = (2.501 - 0.00237 * ta) * MW_WATER
        lv_j_umol.name = "LATENT_HEAT_VAPORIZATION_J_UMOL"

        return lv_j_umol

    def _remove_outliers(self, series: pd.Series):
        ham = HampelDaytimeNighttime(
            series=series, n_sigma_dt=4, n_sigma_nt=4,
            window_length=48 * 5, showplot=True, verbose=True,
            lat=self.lat, lon=self.lon, utc_offset=self.utc_offset)
        ham.calc(repeat=False)
        return ham.filteredseries

    def _calc_aerodynamic_resistance(self, u, ustar, rem_outliers: bool = False) -> pd.Series:
        # Aerodynamic resistance
        # ra: series, aerodynamic resistance (s m-1)
        ra = aerodynamic_resistance(u_ms=u, ustar_ms=ustar)
        if rem_outliers:
            if self.remove_outliers_method == "fast":
                ra = self._remove_outliers_fast(series=ra)
            elif self.remove_outliers_method == "separate":
                ra = self._remove_outliers(series=ra)
            else:
                raise ValueError(f"Unknown remove_outliers_method: {self.remove_outliers_method}")
        return ra

    @staticmethod
    def _remove_outliers_fast(series, plot_title: str = "X", n_sigmas: int = 5):
        """Remove outliers with Hampel filter, using running MAD (median absolute deviation)

        :param series: pandas.Series, data from which outliers are removed
        :param plot_title: Plot title
        :param n_sigmas:
        :return:
        pandas.Series with outliers removed
        """
        # n_sigmas = 4  # Number of sigmas for limits
        k = 1.4826  # Scale factor for Gaussian distribution
        window = 1440  # Rolling time window in number of records
        min_periods = 1  # Min number of records in window

        _series_rolling = series.rolling(window=window, min_periods=min_periods, center=True)
        _series_running_median = _series_rolling.median()
        _series_sub_winmed = series.sub(_series_running_median).abs()  # Data series minus median in time window
        _series_running_mad = _series_sub_winmed.rolling(window=window, min_periods=min_periods,
                                                         center=True).median().multiply(k)
        _diff_series_runmed = np.abs(series - _series_running_median)
        _diff_limit = _series_running_mad.multiply(n_sigmas)  # Limit

        _outliers_ix = _diff_series_runmed > _diff_limit
        series.loc[_outliers_ix] = np.nan  # Remove outliers from series (set to missing)
        return series

        # # Plot
        # figsize = (14, 5)
        # plt.figure()
        # series_orig.plot(figsize=figsize, title=f"{plot_title} BEFORE outlier removal");
        # plt.figure()
        # series.plot(title=f"{plot_title} AFTER outlier removal", figsize=figsize, label="series after outlier removal");
        # _diff_series_runmed.loc[~_outliers_ix].plot(
        #     label="absolute difference of: series - running median of series; for outlier detection")
        # _diff_limit.plot(label="limit from: series running MAD * number of sigmas")
        # _series_running_mad.plot(label="series running MAD (median absolute deviation)")
        # _series_running_median.plot(label="series running median")
        # plt.legend()
        # print(series.describe())

        # return series, series_orig

    def _flux_correction_term_unscaled_bur08(self) -> tuple[pd.Series, pd.Series]:
        """Calculate bulk instrument surface temperature (BUR08)"""

        # TOP OF WINDOW ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_top = 1.005 * self.ta + 0.24
        _ts_bur08_night_top = 1.008 * self.ta - 0.41
        ts_top = pd.Series(index=_ts_bur08_day_top.index)
        ts_top.loc[self.daytime == 1] = _ts_bur08_day_top  # Use daytime Ts in daytime data rows
        ts_top.loc[self.daytime == 0] = _ts_bur08_night_top  # Use nighttime Ts in nighttime data rows
        # Calculate sigma below top window
        l_top = 0.045  # Diameter of the detector housing (m)
        sigma_top = 0.0028 * np.sqrt(l_top / self.u) + (0.00025 / self.u) + 0.0045
        # Calculate top window sensible heat
        r_top = 0.0225  # Radius of the detector sphere (m)
        a = (r_top + sigma_top) * (ts_top - self.ta)
        b = r_top * sigma_top
        S_top = self.k_air * (a / b)

        # BOTTOM WINDOW ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_bottom = 0.944 * self.ta + 2.57
        _ts_bur08_night_bottom = 0.883 * self.ta + 2.17
        ts_bottom = pd.Series(index=_ts_bur08_day_bottom.index)
        ts_bottom.loc[self.daytime == 1] = _ts_bur08_day_bottom  # Use daytime Ts in daytime data rows
        ts_bottom.loc[self.daytime == 0] = _ts_bur08_night_bottom  # Use nighttime Ts in nighttime data rows
        # Calculate sigma above bottom window
        l_bottom = 0.065  # Diameter of the source housing (m)
        sigma_bottom = 0.004 * np.sqrt(l_bottom / self.u) + 0.004
        # Calculate bottom window sensible heat
        S_bottom = self.k_air * ((ts_bottom - self.ta) / sigma_bottom)

        # SPAR ---
        # Surface temperatures day/night
        # Calculate and then combine day and night temperatures
        _ts_bur08_day_spar = 1.01 * self.ta + 0.36
        _ts_bur08_night_spar = 1.01 * self.ta - 0.17
        ts_spar = pd.Series(index=_ts_bur08_day_spar.index)
        ts_spar.loc[self.daytime == 1] = _ts_bur08_day_spar  # Use daytime Ts in daytime data rows
        ts_spar.loc[self.daytime == 0] = _ts_bur08_night_spar  # Use nighttime Ts in nighttime data rows
        # Calculate sigma around spar
        l_spar = 0.005  # Diameter of the spar (m)
        sigma_spar = 0.0058 * np.sqrt(l_spar / self.u)
        # Calculate spar sensible heat
        r_spar = 0.0025  # Radius of the spar cylinder (m)
        a = ts_spar - self.ta
        b = r_spar * np.log((r_spar + sigma_spar) / r_spar)
        S_spar = self.k_air * (a / b)

        # Save specific surfaces to self for plotting
        self.ts_top = ts_top
        self.ts_top.name = "Ts_Top"

        self.ts_bottom = ts_bottom
        self.ts_bottom.name = "Ts_Bottom"

        self.ts_spar = ts_spar
        self.ts_spar.name = "Ts_Spar"

        # Calculate sensible heat from all key instrument surfaces
        S = S_bottom + S_top + 0.15 * S_spar  # W m-2

        # Calculate unscaled flux correction term
        # self.gas_density is in [µmol m-3]
        # Result fct_unsc is in [µmol m-2 s-1]
        fct_unsc = (S / (self.rho_a * self.c_p)) * (self.gas_density / (self.ta + 273.15))

        # print(f"Ts (BUR08), mean = {fct_unsc.mean():.2f}")
        return fct_unsc, S

    def _flux_correction_term_unscaled_jar09_bur06(self, ts: pd.Series) -> pd.Series:
        """
        Calculate unscaled flux correction term.
        Equation (8) in Burba et al. (2006).
        Used by JAR09 and BUR06.

        Args:
            ts: series, instrument surface temperature (°C)
        """
        # Convert temperatures to Kelvin for the denominator
        ta_k = self.ta + 273.15

        # Term A: Surface - Air temp delta * CO2 density
        # gas_density in [µmol m-3] -> term_a in [K * µmol m-3]
        term_a = (ts - self.ta) * self.gas_density

        # Term B: Aerodynamic resistance * Temp
        term_b = self.ra * ta_k

        # Term C: Water vapor correction factor
        term_c = 1 + 1.6077 * (self.rho_v / self.rho_d)

        return (term_a / term_b) * term_c

    def _estimate_surface_temp_bur06(self) -> pd.Series:
        """
        Estimates the surface temperature based on the provided air temperature
        values using the BUR06 model. The BUR06 model applies a quadratic
        relationship to approximate the surface temperature from air temperature.

        Returns:
            pd.Series: Estimated surface temperature values calculated using the
            BUR06 model.
        """
        ts = 0.0025 * self.ta ** 2 + 0.9 * self.ta + 2.07
        return ts

    def _estimate_surface_temp_jar09(self) -> pd.Series:
        """
        Estimates surface temperature using the JAR09 method.

        This method calculates surface temperature based on ambient temperature and
        a binary series representing daytime or nighttime conditions. It applies
        different linear relationships for daytime and nighttime data.

        Returns:
            pd.Series: A Pandas Series representing the estimated surface temperatures.
        """
        ts = pd.Series(index=self.ta.index, dtype=float)

        # Daytime relation
        ts.loc[self.daytime == 1] = 0.93 * self.ta.loc[self.daytime == 1] + 3.17
        # Nighttime relation
        ts.loc[self.daytime == 0] = 1.05 * self.ta.loc[self.daytime == 0] + 1.52

        return ts

    def plot_diel_cycles(self):
        """
        Plots the mean diurnal cycles.
        Auto-detects if BUR08 surfaces exist and plots them individually.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as ticker
        import numpy as np

        print("Plotting Physics Diel Cycles...")

        # Check if BUR08 surfaces exist
        has_bur08 = (self.ts_bottom is not None)

        # Create plot dataframe
        frame = {
            'SWIN': self.swin_pot, 'Ta': self.ta, 'U': self.u, 'Ra': self.ra,
            'FCT_unsc': self.fct_unsc_gf if not self.fct_unsc_gf.empty else self.fct_unsc
        }

        if has_bur08:
            frame.update({
                'Ts_Top': self.ts_top,
                'Ts_Bottom': self.ts_bottom,
                'Ts_Spar': self.ts_spar
            })
            # For heating, we use the weighted average or main window delta
            frame['Delta_T'] = self.ts_bottom - self.ta
        else:
            frame['Ts'] = self.ts
            if 'Ts' in frame and 'Ta' in frame:
                frame['Delta_T'] = self.ts - self.ta
            else:
                frame['Delta_T'] = np.nan

        plot_df = pd.DataFrame(frame)
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            plot_df.index = pd.to_datetime(plot_df.index)

        # --- CONFIGURATION ---
        plot_vars = [
            {'col': 'SWIN', 'title': '1. Incoming Radiation', 'unit': r'$W\ m^{-2}$'},
            {'col': 'Ta', 'title': '2. Air Temperature', 'unit': r'$^{\circ}C$'},
            # Panel 3 is special (Multi-line for BUR08)
            {'col': 'Ts_Multi' if has_bur08 else 'Ts',
             'title': '3. Instrument Surfaces (BUR08)' if has_bur08 else '3. Surface Temp',
             'unit': r'$^{\circ}C$'},
            {'col': 'Delta_T', 'title': r'4. Heating (Main Surf - Ta)', 'unit': r'$\Delta^{\circ}C$'},
            {'col': 'U', 'title': '5. Wind Speed', 'unit': r'$m\ s^{-1}$'},
            {'col': 'Ra', 'title': '6. Aerodynamic Res.', 'unit': r'$s\ m^{-1}$'},
            {'col': 'FCT_unsc', 'title': '7. Unscaled Correction', 'unit': r'$\mu mol$'},
            {'col': None}
        ]

        # Colors
        colors = ['#f1c40f', '#e67e22', '#d35400', '#c0392b', '#3498db', '#9b59b6', '#2c3e50']

        # Settings
        rc_settings = {'font.family': 'sans-serif', 'font.size': 11}

        with plt.rc_context(rc_settings):
            fig = plt.figure(figsize=(22, 12), constrained_layout=True)
            gs = gridspec.GridSpec(2, 4, figure=fig)
            axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]

            for i, var in enumerate(plot_vars):
                ax = axes[i]
                col_name = var.get('col')
                if col_name is None:
                    ax.axis('off');
                    continue

                # --- SPECIAL HANDLING FOR BUR08 SURFACES (PANEL 3) ---
                if col_name == 'Ts_Multi':
                    # Plot annual means of all 3 surfaces + Ta

                    # 1. Ta (Reference)
                    mean_ta = plot_df['Ta'].groupby(plot_df.index.hour).mean()
                    ax.plot(mean_ta.index, mean_ta.values, color='black', ls=':', lw=1.5, label='Ta (Air)')

                    # 2. Surfaces
                    surfs = [('Ts_Bottom', '#c0392b'), ('Ts_Top', '#e67e22'), ('Ts_Spar', '#8e44ad')]
                    for s_col, s_color in surfs:
                        mean_s = plot_df[s_col].groupby(plot_df.index.hour).mean()
                        ax.plot(mean_s.index, mean_s.values, color=s_color, lw=2, label=s_col.split('_')[1])

                    ax.legend(fontsize=9, frameon=False, loc='best')
                    ax.set_title(var['title'], fontweight='bold', loc='left')
                    ax.set_ylabel(var['unit'])
                    ax.set_xlim(0, 23)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
                    continue

                # --- STANDARD PLOTTING ---
                series = plot_df[col_name].dropna()
                if series.empty: continue

                diel = series.groupby([series.index.month, series.index.hour]).mean().unstack()
                annual = series.groupby(series.index.hour).mean()

                ax.plot(annual.index, annual.values, color='gray', ls='--', lw=2, alpha=0.5, zorder=1)

                # Monthly lines
                cmap = plt.get_cmap('Spectral_r', 12)
                for m in diel.index:
                    ax.plot(diel.columns, diel.loc[m], color=cmap((m - 0.5) / 12), alpha=0.8)

                ax.set_title(var['title'], fontweight='bold', loc='left')
                ax.set_ylabel(var['unit'], color='#7f8c8d')
                ax.set_xlim(0, 23)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(6))

                if series.min() < 0 < series.max(): ax.axhline(0, lw=1, color='k')

            fig.suptitle(f"Physics Drivers ({self.flux_type})", fontsize=16, fontweight='bold', y=1.02)
            plt.show()


class ScopOptimizer:
    """
    Optimizes scaling factors using Block Bootstrapping and SciPy minimization.
    """

    def __init__(self,
                 flux_type: FluxType,
                 class_var: pd.Series,
                 n_classes: int,
                 fct_unsc: pd.Series,
                 daytime: pd.Series,
                 n_bootstrap_runs: int,
                 flux_openpath: pd.Series,
                 flux_closedpath: pd.Series,
                 latent_heat_vaporization: Optional[pd.Series] = None):
        """
                Args:
                    flux_type: "CO2" or "H2O"
                    latent_heat_vaporization: Series in [J / µmol].
                                              Required if flux_type="H2O" to convert
                                              correction term to Watts.
                """

        self.flux_type = flux_type
        self.n_classes = n_classes
        self.n_bootstrap = n_bootstrap_runs
        self.cols = ColumnConfig()

        # UNIT MATCHING
        # fct_unsc is always [µmol m-2 s-1]
        # If H2O, fluxes (LE) are in [W m-2]. Must convert FCT to W.
        if self.flux_type == "H2O":
            if latent_heat_vaporization is None:
                raise ValueError("latent_heat_vaporization required for H2O flux optimization")
            # [µmol m-2 s-1] * [J / µmol] = [J m-2 s-1] = [W m-2]
            _fct_for_opt = fct_unsc * latent_heat_vaporization
        else:
            # CO2: [µmol m-2 s-1] matches [µmol m-2 s-1]. No change.
            _fct_for_opt = fct_unsc

        # Combine inputs into a single DataFrame for easier grouping
        self.df = pd.DataFrame({
            'class_var': class_var,
            'target': flux_openpath,
            'reference': flux_closedpath,
            'fct_unsc': fct_unsc,
            self.cols.daytime: daytime
        })

        self.scaling_factors_df = pd.DataFrame()

    def run(self) -> pd.DataFrame:
        """
        Executes the optimization routine.
        Uses list accumulation for speed instead of DataFrame.loc assignment.
        """
        results = []

        # Group by daytime
        for daytime, day_group in self.df.groupby(self.cols.daytime):

            # Create bins (quantiles)
            # 'duplicates=drop' handles cases where data is constant/sparse
            try:
                day_group['bin'] = pd.qcut(day_group['class_var'], self.n_classes, labels=False, duplicates='drop')
            except ValueError:
                # Fallback to linear cut if qcut fails
                day_group['bin'] = pd.cut(day_group['class_var'], self.n_classes, labels=False)

            # 3. Iterate through Bins
            for bin_id, bin_group in day_group.groupby('bin'):
                if bin_group.empty:
                    continue

                # Drop NaNs upfront for this bin
                # (We only need rows where all vars are present)
                valid_bin = bin_group.dropna()
                if len(valid_bin) < 10:
                    continue

                # Prepare numpy arrays (for speed)
                arr_target = valid_bin['target'].values
                arr_ref = valid_bin['reference'].values
                arr_fct = valid_bin['fct_unsc'].values
                n_samples = len(valid_bin)

                # If n_bootstrap == 0, run once on raw data
                runs = self.n_bootstrap if self.n_bootstrap > 0 else 1
                factors = []
                sos_vals = []

                for _ in range(runs):
                    # Sampling
                    if self.n_bootstrap > 0:
                        # Circular block bootstrap (preserves time correlation)
                        # Block size 12 approx 6 hours (if 30min data)
                        indices = self._block_bootstrap_indices(n_samples, block_size=12)
                        s_target, s_ref, s_fct = arr_target[indices], arr_ref[indices], arr_fct[indices]
                    else:
                        # No bootstrapping
                        s_target, s_ref, s_fct = arr_target, arr_ref, arr_fct

                    # Optimization (SciPy)
                    res = minimize_scalar(
                        self._cost_function_numpy,
                        args=(s_fct, s_target, s_ref),
                        method='bounded', bounds=(0, 50.0))

                    factors.append(res.x)
                    sos_vals.append(res.fun)

                results.append({
                    'DAYTIME': daytime,
                    'GROUP_CLASSVAR': bin_id,
                    'GROUP_CLASSVAR_MIN': valid_bin['class_var'].min(),
                    'GROUP_CLASSVAR_MAX': valid_bin['class_var'].max(),
                    'BOOTSTRAP_RUNS': self.n_bootstrap,
                    'SF_MEDIAN': np.median(factors),
                    'SF_Q25': np.percentile(factors, 25),
                    'SF_Q75': np.percentile(factors, 75),
                    'SF_Q01': np.percentile(factors, 1),
                    'SF_Q99': np.percentile(factors, 99),
                    'SOS_MEDIAN': np.median(sos_vals),
                    'NUMVALS_AVG': n_samples
                })

                print(f"Finished group {bin_id} (Daytime {daytime}): Median SF = {np.median(factors):.3f}")

        # Create dataframe
        self.scaling_factors_df = pd.DataFrame(results)
        return self.scaling_factors_df

    @staticmethod
    def _cost_function_numpy(factor, fct_arr, target_arr, ref_arr):
        """
        Vectorized Cost Function using NumPy.
        Objective: Minimize L1 norm of the difference between cumulative sums.
        """
        corrected = target_arr + (fct_arr * factor)
        # diff = Cumulative(Corrected) - Cumulative(Reference)
        diff = np.cumsum(corrected) - np.cumsum(ref_arr)
        return np.sum(np.abs(diff))

    @staticmethod
    def _block_bootstrap_indices(n: int, block_size: int) -> np.ndarray:
        """
        Generates indices for circular block bootstrapping using vectorized NumPy operations.

        Args:
            n: Total number of data points.
            block_size: Length of each contiguous block.

        Returns:
            np.ndarray: An array of resampled indices of length n.
        """
        # Calculate how many blocks are needed to cover the array
        num_blocks = int(np.ceil(n / block_size))

        # Randomly select start positions for each block
        start_indices = np.random.randint(0, n, size=num_blocks)

        # Create a 2D array of indices using broadcasting
        # Shape (num_blocks, 1) + Shape (1, block_size) -> Shape (num_blocks, block_size)
        # logic: start_index + [0, 1, 2, ... block_size-1]
        offsets = np.arange(block_size)
        indices_2d = start_indices[:, None] + offsets[None, :]

        # Flatten the 2D array into a 1D array
        indices_flat = indices_2d.flatten()

        # Handle circularity and trim
        # Modulo n (%) ensures that if a block goes past the end, it wraps to the start
        indices_circular = indices_flat % n

        # Trim to the exact original length n (since num_blocks * block_size >= n)
        return indices_circular[:n]

    def plot(self):
        """Plots optimized scaling factors with confidence intervals."""
        if self.scaling_factors_df.empty:
            return

        print("Plotting Scaling Factors...")
        plot_df = self.scaling_factors_df.copy()

        styles = {
            1: {'color': '#d35400', 'label': 'Daytime', 'marker': 'o'},
            0: {'color': '#2980b9', 'label': 'Nighttime', 'marker': 's'}
        }

        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

        for day_flag, group in plot_df.groupby('DAYTIME'):
            group = group.sort_values('GROUP_CLASSVAR_MIN')
            style = styles.get(day_flag, {'color': 'k', 'label': '?', 'marker': 'x'})

            x = group['GROUP_CLASSVAR_MIN']

            # Confidence Intervals
            ax.fill_between(x, group['SF_Q01'], group['SF_Q99'], color=style['color'], alpha=0.1, edgecolor='none')
            ax.fill_between(x, group['SF_Q25'], group['SF_Q75'], color=style['color'], alpha=0.25, edgecolor='none')

            # Median
            ax.plot(x, group['SF_MEDIAN'], color=style['color'], marker=style['marker'],
                    lw=2, label=f"{style['label']} (N={group['NUMVALS_AVG'].mean():.0f})")

        ax.set_title("Scaling Factors vs Class Variable", fontsize=14, fontweight='bold', loc='left', color='#333333')
        ax.set_ylabel(r"Scaling Factor ($\xi$)", fontsize=11)
        ax.set_xlabel("Class Variable", fontsize=11)
        ax.axhline(0, lw=1, color='k', linestyle='-', zorder=0)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=False)

        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.show()

    def stats(self):
        """
        Prints a professional summary of the optimization results,
        including global averages and per-bin details.
        """
        if self.scaling_factors_df.empty:
            print("(!) No optimization results found. Run .run() first.")
            return

        df = self.scaling_factors_df

        # Helper for separators
        def print_sep(char='-', length=75):
            print(char * length)

        print("\n")
        print_sep('=', 75)
        print(f"{'SCALING FACTOR OPTIMIZATION REPORT':^75}")
        print_sep('=', 75)

        print(f"Flux Type      : {self.flux_type}")
        print(f"Bootstrap Runs : {self.n_bootstrap}")
        print(f"Total Bins     : {len(df)}")
        print_sep('-', 75)

        # --- 1. GLOBAL SUMMARY (Day vs Night) ---
        print(f"{'1. GLOBAL SUMMARY (Median of Bins)':<40}")
        print_sep('.', 75)
        print(f"{'Period':<15} | {'Median SF':>10} | {'Mean Uncertainty (IQR)':>22} | {'N_Avg':>8}")

        summary_groups = df.groupby('DAYTIME')
        for daytime, group in summary_groups:
            period = "Daytime" if daytime == 1 else "Nighttime"
            median_sf = group['SF_MEDIAN'].median()
            mean_iqr = (group['SF_Q75'] - group['SF_Q25']).mean()
            mean_n = group['NUMVALS_AVG'].mean()

            print(f"{period:<15} | {median_sf:>10.3f} | {mean_iqr:>22.3f} | {mean_n:>8.0f}")

        print("\n")

        # --- 2. DETAILED BIN REPORT ---
        print(f"{'2. DETAILED BIN BREAKDOWN':<40}")
        print_sep('.', 75)

        # Header
        # Range column needs space for "123.45 - 123.45"
        header = (f"{'Bin':<4} | {'Class Range':^19} | {'N':>5} | "
                  f"{'Median SF (ξ)':>16} | {'99% CI':^16} | {'Error (SOS)':>10}")
        print(header)
        print_sep('-', 75)

        # Iterate Day then Night
        for daytime in [1, 0]:
            subset = df[df['DAYTIME'] == daytime].sort_values('GROUP_CLASSVAR_MIN')
            if subset.empty:
                continue

            period_label = "DAYTIME" if daytime == 1 else "NIGHTTIME"
            print(f"--- {period_label} ---")

            for _, row in subset.iterrows():
                # Format the range string
                r_min, r_max = row['GROUP_CLASSVAR_MIN'], row['GROUP_CLASSVAR_MAX']
                range_str = f"{r_min:>7.2f} - {r_max:<7.2f}"

                # Format CI string
                ci_str = f"[{row['SF_Q01']:>4.2f}-{row['SF_Q99']:<4.2f}]"

                print(f"{int(row['GROUP_CLASSVAR']):<4} | "
                      f"{range_str:^19} | "
                      f"{int(row['NUMVALS_AVG']):>5} | "
                      f"{row['SF_MEDIAN']:>16.3f} | "
                      f"{ci_str:^16} | "
                      f"{row['SOS_MEDIAN']:>10.2f}")
            print("")  # Spacer between day/night

        print_sep('=', 75)
        print("\n")

    def get(self) -> pd.DataFrame:
        return self.scaling_factors_df


class ScopApplicator:

    def __init__(self,
                 flux_type: FluxType,
                 fct_unsc: pd.Series,
                 scaling_factors_df: pd.DataFrame,
                 flux_openpath: pd.Series,
                 classvar: pd.Series,
                 daytime: pd.Series,
                 latent_heat_vaporization: Optional[pd.Series] = None):
        """
        Args:
            flux_type: "CO2" or "H2O"
            latent_heat_vaporization: Series of Lambda (J/mmol or J/kg depending on inputs).
                                      Required ONLY if flux_type="H2O" and fluxes are in W m-2
                                      but correction is in molar units.
        """
        self.flux_type = flux_type
        self.fct_unsc = fct_unsc.copy()
        self.scaling_factors_df = scaling_factors_df.copy()
        self.flux_openpath = flux_openpath.copy()
        self.classvar = classvar.copy()
        self.daytime = daytime.copy()
        self.latent_heat_vaporization = latent_heat_vaporization

        self.cols = ColumnConfig()

        # Determine output column name
        prefix = 'LE' if self.flux_type == 'H2O' else 'NEE'
        self.col_flux_corr = f"{prefix}{self.cols.flux_corr_suffix}"

        frame = {self.fct_unsc.name: self.fct_unsc, self.flux_openpath.name: self.flux_openpath,
                 self.classvar.name: self.classvar, self.daytime.name: self.daytime}
        self.df = pd.DataFrame(frame, index=self.flux_openpath.index)

        # Add Lv if needed for H2O conversion
        if self.latent_heat_vaporization is not None:
            self.df['Lv'] = self.latent_heat_vaporization

    def get_results(self) -> pd.DataFrame:
        return self.df

    def run(self):

        # Assign scaling factors depending on the class var (e.g. USTAR)
        self.df = self._assign_scaling_factors()

        # Calculate final flux correction term
        # Corrected OP = uncorrected OP + (FCT_unscaled * ScalingFactor)
        fct_molar = self.df[self.cols.fct_unsc_gf] * self.df[self.cols.sf]

        # Apply unit conversion if H2O (Watts)
        if self.flux_type == "H2O" and 'Lv' in self.df.columns:
            # [µmol m-2 s-1] * [J / µmol] = [J m-2 s-1] = [W m-2]
            self.df[self.cols.fct] = fct_molar * self.df['Lv']
        else:
            # CO2 (or H2O if already molar)
            self.df[self.cols.fct] = fct_molar

        # Apply correction
        self.df[self.col_flux_corr] = self.df[self.flux_openpath.name] + self.df[self.cols.fct]
        # self.df[self.cols.fct] = self.df[self.cols.fct_unsc_gf] * self.df[self.cols.sf]
        # self.df[self.cols.flux_corr_suffix] = self.df[self.flux_openpath.name] + self.df[self.cols.fct]

    def _assign_scaling_factors(self) -> pd.DataFrame:

        # --- 1. STANDARD LOOKUP (Primary Method) ---
        # Uses actual class variable (e.g. USTAR) to look up the Scaling Factor

        valid_mask = self.classvar.notna() & self.daytime.notna()

        if valid_mask.sum() > 0:
            df_valid = self.df.loc[valid_mask].copy()

            # Sort copy by USTAR for the merge_asof
            df_valid = df_valid.sort_values(by=self.classvar.name)

            sf_sorted = self.scaling_factors_df.sort_values(by='GROUP_CLASSVAR_MIN')

            # Merge based on USTAR bins (Backward search)
            merged_subset = pd.merge_asof(
                df_valid,
                sf_sorted[['DAYTIME', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR', 'SF_MEDIAN']],
                left_on=self.classvar.name,
                right_on='GROUP_CLASSVAR_MIN',
                by=self.daytime.name,
                direction='backward'
            )

            # Handle Low Outliers (< lowest bin)
            if merged_subset['SF_MEDIAN'].isna().any():
                merged_subset['SF_MEDIAN'] = merged_subset.groupby(self.daytime.name)['SF_MEDIAN'].transform(
                    lambda x: x.bfill())
                merged_subset['GROUP_CLASSVAR'] = merged_subset.groupby(self.daytime.name)['GROUP_CLASSVAR'].transform(
                    lambda x: x.bfill())

            # Assign found SFs back to main DataFrame
            # Note: self.df is NOT sorted here, we use the index to map values back
            self.df.loc[df_valid.index, self.cols.class_var_group] = merged_subset['GROUP_CLASSVAR'].values
            self.df.loc[df_valid.index, self.cols.sf] = merged_subset['SF_MEDIAN'].values

        # --- 2. GAP FILLING: MEDIAN DIURNAL VARIATION (MDV) ---

        missing_sf_mask = self.df[self.cols.sf].isna()
        # missing_sf_mask = self.df[self.flux_openpath.name].notna() & self.df[self.cols.sf].isna()

        if missing_sf_mask.any():
            n_missing = missing_sf_mask.sum()
            print(f"(!) Warning: {n_missing} fluxes missing Scaling Factor (due to missing {self.classvar.name}).")
            print("    Imputing using Month-Daytime-Hour-Minute Diel Cycle Median...")

            # --- KEY CHANGE: USE ORIGINAL FLUX INDEX ---
            # We access the index directly from the original input series stored in self.
            # This ensures strict alignment with the input data structure.
            idx = self.flux_openpath.index

            if not isinstance(idx, pd.DatetimeIndex):
                # Fallback if input series wasn't a time series (unlikely in this context)
                idx = pd.to_datetime(idx)

            # A. PREPARE KEYS USING ORIGINAL INDEX
            # We ensure self.df aligns with this index before assigning columns
            # (In standard usage, self.df already shares this index, but this is explicit safety)
            self.df['__MONTH'] = idx.month
            self.df['__HOUR'] = idx.hour
            self.df['__MINUTE'] = idx.minute

            # B. DEFINE GROUPER
            grouper = [
                self.df['__MONTH'],
                self.df[self.daytime.name],
                self.df['__HOUR'],
                self.df['__MINUTE']
            ]

            # C. CREATE LOOKUP TABLE
            sf_lut_series = self.df.groupby(grouper)[self.cols.sf].median()

            sf_lut_df = sf_lut_series.reset_index()
            sf_lut_df.columns = ['__MONTH', self.daytime.name, '__HOUR', '__MINUTE', 'SF_MEDIAN_LUT']

            # D. MERGE LUT BACK
            # We use reset_index() to preserve the time index during the merge
            df_temp = self.df.reset_index()

            # Merge on the calculated time columns
            df_merged = df_temp.merge(
                sf_lut_df,
                on=['__MONTH', self.daytime.name, '__HOUR', '__MINUTE'],
                how='left'
            )

            # Restore the index from the temp dataframe
            df_merged.set_index(df_temp.columns[0], inplace=True)
            self.df['SF_MEDIAN_LUT'] = df_merged['SF_MEDIAN_LUT']

            # E. FILL GAPS
            self.df.loc[missing_sf_mask, self.cols.sf] = self.df.loc[missing_sf_mask, self.cols.sf].fillna(
                self.df.loc[missing_sf_mask, 'SF_MEDIAN_LUT']
            )

            # F. FALLBACK (Edge Cases)
            # Check what is still missing after applying the fine-grained LUT
            remaining_missing = self.df[self.flux_openpath.name].notna() & self.df[self.cols.sf].isna()

            if remaining_missing.any():
                print(
                    f"    (!) {remaining_missing.sum()} items still NaN (Exact Month-Hour-Minute combo never observed).")

                # --- DIAGNOSTIC: PRINT MISSING COMBINATIONS ---
                print("    [DEBUG] The following (Month, Daytime, Hour, Minute) combinations are missing from the LUT:")

                # Select the relevant columns for the missing rows
                missing_combos = self.df.loc[remaining_missing, ['__MONTH', self.daytime.name, '__HOUR', '__MINUTE']]

                # Count how many times each specific missing combination occurs
                # This returns a readable Series indexed by the combinations
                missing_counts = missing_combos.value_counts().sort_index()

                # Print the list (Month, Daytime, Hour, Minute) -> Count of missing records
                print(missing_counts)
                print("    -----------------------------------------------------------------------------------------")
                # ----------------------------------------------

                print("    -> Filling edge cases with global Month-Daytime median.")

                # Coarser fallback: Just Month + Daytime (ignoring exact hour/minute)
                coarse_grouper = [self.df['__MONTH'], self.df[self.daytime.name]]
                sf_coarse = self.df.groupby(coarse_grouper)[self.cols.sf].transform('median')
                self.df.loc[remaining_missing, self.cols.sf] = self.df.loc[remaining_missing, self.cols.sf].fillna(
                    sf_coarse)

            # G. CLEANUP
            cols_to_drop = ['__MONTH', '__HOUR', '__MINUTE', 'SF_MEDIAN_LUT']
            self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns], inplace=True)

            n_filled = n_missing - (self.df[self.flux_openpath.name].notna() & self.df[self.cols.sf].isna()).sum()
            print(f"    > Successfully imputed {n_filled} missing Scaling Factors.")

        return self.df.sort_index()

    def _gapfilling_lut(self, series):
        """Gap-fill time series using look-up table (LUT)

        """
        # 1. Calculate the mean for every (Month, Hour) group
        # 'transform' calculates the mean and broadcasts it back to the original index size
        lutvals = series.groupby([series.index.month, series.index.hour]).transform('mean')

        # 2. Fill the gaps in the original series using these means
        series_gf = series.fillna(lutvals)

        return series_gf, lutvals

    def stats(self, flux_closedpath: Optional[pd.Series] = None):
        """
        Prints a detailed report including:
        1. Assignment Method (Direct Lookup vs. MDV Imputation).
        2. Correction Magnitude (Shift in fluxes).
        3. Accuracy Metrics (if Reference is provided).
        """
        import numpy as np

        # --- 1. SAFETY CHECKS ---
        if self.col_flux_corr not in self.df.columns:
            print("(!) No corrected data found. Run .run() first.")
            return

        # Prepare Reference if available
        cp_col = None
        if flux_closedpath is not None:
            cp_col = flux_closedpath.name if flux_closedpath.name else 'flux_closedpath'
            self.df[cp_col] = flux_closedpath

        # Define Columns
        col_op = self.flux_openpath.name
        col_corr = self.col_flux_corr
        col_sf = self.cols.sf
        col_key = self.classvar.name

        # --- 2. ASSIGNMENT STATISTICS ---
        # Logic:
        # - Direct: Flux exists AND ClassVar (USTAR) exists.
        # - Imputed: Flux exists BUT ClassVar is missing (so we used MDV).

        # Filter to rows where we actually have Open-Path data
        op_mask = self.df[col_op].notna()
        df_active = self.df.loc[op_mask]

        n_total = len(df_active)
        n_direct = df_active[col_key].notna().sum()
        n_imputed = n_total - n_direct

        pct_direct = (n_direct / n_total) * 100
        pct_imputed = (n_imputed / n_total) * 100

        # --- 3. SCALING FACTOR STATISTICS ---
        mean_sf_day = df_active.loc[df_active[self.cols.daytime] == 1, col_sf].median()
        mean_sf_night = df_active.loc[df_active[self.cols.daytime] == 0, col_sf].median()

        # --- 4. FLUX MAGNITUDE STATS ---
        sum_raw = df_active[col_op].sum()
        sum_corr = df_active[col_corr].sum()
        net_change = sum_corr - sum_raw

        # Helper for printing
        def print_sep(char='-', length=75):
            print(char * length)

        # ================= REPORT =================
        print("\n")
        print_sep('=', 75)
        print(f"{'FLUX CORRECTION REPORT':^75}")
        print_sep('=', 75)
        print(f"Flux Type      : {self.flux_type}")
        print(f"Total Records  : {n_total:,}")
        print_sep('-', 75)

        # SECTION A: ASSIGNMENT DETAILS
        print(f"{'1. SCALING FACTOR ASSIGNMENT':<40} {'(Method Used)':>34}")
        print_sep('.', 75)

        print(f"   Direct Lookup (Valid {col_key}) : {n_direct:>8,}  ({pct_direct:>5.1f}%)")

        # Highlight Imputation
        imp_symbol = "!" if pct_imputed > 10 else "i"
        print(f" {imp_symbol} Gap-Filled    (MDV Imputation) : {n_imputed:>8,}  ({pct_imputed:>5.1f}%)")

        print(f"\n   Median Scaling Factor (Day)     : {mean_sf_day:.3f}")
        print(f"   Median Scaling Factor (Night)   : {mean_sf_night:.3f}")
        print("\n")

        # SECTION B: BUDGET IMPACT
        print(f"{'2. CORRECTION IMPACT (Budget)':<40} {'(Units: Sum)':>34}")
        print_sep('.', 75)
        print(f"   Uncorrected Sum                 : {sum_raw:>15,.0f}")
        print(f"   Corrected Sum                   : {sum_corr:>15,.0f}")
        print(f"   Net Adjustment                  : {net_change:>15,.0f}  ({(net_change / abs(sum_raw)) * 100:+.1f}%)")

        # SECTION C: ACCURACY (Only if Ref provided)
        if cp_col:
            sum_ref = df_active[cp_col].sum()
            resid = df_active[col_corr] - df_active[cp_col]
            rmse = np.sqrt((resid ** 2).mean())
            slope, _ = np.polyfit(df_active[cp_col].fillna(0), df_active[col_corr].fillna(0), 1)

            print("\n")
            print(f"{'3. ACCURACY vs REFERENCE':<40}")
            print_sep('.', 75)
            print(f"   Reference Sum                   : {sum_ref:>15,.0f}")
            print(f"   Recovery (Corrected / Ref)      : {(sum_corr / sum_ref) * 100:>14.1f}%")
            print(f"   RMSE                            : {rmse:>15.3f}")
            print(f"   Slope (m)                       : {slope:>15.3f}")

        print_sep('=', 75)
        print("\n")

    def plot_dashboard(self, flux_closedpath: Optional[pd.Series] = None):
        """
        Generates a master dashboard with INCREASED FONT SIZES for better readability.
        """

        print("Generating Comprehensive Flux Dashboard (Large Font Edition)...")

        # --- 1. DATA PREPARATION ---
        if self.col_flux_corr not in self.df.columns:
            print(f"Warning: '{self.col_flux_corr}' not found. Run .run() first.")
            return

        plot_df = self.df.copy()

        if not isinstance(plot_df.index, pd.DatetimeIndex):
            try:
                plot_df.index = pd.to_datetime(plot_df.index)
            except Exception as e:
                raise TypeError("Index must be DatetimeIndex.") from e

        cp_col_name = None
        if flux_closedpath is not None:
            cp_col_name = flux_closedpath.name if flux_closedpath.name else 'flux_closedpath'
            plot_df[cp_col_name] = flux_closedpath

        OP_COL = self.flux_openpath.name
        CORR_COL = self.col_flux_corr
        DIFF_COL = 'diff_corr_ref'

        if cp_col_name and cp_col_name in plot_df.columns:
            plot_df[DIFF_COL] = plot_df[CORR_COL] - plot_df[cp_col_name]
        else:
            plot_df[DIFF_COL] = np.nan

        # --- 2. GLOBAL STYLE CONFIG ---
        STYLE = {
            OP_COL: {'label': 'Uncorrected (OP)', 'color': '#95a5a6', 'zorder': 1},
            cp_col_name: {'label': 'Reference (CP)', 'color': '#2c3e50', 'zorder': 2},
            CORR_COL: {'label': 'Corrected (OP)', 'color': '#e74c3c', 'zorder': 3},
            DIFF_COL: {'label': 'Residual', 'color': '#8e44ad', 'zorder': 2}
        }

        # --- FONT CONFIGURATION ---
        # We use a context manager to set global defaults larger,
        # then specific elements are tweaked further below.
        rc_settings = {
            'font.family': 'sans-serif',
            'font.size': 14,  # Base font size (was ~10)
            'axes.titlesize': 16,  # Subplot titles
            'axes.labelsize': 14,  # Axis labels
            'xtick.labelsize': 12,  # Ticks
            'ytick.labelsize': 12,
            'legend.fontsize': 13,
            'figure.titlesize': 20
        }

        with plt.rc_context(rc_settings):

            # --- 3. LAYOUT SETUP ---
            fig = plt.figure(figsize=(24, 20), constrained_layout=True)
            gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.2, 0.8, 1, 1])

            ax_ts = fig.add_subplot(gs[0, :])
            ax_day = fig.add_subplot(gs[1, 0])
            ax_night = fig.add_subplot(gs[1, 1])
            ax_all = fig.add_subplot(gs[1, 2:])

            diel_axes = []
            for r in [2, 3]:
                for c in range(4):
                    diel_axes.append(fig.add_subplot(gs[r, c]))

            common_date_fmt = mdates.DateFormatter('%b')

            # =================================================================
            # SECTION A: TIME SERIES (Row 0)
            # =================================================================
            cols_ts = [OP_COL, cp_col_name, CORR_COL]
            for col in cols_ts:
                if col not in plot_df.columns or col is None: continue
                s = STYLE.get(col, {})

                # Scatter
                ax_ts.scatter(plot_df.index, plot_df[col], s=3, color=s['color'],
                              alpha=0.2, edgecolors='none', zorder=s['zorder'])
                # Trend
                rolling = plot_df[col].rolling(window=336, center=True, min_periods=1).mean()
                ax_ts.plot(rolling.index, rolling, color=s['color'], lw=2.5,
                           alpha=0.9, zorder=s['zorder'] + 10, label=s['label'])

            # Larger Title/Label
            ax_ts.set_title("A. Flux Time Series & Trends", fontsize=18, fontweight='bold', loc='left', pad=10)
            ax_ts.set_ylabel(r"$\mu mol\ m^{-2}\ s^{-1}$", fontsize=14, fontweight='medium')
            ax_ts.axhline(0, color='k', lw=1, zorder=5)
            ax_ts.legend(loc='upper left', frameon=False, ncol=3, fontsize=13, markerscale=2)
            ax_ts.set_xlim(plot_df.index.min(), plot_df.index.max())

            # =================================================================
            # SECTION B: BUDGETS (Row 1)
            # =================================================================
            scenarios = [
                {'ax': ax_day, 'title': 'B1. Daytime Budget', 'mask': plot_df[self.cols.daytime] == 1},
                {'ax': ax_night, 'title': 'B2. Nighttime Budget', 'mask': plot_df[self.cols.daytime] == 0},
                {'ax': ax_all, 'title': 'B3. Total Budget', 'mask': slice(None)}
            ]

            for scen in scenarios:
                ax = scen['ax']
                subset = plot_df.loc[scen['mask']].dropna(subset=[c for c in cols_ts if c in plot_df.columns])

                for col in cols_ts:
                    if col not in subset.columns or col is None: continue
                    s = STYLE.get(col, {})
                    cumsum = subset[col].cumsum()

                    ax.plot(subset.index, cumsum, color=s['color'], lw=3, label=s['label'])

                    # Larger Annotation
                    if not cumsum.empty:
                        ax.text(subset.index[-1], cumsum.iloc[-1], f" {cumsum.iloc[-1]:.0f}",
                                color=s['color'], fontsize=11, fontweight='bold', va='center')

                ax.set_title(scen['title'], fontsize=16, fontweight='bold', loc='left')
                ax.xaxis.set_major_formatter(common_date_fmt)
                if scen['ax'] == ax_day: ax.set_ylabel(r"Cumulative $\Sigma$", fontsize=14)
                if scen['ax'] == ax_all: ax.legend(loc='best', frameon=False, fontsize=12)

            # =================================================================
            # SECTION C: DIEL CYCLES (Rows 2 & 3)
            # =================================================================
            diel_vars = [
                {'col': self.cols.fct_unsc_gf, 'title': r'C1. Unscaled Corr. ($FCT_{unsc}$)', 'unit': r'$\mu mol$'},
                {'col': self.cols.sf, 'title': r'C2. Scaling Factor ($\xi$)', 'unit': '-'},
                {'col': self.cols.fct, 'title': 'C3. Final Correction Term', 'unit': r'$\mu mol$'},
                {'col': None},
                {'col': OP_COL, 'title': 'C4. OP Flux (Uncorrected)', 'unit': r'$\mu mol$'},
                {'col': cp_col_name, 'title': 'C5. CP Flux (Reference)', 'unit': r'$\mu mol$'},
                {'col': CORR_COL, 'title': 'C6. OP Flux (Corrected)', 'unit': r'$\mu mol$'},
                {'col': DIFF_COL, 'title': 'C7. Residual (Corrected - Ref)', 'unit': r'$\mu mol$'},
            ]

            cmap = plt.get_cmap('Spectral_r', 12)
            norm = plt.Normalize(vmin=0.5, vmax=12.5)

            for i, var in enumerate(diel_vars):
                ax = diel_axes[i]
                col_name = var.get('col')
                if col_name is None or col_name not in plot_df.columns:
                    ax.axis('off')
                    continue

                series = plot_df[col_name].dropna()
                if series.empty: continue

                diel = series.groupby([series.index.month, series.index.hour]).mean().unstack()
                annual = series.groupby(series.index.hour).mean()

                ax.plot(annual.index, annual.values, color='#7f8c8d', ls='--', lw=2, zorder=1, alpha=0.5)
                for month in diel.index:
                    ax.plot(diel.columns, diel.loc[month], color=cmap(norm(month)),
                            lw=2, alpha=0.8, zorder=2)

                # Larger Subplot Titles and labels
                ax.set_title(var['title'], fontsize=14, fontweight='bold', loc='left', pad=8)
                ax.set_ylabel(var['unit'], fontsize=11, color='#555555')
                ax.set_xlim(0, 23)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(6))  # Keeps ticks clean (0, 6, 12, 18)

                if series.min() < 0 < series.max():
                    ax.axhline(0, color='k', lw=1, zorder=0)

            # --- 4. FINISHING TOUCHES ---

            # Apply global spine formatting
            for ax in [ax_ts, ax_day, ax_night, ax_all] + diel_axes:
                if not ax.axison: continue
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.8)
                ax.spines['bottom'].set_linewidth(0.8)
                ax.grid(True, ls=':', alpha=0.5)

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=diel_axes, orientation='vertical',
                                fraction=0.02, pad=0.02, aspect=35)

            cbar.set_label('Month', rotation=270, labelpad=20, fontsize=13)
            cbar.set_ticks(range(1, 13))
            cbar.set_ticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            cbar.outline.set_visible(False)

            plt.show()


def _example():
    from diive.core.io.files import load_parquet
    df = load_parquet(
        filepath=r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\30_FLUX_PROCESSING_CHAIN\31_SELF-HEATING_CORRECTION\22_MERGED_IRGA75-noSHC+IRGA72_FluxProcessingChain_after-L3.2_NEE-QCF10_LE-QCF11_2016-2017.parquet")
    # [print(c) for c in df.columns if "RH" in c];

    # Parallel measurements starting 27 May 2016
    df = df.loc["2016-05-27 00:15:00":"2017-12-11 23:45:00"].copy()

    # Variables from EddyPro _fluxnet_ output file
    AIR_CP = "AIR_CP_IRGA72"  # air_heat_capacity (J kg-1 K-1)
    AIR_DENSITY = "AIR_DENSITY_IRGA72"  # air_density (kg m-3)
    VAPOR_DENSITY = "VAPOR_DENSITY_IRGA72"  # water_vapor_density (kg m-3)
    U = "U_IRGA72"  # (m s-1)
    USTAR = "USTAR_IRGA72"  # (m s-1)
    TA = "TA_T1_47_1_gfXG_IRGA72"  # (degC)
    SWIN = "SW_IN_T1_47_1_gfXG_IRGA72"  # (W m-2)
    RH = "RH_T1_47_1_IRGA72"  # (RH)

    FLUX_TYPE = "CO2"
    GAS_DENSITY = "CO2_MOLAR_DENSITY_IRGA75"  # (originally in mmol m-3)
    FLUX_72 = "NEE_L3.1_L3.2_QCF_IRGA72"  # (umol m-2 s-1)
    FLUX_75 = "NEE_L3.1_L3.2_QCF_IRGA75"  # (umol m-2 s-1)
    CLASSVAR = USTAR
    # FLUX_TYPE = "H2O"
    # GAS_DENSITY = "H2O_MOLAR_DENSITY_IRGA75"  # (originally in mmol m-3)
    # FLUX_72 = "LE_L3.1_L3.2_QCF_IRGA72"  # (umol m-2 s-1)
    # FLUX_75 = "LE_L3.1_L3.2_QCF_IRGA75"  # (umol m-2 s-1)
    # CLASSVAR = RH

    # Conversions
    df[GAS_DENSITY] = df[GAS_DENSITY] * 1000  # Convert to umol m-3

    tic = time.time()

    # Calculate
    physics = ScopPhysics(
        flux_type=FLUX_TYPE,
        ta=df[TA].copy(),
        gas_density=df[GAS_DENSITY].copy(),
        rho_a=df[AIR_DENSITY].copy(),
        rho_v=df[VAPOR_DENSITY].copy(),
        u=df[U].copy(),
        c_p=df[AIR_CP].copy(),
        ustar=df[USTAR].copy(),
        lat=47.478333,  # CH–LAE
        lon=8.364389,  # CH–LAE
        utc_offset=1,
    )
    physics.run(correction_method_base="JAR09", gapfill=True)
    physics.stats()
    physics.plot_diel_cycles()
    results_physics_df = physics.get_results()

    optimizer = ScopOptimizer(
        flux_type=FLUX_TYPE,
        fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
        class_var=df[USTAR].copy(),
        n_classes=5,
        n_bootstrap_runs=5,
        flux_openpath=df[FLUX_75].copy(),
        flux_closedpath=df[FLUX_72].copy(),
        daytime=results_physics_df["DAYTIME"],
        latent_heat_vaporization=results_physics_df["LATENT_HEAT_VAPORIZATION_J_UMOL"],
    )
    scaling_factors_df = optimizer.run()
    optimizer.stats()
    optimizer.plot()

    applicator = ScopApplicator(
        flux_type=FLUX_TYPE,
        fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
        scaling_factors_df=scaling_factors_df,
        flux_openpath=df[FLUX_75].copy(),
        classvar=df[CLASSVAR].copy(),
        daytime=results_physics_df["DAYTIME"].copy()
    )
    applicator.run()
    applicator.stats(flux_closedpath=df[FLUX_72].copy())
    applicator.plot_dashboard(flux_closedpath=df[FLUX_72].copy())

    toc = time.time()
    print(f"Time elapsed: {toc - tic:.2f} seconds")

    print("End.")


def _example_lae():
    import pandas as pd
    from diive.core.io.files import load_parquet
    # from diive.pkgs.flux.selfheating import ScopPhysics, ScopApplicator
    FILEPATH = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\20_MERGE_DATA\21.4_FLUXES_L1_noSHC_IRGA75+METEO7.parquet"
    print(f"Data will be loaded from the following file:\n{FILEPATH}")
    df = load_parquet(filepath=FILEPATH)
    # df = df.loc[df.index.year == 2014].copy()
    physics = ScopPhysics(
        flux_type="CO2",
        ta=df["TA_T1_47_1"].copy(),
        gas_density=df["CO2_MOLAR_DENSITY"].copy() * 1000,  # Requires umol m-3
        rho_a=df["AIR_DENSITY"].copy(),
        rho_v=df["VAPOR_DENSITY"].copy(),
        u=df["U"].copy(),
        c_p=df["AIR_CP"].copy(),
        ustar=df["USTAR"].copy(),
        lat=47.478333,  # CH–LAE
        lon=8.364389,  # CH–LAE
        utc_offset=1,
    )
    physics.run(correction_method_base="JAR09", gapfill=True)
    results_physics_df = physics.get_results()
    results_physics_df.describe()
    physics.stats()
    physics.plot_diel_cycles()
    sfdf = pd.read_csv(
        r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\30_FLUX_PROCESSING_CHAIN\31_SELF-HEATING_CORRECTION\32_SelfHeatingCorrection_ScalingFactors_NEE.csv")
    applicator = ScopApplicator(
        flux_type="CO2",
        fct_unsc=results_physics_df["FCT_UNSC_gfRF"],
        scaling_factors_df=sfdf,
        flux_openpath=df["FC"].copy(),
        classvar=df["USTAR"].copy(),
        daytime=results_physics_df["DAYTIME"].copy()
    )
    applicator.run()
    applicator.stats()
    applicator.plot_dashboard()


if __name__ == '__main__':
    _example()
    # _example_lae()
