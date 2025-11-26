import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional, List, Tuple

# --- CONFIGURATION ---
@dataclass(frozen=True)
class ColumnConfig:
    """Immutable configuration for internal column names."""
    # Internal Intermediate Columns
    daytime: str = '.DAYTIME'
    class_var_group: str = '.GROUP_CLASSVAR'

    # Physics Variables
    air_thermal_conductivity: str = '.AIR_THERMAL_CONDUCTIVITY'
    aerodynamic_resistance: str = '.AERODYNAMIC_RESISTANCE'
    dry_air_density: str = '.DRY_AIR_DENSITY'
    t_instrument_surface: str = '.T_INSTRUMENT_SURFACE'

    # Correction Terms
    fct_unsc: str = '.FCT_UNSC'          # Unscaled correction (raw physics)
    fct_unsc_gf: str = '.FCT_UNSC_GF'    # Gap-filled unscaled correction
    sf: str = '.SF'                      # Scaling factor (from LUT)
    sf_gf: str = '.SF_GF'                # Gap-filled scaling factor
    fct: str = '.FCT'                    # Final Correction Term (FCT_unsc * SF)
    nee_op_corr: str = '.NEE_OP_CORR'    # Final Corrected Flux


# --- 1. PHYSICS ENGINE ---
class SHCPhysics:
    """
    Handles the calculation of physical drivers and the unscaled correction term
    based on Burba et al. (2006/2008) and Järvi et al. (2009).
    """

    @staticmethod
    def calc_variables(df: pd.DataFrame,
                       cols: ColumnConfig,
                       u_col: str,
                       ustar_col: str,
                       ta_col: str,
                       rho_a_col: str,
                       rho_v_col: str,
                       co2_molar_col: str,
                       swin_col: str) -> pd.DataFrame:
        """
        Calculates all necessary physics variables for the correction.
        """
        # 1. Detect Daytime (Threshold > 20 W/m2)
        df[cols.daytime] = (df[swin_col] > 20).astype(int)

        # 2. Aerodynamic Resistance (Ra) [s m-1]
        # Approximation: Ra = U / u*^2
        with np.errstate(divide='ignore', invalid='ignore'):
            df[cols.aerodynamic_resistance] = df[u_col] / (df[ustar_col] ** 2)
        df[cols.aerodynamic_resistance] = df[cols.aerodynamic_resistance].replace([np.inf, -np.inf], np.nan)

        # 3. Dry Air Density (rho_d) [kg m-3]
        df[cols.dry_air_density] = df[rho_a_col] - df[rho_v_col]

        # 4. Instrument Surface Temperature (Ts) [degC]
        # Implementation of Jarvi et al. (2009)
        # Daytime: Ts = 0.93*Ta + 3.17
        # Nighttime: Ts = 1.05*Ta + 1.52
        df[cols.t_instrument_surface] = np.where(
            df[cols.daytime] == 1,
            0.93 * df[ta_col] + 3.17,
            1.05 * df[ta_col] + 1.52
        )

        # 5. Calculate Unscaled Flux Correction Term (FCT_unsc)
        # Equation 8 from Burba et al. (2006) / Eq 5 from Kittler et al. (2017)
        # Formula: [(Ts - Ta) * qc] / [Ra * (Ta_K)] * [1 + 1.6077 * (rho_v / rho_d)]

        ta_k = df[ta_col] + 273.15
        qc_umol = df[co2_molar_col] * 1000  # Convert mol/m3 to umol/m3

        term_a = (df[cols.t_instrument_surface] - df[ta_col]) * qc_umol
        term_b = df[cols.aerodynamic_resistance] * ta_k
        term_c = 1 + 1.6077 * (df[rho_v_col] / df[cols.dry_air_density])

        df[cols.fct_unsc] = (term_a / term_b) * term_c

        # 6. Basic cleaning of the unscaled term (remove extreme outliers)
        # In a real pipeline, use Hampel/MAD filter here.
        # For now, we just enforce finiteness.
        df[cols.fct_unsc] = df[cols.fct_unsc].replace([np.inf, -np.inf], np.nan)

        # 7. Gap-fill FCT_unsc (Simple interpolation for demo)
        # In production, use RandomForest or similar
        df[cols.fct_unsc_gf] = df[cols.fct_unsc].interpolate(method='time').bfill().ffill()

        return df


# --- 2. OPTIMIZER (TRAINING) ---
class SHCOptimizer:
    """
    Optimizes Scaling Factors (Sf) by comparing Open-Path fluxes to a Reference (Closed-Path).
    This creates the Lookup Table (LUT).
    """

    def __init__(self, config: ColumnConfig):
        self.cols = config

    def derive_factors(self, df: pd.DataFrame,
                       op_col: str,
                       cp_col: str,
                       classvar_col: str,
                       n_classes: int = 5,
                       n_bootstrap: int = 0) -> pd.DataFrame:
        """
        Calculates the optimal scaling factors for each bin of the class variable.
        """
        results = []

        # Ensure necessary columns exist
        if self.cols.fct_unsc_gf not in df.columns:
            raise ValueError("FCT not found. Run physics calculation first.")

        # 1. Group by Daytime
        for daytime, day_group in df.groupby(self.cols.daytime):

            # 2. Create Bins (Quantiles) for the Class Variable (e.g., USTAR)
            # Use qcut for roughly equal number of points per bin
            try:
                day_group['bin'] = pd.qcut(day_group[classvar_col], n_classes, duplicates='drop')
            except ValueError:
                # Fallback to cut if data is sparse/constant
                day_group['bin'] = pd.cut(day_group[classvar_col], n_classes)

            # 3. Process each Bin
            for bin_interval, bin_group in day_group.groupby('bin', observed=True):
                if bin_group.empty:
                    continue

                # Filter for valid data pairs
                valid_mask = bin_group[[op_col, cp_col, self.cols.fct_unsc_gf]].notna().all(axis=1)
                data = bin_group.loc[valid_mask]

                if len(data) < 10:
                    continue

                # Optimize
                # We minimize the difference between Cumulative Corrected Flux and Cumulative Reference Flux
                target = data[op_col].values
                reference = data[cp_col].values
                fct = data[self.cols.fct_unsc_gf].values

                res = minimize_scalar(
                    self._cost_function,
                    args=(fct, target, reference),
                    bounds=(-1.0, 5.0),
                    method='bounded'
                )

                results.append({
                    'DAYTIME': daytime,
                    'GROUP_CLASSVAR': f"{daytime}_{bin_interval.left:.3f}", # ID
                    'GROUP_CLASSVAR_MIN': bin_interval.left,
                    'GROUP_CLASSVAR_MAX': bin_interval.right,
                    'SF_MEDIAN': res.x,
                    'SOS_MEDIAN': res.fun,
                    'count': len(data)
                })

        return pd.DataFrame(results)

    @staticmethod
    def _cost_function(sf, fct, target, ref):
        """
        Cost function: L1 norm of the difference between cumulative sums.
        Ensures the total carbon budget matches the reference.
        """
        corrected = target + (fct * sf)
        diff = np.abs(np.cumsum(corrected) - np.cumsum(ref))
        return np.sum(diff)


# --- 3. APPLICATOR (INFERENCE) ---
class SHCApplicator:
    """
    Applies the Self-Heating Correction using an existing Lookup Table (LUT).
    This class handles the merging logic and final flux calculation.
    """

    def __init__(self, config: ColumnConfig, scaling_factors_df: pd.DataFrame):
        self.cols = config
        self.sf_df = scaling_factors_df

    def apply(self, df: pd.DataFrame, op_col: str, classvar_col: str, lut_gapfill: bool = True) -> pd.DataFrame:
        """
        Main entry point to apply correction.
        """
        # 1. Assign Scaling Factors (The complex merge logic)
        df = self._assign_scaling_factors(df, classvar_col, lut_gapfill)

        # 2. Check for missing factors
        missing_mask = df[op_col].notna() & df[self.cols.sf_gf].isna()
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} flux values have no scaling factor.")

        # 3. Calculate Final Flux
        # Corrected_Flux = Raw_OP + (FCT_unscaled * Scaling_Factor)
        correction_term = df[self.cols.fct_unsc_gf] * df[self.cols.sf_gf]
        df[self.cols.fct] = correction_term
        df[self.cols.nee_op_corr] = df[op_col] + correction_term

        return df

    def _assign_scaling_factors(self, df: pd.DataFrame, classvar_col: str, lut_gapfill: bool) -> pd.DataFrame:
        """
        Assigns scaling factors based on class variable (USTAR) and Daytime using
        backward merge_asof.

        Logic:
        - Exact Match: Daytime
        - Bin Match: USTAR >= Bin_Min (Backward search)
        - High Outliers (> Max Bin): Assigned to Last Bin (Highest Bin)
        - Low Outliers (< Min Bin): Assigned to First Bin (Lowest Bin)
        """
        # 1. Initialize cols
        df[self.cols.class_var_group] = np.nan
        df[self.cols.sf] = np.nan

        # 2. Filter for valid keys
        valid_mask = df[classvar_col].notna() & df[self.cols.daytime].notna()
        if valid_mask.sum() == 0:
            return df

        # 3. Prepare Subsets
        # Left side: Only strictly necessary cols to avoid suffix duplication
        cols_needed = [classvar_col, self.cols.daytime]
        df_valid = df.loc[valid_mask, cols_needed].sort_values(by=classvar_col)

        # Right side: Sort by Min
        sf_sorted = self.sf_df.sort_values(by='GROUP_CLASSVAR_MIN')

        # 4. Backward Merge
        # Finds the bin where [USTAR] >= [Bin_Min]
        merged_subset = pd.merge_asof(
            df_valid,
            sf_sorted[['DAYTIME', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR', 'SF_MEDIAN']],
            left_on=classvar_col,
            right_on='GROUP_CLASSVAR_MIN',
            by=self.cols.daytime,
            direction='backward'
        )

        # 5. Handle Low Outliers
        # Values < Min Bin result in NaN. We fill them with the SF of the 'First' bin
        # (the smallest bin for that Daytime).
        if merged_subset['SF_MEDIAN'].isna().any():
            # Group by daytime and backfill (taking the first valid value in the sorted order)
            merged_subset['SF_MEDIAN'] = merged_subset.groupby(self.cols.daytime)['SF_MEDIAN']\
                                                      .transform(lambda x: x.bfill())
            merged_subset['GROUP_CLASSVAR'] = merged_subset.groupby(self.cols.daytime)['GROUP_CLASSVAR']\
                                                           .transform(lambda x: x.bfill())

        # 6. Assign back to original DF
        df.loc[df_valid.index, self.cols.class_var_group] = merged_subset['GROUP_CLASSVAR'].values
        df.loc[df_valid.index, self.cols.sf] = merged_subset['SF_MEDIAN'].values

        # 7. Gapfilling (Interpolate missing USTAR times)
        if lut_gapfill:
            df[self.cols.sf_gf] = df[self.cols.sf].interpolate(method='time').bfill().ffill()
        else:
            df[self.cols.sf_gf] = df[self.cols.sf]

        return df.sort_index()


# --- 4. MAIN EXECUTION ---
def main():
    # 1. Create Dummy Data
    dates = pd.date_range('2023-01-01', periods=1000, freq='30min')
    df = pd.DataFrame(index=dates)
    df['U'] = np.random.uniform(1, 5, size=len(df))
    df['USTAR'] = np.random.uniform(0.1, 0.8, size=len(df))
    df['TA'] = np.random.uniform(5, 25, size=len(df))
    df['SWIN'] = np.where(dates.hour.isin(range(6,18)), 400, 0) # Simple day/night
    df['RHO_A'] = 1.2
    df['RHO_V'] = 0.01
    df['CO2_MOL'] = 0.015 # mol/m3

    # Fake Fluxes
    df['NEE_CP'] = np.sin(np.linspace(0, 10, len(df))) * 10 # Reference
    # OP has self-heating bias correlated with Solar Radiation
    df['NEE_OP'] = df['NEE_CP'] + (df['SWIN'] * 0.02)

    # 2. Config
    cols = ColumnConfig()

    # 3. Calculate Physics
    print("Calculating Physics...")
    df = SHCPhysics.calc_variables(
        df, cols,
        u_col='U', ustar_col='USTAR', ta_col='TA',
        rho_a_col='RHO_A', rho_v_col='RHO_V',
        co2_molar_col='CO2_MOL', swin_col='SWIN'
    )

    # 4. Train (Optimize) - Calculate Scaling Factors
    print("Optimizing Scaling Factors...")
    optimizer = SHCOptimizer(cols)
    sf_df = optimizer.derive_factors(
        df,
        op_col='NEE_OP',
        cp_col='NEE_CP',
        classvar_col='USTAR',
        n_classes=3 # Small classes for demo
    )

    print("\nDerived Scaling Factors (LUT):")
    print(sf_df[['DAYTIME', 'GROUP_CLASSVAR_MIN', 'GROUP_CLASSVAR_MAX', 'SF_MEDIAN']])

    # 5. Apply (Inference) - Correct Fluxes
    print("\nApplying Correction...")
    applicator = SHCApplicator(cols, sf_df)

    # We purposefully inject a value OUTSIDE the range to test logic
    # USTAR 2.0 is likely higher than the max bin trained on random(0.1, 0.8)
    df.loc[df.index[0], 'USTAR'] = 2.0

    df_final = applicator.apply(df, op_col='NEE_OP', classvar_col='USTAR')

    # 6. Check Results
    print("\nResult Sample:")
    print(df_final[['NEE_OP', cols.fct_unsc_gf, 'USTAR', cols.sf_gf, cols.nee_op_corr]].head())

    # Verify logic: The outlier USTAR 2.0 should have the SF of the highest bin
    high_sf = sf_df.loc[sf_df['GROUP_CLASSVAR_MAX'].idxmax(), 'SF_MEDIAN']
    applied_sf = df_final.loc[df_final.index[0], cols.sf_gf]
    print(f"\nLogic Check - Outlier USTAR 2.0:")
    print(f"Max Bin SF in LUT: {high_sf:.4f}")
    print(f"Applied SF:        {applied_sf:.4f}")

if __name__ == "__main__":
    main()