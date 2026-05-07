"""
Self-heating correction examples for open-path IRGA sensors.

Demonstrates SCOP (Self-heating Correction for Open-Path) methodology to remove
spurious CO2 flux measurements caused by sun-induced heating of instrument surfaces.

Includes physics modeling, optimization of scaling factors, and correction application.

Run this script to see self-heating correction examples:
    python examples/flux/selfheating.py

See Also
--------
diive.pkgs.flux.selfheating : Self-heating correction classes (ScopPhysics, ScopOptimizer, ScopApplicator)
"""
import time

from diive.pkgs.flux.selfheating import ScopPhysics, ScopOptimizer, ScopApplicator


def example_selfheating_ch_lae():
    """Self-heating correction for CH-LAE site with parallel IRGA measurements.

    Demonstrates full SCOP workflow:
    1. ScopPhysics: Calculate unscaled flux correction term (FCT_UNSC)
    2. ScopOptimizer: Optimize scaling factors using closed-path reference
    3. ScopApplicator: Apply final correction to open-path flux

    Uses parallel measurements with open-path IRGA75 and closed-path IRGA72
    to determine optimal scaling factors via bin-based optimization.
    """

    from diive.configs.exampledata import load_exampledata_parquet_lae
    df = load_exampledata_parquet_lae()

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


if __name__ == '__main__':
    example_selfheating_ch_lae()
