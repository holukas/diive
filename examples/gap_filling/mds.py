"""Marginal Distribution Sampling (MDS) gap-filling examples.

Gap-filling after Reichstein et al (2005):
https://doi.org/10.1111/j.1365-2486.2005.001002.x

MDS fills gaps by using the average flux value during similar meteorological
conditions (SWIN, TA, VPD). Uses a hierarchical quality-based approach with
progressively relaxed meteorological similarity windows.
"""
import time

import diive as dv


def example_mds_flux_gapfilling():
    """Marginal Distribution Sampling (MDS) gap-filling for ecosystem flux.

    Demonstrates MDS gap-filling which replaces missing flux values with
    average flux from similar meteorological conditions. Uses hierarchical
    quality levels: starts with strict conditions (7 days, all 3 variables),
    progressively relaxes to 140+ days with fewer constraints.

    Quality levels:
    - A1-A3 (1-3): High quality, all variables within 7-14 days
    - B1-B4 (4-8): Medium quality, variables within 21-28 days
    - C+ (9+): Low quality, variables within 35-140+ days

    Returns:
        None (displays plot and reports)
    """
    df = dv.load_exampledata_parquet()

    # Prepare data: use July 2022 for faster example execution
    # df = df.loc[(df.index.year == 2022)].copy()
    df = df.loc[(df.index.year == 2022) & (df.index.month == 7)].copy()

    # Variables
    flux = 'NEE_CUT_REF_orig'  # Flux to gap-fill
    ta = 'Tair_f'  # Air temperature (°C)
    swin = 'Rg_f'  # Short-wave incoming radiation (W m-2)
    vpd = 'VPD_f'  # Vapor pressure deficit (in this file: hPa)

    # MDS tolerance settings
    swin_tol = [20, 50]  # W m-2 (low radiation: 20, high: 50)
    ta_tol = 2.5  # °C
    vpd_tol = 0.5  # kPa
    avg_min_n_vals = 5  # Minimum flux values to calculate average

    # Convert VPD from hPa to kPa
    df[vpd] = df[vpd].multiply(0.1)

    # Create and run MDS gap-filling with timing
    start_time = time.perf_counter()

    mds = dv.FluxMDS(
        df=df,
        flux=flux,
        ta=ta,
        swin=swin,
        vpd=vpd,
        swin_tol=swin_tol,
        ta_tol=ta_tol,
        vpd_tol=vpd_tol,
        avg_min_n_vals=avg_min_n_vals,
        verbose=1
    )

    mds.run()
    mds.report()
    mds.showplot()

    elapsed_time = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"MDS gap-filling execution time: {elapsed_time:.2f} seconds")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    example_mds_flux_gapfilling()
