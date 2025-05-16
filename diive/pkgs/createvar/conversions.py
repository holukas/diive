import pandas as pd


def latent_heat_of_vaporization(ta: pd.Series) -> pd.Series:
    """Calculate latent heat of vaporization as a function of air temperature.

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/meteorological_variables.r (latent.heat.vaporization)

    Reference:
        Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
        Kluwer Academic Publishers, Dordrecht, Netherlands

    Args:
        ta: air temperature (°C)

    Returns:
        Latent heat of vaporization (J kg-1)
    """
    k1 = 2.501
    k2 = 0.00237
    Lv = (k1 - k2 * ta) * (10 ** 6)
    return Lv


def et_from_le(le: pd.Series, ta: pd.Series) -> pd.Series:
    """Convert latent heat flux (energy) to evapotranspiration (mass).

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/unit_conversions.r (LE.to.ET)

    Args:
        le: latent heat flux (W m-2)
        ta: air temperature (°C)

    Returns:
        Evapotranspiration (mm H2O h-1)
    """
    _lambda = latent_heat_of_vaporization(ta)
    et = le / _lambda  # kg m-2 s-1 = mm s-1
    et = et.multiply(3600)  # mm h-1
    return et


def _example_et_from_le():
    from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN
    df, meta = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
    le = df['LE'].copy()
    et_check = df['ET'].copy()  # Should be in mm h-1
    ta = df['TA_1_1_1'].copy()

    et = et_from_le(le=le, ta=ta)
    print(et)
    print(et_check)

    import matplotlib.pyplot as plt
    et.plot(label="ET")
    et_check.plot(label="ET from EddyPro")
    plt.legend()
    plt.show()

    # # Testing polars vs pandas
    # # Execution time for 100 000 repeats:
    # #     pandas: 11.505 seconds
    # #     polars:  1.975 seconds
    # ta = pl.from_pandas(ta)
    # print(type(ta))
    # start_time = time.perf_counter()
    # for i in range(100_000):
    #     res = latent_heat_of_vaporization(ta=ta)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time} seconds")


if __name__ == '__main__':
    _example_et_from_le()
