# TODO Marius meint aufpassen auf timestamps wegen mm hr-1 !!!!!!!!!!!!!!!!
import pandas as pd


def latent_heat_of_vaporization(ta: pd.Series) -> pd.Series:
    """Calculate latent heat of vaporization as a function of air temperature.

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/meteorological_variables.r#sym-latent.heat.vaporization

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
    Lv = (k1 - k2 * ta) * (10 ^ 6)
    return Lv


def et_from_le(le: pd.Series, ta: pd.Series) -> pd.Series:
    """Convert latent heat flux (energy) to evapotranspiration (mass).

    Kudos:
        Based on code of the R package 'bigleaf':
        https://rdrr.io/cran/bigleaf/src/R/unit_conversions.r#sym-LE.to.ET

    Args:
        le: latent heat flux (W m-2)
        ta: air temperature (°C)

    Returns:
        Evapotranspiration (kg m-2 s-1, equivalent to mm s-1)
    """
    _lambda = latent_heat_of_vaporization(ta)
    et = le / _lambda  # kg m-2 s-1
    # et = et.multiply(3600)  # kg m-2 hr-1
    # et = et.multiply(18.016)  # kg m-2 hr-1

    return et


def _example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    le = df['LE_f'].copy()
    ta = df['Tair_f'].copy()
    et_check = df['ET_f'].copy()

    et = et_from_le(le=le, ta=ta)
    print(et)
    print(et_check)


if __name__ == '__main__':
    _example()
