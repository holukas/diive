from numpy import exp
from pandas import DataFrame, Series


def calc_vpd_from_ta_rh(df: DataFrame, rh_col: str, ta_col: str) -> Series:
    """

    Args:
        df: Data
        rh_col: Name of column in *df* containing relative humidity data in %
        ta_col: Name of column in *df* containing air temperature data in °C

    Returns:
        VPD in kPa as Series

    Original code in ReddyProc:
        VPD.V.n <- 6.1078 * (1 - rH / 100) * exp(17.08085 * Tair / (234.175 + Tair))
        # See Kolle Logger Tools Software 2012 (Magnus coefficients for water between 0 and 100 degC)
        # Data vector of vapour pressure deficit (VPD, hPa (mbar))

    Reference:
        https://github.com/bgctw/REddyProc/blob/3c2b414c24900c17ab624c20b0e1726e6a813267/R/GeoFunctions.R#L96

    Checked against ReddyProc output, virtually the same, differences only 8 digits after comma.

    """

    # Subset
    rh = df[[rh_col, ta_col]].copy()


    # Calculate terms in separate columns
    #                                              |cc---------------------------------------------->|
    #            |a--->|  |b------------------->|     |c-------------------------------------------->|
    # VPD(hPa) = 6.1078 * (1 - df[rh_col] / 100) * exp(17.08085 * df[ta_col] / (234.175 + df[ta_col]))
    rh['a'] = 6.1078
    rh['b'] = 1 - df[rh_col] / 100
    rh['c'] = rh[ta_col].multiply(17.08085) / rh[ta_col].add(234.175)
    rh['cc'] = exp(rh['c'])

    # Calculate VPD in hPa
    rh['VPD'] = rh['a'] * rh['b'] * rh['cc']

    # Convert units for output in kPa
    rh['VPD'] = rh['VPD'].multiply(0.1)  # hPa --> kPa
    return rh['VPD']
