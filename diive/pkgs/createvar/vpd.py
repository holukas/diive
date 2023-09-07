from numpy import exp
from pandas import DataFrame, Series


def calc_vpd_from_ta_rh(df: DataFrame, rh_col: str, ta_col: str) -> Series:
    """
    Calculate VPD from air temperature and relative humidity

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

    - Example notebook available in:
        notebooks/CalculateVariable/Calculate_VPD_from_TA_and_RH.ipynb

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


def example():
    import matplotlib.pyplot as plt

    from diive.core.times.times import insert_timestamp
    from diive.pkgs.gapfilling.randomforest_ts import QuickFillRFTS
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    from diive.configs.exampledata import load_exampledata_pickle

    # Example data
    df = load_exampledata_pickle()

    # Variables
    ta_col = 'Tair_f'  # Already gap-filled
    rh_col = 'RH'
    vpd_col = 'VPD_f'
    vpd_new_col = 'VPD'
    swin_col = 'Rg_f'  # Already gap-filled
    # [print(x) for x in df.columns if "Rg" in x]

    # Subset
    subsetcols = [ta_col, rh_col, vpd_col, swin_col]
    subset_df = df[subsetcols].copy()

    # Number of gaps
    print(subset_df[swin_col].isnull().sum())
    print(subset_df[ta_col].isnull().sum())

    # Calculate VPD
    subset_df[vpd_new_col] = calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_col, rh_col=rh_col)

    # VPD
    gfcols = [vpd_col, swin_col, ta_col]
    gf_df = subset_df[gfcols].copy()
    rfts = QuickFillRFTS(df=gf_df, target_col=vpd_col)
    rfts.fill()
    subset_df[vpd_new_col] = rfts.get_gapfilled()

    rfts.report()

    HeatmapDateTime(series=subset_df[vpd_new_col]).show()
    HeatmapDateTime(series=subset_df[vpd_col]).show()

    subset_df = insert_timestamp(data=subset_df, convention='end')

    subset_df = subset_df[[vpd_new_col, rh_col, ta_col, swin_col]].copy()
    subset_df = subset_df.resample('D').mean()

    subset_df.plot(subplots=True)
    plt.show()


if __name__ == '__main__':
    example()
