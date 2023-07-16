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
    from pathlib import Path
    import matplotlib.pyplot as plt
    from diive.core.io.files import loadfiles
    from diive.core.times.times import insert_timestamp
    from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
    from diive.core.plotting.heatmap_datetime import HeatmapDateTime

    # Dirs
    SOURCEDIR = r"L:\Sync\luhk_work\80 - SITES\CH-FRU\DATA\Shared\2023-06-15 CH-FRU 2020-2022 TA RH VPD\0-sourcefiles"
    OUTDIR = r"L:\Sync\luhk_work\80 - SITES\CH-FRU\DATA\Shared\2023-06-15 CH-FRU 2020-2022 TA RH VPD\1-out"

    # Data
    data_df = loadfiles(filetype='FLUXNET-FULLSET-HH-CSV-30MIN',
                        sourcedir=SOURCEDIR,
                        limit_n_files=None,
                        fileext='.csv',
                        idstr='_fluxnet_')

    # Restrict years
    locs = (data_df.index.year >= 2020) & (data_df.index.year <= 2022)
    data_df = data_df[locs].copy()

    # Variables
    ta_col = 'TA_1_1_1'
    rh_col = 'RH_1_1_1'
    vpd_col = 'VPD_hPa'
    swin_col = 'SW_IN_1_1_1'
    swinpot_col = 'SW_IN_POT'

    subset_df = data_df[[ta_col, rh_col, swinpot_col, swin_col]].copy()

    # Number of gaps
    print(subset_df[swin_col].isnull().sum())
    print(subset_df[ta_col].isnull().sum())

    # Random forest gap-filling
    rfsettings = dict(include_timestamp_as_features=True,
                      lagged_variants=3, use_neighbor_years=True,
                      feature_reduction=False, verbose=1)
    rfsettings_model = dict(n_estimators=99, random_state=42,
                            min_samples_split=2, min_samples_leaf=1, n_jobs=-1)

    # SW_IN
    rfts = RandomForestTS(df=subset_df, target_col=swin_col, **rfsettings)
    rfts.build_models(**rfsettings_model)
    rfts.gapfill_yrs()
    swin_gf_col = f'{swin_col}_gfRF'
    gapfilled_yrs_df, yrpool_gf_results = rfts.get_gapfilled_dataset()
    subset_df[swin_gf_col] = gapfilled_yrs_df[swin_gf_col].copy()

    # TA
    gfcols = [ta_col, swin_gf_col]
    gf_df = subset_df[gfcols].copy()
    rfts = RandomForestTS(df=gf_df, target_col=ta_col, **rfsettings)
    rfts.build_models(**rfsettings_model)
    rfts.gapfill_yrs()
    ta_gf_col = f'{ta_col}_gfRF'
    gapfilled_yrs_df, yrpool_gf_results = rfts.get_gapfilled_dataset()
    subset_df[ta_gf_col] = gapfilled_yrs_df[ta_gf_col].copy()

    # Calculate VPD
    subset_df[vpd_col] = calc_vpd_from_ta_rh(df=subset_df, ta_col=ta_gf_col, rh_col=rh_col)

    # VPD
    gfcols = [vpd_col, swin_gf_col, ta_gf_col]
    gf_df = subset_df[gfcols].copy()
    rfts = RandomForestTS(df=gf_df, target_col=vpd_col, **rfsettings)
    rfts.build_models(**rfsettings_model)
    rfts.gapfill_yrs()
    vpd_gf_col = f'{vpd_col}_gfRF'
    gapfilled_yrs_df, yrpool_gf_results = rfts.get_gapfilled_dataset()
    subset_df[vpd_gf_col] = gapfilled_yrs_df[vpd_gf_col].copy()

    HeatmapDateTime(series=subset_df[vpd_gf_col]).show()
    HeatmapDateTime(series=subset_df[vpd_col]).show()

    subset_df = insert_timestamp(data=subset_df, convention='end')

    subset_df = subset_df[[vpd_gf_col, rh_col, ta_gf_col, swin_gf_col]].copy()
    subset_df = subset_df.resample('D').mean()

    subset_df.plot(subplots=True)
    plt.show()

    # [print(x) for x in data_df.columns if 'SW' in x]

    subset_df.index.name = "TIMESTAMP"
    outfile = Path(OUTDIR) / "CH-FRU_VPD_RH_TA_2020-2022_20230615.csv"
    subset_df.to_csv(outfile)


if __name__ == '__main__':
    example()
