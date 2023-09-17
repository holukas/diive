#  TODO NEEDS FLOW CHECK


def example():
    from pathlib import Path
    SOURCEFOLDER = [Path(r'F:\Sync\luhk_work\TMP\fru')]
    OUTPATH = Path(r'F:\Sync\luhk_work\TMP\fru')

    # ----------------------
    # Load data from files
    # ----------------------
    from diive.core.io.filereader import MultiDataFileReader, search_files
    from diive.core.io.files import save_parquet
    filepaths = search_files(SOURCEFOLDER, "*.csv")
    filepaths = [fp for fp in filepaths if
                 "eddypro_" in fp.stem and "_fluxnet_" in fp.stem and fp.stem.endswith("_adv")]
    loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    df = loaddatafile.data_df
    save_parquet(outpath=OUTPATH, filename="data", data=df)

    # -------------------------------
    # Level-2: Quality flag expansion
    # -------------------------------
    from pandas import read_parquet
    from diive.pkgs.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsLevel2EddyPro
    from diive.core.io.files import save_parquet
    filepath = str(OUTPATH / "data.parquet")
    df = read_parquet(filepath)
    fluxqc = FluxQualityFlagsLevel2EddyPro(fluxcol='H', df=df, levelid='L2')
    fluxqc.missing_vals_test()
    fluxqc.ssitc_test()
    fluxqc.gas_completeness_test()
    fluxqc.spectral_correction_factor_test()
    fluxqc.signal_strength_test(signal_strength_col='CUSTOM_AGC_MEAN',
                                method='discard above', threshold=90)
    fluxqc.raw_data_screening_vm97_tests(spikes=True,
                                         amplitude=True,
                                         dropout=True,
                                         abslim=False,
                                         skewkurt_hf=False,
                                         skewkurt_sf=False,
                                         discont_hf=False,
                                         discont_sf=False)
    fluxqc.angle_of_attack_test()
    # print(fluxqc.fluxflags)
    _df = fluxqc.get()
    save_parquet(outpath=OUTPATH, filename="data_L2", data=_df)

    # -----------------------------
    # Level-3.1: Storage correction
    # -----------------------------
    from pandas import read_parquet
    from diive.pkgs.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
    from diive.core.io.files import save_parquet
    filepath = str(OUTPATH / "data_L2.parquet")
    df = read_parquet(filepath)
    s = FluxStorageCorrectionSinglePointEddyPro(df=df, fluxcol='FC')
    s.storage_correction()
    # s.showplot(maxflux=20)
    # print(s.storage)
    s.report()
    _df = s.get()
    save_parquet(outpath=OUTPATH, filename="data_L3.1", data=_df)

    # -------------------
    # QCF after Level-3.1
    # -------------------
    from pandas import read_parquet
    from diive.pkgs.qaqc.qcf import FlagQCF
    from diive.core.io.files import save_parquet
    filepath = str(OUTPATH / "data_L3.1.parquet")
    _df = read_parquet(filepath)
    qcf = FlagQCF(series=_df['H_L3.1'], df=_df, levelid='L3.1', swinpot=_df['SW_IN_POT'], nighttime_threshold=50)
    qcf.calculate(daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=2)
    # qcf.report_qcf_flags()
    qcf.report_qcf_evolution()
    # qcf.report_qcf_series()
    # qcf.showplot_qcf_heatmaps(maxabsval=10)
    # qcf.showplot_qcf_timeseries()
    _df = qcf.get()
    save_parquet(outpath=OUTPATH, filename="data_L3.1b", data=_df)

    # # todo TESTING hq fluxes
    # df_hq = _df.loc[_df['FLAG_L3.1_NEE_L3.1_QCF'] == 1].copy()
    # df_hq = df_hq.loc[df_hq['USTAR'] > 0.1].copy()
    #
    # dt = df_hq['SW_IN_POT'] > 20
    # nt = df_hq['SW_IN_POT'] <= 20
    #
    # df_hq_dt = df_hq[dt].copy()
    # df_hq_nt = df_hq[nt].copy()
    #
    # df_hq_dt_flux = df_hq_dt['NEE_L3.1_L3.1_QCF'].copy()
    # df_hq_nt_flux = df_hq_nt['NEE_L3.1_L3.1_QCF'].copy()
    #
    # from diive.core.dfun.stats import sstats
    # sstats(df_hq_dt_flux)
    # sstats(df_hq_nt_flux)
    #
    # import matplotlib.pyplot as plt
    # df_hq_dt_flux.plot()
    # df_hq_nt_flux.plot()
    # plt.show()

    # ----------------------------
    # Level-3.2: Outlier detection
    # ----------------------------
    site_lat = 46.815333
    site_lon = 9.855972
    from pandas import read_parquet
    from diive.pkgs.outlierdetection.stepwiseoutlierdetection import StepwiseOutlierDetection
    from diive.core.io.files import save_parquet
    filepath = str(OUTPATH / "data_L3.1b.parquet")
    _df = read_parquet(filepath)

    sod = StepwiseOutlierDetection(dataframe=_df,
                                   col='NEE_L3.1_L3.1_QCF',
                                   site_lat=site_lat,
                                   site_lon=site_lon,
                                   timezone_of_timestamp='UTC+01:00')

    sod.flag_missingvals_test()
    sod.addflag()

    # sod.flag_outliers_abslim_dtnt_test(daytime_minmax=[-50, 50], nighttime_minmax=[-5, 20], showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_manualremoval_test(remove_dates=[['2019-12-31 19:45:00', '2020-01-31 19:45:00']],
    #                             showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_zscore_dtnt_test(threshold=3, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_increments_zcore_test(threshold=10, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_zscoreiqr_test(factor=3, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_zscore_test(threshold=3, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_thymeboost_test(showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_localsd_test(n_sd=3, winsize=480, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_abslim_test(minval=-10, maxval=5, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_stl_rz_test(zfactor=3, decompose_downsampling_freq='6H', repeat=False, showplot=True)
    # sod.addflag()
    # sod.showplot_orig()
    # sod.showplot_cleaned()

    # sod.flag_outliers_lof_dtnt_test(n_neighbors=None, contamination=0.0005, showplot=True, verbose=True)
    # sod.addflag()

    # sod.flag_outliers_lof_test(n_neighbors=None, contamination=0.005, showplot=True, verbose=True)
    # sod.addflag()

    _df = sod.get()



    # todo ? olr = FluxOutlierRemovalLevel32(df=_df, fluxcol='NEE_L3.1_L3.1_QCF', site_lat=site_lat, site_lon=site_lon)

    # # TODO CHECK outlier removal options, remove qcf=1 based on qcf=0?
    save_parquet(outpath=OUTPATH, filename="data_L3.2", data=_df)

    # -------------------
    # QCF after Level-3.2
    # -------------------
    from pandas import read_parquet
    from diive.pkgs.qaqc.qcf import FlagQCF
    from diive.core.io.files import save_parquet
    filepath = str(OUTPATH / "data_L3.2.parquet")
    _df = read_parquet(filepath)
    qcf = FlagQCF(series=_df['NEE_L3.1_L3.1_QCF'], df=_df, levelid='3.2',
                  swinpot=_df['SW_IN_POT'], nighttime_threshold=50)
    qcf.calculate(daytime_accept_qcf_below=2,
                  nighttimetime_accept_qcf_below=2)
    qcf.report_qcf_flags()
    qcf.report_qcf_evolution()
    qcf.report_qcf_series()
    qcf.showplot_qcf_heatmaps(maxabsval=10)
    qcf.showplot_qcf_timeseries()
    _df = qcf.get()
    save_parquet(outpath=OUTPATH, filename="data_L3.2q", data=_df)

    # from diive.core.io.files import load_pickle
    # _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L3.2.pickle")
    #
    # [print(c) for c in _df.columns if str(c).startswith('NEE')]
    # _test = _df['NEE_L3.1_L3.1_QCF_L3.2_QCF'].copy()
    #
    # # https://fitter.readthedocs.io/en/latest/tuto.html
    # # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
    # from fitter import Fitter, get_common_distributions
    # f = Fitter(_test.dropna().to_numpy(), distributions=get_common_distributions())
    # f.fit()
    # f.summary()
    #
    # #          sumsquare_error          aic            bic    kl_div  ks_statistic      ks_pvalue
    # # norm            0.006988  1403.489479 -377396.228381  0.057412      0.061269   4.660184e-82
    # # lognorm         0.007025  1428.315952 -377253.532484  0.057257      0.063151   3.743164e-87
    # # gamma           0.008857  1449.221743 -371458.509961  0.073552      0.077689  9.752981e-132
    # # cauchy          0.009007  1025.350595 -371047.546496  0.064971      0.080458  2.817290e-141
    # # chi2            0.010701  1420.444096 -366728.335918  0.092422      0.090688  1.948207e-179
    #
    # f.get_best(method='sumsquare_error')
    # # {'norm': {'loc': -4.008143649847604, 'scale': 6.44354010359457}}
    #
    # f.fitted_param["norm"]
    # # (-4.008143649847604, 6.44354010359457)
    #
    # pdf = f.fitted_pdf['norm']
    #
    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=_test, vmin=-20, vmax=20).show()
    #
    # import math
    # import numpy as np
    # from scipy.stats import shapiro
    # from scipy.stats import lognorm
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # perform Shapiro-Wilk test for normality
    # shapiro(lognorm_dataset)
    #
    # shapiro(_test.dropna().to_numpy())
    #
    # import math
    # import numpy as np
    # from scipy.stats import kstest
    # from scipy.stats import lognorm
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # perform Kolmogorov-Smirnov test for normality
    # kstest(lognorm_dataset, 'norm')
    #
    # kstest(_test.dropna().to_numpy(), 'norm')
    #
    # import math
    # import numpy as np
    # from scipy.stats import lognorm
    # import statsmodels.api as sm
    # import matplotlib.pyplot as plt
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # create Q-Q plot with 45-degree line added to plot
    # fig = sm.qqplot(lognorm_dataset, line='45')
    #
    # fig = sm.qqplot(_test.dropna().to_numpy(), line='45')
    #
    # plt.show()
    #
    # import math
    # import numpy as np
    # from scipy.stats import lognorm
    # import matplotlib.pyplot as plt
    #
    # # make this example reproducible
    # np.random.seed(1)
    #
    # # generate dataset that contains 1000 log-normal distributed values
    # lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)
    #
    # # create histogram to visualize values in dataset
    # plt.hist(lognorm_dataset, edgecolor='black', bins=20)
    # plt.show()
    #
    # plt.hist(_test.dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    #
    # dt = _df['SW_IN_POT'] > 50
    # plt.hist(_test.loc[dt].dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    # fig = sm.qqplot(_test.loc[dt].dropna().to_numpy(), line='45')
    # plt.show()
    #
    # nt = _df['SW_IN_POT'] < 50
    # plt.hist(_test.loc[nt].dropna().to_numpy(), edgecolor='black', bins=20)
    # plt.show()
    # fig = sm.qqplot(_test.loc[nt].dropna().to_numpy(), line='45')
    # plt.show()
    #
    # # https://www.statology.org/normality-test-python/
    # # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
    #
    # print("X")


if __name__ == '__main__':
    example()
