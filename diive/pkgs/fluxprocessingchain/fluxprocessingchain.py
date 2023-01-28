def example():
    # # ----------------------
    # # Load data from files
    # # ----------------------
    # import os
    # from pathlib import Path
    # from diive.core.io.filereader import MultiDataFileReader
    # from diive.core.io.files import save_as_pickle
    # FOLDER = r"F:\Sync\luhk_work\20 - CODING\26 - NOTEBOOKS\gl-notebooks\__indev__\data\fluxnet"
    # filepaths = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
    # filepaths = [FOLDER + "/" + f for f in filepaths]
    # filepaths = [Path(f) for f in filepaths]
    # [print(f) for f in filepaths]
    # loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    # df = loaddatafile.data_df
    # save_as_pickle(outpath=r'F:\Dropbox\luhk_work\_temp', filename="data", data=df)

    # Load data from pickle (much faster loading)

    # # -------------------------------
    # # Level-2: Quality flag expansion
    # # -------------------------------
    # from diive.pkgs.fluxprocessingchain.level2_qualityflags import QualityFlagsLevel2
    # from diive.core.io.files import load_pickle, save_as_pickle
    # df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data.pickle")
    # # df = df.iloc[0:20000]
    # fluxqc = QualityFlagsLevel2(fluxcol='FC', df=df)
    # fluxqc.missing_vals_test()
    # fluxqc.ssitc_test()
    # fluxqc.gas_completeness_test()
    # fluxqc.spectral_correction_factor_test()
    # fluxqc.signal_strength_test(signal_strength_col='CUSTOM_AGC_MEAN',
    #                             method='discard above', threshold=90)
    # fluxqc.raw_data_screening_vm97()
    # # print(fluxqc.fluxflags)
    # _df = fluxqc.get()
    # save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L2", data=_df)

    # # -----------------------------
    # # Level-3.1: Storage correction
    # # -----------------------------
    # from diive.pkgs.fluxprocessingchain.level31_storagecorrection import StorageCorrectionSinglePoint
    # from diive.core.io.files import load_pickle, save_as_pickle
    # _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L2.pickle")
    # s = StorageCorrectionSinglePoint(df=_df, fluxcol='FC')
    # s.apply_storage_correction()
    # # s.showplot(maxflux=20)
    # # print(s.storage)
    # s.report()
    # _df = s.get()
    # save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L2", data=_df)

    # # -------------------
    # # QCF after Level-3.1
    # # -------------------
    # from diive.pkgs.qaqc.qcf import FlagQCF
    # from diive.core.io.files import load_pickle, save_as_pickle
    # _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L2.pickle")
    # qcf = FlagQCF(series=_df['NEE_L3.1'], df=_df, levelid=3.1, swinpot=_df['SW_IN_POT'], nighttime_threshold=50)
    # qcf.calculate(daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=1)
    # qcf.report_flags()
    # qcf.report_qcf_evolution()
    # qcf.report_series()
    # qcf.showplot_heatmaps(maxflux=10)
    # # qcf.showplot_timeseries()
    # _df = qcf.get()
    # save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L3.1", data=_df)

    # ----------------------------
    # Level-3.2: Outlier detection
    # ----------------------------
    site_lat = 46.815333
    site_lon = 9.855972
    from diive.core.io.files import load_pickle, save_as_pickle
    from diive.pkgs.fluxprocessingchain.level32_outlierremoval import OutlierRemovalLevel32
    _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L3.1.pickle")
    olr = OutlierRemovalLevel32(df=_df, fluxcol='NEE_L3.1_L3.1_QCF', site_lat=site_lat, site_lon=site_lon)
    olr.zscore_dtnt(threshold=4, showplot=True, verbose=True)
    olr.addflag()
    olr.stl_iqrz(zfactor=1.5, decompose_downsampling_freq='3H', repeat=False, showplot=True)
    olr.addflag()
    olr.lof_dtnt(n_neighbors=None, contamination='auto', showplot=True, verbose=True)
    olr.addflag()
    _df = olr.fulldf
    save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L3.2", data=_df)

    # -------------------
    # QCF after Level-3.2
    # -------------------
    from diive.pkgs.qaqc.qcf import FlagQCF
    from diive.core.io.files import load_pickle, save_as_pickle
    _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L3.2.pickle")
    qcf = FlagQCF(series=_df['NEE_L3.1_L3.1_QCF'], df=_df, levelid=3.2,
                  swinpot=_df['SW_IN_POT'], nighttime_threshold=50)
    qcf.calculate(daytime_accept_qcf_below=2,
                  nighttimetime_accept_qcf_below=1)
    qcf.report_flags()
    qcf.report_qcf_evolution()
    qcf.report_series()
    qcf.showplot_heatmaps(maxflux=10)
    qcf.showplot_timeseries()
    _df = qcf.get()
    save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L3.2", data=_df)

    from diive.core.io.files import load_pickle
    _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L3.2.pickle")

    [print(c) for c in _df.columns if str(c).startswith('NEE')]
    _test = _df['NEE_L3.1_L3.1_QCF_L3.2_QCF'].copy()

    # https://fitter.readthedocs.io/en/latest/tuto.html
    # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
    from fitter import Fitter, get_common_distributions
    f = Fitter(_test.dropna().to_numpy(), distributions=get_common_distributions())
    f.fit()
    f.summary()

    #          sumsquare_error          aic            bic    kl_div  ks_statistic      ks_pvalue
    # norm            0.006988  1403.489479 -377396.228381  0.057412      0.061269   4.660184e-82
    # lognorm         0.007025  1428.315952 -377253.532484  0.057257      0.063151   3.743164e-87
    # gamma           0.008857  1449.221743 -371458.509961  0.073552      0.077689  9.752981e-132
    # cauchy          0.009007  1025.350595 -371047.546496  0.064971      0.080458  2.817290e-141
    # chi2            0.010701  1420.444096 -366728.335918  0.092422      0.090688  1.948207e-179

    f.get_best(method='sumsquare_error')
    # {'norm': {'loc': -4.008143649847604, 'scale': 6.44354010359457}}

    f.fitted_param["norm"]
    # (-4.008143649847604, 6.44354010359457)

    pdf = f.fitted_pdf['norm']

    from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    HeatmapDateTime(series=_test, vmin=-20, vmax=20).show()

    import math
    import numpy as np
    from scipy.stats import shapiro
    from scipy.stats import lognorm

    # make this example reproducible
    np.random.seed(1)

    # generate dataset that contains 1000 log-normal distributed values
    lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

    # perform Shapiro-Wilk test for normality
    shapiro(lognorm_dataset)

    shapiro(_test.dropna().to_numpy())

    import math
    import numpy as np
    from scipy.stats import kstest
    from scipy.stats import lognorm

    # make this example reproducible
    np.random.seed(1)

    # generate dataset that contains 1000 log-normal distributed values
    lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

    # perform Kolmogorov-Smirnov test for normality
    kstest(lognorm_dataset, 'norm')

    kstest(_test.dropna().to_numpy(), 'norm')

    import math
    import numpy as np
    from scipy.stats import lognorm
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # make this example reproducible
    np.random.seed(1)

    # generate dataset that contains 1000 log-normal distributed values
    lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

    # create Q-Q plot with 45-degree line added to plot
    fig = sm.qqplot(lognorm_dataset, line='45')

    fig = sm.qqplot(_test.dropna().to_numpy(), line='45')

    plt.show()

    import math
    import numpy as np
    from scipy.stats import lognorm
    import matplotlib.pyplot as plt

    # make this example reproducible
    np.random.seed(1)

    # generate dataset that contains 1000 log-normal distributed values
    lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

    # create histogram to visualize values in dataset
    plt.hist(lognorm_dataset, edgecolor='black', bins=20)
    plt.show()

    plt.hist(_test.dropna().to_numpy(), edgecolor='black', bins=20)
    plt.show()

    dt = _df['SW_IN_POT'] > 50
    plt.hist(_test.loc[dt].dropna().to_numpy(), edgecolor='black', bins=20)
    plt.show()
    fig = sm.qqplot(_test.loc[dt].dropna().to_numpy(), line='45')
    plt.show()

    nt = _df['SW_IN_POT'] < 50
    plt.hist(_test.loc[nt].dropna().to_numpy(), edgecolor='black', bins=20)
    plt.show()
    fig = sm.qqplot(_test.loc[nt].dropna().to_numpy(), line='45')
    plt.show()

    # https://www.statology.org/normality-test-python/
    # https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9

    print("X")


if __name__ == '__main__':
    example()
