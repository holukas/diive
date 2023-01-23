from diive.core.io.files import load_pickle
from diive.pkgs.fluxprocessingchain.level32_outlierremoval import FluxOutlierRemoval


# from diive.core.io.files import load_pickle, save_as_pickle
# from diive.pkgs.fluxprocessingchain.level2_qualityflags import QualityFlagsLevel2
# from diive.pkgs.fluxprocessingchain.level31_storagecorrection import StorageCorrectionSinglePoint
# from diive.pkgs.fluxprocessingchain.level32_outlierremoval import OutlierSTL
# from diive.pkgs.fluxprocessingchain.qcf import QCF

def example():
    # # # Load data from files
    # # import os
    # # from pathlib import Path
    # # from diive.core.io.filereader import MultiDataFileReader
    # # from diive.core.io.files import save_as_pickle
    # # FOLDER = r"L:\Sync\luhk_work\20 - CODING\26 - NOTEBOOKS\gl-notebooks\__indev__\data\fluxnet"
    # # filepaths = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
    # # filepaths = [FOLDER + "/" + f for f in filepaths]
    # # filepaths = [Path(f) for f in filepaths]
    # # [print(f) for f in filepaths]
    # # loaddatafile = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    # # df = loaddatafile.data_df
    # # save_as_pickle(outpath=r'F:\Dropbox\luhk_work\_temp', filename="data", data=df)
    #
    # # Load data from pickle (much faster loading)
    #
    # # Level-2
    # df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data.pickle")
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
    #
    # # Level-3.1
    # _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L2.pickle")
    # s = StorageCorrectionSinglePoint(df=_df, fluxcol='FC')
    # s.apply_storage_correction()
    # # s.showplot(maxflux=20)
    # # print(s.storage)
    # s.report()
    # _df = s.get()
    # save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L2", data=_df)
    #
    # # QCF after Level-3.1
    # _df = load_pickle(filepath=r"F:\Sync\luhk_work\_temp\data_L2.pickle")
    # qcf = QCF(df=_df, fluxcol='NEE_L3.1', level=3.1, swinpotcol='SW_IN_POT',
    #           nighttime_threshold=50,
    #           daytime_accept_qcf_below=2, nighttimetime_accept_qcf_below=1)
    # qcf.calculate()
    # qcf.report_flags()
    # qcf.report_qcf_evolution()
    # qcf.report_flux()
    # qcf.showplot(maxflux=10)
    # _df = qcf.get()
    # save_as_pickle(outpath=r'F:\Sync\luhk_work\_temp', filename="data_L3.1", data=_df)

    # Level-3.2: Outlier detection
    _df = load_pickle(filepath=r"L:\Sync\luhk_work\_temp\data_L3.1.pickle")
    _df = _df[48 * 150:48 * 200]
    outl = FluxOutlierRemoval(df=_df, targetcol='NEE_L3.1_QCF', nighttimecol='NIGHTTIME')
    outl.run()


if __name__ == '__main__':
    example()
