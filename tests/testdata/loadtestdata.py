from pandas import DataFrame

from diive.common.io.filereader import ReadFileType


def loadtestdata() -> DataFrame:
    # configfilepath = r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\diive\configs\filetypes\DIIVE_CSV_30MIN.yml'
    # filetypeconfig = ConfigFileReader(configfilepath=configfilepath).read()
    filetype = 'CSV_TS-FULL-MIDDLE_30MIN'
    loaddatafile = ReadFileType(
        filetype=filetype,
        # filetypeconfig=filetypeconfig,
        filepath=r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2016-2020_mayToSep.csv')
    data = loaddatafile._readfile()
    return data


if __name__ == '__main__':
    loadtestdata()
