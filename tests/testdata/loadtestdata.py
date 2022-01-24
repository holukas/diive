from pandas import DataFrame

from diive.common.io.filereader import ReadFileType


def loadtestdata() -> DataFrame:
    loaddatafile = ReadFileType(
        filetype='DIIVE_CSV_30MIN',
        filepath=r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\tests\testdata\testfile_ch-dav_2016-2020.diive.csv')
    df = loaddatafile._readfile()
    return df


if __name__ == '__main__':
    loadtestdata()
