from pathlib import Path

from pandas import DataFrame

from diive.core.io.filereader import ReadFileType


def loadtestdata(filetype: str,
                 filepath: str) -> tuple[DataFrame, DataFrame]:
    # configfilepath = r'L:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive\diive\configs\filetypes\DIIVE_CSV_30MIN.yml'
    # filetypeconfig = ConfigFileReader(configfilepath=configfilepath).read()
    filepath = Path(filepath)
    loaddatafile = ReadFileType(
        filetype=filetype,
        # filetypeconfig=filetypeconfig,
        filepath=filepath)
    data_df, metadata_df = loaddatafile._readfile()
    return data_df, metadata_df


if __name__ == '__main__':
    pass
