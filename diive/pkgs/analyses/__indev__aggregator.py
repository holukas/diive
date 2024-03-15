"""
todo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

CURRENTLY IN DEV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
todo
"""
import numpy as np
import pandas as pd

from diive.core.io.filereader import ReadFileType


class MonthlyDielCycles:

    def __init__(self):
        pass


def example():
    SOURCEFILE = r"F:\Downloads\_temp2\wsl_19256_data_10min.csv"

    # Read data from precip files to dataframe
    rft = ReadFileType(filepath=SOURCEFILE, filetype='TOA5-CSV-10MIN', output_middle_timestamp=True)
    df = rft.data_df

    d = df[['ghi_Avg']].copy()
    avg = d.resample('1H').mean()
    sd = d.resample('1H').std()
    avg['MONTH'] = avg.index.month
    avg['TIME'] = avg.index.time

    piv = pd.pivot_table(avg, index='TIME', columns='MONTH',
                         values='ghi_Avg', aggfunc=np.mean)

    import matplotlib.pyplot as plt
    piv.plot(title="ghi_Avg")
    plt.show()

    # MonthlyDielCycles()


if __name__ == '__main__':
    example()
