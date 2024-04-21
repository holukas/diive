import unittest

import pandas as pd
from pandas import DataFrame

import diive.configs.exampledata as ed



class TestLoadFiletypes(unittest.TestCase):

    def test_load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN(self):
        """Load EddyPro _fluxnet_ file"""
        data_df, metadata_df = ed.load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 488)
        self.assertEqual(len(data_df), 1488)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 489)  # One more col than data b/c TIMESTAMP_END removed during parsing
        self.assertEqual(data_df.sum().sum(), 307678629054925.25)
        self.assertEqual(data_df.index[0], pd.Timestamp('2021-07-01 00:15:00'))
        self.assertEqual(data_df.index[999], pd.Timestamp('2021-07-21 19:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2021-07-31 23:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')

    def test_load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN_datafilereader_parameters(self):
        """Load EddyPro _fluxnet_ file by providing parameters to DataFileReader"""
        data_df, metadata_df = ed.load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN_with_datafilereader_parameters()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 488)
        self.assertEqual(len(data_df), 1488)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 489)  # One more col than data b/c TIMESTAMP_END removed during parsing
        self.assertEqual(data_df.sum().sum(), 307678629054925.25)
        self.assertEqual(data_df.index[0], pd.Timestamp('2021-07-01 00:15:00'))
        self.assertEqual(data_df.index[999], pd.Timestamp('2021-07-21 19:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2021-07-31 23:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')

    def test_load_exampledata_multiple_EDDYPRO_FLUXNET_CSV_30MIN(self):
        """Load and merge multiple EddyPro _fluxnet_ files"""
        data_df, metadata_df = ed.load_exampledata_multiple_EDDYPRO_FLUXNET_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 487)
        self.assertEqual(len(data_df), 188563)
        self.assertEqual(data_df['W_UNROT'].describe()['count'], 103)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 488)  # One more col than data b/c TIMESTAMP_END removed during parsing
        self.assertEqual(data_df.sum().sum(), 21263415278351.82)
        self.assertEqual(data_df.index[0], pd.Timestamp('2012-06-09 15:15:00'))
        self.assertEqual(data_df.index[37423], pd.Timestamp('2014-07-29 06:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2023-03-13 00:15:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')

    def test_load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN(self):
        """Load EddyPro _fluxnet_ file"""
        data_df, metadata_df = ed.load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 120)
        self.assertEqual(len(data_df), 468)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 122)  # Two more cols than data b/c date and time cols removed during parsing
        self.assertEqual(data_df['co2_flux'].sum(), -1781.7114716)
        self.assertEqual(data_df.sum().sum(), 2998405615966.3564)
        self.assertEqual(data_df.index[0], pd.Timestamp('2024-03-29 01:15:00'))
        self.assertEqual(data_df.index[10], pd.Timestamp('2024-03-29 06:15:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2024-04-07 18:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')

    def test_load_exampledata_DIIVE_CSV_30MIN(self):
        """Load DIIVE-CSV-30MIN file"""
        data_df, metadata_df = ed.load_exampledata_DIIVE_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 101)
        self.assertEqual(len(data_df), 1488)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df),
                         102)  # One more col than data b/c TIMESTAMP_MIDDLE col removed during parsing
        self.assertEqual(data_df['NEE_CUT_REF_f'].sum(), -1038.7633654702822)
        self.assertEqual(data_df.sum().sum(), 672538188.9187319)
        self.assertEqual(data_df.index[0], pd.Timestamp('2022-07-01 00:15:00'))
        self.assertEqual(data_df.index[1234], pd.Timestamp('2022-07-26 17:15:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2022-07-31 23:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')

    def test_load_exampledata_ETH_RECORD_TOA5_CSVGZ_20HZ(self):
        """Load ETH-RECORD-TOA5-CSVGZ-20HZ file"""
        data_df, metadata_df = ed.load_exampledata_ETH_RECORD_TOA5_CSVGZ_20HZ()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 14)
        self.assertEqual(len(data_df), 96)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 14)
        self.assertEqual(data_df['U'].sum(), 67.60999999999999)
        self.assertEqual(data_df.sum().sum(), 112165.0909)

    def test_load_exampledata_ETH_SONICREAD_BICO_CSVGZ_20HZ(self):
        """Load ETH-SONICREAD-BICO-CSVGZ-20HZ"""
        data_df, metadata_df = ed.load_exampledata_ETH_SONICREAD_BICO_CSVGZ_20HZ()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 16)
        self.assertEqual(len(data_df), 97)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 16)
        self.assertEqual(data_df['CO2_CONC_[IRGA75-A]'].sum(), 1464.8097999999998)
        self.assertEqual(data_df.sum().sum(), 199884.2356)

    def test_load_exampledata_ICOS_H2R_CSVZIP_10S(self):
        """Load ICOS-H2R-CSVZIP-10S file"""
        data_df, metadata_df = ed.load_exampledata_ICOS_H2R_CSVZIP_10S()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 26)
        self.assertEqual(len(data_df), 98)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 27)
        self.assertEqual(data_df.index[0], pd.Timestamp('2023-03-28 00:00:05'))
        self.assertEqual(data_df.index[37], pd.Timestamp('2023-03-28 00:06:15'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2023-03-28 00:16:15'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '10s')
        self.assertEqual(data_df['SW_IN_1_1_1'].sum(), -459.56755200000003)
        self.assertEqual(data_df.sum().sum(), 136644916.2610227)

    def test_load_exampledata_TOA5_DAT_1MIN(self):
        """Load TOA5-DAT-1MIN"""
        data_df, metadata_df = ed.load_exampledata_TOA5_DAT_1MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 95)
        self.assertEqual(len(data_df), 96)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 96)
        self.assertEqual(data_df.index[0], pd.Timestamp('2022-06-29 17:13:30'))
        self.assertEqual(data_df.index[37], pd.Timestamp('2022-06-29 17:50:30'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2022-06-29 18:48:30'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, 'min')
        self.assertEqual(data_df['PPFD_IN_M1_1_1_Avg'].sum(), 59354.815200000005)
        self.assertEqual(data_df.sum().sum(), 9590461.12998073)

    def test_load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_NS_20HZ(self):
        """Load GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-NS-20HZ file"""
        data_df, metadata_df = ed.load_exampledata_GENERIC_CSV_HEADER_1ROW_TS_MIDDLE_FULL_NS_20HZ()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 27)
        self.assertEqual(len(data_df), 98)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 28)
        self.assertEqual(data_df.index[0], pd.Timestamp('2023-05-13 08:30:00'))
        self.assertEqual(data_df.index[42], pd.Timestamp('2023-05-13 08:30:02.100000'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2023-05-13 08:30:04.850000'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '50ms')
        self.assertEqual(data_df['H2O_CONC_[IRGA75-A]'].sum(), 1641495.983)
        self.assertEqual(data_df.sum().sum(), 908013477.9108758)

    def test_load_exampledata_FLUXNET_FULLSET_HH_CSV_30MIN(self):
        """Load FLUXNET-FULLSET-HH-CSV-30MIN file"""
        data_df, metadata_df = ed.load_exampledata_FLUXNET_FULLSET_HH_CSV_30MIN()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(type(metadata_df), DataFrame)
        self.assertEqual(len(data_df.columns), 266)
        self.assertEqual(len(data_df), 99)
        self.assertEqual(len(metadata_df.columns), 4)
        self.assertEqual(len(metadata_df), 267)
        self.assertEqual(data_df.index[0], pd.Timestamp('2005-01-01 00:15:00'))
        self.assertEqual(data_df.index[42], pd.Timestamp('2005-01-01 21:15:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2005-01-03 01:15:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')
        self.assertEqual(data_df['RECO_NT_CUT_USTAR50'].sum(), 642.1616700000001)
        self.assertEqual(data_df.sum().sum(), 19849600880657.418)

    def test_exampledata_pickle(self):
        """Load pickled file"""
        data_df = ed.load_exampledata_pickle()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(len(data_df.columns), 101)
        self.assertEqual(len(data_df), 100)
        self.assertEqual(data_df.index[0], pd.Timestamp('2022-01-01 00:15:00'))
        self.assertEqual(data_df.index[37], pd.Timestamp('2022-01-01 18:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2022-01-03 01:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')
        self.assertEqual(data_df['Tair_f'].sum(), 510.6966666639999)
        self.assertEqual(data_df.sum().sum(), 44017328.76773916)

    def test_exampledata_parquet(self):
        """Load parquet file"""
        data_df = ed.load_exampledata_parquet()
        self.assertEqual(type(data_df), DataFrame)
        self.assertEqual(len(data_df.columns), 37)
        self.assertEqual(len(data_df), 175296)
        self.assertEqual(data_df.index[0], pd.Timestamp('2013-01-01 00:15:00'))
        self.assertEqual(data_df.index[37373], pd.Timestamp('2015-02-18 14:45:00'))
        self.assertEqual(data_df.index[-1], pd.Timestamp('2022-12-31 23:45:00'))
        self.assertEqual(data_df.index.name, 'TIMESTAMP_MIDDLE')
        self.assertEqual(data_df.index.freqstr, '30min')
        self.assertEqual(data_df['NEE_CUT_REF_f'].sum(), -82192.81800000001)
        self.assertEqual(data_df.sum().sum(), 328832896.8160001)


if __name__ == '__main__':
    unittest.main()
