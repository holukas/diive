import unittest

import numpy as np
import pandas as pd

from diive.qaqc import FlagQCF


class TestFlagQCF(unittest.TestCase):

    def _build(self):
        """Build a small flux series + per-test flag columns covering each QCF case."""
        idx = pd.date_range('2022-06-01', periods=6, freq='30min', name='TIMESTAMP_END')
        # Five soft/hard-capable test columns. Column names must contain '_FC_'
        # and end with '_TEST' to be picked up by FlagQCF for target_col='FC'.
        cols = [f'FLAG_FC_T{i}_TEST' for i in range(1, 6)]
        rows = [
            [0, 0, 0, 0, 0],  # all pass            -> QCF 0
            [1, 0, 0, 0, 0],  # 1 soft              -> QCF 1
            [1, 1, 0, 0, 0],  # 2 soft              -> QCF 1
            [1, 1, 1, 1, 0],  # 4 soft (>3)         -> QCF 2
            [2, 0, 0, 0, 0],  # 1 hard              -> QCF 2
            [2, 1, 1, 0, 0],  # 1 hard + 2 soft     -> QCF 2
        ]
        df = pd.DataFrame(rows, columns=cols, index=idx, dtype=float)
        df['FC'] = np.arange(1.0, 7.0)  # the flux series values 1..6
        return df

    def test_qcf_aggregation(self):
        df = self._build()
        qcf = FlagQCF(df=df, target_col='FC')
        qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        out = qcf.get()

        flag = out[qcf.flagqcfcol].to_numpy()
        self.assertEqual(list(flag), [0, 1, 1, 2, 2, 2])

        # A single hard flag must yield QCF=2 (the documented >=1 hard-flag rule).
        self.assertEqual(flag[4], 2)

    def test_filtered_series_drops_qcf2(self):
        df = self._build()
        qcf = FlagQCF(df=df, target_col='FC')
        qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        out = qcf.get()

        filtered = out[qcf.filteredseriescol]
        # QCF=2 rows (indices 3,4,5) are set to NaN; the rest keep their value.
        self.assertTrue(filtered.iloc[[3, 4, 5]].isna().all())
        self.assertTrue(np.allclose(filtered.iloc[[0, 1, 2]].to_numpy(), [1.0, 2.0, 3.0]))

        # The highest-quality series (QCF0) keeps only QCF==0 records.
        hq = out[qcf.filteredseriescol_hq]
        self.assertEqual(int(hq.notna().sum()), 1)


if __name__ == '__main__':
    unittest.main()
