import unittest

import numpy as np
import pandas as pd

from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

FIELD = 'SWC_FF1_0.05_1'
TAGS = {'units': '%', 'varname': FIELD, 'site': 'CH-LAE', 'hpos': 'FF1', 'vpos': '0.05',
        'repl': '1', 'freq': '10min', 'data_version': 'raw', 'gain': '100.0', 'offset': '0.0'}


def _detailed(index) -> pd.DataFrame:
    """Build a data_detailed frame (one variable plus its database tags)."""
    index.name = 'TIMESTAMP_END'
    df = pd.DataFrame(index=index)
    df[FIELD] = np.linspace(30.0, 35.0, len(index))
    for tag, value in TAGS.items():
        df[tag] = value
    return df


def _screening(df: pd.DataFrame) -> StepwiseMeteoScreeningDb:
    return StepwiseMeteoScreeningDb(site='ch-lae', data_detailed={FIELD: df}, fields=[FIELD],
                                    site_lat=47.478333, site_lon=8.364389, utc_offset=1)


class TestMixedTimeResolution(unittest.TestCase):
    """Input whose raw resolution changes partway through, e.g. after a logger
    program change. The coarse era is upsampled onto the finest grid."""

    def _mixed(self):
        # Two days at 10MIN followed by two days at 1MIN.
        coarse = pd.date_range('2020-04-10 15:00', '2020-04-12 15:00', freq='10min')
        fine = pd.date_range('2020-04-12 15:01', '2020-04-14 15:00', freq='1min')
        return _detailed(coarse.union(fine)), coarse, fine

    def test_mixed_resolutions_are_harmonized(self):
        df, coarse, fine = self._mixed()
        out = _screening(df).data_detailed[FIELD]

        # Everything ends up on the finest resolution, on middle timestamps.
        self.assertEqual(out.index.freqstr, 'min')
        self.assertEqual(out.index.name, 'TIMESTAMP_MIDDLE')

        # No gaps introduced: the upsampled era is fully back-filled.
        early = out.loc[:'2020-04-12 14:00', FIELD]
        self.assertEqual(early.isna().sum(), 0)

        # A 10MIN record is valid for ten 1MIN records, so values repeat in runs of 10.
        runs = early.groupby((early != early.shift()).cumsum()).size()
        self.assertEqual(runs.max(), 10)

        # The values themselves are untouched, only repeated: every value in the
        # upsampled era is one that was actually measured in the coarse era.
        self.assertTrue(set(early.unique()).issubset(set(df.loc[coarse, FIELD])))

    def test_single_resolution_is_left_alone(self):
        """The common case: one resolution, so no upsampling happens."""
        index = pd.date_range('2020-04-12 15:01', '2020-04-14 15:00', freq='1min')
        df = _detailed(index)
        out = _screening(df).data_detailed[FIELD]
        self.assertEqual(out.index.freqstr, 'min')
        self.assertEqual(len(out), len(index))
        self.assertEqual(out[FIELD].isna().sum(), 0)


if __name__ == '__main__':
    unittest.main()
