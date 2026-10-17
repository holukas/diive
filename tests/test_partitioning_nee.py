import unittest

import pandas as pd


class TestPartitioningNEE(unittest.TestCase):
    """PartitioningNEE routes to the four backends and returns their results unchanged."""

    @classmethod
    def setUpClass(cls):
        import diive as dv
        df = dv.load_exampledata_parquet()
        # One summer month is enough for the daytime fits. The nighttime
        # methods need a longer record to constrain E0, so they get a year.
        cls.df = df.loc['2017-06-01':'2017-06-30'].copy()
        cls.year = df.loc[df.index.year == 2017].copy()
        cls.site = dict(lat=46.815, lon=9.855, utc_offset=1)

    def _frame(self, method='daytime'):
        return self.year if method == 'nighttime' else self.df

    def _all_inputs(self, method='daytime'):
        df = self._frame(method)
        return dict(ta=df['Tair_orig'], sw_in=df['Rg_orig'], ta_f=df['Tair_f'],
                    sw_in_f=df['Rg_f'], nee_f=df['NEE_CUT_REF_f'],
                    vpd=df['VPD_f'], **self.site)

    def _direct(self, method, mode):
        from diive.flux.partitioning import (
            DaytimePartitioningOneFlux, DaytimePartitioningReddyProc,
            NighttimePartitioningOneFlux, NighttimePartitioningReddyProc)
        df, s = self._frame(method), self.site
        nee = df['NEE_CUT_REF_orig']
        if (method, mode) == ('nighttime', 'oneflux'):
            return NighttimePartitioningOneFlux(
                nee=nee, ta=df['Tair_orig'], sw_in=df['Rg_orig'],
                nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], lat=s['lat'],
                verbose=0).run()
        if (method, mode) == ('nighttime', 'reddyproc'):
            return NighttimePartitioningReddyProc(
                nee=nee, ta=df['Tair_orig'], sw_in=df['Rg_orig'],
                nee_f=df['NEE_CUT_REF_f'], ta_f=df['Tair_f'], verbose=0, **s).run()
        if (method, mode) == ('daytime', 'oneflux'):
            return DaytimePartitioningOneFlux(
                nee=nee, ta=df['Tair_orig'], sw_in=df['Rg_orig'], ta_f=df['Tair_f'],
                sw_in_f=df['Rg_f'], vpd=df['VPD_f'], nee_f=df['NEE_CUT_REF_f'],
                verbose=0).run()
        return DaytimePartitioningReddyProc(
            nee=nee, ta=df['Tair_f'], vpd=df['VPD_f'], sw_in=df['Rg_f'],
            verbose=0, **s).run()

    def test_all_combinations_equal_backend(self):
        from diive.flux.partitioning import PartitioningNEE
        expected_cls = {
            ('nighttime', 'oneflux'): ('NighttimePartitioningOneFlux', '_NT_OF'),
            ('nighttime', 'reddyproc'): ('NighttimePartitioningReddyProc', '_NT_RP'),
            ('daytime', 'oneflux'): ('DaytimePartitioningOneFlux', '_DT_OF'),
            ('daytime', 'reddyproc'): ('DaytimePartitioningReddyProc', '_DT_RP'),
        }
        for (method, mode), (cls_name, suffix) in expected_cls.items():
            with self.subTest(method=method, mode=mode):
                part = PartitioningNEE(self._frame(method)['NEE_CUT_REF_orig'],
                                       method=method, mode=mode, verbose=0,
                                       **self._all_inputs(method)).run()
                direct = self._direct(method, mode)
                self.assertEqual(type(part.backend).__name__, cls_name)
                self.assertEqual((part.method, part.mode), (method, mode))
                pd.testing.assert_frame_equal(part.results, direct.results,
                                              check_exact=True)
                pd.testing.assert_series_equal(part.reco, direct.results[f'RECO{suffix}'],
                                               check_exact=True)
                pd.testing.assert_series_equal(part.gpp, direct.results[f'GPP{suffix}'],
                                               check_exact=True)
                # The month must give values, else equality would be trivial.
                self.assertGreater(part.reco.notna().sum(), 0)
                self.assertGreater(part.gpp.notna().sum(), 0)

    def test_default_mode_is_oneflux(self):
        from diive.flux.partitioning import PartitioningNEE
        part = PartitioningNEE(self.year['NEE_CUT_REF_orig'], method='nighttime',
                               verbose=0, **self._all_inputs('nighttime'))
        self.assertEqual(part.mode, 'oneflux')
        self.assertEqual(type(part.backend).__name__, 'NighttimePartitioningOneFlux')

    def test_functional_wrapper(self):
        from diive.flux.partitioning import partition_nee
        res = partition_nee(self.df['NEE_CUT_REF_orig'], method='daytime',
                            mode='reddyproc', verbose=0, **self._all_inputs())
        pd.testing.assert_frame_equal(res, self._direct('daytime', 'reddyproc').results,
                                      check_exact=True)

    def test_backend_option_passed_through(self):
        from diive.flux.partitioning import PartitioningNEE
        df = self.df
        part = PartitioningNEE(df['NEE_CUT_REF_orig'], method='daytime',
                               verbose=0, reject_alpha_at_start=True,
                               **self._all_inputs())
        self.assertTrue(part.backend.reject_alpha_at_start)
        nee_sd = (df['NEE_CUT_REF_orig'].abs() * 0.2).clip(lower=0.7)
        part = PartitioningNEE(df['NEE_CUT_REF_orig'], method='daytime',
                               mode='reddyproc', verbose=0, nee_sd=nee_sd,
                               **self._all_inputs())
        self.assertIn('nee_sd', part.backend._inputs.columns)

    def test_unknown_backend_option_rejected(self):
        from diive.flux.partitioning import PartitioningNEE
        inputs = self._all_inputs()
        nee = self.df['NEE_CUT_REF_orig']
        # nee_sd belongs to daytime REddyProc only.
        with self.assertRaisesRegex(TypeError, 'nee_sd'):
            PartitioningNEE(nee, method='daytime', mode='oneflux',
                            nee_sd=nee, **inputs)
        with self.assertRaisesRegex(TypeError, 'reject_alpha_at_start'):
            PartitioningNEE(nee, method='nighttime', mode='reddyproc',
                            reject_alpha_at_start=True, **inputs)

    def test_missing_inputs_named(self):
        from diive.flux.partitioning import PartitioningNEE
        df = self.df
        nee = df['NEE_CUT_REF_orig']
        with self.assertRaisesRegex(ValueError, r'Missing: nee_f, lat\.'):
            PartitioningNEE(nee, method='nighttime', ta=df['Tair_orig'],
                            sw_in=df['Rg_orig'], ta_f=df['Tair_f'])
        with self.assertRaisesRegex(ValueError, r'Missing: lon, utc_offset\.'):
            PartitioningNEE(nee, method='nighttime', mode='reddyproc',
                            ta=df['Tair_orig'], sw_in=df['Rg_orig'],
                            ta_f=df['Tair_f'], nee_f=df['NEE_CUT_REF_f'], lat=46.8)
        with self.assertRaisesRegex(ValueError, r'Missing: ta, sw_in\.'):
            PartitioningNEE(nee, method='daytime', ta_f=df['Tair_f'],
                            sw_in_f=df['Rg_f'], vpd=df['VPD_f'])
        # Daytime REddyProc does not need the measured drivers.
        PartitioningNEE(nee, method='daytime', mode='reddyproc', ta_f=df['Tair_f'],
                        sw_in_f=df['Rg_f'], vpd=df['VPD_f'], **self.site)

    def test_invalid_method_or_mode(self):
        from diive.flux.partitioning import PartitioningNEE
        nee = self.df['NEE_CUT_REF_orig']
        with self.assertRaisesRegex(ValueError, 'method'):
            PartitioningNEE(nee, method='both', **self._all_inputs())
        with self.assertRaisesRegex(ValueError, 'mode'):
            PartitioningNEE(nee, method='daytime', mode='diive', **self._all_inputs())

    def test_results_before_run_raises(self):
        from diive.flux.partitioning import PartitioningNEE
        part = PartitioningNEE(self.df['NEE_CUT_REF_orig'], method='nighttime',
                               **self._all_inputs())
        with self.assertRaises(RuntimeError):
            _ = part.results

    def test_partitioning_inputs(self):
        from diive.flux.partitioning import partitioning_inputs
        self.assertEqual(partitioning_inputs('daytime', 'reddyproc'),
                         ('ta_f', 'sw_in_f', 'vpd', 'lat', 'lon', 'utc_offset'))
        self.assertEqual(partitioning_inputs('nighttime'),
                         ('ta', 'sw_in', 'nee_f', 'ta_f', 'lat'))

    def test_partitioning_optional_inputs(self):
        from diive.flux.partitioning import partitioning_optional_inputs
        self.assertEqual(partitioning_optional_inputs('daytime', 'oneflux'), ('nee_f',))
        self.assertEqual(partitioning_optional_inputs('daytime', 'reddyproc'), ('nee_sd',))
        self.assertEqual(partitioning_optional_inputs('nighttime', 'oneflux'), ())
        self.assertEqual(partitioning_optional_inputs('nighttime', 'reddyproc'), ())

    def test_daytime_oneflux_takes_nee_f_only_when_given(self):
        # Daytime ONEFlux reads the gap-filled NEE for windows without measured
        # nighttime NEE (L215); it is optional, so leaving it out is no error.
        from diive.flux.partitioning import PartitioningNEE
        inputs = self._all_inputs()
        nee = self.df['NEE_CUT_REF_orig']
        with_nee_f = PartitioningNEE(nee, method='daytime', verbose=0, **inputs)
        self.assertIn('nee_f', with_nee_f.backend._inputs.columns)
        inputs['nee_f'] = None
        without = PartitioningNEE(nee, method='daytime', verbose=0, **inputs)
        self.assertNotIn('nee_f', without.backend._inputs.columns)


if __name__ == '__main__':
    unittest.main()
