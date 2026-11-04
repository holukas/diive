"""Tests for the flux footprint prediction (FFP) of Kljun et al. (2015).

The expected numbers come from the FFP Python code of the authors (version 1.42), run on
the same inputs. The full comparison (bitwise, 50 cases) is the parity harness in
diive-data/references/footprint_parity/.
"""
import unittest

import numpy as np
import pandas as pd

from diive.flux.lowres.footprint import ffp_climatology, ffp_footprint

RTOL = 1e-12

# Eight records. Record 3 (h = 9 m) and record 4 (ustar = 0.05) are invalid.
SERIES = dict(zm=[20., 22., 18., 20., 25., 15., 20., 21.],
              h=[1500., 900., 1200., 9., 2000., 600., 800., 1100.],
              ol=[-50., 200., -500., -100., 30., -20., 1000., -2000.],
              sigmav=[0.6, 0.4, 0.8, 0.6, 0.3, 1.1, 0.5, 0.7],
              ustar=[0.4, 0.3, 0.5, 0.4, 0.05, 0.25, 0.35, 0.6],
              wind_dir=[210., 225., 190., 200., 45., 260., 300., 15.])
UMEAN = [3., 4., 5., 3., 2., 6., 3., 4.]


class TestFootprintSingle(unittest.TestCase):

    def test_z0_matches_reference(self):
        res = ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15,
                            wind_dir=30., verbose=0)
        np.testing.assert_allclose(res.x_ci_max, 72.68758765611294, rtol=RTOL)
        np.testing.assert_allclose(
            res.f_ci[[0, 10, 100, 500, -1]],
            [9.830452903555068e-21, 0.0018571070028968345, 0.001189943748278819,
             7.223264520977381e-05, 1.9113813325184526e-05], rtol=RTOL)
        self.assertEqual(res.f_2d.shape, (1001, 1501))
        np.testing.assert_allclose(res.f_2d.sum(), 0.1533259468996805, rtol=RTOL)
        np.testing.assert_allclose(
            res.f_2d[100, 700:705],
            [1.206876444775538e-07, 1.4455542506445563e-07, 1.7251335241843254e-07,
             2.0512933410891955e-07, 2.4302422189780646e-07], rtol=RTOL)
        np.testing.assert_allclose([res.x_2d[100, 700], res.y_2d[100, 700]],
                                   [239.44493039203954, 165.51368491323737], rtol=RTOL)
        np.testing.assert_allclose(
            res.fr,
            [6.36492706904385e-05, 4.55605064332882e-05, 3.107597791281308e-05,
             2.0275786179173592e-05, 1.2460156025758266e-05, 6.8777003047535125e-06,
             3.2317215095801916e-06, 1.1367507878256856e-06], rtol=RTOL)
        self.assertEqual(len(res.xr[7]), 717)
        np.testing.assert_allclose(res.xr[7][:3], [18.314632647710702, 16.00391750237121,
                                                   13.76195988731779], rtol=RTOL)
        np.testing.assert_allclose(res.yr[7][:3], [16.76884825936056, 17.750914228625014,
                                                   18.852071733747913], rtol=RTOL)

    def test_umean_matches_reference(self):
        res = ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, umean=3.2,
                            wind_dir=30., rs=None, verbose=0)
        np.testing.assert_allclose(res.x_ci_max, 56.44263609195215, rtol=RTOL)
        np.testing.assert_allclose(res.f_2d.sum(), 0.254285495974613, rtol=RTOL)
        np.testing.assert_allclose(res.f_ci[100], 0.0015324256005327325, rtol=RTOL)
        self.assertIsNone(res.fr)

    def test_z0_wins_over_umean(self):
        kw = dict(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, rs=None, verbose=0)
        a = ffp_footprint(**kw, z0=0.15)
        b = ffp_footprint(**kw, z0=0.15, umean=3.2)
        np.testing.assert_array_equal(a.f_2d, b.f_2d)

    def test_footprint_lies_upwind(self):
        # Wind from the north: the footprint peak lies north of the tower
        res = ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15,
                            wind_dir=0., rs=None, verbose=0)
        i, j = np.unravel_index(np.argmax(res.f_2d), res.f_2d.shape)
        self.assertGreater(res.y_2d[i, j], 0.)
        self.assertAlmostEqual(res.x_2d[i, j], 0., delta=1.)

    def test_crop_keeps_orientation(self):
        kw = dict(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15, verbose=0)
        full = ffp_footprint(**kw)
        cropped = ffp_footprint(**kw, crop=True)
        self.assertLess(cropped.f_2d.size, full.f_2d.size)
        # x still runs along axis 0, the cropped grid is a block of the full one
        self.assertTrue(np.all(cropped.y_2d[:, 0] == cropped.y_2d[0, 0]))
        i0 = np.where(full.x_2d[:, 0] == cropped.x_2d[0, 0])[0][0]
        j0 = np.where(full.y_2d[0] == cropped.y_2d[0, 0])[0][0]
        ni, nj = cropped.f_2d.shape
        np.testing.assert_array_equal(cropped.f_2d, full.f_2d[i0:i0 + ni, j0:j0 + nj])
        # The 80 % contour lies inside the cropped grid
        self.assertGreater(cropped.xr[-1].min(), cropped.x_2d.min())
        self.assertLess(cropped.xr[-1].max(), cropped.x_2d.max())

    def test_rs_forms(self):
        kw = dict(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15, verbose=0)
        self.assertEqual(ffp_footprint(**kw, rs=80).rs, [0.8])
        self.assertEqual(ffp_footprint(**kw, rs=95).rs, [0.9])
        self.assertEqual(ffp_footprint(**kw, rs=[50., 20., 95.]).rs, [0.2, 0.5])

    def test_invalid_input_raises(self):
        kw = dict(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15)
        for change in (dict(ustar=0.05), dict(h=9.), dict(zm=2000.), dict(sigmav=0.),
                       dict(ol=-1.), dict(wind_dir=361.), dict(z0=None), dict(ustar=np.nan),
                       dict(z0=2.)):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    ffp_footprint(**{**kw, **change}, verbose=0)
        # In the roughness sublayer only with rslayer=True
        res = ffp_footprint(**{**kw, 'z0': 2.}, rslayer=True, rs=None, verbose=0)
        self.assertTrue(np.isfinite(res.x_ci_max))


class TestFootprintClimatology(unittest.TestCase):

    def test_z0_matches_reference(self):
        res = ffp_climatology(**SERIES, z0=0.2, nx=200, verbose=0)
        self.assertEqual(res.n, 6)
        np.testing.assert_array_equal(res.valid, [1, 1, 1, 0, 0, 1, 1, 1])
        self.assertEqual(res.fclim_2d.shape, (201, 201))
        np.testing.assert_allclose(res.fclim_2d.sum(), 0.008912522700363978, rtol=RTOL)
        np.testing.assert_allclose(
            res.fclim_2d[94, 96:99],
            [2.8550753313385744e-05, 3.0453638291777903e-05, 2.8899531022502e-05], rtol=RTOL)
        np.testing.assert_allclose(
            res.fr,
            [1.6445713974268354e-05, 1.1131434730762096e-05, 7.653954722274407e-06,
             4.916029682841799e-06, 2.844484502191049e-06, 1.5111685668716127e-06,
             6.617051078401439e-07, 2.0978478343427679e-07], rtol=RTOL)
        self.assertEqual(len(res.xr[4]), 129)
        np.testing.assert_allclose(res.xr[4][:3], [-60.00000000000001, -50.64150299202055, -50.0],
                                   rtol=RTOL)
        np.testing.assert_allclose(res.yr[4][:3], [-230.5589574022812, -230.0, -229.96705695542116],
                                   rtol=RTOL)

    def test_umean_unsmoothed_matches_reference(self):
        res = ffp_climatology(**SERIES, umean=UMEAN, nx=200, smooth_data=False, rs=None, verbose=0)
        self.assertEqual(res.n, 6)
        np.testing.assert_allclose(res.fclim_2d.sum(), 0.008894159246219651, rtol=RTOL)
        np.testing.assert_allclose(res.fclim_2d.max(), 3.3391206717052126e-05, rtol=RTOL)

    def test_series_with_missing_values(self):
        # A record with a missing input is skipped, as if it were not there
        df = pd.DataFrame(SERIES)
        df.loc[2, 'ol'] = np.nan
        res = ffp_climatology(**{c: df[c] for c in df}, z0=0.2, nx=200, rs=None, verbose=0)
        dropped = df.drop(index=2)
        ref = ffp_climatology(**{c: dropped[c] for c in dropped}, z0=0.2, nx=200, rs=None, verbose=0)
        self.assertEqual(res.n, 5)
        self.assertFalse(res.valid[2])
        np.testing.assert_array_equal(res.fclim_2d, ref.fclim_2d)

    def test_worker_processes_give_same_result(self):
        rng = np.random.default_rng(5)
        n = 120
        kw = dict(zm=20., h=rng.uniform(300., 2000., n), ol=rng.uniform(-500., -20., n),
                  sigmav=rng.uniform(0.3, 1., n), ustar=rng.uniform(0.2, 0.7, n),
                  wind_dir=rng.uniform(0., 360., n), z0=0.2, nx=150, rs=None, verbose=0)
        one = ffp_climatology(**kw, n_jobs=1)
        two = ffp_climatology(**kw, n_jobs=2)
        np.testing.assert_array_equal(one.fclim_2d, two.fclim_2d)

    def test_domain_from_dx_and_nx(self):
        res = ffp_climatology(**SERIES, z0=0.2, dx=4., nx=100, rs=None, verbose=0)
        self.assertEqual(res.x_2d.shape, (101, 101))
        self.assertEqual((res.x_2d.min(), res.x_2d.max()), (-200., 200.))

    def test_crop(self):
        full = ffp_climatology(**SERIES, z0=0.2, nx=200, verbose=0)
        cropped = ffp_climatology(**SERIES, z0=0.2, nx=200, crop=True, verbose=0)
        self.assertLess(cropped.fclim_2d.size, full.fclim_2d.size)
        self.assertGreater(cropped.xr[-1].min(), cropped.x_2d.min())
        self.assertLess(cropped.yr[-1].max(), cropped.y_2d.max())

    def test_errors(self):
        with self.assertRaises(ValueError):
            ffp_climatology(**{**SERIES, 'ol': SERIES['ol'][:-1]}, z0=0.2, nx=200, verbose=0)
        with self.assertRaises(ValueError):
            ffp_climatology(**{**SERIES, 'ustar': [0.05] * 8}, z0=0.2, nx=200, verbose=0)
        with self.assertRaises(ValueError):
            ffp_climatology(**SERIES, nx=200, verbose=0)


if __name__ == '__main__':
    unittest.main()
