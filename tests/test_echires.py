import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import diive as dv
from diive.configs.exampledata import load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz


class TestEcHires(unittest.TestCase):

    def test_flux_detection_limit(self):
        df = load_exampledata_GENERIC_TXT_EDDY_COVARIANCE_10Hz()
        df = df[['x', 'y', 'z', 'N2Od', 'Ts', 'H2O']].copy()
        df['pressure'] = 100000
        df['N2Od'] = df['N2Od'].multiply(10 ** 3)  # Convert from umol mol-1 to nmol mol-1
        df['H2O'] = df['H2O'].div(10 ** 6)  # Convert from umol mol-1 to mol mol-1
        df['Ts'] = df['Ts'].add(273.15)  # From degC to K
        fdl = dv.flux.FluxDetectionLimit(
            df=df,
            u_col='x',  # m s-1
            v_col='y',  # m s-1
            w_col='z',  # m s-1
            c_col='N2Od',  # nmol mol-1 (ppb)
            ts_col='Ts',  # degC
            h2o_col='H2O',  # mol mol-1
            press_col='pressure',  # Pa
            noise_range=20,  # seconds
            default_lag=2.8,  # seconds
            lag_range=[-180, 180],  # seconds, calculate covariance for all steps between -180s and +180s
            lag_stepsize=1,  # number of records, step size for lag search
            sampling_rate=10,  # Hz
            create_covariance_plot=True,
            title_covariance_plot="Covariance vs time lag for example file")
        fdl.run()
        results = fdl.get_detection_limit()
        self.assertEqual(len(fdl.hires_df.columns), 11)
        self.assertIn("e", fdl.hires_df.columns)  # e_col
        self.assertIn("pd", fdl.hires_df.columns)  # pd_col
        self.assertIsInstance(fdl.hires_df, pd.DataFrame)
        self.assertEqual(fdl.lag_from, -1800)
        self.assertEqual(fdl.lag_to, 1800)
        self.assertIn('flux_detection_limit', results)
        self.assertIn('flux_noise_rmse', results)
        self.assertAlmostEqual(results['flux_detection_limit'], 1.9300179626373497, places=10)
        self.assertAlmostEqual(results['flux_noise_rmse'], 0.6433393208791166, places=10)

        # Verify that the flux conversion factor is applied correctly
        flux_conversion_factor = fdl.cov_df['cov_flux'] / fdl.cov_df['cov']
        calculated_flux_conversion_factor = (
                1 / ((8.31446261815324 * fdl.hires_df['Ts'].mean()) / fdl.hires_df['pd'].mean()))
        # Ensure the calculated factor matches the expected value
        self.assertEqual(flux_conversion_factor.iloc[0], calculated_flux_conversion_factor)

        # Ensure that the flux detection limit and noise RMSE are calculated correctly
        detection_limit = fdl.results['flux_detection_limit']
        noise_rmse = fdl.results['flux_noise_rmse']
        # Check the detection limit follows 3 * RMSE rule
        self.assertEqual(detection_limit, 3 * noise_rmse)

    def test_reynolds_decomposition(self):
        import numpy as np
        from diive.flux import reynolds_decomposition
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name='w')
        xprime = reynolds_decomposition(x)
        # x' = x - mean(x); fluctuations sum to (approximately) zero
        self.assertAlmostEqual(xprime.mean(), 0.0, places=10)
        self.assertTrue(np.allclose(xprime.to_numpy(), x.to_numpy() - x.mean()))
        self.assertEqual(xprime.name, 'w')

    def test_wind_double_rotation(self):
        import numpy as np
        from diive.flux import WindDoubleRotation
        rng = np.random.RandomState(0)
        n = 2000
        # Mean wind tilted in both horizontal and vertical: u has offset, v and w
        # carry a mean (tilt) plus turbulence.
        u = pd.Series(3.0 + rng.normal(0, 0.5, n), name='u')
        v = pd.Series(1.0 + rng.normal(0, 0.5, n), name='v')
        w = pd.Series(0.4 + rng.normal(0, 0.2, n), name='w')
        wr = WindDoubleRotation(u=u, v=v, w=w)
        # After double rotation: mean(v2) and mean(w2) are ~0 (defining property),
        # and the rotated streamwise component aligns with the mean horizontal wind.
        self.assertAlmostEqual(wr.v2.mean(), 0.0, places=8)
        self.assertAlmostEqual(wr.w2.mean(), 0.0, places=8)
        self.assertGreater(wr.u2.mean(), 0.0)


class TestPwbPerGasWindow(unittest.TestCase):
    """Per-gas time-lag search windows for PWB (lws/uws + per_gas_lag)."""

    @staticmethod
    def _lag(sig, k):
        import numpy as np
        return np.r_[np.zeros(k), sig[:-k]]

    def _core_fixture(self, scalar_lag_records, noise=0.5, seed=0, n=6000):
        """Synthetic (w, scalar, t_sonic) with the scalar lagged behind w."""
        import numpy as np
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n)
        s = self._lag(w, scalar_lag_records) + noise * rng.standard_normal(n)
        t = 0.7 * w + 0.3 * rng.standard_normal(n)
        return pd.DataFrame({'w': w, 's': s, 't': t})

    def _detect(self, df, **kw):
        from diive.flux.hires.lag_pwb import PreWhiteningBootstrap
        pwb = PreWhiteningBootstrap(df, 'w', 's', 't', hz=20, lag_max_s=10,
                                    n_bootstrap=29, random_state=123, **kw)
        pwb.run()
        return pwb.results['tlag_s']

    # ---- compact LABEL:column@... spec parser ----
    def test_parse_scalar_spec(self):
        from diive.flux.hires.detect_and_remove_tlag import parse_scalar_spec
        self.assertEqual(parse_scalar_spec('CH4:ch4'), ('CH4', 'ch4', {}))
        # column names with brackets survive; @ introduces per-gas overrides
        self.assertEqual(
            parse_scalar_spec('H2O:H2O_DRY_[LGR-A]@lag=30;uws=25'),
            ('H2O', 'H2O_DRY_[LGR-A]', {'lag_max_s': 30.0, 'uws': 25.0}))
        self.assertEqual(
            parse_scalar_spec('N2O:n2o@lag=15;lws=0;uws=10;block=30'),
            ('N2O', 'n2o',
             {'lag_max_s': 15.0, 'lws': 0.0, 'uws': 10.0, 'block_length_s': 30.0}))
        for bad in ('noColon', 'X:y@foo=1', 'X:y@lag', 'X:y@lag=abc', ':y', 'X:'):
            with self.assertRaises(ValueError):
                parse_scalar_spec(bad)

    # ---- per-gas resolution + R block-coupling ----
    def test_resolve_gas_lag(self):
        from diive.flux.hires.detect_and_remove_tlag import _resolve_gas_lag
        ov = {'H2O': {'lag_max_s': 30.0, 'uws': 25.0}}
        # Overriding lag_max re-couples the block to 2*lag_max (R: l=LAG.MAX*2).
        self.assertEqual(_resolve_gas_lag('H2O', 10.0, 20.0, None, None, ov),
                         (30.0, 60.0, None, 25.0))
        # A gas without an entry uses the global values.
        self.assertEqual(_resolve_gas_lag('CH4', 10.0, 20.0, None, None, ov),
                         (10.0, 20.0, None, None))
        # Global lws/uws are inherited when no per-gas window is set.
        self.assertEqual(_resolve_gas_lag('CH4', 10.0, 20.0, 0.0, 5.0, ov),
                         (10.0, 20.0, 0.0, 5.0))
        # An explicit block override is kept; lag stays global.
        self.assertEqual(
            _resolve_gas_lag('X', 10.0, 20.0, None, None,
                             {'X': {'block_length_s': 33.0}}),
            (10.0, 33.0, None, None))

    # ---- core: an explicit full window must equal the no-window default ----
    def test_window_full_equals_default(self):
        df = self._core_fixture(30)  # scalar lags w by +1.5 s
        default = self._detect(df)
        full = self._detect(df, lws=-10, uws=10)  # full symmetric window
        self.assertAlmostEqual(default, 1.5, places=6)
        self.assertEqual(default, full)

    # ---- core: a window clips a lag that lies outside it ----
    def test_window_clips_outside_lag(self):
        df = self._core_fixture(160, noise=0.3)  # +8 s lag
        default = self._detect(df)
        clipped = self._detect(df, lws=0, uws=3)  # window excludes the 8 s peak
        self.assertAlmostEqual(default, 8.0, places=6)
        self.assertLessEqual(clipped, 3.0)
        self.assertNotAlmostEqual(clipped, 8.0, places=3)

    # ---- an edge-pinned detection is a failed detection (never applied) ----
    def test_edge_pinned_detection_rejected(self):
        import numpy as np
        from diive.flux.hires.lag_pwb import PreWhiteningBootstrap
        df = self._core_fixture(30)
        pwb = PreWhiteningBootstrap(df, 'w', 's', 't', hz=20, lag_max_s=10,
                                    n_bootstrap=29, random_state=123)
        # Symmetric +/-10 s window -> edges at +/-200 records. A zero-width HDI
        # (every replicate agreed) is exactly what a real edge pin produces.
        pwb._win_lo_idx, pwb._win_hi_idx, pwb._lag_max_records = 0, 400, 200
        pwb._hdi_lo_s = pwb._hdi_hi_s = 0.0
        # Interior mode -> usable and reliable.
        pwb._tlag_records = 16
        self.assertFalse(pwb.is_edge_pinned)
        self.assertAlmostEqual(pwb.tlag_s, 0.8)
        self.assertTrue(pwb.is_reliable)
        # Edge mode (window boundary) -> rejected: NaN lag + HDI, not reliable.
        pwb._tlag_records = -200
        self.assertTrue(pwb.is_edge_pinned)
        self.assertTrue(np.isnan(pwb.tlag_s))
        self.assertFalse(pwb.is_reliable)
        self.assertTrue(np.isnan(pwb.hdi_range_s))
        # Per-gas window [0, 5] s -> edges at its OWN bounds (0 and 100 records),
        # not the global +/-10 s.
        pwb._win_lo_idx, pwb._win_hi_idx = 200, 300
        pwb._tlag_records = 0    # lower bound (0 s)
        self.assertTrue(pwb.is_edge_pinned)
        pwb._tlag_records = 100  # upper bound (5 s)
        self.assertTrue(pwb.is_edge_pinned)
        pwb._tlag_records = 60   # 3 s, interior -> kept
        self.assertFalse(pwb.is_edge_pinned)
        self.assertAlmostEqual(pwb.tlag_s, 3.0)

    # ---- pipeline rejects a malformed per_gas_lag ----
    def test_pipeline_rejects_bad_per_gas_lag(self):
        import tempfile
        from pathlib import Path
        from diive.flux.hires.detect_and_remove_tlag import PerFilePipeline
        d = Path(tempfile.mkdtemp())
        scalars = {'CH4': 'ch4', 'H2O': 'h2o'}
        with self.assertRaises(ValueError):  # unknown gas label
            PerFilePipeline(d, d, 'u', 'v', 'w', 'ts', scalars,
                            per_gas_lag={'XX': {'lag_max_s': 30}})
        with self.assertRaises(ValueError):  # unknown override key
            PerFilePipeline(d, d, 'u', 'v', 'w', 'ts', scalars,
                            per_gas_lag={'H2O': {'bogus': 1}})

    # ---- window [lws, uws] -> PWB lag params (lag_max + coupled block) ----
    def test_window_to_lag_params(self):
        from diive.flux.hires.detect_and_remove_tlag import window_to_lag_params
        # Asymmetric long-inlet window: lag_max = upper bound, block = 2x.
        self.assertEqual(
            window_to_lag_params(0.0, 25.0),
            {'lag_max_s': 25.0, 'block_length_s': 50.0, 'lws': 0.0, 'uws': 25.0})
        # Symmetric window reduces to a plain lag_max with block = 2x.
        self.assertEqual(
            window_to_lag_params(-10.0, 10.0),
            {'lag_max_s': 10.0, 'block_length_s': 20.0, 'lws': -10.0, 'uws': 10.0})
        # The larger absolute bound drives lag_max.
        self.assertEqual(window_to_lag_params(-30.0, 5.0)['lag_max_s'], 30.0)
        # A narrow window floors the block at the paper's 20 s (not 2*half=10).
        self.assertEqual(
            window_to_lag_params(0.0, 5.0),
            {'lag_max_s': 5.0, 'block_length_s': 20.0, 'lws': 0.0, 'uws': 5.0})
        with self.assertRaises(ValueError):
            window_to_lag_params(10.0, 5.0)  # upper <= lower

    # ---- TUI: the Win field auto-syncs to the selected scalars ----
    def test_cli_writes_tui_loadable_settings_yaml(self):
        import tempfile, yaml
        from pathlib import Path
        from diive.flux.hires.detect_and_remove_tlag import (
            _build_parser, parse_scalar_spec)
        try:
            from diive.flux.hires.detect_and_remove_tlag_tui import (
                write_run_settings_yaml, parse_win_ranges,
                _FIELD_IDS, _SWITCHES)
        except Exception:
            self.skipTest('textual TUI not importable')

        args = _build_parser().parse_args([
            '--input-dir', '.', '--output-dir', '.',
            '--col-u', 'u', '--col-v', 'v', '--col-w', 'w', '--col-tsonic', 'ts',
            '--scalar', 'CH4:ch4', '--scalar', 'H2O:h2o@lws=0;uws=20',
            '--lag-max', '10', '--hdi-prefilter', '1.0',
            '--lws', '0', '--uws', '5', '--random-state', '42', '--save-plots',
        ])
        scalars, pgl = {}, {}
        for tok in args.scalars:
            lbl, col, ov = parse_scalar_spec(tok)
            scalars[lbl] = col
            if ov:
                pgl[lbl] = ov

        out = Path(tempfile.mkdtemp())
        path = write_run_settings_yaml(out, args, scalars, pgl)
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).name, 'detect_remove_tui_settings.yaml')
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))

        # Every key must be one the TUI's loader recognises (no silent drops).
        known = set(_FIELD_IDS) | set(_SWITCHES)
        self.assertTrue(set(data).issubset(known), set(data) - known)
        # Scalars carry columns only; windows live in the 'Win s' field.
        self.assertEqual(data['scalars'], 'CH4:ch4,H2O:h2o')
        # Per-gas override (H2O) and the global window inherited by CH4 both
        # reconstruct, and parse straight back through the TUI loader.
        self.assertEqual(parse_win_ranges(data['winranges']),
                         {'CH4': (0.0, 5.0), 'H2O': (0.0, 20.0)})
        self.assertEqual(data['randomstate'], '42')
        self.assertTrue(data['saveplots'])

    def test_tui_win_field_autosync(self):
        try:
            import asyncio
            from diive.flux.hires.detect_and_remove_tlag_tui import DetectRemoveTUI
            from textual.widgets import Input, Button
        except Exception:
            self.skipTest('textual TUI not importable')

        async def scenario():
            app = DetectRemoveTUI(demo=False)
            async with app.run_test(size=(120, 60)) as pilot:
                await pilot.pause()
                win = lambda: app.query_one('#winranges', Input).value
                scal = app.query_one('#scalars', Input)

                # Typing scalars seeds a symmetric window per gas from Lag max.
                scal.value = 'CH4:ch4,N2O:n2o'
                await pilot.pause()
                self.assertEqual(win(), 'CH4:[-10,10],N2O:[-10,10]')

                # Editing a window + changing Lag max + adding a gas: the edit is
                # preserved, the new gas is seeded at the new Lag max.
                app.query_one('#winranges', Input).value = \
                    'CH4:[-10,10],N2O:[-10,10],H2O:[0,25]'
                app.query_one('#lagmax', Input).value = '15'
                scal.value = 'CH4:ch4,N2O:n2o,H2O:h2o,CO2:co2'
                await pilot.pause()
                self.assertEqual(
                    win(), 'CH4:[-10,10],N2O:[-10,10],H2O:[0,25],CO2:[-15,15]')

                # Removing a gas drops its window.
                scal.value = 'CH4:ch4,H2O:h2o,CO2:co2'
                await pilot.pause()
                self.assertEqual(win(), 'CH4:[-10,10],H2O:[0,25],CO2:[-15,15]')

                # The reseed button rewrites all windows to the symmetric default.
                app.query_one('#reseed_winranges', Button).press()
                await pilot.pause()
                self.assertEqual(
                    win(), 'CH4:[-15,15],H2O:[-15,15],CO2:[-15,15]')

                # _collect turns the Win field into per-gas lag params; the
                # Scalars field stays a pure {label: column} map.
                app.query_one('#winranges', Input).value = \
                    'CH4:[-10,10],H2O:[0,25],CO2:[-10,10]'
                app.query_one('#input_dir', Input).value = '.'
                app.query_one('#output_dir', Input).value = '.'
                cfg = app._collect()
                self.assertEqual(set(cfg['scalars']), {'CH4', 'H2O', 'CO2'})
                self.assertEqual(
                    cfg['per_gas_lag']['H2O'],
                    {'lag_max_s': 25.0, 'block_length_s': 50.0,
                     'lws': 0.0, 'uws': 25.0})
                self.assertNotIn('lws', cfg)
                self.assertNotIn('block_length_s', cfg)

        asyncio.run(scenario())

    # ---- end-to-end: a per-gas window finds a lag a global one cannot ----
    def test_pipeline_per_gas_window_end_to_end(self):
        import numpy as np
        import tempfile
        from pathlib import Path
        from diive.flux.hires.detect_and_remove_tlag import PerFilePipeline
        rng = np.random.default_rng(1)
        m = 7000
        w = rng.standard_normal(m)
        df = pd.DataFrame({
            'u': rng.standard_normal(m), 'v': rng.standard_normal(m), 'w': w,
            'ts': 0.8 * w + 0.2 * rng.standard_normal(m),
            'ch4': self._lag(w, 20) + 0.2 * rng.standard_normal(m),    # 1 s
            'h2o': self._lag(w, 240) + 0.2 * rng.standard_normal(m),   # 12 s, > default 10 s window
        })

        def run(per_gas_lag):
            ind = Path(tempfile.mkdtemp())
            df.to_csv(ind / 'site_202401010000.csv', index=False)
            out = Path(tempfile.mkdtemp())
            pipe = PerFilePipeline(
                ind, out, 'u', 'v', 'w', 'ts', {'CH4': 'ch4', 'H2O': 'h2o'},
                hz=20, n_bootstrap=19, chunk_seconds=300, min_chunk_seconds=100,
                extra_rows=0, n_workers=1, random_state=42,
                per_gas_lag=per_gas_lag)
            row = pipe.run().iloc[0]
            self.assertTrue(any((out / '2_lag_removed').glob('*.csv')))
            return float(row['ch4_tlag_s']), float(row['h2o_tlag_s'])

        ch4_def, h2o_def = run(None)
        ch4_pg, h2o_pg = run({'H2O': {'lag_max_s': 20}})

        # CH4's ~1 s lag is recovered in both runs (inside the default window).
        self.assertAlmostEqual(ch4_def, 1.0, places=1)
        self.assertAlmostEqual(ch4_pg, 1.0, places=1)
        # The default +/-10 s window cannot reach H2O's 12 s lag...
        self.assertLess(h2o_def, 11.0)
        # ...but a per-gas wide window recovers it (~12 s), CH4 untouched.
        self.assertGreater(h2o_pg, 11.0)


class TestWindRotationInvariants(unittest.TestCase):
    """Properties of the double rotation that the mean-based test cannot see.

    Checking only that mean(v2) and mean(w2) are ~0 leaves a sign error or a
    swapped trig term free to pass, because several wrong rotations still
    zero those means.
    """

    @staticmethod
    def _tilted_wind(n: int = 500):
        import numpy as np
        rng = np.random.default_rng(0)
        return (pd.Series(3.0 + rng.normal(0, 0.5, n), name='u'),
                pd.Series(1.0 + rng.normal(0, 0.5, n), name='v'),
                pd.Series(0.4 + rng.normal(0, 0.2, n), name='w'))

    def test_rotation_preserves_wind_speed(self):
        import numpy as np
        from diive.flux import WindDoubleRotation
        u, v, w = self._tilted_wind()
        wr = WindDoubleRotation(u=u, v=v, w=w)
        # Both rotations are orthogonal, so the 3D speed of every single sample
        # is unchanged. This is what pins the trig down.
        before = u ** 2 + v ** 2 + w ** 2
        after = wr.u2 ** 2 + wr.v2 ** 2 + wr.w2 ** 2
        self.assertTrue(np.allclose(before.to_numpy(), after.to_numpy()))

    def test_already_aligned_wind_is_untouched(self):
        import numpy as np
        from diive.flux import WindDoubleRotation
        # Mean wind exactly along +x, no crosswind, no tilt -> nothing to rotate.
        u = pd.Series([5.0, 6.0, 4.0, 5.5], name='u')
        zero = pd.Series([0.0] * 4, name='z')
        wr = WindDoubleRotation(u=u, v=zero, w=zero)
        self.assertAlmostEqual(wr.theta, 0.0, places=12)
        self.assertAlmostEqual(wr.phi, 0.0, places=12)
        self.assertTrue(np.allclose(wr.u2.to_numpy(), u.to_numpy()))
        self.assertTrue(np.allclose(wr.v2.to_numpy(), 0.0))
        self.assertTrue(np.allclose(wr.w2.to_numpy(), 0.0))

    def test_rotation_angle_for_crosswind_only(self):
        import math
        from diive.flux import WindDoubleRotation
        # Mean wind purely along +y: the first rotation must turn by +pi/2, and
        # the rotated streamwise component must carry the full wind speed.
        zero = pd.Series([0.0] * 10, name='z')
        v = pd.Series([4.0] * 10, name='v')
        wr = WindDoubleRotation(u=zero, v=v, w=zero)
        self.assertAlmostEqual(wr.theta, math.pi / 2, places=12)
        self.assertAlmostEqual(float(wr.u2.mean()), 4.0, places=12)


class TestMaxCovarianceLagDetection(unittest.TestCase):
    """MaxCovariance must recover a lag that was put in deliberately."""

    @staticmethod
    def _shifted_pair(n: int, lag: int):
        import numpy as np
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, n)
        # A positive lag means the scalar arrives *later* than the reference.
        lagged = pd.Series(reference).shift(lag).fillna(0.0).to_numpy()
        return pd.DataFrame({'w': reference, 'c': lagged})

    def _detected_shift(self, df, winsize_from=-50, winsize_to=50):
        from diive.flux.hires.lag import MaxCovariance
        mc = MaxCovariance(df=df, var_reference='w', var_lagged='c',
                           lgs_winsize_from=winsize_from, lgs_winsize_to=winsize_to)
        mc.run()
        cov_df, _ = mc.get()
        peak = cov_df.loc[cov_df['flag_peak_max_cov_abs'], 'shift']
        return int(peak.iloc[0])

    def test_recovers_injected_lag_including_sign(self):
        # The sign convention is documented on MaxCovariance.__init__: a positive
        # lag means var_lagged arrives later. Pin it, so a sign flip is caught.
        for injected in (0, 5, -7, 20):
            with self.subTest(injected=injected):
                df = self._shifted_pair(n=3000, lag=injected)
                self.assertEqual(self._detected_shift(df), injected)

    def test_peak_stays_inside_the_search_window(self):
        # A lag beyond the window cannot be reported; whatever is picked must
        # still lie within the requested bounds rather than run off the end.
        df = self._shifted_pair(n=3000, lag=40)
        found = self._detected_shift(df, winsize_from=-10, winsize_to=10)
        self.assertGreaterEqual(found, -10)
        self.assertLessEqual(found, 10)


class TestApplyTlagHelpers(unittest.TestCase):
    """Filename-key extraction decides which lag is applied to which raw file."""

    def test_extract_key_variants(self):
        from diive.flux.hires.apply_tlag import _extract_key
        name = 'site_20240115_1130_raw.csv'
        # No pattern: the whole filename is the key.
        self.assertEqual(_extract_key(None, name), name)
        # No capture group: the whole match is the key.
        self.assertEqual(_extract_key(r'\d{8}', name), '20240115')
        # Several groups: all non-None groups concatenated, so a date and a time
        # part can be combined into one key.
        self.assertEqual(_extract_key(r'(\d{8})_(\d{4})', name), '202401151130')
        # No match: None, which the caller reads as "no raw file for this period".
        self.assertIsNone(_extract_key(r'ZZZ(\d+)', name))

    def test_build_filename_map_rejects_key_collision(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from diive.flux.hires.apply_tlag import _build_filename_map
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'site_20240115_1130.csv').write_text('x', encoding='utf-8')
            (d / 'site_20240115_1200.csv').write_text('x', encoding='utf-8')

            # Date + time is unique.
            unique = _build_filename_map(d, r'(\d{8})_(\d{4})')
            self.assertEqual(sorted(unique), ['202401151130', '202401151200'])

            # Date alone collides. Silently keeping one would apply the wrong
            # file's lag and drop the other, so it must raise and name both.
            with self.assertRaises(ValueError) as ctx:
                _build_filename_map(d, r'\d{8}')
            msg = str(ctx.exception)
            self.assertIn('site_20240115_1130.csv', msg)
            self.assertIn('site_20240115_1200.csv', msg)

    def test_build_filename_map_skips_non_matching_files(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from diive.flux.hires.apply_tlag import _build_filename_map
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'site_20240115_1130.csv').write_text('x', encoding='utf-8')
            (d / 'notes.txt').write_text('x', encoding='utf-8')
            (d / 'sub').mkdir()  # directories are not files, must be ignored
            result = _build_filename_map(d, r'(\d{8})_(\d{4})')
            self.assertEqual(list(result), ['202401151130'])


class TestPwbopt(unittest.TestCase):
    """PWBOPT S1/S2/S3 selection decides which lag is removed from the data.

    ``apply_pwbopt`` + ``apply_hdi_prefilter`` + ``fill_tlag_gaps`` produce
    ``{prefix}_tlag_final_pf_s``, the column ``TlagApplier`` reads. A wrong
    decision here does not raise -- it shifts real measurements by the wrong
    amount.
    """

    @staticmethod
    def _pwbopt(tlag, hdi, **kwargs):
        from diive.flux.hires.lag_pwb import PwbBatchDetection
        return PwbBatchDetection.apply_pwbopt(tlag, hdi, **kwargs)

    @staticmethod
    def _prefilter(tlag, hdi, threshold):
        from diive.flux.hires.lag_pwb import PwbBatchDetection
        return PwbBatchDetection.apply_hdi_prefilter(tlag, hdi, threshold)

    @staticmethod
    def _fill(pwbopt_s, **kwargs):
        from diive.flux.hires.lag_pwb import PwbBatchDetection
        return PwbBatchDetection.fill_tlag_gaps(pwbopt_s, **kwargs)

    # ---- S1 / S2 / S3 ----
    def test_s1_accepts_every_narrow_hdi_detection(self):
        out = self._pwbopt([1.0, 1.4, 0.9], [0.1, 0.2, 0.3], hdi_thresh=0.5)
        self.assertEqual(list(out['flag']), ['S1_optimal'] * 3)
        self.assertEqual(list(out['pwbopt_s']), [1.0, 1.4, 0.9])

    def test_s1_threshold_is_strict(self):
        # hdi == hdi_thresh is NOT reliable (the check is `<`), so it can only
        # get in through S2. Pin the boundary so it cannot drift.
        out = self._pwbopt([1.0, 1.0], [0.5, 0.4999], hdi_thresh=0.5)
        self.assertEqual(list(out['flag']), ['S3_unreliable', 'S1_optimal'])

    def test_s2_accepts_an_uncertain_lag_close_to_the_last_optimal(self):
        out = self._pwbopt([1.0, 1.3], [0.1, 2.0], hdi_thresh=0.5, dev_thresh=0.5)
        self.assertEqual(list(out['flag']), ['S1_optimal', 'S2_optimal'])
        self.assertEqual(out['pwbopt_s'].iloc[1], 1.3)

    def test_s2_updates_the_reference_so_the_lag_can_drift(self):
        # Every S2 acceptance becomes the new reference, so a run of small steps
        # walks the lag away from the only reliable detection: dev_thresh bounds
        # the step, not the total excursion. Documented here because it looks
        # like a bug from the outside.
        out = self._pwbopt([1.0, 1.4, 1.8, 2.2], [0.1, 2.0, 2.0, 2.0],
                           hdi_thresh=0.5, dev_thresh=0.5)
        self.assertEqual(list(out['flag']),
                         ['S1_optimal', 'S2_optimal', 'S2_optimal', 'S2_optimal'])
        self.assertAlmostEqual(out['pwbopt_s'].iloc[-1], 2.2)

    def test_s3_repeats_the_last_optimal_instead_of_the_detection(self):
        # Wide HDI and far from the last optimal: the detection is discarded and
        # never becomes the new reference, so the third period is judged against
        # 1.0 as well.
        out = self._pwbopt([1.0, 7.0, 7.1], [0.1, 3.0, 3.0],
                           hdi_thresh=0.5, dev_thresh=0.5)
        self.assertEqual(list(out['flag']),
                         ['S1_optimal', 'S3_unreliable', 'S3_unreliable'])
        self.assertEqual(list(out['pwbopt_s']), [1.0, 1.0, 1.0])

    def test_nothing_is_carried_forward_before_the_first_reliable_lag(self):
        # Leading unreliable periods stay NaN; filling them is fill_tlag_gaps'
        # job. Emitting 0.0 here would apply a zero lag to real data.
        out = self._pwbopt([5.0, 5.2, 1.0], [3.0, 3.0, 0.1],
                           hdi_thresh=0.5, dev_thresh=0.5)
        self.assertTrue(np.isnan(out['pwbopt_s'].iloc[0]))
        self.assertTrue(np.isnan(out['pwbopt_s'].iloc[1]))
        self.assertEqual(out['pwbopt_s'].iloc[2], 1.0)

    def test_a_failed_detection_is_a_gap_not_a_value(self):
        # NaN lag (detection failed, chunk errored) and NaN HDI (unknown
        # reliability) both carry the previous optimal forward.
        out = self._pwbopt([1.0, np.nan, 2.0], [0.1, np.nan, np.nan])
        self.assertEqual(list(out['pwbopt_s']), [1.0, 1.0, 1.0])

    # ---- HDI pre-filter ----
    def test_prefilter_drops_only_hdi_above_the_threshold(self):
        pf = self._prefilter([1.0, 1.3, 1.5], [0.1, 2.0, 1.0], threshold=1.0)
        self.assertEqual(pf[0], 1.0)
        self.assertTrue(np.isnan(pf[1]))
        self.assertEqual(pf[2], 1.5)  # hdi == threshold is kept

    def test_prefilter_keeps_a_lag_whose_hdi_is_unknown(self):
        # NaN HDI is not "wide", it is unknown, so the prefilter leaves it
        # alone; apply_pwbopt then treats it as a gap anyway.
        pf = self._prefilter([1.0, 2.0], [np.nan, 5.0], threshold=1.0)
        self.assertEqual(pf[0], 1.0)
        self.assertTrue(np.isnan(pf[1]))

    def test_prefilter_is_stricter_than_plain_pwbopt(self):
        # The reason the _pf_ column exists: a wide-HDI lag that S2 would accept
        # (it happens to sit close to the last optimal) is removed *before*
        # selection, so it can never be applied.
        tlag, hdi = [1.0, 1.3], [0.1, 2.0]
        std = self._pwbopt(tlag, hdi, hdi_thresh=0.5, dev_thresh=0.5)
        pf = self._pwbopt(self._prefilter(tlag, hdi, threshold=1.0), hdi,
                          hdi_thresh=0.5, dev_thresh=0.5)
        self.assertEqual(std['pwbopt_s'].iloc[1], 1.3)
        self.assertEqual(pf['pwbopt_s'].iloc[1], 1.0)

    # ---- gap filling ----
    def test_fill_backfills_the_leading_gap(self):
        filled = self._fill([np.nan, np.nan, 2.0, 2.0])
        self.assertEqual(list(filled), [2.0] * 4)

    def test_fill_falls_back_to_the_median_of_the_raw_lags(self):
        # No S1/S2 anywhere: rather than leave the period unusable, the median
        # of the raw detections is used.
        filled = self._fill([np.nan] * 3, tlag_s_raw=[1.0, 5.0, 3.0])
        self.assertEqual(list(filled), [3.0] * 3)

    def test_fill_uses_the_explicit_fallback_as_last_resort(self):
        filled = self._fill([np.nan, np.nan], tlag_s_raw=[np.nan, np.nan],
                            fallback=0.3)
        self.assertEqual(list(filled), [0.3, 0.3])

    def test_fill_leaves_nan_when_nothing_can_be_inferred(self):
        # Better an explicit NaN (skipped: lag_nan) than a fabricated lag.
        filled = self._fill([np.nan, np.nan], tlag_s_raw=[np.nan, np.nan])
        self.assertTrue(np.isnan(filled).all())


class TestPwboptFinalLags(unittest.TestCase):
    """_pwbopt_final_lags turns per-chunk detections into the applied lag."""

    SCALARS = {'CH4': 'ch4'}

    @staticmethod
    def _rows(tlags, hdis):
        return [{'chunk_index': i, 'ch4_tlag_s': t, 'ch4_hdi_range_s': h}
                for i, (t, h) in enumerate(zip(tlags, hdis, strict=True))]

    def _final(self, rows, template='{prefix}_tlag_final_pf_s'):
        from diive.flux.hires.detect_and_remove_tlag import _pwbopt_final_lags
        return _pwbopt_final_lags(rows, self.SCALARS, hdi_thresh=0.5,
                                  dev_thresh=0.5, hdi_prefilter=1.0,
                                  lag_column_template=template)

    def test_chunks_are_ordered_before_selection(self):
        # Workers finish out of order. PWBOPT is a carry-forward in time, so an
        # unsorted sequence would propagate lags backwards.
        rows = self._rows([1.0, 9.0], [0.1, 3.0])
        out = self._final([rows[1], rows[0]])
        self.assertEqual(out[(0, 'CH4')], 1.0)
        self.assertEqual(out[(1, 'CH4')], 1.0)  # S3 -> carried forward

    def test_default_template_selects_the_prefiltered_series(self):
        # The two columns differ exactly when S2 accepted a wide-HDI lag; the
        # default must be the conservative one.
        rows = self._rows([1.0, 1.3], [0.1, 2.0])
        self.assertEqual(self._final(rows, '{prefix}_tlag_final_s')[(1, 'CH4')], 1.3)
        self.assertEqual(self._final(rows, '{prefix}_tlag_final_pf_s')[(1, 'CH4')], 1.0)

    def test_an_errored_chunk_still_gets_a_usable_lag(self):
        # A chunk that failed detection carries no tlag keys at all. It must
        # still receive a lag, otherwise its data would go out unaligned.
        rows = self._rows([1.0, np.nan, 1.1], [0.1, np.nan, 0.1])
        rows[1] = {'chunk_index': 1}  # what _empty_detect_row leaves behind
        out = self._final(rows)
        self.assertEqual(out[(1, 'CH4')], 1.0)
        self.assertFalse(np.isnan(list(out.values())).any())

    def test_a_leading_errored_chunk_is_backfilled(self):
        rows = self._rows([np.nan, 1.2], [np.nan, 0.1])
        self.assertEqual(self._final(rows)[(0, 'CH4')], 1.2)

    def test_no_rows_is_not_an_error(self):
        self.assertEqual(self._final([]), {})


class TestChunkGridAlignment(unittest.TestCase):
    """Off-grid raw files must still yield chunks on the wall-clock grid.

    Downstream software bins 30-min files by clock time, so a chunk starting at
    10:10 instead of 10:30 lands in the wrong averaging interval. The
    end-to-end pipeline test only ever uses an on-grid start time, so the
    leading-partial path is otherwise never executed.
    """

    HZ = 20
    CHUNK = 1800

    @staticmethod
    def _lead(*args):
        from diive.flux.hires.detect_and_remove_tlag import _grid_lead_seconds
        return _grid_lead_seconds(*args)

    @staticmethod
    def _start(*args):
        from diive.flux.hires.detect_and_remove_tlag import _chunk_start_time
        return _chunk_start_time(*args)

    @staticmethod
    def _slice(*args):
        from diive.flux.hires.detect_and_remove_tlag import _chunk_row_slice
        return _chunk_row_slice(*args)

    def test_a_file_on_the_grid_has_no_lead(self):
        for t in (datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 10, 30),
                  datetime(2024, 1, 15, 0, 0)):
            with self.subTest(t=t):
                self.assertEqual(self._lead(t, self.CHUNK), 0.0)

    def test_the_lead_reaches_the_next_boundary(self):
        self.assertEqual(self._lead(datetime(2024, 1, 15, 10, 10), 1800), 1200.0)
        self.assertEqual(self._lead(datetime(2024, 1, 15, 10, 50), 1800), 600.0)
        # Sub-minute offsets count too, otherwise a chunk drifts off the grid.
        self.assertEqual(self._lead(datetime(2024, 1, 15, 10, 29, 59), 1800), 1.0)

    def test_without_a_start_time_the_grid_is_unknown(self):
        self.assertEqual(self._lead(None, 1800), 0.0)
        self.assertIsNone(self._start(None, 3, 1800))
        # Legacy fixed-offset chunking.
        self.assertEqual(self._slice(None, 3, 1800, self.HZ), (3 * 36000, 36000))

    def test_an_offgrid_file_snaps_every_later_chunk_to_the_grid(self):
        start = datetime(2024, 1, 15, 10, 10)
        starts = [self._start(start, i, self.CHUNK) for i in range(4)]
        # Chunk 0 is the short partial and keeps the file's real start time.
        self.assertEqual(starts[0], start)
        self.assertEqual(starts[1], datetime(2024, 1, 15, 10, 30))
        self.assertEqual(starts[2], datetime(2024, 1, 15, 11, 0))
        self.assertEqual(starts[3], datetime(2024, 1, 15, 11, 30))
        for s in starts[1:]:
            self.assertIn(s.minute, (0, 30))
            self.assertEqual(s.second, 0)

    def test_an_ongrid_file_uses_fixed_steps(self):
        start = datetime(2024, 1, 15, 10, 30)
        self.assertEqual(self._start(start, 0, self.CHUNK), start)
        self.assertEqual(self._start(start, 2, self.CHUNK),
                         datetime(2024, 1, 15, 11, 30))

    def test_the_leading_partial_is_shorter_than_a_full_chunk(self):
        start = datetime(2024, 1, 15, 10, 10)
        self.assertEqual(self._slice(start, 0, self.CHUNK, self.HZ),
                         (0, 1200 * self.HZ))
        self.assertEqual(self._slice(start, 1, self.CHUNK, self.HZ),
                         (1200 * self.HZ, self.CHUNK * self.HZ))

    def test_row_slices_tile_the_file_without_gap_or_overlap(self):
        # Every raw record must land in exactly one chunk.
        for start in (datetime(2024, 1, 15, 10, 10),
                      datetime(2024, 1, 15, 10, 30), None):
            with self.subTest(start=start):
                slices = [self._slice(start, i, self.CHUNK, self.HZ)
                          for i in range(4)]
                self.assertEqual(slices[0][0], 0)
                for (off, n), (nxt, _) in zip(slices, slices[1:], strict=False):
                    self.assertEqual(off + n, nxt)

    def test_the_row_offset_agrees_with_the_chunk_start_time(self):
        # One helper names the output file, the other reads its rows. If they
        # disagree, a chunk is labelled with a time it does not contain.
        start = datetime(2024, 1, 15, 10, 10)
        for i in range(4):
            with self.subTest(chunk=i):
                offset, _ = self._slice(start, i, self.CHUNK, self.HZ)
                self.assertEqual(self._start(start, i, self.CHUNK),
                                 start + timedelta(seconds=offset / self.HZ))


class TestChunkFilename(unittest.TestCase):
    """Each output chunk is named after its own wall-clock start time."""

    FMT = '%Y%m%d%H%M'

    @staticmethod
    def _name(*args):
        from diive.flux.hires.detect_and_remove_tlag import _chunk_filename
        return _chunk_filename(*args)

    def test_starttime_advances_with_the_chunk(self):
        # The documented CH-CHA case: one long file -> per-chunk files named by
        # each chunk's own start.
        p = Path('CH-CHA_202107271300.csv')
        out = [self._name(p, i, 1800, 'CH-CHA_{starttime}{suffix}',
                          r'(\d{12})', self.FMT) for i in range(3)]
        self.assertEqual([n for n, _ in out],
                         ['CH-CHA_202107271300.csv', 'CH-CHA_202107271330.csv',
                          'CH-CHA_202107271400.csv'])
        self.assertEqual(out[1][1], datetime(2021, 7, 27, 13, 30))

    def test_index_template_needs_no_regex(self):
        name, t = self._name(Path('raw_001.txt'), 2, 1800,
                             '{stem}_c{index}{suffix}', None, self.FMT)
        self.assertEqual(name, 'raw_001_c2.txt')
        self.assertIsNone(t)

    def test_starttime_without_a_regex_is_a_config_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._name(Path('raw.csv'), 0, 1800, '{starttime}{suffix}',
                       None, self.FMT)
        self.assertIn('start-time-regex', str(ctx.exception))

    def test_an_unmatched_regex_only_matters_when_starttime_is_used(self):
        p = Path('raw_without_a_timestamp.csv')
        # Needed but unobtainable -> fail loudly, naming the offending file.
        with self.assertRaises(ValueError) as ctx:
            self._name(p, 0, 1800, '{starttime}{suffix}', r'(\d{12})', self.FMT)
        self.assertIn(p.name, str(ctx.exception))
        # Not needed -> the unmatched regex is harmless.
        name, t = self._name(p, 0, 1800, '{stem}_{index}{suffix}',
                             r'(\d{12})', self.FMT)
        self.assertEqual(name, 'raw_without_a_timestamp_0.csv')
        self.assertIsNone(t)

    def test_a_format_mismatch_reports_the_format(self):
        with self.assertRaises(ValueError) as ctx:
            self._name(Path('site_202107271300.csv'), 0, 1800,
                       '{starttime}{suffix}', r'(\d{12})', '%Y-%m-%d')
        self.assertIn('%Y-%m-%d', str(ctx.exception))

    def test_an_unknown_placeholder_lists_the_available_ones(self):
        with self.assertRaises(ValueError) as ctx:
            self._name(Path('a.csv'), 0, 1800, '{nope}', None, self.FMT)
        msg = str(ctx.exception)
        self.assertIn('nope', msg)
        self.assertIn('stem', msg)

    def test_parse_file_start_time(self):
        from diive.flux.hires.detect_and_remove_tlag import _parse_file_start_time
        expected = datetime(2021, 7, 27, 13, 0)
        self.assertEqual(
            _parse_file_start_time('CH-CHA_202107271300.csv', r'(\d{12})', self.FMT),
            expected)
        # Split groups are concatenated, so a hyphenated EddyPro name works.
        self.assertEqual(
            _parse_file_start_time('20210727-1300_adv.txt', r'(\d{8})-(\d{4})', self.FMT),
            expected)
        # No regex / no match / unparseable -> None, caller falls back to
        # filename order rather than crashing the run.
        self.assertIsNone(_parse_file_start_time('CH-CHA_202107271300.csv', None, self.FMT))
        self.assertIsNone(_parse_file_start_time('no_timestamp.csv', r'(\d{12})', self.FMT))
        self.assertIsNone(_parse_file_start_time('CH-CHA_209902991300.csv', r'(\d{12})', self.FMT))


class TestRawFileRoundTrip(unittest.TestCase):
    """A written chunk must be a drop-in replacement for the input file.

    Everything except the shifted scalar has to survive byte-for-byte in
    meaning: metadata block, header, units rows, missing-value sentinel and the
    line terminator the logger used.
    """

    HEADER = ['# raw data CH-CHA', 'u,v,w,ch4', 'm/s,m/s,m/s,ppb',
              'sonic,sonic,sonic,lgr']

    def _make_file(self, directory, name='raw.csv', lineterm='\n',
                   data=None):
        lines = list(self.HEADER)
        lines += data if data is not None else [
            f'{i}.0,1.0,2.0,{i * 10}.5' for i in range(5)]
        p = Path(directory) / name
        p.write_bytes((lineterm.join(lines) + lineterm).encode('utf-8'))
        return p

    @staticmethod
    def _read(*args, **kwargs):
        from diive.flux.hires.detect_and_remove_tlag import _read_raw_file
        return _read_raw_file(*args, **kwargs)

    @staticmethod
    def _write(*args, **kwargs):
        from diive.flux.hires.detect_and_remove_tlag import _write_raw_file
        return _write_raw_file(*args, **kwargs)

    def test_detects_the_line_terminator_the_input_used(self):
        from diive.flux.hires.detect_and_remove_tlag import (
            _detect_lineterm, _resolve_lineterm)
        with TemporaryDirectory() as tmp:
            crlf = self._make_file(tmp, 'win.csv', '\r\n')
            lf = self._make_file(tmp, 'unix.csv', '\n')
            self.assertEqual(_detect_lineterm(crlf), '\r\n')
            self.assertEqual(_detect_lineterm(lf), '\n')
            # No newline at all -> default.
            nolines = Path(tmp) / 'oneline.csv'
            nolines.write_bytes(b'no newline here')
            self.assertEqual(_detect_lineterm(nolines), '\n')
            # 'auto' resolves against the input; an explicit choice wins.
            self.assertEqual(_resolve_lineterm('auto', crlf), '\r\n')
            self.assertEqual(_resolve_lineterm('\n', crlf), '\n')

    def test_header_block_and_data_survive_the_round_trip(self):
        with TemporaryDirectory() as tmp:
            src = self._make_file(tmp)
            preserved, df = self._read(src, skiprows=1, extra_rows=2, sep=',',
                                       na_values=['-9999'])
            self.assertEqual(list(df.columns), ['u', 'v', 'w', 'ch4'])
            self.assertEqual(len(df), 5)  # the two units rows are not data

            out = Path(tmp) / 'out.csv'
            self._write(out, preserved, df, sep=',', lineterm='\n', na_rep='-9999')
            preserved2, df2 = self._read(out, skiprows=1, extra_rows=2, sep=',',
                                         na_values=['-9999'])
            self.assertEqual([ln.rstrip('\r\n') for ln in preserved],
                             [ln.rstrip('\r\n') for ln in preserved2])
            pd.testing.assert_frame_equal(df, df2)

    def test_the_output_never_mixes_line_terminators(self):
        # Header lines come back from a text-mode read LF-terminated; writing
        # them verbatim next to CRLF data rows would produce a half-CRLF file
        # that some readers choke on.
        with TemporaryDirectory() as tmp:
            src = self._make_file(tmp, 'win.csv', '\r\n')
            preserved, df = self._read(src, skiprows=1, extra_rows=2, sep=',',
                                       na_values=['-9999'])
            out = Path(tmp) / 'out.csv'
            self._write(out, preserved, df, sep=',', lineterm='\r\n',
                        na_rep='-9999')
            raw = out.read_bytes()
            self.assertGreater(raw.count(b'\n'), 0)
            self.assertEqual(raw.count(b'\n'), raw.count(b'\r\n'))

    def test_the_missing_value_sentinel_round_trips(self):
        with TemporaryDirectory() as tmp:
            src = self._make_file(tmp, data=['0.0,1.0,2.0,-9999',
                                             '1.0,1.0,2.0,5.5'])
            preserved, df = self._read(src, skiprows=1, extra_rows=2, sep=',',
                                       na_values=['-9999'])
            self.assertTrue(np.isnan(df['ch4'].iloc[0]))
            out = Path(tmp) / 'out.csv'
            self._write(out, preserved, df, sep=',', lineterm='\n',
                        na_rep='-9999')
            self.assertIn('-9999', out.read_text(encoding='utf-8').splitlines()[4])

    def test_whitespace_separated_files_round_trip(self):
        from diive.flux.hires.detect_and_remove_tlag import _WHITESPACE_SEP
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / 'ep.txt'
            src.write_text('meta\nu v w ch4\n0.0  1.0 2.0 3.0\n1.0 1.0 2.0 4.0\n',
                           encoding='utf-8')
            preserved, df = self._read(src, skiprows=1, extra_rows=0,
                                       sep=_WHITESPACE_SEP, na_values=[])
            self.assertEqual(list(df.columns), ['u', 'v', 'w', 'ch4'])
            out = Path(tmp) / 'out.txt'
            self._write(out, preserved, df, sep=_WHITESPACE_SEP, lineterm='\n',
                        na_rep='-9999')
            _, df2 = self._read(out, skiprows=1, extra_rows=0,
                                sep=_WHITESPACE_SEP, na_values=[])
            pd.testing.assert_frame_equal(df, df2)

    def test_a_header_data_mismatch_names_the_flags_to_check(self):
        # The most common misconfiguration; the message has to point at the
        # flag that is wrong, not at a pandas traceback.
        with TemporaryDirectory() as tmp:
            src = self._make_file(tmp)
            with self.assertRaises(ValueError) as ctx:
                self._read(src, skiprows=0, extra_rows=2, sep=',',
                           na_values=['-9999'])
            msg = str(ctx.exception)
            self.assertIn('skiprows', msg)
            self.assertIn(src.name, msg)


class TestWhitespaceSeparatedPipeline(unittest.TestCase):
    """EddyPro rotated files are whitespace-separated, which needs pandas'
    python parser -- and that parser rejects some C-parser options.

    ``detect_one_chunk`` and ``remove_one_chunk`` each open the file with their
    own ``read_csv`` call, so one end-to-end run is the only thing that covers
    all of them. Every other pipeline test uses comma-separated input.
    """

    def test_a_whitespace_file_runs_through_detect_and_remove(self):
        from diive.flux.hires.detect_and_remove_tlag import (
            _WHITESPACE_SEP, PerFilePipeline)
        rng = np.random.default_rng(3)
        n = 2400
        w = rng.standard_normal(n)
        df = pd.DataFrame({
            'u': rng.standard_normal(n), 'v': rng.standard_normal(n), 'w': w,
            'ts': 0.8 * w + 0.2 * rng.standard_normal(n),
            'ch4': np.r_[np.zeros(20), w[:-20]] + 0.2 * rng.standard_normal(n),
        })
        with TemporaryDirectory() as ind, TemporaryDirectory() as out:
            src = Path(ind) / 'site_202401010000.txt'
            with open(src, 'w', encoding='utf-8', newline='') as fh:
                fh.write(' '.join(df.columns) + '\n')
                df.to_csv(fh, sep=' ', index=False, header=False,
                          lineterminator='\n')
            summary = PerFilePipeline(
                Path(ind), Path(out), 'u', 'v', 'w', 'ts', {'CH4': 'ch4'},
                hz=20, n_bootstrap=19, chunk_seconds=60, min_chunk_seconds=30,
                sep=_WHITESPACE_SEP, extra_rows=0, n_workers=1,
                file_pattern='*.txt', random_state=42).run()
            written = sorted((Path(out) / '2_lag_removed').glob('*.txt'))
            first = written[0].read_text(encoding='utf-8').splitlines()

        errors = [e for e in summary['error'].fillna('') if e]
        self.assertEqual(errors, [])
        self.assertEqual(len(written), 2)  # 2400 rows at 20 Hz = two 60 s chunks
        self.assertEqual(first[0].split(), list(df.columns))
        self.assertAlmostEqual(float(summary['ch4_tlag_s'].iloc[0]), 1.0, delta=0.3)


class TestTlagApplierShift(unittest.TestCase):
    """The shift is the only step that rewrites measurement data.

    A sign or rounding error does not raise: it silently mis-aligns every
    scalar and corrupts every flux computed downstream. Only the filename
    helpers were covered before.
    """

    HZ = 20

    def _write_input(self, directory, df):
        src = Path(directory) / 'in.csv'
        with open(src, 'w', encoding='utf-8', newline='') as fh:
            fh.write(','.join(df.columns) + '\n')
            fh.write(','.join(['-'] * len(df.columns)) + '\n')  # units row
            df.to_csv(fh, index=False, header=False, na_rep='-9999',
                      lineterminator='\n')
        return src

    def _apply(self, directory, df, lags, scalars=None, strict=False):
        """Run the worker directly: same code path as run(), no process pool."""
        from diive.flux.hires.apply_tlag import _apply_tlag_file_worker
        scalars = scalars if scalars is not None else {'CH4': 'ch4'}
        src = self._write_input(directory, df)
        out = Path(directory) / 'out.csv'
        row = _apply_tlag_file_worker((
            str(src), str(out), scalars, lags, self.HZ,
            0, 1, ',', '\n', ['-9999'], '-9999', strict))
        return row, out

    @staticmethod
    def _read_output(path, columns):
        got = pd.read_csv(path, skiprows=2, header=None, na_values=['-9999'])
        got.columns = columns
        return got

    @staticmethod
    def _delayed_pair(n=200, records=30, seed=0):
        """A scalar that is the wind delayed by `records` rows (tube delay)."""
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n)
        return pd.DataFrame({'w': w, 'ch4': np.r_[np.zeros(records), w[:-records]]})

    def test_removing_the_lag_realigns_the_scalar_with_the_wind(self):
        # The sign convention in one assertion: a scalar delayed by 1.5 s must
        # come back into phase with the wind. A forward shift would double the
        # offset instead, and nothing else in the pipeline would notice.
        records = 30
        df = self._delayed_pair(records=records)
        with TemporaryDirectory() as tmp:
            row, out = self._apply(tmp, df, {'CH4': records / self.HZ})
            got = self._read_output(out, ['w', 'ch4'])
        valid = got['ch4'].notna()
        self.assertEqual(int(valid.sum()), len(df) - records)
        self.assertTrue(np.allclose(got.loc[valid, 'ch4'], got.loc[valid, 'w']))
        self.assertEqual(row['ch4_applied_records'], records)
        self.assertEqual(row['ch4_status'], 'ok')

    def test_the_rows_shifted_out_become_the_missing_value_sentinel(self):
        records = 30
        df = self._delayed_pair(records=records)
        with TemporaryDirectory() as tmp:
            _, out = self._apply(tmp, df, {'CH4': records / self.HZ})
            text = out.read_text(encoding='utf-8')
            got = self._read_output(out, ['w', 'ch4'])
        # Exactly one sentinel per row that lost its value, and no more.
        self.assertEqual(text.count('-9999'), records)
        self.assertTrue(got['ch4'].tail(records).isna().all())

    def test_only_the_requested_column_moves(self):
        df = self._delayed_pair(records=30)
        with TemporaryDirectory() as tmp:
            _, out = self._apply(tmp, df, {'CH4': 1.5})
            got = self._read_output(out, ['w', 'ch4'])
        self.assertTrue(np.allclose(got['w'], df['w']))

    def test_a_nan_lag_is_skipped_instead_of_applied_as_zero(self):
        df = self._delayed_pair(records=30)
        with TemporaryDirectory() as tmp:
            row, out = self._apply(tmp, df, {'CH4': np.nan})
            got = self._read_output(out, ['w', 'ch4'])
        self.assertEqual(row['ch4_status'], 'skipped:lag_nan')
        self.assertEqual(row['status'], 'ok')  # the file is still written
        self.assertTrue(np.allclose(got['ch4'], df['ch4']))

    def test_a_missing_column_is_reported_not_fatal(self):
        # One gas absent from a file must not lose the other gases' correction.
        df = self._delayed_pair(records=30)
        with TemporaryDirectory() as tmp:
            row, out = self._apply(tmp, df, {'CH4': 1.5, 'N2O': 2.0},
                                   scalars={'CH4': 'ch4', 'N2O': 'n2o'})
            self.assertTrue(out.exists())
        self.assertEqual(row['n2o_status'], 'skipped:column_missing')
        self.assertEqual(row['ch4_status'], 'ok')
        self.assertEqual(row['status'], 'ok')

    def test_whitespace_separated_files_are_supported(self):
        # The default separator of diive-tlag-apply-batch, and the format of the
        # EddyPro rotated files the module is written around. It needs pandas'
        # python engine, which rejects some C-parser options.
        from diive.flux.hires.apply_tlag import (
            _WHITESPACE_SEP, _apply_tlag_file_worker)
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / 'ep.txt'
            src.write_text('meta line\nu v w ch4\n0.0 1.0 2.0 3.0\n'
                           '1.0 1.0 2.0 4.0\n', encoding='utf-8')
            out = Path(tmp) / 'out.txt'
            row = _apply_tlag_file_worker((
                str(src), str(out), {'CH4': 'ch4'}, {'CH4': 0.05}, self.HZ,
                1, 0, _WHITESPACE_SEP, '\n', ['-9999'], '-9999', False))
            self.assertEqual(row['status'], 'ok', msg=row['error'])
            self.assertEqual(row['ch4_applied_records'], 1)
            lines = out.read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines[:2], ['meta line', 'u v w ch4'])
        self.assertEqual(lines[2].split()[-1], '4.0')   # shifted up by one row
        self.assertEqual(lines[3].split()[-1], '-9999')  # shifted out

    def test_the_lag_is_rounded_to_whole_records(self):
        # Sub-record lags cannot be applied; nearest-record rounding is the
        # documented behaviour (0.07 s * 20 Hz = 1.4 -> 1 record).
        df = self._delayed_pair(records=10)
        for lag_s, expected in ((0.07, 1), (0.08, 2), (0.02, 0), (-0.1, -2)):
            with self.subTest(lag_s=lag_s), TemporaryDirectory() as tmp:
                row, _ = self._apply(tmp, df, {'CH4': lag_s})
                self.assertEqual(row['ch4_applied_records'], expected)

    def test_a_broken_file_is_captured_unless_strict(self):
        # One malformed file must not abort a batch of thousands.
        with TemporaryDirectory() as tmp:
            # Four column names but three data columns: the mismatch the
            # skiprows/extra-rows/sep flags are usually to blame for.
            src = Path(tmp) / 'in.csv'
            src.write_text('u,v,w,ch4\n-,-,-,-\n0.0,1.0,2.0\n1.0,1.0,2.0\n',
                           encoding='utf-8')

            from diive.flux.hires.apply_tlag import _apply_tlag_file_worker
            args = (str(src), str(Path(tmp) / 'out.csv'), {'CH4': 'ch4'},
                    {'CH4': 1.5}, self.HZ, 0, 1, ',', '\n', ['-9999'], '-9999',
                    False)
            row = _apply_tlag_file_worker(args)
            self.assertEqual(row['status'], 'error')
            self.assertTrue(row['error'])
            self.assertEqual(row['period'], src.name)

            with self.assertRaises(Exception):
                _apply_tlag_file_worker(args[:-1] + (True,))


class TestTlagApplierRun(unittest.TestCase):
    """Guards that must fire before any file is written."""

    @staticmethod
    def _results_csv(directory, rows, columns):
        path = Path(directory) / 'tlag_results.csv'
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path

    @staticmethod
    def _applier(tmp, results, **kwargs):
        from diive.flux.hires.apply_tlag import TlagApplier
        return TlagApplier(input_dir=Path(tmp), output_dir=Path(tmp) / 'out',
                           results_csv=results, scalars={'CH4': 'ch4'},
                           sep=',', skiprows=0, extra_rows=1, hz=20,
                           n_workers=1, **kwargs)

    def test_a_missing_lag_column_names_the_template(self):
        # Wrong --lag-column-template is a silent no-op risk; it must be caught
        # before the pool starts, with the available columns listed.
        with TemporaryDirectory() as tmp:
            results = self._results_csv(tmp, [['f1.csv', 1.5]],
                                        ['period', 'ch4_tlag_s'])
            with self.assertRaises(ValueError) as ctx:
                self._applier(tmp, results).run()
            msg = str(ctx.exception)
            self.assertIn('ch4_tlag_final_pf_s', msg)
            self.assertIn('ch4_tlag_s', msg)  # what the CSV actually offers

    def test_two_periods_resolving_to_one_output_are_rejected(self):
        # An over-broad --period-key-regex would make one period silently
        # overwrite the other with a different lag.
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'CH-CHA_202401151130.csv').write_text('x', encoding='utf-8')
            results = self._results_csv(
                tmp, [['20240115-1130', 1.5], ['20240115-1200', 2.5]],
                ['period', 'ch4_tlag_final_pf_s'])
            applier = self._applier(tmp, results,
                                    period_key_regex=r'(\d{8})',
                                    file_key_regex=r'(\d{8})')
            with self.assertRaises(ValueError) as ctx:
                applier.run()
            self.assertIn('20240115-1130', str(ctx.exception))

    def test_a_period_without_a_raw_file_is_reported_not_dropped(self):
        # No worker runs, but the summary must still carry the period with an
        # explicit failure -- a vanished row would read as success. The
        # detection and application dirs routinely hold different files.
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'CH-CHA_202401151130.csv').write_text('x', encoding='utf-8')
            results = self._results_csv(
                tmp, [['20240115-2359_adv.txt', 1.5]],
                ['period', 'ch4_tlag_final_pf_s'])
            summary = self._applier(tmp, results,
                                    period_key_regex=r'(\d{8})-(\d{4})',
                                    file_key_regex=r'(\d{12})').run()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary['status'].iloc[0], 'error')
        self.assertEqual(summary['ch4_status'].iloc[0], 'skipped:no_raw_file')
        self.assertIn('202401152359', summary['error'].iloc[0])

    def test_run_applies_the_lag_and_keeps_a_numeric_period_a_string(self):
        # The one test that goes through the process pool. Read as int,
        # '202401151130' would become a float and never match a filename again.
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / '202401151130'  # no suffix: numeric-looking name
            src.write_text('w,ch4\n-,-\n1.0,10.0\n2.0,20.0\n3.0,30.0\n',
                           encoding='utf-8')
            results = self._results_csv(
                tmp, [['202401151130', 0.05]], ['period', 'ch4_tlag_final_pf_s'])
            summary = self._applier(tmp, results).run()
            written = (Path(tmp) / 'out' / '202401151130').read_text(
                encoding='utf-8').splitlines()
        self.assertEqual(summary['period'].iloc[0], '202401151130')
        self.assertEqual(summary['status'].iloc[0], 'ok',
                         msg=summary['error'].iloc[0])
        self.assertEqual(summary['ch4_applied_records'].iloc[0], 1)
        # One record of lag removed: ch4 moves up a row, the last row empties.
        self.assertEqual([ln.split(',')[-1] for ln in written[2:]],
                         ['20.0', '30.0', '-9999'])


class TestPwbReproducibility(unittest.TestCase):
    """A detected lag must be reproducible, and reliability must mean something."""

    @staticmethod
    def _fixture(n=3000, records=30, noise=0.5, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal(n)
        s = np.r_[np.zeros(records), w[:-records]] + noise * rng.standard_normal(n)
        return pd.DataFrame({'w': w, 's': s, 't': 0.7 * w + 0.3 * rng.standard_normal(n)})

    @staticmethod
    def _run(df, **kwargs):
        from diive.flux.hires.lag_pwb import PreWhiteningBootstrap
        pwb = PreWhiteningBootstrap(df, 'w', 's', 't', hz=20, lag_max_s=5,
                                    n_bootstrap=19, **kwargs)
        pwb.run()
        return pwb.results

    def test_the_same_seed_reproduces_the_lag_and_its_uncertainty(self):
        # The bootstrap is random; without a pinned seed a re-run of the same
        # raw file would write different lags, and PWBOPT's S1/S2 decisions
        # would flip with it.
        df = self._fixture()
        first = self._run(df, random_state=7)
        second = self._run(df, random_state=7)
        for key in ('tlag_s', 'hdi_lo_s', 'hdi_hi_s', 'best_combination'):
            with self.subTest(key=key):
                self.assertEqual(first[key], second[key])

    def test_the_selected_combination_is_one_of_the_four(self):
        res = self._run(self._fixture(), random_state=7)
        # The four RFlux v3.2.0 combinations; anything else means the selection
        # step returned a key the rest of the class cannot interpret.
        self.assertIn(res['best_combination'], ('cw', 'wc', 'ct', 'tc'))
        # Whichever combination wins must still point at the injected 1.5 s lag.
        # A tolerance, not an equality: 19 bootstrap resamples put the mode
        # within a few records, and exact recovery is pinned elsewhere
        # (TestPwbPerGasWindow.test_window_full_equals_default).
        self.assertAlmostEqual(res['tlag_s'], 1.5, delta=0.25)

    def test_a_noisier_signal_is_reported_as_less_certain(self):
        # HDI width is the S1 criterion, so it has to track detection quality
        # rather than just being a number.
        clean = self._run(self._fixture(noise=0.05), random_state=7)
        noisy = self._run(self._fixture(noise=8.0), random_state=7)
        self.assertLess(clean['hdi_range_s'], noisy['hdi_range_s'])
        self.assertTrue(clean['is_reliable'])


if __name__ == '__main__':
    unittest.main()
