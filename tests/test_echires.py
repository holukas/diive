import unittest

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


if __name__ == '__main__':
    unittest.main()
