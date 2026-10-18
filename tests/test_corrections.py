"""
TEST_CORRECTIONS: the dv.corrections namespace
==============================================

Covers all ten public symbols of ``dv.corrections``, the ``apply_corrections``
dispatch table, and the ``dv.corrections`` namespace module itself -- which no
test imported at all, leaving its ``__all__`` unverified (a symbol could have
been dropped from the re-export list with the whole suite still green).

Run: pytest tests/test_corrections.py -v
"""
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

# Site coordinates used for every day/night-dependent correction.
LAT, LON, UTC_OFFSET = 46.6, 9.8, 1


def _radiation_series(offset: float = 3.0, days: int = 10, name: str = "SW_IN"):
    """Synthetic radiation carrying a known constant offset everywhere.

    Values follow potential radiation, so the day/night structure is real, plus
    ``offset``. The daily nighttime mean is therefore exactly ``offset``, which
    makes the detected offset an exact expected value rather than an approximation.
    """
    import diive as dv
    idx = pd.date_range("2021-06-01", periods=48 * days, freq="30min",
                        name="TIMESTAMP_MIDDLE")
    swinpot = dv.variables.potrad(timestamp_index=idx, lat=LAT, lon=LON,
                                  utc_offset=UTC_OFFSET)
    values = np.where(swinpot > 0.001, swinpot * 0.7, 0.0) + offset
    return pd.Series(values, index=idx, name=name)


class TestCorrectionsNamespace(unittest.TestCase):
    """`__all__` resolution for every namespace lives in tests/test_imports.py;
    this pins the re-exports to their implementation objects, which the generic
    test cannot do (it does not know each namespace's backing module)."""

    def test_exports_are_the_implementation_objects(self):
        # The namespace must re-export, not shadow with a look-alike.
        import diive as dv
        from diive.preprocessing.corrections import (
            apply_corrections, remove_nighttime_zero_offset, setto_threshold)
        self.assertIs(dv.corrections.apply_corrections, apply_corrections)
        self.assertIs(dv.corrections.remove_nighttime_zero_offset,
                      remove_nighttime_zero_offset)
        self.assertIs(dv.corrections.setto_threshold, setto_threshold)


class TestCorrections(unittest.TestCase):

    def test_settomissing(self):
        import numpy as np
        import pandas as pd
        from diive.preprocessing.corrections import set_exact_values_to_missing
        series = pd.Series([1, 2, 0, 4, 5, 6, 7, 0, 9, 10], name="testdata")
        series_corr = set_exact_values_to_missing(series=series, values=[0, 1, 10], showplot=False)
        expected_series = pd.Series([np.nan, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, np.nan], name="testdata")
        pd.testing.assert_series_equal(series_corr, expected_series)

    def test_winddiroffset(self):
        from diive.configs.exampledata import load_exampledata_winddir
        from diive.preprocessing.corrections import WindDirOffset
        df = load_exampledata_winddir()
        # Get wind direction time series as series
        winddir = df['wind_dir'].copy()
        locs = (winddir.index.year >= 2020) & (winddir.index.year <= 2022)
        winddir = winddir.loc[locs]
        winddir = winddir.dropna()
        wds = WindDirOffset(winddir=winddir, offset_start=-50, offset_end=50,
                            hist_ref_years=[2021, 2022], hist_n_bins=360)
        yearlyoffsets_df = wds.get_yearly_offsets()
        winddir_corrected = wds.get_corrected_wind_directions()
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2020, 'OFFSET'].values, -2)
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2021, 'OFFSET'].values, 0)
        self.assertEqual(yearlyoffsets_df.loc[yearlyoffsets_df['YEAR'] == 2022, 'OFFSET'].values, 0)
        self.assertEqual(winddir_corrected.sum(), 7495054.8)
        self.assertEqual(winddir_corrected.max(), 359.9)
        self.assertEqual(winddir_corrected.min(), 0)

    def test_winddiroffset_passes_verbose_to_the_year_lines(self):
        """`verbose` must win over the module default in both directions."""
        from diive.core.utils.console import (console, get_verbosity, set_verbosity,
                                              VERBOSE_DEBUG, VERBOSE_PROGRESS, VERBOSE_SILENT)
        from diive.preprocessing.corrections import WindDirOffset
        idx = pd.date_range("2021-01-01", "2022-12-31", freq="1D")
        winddir = pd.Series(np.linspace(0, 359, len(idx)), index=idx, name="WD")
        kwargs = dict(winddir=winddir, hist_ref_years=[2021],
                      offset_start=-2, offset_end=2, hist_n_bins=36)

        def _output(default, verbose):
            set_verbosity(default)
            with console.capture() as cap:
                WindDirOffset(**kwargs, verbose=verbose)
            return cap.get()

        saved = get_verbosity()
        try:
            self.assertNotIn("Working on year", _output(VERBOSE_DEBUG, VERBOSE_SILENT))
            self.assertIn("Working on year", _output(VERBOSE_PROGRESS, VERBOSE_DEBUG))
            # None keeps following the module default
            self.assertIn("Working on year", _output(VERBOSE_DEBUG, None))
        finally:
            set_verbosity(saved)


class TestCorrectionsDoNotMutateTheInput(unittest.TestCase):
    """A correction must not touch the Series the caller handed it.

    `setto_threshold`, `set_exact_values_to_missing` and
    `remove_relativehumidity_offset` used to rename the parameter in place
    (`series.name = "input_data"`) so the quickplot panels could tell input from
    output. The returned series was named correctly, but the caller's object was
    left called "input_data". `apply_corrections` copies first, which is why it
    went unnoticed; a direct library call did not.
    """

    def _cases(self):
        import diive as dv
        idx = pd.date_range("2021-01-01", periods=48 * 2, freq="30min",
                            name="TIMESTAMP_MIDDLE")
        values = np.linspace(1.0, 120.0, len(idx))
        radiation = _radiation_series(days=2)
        return [
            ("setto_threshold(max)", pd.Series(values, index=idx, name="MYVAR"),
             lambda s: dv.corrections.setto_threshold(series=s, threshold=10.0, type="max")),
            ("setto_threshold(min)", pd.Series(values, index=idx, name="MYVAR"),
             lambda s: dv.corrections.setto_threshold(series=s, threshold=10.0, type="min")),
            ("set_exact_values_to_missing", pd.Series(values, index=idx, name="MYVAR"),
             lambda s: dv.corrections.set_exact_values_to_missing(series=s, values=[1.0])),
            ("remove_relativehumidity_offset", pd.Series(values, index=idx, name="MYVAR"),
             lambda s: dv.corrections.remove_relativehumidity_offset(series=s)),
            ("remove_nighttime_zero_offset", radiation,
             lambda s: dv.corrections.remove_nighttime_zero_offset(
                 series=s, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)),
            ("setto_value", pd.Series(values, index=idx, name="MYVAR"),
             lambda s: dv.corrections.setto_value(series=s, dates=[str(idx[0])], value=0.0)),
        ]

    def test_input_name_is_preserved(self):
        for label, series, call in self._cases():
            with self.subTest(correction=label):
                name_before = series.name
                call(series)
                self.assertEqual(series.name, name_before,
                                 f"{label} renamed the caller's series")

    def test_input_values_and_index_are_preserved(self):
        for label, series, call in self._cases():
            with self.subTest(correction=label):
                before = series.copy()
                call(series)
                pd.testing.assert_series_equal(series, before)

    def test_output_carries_the_input_name(self):
        for label, series, call in self._cases():
            with self.subTest(correction=label):
                self.assertEqual(call(series).name, series.name)

    def test_a_rejected_call_leaves_the_input_alone(self):
        # setto_threshold validates `type` after the point where it used to
        # rename, so a rejected call still renamed the caller's series.
        import diive as dv
        series = pd.Series([1.0, 2.0], name="MYVAR",
                           index=pd.date_range("2021-01-01", periods=2, freq="30min"))
        with self.assertRaises(ValueError):
            dv.corrections.setto_threshold(series=series, threshold=1.0, type="middle")
        self.assertEqual(series.name, "MYVAR")


class TestSettoCorrections(unittest.TestCase):
    """`setto_value` and `setto_threshold` (`set_exact_values_to_missing` above)."""

    def setUp(self):
        self.idx = pd.date_range("2021-01-01", periods=6, freq="30min")
        self.series = pd.Series([1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                                index=self.idx, name="TA")

    def test_setto_threshold_max_caps_from_above(self):
        from diive.preprocessing.corrections import setto_threshold
        out = setto_threshold(series=self.series.copy(), threshold=12.0, type="max")
        pd.testing.assert_series_equal(
            out, pd.Series([1.0, 5.0, 10.0, 12.0, 12.0, 12.0],
                           index=self.idx, name="TA"))

    def test_setto_threshold_min_floors_from_below(self):
        from diive.preprocessing.corrections import setto_threshold
        out = setto_threshold(series=self.series.copy(), threshold=12.0, type="min")
        pd.testing.assert_series_equal(
            out, pd.Series([12.0, 12.0, 12.0, 15.0, 20.0, 25.0],
                           index=self.idx, name="TA"))

    def test_setto_threshold_rejects_an_unknown_type(self):
        from diive.preprocessing.corrections import setto_threshold
        with self.assertRaises(ValueError):
            setto_threshold(series=self.series.copy(), threshold=1.0, type="middle")

    def test_setto_value_single_timestamp_and_range(self):
        from diive.preprocessing.corrections import setto_value
        # A bare string selects one record; a [start, end] list selects an
        # inclusive range.
        out = setto_value(series=self.series.copy(),
                          dates=[str(self.idx[0]), [str(self.idx[3]), str(self.idx[5])]],
                          value=-9.0)
        pd.testing.assert_series_equal(
            out, pd.Series([-9.0, 5.0, 10.0, -9.0, -9.0, -9.0],
                           index=self.idx, name="TA"))

    def test_setto_value_leaves_the_input_untouched(self):
        from diive.preprocessing.corrections import setto_value
        original = self.series.copy()
        setto_value(series=self.series, dates=[str(self.idx[0])], value=-9.0)
        pd.testing.assert_series_equal(self.series, original)

    # L307: an InfluxDB integer field arrives as int64, and pandas 3 raises when
    # a decimal value is assigned into it.
    def _int_series(self):
        return pd.Series([1, 5, 10, 15, 20, 25], index=self.idx, name="TA", dtype="int64")

    def test_setto_threshold_decimal_on_int64_returns_float(self):
        from diive.preprocessing.corrections import setto_threshold
        out = setto_threshold(series=self._int_series(), threshold=12.5, type="max")
        pd.testing.assert_series_equal(
            out, pd.Series([1.0, 5.0, 10.0, 12.5, 12.5, 12.5],
                           index=self.idx, name="TA"))
        out = setto_threshold(series=self._int_series(), threshold=12.5, type="min")
        pd.testing.assert_series_equal(
            out, pd.Series([12.5, 12.5, 12.5, 15.0, 20.0, 25.0],
                           index=self.idx, name="TA"))

    def test_setto_threshold_integer_on_int64_keeps_int64(self):
        from diive.preprocessing.corrections import setto_threshold
        out = setto_threshold(series=self._int_series(), threshold=12, type="max")
        pd.testing.assert_series_equal(
            out, pd.Series([1, 5, 10, 12, 12, 12], index=self.idx, name="TA", dtype="int64"))

    def test_setto_value_decimal_on_int64_returns_float(self):
        from diive.preprocessing.corrections import setto_value
        series = self._int_series()
        original = series.copy()
        out = setto_value(series=series, dates=[[str(self.idx[3]), str(self.idx[4])]], value=3.7)
        pd.testing.assert_series_equal(
            out, pd.Series([1.0, 5.0, 10.0, 3.7, 3.7, 25.0], index=self.idx, name="TA"))
        pd.testing.assert_series_equal(series, original)

    def test_setto_value_integer_on_int64_keeps_int64(self):
        from diive.preprocessing.corrections import setto_value
        out = setto_value(series=self._int_series(), dates=[str(self.idx[0])], value=3)
        pd.testing.assert_series_equal(
            out, pd.Series([3, 5, 10, 15, 20, 25], index=self.idx, name="TA", dtype="int64"))

    def test_setto_float32_series_keeps_its_dtype(self):
        from diive.preprocessing.corrections import setto_threshold, setto_value
        series = self.series.astype("float32")
        out = setto_threshold(series=series, threshold=12.5, type="max")
        self.assertEqual(out.dtype, np.float32)
        out = setto_value(series=series, dates=[str(self.idx[0])], value=3.7)
        self.assertEqual(out.dtype, np.float32)


class TestOffsetCorrections(unittest.TestCase):
    """`remove_relativehumidity_offset` and `MeasurementOffsetFromReplicate`."""

    def test_relativehumidity_offset_pulls_the_series_under_100(self):
        from diive.preprocessing.corrections import remove_relativehumidity_offset
        # Two days: the first drifts above 100%, the second is well-behaved.
        idx = pd.date_range("2021-01-01", periods=48 * 2, freq="30min", name="TIMESTAMP")
        values = np.concatenate([np.full(48, 110.0), np.full(48, 80.0)])
        series = pd.Series(values, index=idx, name="RH")
        out = remove_relativehumidity_offset(series=series.copy())
        self.assertLessEqual(float(out.max()), 100.0)
        self.assertEqual(out.name, "RH")

    def test_relativehumidity_offset_is_a_noop_without_exceedance(self):
        from diive.preprocessing.corrections import remove_relativehumidity_offset
        idx = pd.date_range("2021-01-01", periods=48, freq="30min", name="TIMESTAMP")
        series = pd.Series(np.full(48, 55.0), index=idx, name="RH")
        out = remove_relativehumidity_offset(series=series.copy())
        # No value exceeds 100, so the offset is zero and nothing moves.
        np.testing.assert_allclose(out.to_numpy(), series.to_numpy())

    def test_measurement_offset_from_replicate_recovers_a_known_offset(self):
        from diive.preprocessing.corrections import MeasurementOffsetFromReplicate
        idx = pd.date_range("2021-01-01", periods=200, freq="30min")
        replicate = pd.Series(np.linspace(0, 20, 200), index=idx, name="TA_REF")
        # The measurement reads 5 units low; the scan must recover exactly +5.
        measurement = (replicate - 5.0).rename("TA")
        corr = MeasurementOffsetFromReplicate(measurement=measurement,
                                              replicate=replicate,
                                              offset_start=-10, offset_end=10,
                                              offset_stepsize=0.5)
        self.assertAlmostEqual(float(corr.get_offset()), 5.0, places=6)
        np.testing.assert_allclose(corr.get_corrected_measurement().to_numpy(),
                                   replicate.to_numpy(), atol=1e-9)

    def test_measurement_offset_from_replicate_passes_verbose_to_the_scan_lines(self):
        """`verbose` must win over the module default in both directions.

        The per-offset line is a `detail` call. Without `verbose=` it followed
        only the level set with `set_verbosity()`, so a caller could neither
        silence nor show it.
        """
        from diive.core.utils.console import (console, get_verbosity, set_verbosity,
                                              VERBOSE_DEBUG, VERBOSE_PROGRESS, VERBOSE_SILENT)
        from diive.preprocessing.corrections import MeasurementOffsetFromReplicate
        idx = pd.date_range("2021-01-01", periods=20, freq="30min")
        replicate = pd.Series(np.linspace(0, 2, 20), index=idx, name="TA_REF")
        measurement = (replicate - 1.0).rename("TA")
        kwargs = dict(measurement=measurement, replicate=replicate,
                      offset_start=-2, offset_end=2, offset_stepsize=1)

        def _scan_output(default, verbose):
            set_verbosity(default)
            with console.capture() as cap:
                MeasurementOffsetFromReplicate(**kwargs, verbose=verbose)
            return cap.get()

        saved = get_verbosity()
        try:
            self.assertNotIn("trying with offset", _scan_output(VERBOSE_DEBUG, VERBOSE_SILENT))
            self.assertIn("trying with offset", _scan_output(VERBOSE_PROGRESS, VERBOSE_DEBUG))
            # None keeps following the module default
            self.assertIn("trying with offset", _scan_output(VERBOSE_DEBUG, None))
        finally:
            set_verbosity(saved)


class TestNighttimeZeroOffset(unittest.TestCase):
    """`remove_nighttime_zero_offset`, its diagnostics, and the result dataclass."""

    @classmethod
    def setUpClass(cls):
        cls.series = _radiation_series(offset=3.0)

    def test_nighttime_is_forced_to_zero(self):
        import diive as dv
        out = dv.corrections.remove_nighttime_zero_offset(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        night = result.nighttime_flag == 1
        self.assertGreater(int(night.sum()), 0)
        np.testing.assert_allclose(out[night].to_numpy(), 0.0)
        self.assertEqual(out.name, "SW_IN")

    def test_detected_offset_matches_the_injected_one(self):
        import diive as dv
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        # The nighttime values are exactly the injected offset, so the daily
        # nighttime mean must recover it.
        self.assertAlmostEqual(float(result.offset.median()), 3.0, places=6)

    def test_diagnostics_corrected_equals_the_plain_correction(self):
        # Documented contract: result.corrected is identical to the function's output.
        import diive as dv
        out = dv.corrections.remove_nighttime_zero_offset(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        pd.testing.assert_series_equal(out, result.corrected)

    def test_clamp_negatives_toggle(self):
        import diive as dv
        # Dip a few daytime records below the offset so the corrected value goes
        # negative -- the only case where the flag makes a difference.
        series = self.series.copy()
        daytime_positions = np.where(series.to_numpy() > 100.0)[0][:5]
        series.iloc[daytime_positions] = 1.0  # below the 3.0 offset

        clamped = dv.corrections.nighttime_zero_offset_diagnostics(
            series=series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET,
            clamp_negatives=True)
        kept = dv.corrections.nighttime_zero_offset_diagnostics(
            series=series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET,
            clamp_negatives=False)

        self.assertEqual(clamped.n_below_zero_after, 0)
        self.assertGreater(kept.n_below_zero_after, 0)
        # Nighttime is forced to zero either way.
        self.assertEqual(clamped.n_below_zero_after_night, 0)
        self.assertEqual(kept.n_below_zero_after_night, 0)

    def test_result_fields_are_consistent(self):
        import diive as dv
        from diive.preprocessing.corrections import NighttimeZeroOffsetResult
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=self.series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        self.assertIsInstance(result, NighttimeZeroOffsetResult)
        # Every series shares the input index.
        for field in ("input", "offset", "corrected_by_offset", "corrected",
                      "nighttime_flag"):
            with self.subTest(field=field):
                self.assertTrue(getattr(result, field).index.equals(self.series.index))
        self.assertEqual(result.n_night, int((result.nighttime_flag == 1).sum()))
        self.assertLess(result.n_night, len(self.series))  # daytime exists too


class TestApplyCorrections(unittest.TestCase):
    """`apply_corrections` is the dispatch table every correction tab routes through."""

    def setUp(self):
        self.idx = pd.date_range("2021-01-01", periods=6, freq="30min")
        self.series = pd.Series([1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                                index=self.idx, name="TA")

    def test_dispatch_matches_calling_the_function_directly(self):
        from diive.preprocessing.corrections import (
            apply_corrections, set_exact_values_to_missing, setto_threshold,
            setto_value)
        cases = [
            ({"key": "setto_max", "kwargs": {"threshold": 12.0}},
             lambda s: setto_threshold(series=s, threshold=12.0, type="max")),
            ({"key": "setto_min", "kwargs": {"threshold": 12.0}},
             lambda s: setto_threshold(series=s, threshold=12.0, type="min")),
            ({"key": "setto_value", "kwargs": {"dates": [str(self.idx[0])], "value": -9.0}},
             lambda s: setto_value(series=s, dates=[str(self.idx[0])], value=-9.0)),
            ({"key": "set_exact_to_missing", "kwargs": {"values": [10.0]}},
             lambda s: set_exact_values_to_missing(series=s, values=[10.0])),
        ]
        for correction, direct in cases:
            with self.subTest(key=correction["key"]):
                via_dispatch = apply_corrections(self.series.copy(), [correction])
                expected = direct(self.series.copy())
                pd.testing.assert_series_equal(via_dispatch, expected)

    def test_dispatch_covers_the_radiation_offset(self):
        import diive as dv
        from diive.preprocessing.corrections import apply_corrections
        series = _radiation_series(offset=3.0, days=3)
        via_dispatch = apply_corrections(
            series.copy(), [{"key": "radiation_zero_offset", "kwargs": {}}],
            lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        expected = dv.corrections.remove_nighttime_zero_offset(
            series=series.copy(), lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        pd.testing.assert_series_equal(via_dispatch, expected)

    def test_dispatch_passes_clamp_negatives_through(self):
        from diive.preprocessing.corrections import apply_corrections
        series = _radiation_series(offset=3.0, days=3)
        daytime_positions = np.where(series.to_numpy() > 100.0)[0][:5]
        series.iloc[daytime_positions] = 1.0
        clamped = apply_corrections(
            series.copy(), [{"key": "radiation_zero_offset", "kwargs": {}}],
            lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        kept = apply_corrections(
            series.copy(),
            [{"key": "radiation_zero_offset", "kwargs": {"clamp_negatives": False}}],
            lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        self.assertEqual(int((clamped < 0).sum()), 0)
        self.assertGreater(int((kept < 0).sum()), 0)

    def test_dispatch_covers_the_relativehumidity_offset(self):
        from diive.preprocessing.corrections import (
            apply_corrections, remove_relativehumidity_offset)
        idx = pd.date_range("2021-01-01", periods=48, freq="30min")
        series = pd.Series(np.full(48, 110.0), index=idx, name="RH")
        via_dispatch = apply_corrections(
            series.copy(), [{"key": "relativehumidity_offset", "kwargs": {}}])
        expected = remove_relativehumidity_offset(series=series.copy())
        pd.testing.assert_series_equal(via_dispatch, expected)

    def test_every_registered_correction_key_is_dispatchable(self):
        # A CorrectionSpec the GUI offers but apply_corrections cannot route
        # would only fail when a user clicks it.
        from diive.preprocessing.corrections import apply_corrections
        from diive.qaqc import CORRECTIONS
        kwargs_for = {
            "radiation_zero_offset": {},
            "relativehumidity_offset": {},
            "setto_max": {"threshold": 12.0},
            "setto_min": {"threshold": 12.0},
            "setto_value": {"dates": [str(self.idx[0])], "value": 0.0},
            "set_exact_to_missing": {"values": [10.0]},
        }
        self.assertEqual({spec.key for spec in CORRECTIONS}, set(kwargs_for),
                         "the CORRECTIONS registry changed -- update this test")
        for spec in CORRECTIONS:
            with self.subTest(key=spec.key):
                series = (_radiation_series(days=3) if spec.needs_coords
                          else self.series.copy())
                out = apply_corrections(
                    series, [{"key": spec.key, "kwargs": kwargs_for[spec.key]}],
                    lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
                self.assertEqual(len(out), len(series))

    def test_unknown_key_raises(self):
        from diive.preprocessing.corrections import apply_corrections
        with self.assertRaises(ValueError) as ctx:
            apply_corrections(self.series.copy(), [{"key": "no_such_correction"}])
        self.assertIn("no_such_correction", str(ctx.exception))

    def test_corrections_apply_in_order(self):
        # Each correction operates on the previous one's output, so order matters:
        # capping at 12 and then flooring at 20 leaves everything at 20, which is
        # only true if the second step really saw the first step's output.
        from diive.preprocessing.corrections import apply_corrections
        out = apply_corrections(self.series.copy(), [
            {"key": "setto_max", "kwargs": {"threshold": 12.0}},
            {"key": "setto_min", "kwargs": {"threshold": 20.0}},
        ])
        np.testing.assert_allclose(out.to_numpy(), 20.0)
        # The reverse order clamps into [20, ...] first, then caps at 12.
        reverse = apply_corrections(self.series.copy(), [
            {"key": "setto_min", "kwargs": {"threshold": 20.0}},
            {"key": "setto_max", "kwargs": {"threshold": 12.0}},
        ])
        np.testing.assert_allclose(reverse.to_numpy(), 12.0)

    def test_empty_correction_list_returns_an_unchanged_copy(self):
        from diive.preprocessing.corrections import apply_corrections
        out = apply_corrections(self.series, [])
        pd.testing.assert_series_equal(out, self.series)
        self.assertIsNot(out, self.series)

    def test_input_values_are_not_mutated(self):
        from diive.preprocessing.corrections import apply_corrections
        original = self.series.copy()
        apply_corrections(self.series, [
            {"key": "setto_max", "kwargs": {"threshold": 2.0}},
            {"key": "set_exact_to_missing", "kwargs": {"values": [1.0]}},
        ])
        pd.testing.assert_series_equal(self.series, original)


if __name__ == '__main__':
    unittest.main()


class TestWindDirOffsetCircularBinning(unittest.TestCase):
    """WindDirOffset compares direction histograms bin by bin, so the bins must
    mean the same thing in both — over the full circle, where 360 deg == 0 deg."""

    N_PER_YEAR = 3000
    PLANTED_OFFSET = 40  # degrees the second year is rotated by

    def _winddir(self, centre: float) -> pd.Series:
        """Two years of directions drawn from one asymmetric distribution; the
        second year is the first rotated by PLANTED_OFFSET (wrapped)."""
        rng = np.random.RandomState(42)
        base = rng.normal(loc=centre, scale=30.0, size=self.N_PER_YEAR) % 360
        ix = pd.DatetimeIndex(
            list(pd.date_range('2020-01-01', periods=self.N_PER_YEAR, freq='1h'))
            + list(pd.date_range('2021-01-01', periods=self.N_PER_YEAR, freq='1h')),
            name='TIMESTAMP_MIDDLE')
        vals = np.concatenate([base, (base + self.PLANTED_OFFSET) % 360])
        return pd.Series(vals, index=ix, name='WD')

    def _detect(self, winddir, n_bins):
        from diive.preprocessing.corrections import WindDirOffset
        wds = WindDirOffset(winddir=winddir, offset_start=-60, offset_end=60,
                            hist_ref_years=[2020], hist_n_bins=n_bins)
        off = wds.get_yearly_offsets().set_index('YEAR')['OFFSET']
        return wds, off

    def test_coarse_bins_still_recover_the_planted_offset(self):
        # The per-year histogram used to be hardcoded to 360 bins while the
        # reference used hist_n_bins, so with any other bin count the two COUNTS
        # series had different lengths and .corr() silently aligned them on the
        # RangeIndex and correlated the overlap only.
        winddir = self._winddir(centre=90.0)
        for n_bins in (36, 72, 360):
            with self.subTest(n_bins=n_bins):
                wds, off = self._detect(winddir, n_bins)
                self.assertEqual(len(wds.ref_results), n_bins,
                                 "reference histogram must honour hist_n_bins")
                self.assertEqual(off[2020], 0)
                self.assertEqual(off[2021], -self.PLANTED_OFFSET)

    def test_offset_across_north_is_recovered(self):
        # Directions centred near 350 deg: the planted rotation carries a large
        # share of them past 360 and back onto 0, so recovering it exercises the
        # wrap rather than a plain shift.
        winddir = self._winddir(centre=350.0)
        crossed = ((winddir.loc[winddir.index.year == 2021] < self.PLANTED_OFFSET).sum())
        self.assertGreater(crossed, 0, "test data must actually cross north")
        _wds, off = self._detect(winddir, n_bins=360)
        self.assertEqual(off[2021], -self.PLANTED_OFFSET)

    def test_corrected_directions_stay_on_the_circle(self):
        winddir = self._winddir(centre=350.0)
        wds, _off = self._detect(winddir, n_bins=360)
        corrected = wds.get_corrected_wind_directions()
        self.assertGreaterEqual(corrected.min(), 0)
        self.assertLess(corrected.max(), 360, "360 deg is 0 deg; it must not appear")

    def test_wrapping_handles_any_magnitude(self):
        from diive.preprocessing.corrections import WindDirOffset
        s = pd.Series([0.0, 360.0, 370.0, -10.0, 720.0, -370.0, np.nan])
        wrapped = WindDirOffset._correct_degrees(s)
        expected = [0.0, 0.0, 10.0, 350.0, 0.0, 350.0]
        self.assertEqual(wrapped.dropna().tolist(), expected)

    def test_histogram_spans_the_full_circle(self):
        # Bin edges are fixed, not data-derived: a series covering only one
        # quadrant must still produce hist_n_bins bins over 0-360, otherwise
        # bin i would mean a different direction in each histogram.
        from diive.preprocessing.corrections import WindDirOffset
        winddir = self._winddir(centre=90.0)
        wds, _ = self._detect(winddir, n_bins=36)
        narrow = pd.Series(np.linspace(10, 80, 500))
        counts = wds._wind_histogram(narrow)
        self.assertEqual(len(counts), 36)
        self.assertEqual(counts.sum(), 500)
        self.assertEqual(counts[20:].sum(), 0, "no counts above 200 deg expected")


# Abisko-like site in polar day in June. potrad's solar-noon shift there vacates
# the last ~17 minutes of each calendar day, which potrad sets to 0 (L244).
POLAR_LAT, POLAR_LON, POLAR_UTC = 68.35, 19.05, 1


def _polar_series(start="2026-06-10", days=4, freq="10min", offset=2.0,
                  lat=POLAR_LAT, lon=POLAR_LON, utc_offset=POLAR_UTC):
    """Radiation with a sensor offset, on TIMESTAMP_MIDDLE, ending at midnight.

    The true signal is potential radiation with potrad's midnight zeros replaced
    by their neighbours, because the sky is not dark at midnight in polar day.
    """
    import diive as dv
    step = pd.Timedelta(freq)
    idx = pd.date_range(start, periods=int(pd.Timedelta(days=days) / step), freq=freq) + step / 2
    idx.name = "TIMESTAMP_MIDDLE"
    pr = dv.variables.potrad(timestamp_index=idx, lat=lat, lon=lon, utc_offset=utc_offset)
    true = pr.where(pr > 0.001).interpolate().bfill().ffill()
    measured = pd.Series(0.7 * true.to_numpy() + offset, index=idx, name="SW_IN")
    return measured, 0.7 * true, pr


def _true_night(index, lat, lon, utc_offset, period_min):
    """Night where potential radiation is below 0.001 W m-2 with the minutes
    potrad's solar-noon shift vacates taken from the neighbouring day's curve."""
    from diive.variables.radiation import _noaa_solar_noon_minutes, _oneflux_daily_rpot
    starts = index - pd.Timedelta(minutes=period_min / 2)
    epoch = starts.min().normalize() - pd.Timedelta(days=1)
    off = np.floor((starts - epoch) / pd.Timedelta(minutes=1) + 0.5).astype(np.int64)
    dates = pd.date_range(epoch, periods=int((off.max() + period_min) // 1440) + 3, freq="D")
    shifts = np.array([_noaa_solar_noon_minutes(d.year, d.month, d.day, -lon, -utc_offset) - 720
                       for d in dates])
    rows = np.array([d.dayofyear - 1 for d in dates])
    total = np.zeros(len(index))
    for k in range(period_min):
        minute = off + k
        day = minute // 1440
        i = (minute % 1440) - shifts[day]
        day = np.where(i < 0, day - 1, np.where(i >= 1440, day + 1, day))
        i = np.mod(i, 1440)
        total += _oneflux_daily_rpot(lat, rows[day] + i / 1440.0 + 1.0, i / 60.0)
    return total / period_min < 0.001


class TestNighttimeZeroOffsetPolarDay(unittest.TestCase):
    """potrad's midnight zeros in polar day are not nighttime (L244)."""

    def _diag(self, series, lat=POLAR_LAT, lon=POLAR_LON, utc_offset=POLAR_UTC, **kwargs):
        import diive as dv
        return dv.corrections.nighttime_zero_offset_diagnostics(
            series=series, lat=lat, lon=lon, utc_offset=utc_offset, **kwargs)

    def test_10min_midnight_zeros_are_not_zeroed_or_used_as_offset(self):
        measured, true, pr = _polar_series(freq="10min")
        # Precondition: potrad has one zero record per day, the last day's at
        # the very end of the series.
        self.assertEqual(int((pr <= 0.001).sum()), 4)
        self.assertTrue(pr.iloc[-1] <= 0.001)
        result = self._diag(measured.copy())
        self.assertEqual(result.n_night, 0)
        # No night, so no offset can be estimated and none is removed. Before
        # the fix the real values at the zero records (17-19 W m-2) were the offset.
        np.testing.assert_allclose(result.offset.to_numpy(), 0.0)
        pd.testing.assert_series_equal(result.corrected, measured)
        self.assertAlmostEqual(float((result.corrected - true).mean()), 2.0, places=9)

    def test_1min_midnight_zero_runs_are_not_nighttime(self):
        measured, _, pr = _polar_series(freq="1min", days=2)
        self.assertGreater(int((pr <= 0.001).sum()), 20)  # one run of ~17 per day
        result = self._diag(measured.copy())
        self.assertEqual(result.n_night, 0)
        pd.testing.assert_series_equal(result.corrected, measured)

    def test_no_nighttime_at_all_leaves_the_offset_at_zero(self):
        # 30-min polar day has no zero record, so no nighttime at all. The offset
        # used to be NaN there, which made every corrected record NaN.
        measured, _, pr = _polar_series(freq="30min")
        self.assertFalse(bool((pr <= 0.001).any()))
        result = self._diag(measured.copy())
        self.assertEqual(result.n_night, 0)
        self.assertFalse(result.corrected.isna().any())
        pd.testing.assert_series_equal(result.corrected, measured)

    def test_real_nights_next_to_the_midnight_zeros_stay_nighttime(self):
        # Utqiagvik in early May: the shift vacates 00:00-01:27, the real night
        # lasts from about 00:10 to 02:45. Records of the real night stay
        # nighttime, the vacated records before it with the sun up do not (L266).
        lat, lon, utc = 71.29, -156.79, -9
        measured, _, pr = _polar_series(start="2026-05-10", days=4, lat=lat, lon=lon,
                                        utc_offset=utc)
        result = self._diag(measured.copy(), lat=lat, lon=lon, utc_offset=utc)
        truth = _true_night(measured.index, lat, lon, utc, period_min=10)
        self.assertGreater(int(truth.sum()), 50)
        self.assertGreater(int(((pr <= 0.001) & ~truth).sum()), 0)  # precondition
        np.testing.assert_array_equal(result.nighttime_flag.to_numpy() == 1, truth)

    def test_late_solar_noon_follows_the_sun_across_midnight(self):
        # L266: where solar noon is late, nights mixing the midnight zeros with a
        # real night kept sunlit records as nighttime (10-min), and the old
        # length tolerance made short real nights daytime (hourly).
        for lat, lon, utc, freq in ((69.0, -30.0, 0, "10min"), (71.3, -156.6, -9, "10min"),
                                    (67.0, -110.0, -6, "60min"), (69.0, -30.0, 0, "60min")):
            with self.subTest(lon=lon, freq=freq):
                step = pd.Timedelta(freq)
                idx = pd.date_range("2023-01-01", "2024-01-01", freq=freq, inclusive="left") + step / 2
                series = pd.Series(1.0, index=idx, name="SW_IN")
                result = self._diag(series, lat=lat, lon=lon, utc_offset=utc)
                truth = _true_night(idx, lat, lon, utc, period_min=int(step / pd.Timedelta("1min")))
                np.testing.assert_array_equal(result.nighttime_flag.to_numpy() == 1, truth)

    def test_normal_latitudes_keep_the_potrad_night(self):
        # Bit-identity guard: away from polar day the night flag is potrad's own.
        import diive as dv
        from diive.variables import daytime_nighttime_flag_from_swinpot
        for freq in ("30min", "10min", "1min"):
            with self.subTest(freq=freq):
                step = pd.Timedelta(freq)
                idx = pd.date_range("2021-06-01", periods=int(pd.Timedelta(days=3) / step),
                                    freq=freq) + step / 2
                pr = dv.variables.potrad(timestamp_index=idx, lat=LAT, lon=LON,
                                         utc_offset=UTC_OFFSET)
                series = pd.Series(np.where(pr > 0.001, 0.7 * pr, 0.0) + 3.0, index=idx,
                                   name="SW_IN")
                _, raw_night = daytime_nighttime_flag_from_swinpot(pr, nighttime_threshold=0.001)
                result = self._diag(series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
                pd.testing.assert_series_equal(result.nighttime_flag, raw_night)
                self.assertAlmostEqual(float(result.offset.median()), 3.0, places=9)

    def test_night_next_to_an_index_gap_stays_nighttime(self):
        # The evening night of 1 June ends at midnight because the records
        # 00:00-06:00 of 2 June are missing. It is a real night.
        series = _radiation_series(offset=3.0, days=3)
        gap = (series.index >= "2021-06-02 00:00") & (series.index < "2021-06-02 06:00")
        series = series[~gap]
        result = self._diag(series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        evening = (series.index >= "2021-06-01 22:00") & (series.index < "2021-06-02 00:00")
        self.assertTrue((result.nighttime_flag[evening] == 1).all())
        self.assertTrue((result.corrected[evening] == 0).all())


class TestNighttimeZeroOffsetSwinPot(unittest.TestCase):
    """The `swin_pot` parameter replaces the computed potential radiation."""

    def test_swin_pot_from_potrad_gives_the_default_result(self):
        import diive as dv
        series = _radiation_series(offset=3.0)
        swin_pot = dv.variables.potrad(timestamp_index=series.index, lat=LAT, lon=LON,
                                       utc_offset=UTC_OFFSET)
        default = dv.corrections.nighttime_zero_offset_diagnostics(
            series=series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET)
        given = dv.corrections.nighttime_zero_offset_diagnostics(
            series=series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET, swin_pot=swin_pot)
        for field in ("offset", "corrected_by_offset", "corrected", "nighttime_flag"):
            with self.subTest(field=field):
                pd.testing.assert_series_equal(getattr(given, field), getattr(default, field))
        pd.testing.assert_series_equal(
            dv.corrections.remove_nighttime_zero_offset(
                series=series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET, swin_pot=swin_pot),
            default.corrected)

    def test_swin_pot_with_one_extra_record_per_end_matches_the_default_in_polar_day(self):
        import diive as dv
        measured, _, _ = _polar_series(freq="10min")
        step = pd.Timedelta("10min")
        wider = pd.DatetimeIndex([measured.index[0] - step]).append(measured.index).append(
            pd.DatetimeIndex([measured.index[-1] + step]))
        swin_pot = dv.variables.potrad(timestamp_index=wider, lat=POLAR_LAT, lon=POLAR_LON,
                                       utc_offset=POLAR_UTC)
        default = dv.corrections.nighttime_zero_offset_diagnostics(
            series=measured, lat=POLAR_LAT, lon=POLAR_LON, utc_offset=POLAR_UTC)
        given = dv.corrections.nighttime_zero_offset_diagnostics(
            series=measured, lat=POLAR_LAT, lon=POLAR_LON, utc_offset=POLAR_UTC,
            swin_pot=swin_pot)
        pd.testing.assert_series_equal(given.corrected, default.corrected)
        pd.testing.assert_series_equal(given.nighttime_flag, default.nighttime_flag)

    def test_swin_pot_on_the_series_index_finds_the_zero_record_at_the_series_end(self):
        # Each record is judged on its own window, so no record beyond the end is needed
        import diive as dv
        measured, _, _ = _polar_series(freq="10min")
        swin_pot = dv.variables.potrad(timestamp_index=measured.index, lat=POLAR_LAT,
                                       lon=POLAR_LON, utc_offset=POLAR_UTC)
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=measured, lat=POLAR_LAT, lon=POLAR_LON, utc_offset=POLAR_UTC,
            swin_pot=swin_pot)
        self.assertEqual(result.n_night, 0)

    def test_swin_pot_on_a_finer_grid_uses_each_record_period(self):
        # Meteo screening's mixed grids: 10-min records at their END slot of a
        # 1-min grid (slot = END - 30 s), the slots in between empty (NaN).
        import diive as dv
        lat, lon, utc = 69.0, -30.0, 0
        ends = pd.date_range("2023-06-10 00:10", "2023-06-20 00:00", freq="10min")
        middles = ends - pd.Timedelta("5min")
        pr = dv.variables.potrad(timestamp_index=middles, lat=lat, lon=lon, utc_offset=utc)
        slots = pd.date_range(ends[0], ends[-1], freq="1min") - pd.Timedelta("30s")
        swin_pot = pd.Series(np.nan, index=slots)
        swin_pot.loc[ends - pd.Timedelta("30s")] = pr.to_numpy()
        series = pd.Series(1.0, index=ends - pd.Timedelta("30s"), name="SW_IN")
        result = dv.corrections.nighttime_zero_offset_diagnostics(
            series=series, lat=lat, lon=lon, utc_offset=utc, swin_pot=swin_pot)
        truth = _true_night(middles, lat, lon, utc, period_min=10)
        self.assertGreater(int(((pr <= 0.001).to_numpy() & ~truth).sum()), 0)  # precondition
        np.testing.assert_array_equal(result.nighttime_flag.to_numpy() == 1, truth)

    @staticmethod
    def _coarse_then_fine(t_end, lat, lon, utc):
        """10-min records up to END *t_end*, then 1-min records, on a 1-min slot grid."""
        import diive as dv
        ends10 = pd.date_range(t_end - pd.Timedelta("1D"), t_end, freq="10min")
        ends1 = pd.date_range(t_end + pd.Timedelta("1min"), periods=600, freq="1min")
        m10, m1 = ends10 - pd.Timedelta("5min"), ends1 - pd.Timedelta("30s")
        slots = pd.date_range(ends10[0], ends1[-1], freq="1min") - pd.Timedelta("30s")
        swin_pot = pd.Series(np.nan, index=slots)
        swin_pot.loc[ends10 - pd.Timedelta("30s")] = dv.variables.potrad(
            timestamp_index=m10, lat=lat, lon=lon, utc_offset=utc).to_numpy()
        swin_pot.loc[m1] = dv.variables.potrad(
            timestamp_index=m1, lat=lat, lon=lon, utc_offset=utc).to_numpy()
        record_slots = ends10.append(ends1) - pd.Timedelta("30s")
        period_sec = pd.Series(np.r_[np.full(len(ends10), 600.0), np.full(len(ends1), 60.0)],
                               index=record_slots)
        truth = np.r_[_true_night(m10, lat, lon, utc, period_min=10),
                      _true_night(m1, lat, lon, utc, period_min=1)]
        return pd.Series(1.0, index=record_slots, name="SW_IN"), swin_pot, period_sec, truth

    def test_swin_pot_period_sec_gives_the_last_coarse_record_its_own_window(self):
        # L278: without periods the 10-min record ending 00:20, just before the
        # 1-min block, got a 1-min window (the spacing to its nearer neighbour).
        # Utqiagvik in May: the real night begins inside the minutes potrad vacates.
        import diive as dv
        lat, lon, utc = 71.29, -156.79, -9
        t_end = pd.Timestamp("2026-05-12 00:20")
        series, swin_pot, period_sec, truth = self._coarse_then_fine(t_end, lat, lon, utc)
        kwargs = dict(series=series, lat=lat, lon=lon, utc_offset=utc, swin_pot=swin_pot)
        spacing = dv.corrections.nighttime_zero_offset_diagnostics(**kwargs)
        own = dv.corrections.nighttime_zero_offset_diagnostics(**kwargs, swin_pot_period_sec=period_sec)
        last_coarse = series.index.get_loc(t_end - pd.Timedelta("30s"))
        wrong = np.flatnonzero((spacing.nighttime_flag.to_numpy() == 1) != truth)
        np.testing.assert_array_equal(wrong, [last_coarse])  # precondition
        np.testing.assert_array_equal(own.nighttime_flag.to_numpy() == 1, truth)

    def test_swin_pot_period_sec_falls_back_to_the_spacing(self):
        # NaN or missing periods use the spacing, so all-NaN periods change nothing
        import diive as dv
        lat, lon, utc = 71.29, -156.79, -9
        series, swin_pot, period_sec, _ = self._coarse_then_fine(
            pd.Timestamp("2026-05-12 00:20"), lat, lon, utc)
        kwargs = dict(series=series, lat=lat, lon=lon, utc_offset=utc, swin_pot=swin_pot)
        spacing = dv.corrections.nighttime_zero_offset_diagnostics(**kwargs)
        for periods in (period_sec * np.nan, period_sec.iloc[:0]):
            with self.subTest(n=len(periods)):
                given = dv.corrections.nighttime_zero_offset_diagnostics(
                    **kwargs, swin_pot_period_sec=periods)
                pd.testing.assert_series_equal(given.nighttime_flag, spacing.nighttime_flag)
                pd.testing.assert_series_equal(given.corrected, spacing.corrected)

    def test_swin_pot_is_used_instead_of_the_coordinates(self):
        import diive as dv
        series = _radiation_series(offset=3.0, days=2)
        # All zero: every record is nighttime, so everything is zeroed.
        swin_pot = pd.Series(0.0, index=series.index)
        out = dv.corrections.remove_nighttime_zero_offset(
            series=series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET, swin_pot=swin_pot)
        np.testing.assert_allclose(out.to_numpy(), 0.0)

    def test_swin_pot_must_cover_the_series(self):
        import diive as dv
        series = _radiation_series(offset=3.0, days=2)
        swin_pot = pd.Series(0.0, index=series.index[1:])
        with self.assertRaises(ValueError):
            dv.corrections.remove_nighttime_zero_offset(
                series=series, lat=LAT, lon=LON, utc_offset=UTC_OFFSET, swin_pot=swin_pot)


class TestRelativeHumidityOffsetCarry(unittest.TestCase):
    """Documented: the daily offset is interpolated and carried to all days (L247)."""

    def test_one_foggy_day_lowers_every_record(self):
        from diive.preprocessing.corrections import remove_relativehumidity_offset
        idx = pd.date_range("2026-03-01 00:05", periods=144 * 10, freq="10min")
        hour = idx.hour + idx.minute / 60
        rh = np.asarray(75 + 20 * np.cos(2 * np.pi * (hour - 4) / 24), dtype=float)
        rh[(idx.day == 1) & (hour >= 3) & (hour < 5)] = 103.0
        series = pd.Series(rh, index=idx, name="RH")
        out = remove_relativehumidity_offset(series=series.copy())
        np.testing.assert_allclose((series - out).to_numpy(), 3.0)


class TestSetExactValuesToMissingTolerance(unittest.TestCase):
    """float32 storage must still match the value shown (L249)."""

    def test_float32_value_matches(self):
        from diive.preprocessing.corrections import set_exact_values_to_missing
        values = np.array([np.float32(0.1), 0.1, 0.1000002, 5.0, np.float32(23.7)], dtype=float)
        series = pd.Series(values, name="TA")
        out = set_exact_values_to_missing(series=series, values=[0.1, 23.7], verbose=0)
        # float32 0.1 and 23.7 match, 0.1000002 (2e-6 relative) does not
        self.assertEqual(out.isna().tolist(), [True, True, False, False, True])

    def test_zero_and_missing_code_match_exactly(self):
        from diive.preprocessing.corrections import set_exact_values_to_missing
        series = pd.Series([0.0, 1e-12, -1e-30, -9999.0, -9998.0, np.float32(-9999.0)], name="TA")
        out = set_exact_values_to_missing(series=series, values=[0, -9999], verbose=0)
        self.assertEqual(out.isna().tolist(), [True, False, False, True, False, True])

    def test_non_numeric_values_raise_a_clear_error(self):
        # L267: these raised TypeError/DTypePromotionError from np.isclose
        from diive.preprocessing.corrections import set_exact_values_to_missing
        series = pd.Series([-9999.0, 1.0], name="TA")
        for values in ([None], ["-9999"], [pd.NA], [-9999, True], "-9999", -9999):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "values must be"):
                    set_exact_values_to_missing(series=series, values=values, verbose=0)

    def test_numpy_numbers_and_nan_are_accepted(self):
        from diive.preprocessing.corrections import set_exact_values_to_missing
        series = pd.Series([-9999.0, 1.0, np.nan], name="TA")
        out = set_exact_values_to_missing(series=series, values=[np.int64(-9999), np.nan],
                                          verbose=0)
        self.assertEqual(out.isna().tolist(), [True, False, True])


class TestSettoVerbose(unittest.TestCase):
    """`verbose` reaches the reports of the set-to corrections (L252)."""

    def _output(self, func, default, **kwargs):
        from diive.core.utils.console import console, set_verbosity
        set_verbosity(default)
        with console.capture() as cap:
            func(**kwargs)
        return cap.get()

    def test_set_exact_values_to_missing(self):
        from diive.core.utils.console import (get_verbosity, set_verbosity,
                                              VERBOSE_DEBUG, VERBOSE_PROGRESS)
        from diive.preprocessing.corrections import set_exact_values_to_missing
        kwargs = dict(series=pd.Series([0.0, 1.0], name="TA"), values=[0])
        func = set_exact_values_to_missing
        saved = get_verbosity()
        try:
            self.assertEqual(self._output(func, VERBOSE_DEBUG, verbose=0, **kwargs), "")
            # The default prints the report as before
            out = self._output(func, VERBOSE_PROGRESS, **kwargs)
            self.assertIn("Number of records set to missing: 1", out)
            self.assertNotIn("Locations", out)
            self.assertIn("Locations", self._output(func, VERBOSE_PROGRESS, verbose=VERBOSE_DEBUG,
                                                    **kwargs))
        finally:
            set_verbosity(saved)

    def test_setto_threshold(self):
        from diive.core.utils.console import (get_verbosity, set_verbosity,
                                              VERBOSE_DEBUG, VERBOSE_PROGRESS)
        from diive.preprocessing.corrections import setto_threshold
        kwargs = dict(series=pd.Series([1.0, 20.0], name="TA"), threshold=10.0, type="max")
        saved = get_verbosity()
        try:
            self.assertEqual(self._output(setto_threshold, VERBOSE_DEBUG, verbose=0, **kwargs), "")
            self.assertIn("QA/QC set to threshold value",
                          self._output(setto_threshold, VERBOSE_PROGRESS, **kwargs))
        finally:
            set_verbosity(saved)
