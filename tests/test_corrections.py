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
