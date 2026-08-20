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

    def test_screening_report(self):
        # The per-step report: potential/measured/retained/rejected, one row per
        # test (cumulative), with percentages that add up.
        df = self._build()
        qcf = FlagQCF(df=df, target_col='FC')
        qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        table, text = qcf.screening_report()

        # No swinpot -> overall only; one row per test column (5 tests).
        self.assertEqual(set(table['period']), {'OVERALL'})
        self.assertEqual(len(table), 5)

        # Final (last-step) cumulative numbers match the known QCF distribution:
        # 6 measured records, QCF == [0,1,1,2,2,2] -> 3 retained, 3 rejected.
        final = table.iloc[-1]
        self.assertEqual(int(final['n_potential']), 6)
        self.assertEqual(int(final['n_measured']), 6)
        self.assertEqual(int(final['n_retained']), 3)
        self.assertEqual(int(final['n_rejected']), 3)
        self.assertEqual(int(final['n_retained'] + final['n_rejected']),
                         int(final['n_measured']))
        self.assertAlmostEqual(float(final['perc_rejected']), 50.0, places=4)
        self.assertIn('STEPWISE SCREENING REPORT', text)


def _flag_frame(n_records: int, flag_rows=None, with_swinpot: bool = False):
    """Frame of five FLAG_FC_*_TEST columns plus the FC series.

    Column names must contain '_FC_' and end with '_TEST' for FlagQCF to pick
    them up for target_col='FC'.
    """
    idx = pd.date_range('2022-06-01', periods=n_records, freq='30min',
                        name='TIMESTAMP_END')
    cols = [f'FLAG_FC_T{i}_TEST' for i in range(1, 6)]
    if flag_rows is None:
        df = pd.DataFrame({c: np.zeros(n_records) for c in cols}, index=idx)
    else:
        df = pd.DataFrame(flag_rows, columns=cols, index=idx, dtype=float)
    df['FC'] = np.arange(1.0, n_records + 1.0)
    if with_swinpot:
        # A real diel cycle so the 20 W/m2 nighttime threshold splits the record.
        df['SW_IN_POT'] = np.clip(
            600 * np.sin(2 * np.pi * ((np.arange(n_records) % 48) - 12) / 48), 0, None)
    return df


class TestFlagQCFRules(unittest.TestCase):
    """The QCF decision rules, at the boundaries the existing tests skip."""

    def test_soft_flag_threshold_boundary(self):
        from diive.qaqc import FlagQCF
        # The rule is "more than three soft flags -> QCF 2", so exactly three
        # must still be 1. The existing aggregation test jumps from 2 to 4 soft
        # flags and never touches the boundary itself.
        rows = [
            [0, 0, 0, 0, 0],  # none          -> 0
            [1, 1, 1, 0, 0],  # exactly 3 soft -> 1
            [1, 1, 1, 1, 0],  # 4 soft         -> 2
            [2, 0, 0, 0, 0],  # a single hard  -> 2
        ]
        qcf = FlagQCF(df=_flag_frame(4, rows), target_col='FC')
        qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        self.assertEqual(list(qcf.get()[qcf.flagqcfcol].astype(int)), [0, 1, 2, 2])

    def test_flag_sums_are_exposed(self):
        from diive.qaqc import FlagQCF
        rows = [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 0, 0, 0]]
        qcf = FlagQCF(df=_flag_frame(3, rows), target_col='FC')
        qcf.calculate()
        out = qcf.get()
        # Hard flags are summed by *value* (2 each), soft flags by value 1 --
        # so two hard flags sum to 4, not 2.
        self.assertEqual(list(out[qcf.sumsoftflagscol]), [0.0, 2.0, 0.0])
        self.assertEqual(list(out[qcf.sumhardflagscol]), [0.0, 0.0, 4.0])
        self.assertEqual(list(out[qcf.sumflagscol]), [0.0, 2.0, 4.0])


class TestFlagQCFDayNight(unittest.TestCase):
    """The `swinpot_col` path: separate day/night acceptance thresholds.

    Without swinpot no day/night split happens at all, which is the only branch
    the existing tests exercised.
    """

    @classmethod
    def setUpClass(cls):
        # One soft flag on every record -> QCF 1 everywhere before thresholds.
        cls.df = _flag_frame(48 * 4, with_swinpot=True)
        cls.df['FLAG_FC_T1_TEST'] = 1.0
        cls.daytime = cls.df['SW_IN_POT'] >= 20

    def _qcf(self, daytime_below, nighttime_below):
        from diive.qaqc import FlagQCF
        qcf = FlagQCF(df=self.df, target_col='FC', swinpot_col='SW_IN_POT',
                      nighttime_threshold=20)
        qcf.calculate(daytime_accept_qcf_below=daytime_below,
                      nighttime_accept_qcf_below=nighttime_below)
        return qcf

    def test_stricter_daytime_threshold_rejects_marginal_daytime_only(self):
        qcf = self._qcf(daytime_below=1, nighttime_below=2)
        flag = qcf.get()[qcf.flagqcfcol]
        # QCF 1 clears the nighttime threshold of 2 but not the daytime one of 1.
        self.assertEqual(set(flag[self.daytime].astype(int)), {2})
        self.assertEqual(set(flag[~self.daytime].astype(int)), {1})

    def test_equal_thresholds_leave_marginal_records_alone(self):
        qcf = self._qcf(daytime_below=2, nighttime_below=2)
        flag = qcf.get()[qcf.flagqcfcol]
        self.assertEqual(set(flag.astype(int)), {1})

    def test_screening_report_splits_day_and_night(self):
        # Without swinpot the report is OVERALL-only (asserted in
        # TestFlagQCF.test_screening_report); with it, both periods appear.
        qcf = self._qcf(daytime_below=2, nighttime_below=2)
        table, text = qcf.screening_report()
        self.assertEqual(set(table['period']), {'OVERALL', 'DAYTIME', 'NIGHTTIME'})
        self.assertIn('STEPWISE SCREENING REPORT', text)
        # Day and night counts must partition the overall record count.
        per_period = table.groupby('period')['n_potential'].max()
        self.assertEqual(int(per_period['DAYTIME'] + per_period['NIGHTTIME']),
                         int(per_period['OVERALL']))


class TestFlagQCFReports(unittest.TestCase):
    """The three console reports, none of which had any coverage."""

    @classmethod
    def setUpClass(cls):
        from diive.qaqc import FlagQCF
        df = _flag_frame(48 * 4, with_swinpot=True)
        df['FLAG_FC_T1_TEST'] = 1.0
        df.loc[df.index[:20], 'FLAG_FC_T2_TEST'] = 2.0
        cls.qcf = FlagQCF(df=df, target_col='FC', swinpot_col='SW_IN_POT')
        cls.qcf.calculate()

    def _emits(self, method_name):
        """Return what the report wrote to stdout.

        Captured from stdout rather than via `add_console_sink`: that registers
        the mirror on the *current* module-level console, while `qcf.py` did
        `from ... import console as _console` at import time and so keeps
        whatever object existed then. `refresh_console()` (which
        `tests/test_console.py` exercises) rebinds the module global, and the
        docstring says so explicitly -- "modules that imported console by name
        keep the previous object". A sink-based check therefore passes or fails
        depending on test file order. Rich resolves `sys.stdout` at write time,
        so redirect_stdout catches the output whichever console object is used.
        """
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            getattr(self.qcf, method_name)()
        return buffer.getvalue()

    def test_reports_emit_output(self):
        expected = {
            'report_qcf_flags': 'INDIVIDUAL TEST FLAG STATISTICS',
            'report_qcf_evolution': 'QCF EVOLUTION',
            'report_qcf_series': 'QCF QUALITY CONTROL REPORT',
        }
        for method, heading in expected.items():
            with self.subTest(report=method):
                text = self._emits(method)
                self.assertTrue(text.strip(),
                                f"{method}() produced no console output")
                self.assertIn(heading, text)
                self.assertIn('FC', text)


class TestFlagQCFPlots(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from diive.qaqc import FlagQCF
        df = _flag_frame(48 * 4, with_swinpot=True)
        df['FLAG_FC_T1_TEST'] = 1.0
        cls.qcf = FlagQCF(df=df, target_col='FC', swinpot_col='SW_IN_POT')
        cls.qcf.calculate()

    def test_plots_produce_a_figure(self):
        import matplotlib.pyplot as plt
        for method in ('showplot_qcf_heatmaps', 'showplot_qcf_timeseries'):
            with self.subTest(plot=method):
                plt.close('all')
                getattr(self.qcf, method)()
                self.assertGreater(len(plt.get_fignums()), 0)
                plt.close('all')


class TestValidateIdString(unittest.TestCase):
    """`diive.core.funcs.funcs.validate_id_string`, the shared idstr normaliser.

    Every caller interpolates the result straight into a column name, so it must
    return a string in all cases -- returning None for a falsy input produced
    names like `FLAGNone_FC_QCF`.
    """

    def test_normalisation(self):
        from diive.core.funcs.funcs import validate_id_string
        cases = {None: '', '': '', 'L2': '_L2', '_L2': '_L2', 'L3.1': '_L3.1'}
        for given, expected in cases.items():
            with self.subTest(idstr=given):
                self.assertEqual(validate_id_string(idstr=given), expected)

    def test_result_is_always_a_string(self):
        from diive.core.funcs.funcs import validate_id_string
        for given in (None, '', 'L2'):
            with self.subTest(idstr=given):
                self.assertIsInstance(validate_id_string(idstr=given), str)

    def test_falsy_result_stays_falsy(self):
        # Callers such as FlagBase and the USTAR flaggers branch on `if idstr:`.
        # Normalising None to '' must not change that decision.
        from diive.core.funcs.funcs import validate_id_string
        self.assertFalse(validate_id_string(idstr=None))
        self.assertFalse(validate_id_string(idstr=''))
        self.assertTrue(validate_id_string(idstr='L2'))


class TestFlagQCFColumnNames(unittest.TestCase):
    """Output column names, with and without the optional `idstr`.

    Omitting `idstr` used to yield `FLAGNone_FC_QCF` / `FCNone_QCF`, because the
    normaliser returned None and the f-string rendered it as text.
    """

    def _names(self, **kwargs):
        from diive.qaqc import FlagQCF
        qcf = FlagQCF(df=_flag_frame(4), target_col='FC', **kwargs)
        qcf.calculate()
        return qcf, (qcf.flagqcfcol, qcf.filteredseriescol, qcf.filteredseriescol_hq)

    def test_names_without_idstr(self):
        _, names = self._names()
        self.assertEqual(names, ('FLAG_FC_QCF', 'FC_QCF', 'FC_QCF0'))

    def test_names_with_idstr(self):
        _, names = self._names(idstr='L2')
        self.assertEqual(names, ('FLAG_L2_FC_QCF', 'FC_L2_QCF', 'FC_L2_QCF0'))

    def test_leading_underscore_is_not_doubled(self):
        # 'L2' and '_L2' are the same identifier.
        self.assertEqual(self._names(idstr='_L2')[1], self._names(idstr='L2')[1])

    def test_empty_idstr_matches_an_omitted_one(self):
        self.assertEqual(self._names(idstr='')[1], self._names()[1])

    def test_no_output_column_carries_a_literal_none(self):
        qcf, _ = self._names()
        offenders = [c for c in qcf.get().columns if 'None' in str(c)]
        self.assertEqual([], offenders)


class TestFlagQCFValidation(unittest.TestCase):

    def test_missing_columns_raise(self):
        from diive.qaqc import FlagQCF
        df = _flag_frame(4, with_swinpot=True)
        with self.subTest(column='target_col'):
            with self.assertRaises(KeyError):
                FlagQCF(df=df, target_col='NOT_A_COLUMN')
        with self.subTest(column='swinpot_col'):
            with self.assertRaises(KeyError):
                FlagQCF(df=df, target_col='FC', swinpot_col='NOT_A_COLUMN')


class TestFlagQCFNeverNaN(unittest.TestCase):
    """QCF is always 0/1/2, never NaN (finding L8).

    The flag sums treat a NaN test flag as 0, so a record whose every test flag
    is NaN lands at QCF=0 ("nothing flagged it"), not at NaN. The NaN in
    `_calculate_flag_qcf` is only the column initialisation.
    """

    def test_all_nan_test_flags_give_qcf_zero(self):
        from diive.qaqc import FlagQCF
        rows = [
            [np.nan] * 5,  # no flag available at all -> 0, not NaN
            [np.nan, np.nan, 0, np.nan, np.nan],  # one test passed -> 0
            [np.nan, 1, np.nan, np.nan, np.nan],  # one soft flag   -> 1
            [np.nan, np.nan, 2, np.nan, np.nan],  # one hard flag   -> 2
        ]
        qcf = FlagQCF(df=_flag_frame(4, rows), target_col='FC')
        qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
        flag = qcf.get()[qcf.flagqcfcol]
        self.assertFalse(flag.isna().any(), "QCF must never be NaN")
        self.assertEqual(list(flag.astype(int)), [0, 0, 1, 2])


class TestEddyProFlags(unittest.TestCase):
    """Conversion of EddyPro output flags to diive format (0/1/2)."""

    def _idx(self, n: int):
        return pd.date_range('2022-06-01', periods=n, freq='30min', name='TIMESTAMP_END')

    def test_ssitc_values_are_passed_through(self):
        # Finding L15: SSITC is deliberately NOT thresholded -- EddyPro's 0/1/2
        # already is the diive scale, so intermediate quality 1 stays a soft flag.
        from diive.preprocessing.qaqc.eddyproflags import flag_ssitc_eddypro_test
        df = pd.DataFrame({'FC_SSITC_TEST': [0.0, 1.0, 2.0]}, index=self._idx(3))
        flag = flag_ssitc_eddypro_test(df=df, flux='FC')
        self.assertEqual(list(flag), [0.0, 1.0, 2.0])

        # ... and `setflag_timeperiod` is what promotes 1 to 2 for chosen periods.
        promoted = flag_ssitc_eddypro_test(
            df=df, flux='FC', setflag_timeperiod={2: [[1, '2022-06-01', '2022-06-02']]})
        self.assertEqual(list(promoted), [0.0, 2.0, 2.0])

    def test_negative_code_is_not_testable(self):
        """L89: a negative code is not a valid EddyPro flag and must not read as one.

        The digit is taken from the string form, where a minus sign shifts every
        position: -9999 becomes '-9999.0', whose character 6 is '0' -- a junk
        sentinel arriving as "tested and good". At position 1 the same value read
        as '9' (missing) by luck, while -1 read as a soft warning.
        """
        import numpy as np
        import pandas as pd
        from diive.preprocessing.qaqc.eddyproflags import (
            _extract_and_convert_flag_from_multidigit as extract)
        df = pd.DataFrame({'C': [800000000, -9999, -1, 811111111, np.nan]})
        for position in (1, 6):
            with self.subTest(position=position):
                out = extract(df, 'C', position, is_hard_flag=False)
                self.assertEqual(out.iloc[0], 0.0)          # valid all-pass code
                self.assertTrue(np.isnan(out.iloc[1]))      # -9999 -> not testable
                self.assertTrue(np.isnan(out.iloc[2]))      # -1    -> not testable
                self.assertTrue(np.isnan(out.iloc[4]))      # NaN stays NaN
        # A real flag at that position is still read, so the guard is not a blanket veto.
        self.assertEqual(extract(df, 'C', 1, is_hard_flag=False).iloc[3], 1.0)

    def test_scalar_zero_code_is_flag_zero(self):
        # Finding L25: a raw code of 0 (no flag raised) must extract as flag 0.
        # It used to become NaN, because the float->string round-trip made it
        # '0.0' and the digit at the requested position was the '.'.
        from diive.preprocessing.qaqc.eddyproflags import (
            flag_steadiness_horizontal_wind_eddypro_test, flags_vm97_eddypro_fluxnetfile_tests)
        df = pd.DataFrame({'VM97_NSHW_HF': [80.0, 81.0, 0.0, np.nan]}, index=self._idx(4))
        flag = flag_steadiness_horizontal_wind_eddypro_test(df=df, flux='FC')
        # 80 -> passed (0), 81 -> hard fail (2), 0 -> passed (0), missing -> NaN.
        self.assertEqual(list(flag[:3]), [0.0, 2.0, 0.0])
        self.assertTrue(np.isnan(flag.iloc[3]))

        # Same for the multi-digit VM97 column, at a position beyond the first.
        vm97 = pd.DataFrame({'CO2_VM97_TEST': [800000000.0, 810000000.0, 0.0]},
                            index=self._idx(3))
        flags = flags_vm97_eddypro_fluxnetfile_tests(df=vm97, flux='FC', fluxbasevar='CO2',
                                                     spikes=True, dropout=True)
        self.assertEqual(list(flags['FLAG_FC_CO2_VM97_SPIKE_HF_TEST']), [0.0, 2.0, 0.0])
        self.assertEqual(list(flags['FLAG_FC_CO2_VM97_DROPOUT_TEST']), [0.0, 0.0, 0.0])


class TestMeasurementsRegistry(unittest.TestCase):
    """The measurement/correction registry: which measurement is a column, and
    which corrections are physically meaningful for it.

    Pure lookup tables and one string heuristic -- the kind of code that
    regresses silently, since a wrong answer still returns a plausible value.
    """

    def test_measurements_table_is_well_formed(self):
        from diive.qaqc import MEASUREMENTS, Measurement
        codes = [m.code for m in MEASUREMENTS]
        self.assertEqual(len(codes), len(set(codes)), "measurement codes must be unique")
        for m in MEASUREMENTS:
            with self.subTest(code=m.code):
                self.assertIsInstance(m, Measurement)
                self.assertTrue(m.code and m.description)
                self.assertEqual(m.code, m.code.upper())

    def test_corrections_table_is_well_formed(self):
        from diive.qaqc import CORRECTIONS, CorrectionSpec
        keys = [c.key for c in CORRECTIONS]
        self.assertEqual(len(keys), len(set(keys)), "correction keys must be unique")
        for c in CORRECTIONS:
            with self.subTest(key=c.key):
                self.assertIsInstance(c, CorrectionSpec)
                self.assertTrue(c.key and c.label and c.description)
                self.assertIsInstance(c.needs_coords, bool)
        # Only the day/night-dependent correction needs site coordinates.
        needs = {c.key for c in CORRECTIONS if c.needs_coords}
        self.assertEqual(needs, {'radiation_zero_offset'})

    def test_measurement_label(self):
        from diive.qaqc import measurement_label
        self.assertEqual(measurement_label('TA'), 'TA - air temperature')
        self.assertEqual(measurement_label('SW'), 'SW - shortwave radiation')
        # An unknown code falls back to the code itself rather than raising.
        self.assertEqual(measurement_label('NOPE'), 'NOPE')

    def test_correction_spec_lookup(self):
        from diive.qaqc import correction_spec
        spec = correction_spec('radiation_zero_offset')
        self.assertIsNotNone(spec)
        self.assertEqual(spec.key, 'radiation_zero_offset')
        self.assertTrue(spec.needs_coords)
        self.assertIsNone(correction_spec('no_such_key'))

    def test_corrections_for_measurement_puts_specific_before_generic(self):
        from diive.qaqc import corrections_for_measurement
        generic = ['setto_max', 'setto_min', 'setto_value', 'set_exact_to_missing']

        # Radiation measurements additionally get the zero-offset correction,
        # and it comes first.
        for code in ('SW', 'PPFD'):
            with self.subTest(code=code):
                keys = corrections_for_measurement(code)
                self.assertEqual(keys[0], 'radiation_zero_offset')
                self.assertEqual(keys[1:], generic)

        # RH gets its own offset correction, not the radiation one.
        rh = corrections_for_measurement('RH')
        self.assertEqual(rh[0], 'relativehumidity_offset')
        self.assertNotIn('radiation_zero_offset', rh)

        # A measurement with no specific physics, an unknown code, and None all
        # yield the generic corrections only.
        for code in ('TA', 'UNKNOWN_CODE', None):
            with self.subTest(code=code):
                self.assertEqual(corrections_for_measurement(code), generic)

    def test_corrections_for_measurement_returns_known_keys_only(self):
        from diive.qaqc import CORRECTIONS, MEASUREMENTS, corrections_for_measurement
        known = {c.key for c in CORRECTIONS}
        for m in MEASUREMENTS:
            with self.subTest(code=m.code):
                keys = corrections_for_measurement(m.code)
                self.assertTrue(set(keys) <= known)
                self.assertEqual(len(keys), len(set(keys)), "no duplicates")

    def test_detect_measurement_matches_naming_conventions(self):
        from diive.qaqc import detect_measurement
        cases = {
            'SW_IN_T1_2_1': 'SW',
            'SW_OUT': 'SW',
            'RH_T1_2_1': 'RH',
            'TA_T1_2_1': 'TA',
            'Tair_f': 'TA',
            'VPD_f': 'VPD',
            'LW_IN': 'LW',
            'PPFD_IN_T1_1_1': 'PPFD',
            'PA_T1_2_1': 'PA',
            'PREC_TOT_T1_2_1': 'PREC',
            'WS_T1_2_1': 'WS',
            'WD_T1_2_1': 'WD',
            'TS_GF1_0.05_1': 'TS',
            'G_GF1_0.03_1': 'G',
        }
        for varname, expected in cases.items():
            with self.subTest(varname=varname):
                self.assertEqual(detect_measurement(varname), expected)

    def test_detect_measurement_prefers_the_more_specific_prefix(self):
        from diive.qaqc import detect_measurement
        # SWC (soil water content) must win over SW (shortwave radiation),
        # otherwise a soil probe would be offered the radiation zero-offset.
        self.assertEqual(detect_measurement('SWC_GF1_0.05_1'), 'SWC')
        self.assertEqual(detect_measurement('SWC'), 'SWC')
        self.assertEqual(detect_measurement('SW_IN'), 'SW')

    def test_detect_measurement_returns_none_when_nothing_matches(self):
        from diive.qaqc import detect_measurement
        for varname in ('FC', 'NEE_CUT_REF_f', 'H', 'LE', ''):
            with self.subTest(varname=varname):
                self.assertIsNone(detect_measurement(varname))
        # Non-string input is tolerated rather than raising.
        self.assertIsNone(detect_measurement(None))
        self.assertIsNone(detect_measurement(42))

    def test_detected_measurements_are_in_the_registry(self):
        # Every code detect_measurement can return must be a known measurement,
        # or the label/correction lookups downstream would silently degrade.
        from diive.qaqc import MEASUREMENTS, detect_measurement
        known = {m.code for m in MEASUREMENTS}
        samples = ['SWC_1', 'SW_IN', 'PPFD_IN', 'LW_OUT', 'RH_1', 'VPD_1',
                   'TA_1', 'Tair_f', 'TS_1', 'PREC_1', 'PA_1', 'WS_1', 'WD_1',
                   'G_1']
        for varname in samples:
            with self.subTest(varname=varname):
                code = detect_measurement(varname)
                self.assertIn(code, known)


if __name__ == '__main__':
    unittest.main()
