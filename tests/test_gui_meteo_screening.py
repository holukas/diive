"""
TEST_GUI_METEO_SCREENING: END-TO-END TESTS OF THE METEO SCREENING (DATABASE) TAB
================================================================================

Offscreen tests of the GUI tab "Meteo screening (database)" on downloads shaped
like the Database explorer's ``data_detailed``: one field on a TIMESTAMP_END
index without freq, all database tags as columns. Each download goes through the
real hand-off, ``db.manager.request_screening(payload)`` into a ``MainWindow``,
then through the tab's outlier chain, corrections and resampling.

Scenarios: single 10-min, single 30-min, 10-min then 1-min, 10-min then 1-min
with the switch inside a half hour, 10-min then 15-min, three resolutions, and
downloads the library rejects. Results are checked against references computed
here from the raw records, not against the tab's own intermediate series.

Run: pytest tests/test_gui_meteo_screening.py -v

Part of the diive library: https://github.com/holukas/diive
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import inspect
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

import diive as dv
from diive.core.io.db.influx.common import TAGS

TAB_LABEL = "Meteo screening (database)"
FIELD = "TA_T1_2_1"
START = pd.Timestamp("2021-07-01 00:00")  # inside the example year the window loads
P = pd.Timedelta("30min")
LAT, LON, UTC = 47.29, 7.73, 1
ABSLIM_SPIKE = 80.0   # caught by absolute limits
HAMPEL_SPIKE = 8.0    # added to the value; caught by Hampel with differencing
MINCOUNTS = 0.9       # the tab's default

#: The chain every scenario runs: absolute limits for the 80-degree spikes, then
#: Hampel with double differencing for the +8 spikes. n_sigma=24 sits between the
#: spike (~33 sigma of the differenced noise) and its two neighbours (~16 sigma).
STEPS = [
    {"method": "flag_outliers_abslim_test",
     "kwargs": {"minval": -20.0, "maxval": 50.0, "separate_day_night": False},
     "enabled": True},
    {"method": "flag_outliers_hampel_test",
     "kwargs": {"window_length": 49, "n_sigma": 24.0, "use_differencing": True,
                "separate_day_night": False, "repeat": True},
     "enabled": True},
]


# --- fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt slot or a worker thread raised (PySide6 swallows both).

    Copied from tests/test_gui.py: a slot exception goes to ``sys.excepthook``
    and the emitting call returns normally, and a worker-thread exception only
    prints, so without this a crashed load or run would look like a no-op.
    """
    captured = []

    def _hook(etype, value, tb):
        captured.append("".join(traceback.format_exception(etype, value, tb)))

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        captured.append(
            f"in thread {getattr(args.thread, 'name', '?')!r}:\n"
            + "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback)))

    prev_hook, prev_thread_hook = sys.excepthook, threading.excepthook
    sys.excepthook, threading.excepthook = _hook, _thread_hook
    try:
        yield captured
    finally:
        sys.excepthook, threading.excepthook = prev_hook, prev_thread_hook
    if captured:
        pytest.fail(
            f"{len(captured)} exception(s) raised inside a Qt slot or worker "
            "thread and swallowed by PySide6:\n\n" + "\n".join(captured),
            pytrace=False)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(app):
    """One MainWindow for the module, on one year of example data (30-min MIDDLE).

    The tab is single-instance and every scenario re-stages it, as a user sending
    one field after another from the Database explorer would.
    """
    from diive.gui import events as _events
    from diive.gui import site
    import diive

    example = dv.times.keep_daterange(
        dv.load_exampledata_parquet(), "2021-01-01", "2021-12-31 23:30")
    site_before = site.manager.as_dict()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(diive, "load_exampledata_parquet", lambda: example.copy())
        site.manager.update(name="CH-XYZ", latitude=LAT, longitude=LON,
                            elevation=500.0, utc_offset=UTC)
        from diive.gui.app import MainWindow
        win = MainWindow()
        win.show()
        app.processEvents()
        win._wait_for_io()
        app.processEvents()
        win._test_baseline_threads = set(threading.enumerate())
        yield win
        QApplication.processEvents()
        win.close()
        shiboken6.delete(win)
        win._tabs.clear()
        win._menu_tab_list.clear()
        win._pinned.clear()
        gc.collect()
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    site.manager.load_dict(site_before)
    site.manager.configured = site_before.get("configured", False)
    _events.manager.events.clear()


# --- download frames and references --------------------------------------------

def _tagvalues(field, freq, units):
    return dict(site="CH-XYZ", varname=field, units=units, raw_varname=f"{field}_raw",
                raw_units=units, hpos="T1", vpos="2", repl="1", data_raw_freq=freq,
                freq=freq, filegroup="meteo", config_filetype="TEST",
                data_version="raw", gain="1", offset="0")


def _download(segments, field=FIELD, units="degC", value_fn=None,
              abslim_spikes=(), hampel_spikes=()):
    """A download as the Database explorer hands it over.

    segments: [(freq, span), ...] in time order; each segment continues one step
    after the previous segment's last END. Returns (frame, duration), duration
    being each record's resolution (the time it covers).
    """
    parts, t = [], START
    for freq, span in segments:
        idx = pd.date_range(t + pd.Timedelta(freq), t + pd.Timedelta(span), freq=freq)
        part = pd.DataFrame(_tagvalues(field, freq, units), index=idx)
        part["_duration"] = pd.Timedelta(freq)
        parts.append(part)
        t = idx[-1]
    df = pd.concat(parts)
    if value_fn is None:
        hour = df.index.hour.to_numpy() + df.index.minute.to_numpy() / 60
        rng = np.random.default_rng(42)
        df[field] = 15 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.2, len(df))
    else:
        df[field] = value_fn(df.index)
    for ts in abslim_spikes:
        assert pd.Timestamp(ts) in df.index, ts
        df.loc[pd.Timestamp(ts), field] = ABSLIM_SPIKE
    for ts in hampel_spikes:
        assert pd.Timestamp(ts) in df.index, ts
        df.loc[pd.Timestamp(ts), field] += HAMPEL_SPIKE
    df.index = pd.DatetimeIndex(df.index.to_numpy(), name="TIMESTAMP_END")  # no freq
    duration = df.pop("_duration")
    return df[TAGS + [field]], duration


def _grid_step(ends: pd.DatetimeIndex, duration: pd.Series) -> pd.Timedelta:
    """Step of a grid that holds every record at its own END: the greatest common
    divisor of the resolutions and of the offsets between END timestamps."""
    ns = np.concatenate([duration.dt.total_seconds().to_numpy() * 1e9,
                         (ends - ends[0]).total_seconds().to_numpy() * 1e9]).astype(np.int64)
    return pd.Timedelta(int(np.gcd.reduce(ns)), unit="ns")


def _reference(values: pd.Series, duration: pd.Series, agg: str, mincounts: float,
               step: pd.Timedelta) -> pd.Series:
    """Time-weighted 30-min aggregate of the kept raw records, per period END T.

    A period collects the records whose END is in (T - 30min, T]. The mean weights
    each record by the time it covers; the sum adds each record once. A period is
    kept if its records cover at least int(slots * mincounts) slots of the grid
    (at least one slot if that minimum is below three), the rule the Resample
    page documents.
    """
    target = values.index.ceil(P)
    w = duration.reindex(values.index).dt.total_seconds()
    covered_slots = w.groupby(target).sum() / step.total_seconds()
    if agg == "mean":
        out = (values * w).groupby(target).sum() / w.groupby(target).sum()
    else:
        out = values.groupby(target).sum()
    slots = int(round(P / step))
    minslots = int(slots * mincounts)
    minslots = 1 if minslots < 3 else minslots
    return out.where(covered_slots >= minslots - 1e-9)


# --- driving the tab ---------------------------------------------------------------

def _meteo_tab(win):
    """The (single-instance) Meteo screening tab, or None before the first hand-off."""
    return next((t for t in win._menu_tab_list
                 if getattr(t, "_menu_label", None) == TAB_LABEL), None)


def _busy(tab, win) -> bool:
    from diive.gui.widgets.worker import LatestRunner, WorkerRunner
    if getattr(tab, "_running", False):
        return True
    for v in vars(tab).values():
        if isinstance(v, LatestRunner) and v.is_busy:
            return True
        if isinstance(v, WorkerRunner) and v.is_running:
            return True
    baseline = getattr(win, "_test_baseline_threads", set())
    for t in threading.enumerate():
        if t in baseline or t is threading.main_thread() or not t.is_alive():
            continue
        if any(s in t.name for s in ("Executor", "QueueFeeder", "pydevd")):
            continue
        return True
    return False


def _settle(win, tab=None, timeout: float = 240.0) -> None:
    """Pump events until the tab's background work has run and been delivered.

    The tab may build the library object, run the chain and resample on worker
    threads; a result counts as delivered once no runner or worker thread is busy
    for a few consecutive event-loop passes.
    """
    deadline = time.monotonic() + timeout
    calm = 0
    while calm < 5:
        QApplication.processEvents()
        if tab is not None and _busy(tab, win):
            calm = 0
            if time.monotonic() > deadline:
                raise AssertionError(f"meteo tab still busy after {timeout:.0f} s")
            time.sleep(0.005)
        else:
            calm += 1
    QApplication.processEvents()


def _stage(win, frame, field=FIELD):
    from diive.gui import db
    db.manager.request_screening({
        "data_detailed": {field: frame}, "field": field, "measurement": "TA",
        "bucket": "ch-xyz_raw", "data_version": "raw", "utc_offset": UTC})
    tab = _meteo_tab(win)
    _settle(win, tab)
    return tab


def _disable_corrections(tab):
    if tab is None:
        return
    for row in tab.corrections_panel._rows.values():
        row.enable.setChecked(False)


def _run_chain(win, tab, steps):
    tab._steps = [dict(s) for s in steps]
    tab._apply_chain_change()
    tab.run_outliers_btn.click()
    _settle(win, tab)


def _emitted(win, tab, agg, mincounts=MINCOUNTS, freq="30min"):
    tab._freq.setCurrentText(freq)
    tab._agg.setCurrentText(agg)
    tab._mincounts.setValue(mincounts)
    _settle(win, tab)
    rdf = tab._result_df
    return None if rdf is None else rdf.copy()


def _mscr(tab):
    """The StepwiseMeteoScreeningDb the tab runs on, wherever the tab keeps it:
    the last run's instance if the run result holds one, else the loaded one."""
    from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb
    payload = getattr(tab, "_payload", None)
    if isinstance(payload, dict):
        for w in payload.values():
            if isinstance(w, StepwiseMeteoScreeningDb):
                return w
    for v in vars(tab).values():
        if isinstance(v, StepwiseMeteoScreeningDb):
            return v
        if isinstance(v, dict):
            for w in v.values():
                if isinstance(w, StepwiseMeteoScreeningDb):
                    return w
    return None


def _qcf(tab, field=FIELD):
    m = _mscr(tab)
    if m is not None and field in m.outlier_detection_qcf:
        return m.outlier_detection_qcf[field]
    payload = getattr(tab, "_payload", None)
    return payload["qcf"] if payload else None


def _test_flags(tab, field=FIELD) -> pd.DataFrame:
    m = _mscr(tab)
    if m is not None and field in m.outlier_detection:
        return m.outlier_detection[field].flags
    return tab._payload["detector"].flags


def _slot_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """END timestamps of the screening grid's slots (the library screens on MIDDLE)."""
    if index.name == "TIMESTAMP_END":
        return index
    step = pd.Timedelta(index.freq) if index.freq is not None else index.to_series().diff().min()
    return index + step / 2


def _norm(text: str) -> str:
    """Drop thousands separators so '4,752 records' reads as '4752 records'."""
    return re.sub(r"(?<=\d)[,'   ](?=\d{3}\b)", "", text)


def _res_token(freq: str) -> str:
    n = pd.Timedelta(freq) // pd.Timedelta("1min")
    return rf"(?<![\d.]){n}\s?min"


def _report_counts(text: str, period: str) -> dict:
    """'Potential/Measured/Retained/Rejected' counts of one section of the QCF report."""
    m = re.search(rf"--- {period} ---(.*?)(?:\n---|\n=)", text, flags=re.S)
    assert m, f"no {period} section in the report:\n{text}"
    out = {}
    for key in ("Potential", "Measured", "Retained", "Rejected"):
        k = re.search(rf"{key}[^:]*:\s*(\d+)", m.group(1))
        assert k, f"no '{key}' count in the {period} section:\n{m.group(1)}"
        out[key] = int(k.group(1))
    return out


# --- scenarios -------------------------------------------------------------------

@dataclass
class Scenario:
    segments: list
    abslim_spikes: list
    hampel_spikes: list
    coarse_hampel_spikes: list = dc_field(default_factory=list)


SCENARIOS = {
    "single_10min": Scenario([("10min", "2D")], ["2021-07-01 03:00"], ["2021-07-01 14:00"]),
    "single_30min": Scenario([("30min", "3D")], ["2021-07-01 03:00"], ["2021-07-02 14:00"]),
    "mixed_10min_1min": Scenario(
        [("10min", "2D"), ("1min", "2D")],
        ["2021-07-01 03:00", "2021-07-03 05:13"],
        ["2021-07-01 14:00", "2021-07-03 14:07"], ["2021-07-01 14:00"]),
    # 10-min records up to END 07-03 12:10, then 1-min: the half hour (12:00, 12:30]
    # holds one 10-min record and twenty 1-min records.
    "mixed_switch_in_halfhour": Scenario(
        [("10min", "2D 12:10:00"), ("1min", "1D 11:50:00")],
        ["2021-07-01 03:00", "2021-07-04 05:13"],
        ["2021-07-02 14:00", "2021-07-04 14:07"], ["2021-07-02 14:00"]),
    # 15 min is not a multiple of 10 min: the grid must be 5 min.
    "mixed_10min_15min": Scenario(
        [("10min", "2D"), ("15min", "2D")],
        ["2021-07-01 03:00", "2021-07-03 05:15"],
        ["2021-07-01 14:00", "2021-07-04 14:15"], ["2021-07-01 14:00"]),
    "three_resolutions": Scenario(
        [("10min", "1D"), ("5min", "12h"), ("1min", "144min")],
        ["2021-07-01 03:00", "2021-07-02 05:00", "2021-07-02 13:13"],
        ["2021-07-01 14:00"], ["2021-07-01 14:00"]),
}

_CACHE: dict = {}


@dataclass
class Result:
    raw: pd.DataFrame
    duration: pd.Series
    planted: pd.DatetimeIndex
    status_loaded: str
    status_run: str
    qcf_label: str
    report: str
    mscr: object
    qcf: object
    flags: pd.DataFrame
    emitted: dict            # {(agg, mincounts): DataFrame or None}
    swallowed: list


def _screened(win, captured, name) -> Result:
    """Stage, screen and resample scenario *name* once; later tests reuse it."""
    if name in _CACHE:
        res = _CACHE[name]
        if isinstance(res, BaseException):
            raise res
        assert res.swallowed == [], "exceptions swallowed while screening:\n" + "\n".join(res.swallowed)
        return res
    sc = SCENARIOS[name]
    n0 = len(captured)
    try:
        raw, duration = _download(sc.segments, abslim_spikes=sc.abslim_spikes,
                                  hampel_spikes=sc.hampel_spikes)
        _disable_corrections(_meteo_tab(win))
        tab = _stage(win, raw)
        _disable_corrections(tab)
        status_loaded = tab.status.text()
        _run_chain(win, tab, STEPS)
        status_run = tab.status.text()
        emitted = {}
        for key in (("mean", MINCOUNTS), ("sum", MINCOUNTS), ("mean", 0.0)):
            emitted[key] = _emitted(win, tab, *key)
        _emitted(win, tab, "mean", MINCOUNTS)  # leave the default settings
        qcf = _qcf(tab)
        res = Result(raw=raw, duration=duration,
                     planted=pd.DatetimeIndex(sc.abslim_spikes + sc.hampel_spikes),
                     status_loaded=status_loaded, status_run=status_run,
                     qcf_label=tab.qcf_label.text(), report=tab.report_text.toPlainText(),
                     mscr=_mscr(tab), qcf=qcf,
                     flags=_test_flags(tab) if qcf is not None else None,
                     emitted=emitted, swallowed=list(captured[n0:]))
    except BaseException as err:
        _CACHE[name] = err
        raise
    _CACHE[name] = res
    return res


ALL = list(SCENARIOS)


# --- tests: loading --------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_status_reports_records_and_resolutions(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    status = _norm(res.status_loaded)
    assert re.search(rf"(?<!\d){len(res.raw)}(?!\d)\s*records", status), \
        f"status does not report the {len(res.raw)} records: {res.status_loaded!r}"
    for freq in res.raw["freq"].unique():
        assert re.search(_res_token(freq), status), \
            f"status does not name the {freq} resolution: {res.status_loaded!r}"


@pytest.mark.parametrize("name", ALL)
def test_all_records_kept(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    m = res.mscr
    assert m is not None, "the tab does not run on a StepwiseMeteoScreeningDb"
    assert m.records_count(FIELD) == len(res.raw)
    orig = m.series_hires_orig[FIELD]
    assert int(orig.notna().sum()) == len(res.raw)
    # every raw record sits at its own END with its own value
    at_end = pd.Series(orig.dropna().to_numpy(), index=_slot_ends(orig.index)[orig.notna().to_numpy()])
    assert at_end.index.equals(res.raw.index)
    np.testing.assert_allclose(at_end.to_numpy(), res.raw[FIELD].to_numpy())
    got = sorted(pd.Timedelta(f) for f in m.resolutions(FIELD))
    assert got == sorted(pd.Timedelta(f) for f in res.raw["freq"].unique())


# --- tests: outlier chain and QCF ------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_planted_spikes_and_only_those_rejected(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    assert res.qcf is not None, f"no QCF after the run; status: {res.status_run!r}"
    flag = res.qcf.flagqcf
    rejected = _slot_ends(flag.index)[(flag == 2).to_numpy()]
    assert sorted(rejected) == sorted(res.planted)


@pytest.mark.parametrize("name", [n for n in ALL if SCENARIOS[n].coarse_hampel_spikes])
def test_hampel_with_differencing_catches_coarse_spike(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    cols = [c for c in res.flags.columns if "HAMPEL" in str(c).upper()]
    assert len(cols) == 1, list(res.flags.columns)
    hampel = res.flags[cols[0]]
    hampel.index = _slot_ends(hampel.index)
    for ts in SCENARIOS[name].coarse_hampel_spikes:
        assert hampel.loc[pd.Timestamp(ts)] == 2, f"coarse spike at END {ts} not flagged by Hampel"
    # the coarse records were tested, not passed untested
    coarse = res.raw.index[res.raw["freq"] == res.raw["freq"].iloc[0]]
    tested = hampel.reindex(coarse).notna().mean()
    assert tested > 0.95, f"only {tested:.0%} of the coarse records carry a Hampel flag"


@pytest.mark.parametrize("name", ALL)
def test_qcf_report_counts_records_only(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    n, n_rej = len(res.raw), len(res.planted)
    assert int(res.qcf.flagqcf.notna().sum()) == n
    overall = _report_counts(res.report, "OVERALL")
    assert overall["Potential"] == n
    assert overall["Measured"] == n
    assert overall["Rejected"] == n_rej
    assert overall["Retained"] == n - n_rej
    label = {int(k): int(v) for k, v in re.findall(r"\b([012])\s*:\s*(\d+)", res.qcf_label)}
    assert label, f"no QCF counts in the label: {res.qcf_label!r}"
    assert sum(label.values()) == n, res.qcf_label
    assert label.get(2, 0) == n_rej, res.qcf_label


def test_daynight_at_timestamp_middle_single_30min(window, slot_exceptions):
    """Day/night of a 30-min download follows potential radiation at TIMESTAMP_MIDDLE."""
    res = _screened(window, slot_exceptions, "single_30min")
    middle = pd.DatetimeIndex(res.raw.index - P / 2, freq="30min")
    swinpot = dv.variables.potrad(middle, LAT, LON, UTC)
    expected_day = (swinpot >= 20).to_numpy()

    qcf = res.qcf
    assert qcf.daytime is not None, "QCF has no day/night split (site coordinates not passed?)"
    daytime = qcf.daytime.copy()
    daytime.index = _slot_ends(daytime.index)
    daytime = daytime.reindex(res.raw.index)
    assert daytime.notna().all()
    flips = res.raw.index[(daytime.to_numpy() == 1) != expected_day]
    assert len(flips) == 0, f"{len(flips)} records with the wrong day/night: {list(flips[:6])}"

    pot = qcf.swinpot_data.copy()
    pot.index = _slot_ends(pot.index)
    np.testing.assert_allclose(pot.reindex(res.raw.index).to_numpy(), swinpot.to_numpy(),
                               atol=1e-6)

    day = _report_counts(res.report, "DAYTIME")
    night = _report_counts(res.report, "NIGHTTIME")
    assert day["Potential"] == int(expected_day.sum())
    assert night["Potential"] == int((~expected_day).sum())


# --- tests: resampling and emit ----------------------------------------------------

@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("agg,mincounts", [("mean", MINCOUNTS), ("sum", MINCOUNTS), ("mean", 0.0)])
def test_emitted_matches_time_weighted_reference(window, slot_exceptions, name, agg, mincounts):
    res = _screened(window, slot_exceptions, name)
    out = res.emitted[(agg, mincounts)]
    assert out is not None and not out.empty, f"nothing to emit ({agg}, mincounts {mincounts})"
    got = out.iloc[:, 0].copy()
    got.index = got.index + P / 2  # MIDDLE -> END, checked in test_emitted_timestamps
    kept = res.raw[FIELD].drop(res.planted)
    step = _grid_step(res.raw.index, res.duration)
    ref = _reference(kept, res.duration, agg, mincounts, step)
    got_valid, ref_valid = got.dropna(), ref.dropna()
    missing = ref_valid.index.difference(got_valid.index)
    extra = got_valid.index.difference(ref_valid.index)
    assert len(missing) == 0 and len(extra) == 0, (
        f"periods (END) the reference has but the tab does not: {list(missing[:6])} "
        f"({len(missing)}); the tab has but the reference does not: {list(extra[:6])} ({len(extra)})")
    np.testing.assert_allclose(got_valid.to_numpy(), ref_valid.reindex(got_valid.index).to_numpy(),
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", ["single_10min", "mixed_10min_1min"])
def test_emitted_timestamps_middle_on_dataset_grid(window, slot_exceptions, name):
    res = _screened(window, slot_exceptions, name)
    out = res.emitted[("mean", MINCOUNTS)]
    idx = out.index
    assert idx.name == "TIMESTAMP_MIDDLE"
    assert ((idx - P / 2) == (idx - P / 2).floor(P)).all(), "timestamps are not 30-min MIDDLE"
    assert idx.isin(window._data.index).all(), "emitted timestamps are off the dataset grid"
    first_end = res.raw.index[0].ceil(P)
    assert idx[out.iloc[:, 0].notna().to_numpy()][0] == first_end - P / 2

    # Add it to the working dataset: the values land on the same timestamps. The
    # tab may hold another scenario by now, so this one is staged and run again.
    tab = _stage(window, res.raw)
    _disable_corrections(tab)
    _run_chain(window, tab, STEPS)
    out = _emitted(window, tab, "mean")
    before = set(window._full_data.columns)
    assert tab.add_btn.isEnabled()
    tab.add_btn.click()
    _settle(window, tab)
    new = [c for c in window._full_data.columns if c not in before]
    try:
        assert len(new) == 1, new
        merged = window._full_data[new[0]].reindex(idx)
        np.testing.assert_allclose(merged.to_numpy(), out.iloc[:, 0].to_numpy(), equal_nan=True)
    finally:
        if new:
            window._full_data = window._full_data.drop(columns=new)
            for c in new:
                window._created.discard(c)
            window._apply_range()
            _settle(window, tab)


# --- tests: corrections --------------------------------------------------------------

SW_FIELD = "SW_IN_T1_2_1"


def _sw_values(index):
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    rng = np.random.default_rng(7)
    day = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None)
    return day - 4.0 + rng.normal(0, 0.5, len(index))  # nighttime offset of -4


def test_zero_offset_correction_leaves_empty_slots_empty(window, slot_exceptions):
    raw, duration = _download([("10min", "2D"), ("1min", "2D")], field=SW_FIELD,
                              units="W m-2", value_fn=_sw_values)
    _disable_corrections(_meteo_tab(window))
    tab = _stage(window, raw, field=SW_FIELD)
    try:
        _disable_corrections(tab)
        tab._steps = []
        tab._apply_chain_change()
        rows = tab.corrections_panel._rows
        assert "radiation_zero_offset" in rows, list(rows)
        rows["radiation_zero_offset"].enable.setChecked(True)
        tab.run_corrections_btn.click()
        _settle(window, tab)

        # the corrected series the tab holds (and emits from), not the loaded one
        corrected = tab._corrected
        assert corrected is not None, f"no corrected series; status: {tab.status.text()!r}"
        assert (corrected.dropna() == 0).any(), "the zero-offset correction did not run"
        assert int(corrected.notna().sum()) == len(raw), \
            "the correction wrote values into empty grid slots"
        ends = _slot_ends(corrected.index)[corrected.notna().to_numpy()]
        assert ends.equals(raw.index)

        mean = _emitted(window, tab, "mean")
        total = _emitted(window, tab, "sum")
        assert mean is not None and total is not None
        mean_s, sum_s = mean.iloc[:, 0], total.iloc[:, 0]
        mean_s.index = mean_s.index + P / 2
        sum_s.index = sum_s.index + P / 2
        n_records = raw[SW_FIELD].groupby(raw.index.ceil(P)).count()
        periods = n_records.index
        # every half hour holds all its records, so none may be missing
        assert mean_s.reindex(periods).notna().all()
        # sum = records x mean: 3 in the 10-min part, 30 in the 1-min part
        daytime = mean_s.reindex(periods) > 1
        ratio = (sum_s.reindex(periods) / mean_s.reindex(periods))[daytime]
        np.testing.assert_allclose(ratio.to_numpy(), n_records[daytime].to_numpy(), rtol=1e-9)
    finally:
        _disable_corrections(tab)
        _emitted(window, tab, "mean")


# --- tests: failed loads ---------------------------------------------------------------

def _too_small():
    frame, _ = _download([("10min", "2D")], field="TA_T2_2_1")
    return frame.iloc[:1], "too few"


def _mixed_units():
    frame, _ = _download([("10min", "2D")], field="TA_T2_2_1")
    frame = frame.copy()
    frame.iloc[:10, frame.columns.get_loc("units")] = "K"
    return frame, "units"


@pytest.mark.parametrize("make_bad", [_too_small, _mixed_units], ids=["too_small", "mixed_units"])
def test_failed_load_shows_error_and_drops_previous_field(window, slot_exceptions, make_bad):
    good, _ = _download([("10min", "2D")])
    tab = _stage(window, good)
    assert re.search(rf"{len(good)}\s*records", _norm(tab.status.text())), tab.status.text()
    _run_chain(window, tab, STEPS)
    assert tab._result_df is not None  # the previous field is fully screened

    bad, fragment = make_bad()
    tab = _stage(window, bad, field="TA_T2_2_1")
    status = tab.status.text()
    assert fragment in status.lower(), f"status does not explain the failure: {status!r}"
    assert "TA_T2_2_1" in status, status
    # nothing of the previous field is left to screen, plot or add
    m = _mscr(tab)
    assert m is None or FIELD not in m.fields
    assert getattr(tab, "_var", None) != FIELD
    df = getattr(tab, "_df", None)
    assert df is None or FIELD not in df.columns
    assert tab._result_df is None
    assert not tab.add_btn.isEnabled()
    tab.run_outliers_btn.click()
    _settle(window, tab)
    assert tab._result_df is None
    assert not tab.add_btn.isEnabled()


# --- tests: step picker ------------------------------------------------------------------

def test_absolute_limits_in_step_picker(app):
    from diive.gui.widgets.stepwise_cards import StepEditorDialog
    from diive.gui.widgets.stepwise_method_params import STEP_METHOD_BY_KEY, method_labels
    from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

    key = "flag_outliers_abslim_test"
    assert key in dict(method_labels())
    dlg = StepEditorDialog(None)
    try:
        assert key in [dlg.method.itemData(i) for i in range(dlg.method.count())]
    finally:
        dlg.deleteLater()

    step = {"method": key, "kwargs": {
        "minval": -10.0, "maxval": 40.0, "separate_day_night": True,
        "minval_daytime": -5.0, "maxval_daytime": 45.0,
        "minval_nighttime": -12.0, "maxval_nighttime": 30.0}}
    dlg = StepEditorDialog(None, step)
    try:
        out = dlg.step()
    finally:
        dlg.deleteLater()
    assert out["method"] == key
    for k, v in step["kwargs"].items():
        assert out["kwargs"][k] == pytest.approx(v), k

    # Per-period limits are seeded from the global ones when the split is turned on.
    w = STEP_METHOD_BY_KEY[key]()
    w.load({"minval": -7.0, "maxval": 33.0, "separate_day_night": True})
    kw = w.step()["kwargs"]
    assert kw["minval_daytime"] == pytest.approx(-7.0)
    assert kw["minval_nighttime"] == pytest.approx(-7.0)
    assert kw["maxval_daytime"] == pytest.approx(33.0)
    assert kw["maxval_nighttime"] == pytest.approx(33.0)

    w.load({"minval": -7.0, "maxval": 33.0, "separate_day_night": False})
    kw = w.step()["kwargs"]
    assert kw["minval"] == pytest.approx(-7.0) and kw["maxval"] == pytest.approx(33.0)
    assert kw["separate_day_night"] is False
    assert all(kw.get(k) is None for k in ("minval_daytime", "maxval_daytime",
                                           "minval_nighttime", "maxval_nighttime"))
    # the step dispatches onto the library method unchanged
    params = inspect.signature(StepwiseMeteoScreeningDb.flag_outliers_abslim_test).parameters
    assert set(out["kwargs"]) <= set(params)
    w.deleteLater()
