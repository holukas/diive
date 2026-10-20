"""
TEST_GUI_TAB_FIXES: GUI TESTS FOR SCREENING, VARIABLE SUBSET AND RECORD SELECTION
=================================================================================

Offscreen tests for three GUI fixes: Stepwise screening refuses a day/night
step without a site (L319), reopening a project keeps the variable subset it
was saved with (L336), and Select records by condition offers only numeric
condition variables (L355).

Run: pytest tests/test_gui_tab_fixes.py -v

Part of the diive library: https://github.com/holukas/diive
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
import threading
import time
import traceback

import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import diive as dv

_SITE = dict(name="X", latitude=47.29, longitude=7.73, elevation=0.0, utc_offset=1)


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or a worker thread raised (PySide6
    hands those to the excepthooks and returns normally)."""
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


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """No message box may block a headless test (e.g. a prompt on close)."""
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)


@pytest.fixture
def site_snapshot():
    """Restore the app-wide site after the test."""
    from diive.gui import site
    before = site.manager.as_dict()
    yield site.manager
    site.manager.load_dict(before)
    site.manager.configured = before.get("configured", False)


@pytest.fixture(scope="module")
def example_year():
    df = dv.load_exampledata_parquet()
    return dv.times.keep_daterange(df, "2021-01-01", "2021-12-31 23:30")


def _destroy_window(win):
    """Really destroy a MainWindow (see tests/test_gui.py for why)."""
    QApplication.processEvents()
    win.close()
    shiboken6.delete(win)
    win._tabs.clear()
    win._menu_tab_list.clear()
    win._pinned.clear()
    gc.collect()
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def window(app, monkeypatch, example_year):
    import diive
    monkeypatch.setattr(diive, "load_exampledata_parquet", lambda: example_year.copy())
    # Answer the unsaved-changes prompt without showing it.
    from diive.gui import app as app_mod
    monkeypatch.setattr(app_mod, "ask_save_changes", lambda *a, **k: "discard",
                        raising=False)
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    yield win
    _destroy_window(win)
    from diive.gui import events as _events
    _events.manager.events.clear()
    _events.manager.visible = True
    _events.manager.categories = dict(_events.DEFAULT_CATEGORIES)


def _wait_for_chain(tab, timeout: float = 120.0) -> None:
    """Pump the event loop until the screening tab's chain run has finished."""
    deadline = time.monotonic() + timeout
    while tab._running:
        if time.monotonic() > deadline:
            raise AssertionError("chain still running")
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()


def _noisy():
    df = dv.variables.generate_noisy_timeseries(
        start_date="2024-01-01", periods=48 * 10, freq="30min", trend_slope=0.0,
        seasonal_strength=5, noise_level=1, outlier_fraction=0.05)
    df.index.name = "TIMESTAMP_END"
    return df


def _zscore(daynight: bool) -> dict:
    return {"method": "flag_outliers_zscore_test",
            "kwargs": {"thres_zscore": 4.0, "separate_day_night": daynight,
                       "repeat": True},
            "enabled": True}


# --- L319: Stepwise screening needs the site for a day/night step ---------------

def test_stepwise_daynight_step_needs_the_site(app, site_snapshot):
    from diive.gui.tabs._screening_base import _NO_SITE_MESSAGE
    from diive.gui.tabs.stepwise import StepwiseScreeningTab

    site_snapshot.configured = False
    df = _noisy()
    tab = StepwiseScreeningTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._select("observed_value")
    tab._steps = [_zscore(daynight=False), _zscore(daynight=True)]
    run_id = tab._run_id

    tab.run_outliers_btn.click()
    assert tab.status.text() == _NO_SITE_MESSAGE
    assert not tab._running and tab._run_id == run_id
    assert tab._payload is None

    # A disabled day/night step does not need the site.
    tab._steps[1] = {**tab._steps[1], "enabled": False}
    tab.run_outliers_btn.click()
    _wait_for_chain(tab)
    assert tab._payload is not None
    assert len(tab._payload["removed"]) == 2

    # Once the site is set, the day/night step runs.
    tab._steps[1] = {**tab._steps[1], "enabled": True}
    site_snapshot.update(**_SITE)
    QApplication.processEvents()
    tab.run_outliers_btn.click()
    _wait_for_chain(tab)
    assert tab._payload is not None
    assert tab.status.text() != _NO_SITE_MESSAGE
    assert tab._committed_steps[1]["kwargs"]["separate_day_night"] is True


def test_stepwise_step_without_daynight_runs_without_site(app, site_snapshot):
    from diive.gui.tabs.stepwise import StepwiseScreeningTab

    site_snapshot.configured = False
    tab = StepwiseScreeningTab()
    tab.widget()
    tab.on_data_loaded(_noisy())
    tab._select("observed_value")
    tab._steps = [_zscore(daynight=False)]
    tab.run_outliers_btn.click()
    _wait_for_chain(tab)
    assert tab._payload is not None
    assert "removed overall" in tab.status.text()


def test_step_needs_site_rules():
    from diive.gui.tabs._screening_base import _step_needs_site

    # Hampel splits day and night by default, as the library does.
    assert _step_needs_site({"method": "flag_outliers_hampel_test", "kwargs": {}})
    assert not _step_needs_site({"method": "flag_outliers_zscore_test", "kwargs": {}})
    assert not _step_needs_site({"method": "flag_outliers_hampel_test",
                                 "kwargs": {"separate_day_night": False}})
    # Per-period limits turn the split on in the library.
    assert _step_needs_site({"method": "flag_outliers_abslim_test",
                             "kwargs": {"minval": 0, "minval_daytime": 1}})
    assert _step_needs_site({"method": "flag_outliers_trim_low_test",
                             "kwargs": {"trim_nighttime": True}})
    assert not _step_needs_site({"method": "flag_missingvals_test", "kwargs": {}})


def test_meteo_screening_daynight_step_needs_the_site(app, site_snapshot):
    """The database variant shares the guard."""
    from diive.gui.tabs._screening_base import _NO_SITE_MESSAGE
    from diive.gui.tabs.meteo_screening import MeteoScreeningTab

    site_snapshot.configured = False
    tab = MeteoScreeningTab()
    tab.widget()
    tab._df, tab._var = _noisy(), "observed_value"
    tab._steps = [_zscore(daynight=True)]
    tab._run()
    assert tab.status.text() == _NO_SITE_MESSAGE
    assert not tab._running


# --- L336: the saved subset decides what a reopened project shows --------------

def test_reset_subset_then_reopen_keeps_all_variables(window, tmp_path):
    n_all = window._full_data.shape[1]
    names = [str(c) for c in window._full_data.columns]
    window._open_menu_tab("Select variables")
    sel = window._menu_tab_list[-1]
    for n in names[:3]:
        sel._select(n)
    sel._confirm()
    QApplication.processEvents()
    assert window._data.shape[1] == 3

    window._reset_var_subset()
    QApplication.processEvents()
    assert window._var_subset is None
    assert sel.picker.selected_names() == []    # the reset clears the picks too

    folder = tmp_path / "Reset.diive"
    assert window._write_project(folder, "Reset")
    assert window._load_project_folder(folder)
    QApplication.processEvents()

    assert window._var_subset is None
    assert window._data.shape[1] == n_all
    restored = next(t for t in window._menu_tab_list
                    if t._menu_label == "Select variables")
    assert restored.picker.selected_names() == []
    assert "Applied" not in restored.status.text()


def test_reopen_keeps_saved_subset(window, tmp_path):
    names = [str(c) for c in window._full_data.columns]
    window._open_menu_tab("Select variables")
    sel = window._menu_tab_list[-1]
    sel._select(names[4])
    sel._select(names[2])
    sel._confirm()
    QApplication.processEvents()

    folder = tmp_path / "Subset.diive"
    assert window._write_project(folder, "Subset")
    assert window._load_project_folder(folder)
    QApplication.processEvents()
    assert window._var_subset == [names[4], names[2]]
    assert list(window._data.columns) == [names[4], names[2]]


# --- L355: only numeric condition variables ------------------------------------

def test_select_records_offers_only_numeric_conditions(app, example_year):
    from diive.gui.tabs.select_records import SelectRecordsTab

    df = example_year.copy()
    df["TS_END"] = df.index.to_series()        # datetime, kept as datetime
    df["IS_DAY"] = df["Tair_f"] > 10            # boolean
    tab = SelectRecordsTab()
    tab.widget()
    tab.on_data_loaded(df, set())
    QApplication.processEvents()

    offered = [tab.cond_combo.itemText(i) for i in range(tab.cond_combo.count())]
    assert "TS_END" not in offered and "IS_DAY" not in offered
    assert "Tair_f" in offered

    # A datetime target still works with a numeric condition.
    tab._select("TS_END")
    tab.cond_combo.setCurrentText("Tair_f")
    QApplication.processEvents()
    assert tab.lower.value() == pytest.approx(float(df["Tair_f"].min()), abs=1e-4)
    tab.lower.setValue(15.0)
    tab.upper.setValue(20.0)
    tab._apply_op()
    QApplication.processEvents()
    assert len(tab._steps) == 1
    expected = df["Tair_f"].between(15.0, 20.0)
    assert int(tab._keep_mask.sum()) == int(expected.sum())

    # Nothing numeric to pick: the operation is refused with a message.
    tab.on_data_loaded(df[["TS_END", "IS_DAY"]], set())
    tab._select("TS_END")
    assert tab.cond_combo.count() == 0
    tab._apply_op()
    assert tab.status.text() == "Pick a numeric condition variable."
    assert tab._steps == []


def test_restored_datetime_condition_is_ignored(app, example_year):
    """A project saved with a condition column that is now a datetime column
    must not reach the limits code."""
    from diive.gui.tabs.select_records import SelectRecordsTab

    df = example_year[["Tair_f", "VPD_f"]].copy()
    df["TS_END"] = df.index.to_series()
    tab = SelectRecordsTab()
    tab.widget()
    tab.on_data_loaded(df, set())
    tab.cond_combo.setCurrentText("VPD_f")
    lo = tab.lower.value()
    tab.restore_state({"target": "Tair_f", "controls": {"cond": "TS_END"}})
    QApplication.processEvents()
    assert tab.cond_combo.currentText() == "VPD_f"
    assert tab.lower.value() == lo
