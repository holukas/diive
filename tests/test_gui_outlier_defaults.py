"""
TEST_GUI_OUTLIER_DEFAULTS: repeat defaults of the GUI outlier tabs
==================================================================

Offscreen tests for two outlier tab defaults: the Local Outlier Factor tab runs
one pass unless Repeat is ticked (L317), and the Absolute limits tab with
day/night separation and its default Repeat finishes (L316).

Run: pytest tests/test_gui_outlier_defaults.py -v

Part of the diive library: https://github.com/holukas/diive
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import threading
import traceback

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt slot or worker thread raised (PySide6 swallows it).

    Same pattern as the fixture in tests/test_gui.py.
    """
    captured = []

    def _hook(etype, value, tb):
        captured.append("".join(traceback.format_exception(etype, value, tb)))

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        captured.append("".join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback)))

    prev_hook, prev_thread_hook = sys.excepthook, threading.excepthook
    sys.excepthook, threading.excepthook = _hook, _thread_hook
    try:
        yield captured
    finally:
        sys.excepthook, threading.excepthook = prev_hook, prev_thread_hook
    if captured:
        pytest.fail("exception(s) raised inside a Qt slot or worker thread:\n\n"
                    + "\n".join(captured), pytrace=False)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _diel_frame(periods: int = 48 * 10) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=periods, freq="30min")
    hours = idx.hour + idx.minute / 60
    rng = np.random.default_rng(0)
    ta = 12 * np.sin((hours - 6) / 24 * 2 * np.pi) + 8 + rng.normal(0, 1.5, periods)
    return pd.DataFrame({"TA": ta}, index=idx)


def _make_tab(cls, df):
    tab = cls()
    tab.widget()
    tab.on_data_loaded(df)
    tab._select("TA")
    QApplication.processEvents()
    return tab


def _payload_with_timeout(tab, kwargs, repeat, separate, timeout=60.0):
    """Run the tab's detection off-thread so a run that never stops fails the
    test instead of hanging it."""
    out = {}

    def target():
        out["payload"] = tab._compute_payload(tab._df["TA"], kwargs, repeat, separate)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"detection still running after {timeout:.0f} s"
    QApplication.processEvents()
    return out["payload"]


def test_lof_tab_runs_one_pass_by_default(app):
    from diive.gui.tabs.outliers_lof import LocalOutlierFactorTab
    tab = _make_tab(LocalOutlierFactorTab, _diel_frame())
    assert tab.repeat_cb.isChecked() is False

    iterations = []
    tab._sig.progress.connect(lambda it, *_: iterations.append(it))
    kwargs = tab._current_kwargs()
    assert kwargs["contamination"] == 0.01
    payload = _payload_with_timeout(tab, kwargs, tab.repeat_cb.isChecked(),
                                    tab.daynight_cb.isChecked())
    assert iterations == [1]
    # About 1 % of the records, not most of the series.
    assert 0 < payload["n_outliers"] < 0.02 * tab._df["TA"].count()
    assert ").run()" in tab._python_code()


def test_other_outlier_tabs_keep_repeat_on(app):
    from diive.gui.tabs.outliers_absolutelimits import AbsoluteLimitsTab
    tab = _make_tab(AbsoluteLimitsTab, _diel_frame())
    assert tab.repeat_cb.isChecked() is True


def test_absolutelimits_tab_daytime_nighttime_with_repeat_finishes(app):
    from diive.gui.tabs.outliers_absolutelimits import AbsoluteLimitsTab
    tab = _make_tab(AbsoluteLimitsTab, _diel_frame())
    tab.daynight_cb.setChecked(True)
    tab.lat.setValue(46.815333)
    tab.lon.setValue(9.855972)
    tab.dt_min.setValue(4.0)
    tab.dt_max.setValue(20.0)
    tab.nt_min.setValue(-2.0)
    tab.nt_max.setValue(10.0)
    assert tab.repeat_cb.isChecked() is True

    iterations = []
    tab._sig.progress.connect(lambda it, *_: iterations.append(it))
    kwargs = tab._current_kwargs()
    assert kwargs["separate_day_night"] is True
    payload = _payload_with_timeout(tab, kwargs, True, True)
    # Fixed limits: the second pass on the cleaned series finds nothing.
    assert iterations == [1, 2]
    assert payload["n_outliers"] > 0
