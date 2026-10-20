"""
TEST_GUI_OVERVIEW_NONNUMERIC: the Overview with non-numeric columns
===================================================================

The Overview must never plot a text, datetime or boolean column (L333): a text
column became a categorical axis with one category per record and froze the
GUI for minutes. Adding such a column keeps the current selection, and
selecting one shows a note instead of the panels.

Run: pytest tests/test_gui_overview_nonnumeric.py -v
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
from PySide6.QtWidgets import QApplication

import diive as dv


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or worker thread raised.

    PySide6 hands such exceptions to ``sys.excepthook`` and carries on, so a
    test would otherwise pass with a crashed handler (copied from
    tests/test_gui.py)."""
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
        pytest.fail(
            f"{len(captured)} exception(s) raised inside a Qt slot or worker "
            "thread and swallowed by PySide6:\n\n" + "\n".join(captured),
            pytrace=False)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def example_year():
    df = dv.load_exampledata_parquet()
    return dv.times.keep_daterange(df, "2021-01-01", "2021-12-31 23:30")


def _destroy_window(win):
    """Destroy a MainWindow completely (see tests/test_gui.py for why)."""
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
    from diive.gui import config
    monkeypatch.setattr(config, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(diive, "load_exampledata_parquet",
                        lambda: example_year.copy())
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


def _settle(overview):
    """Run the deferred Overview render (VariablePanel.run_with_loading)."""
    overview.varpanel.flush_pending()
    QApplication.processEvents()


def _figure_texts(canvas):
    return [t.get_text() for t in canvas.fig.texts]


def _text_timestamp(win):
    """The column Variables > Add timestamp column builds with its default
    text format."""
    from diive.times import format_timestamp
    return format_timestamp(win._full_data, convention="end",
                            fmt="%Y-%m-%d %H:%M:%S").rename("TIMESTAMP_TEXT")


def test_adding_text_column_keeps_selection_and_is_fast(window):
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    _settle(overview)
    prev_axes = list(overview.canvas.fig.axes)
    assert prev_axes

    t0 = time.perf_counter()
    window._add_features(_text_timestamp(window).to_frame())
    _settle(overview)
    elapsed = time.perf_counter() - t0

    assert "TIMESTAMP_TEXT" in overview._df.columns
    assert "TIMESTAMP_TEXT" in overview._created  # a new column, yet skipped
    assert overview._current == "NEE_CUT_REF_f"
    assert "TIMESTAMP_TEXT" in overview.varpanel.names()
    # The figure is the numeric variable again: datetime panels, no note.
    assert overview._shared_x_ax in overview.canvas.fig.axes
    assert not any("not numeric" in t for t in _figure_texts(overview.canvas))
    # Before the fix one year of text records did not finish drawing in 600 s.
    assert elapsed < 30


def test_selecting_text_column_shows_note_instead_of_plot(window):
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    _settle(overview)
    window._add_features(_text_timestamp(window).to_frame())
    _settle(overview)

    t0 = time.perf_counter()
    overview._on_select("TIMESTAMP_TEXT")
    _settle(overview)
    assert time.perf_counter() - t0 < 30

    assert overview._current == "TIMESTAMP_TEXT"
    assert overview.canvas.fig.axes == []
    assert _figure_texts(overview.canvas) == [
        "TIMESTAMP_TEXT is not numeric, nothing to plot."]
    assert overview._shared_x_ax is None and overview._zoom_series is None
    # The hero still shows the record extent, the numeric stats stay empty.
    assert overview.hero._name.text() == "TIMESTAMP_TEXT"
    assert overview.hero._slots["NOV"]._value.text() == f"{len(overview._df):,}"
    assert overview.hero._slots["MEAN ± SD"]._value.text() == "—"

    # Event focus and an event-overlay refresh find nothing to draw on.
    overview.focus_on(pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-10"))
    overview.refresh_events()
    assert overview.canvas.fig.axes == []


@pytest.mark.parametrize("kind", ["datetime", "bool", "object_bool"])
def test_other_non_numeric_columns_show_note(window, kind):
    overview = window._tabs[0]
    nee = window._full_data["NEE_CUT_REF_f"]
    col = {
        "datetime": pd.Series(nee.index, index=nee.index),
        "bool": nee > 0,
        "object_bool": (nee > 0).astype(object),
    }[kind].rename(f"X_{kind}")
    window._add_features(col.to_frame())
    _settle(overview)
    assert overview._current != col.name  # not auto-selected

    overview._on_select(col.name)
    _settle(overview)
    assert overview.canvas.fig.axes == []
    assert _figure_texts(overview.canvas) == [
        f"{col.name} is not numeric, nothing to plot."]


def test_numeric_column_still_renders_after_note(window):
    overview = window._tabs[0]
    window._add_features(_text_timestamp(window).to_frame())
    _settle(overview)
    overview._on_select("TIMESTAMP_TEXT")
    _settle(overview)
    assert overview.canvas.fig.axes == []

    # A new numeric column is still auto-selected and drawn in full.
    new = (window._full_data["NEE_CUT_REF_f"] * 2).rename("NEE_DOUBLE")
    window._add_features(new.to_frame())
    _settle(overview)
    assert overview._current == "NEE_DOUBLE"
    assert overview.canvas.fig.texts == []
    assert len(overview.canvas.fig.axes) >= 7  # seven panels (+ colourbar)
    assert overview._shared_x_ax in overview.canvas.fig.axes
    assert overview._zoom_series is not None

    prev_axes = list(overview.canvas.fig.axes)
    overview._on_select("NEE_CUT_REF_f")
    _settle(overview)
    assert all(ax not in overview.canvas.fig.axes for ax in prev_axes)
    assert overview._shared_x_ax in overview.canvas.fig.axes
