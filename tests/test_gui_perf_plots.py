"""
TEST_GUI_PERF_PLOTS: plot tabs render once, thin long lines, draw heatmap images
================================================================================

Checks for the speed-ups of the Plot menu tabs and the heatmap previews (P9 in
devnotes/GUI_PERFORMANCE.md):

- a plot tab renders once when opened, at its final size, and a render asked
  for while the tab is hidden (a theme change) waits until it is shown;
- the time series is drawn thinned to the screen columns, is thinned again
  for every zoom, pan and resize, and the hover still reads every record;
- the Heatmap tab and the heatmap previews draw one image instead of a mesh.

Run: pytest tests/test_gui_perf_plots.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
import threading
import traceback
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import diive as dv


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or worker thread raised (copied from
    tests/test_gui.py: PySide6 hands such exceptions to ``sys.excepthook``)."""
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
def example_data():
    # The full ten years: a year is too short to be thinned on a wide canvas.
    return dv.load_exampledata_parquet()


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
def window(app, monkeypatch, example_data):
    import diive
    from diive.gui import config
    monkeypatch.setattr(config, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(diive, "load_exampledata_parquet",
                        lambda: example_data.copy())
    discard = QMessageBox.StandardButton.Discard
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: discard)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("diive.gui.app.ask_save_changes",
                        lambda *a, **k: "discard", raising=False)
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.resize(1600, 1000)
    win.show()
    app.processEvents()
    win._load_example()
    app.processEvents()
    yield win
    _destroy_window(win)


def _open(win, label):
    win._open_menu_tab(label)
    tab = win._menu_tab_list[-1]
    tab.varpanel.flush_pending()
    QApplication.processEvents()
    return tab


def _count_draws(monkeypatch, fig):
    """Record the size of every full render of `fig`."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    sizes = []
    orig = FigureCanvasAgg.draw

    def draw(self, *a, **k):
        if self.figure is fig:
            sizes.append(tuple(fig.bbox.size))
        return orig(self, *a, **k)

    monkeypatch.setattr(FigureCanvasAgg, "draw", draw)
    return sizes


# --- render once, when shown ----------------------------------------------------

@pytest.mark.parametrize("label", ["Diel cycle", "Cumulative", "Time series"])
def test_plot_tab_renders_once_when_shown(window, monkeypatch, label):
    from diive.gui.tabs import plotting

    created = []
    orig_build = plotting.PlottingTab.build

    def build(self):
        root = orig_build(self)
        created.append(_count_draws(monkeypatch, self.canvas.fig))
        return root

    monkeypatch.setattr(plotting.PlottingTab, "build", build)
    window._open_menu_tab(label)
    tab = window._menu_tab_list[-1]
    for _ in range(5):
        QApplication.processEvents()
    sizes = created[0]
    # Rendered once, at the size the canvas has now.
    assert tab._pending_render is None
    assert tab.canvas.fig.axes and tab._panels == ["NEE_CUT_REF_f"]
    assert len(sizes) == 1, sizes
    assert sizes[0] == tuple(tab.canvas.fig.bbox.size)


def test_hidden_plot_tab_renders_when_shown_again(window):
    # A theme change re-renders a Time series tab; while another tab is in
    # front that render waits until the tab is shown again.
    from diive.gui import theme
    tab = _open(window, "Time series")
    tw = window._tabwidget
    tw.setCurrentIndex(0)
    QApplication.processEvents()
    before = list(tab.canvas.fig.axes)
    theme.manager.changed.emit()
    assert tab._pending_render is False
    assert tab.canvas.fig.axes == before  # nothing re-rendered while hidden
    tw.setCurrentWidget(tab.widget())
    QApplication.processEvents()
    QApplication.processEvents()
    assert tab._pending_render is None
    assert all(ax not in tab.canvas.fig.axes for ax in before)


def test_unshown_tab_still_renders_at_once(app, example_data):
    # A tab outside a shown window (tests, a standalone build) renders right away.
    from diive.gui.tabs.plotting import PlottingTab
    from diive.gui.widgets.plot_settings import DIELCYCLE
    tab = PlottingTab(DIELCYCLE, "Diel cycle")
    tab.widget()
    tab.on_data_loaded(example_data.loc["2021"], set())
    assert tab._pending_render is None
    assert tab.canvas.fig.axes and tab.canvas.fig.axes[0].get_lines()


# --- thinned time series ------------------------------------------------------------

def _ts_line(tab):
    line, x, y = tab._thinned[0]
    return line, x, y


def test_time_series_is_thinned_and_rethinned_on_zoom(window):
    tab = _open(window, "Time series")
    series = window._data["NEE_CUT_REF_f"]
    line, full_x, full_y = _ts_line(tab)
    ax = line.axes
    np.testing.assert_array_equal(full_y, series.to_numpy(float))
    assert line._diive_hover_xy[1] is full_y
    drawn_y = np.asarray(line.get_ydata(orig=False), float)
    assert len(drawn_y) < len(series) / 5
    # Every extreme is kept.
    assert np.nanmax(drawn_y) == series.max() and np.nanmin(drawn_y) == series.min()

    # Zoom to three days: the line holds every record of the window again.
    i = 100_000
    ax.set_xlim(full_x[i], full_x[i + 144])
    x_now = np.asarray(line.get_xdata(orig=False), float)
    in_window = full_x[i - 1:i + 146]
    np.testing.assert_array_equal(x_now, in_window)

    # Home (the full range) thins it again.
    ax.set_xlim(full_x[0], full_x[-1])
    assert len(line.get_xdata()) < len(series) / 5


def test_time_series_rethinned_on_resize(window):
    tab = _open(window, "Time series")
    line, _, _ = _ts_line(tab)
    n_wide = len(line.get_xdata())
    from matplotlib.backend_bases import ResizeEvent
    tab.canvas.fig.set_size_inches(tab.canvas.fig.get_size_inches() / 2)
    ResizeEvent("resize_event", tab.canvas.fig.canvas)._process()
    n_narrow = len(line.get_xdata())
    assert n_narrow < 0.7 * n_wide


def test_time_series_hover_reads_records_left_out(window):
    tab = _open(window, "Time series")
    series = window._data["NEE_CUT_REF_f"]
    line, full_x, full_y = _ts_line(tab)
    ax = line.axes
    drawn_x = set(np.asarray(line.get_xdata(orig=False), float))
    i = next(k for k in range(5000, len(full_x)) if full_x[k] not in drawn_x
             and np.isfinite(full_y[k]))
    px, py = ax.transData.transform((full_x[i], full_y[i]))
    ev = types.SimpleNamespace(inaxes=ax, xdata=full_x[i], ydata=full_y[i], x=px, y=py)
    hx, hy, text, _ = tab.canvas.hover._value_at(ax, ev)
    assert (hx, hy) == (full_x[i], full_y[i])
    assert series.index[i].strftime("%Y-%m-%d %H:%M") in text


def test_stacked_time_series_panels_both_rethin(window):
    tab = _open(window, "Time series")
    tab._on_selected("Tair_f", True)
    tab.varpanel.flush_pending()
    QApplication.processEvents()
    assert tab._panels == ["NEE_CUT_REF_f", "Tair_f"]
    assert len(tab._thinned) == 2
    (l1, x1, _), (l2, _, _) = tab._thinned
    l1.axes.set_xlim(x1[50_000], x1[50_000 + 96])  # shared x: both follow
    assert len(l1.get_xdata()) <= 99 and len(l2.get_xdata()) <= 99


def test_time_series_with_markers_is_not_thinned(window):
    tab = _open(window, "Time series")
    tab.settings.marker.setChecked(True)
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab._thinned == []
    line = tab.canvas.fig.axes[0].get_lines()[0]
    assert len(line.get_xdata()) == len(window._data)


def test_thinned_line_looks_like_the_full_line(window, monkeypatch):
    # Rendered pixels of the thinned and the full line agree, apart from
    # antialiasing at a small share of line edges.
    from diive.gui.tabs import plotting
    tab = _open(window, "Time series")

    def pixels():
        tab._render()
        QApplication.processEvents()
        return np.asarray(tab.canvas.fig.canvas.buffer_rgba())[..., :3].astype(int)

    thinned = pixels()
    monkeypatch.setattr(plotting.PlottingTab, "_keep_thinned", lambda *a, **k: None)
    full = pixels()
    diff = np.abs(thinned - full).max(axis=2)
    assert thinned.shape == full.shape
    assert np.count_nonzero(diff > 96) / diff.size < 0.0005
    assert np.count_nonzero(diff) / diff.size < 0.02


# --- heatmaps as images ----------------------------------------------------------------

def test_heatmap_tab_draws_an_image_that_zooms_and_hovers(window):
    from matplotlib.collections import QuadMesh
    tab = _open(window, "Heatmap date/time")
    ax = tab.canvas.fig.axes[0]
    image = ax.get_images()[0]
    assert not any(isinstance(c, QuadMesh) for c in ax.collections)
    series = window._data["NEE_CUT_REF_f"]
    values = np.ma.asarray(image.get_array()).filled(np.nan)
    assert values.shape == (len(series) // 48, 48)

    # One row per day, one column per half-hour, in record order (the record
    # covers whole days).
    np.testing.assert_array_equal(values.ravel(), series.to_numpy(float))

    # Hover reads the cell under the cursor: 2021-06-01, the half-hour around
    # 12:15.
    import matplotlib.dates as mdates
    day = mdates.date2num(pd.Timestamp("2021-06-01")) + 0.5
    ev = types.SimpleNamespace(inaxes=ax, xdata=12.25, ydata=day, x=0, y=0)
    _, _, text, _ = tab.canvas.hover._value_at(ax, ev)
    expected = series.loc["2021-06-01 12:15"]
    assert f"{expected:.4g}" in text

    # Zoom to one month: the image stays, the view follows.
    lo = mdates.date2num(pd.Timestamp("2021-06-01"))
    ax.set_ylim(lo, lo + 30)
    tab.canvas.draw()
    assert ax.get_images()[0] is image
    assert ax.get_ylim() == (lo, lo + 30)


def test_heatmap_image_looks_like_the_mesh(app, example_data):
    # Same picture: one year at the plot tab's size, image vs mesh.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    series = example_data["NEE_CUT_REF_f"].loc["2021"]

    def pixels(as_image):
        fig = Figure(figsize=(6, 5), dpi=100, layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        dv.plotting.HeatmapDateTime(series).plot(ax=ax, fig=fig, as_image=as_image)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)

    diff = np.abs(pixels(True) - pixels(False)).max(axis=2)
    # Cell edges can fall one pixel apart; colours elsewhere agree.
    assert np.count_nonzero(diff > 64) / diff.size < 0.02


def test_preview_heatmaps_are_images(app, example_data):
    from diive.gui.tabs.combine_variables import CombineVariablesTab
    df = example_data[["NEE_CUT_REF_f", "Tair_f"]].loc["2021"]
    tab = CombineVariablesTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._assign(1, "NEE_CUT_REF_f")
    QApplication.processEvents()
    ax = tab.slot1.canvas.fig.axes[0]
    image = ax.get_images()[0]
    assert not ax.collections or all(c.__class__.__name__ != "QuadMesh"
                                     for c in ax.collections)
    values = np.ma.asarray(image.get_array()).filled(np.nan)
    np.testing.assert_array_equal(
        np.sort(values[np.isfinite(values)]),
        np.sort(df["NEE_CUT_REF_f"].dropna().to_numpy()))


# --- project restore ------------------------------------------------------------------

def test_restored_plot_tab_renders_once_with_restored_settings(window):
    # restore_state replays the saved clicks without their queued renders and
    # renders once, with the restored panels and per-panel settings.
    tab = _open(window, "Time series")
    tab._on_selected("Tair_f", True)
    tab.varpanel.flush_pending()
    tab.settings.linewidth.setValue(5.0)  # the second (active) panel only
    tab.update_btn.click()
    QApplication.processEvents()
    state = tab.save_state()

    fresh = _open(window, "Time series")
    calls = []
    render = fresh._render

    def counting(*a, **k):
        calls.append(a)
        return render(*a, **k)

    fresh._render = counting
    fresh.restore_state(state)
    for _ in range(3):
        QApplication.processEvents()
    fresh.varpanel.flush_pending()
    assert len(calls) == 1
    assert fresh._panels == ["NEE_CUT_REF_f", "Tair_f"]
    assert fresh._active_panel == "Tair_f"
    assert fresh.settings.linewidth.value() == 5.0
    axes = fresh.canvas.fig.axes
    titles = [ax.get_title(loc) for ax in axes for loc in ("left", "center")]
    assert any("NEE_CUT_REF_f" in t for t in titles)
    assert any("Tair_f" in t for t in titles)
    widths = [ax.get_lines()[0].get_linewidth() for ax in axes[:2]]
    assert widths[1] == pytest.approx(5.0) and widths[0] != pytest.approx(5.0)
