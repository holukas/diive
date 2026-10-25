"""
TEST_GUI_PERF_ANALYSIS: faster redraws of the Spectrogram and Seasonal trend tabs
=================================================================================

The Spectrogram tab draws only the mesh cells in view (`_ViewCroppedMesh`) and
the Seasonal trend tab computes its date ticks once per view
(`_ViewMemoDateLocator`). Both must paint exactly what the plain matplotlib
artists paint (GUI_PERFORMANCE.md P11).

Run: pytest tests/test_gui_perf_analysis.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
import threading
import time
import traceback

import pytest

pytest.importorskip("PySide6")

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import shiboken6
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

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


@pytest.fixture(scope="module", autouse=True)
def _worker_process():
    """Stop the shared worker process (Seasonal trend) after this module."""
    yield
    import multiprocessing
    from diive.gui.widgets.worker import shutdown_process_pool
    shutdown_process_pool()
    assert multiprocessing.active_children() == []


@pytest.fixture(scope="module")
def example_two_years():
    # The decomposition needs two annual cycles of daily data.
    df = dv.load_exampledata_parquet()
    return dv.times.keep_daterange(df, "2020-01-01", "2021-12-31 23:30")


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
def window(app, monkeypatch, example_two_years):
    import diive
    from diive.gui import config
    monkeypatch.setattr(config, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(diive, "load_exampledata_parquet",
                        lambda: example_two_years.copy())
    discard = QMessageBox.StandardButton.Discard
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: discard)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("diive.gui.app.ask_save_changes",
                        lambda *a, **k: "discard", raising=False)
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    yield win
    _destroy_window(win)


def _wait_for_worker(tab, timeout: float = 120.0) -> None:
    """Pump the event loop until the tab's background compute has been drawn."""
    deadline = time.monotonic() + timeout
    while tab._runner.is_busy:
        if time.monotonic() > deadline:
            raise AssertionError(f"{type(tab).__name__}: worker still busy "
                                 f"after {timeout:.0f} s")
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _count_triangles(fig):
    """Draw `fig` and return the number of gouraud triangles sent to Agg."""
    renderer = fig.canvas.get_renderer()
    sent = []
    draw = renderer.draw_gouraud_triangles

    def record(gc, triangles, colors, transform):
        sent.append(len(triangles))
        return draw(gc, triangles, colors, transform)

    renderer.draw_gouraud_triangles = record
    try:
        fig.canvas.draw()
    finally:
        renderer.draw_gouraud_triangles = draw
    return sum(sent)


# --- Spectrogram: mesh cropped to the view ---------------------------------

def _spectrogram_figure(cropped, xlim=None, ylim=(0, 4), nan_block=False):
    from diive.gui.tabs.spectrogram import _ViewCroppedMesh
    rng = np.random.default_rng(3)
    x = mdates.date2num(pd.date_range("2021-01-01", periods=300, freq="D"))
    y = np.linspace(0, 24, 129)
    power = rng.normal(size=(len(y), len(x))).cumsum(axis=1)
    if nan_block:
        power[10:20, 50:80] = np.nan
    fig = Figure(figsize=(7.3, 4.1), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(power), shading="gouraud")
    if cropped:
        mesh.__class__ = _ViewCroppedMesh
    ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.xaxis_date()
    fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01)
    return fig, mesh


@pytest.mark.parametrize("view", [
    dict(ylim=(0, 4)),
    dict(ylim=(0, 24)),
    dict(ylim=(2.1, 3.37)),
    dict(ylim=(4, 0)),
    dict(ylim=(0, 4), xlim=(18700, 18760.3)),
    dict(ylim=(30, 40)),
    dict(ylim=(0, 6), nan_block=True),
])
def test_cropped_mesh_paints_the_same_pixels(view):
    plain, _ = _spectrogram_figure(False, **view)
    cropped, mesh = _spectrogram_figure(True, **view)
    np.testing.assert_array_equal(_pixels(cropped), _pixels(plain))
    # The full mesh is still there outside the draw.
    assert mesh.get_array().shape == (129, 300)
    assert mesh._coordinates.shape == (129, 300, 2)


def test_cropped_mesh_sends_only_the_cells_in_view():
    plain, _ = _spectrogram_figure(False, ylim=(0, 4))
    cropped, mesh = _spectrogram_figure(True, ylim=(0, 4))
    n_plain = _count_triangles(plain)
    n_cropped = _count_triangles(cropped)
    assert n_plain == 4 * 128 * 299
    assert 0 < n_cropped < n_plain / 4
    # Raising the limit needs no rebuild: the next draw takes more rows.
    mesh.axes.set_ylim(0, 24)
    assert _count_triangles(cropped) == n_plain


def test_spectrogram_tab_draws_cropped_mesh(window):
    from diive.gui.tabs.spectrogram import _ViewCroppedMesh
    window._open_menu_tab("Spectrogram")
    tab = window._menu_tab_list[-1]
    _wait_for_worker(tab)
    mesh = tab._mesh
    assert type(mesh) is _ViewCroppedMesh
    fig = tab.canvas.fig
    n_rows = mesh.get_array().shape[0]

    for limit in (4.0, 12.0):
        tab.max_freq.setValue(limit)
        QApplication.processEvents()
        assert tab._mesh is mesh
        assert mesh.axes.get_ylim()[1] == pytest.approx(limit)
        fast = _pixels(fig)
        mesh.__class__ = QuadMesh
        try:
            plain = _pixels(fig)
        finally:
            mesh.__class__ = _ViewCroppedMesh
        np.testing.assert_array_equal(fast, plain)
    assert mesh.get_array().shape[0] == n_rows


# --- Seasonal trend: date ticks computed once per view ---------------------

def _date_panels(memo):
    from diive.gui.tabs.seasonaltrend import _ViewMemoDateLocator
    idx = pd.date_range("2013-01-01", "2022-12-31", freq="D")
    values = np.sin(np.arange(len(idx)) / 58.0)
    fig = Figure(figsize=(9, 7), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 1, sharex=True)
    for ax in axes:
        ax.plot(idx, values, linewidth=1.0)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    locator = axes[0].xaxis.get_major_locator()
    assert type(locator) is mdates.AutoDateLocator
    if memo:
        locator.__class__ = _ViewMemoDateLocator
    return fig, axes


def test_memo_date_locator_gives_the_same_ticks_and_pixels():
    plain_fig, plain_axes = _date_panels(False)
    memo_fig, memo_axes = _date_panels(True)
    np.testing.assert_array_equal(_pixels(memo_fig), _pixels(plain_fig))
    # A new view gets new ticks, not the remembered ones.
    for lo, hi in (("2016-03-01", "2016-09-01"), ("2014-01-01", "2021-01-01")):
        for axes in (plain_axes, memo_axes):
            axes[0].set_xlim(pd.Timestamp(lo), pd.Timestamp(hi))
        np.testing.assert_array_equal(_pixels(memo_fig), _pixels(plain_fig))
        np.testing.assert_array_equal(memo_axes[-1].get_xticks(),
                                      plain_axes[-1].get_xticks())
        assert ([t.get_text() for t in memo_axes[-1].get_xticklabels()]
                == [t.get_text() for t in plain_axes[-1].get_xticklabels()])


def test_memo_date_locator_asks_the_rule_once_per_view():
    fig, axes = _date_panels(True)
    locator = axes[0].xaxis.get_major_locator()
    calls = []
    get_locator = locator.get_locator

    def counting(dmin, dmax):
        calls.append((dmin, dmax))
        return get_locator(dmin, dmax)

    locator.get_locator = counting
    fig.canvas.draw()
    assert len(calls) == 1


def test_seasonal_trend_tab_uses_memo_locator(window):
    from diive.gui.tabs.seasonaltrend import _ViewMemoDateLocator
    window._open_menu_tab("Seasonal trend & anomalies")
    tab = window._menu_tab_list[-1]
    _wait_for_worker(tab)
    assert tab._decomp is not None
    fig = tab.canvas.fig
    assert len(fig.axes) == 4
    locator = fig.axes[0].xaxis.get_major_locator()
    assert type(locator) is _ViewMemoDateLocator
    assert all(ax.xaxis.get_major_locator() is locator for ax in fig.axes)
    fast = _pixels(fig)
    locator.__class__ = mdates.AutoDateLocator
    try:
        plain = _pixels(fig)
    finally:
        locator.__class__ = _ViewMemoDateLocator
    np.testing.assert_array_equal(fast, plain)
