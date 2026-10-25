"""
TEST_GUI_PERF_OVERVIEW: speed-ups of the Overview tab (P5, P15)
================================================================

The Overview computes its ticks once per view, remembers where its 'best'
legends go, reuses its tag chips and settles the hero band before the first
render. Each shortcut must give the same result as the work it skips.

Run: pytest tests/test_gui_perf_overview.py -v
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
import shiboken6
from matplotlib.legend import Legend
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

import diive as dv


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or worker thread raised (copied
    from tests/test_gui.py: PySide6 hands such exceptions to
    ``sys.excepthook`` and carries on)."""
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


@pytest.fixture(scope="module", autouse=True)
def _no_save_prompt():
    """Answer "Save changes?" with Don't save (see tests/test_gui.py)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("diive.gui.app.ask_save_changes", lambda *a, **k: "discard")
        yield


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
def no_config(monkeypatch, example_year):
    import diive
    from diive.gui import config
    monkeypatch.setattr(config, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(diive, "load_exampledata_parquet",
                        lambda: example_year.copy())


@pytest.fixture
def window(app, no_config):
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    yield win
    _destroy_window(win)


def _overview(win, name="NEE_CUT_REF_f"):
    overview = win._tabs[0]
    overview._on_select(name)
    overview.varpanel.flush_pending()
    QApplication.processEvents()
    return overview


def _wait(ms):
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _plain_ticks(locator):
    """The ticks the locator computes without its per-view memo."""
    from diive.gui.tabs.overview import _TicksPerView
    return np.asarray(super(_TicksPerView, locator).__call__())


def test_ticks_per_view_equal_the_computed_ticks(window):
    from diive.gui.tabs.overview import (
        _AutoTicksPerView, _DateTicksPerView, _FittedDateLocator,
        _MaxNTicksPerView, _TicksPerView)
    overview = _overview(window)
    fig = overview.canvas.fig
    kinds = {type(a.yaxis.get_major_locator()) for a in fig.axes}
    assert _AutoTicksPerView in kinds  # value axes
    assert type(overview._heatmap_ax.yaxis.get_major_locator()) is _DateTicksPerView
    assert type(overview._hist_ax.xaxis.get_major_locator()) is _MaxNTicksPerView
    assert type(overview._shared_x_ax.xaxis.get_major_locator()) is _FittedDateLocator

    def check():
        overview.canvas._canvas.draw()
        checked = 0
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                locator = axis.get_major_locator()
                if isinstance(locator, _TicksPerView):
                    np.testing.assert_array_equal(locator(), _plain_ticks(locator))
                    checked += 1
        assert checked >= 10

    check()
    x0, x1 = overview._shared_x_ax.get_xlim()
    for lo, hi in ((0.1, 0.5), (0.4, 0.45), (0.0, 1.0)):
        overview._shared_x_ax.set_xlim(x0 + (x1 - x0) * lo, x0 + (x1 - x0) * hi)
        check()
    overview._zoom_debounce.flush()  # rebuilt diel cycle and histogram
    check()
    fig.set_size_inches(fig.get_size_inches() * 0.6)  # fewer ticks fit
    check()


def test_one_draw_computes_the_date_ticks_once_per_axis(window, monkeypatch):
    overview = _overview(window)
    ax = overview._shared_x_ax
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x0 + (x1 - x0) * 0.3)
    overview.canvas._canvas.draw()
    calls = []
    original = mdates.AutoDateLocator.__call__
    monkeypatch.setattr(mdates.AutoDateLocator, "__call__",
                        lambda self: (calls.append(1), original(self))[1])
    ax.set_xlim(x0 + (x1 - x0) * 0.1, x0 + (x1 - x0) * 0.4)  # a pan step
    overview.canvas._canvas.draw()
    # The linked date axes plus the heatmap's date axis; without the memo
    # each draw asked about 25 times.
    assert 1 <= len(calls) <= 2
    overview._zoom_debounce.cancel()


def _legends(ax):
    return [a for a in ax.get_children() if isinstance(a, Legend)]


def test_placed_legend_lands_where_matplotlib_puts_it(window):
    from diive.gui.tabs.overview import _PlacedLegend
    overview = _overview(window)
    canvas = overview.canvas._canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    for ax in (overview._diel_ax, overview._hist_ax):
        legends = _legends(ax)
        assert len(legends) == 2
        assert all(type(leg) is _PlacedLegend for leg in legends)

    def check(ax):
        for leg in _legends(ax):
            box = leg._legend_box.get_window_extent(renderer)
            got = leg._find_best_position(box.width, box.height, renderer)
            assert got == Legend._find_best_position(leg, box.width, box.height, renderer)
            assert leg._best[0] is not None

    check(overview._diel_ax)
    check(overview._hist_ax)
    # Moving the data under a legend must move the cached answer too.
    diel = overview._diel_ax
    before = [leg._best for leg in _legends(diel)]
    lo, hi = diel.get_ylim()
    diel.set_ylim(lo - (hi - lo), hi)
    check(diel)
    assert [leg._best for leg in _legends(diel)] != before


def test_first_render_waits_for_the_hero_band(app, no_config, monkeypatch):
    # The hero band's stats wrap onto more rows once they have text, which
    # takes several layout passes to reach the canvas. The first render must
    # wait for them, or the canvas shrinks right after it and the figure is
    # laid out and drawn a second time (seen at startup).
    from matplotlib.figure import Figure
    from diive.gui.app import MainWindow
    from diive.gui.widgets.mpl_canvas import MplCanvas
    win = MainWindow(None, autoload=False)
    try:
        win.resize(800, 800)
        win.show()
        app.processEvents()
        overview = win._tabs[0]
        draws, solves = [], []
        draw, solve = Figure.draw, MplCanvas._solve_layout
        monkeypatch.setattr(
            Figure, "draw",
            lambda fig, r: (draws.append(overview.canvas.height()), draw(fig, r))[1])
        monkeypatch.setattr(MplCanvas, "_solve_layout",
                            lambda self: (solves.append(1), solve(self))[1])
        win._load_example()
        overview.varpanel.flush_pending()
        _wait(400)  # longer than the canvas' resize settle time
        assert overview._current
        assert draws == [overview.canvas.height()]
        assert solves == []
    finally:
        _destroy_window(win)


def test_hero_tag_chips_are_reused(window):
    from diive.gui import metadata_store
    overview = _overview(window, "Tair_f")
    hero = overview.hero
    for tag in ("zeta", "alpha", "mid"):
        metadata_store.manager.add_user_tag("Tair_f", tag)
    overview = _overview(window, "Tair_f")
    chips = list(hero._tag_chips)
    assert [c.text() for c in chips] == ["alpha", "mid", "zeta"]
    assert all(not c.isHidden() for c in chips)
    assert "border-radius" in chips[0].styleSheet()

    metadata_store.manager.add_user_tag("VPD_f", "only")
    _overview(window, "VPD_f")
    assert hero._tag_chips == chips  # same widgets, no new ones
    assert chips[0].text() == "only" and not chips[0].isHidden()
    assert all(c.isHidden() for c in chips[1:])

    _overview(window, "Rg_f")  # no tags
    assert all(c.isHidden() for c in hero._tag_chips)


def test_select_removes_the_previous_panels(window):
    overview = _overview(window, "Tair_f")
    old_axes = list(overview.canvas.fig.axes)
    overview = _overview(window, "VPD_f")
    axes = overview.canvas.fig.axes
    assert len(axes) == len(old_axes)
    assert not set(map(id, axes)) & set(map(id, old_axes))
    assert all(ax.figure is None or ax not in ax.figure.axes for ax in old_axes)
    # Zooming the new panels still reaches the heatmap and summaries.
    x0, x1 = overview._shared_x_ax.get_xlim()
    heat = overview._heatmap_ax.get_ylim()
    overview._shared_x_ax.set_xlim(x0 + (x1 - x0) * 0.2, x0 + (x1 - x0) * 0.4)
    assert overview._heatmap_ax.get_ylim() != heat
    assert overview._zoom_debounce.pending()
    overview._zoom_debounce.cancel()
