"""
TEST_GUI_PERF_APP: TAB OPENS, CANVAS RESIZES AND BACKGROUND IMPORTS
===================================================================

Checks for the GUI performance items P6, P8 and P16 (devnotes/GUI_PERFORMANCE.md):

- a resize that only bounces (the size changes and returns before the settle
  render, as when a menu tab is added) neither solves the layout nor renders;
- a canvas in a hidden tab page settles when the page is shown again;
- a menu tab renders once, at the size it is shown at;
- the background import of the library modules finishes before a tab opens;
- a visibility-only event toggle does not rebuild the event flag columns.

Run: pytest tests/test_gui_perf_app.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
import threading
import traceback

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QStackedWidget, QWidget

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
    discard = QMessageBox.StandardButton.Discard
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: discard)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("diive.gui.app.ask_save_changes",
                        lambda *a, **k: "discard", raising=False)
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    app.processEvents()
    win._tabs[0].varpanel.flush_pending()  # the Overview's first render
    app.processEvents()
    yield win
    _destroy_window(win)
    from diive.gui import events as _events
    _events.manager.events.clear()
    _events.manager.visible = True
    _events.manager.categories = dict(_events.DEFAULT_CATEGORIES)


def _pump():
    for _ in range(3):
        QApplication.sendPostedEvents()
        QApplication.processEvents()


def _line_canvas(parent=None):
    """A rendered MplCanvas with one black line. Records every real (Agg)
    render as the widget size and every layout solve."""
    from diive.gui.widgets.mpl_canvas import MplCanvas
    canvas = MplCanvas(parent, show_toolbar=False)
    # A slow test run must not let the settle timer fire on its own.
    canvas._relayout_timer.setInterval(60_000)
    fc = canvas._canvas
    renders, solves = [], []
    orig_draw, orig_solve = fc.draw, canvas._solve_layout
    fc.draw = lambda: (renders.append((fc.width(), fc.height())), orig_draw())[1]
    canvas._solve_layout = lambda: (solves.append((fc.width(), fc.height())),
                                    orig_solve())[1]
    return canvas, fc, renders, solves


def _render_line(canvas):
    ax = canvas.new_axes(1)[0]
    ax.plot(range(50), color="black", linewidth=3)
    canvas.draw()
    QApplication.processEvents()


def _shows_line(fc) -> bool:
    shot = fc.grab().toImage()
    return any(shot.pixelColor(x, y).lightness() < 60
               for x in range(0, shot.width(), 4)
               for y in range(0, shot.height(), 4))


def _bounce(fc, dh=-2):
    w, h = fc.width(), fc.height()
    fc.resize(w, h + dh)
    fc.resize(w, h)
    _pump()


# --- P16: canvas resizes ------------------------------------------------------

def test_bounce_after_render_neither_solves_nor_renders(app):
    # The first resize after a render used to solve the layout at once; when
    # the size returns before the queued render, nothing is redone.
    canvas, fc, renders, solves = _line_canvas()
    try:
        canvas.resize(500, 350)
        canvas.show()
        _render_line(canvas)
        renders.clear(), solves.clear()
        renderer = fc.renderer
        _bounce(fc)
        assert renders == [] and solves == []
        assert not canvas._relayout_timer.isActive()
        assert not canvas._solve_pending and not fc._draw_pending
        assert fc.renderer is renderer
        assert _shows_line(fc)
    finally:
        canvas.close()


def test_bounce_after_a_settled_resize_neither_solves_nor_renders(app):
    canvas, fc, renders, solves = _line_canvas()
    try:
        canvas.resize(500, 350)
        canvas.show()
        _render_line(canvas)
        w0, h0 = fc.width(), fc.height()
        fc.resize(w0 + 40, h0 + 20)  # first resize: solved and rendered at once
        _pump()
        assert renders[-1] == (w0 + 40, h0 + 20)
        assert solves[-1] == (w0 + 40, h0 + 20)
        renders.clear(), solves.clear()
        _bounce(fc, dh=4)
        assert renders == [] and solves == []
        assert not canvas._relayout_timer.isActive()
        assert fc._last_frame is None
        assert _shows_line(fc)
        # A real change of size still settles: one solve and one render.
        fc.resize(w0 + 40, h0 + 60)
        _pump()
        assert canvas._relayout_timer.isActive() and renders == []
        canvas._relayout_timer.timeout.emit()
        _pump()
        assert solves == [(w0 + 40, h0 + 60)]
        assert renders == [(w0 + 40, h0 + 60)]
    finally:
        canvas.close()


def test_bounce_keeps_a_render_queued_for_a_figure_change(app):
    # A render the figure itself asked for is not dropped with the bounce.
    canvas, fc, renders, solves = _line_canvas()
    try:
        canvas.resize(500, 350)
        canvas.show()
        _render_line(canvas)
        renders.clear()
        canvas.fig.axes[0].lines[0].set_color("red")
        canvas.draw_idle()
        _bounce(fc)
        assert renders == [(fc.width(), fc.height())]
    finally:
        canvas.close()


def test_hidden_page_settles_when_shown_again(app):
    # A settle that comes due while the canvas sits in a hidden page (a menu
    # tab opened over it) waits until the page is shown again.
    stack = QStackedWidget()
    canvas, fc, renders, solves = _line_canvas()
    stack.addWidget(canvas)
    stack.addWidget(QWidget())
    try:
        stack.resize(500, 350)
        stack.show()
        _pump()
        _render_line(canvas)
        w0, h0 = fc.width(), fc.height()
        fc.resize(w0, h0 - 30)  # first resize after the render
        _pump()
        fc.resize(w0, h0 - 20)  # a later one: settles on the timer
        _pump()
        assert canvas._relayout_timer.isActive()
        renders.clear(), solves.clear()
        stack.setCurrentIndex(1)
        canvas._relayout_timer.timeout.emit()
        _pump()
        assert solves == [] and renders == []
        assert canvas._relayout_owed
        stack.setCurrentIndex(0)
        _pump()
        assert not canvas._relayout_owed
        assert canvas._relayout_timer.isActive()
        canvas._relayout_timer.timeout.emit()
        _pump()
        assert solves == [(w0, h0 - 20)]
        assert renders == [(w0, h0 - 20)]
        assert canvas._solved_size == canvas._fig_size()
    finally:
        stack.close()


# --- P16: menu tab opens --------------------------------------------------------

def test_menu_tab_renders_at_its_shown_size_and_tabs_behind_wait(window, monkeypatch):
    # The new tab is shown before it gets the data, so it renders at its real
    # size, without a second layout solve. The tabs behind it are neither
    # re-solved nor rendered, and the Overview is laid out for its current
    # size once it is shown again. The first menu tab is opened beforehand:
    # offscreen, its close button makes the tab bar 2 px taller for good.
    from diive.gui.widgets.mpl_canvas import MplCanvas, _ResizeDeferringCanvas
    window._open_menu_tab("Time series")
    first = window._menu_tab_list[-1]
    first.varpanel.flush_pending()
    _pump()
    renders, solves = [], []
    orig_draw = _ResizeDeferringCanvas.draw
    orig_solve = MplCanvas._solve_layout
    monkeypatch.setattr(
        _ResizeDeferringCanvas, "draw",
        lambda self: (renders.append((self, self.width(), self.height())),
                      orig_draw(self))[1])
    monkeypatch.setattr(
        MplCanvas, "_solve_layout",
        lambda self: (solves.append(self), orig_solve(self))[1])
    overview = window._tabs[0].canvas
    behind = [overview, first.canvas]
    for label in ("Diel cycle", "Histogram"):
        renders.clear(), solves.clear()
        window._open_menu_tab(label)
        tab = window._menu_tab_list[-1]
        tab.varpanel.flush_pending()
        _pump()
        fc = tab.canvas._canvas
        own = [(w, h) for c, w, h in renders if c is fc]
        assert own and set(own) == {(fc.width(), fc.height())}
        assert tab.canvas not in solves
        for canvas in behind:
            assert all(c is not canvas._canvas for c, _, _ in renders)
            assert canvas not in solves
            assert not canvas._relayout_timer.isActive()
        behind.append(tab.canvas)
    window._tabwidget.setCurrentIndex(0)
    _pump()
    if overview._relayout_timer.isActive():
        overview._relayout_timer.timeout.emit()
        _pump()
    assert not overview._relayout_owed
    assert overview._solved_size == overview._fig_size()


def test_tab_rendering_on_push_renders_once_at_its_shown_size(window, monkeypatch):
    # A tab that renders synchronously when it gets the data used to render
    # at the default size of a hidden page, then solve and render again when
    # shown. It now gets the data after it is shown.
    from diive.gui import app as app_module
    from diive.gui.tabs.base import DiiveTab
    from diive.gui.widgets.mpl_canvas import MplCanvas

    class _SyncPlotTab(DiiveTab):
        title = "Sync plot"

        def build(self):
            root = QWidget()
            from PySide6.QtWidgets import QVBoxLayout
            QVBoxLayout(root).addWidget(MplCanvas(show_toolbar=False))
            self.canvas = root.findChild(MplCanvas)
            return root

        def on_data_loaded(self, df, created=None):
            ax = self.canvas.new_axes(1)[0]
            ax.plot(df.iloc[:200, 0].to_numpy())
            self.canvas.draw()

    monkeypatch.setitem(app_module.MENU_TAB_CLASSES, "Sync plot", _SyncPlotTab)
    window._open_menu_tab("Time series")  # the first menu tab settles the tab bar
    window._menu_tab_list[-1].varpanel.flush_pending()
    _pump()

    renders, solves = [], []
    orig_solve = MplCanvas._solve_layout
    monkeypatch.setattr(
        MplCanvas, "_solve_layout",
        lambda self: (solves.append(self), orig_solve(self))[1])
    from diive.gui.widgets.mpl_canvas import _ResizeDeferringCanvas
    orig_draw = _ResizeDeferringCanvas.draw
    monkeypatch.setattr(
        _ResizeDeferringCanvas, "draw",
        lambda self: (renders.append((self, self.width(), self.height())),
                      orig_draw(self))[1])
    window._open_menu_tab("Sync plot")
    _pump()
    tab = window._menu_tab_list[-1]
    fc = tab.canvas._canvas
    own = [(w, h) for c, w, h in renders if c is fc]
    assert own == [(fc.width(), fc.height())]
    assert tab.canvas not in solves


# --- P8: background import --------------------------------------------------------

def test_menu_tab_waits_for_the_background_import(window):
    from diive.gui.app import _PREIMPORT_MODULES
    window.start_preimport()
    thread = window._preimport_thread
    assert thread is not None and thread.name == "diive-preimport"
    window._open_menu_tab("Appearance")
    assert not thread.is_alive()
    assert all(name in sys.modules for name in _PREIMPORT_MODULES)
    assert window._menu_tab_list[-1]._menu_label == "Appearance"


# --- events: visibility toggle --------------------------------------------------

def test_event_visibility_toggle_does_not_rebuild_flags(window, monkeypatch):
    import pandas as pd
    import diive
    from diive.events import Event
    from diive.gui import events
    events.manager.add(Event(name="harvest", start=pd.Timestamp("2021-06-01"),
                             end=pd.Timestamp("2021-06-03")))
    _pump()
    col = "EVENT_harvest"
    assert col in window._full_data.columns
    n_before = int(window._full_data[col].sum())

    calls = []
    orig = diive.events.event_to_flag
    monkeypatch.setattr(diive.events, "event_to_flag",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    events.manager.set_visible(False)
    _pump()
    assert calls == []
    assert not window._show_events_act.isChecked()
    assert col in window._full_data.columns

    # An edit still rebuilds the column.
    events.manager.replace(0, Event(name="harvest", start=pd.Timestamp("2021-06-01"),
                                    end=pd.Timestamp("2021-06-10")))
    _pump()
    assert calls
    assert int(window._full_data[col].sum()) > n_before
