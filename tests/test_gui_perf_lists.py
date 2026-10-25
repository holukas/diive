"""
TEST_GUI_PERF_LISTS: variable lists and hover tooltip stay fast and correct
==========================================================================

Covers the speed-ups of GUI_PERFORMANCE.md P13 and P14:

- ``VariablePanel`` re-sequences its rows with one model sort instead of taking
  every row out and adding it back (quadratic, about 0.5 s per filter keystroke
  at 3000 variables). Order, hidden rows and row data must be unchanged.
- ``VariableDelegate`` caches its fonts and parsed colours, but still follows
  theme edits.
- ``HoverAnnotator`` restores and repaints only the old and new tooltip areas,
  skips moves that land on the same sample, and finds the nearest sample of
  an ascending line with a binary search.

Run: pytest tests/test_gui_perf_lists.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import random
import sys
import threading
import traceback

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or worker thread raised (copied from
    tests/test_gui.py: PySide6 swallows those exceptions)."""
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


@pytest.fixture
def clean_metadata():
    from diive.gui import metadata_store
    metadata_store.manager.store.clear_user_data()
    yield metadata_store.manager
    metadata_store.manager.store.clear_user_data()


# --- VariablePanel (P14) ---------------------------------------------------

def _names(n):
    stems = ["NEE_CUT_REF_f", "Tair_f", "VPD_f", "Rg_f", "LE_f", "SWC_1",
             "FCH4", "QC_NEE", "H_f", "PPFD_IN"]
    return [f"{stems[i % len(stems)]}_{i}" for i in range(n)]


def _shown(panel):
    """Names in row order, and the set of hidden names."""
    from diive.gui.widgets.variable_delegate import NAME_ROLE
    rows = [panel.list.item(i) for i in range(panel.list.count())]
    return ([it.data(NAME_ROLE) for it in rows],
            {it.data(NAME_ROLE) for it in rows if it.isHidden()})


def _expected(names, text, favs):
    """Reference order of the fuzzy filter, computed independently."""
    from diive.gui.widgets.variable_panel import _fuzzy_score, _normalize
    needle = _normalize(text)
    order = {n: i for i, n in enumerate(names)}
    if not needle:
        return sorted(names, key=lambda n: (n not in favs, order[n])), set()
    matched, unmatched = [], []
    for n in names:
        score = _fuzzy_score(needle, _normalize(n))
        if score is None:
            unmatched.append(n)
        else:
            matched.append((n not in favs, -score, order[n], n))
    matched.sort(key=lambda t: t[:3])
    return [t[3] for t in matched] + unmatched, set(unmatched)


def test_panel_filter_order_hidden_and_identity(app, clean_metadata):
    from diive.core.metadata import FAVORITE
    from diive.gui.widgets.variable_panel import VariablePanel

    names = _names(400)
    panel = VariablePanel()
    panel.resize(300, 600)
    panel.show()
    panel.set_variables(names, created={names[7]})
    app.processEvents()
    assert panel.list.uniformItemSizes()
    items_before = {id(panel.list.item(i)) for i in range(panel.list.count())}

    rng = random.Random(3)
    favs = set()
    for text in ["", "nee", "tair1", "vpd_f_3", "zzz", "", "f", "rg 2", ""]:
        # Toggle a favourite now and then: favourites sort first in every view.
        fav = rng.choice(names)
        clean_metadata.toggle_user_tag(fav, FAVORITE)
        favs ^= {fav}
        panel.search.setText(text)
        app.processEvents()
        order, hidden = _shown(panel)
        exp_order, exp_hidden = _expected(names, text, favs)
        assert order == exp_order, text
        assert hidden == exp_hidden, text

    # Rows were moved, not rebuilt: the same item objects are still there.
    assert {id(panel.list.item(i))
            for i in range(panel.list.count())} == items_before
    panel.close()


def test_panel_row_data_survives_reorder(app, clean_metadata):
    from diive.core.metadata import FAVORITE
    from diive.gui.widgets.variable_delegate import (
        CREATED_ROLE, LOADING_ROLE, NAME_ROLE, PANEL_ROLE)
    from diive.gui.widgets.variable_panel import VariablePanel

    names = _names(50)
    panel = VariablePanel()
    panel.set_variables(names, created={names[10]})
    panel.set_panels([names[20], names[30]])
    panel.set_loading(names[40])
    clean_metadata.toggle_user_tag(names[30], FAVORITE)  # reorders the list
    panel.search.setText("f")

    rows = {panel.list.item(i).data(NAME_ROLE): panel.list.item(i)
            for i in range(panel.list.count())}
    assert panel.list.item(0).data(NAME_ROLE) == names[30]  # favourite first
    assert rows[names[20]].data(PANEL_ROLE) == 1
    assert rows[names[30]].data(PANEL_ROLE) == 2
    assert rows[names[0]].data(PANEL_ROLE) == 0
    assert rows[names[10]].data(CREATED_ROLE) is True
    assert rows[names[40]].data(LOADING_ROLE) is True
    assert sum(bool(it.data(LOADING_ROLE)) for it in rows.values()) == 1

    panel.clear_loading()
    panel.set_panels([names[5]])
    assert not any(it.data(LOADING_ROLE) for it in rows.values())
    assert rows[names[5]].data(PANEL_ROLE) == 1
    assert rows[names[20]].data(PANEL_ROLE) == 0
    assert rows[names[30]].data(PANEL_ROLE) == 0

    # add / remove keep working with the sorted rows.
    panel.search.clear()
    panel.add_name("ZZ_new", created=True)
    assert panel.names()[-1] == "ZZ_new"
    panel.remove_name(names[3])
    assert names[3] not in panel.names()
    assert len(panel.names()) == len(names)
    panel.close()


def test_panel_filter_keeps_scroll_position_when_order_is_unchanged(app, clean_metadata):
    from diive.gui.widgets.variable_panel import VariablePanel

    panel = VariablePanel()
    panel.resize(300, 400)
    panel.show()
    panel.set_variables(_names(300))
    app.processEvents()
    bar = panel.list.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    value = bar.value()
    assert value > 0
    clean_metadata.notify()  # nothing changed: rows stay where they are
    app.processEvents()
    assert bar.value() == value
    panel.close()


# --- VariableDelegate (P14) ------------------------------------------------

def test_delegate_caches_fonts_and_follows_theme(app, clean_metadata):
    from PySide6.QtGui import QFont
    from diive.gui import theme
    from diive.gui.widgets import variable_delegate as vd
    from diive.gui.widgets.variable_panel import VariablePanel

    panel = VariablePanel()
    panel.resize(300, 300)
    panel.set_variables(["NEE_CUT_REF_f", "Tair_f", "QC_NEE"], created={"Tair_f"})
    panel.set_panels(["NEE_CUT_REF_f"])
    panel.show()
    app.processEvents()
    # Paint the rows (pills, NEW, highlight) with the cached fonts and colours:
    # the highlighted first row carries the primary background colour.
    shot = panel.list.viewport().grab().toImage()
    rect = panel.list.visualItemRect(panel.list.item(0))
    primary = vd._tok("PRIMARY_BG").name()
    assert any(shot.pixelColor(x, rect.center().y()).name() == primary
               for x in range(rect.left() + 4, rect.left() + 40))

    delegate = panel._delegate
    base = QFont("Arial", 10)
    first = delegate._fonts_for(base)
    assert delegate._fonts_for(QFont(base)) is first  # cached for an equal font
    small, small_bold, normal, bold = first
    assert small.pointSizeF() == pytest.approx(9.0) and small_bold.bold()
    assert bold.bold() and not normal.bold()
    bigger = QFont("Arial", 14)
    second = delegate._fonts_for(bigger)
    assert second is not first and second[3].pointSizeF() == pytest.approx(14.0)

    old = theme.manager.tokens["PRIMARY_BG"]
    try:
        theme.manager.tokens["PRIMARY_BG"] = "#123456"
        assert vd._tok("PRIMARY_BG").name() == "#123456"
    finally:
        theme.manager.tokens["PRIMARY_BG"] = old
    assert vd._tok("PRIMARY_BG").name().lower() == old.lower()
    panel.close()


# --- HoverAnnotator (P13) --------------------------------------------------

def test_nearest_ascending_matches_nanargmin():
    from diive.gui.widgets.hover import _nearest_ascending

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        # Integer steps give repeated values and exact midpoint ties.
        x = np.cumsum(rng.integers(0, 3, n)).astype(float)
        for value in np.concatenate([x, x + 0.5, x - 0.5, [x[0] - 10, x[-1] + 10],
                                     rng.uniform(x[0] - 2, x[-1] + 2, 10)]):
            assert _nearest_ascending(x, value) == int(np.nanargmin(np.abs(x - value)))


def _hover_canvas(app, n=20000):
    import pandas as pd
    from diive.gui.widgets.mpl_canvas import MplCanvas

    canvas = MplCanvas()
    canvas.resize(900, 600)
    canvas.show()
    app.processEvents()
    ax = canvas.new_axes(1)[0]
    idx = pd.date_range("2021-01-01", periods=n, freq="30min")
    y = np.sin(np.arange(n) / 3000.0) * 10
    y[100:200] = np.nan
    ax.plot(idx, y)
    canvas.draw()
    app.processEvents()
    return canvas, ax


def _move(canvas, ax, frac_x, frac_y=0.5):
    from matplotlib.backend_bases import MouseEvent
    x0, y0, x1, y1 = ax.bbox.extents
    MouseEvent("motion_notify_event", canvas._canvas,
               x0 + frac_x * (x1 - x0), y0 + frac_y * (y1 - y0))._process()


def test_hover_repaints_only_the_tooltip_area(app, monkeypatch):
    from matplotlib.transforms import Bbox
    canvas, ax = _hover_canvas(app)
    fc = canvas._canvas
    hover = canvas.hover
    clean = np.asarray(fc.buffer_rgba()).copy()
    blits = []
    real_blit = fc.blit
    monkeypatch.setattr(fc, "blit", lambda bbox=None: (blits.append(bbox), real_blit(bbox)))

    fig_area = canvas.fig.bbox.width * canvas.fig.bbox.height
    # A small step repaints one box covering the old and the new tooltip.
    _move(canvas, ax, 0.30)
    first = hover._drawn
    _move(canvas, ax, 0.31)
    assert hover._visible and len(blits) == 2
    for box in (first, hover._drawn):
        visible = Bbox.intersection(box, canvas.fig.bbox)  # blits are clipped
        assert blits[1].contains(*visible.p0) and blits[1].contains(*visible.p1)
    # A jump repaints the old and the new box separately.
    _move(canvas, ax, 0.7)
    assert len(blits) == 4
    for bbox in blits:
        assert bbox is not None and bbox.width * bbox.height < 0.05 * fig_area

    # Same frame as restoring the whole background and drawing the artists.
    partial = np.asarray(fc.buffer_rgba()).copy()
    fc.restore_region(hover._bg)
    ann, marker = hover._annotations[ax], hover._markers[ax]
    ax.draw_artist(marker)
    ax.draw_artist(ann)
    assert np.array_equal(partial, np.asarray(fc.buffer_rgba()))

    # A move onto the same sample changes nothing and repaints nothing.
    _move(canvas, ax, 0.7)
    assert len(blits) == 4

    # Hiding puts back exactly the clean frame.
    hover._hide()
    assert not hover._visible and len(blits) == 5
    assert np.array_equal(np.asarray(fc.buffer_rgba()), clean)
    canvas.close()


def _line_value_reference(ax, x, y, ev):
    """The lookup before P13: a full nanargmin scan, then the neighbour
    nearest in pixels."""
    best = None
    with np.errstate(invalid="ignore"):
        dx = np.abs(x - ev.xdata)
    idx = int(np.nanargmin(dx))
    for i in (idx - 1, idx, idx + 1):
        if not (0 <= i < x.size) or not np.isfinite(y[i]):
            continue
        px, py = ax.transData.transform((x[i], y[i]))
        dist = (px - ev.x) ** 2 + (py - ev.y) ** 2
        if best is None or dist < best[0]:
            best = (dist, x[i], y[i])
    return None if best is None else best[1:]


def test_hover_line_lookup_matches_full_scan_and_follows_new_data(app):
    import types
    canvas, ax = _hover_canvas(app)
    hover = canvas.hover
    line = ax.get_lines()[0]

    def event(xd, yd):
        px, py = ax.transData.transform((xd, yd))
        return types.SimpleNamespace(inaxes=ax, xdata=xd, ydata=yd, x=px, y=py)

    def check(n_events=300):
        x = np.asarray(line.get_xdata(orig=False), float)
        y = np.asarray(line.get_ydata(orig=False), float)
        rng = np.random.default_rng(1)
        lo, hi = np.nanmin(x), np.nanmax(x)
        xds = np.concatenate([rng.uniform(lo - 1, hi + 1, n_events), x[95:205]])
        for xd in xds:
            ev = event(xd, float(rng.uniform(-12, 12)))
            got = hover._line_value(ax, [line], ev)
            ref = _line_value_reference(ax, x, y, ev)
            assert (got is None) == (ref is None)
            if got is not None:
                assert (got[0], got[1]) == ref
        return x, y

    assert hover._line_arrays(line)[2] is True  # ascending: binary search path
    x, y = check()

    # New data on the same line must not be served from the cache.
    line.set_ydata(y + 100.0)
    _, hy, _, _ = hover._line_value(ax, [line], event(x[5000], y[5000] + 100.0))
    assert hy == pytest.approx(y[5000] + 100.0)

    # A line that is not ascending takes the full scan, with the same result.
    line.set_data(x[::-1], y[::-1])
    assert hover._line_arrays(line)[2] is False
    check(100)
    canvas.close()
