"""
TEST_GUI_PERF_THEME: theme changes and event cards restyle only what changed
============================================================================

``app.setStyleSheet`` and ``QWidget.setStyleSheet`` re-polish their widgets even
when the text is the same, and the app-wide call reaches every live widget (P4,
P15 in devnotes/GUI_PERFORMANCE.md). These tests count the ``StyleChange``
events a restyle sends, so they check that a theme change without a stylesheet
change, opening the Appearance tab and adding a sub-tab page restyle nothing,
and that the Events tab reuses its cards instead of rebuilding all of them.

Run: pytest tests/test_gui_perf_theme.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import threading
import traceback

import pytest

pytest.importorskip("PySide6")

import pandas as pd
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QPushButton, QWidget


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
    from diive.gui import theme
    app = QApplication.instance() or QApplication([])
    theme.manager.apply()  # the app-wide stylesheet, as at startup
    yield app


@pytest.fixture
def saved_theme():
    """Restore the theme singleton after a test edits it."""
    from diive.gui import theme
    saved = theme.manager.as_dict()
    yield theme.manager
    theme.manager.reset(silent=True)
    theme.manager.load_dict(saved)
    theme.manager.apply()


class _StyleChanges(QObject):
    """Counts the StyleChange events (one per restyle) sent to widgets."""

    def __init__(self, *widgets):
        super().__init__()
        self.count = 0
        for w in widgets:
            w.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.StyleChange:
            self.count += 1
        return False


class _Counter(QObject):
    count = 0

    def hit(self):
        self.count += 1


def test_theme_apply_restyles_only_on_stylesheet_change(app, saved_theme):
    probe = QPushButton("probe")
    probe.show()
    QApplication.processEvents()
    styles = _StyleChanges(probe)
    counter = _Counter()
    saved_theme.changed.connect(counter.hit)
    try:
        # Not part of the stylesheet: listeners hear `changed`, nothing restyles.
        saved_theme.list_width += 10
        saved_theme.heatmap_cmap = "cividis"
        saved_theme.pills["GPP"][1] = "#000000"
        saved_theme.ts_colors[0] = "#123456"
        saved_theme.apply()
        assert counter.count == 1
        assert styles.count == 0

        # A token is: the app-wide stylesheet changes and reaches the widgets.
        saved_theme.tokens["ACCENT"] = "#010203"
        saved_theme.apply()
        assert counter.count == 2
        assert "#010203" in app.styleSheet()
        assert styles.count >= 1
    finally:
        saved_theme.changed.disconnect(counter.hit)


def test_opening_appearance_does_not_restyle_the_app(app, saved_theme):
    from diive.gui.tabs.settings import SettingsTab

    probe = QPushButton("probe")
    probe.show()
    QApplication.processEvents()
    styles = _StyleChanges(probe)
    counter = _Counter()
    saved_theme.changed.connect(counter.hit)
    before = app.styleSheet()
    try:
        tab = SettingsTab()
        w = tab.widget()
        w.show()
        QApplication.processEvents()
        assert tab.width_spin.value() == saved_theme.list_width
        assert counter.count == 0
        assert styles.count == 0
        assert app.styleSheet() == before
        w.hide()
    finally:
        saved_theme.changed.disconnect(counter.hit)


def test_appearance_swatches_restyle_only_the_changed_one(app, saved_theme):
    from diive.gui.tabs.settings import SettingsTab

    tab = SettingsTab()
    tab.widget()
    pill_kind = next(iter(saved_theme.pills))
    swatch = tab._swatches[0]  # the first pill colour
    other = tab._swatches[1]
    styles_swatch = _StyleChanges(swatch)
    styles_other = _StyleChanges(other)

    saved_theme.apply()  # nothing changed
    assert styles_swatch.count == 0 and styles_other.count == 0

    tab._set_pill(pill_kind, "#0A0B0C")
    assert "#0A0B0C" in swatch.styleSheet()
    assert styles_swatch.count == 1
    assert styles_other.count == 0


def test_subtabs_add_page_styles_only_the_new_pill(app, saved_theme):
    from diive.gui.widgets.sub_tabs import SubTabs

    tabs = SubTabs()
    tabs.add_page("a", QWidget())
    first = tabs._buttons[0]
    styles = _StyleChanges(first)
    tabs.add_page("b", QWidget())
    tabs.add_page("c", QWidget())
    assert styles.count == 0
    # Same look on every pill.
    assert {b.styleSheet() for b in tabs._buttons} == {first.styleSheet()}
    assert saved_theme.tokens["ACCENT"] in first.styleSheet()

    saved_theme.list_width += 10
    saved_theme.apply()  # no token change: the pills keep their style
    assert styles.count == 0

    saved_theme.tokens["ACCENT"] = "#0D0E0F"
    saved_theme.apply()
    assert all("#0D0E0F" in b.styleSheet() for b in tabs._buttons)


@pytest.fixture
def events_tab(app):
    """A shown Events tab on four events. The event list is restored after."""
    from diive.events import Event
    from diive.gui import events as events_store
    from diive.gui.tabs.events import EventsTab

    saved = (list(events_store.manager.events), events_store.manager.visible)
    idx = pd.date_range("2021-01-01 00:15", periods=17520, freq="30min")
    events_store.manager.events = [
        Event(name="Harvest A", start=idx[100], end=idx[200], category="harvest"),
        Event(name="Fertilization", start=idx[3000], category="fertilization"),
        Event(name="Harvest B", start=idx[9000], category="harvest"),
        Event(name="Grazing", start=idx[12000], end=idx[12500], category="grazing"),
    ]
    tab = EventsTab()
    w = tab.widget()
    w.resize(1200, 800)
    w.show()
    tab.on_data_loaded(pd.DataFrame(index=idx))
    QApplication.processEvents()
    yield tab, idx
    w.hide()
    events_store.manager.events, events_store.manager.visible = saved


def _board_cards(tab):
    from diive.gui.tabs.events import _EventCard
    return tab.scroll.widget().findChildren(_EventCard)


def _is_among(card, cards):
    return any(card is c for c in cards)


def test_events_filter_reuses_cards(events_tab):
    tab, _ = events_tab
    cards = _board_cards(tab)
    assert len(cards) == 4

    tab.filter_edit.setText("harv")
    QApplication.processEvents()
    shown = _board_cards(tab)
    assert len(shown) == 2
    assert all(_is_among(c, cards) for c in shown)  # the same cards, not new ones
    assert all(c.isVisible() for c in shown)

    tab.filter_edit.clear()
    QApplication.processEvents()
    back = _board_cards(tab)
    assert len(back) == 4 and all(_is_among(c, cards) for c in back)
    assert all(c.isVisible() for c in back)


def test_events_visibility_toggle_and_same_range_push_keep_the_board(events_tab):
    from diive.gui import events as events_store

    tab, idx = events_tab
    board = tab.scroll.widget()
    events_store.manager.set_visible(not events_store.manager.visible)
    QApplication.processEvents()
    assert tab.show_chk.isChecked() == events_store.manager.visible
    assert tab.scroll.widget() is board

    tab.on_data_loaded(pd.DataFrame(index=idx))  # same range
    QApplication.processEvents()
    assert tab.scroll.widget() is board


def test_events_edit_rebuilds_only_that_card(events_tab):
    from diive.events import Event
    from diive.gui import events as events_store
    from PySide6.QtWidgets import QLabel

    tab, idx = events_tab
    before = _board_cards(tab)
    events_store.manager.replace(
        1, Event(name="Fertilization N", start=idx[3000], category="fertilization"))
    QApplication.processEvents()
    after = _board_cards(tab)
    new = [c for c in after if not _is_among(c, before)]
    assert len(after) == 4 and len(new) == 1
    assert "Fertilization N" in [lbl.text() for lbl in new[0].findChildren(QLabel)]

    # A changed range moves every span bar: all cards are rebuilt.
    tab.on_data_loaded(pd.DataFrame(index=idx[:8000]))
    QApplication.processEvents()
    assert not any(_is_among(c, after) for c in _board_cards(tab))
