"""
GUI.TABS.EVENTS: EVENTS MANAGER
===============================

Central place to manage *events* (fertilization, harvest, grazing, management
interventions) and decide whether they show on the plots. Each event is a **card**
— a title, its date settings, a relative-time hint and a mini bar showing where it
sits in the loaded record — and the cards reflow into rows on a soft grey board so
the white cards stand out. The grid can be **filtered**, **grouped** (by category
or year) and shown at two **densities**; each card carries a trashcan delete and a
⋯ menu (show on the Overview, edit, duplicate, shift). A **Manage categories**
dialog defines the category palette; a single **Show events on plots** checkbox is
the master visibility toggle.

The tab edits the app-wide ``events.manager`` (the live event list, the category
palette, and the visibility flag). The main window reacts to ``events.manager``
signals to (re)build each event's 0/1 flag column, redraw overlays, and focus the
Overview — so this tab holds no data or plotting logic itself.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from diive.gui import events as events_store
from diive.gui import icons, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.add_event_dialog import AddEventDialog
from diive.gui.widgets.category_dialog import CategoryDialog
from diive.gui.widgets.flow_layout import FlowLayout
from diive.gui.widgets.menu import studio_menu

#: Board behind the cards — a touch darker than the white cards so they pop.
_BOARD_BG = "#ECECE8"
#: Small per-category glyphs (decorative, stable per name) shown on the pill.
_GLYPHS = ["◆", "●", "■", "▲", "★", "✿", "✚", "◇", "▸", "❖"]


def _contrast(hex_color: str) -> str:
    """Readable text colour for a filled chip of ``hex_color``."""
    return "white" if QColor(hex_color).lightnessF() < 0.55 else "#1A2327"


def _tint(hex_color: str, amount: float) -> str:
    """Blend a colour toward white (``amount`` = fraction of white)."""
    c = QColor(hex_color)
    r = round(c.red() * (1 - amount) + 255 * amount)
    g = round(c.green() * (1 - amount) + 255 * amount)
    b = round(c.blue() * (1 - amount) + 255 * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _glyph(name: str) -> str:
    """A stable decorative glyph for a category name (same name → same glyph)."""
    return _GLYPHS[sum(ord(c) for c in name) % len(_GLYPHS)]


def _fmt_duration(td: pd.Timedelta) -> str:
    """Compact human-readable duration (days/hours)."""
    total_h = td.total_seconds() / 3600.0
    if total_h >= 48:
        return f"{total_h / 24:.1f} d"
    if total_h >= 1:
        return f"{total_h:.1f} h"
    return f"{td.total_seconds() / 60:.0f} min"


def _relative_time(event, now: pd.Timestamp) -> str:
    """A compact 'in 3 mo' / '5 d ago' / 'ongoing' hint relative to ``now``."""
    if event.is_range and event.start <= now <= event.end:
        return "ongoing"
    delta = event.start - now
    past = delta.total_seconds() < 0
    days = abs(delta.total_seconds()) / 86400.0
    if days >= 365:
        n, u = round(days / 365), "yr"
    elif days >= 30:
        n, u = round(days / 30), "mo"
    elif days >= 1:
        n, u = round(days), "d"
    else:
        n, u = round(abs(delta.total_seconds()) / 3600.0), "h"
    if n == 0:
        return "now"
    return f"{n} {u} ago" if past else f"in {n} {u}"


class _SpanBar(QWidget):
    """A thin track showing where the event falls inside the loaded record: a dot
    for an instant, a filled segment for a period."""

    def __init__(self, frac_start: float, frac_end: float | None,
                 color: str) -> None:
        super().__init__()
        self._s = min(max(frac_start, 0.0), 1.0)
        self._e = None if frac_end is None else min(max(frac_end, 0.0), 1.0)
        self._color = color
        self.setFixedHeight(7)
        self.setMinimumWidth(40)
        self.setToolTip("Position within the loaded record")

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        p.setBrush(QColor("#E0E0DB"))
        p.drawRoundedRect(0, 1, w, h - 2, (h - 2) / 2, (h - 2) / 2)
        c = QColor(self._color)
        p.setBrush(c)
        if self._e is None:
            x = self._s * (w - h)
            p.drawEllipse(int(x), 0, h, h)
        else:
            x0 = self._s * w
            x1 = max(self._e * w, x0 + 3)
            p.drawRoundedRect(int(x0), 1, int(x1 - x0), h - 2,
                              (h - 2) / 2, (h - 2) / 2)
        p.end()


class _EventCard(QFrame):
    """One event as a card: category pill, title, date settings, span bar.

    White surface with a soft shadow and a coloured left accent + category pill.
    Double-click edits it; the trashcan deletes it; the ⋯ menu shows it on the
    Overview, duplicates it, or shifts its date. The action callbacks are bound to
    the event's index in the manager list (the tab rebuilds all cards on every
    change, so the index is valid for the card's lifetime)."""

    def __init__(self, event, color: str, span, now, *, compact: bool,
                 on_edit, on_delete, on_focus, on_duplicate, on_shift) -> None:
        super().__init__()
        self._on_edit = on_edit
        self._on_focus = on_focus
        self._on_duplicate = on_duplicate
        self._on_shift = on_shift
        self.setObjectName("eventcard")
        width = 174 if compact else 226
        self.setFixedWidth(width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame#eventcard { background: #FFFFFF; border: 1px solid #E3E6E8;"
            f" border-left: 4px solid {color}; border-radius: 12px; }}"
            f" QFrame#eventcard:hover {{ border: 1px solid {color};"
            f" border-left: 4px solid {color}; background: {_tint(color, 0.96)}; }}"
            # A local stylesheet detaches this card's tooltip from the app-wide
            # QToolTip rule; re-attach the light look explicitly.
            + theme.manager.tooltip_qss())
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(3)
        shadow.setColor(QColor(38, 50, 56, 48))
        self.setGraphicsEffect(shadow)

        lay = QVBoxLayout(self)
        m = (10, 8, 9, 9) if compact else (14, 11, 11, 12)
        lay.setContentsMargins(*m)
        lay.setSpacing(4 if compact else 7)

        # Header: category pill + ⋯ menu + trashcan delete.
        head = QHBoxLayout()
        head.setSpacing(4)
        cat = (event.category or "uncategorized").strip() or "uncategorized"
        pill = QLabel(f"{_glyph(cat)} {theme.manager.label_text(cat)}")
        pf = theme.manager.tracked_font(pill.font())
        pf.setBold(True)
        pf.setPointSizeF(max(6.5, pf.pointSizeF() - 2.0))
        pill.setFont(pf)
        pill.setStyleSheet(
            f"background: {color}; color: {_contrast(color)};"
            f" border-radius: 8px; padding: 2px 8px;")
        head.addWidget(pill)
        head.addStretch(1)
        head.addWidget(self._icon_button(
            icons.locate_icon("#607D8B"), "Show this event on the Overview plot",
            on_focus, hover_bg="#E3F2FD"))
        head.addWidget(self._menu_button())
        head.addWidget(self._delete_button(on_delete))
        lay.addLayout(head)

        # Title (event name).
        name_lbl = QLabel(event.name)
        name_lbl.setWordWrap(True)
        nf = name_lbl.font()
        nf.setBold(True)
        nf.setPointSizeF(nf.pointSizeF() + (1.0 if compact else 2.0))
        name_lbl.setFont(nf)
        name_lbl.setStyleSheet("background: transparent; color: #1F2933;")
        lay.addWidget(name_lbl)

        # Date settings (kind + dates).
        if event.is_range:
            kind = f"Period · {_fmt_duration(event.duration)}"
            dates = (f"{event.start:%Y-%m-%d %H:%M}" if compact
                     else f"{event.start:%Y-%m-%d %H:%M}  →\n"
                          f"{event.end:%Y-%m-%d %H:%M}")
        else:
            kind = "Instant"
            dates = f"{event.start:%Y-%m-%d %H:%M}"
        kind_lbl = QLabel(kind)
        kf = theme.manager.tracked_font(kind_lbl.font())
        kf.setBold(True)
        kf.setPointSizeF(max(6.5, kf.pointSizeF() - 2.0))
        kind_lbl.setFont(kf)
        kind_lbl.setStyleSheet("background: transparent; color: #78909C;")
        lay.addWidget(kind_lbl)

        date_lbl = QLabel(dates)
        dfn = date_lbl.font()
        dfn.setPointSizeF(dfn.pointSizeF() - 0.5)
        date_lbl.setFont(dfn)
        date_lbl.setStyleSheet("background: transparent; color: #455A64;")
        lay.addWidget(date_lbl)

        if compact:
            return  # compact density stops at the essentials

        # Relative-time hint.
        rel_lbl = QLabel(_relative_time(event, now))
        rel_lbl.setFont(kind_lbl.font())
        rel_lbl.setStyleSheet("background: transparent; color: #B0883A;")
        lay.addWidget(rel_lbl)

        # Position-in-record mini bar.
        bar = self._span_bar(event, span, color)
        if bar is not None:
            lay.addWidget(bar)

        # Optional description preview (elided to one line).
        desc = (event.description or "").strip()
        if desc:
            d_lbl = QLabel()
            d_lbl.setFont(date_lbl.font())
            d_lbl.setStyleSheet(
                "background: transparent; color: #90A4AE; font-style: italic;")
            fm = QFontMetrics(d_lbl.font())
            d_lbl.setText(fm.elidedText(desc, Qt.TextElideMode.ElideRight,
                                        width - 28))
            lay.addWidget(d_lbl)
        self.setToolTip((desc + "\n\n" if desc else "")
                        + "Double-click to edit · ⋯ for more")

    # --- header buttons -----------------------------------------------
    @staticmethod
    def _icon_button(icon, tip, on_click, *, hover_bg: str) -> QPushButton:
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setFixedSize(22, 22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip)
        btn.setStyleSheet(
            "QPushButton { background: transparent; border: none;"
            " border-radius: 6px; }"
            f" QPushButton:hover {{ background: {hover_bg}; }}"
            # Local stylesheet detaches the tooltip from the app-wide rule.
            + theme.manager.tooltip_qss())
        btn.clicked.connect(on_click)
        return btn

    def _menu_button(self) -> QPushButton:
        btn = self._icon_button(icons.dots_icon("#90A4AE"), "More actions",
                                lambda: None, hover_bg="#ECEFF1")
        btn.clicked.connect(lambda: self._open_menu(btn))
        return btn

    def _delete_button(self, on_delete) -> QPushButton:
        btn = self._icon_button(icons.trash_icon("#90A4AE"), "Delete event",
                                on_delete, hover_bg="#FFEBEE")
        # Swap to a red glyph on hover via enter/leave (icons can't theme per QSS).
        btn.installEventFilter(self)
        self._del_btn = btn
        return btn

    def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
        from PySide6.QtCore import QEvent
        if obj is getattr(self, "_del_btn", None):
            if event.type() == QEvent.Type.Enter:
                obj.setIcon(icons.trash_icon("#E53935"))
            elif event.type() == QEvent.Type.Leave:
                obj.setIcon(icons.trash_icon("#90A4AE"))
        return super().eventFilter(obj, event)

    def _open_menu(self, anchor) -> None:
        menu = studio_menu(self)
        menu.addAction("Show on Overview", self._on_focus)
        menu.addAction("Edit…", self._on_edit)
        menu.addAction("Duplicate", self._on_duplicate)
        menu.addSeparator()
        menu.addAction("Shift 1 day later",
                       lambda: self._on_shift(pd.Timedelta(days=1)))
        menu.addAction("Shift 1 day earlier",
                       lambda: self._on_shift(pd.Timedelta(days=-1)))
        menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    @staticmethod
    def _span_bar(event, span, color: str):
        """Build the position-in-record bar, or None when there's no usable span."""
        if not span:
            return None
        start, end = span
        total = (end - start).total_seconds()
        if total <= 0:
            return None
        fs = (event.start - start).total_seconds() / total
        fe = ((event.end - start).total_seconds() / total
              if event.is_range else None)
        return _SpanBar(fs, fe, color)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._on_edit()
        super().mouseDoubleClickEvent(event)


class _AddCard(QFrame):
    """A dashed ghost card at the end of the grid that adds a new event."""

    def __init__(self, on_click, *, compact: bool) -> None:
        super().__init__()
        self._cb = on_click
        self.setFixedWidth(174 if compact else 226)
        self.setMinimumHeight(96 if compact else 132)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame { background: transparent; border: 1.5px dashed #BFC8CE;"
            " border-radius: 12px; }"
            " QFrame:hover { border-color: #42A5F5;"
            " background: rgba(66, 165, 245, 0.08); }")
        lay = QVBoxLayout(self)
        lbl = QLabel("＋\nAdd event")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: none; color: #78909C; background: transparent;")
        lay.addWidget(lbl)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._cb()
        super().mousePressEvent(event)


class EventsTab(DiiveTab):
    """Card board of events; filter / group / density + manage categories."""

    title = "Events"

    def build(self) -> QWidget:
        self._df = None
        root = QWidget()
        root.setObjectName("eventsroot")
        root.setStyleSheet(f"QWidget#eventsroot {{ background: {_BOARD_BG}; }}")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QLabel(theme.manager.label_text("Events"))
        tf = theme.manager.tracked_font(title.font())
        tf.setBold(True)
        tf.setPointSizeF(tf.pointSizeF() + 1.0)
        title.setFont(tf)
        title.setStyleSheet("background: transparent;")
        outer.addWidget(title)

        intro = QLabel(
            "Mark when something happened — fertilization, harvest, grazing, a "
            "management step. Each event is a card (stored as a 0/1 column, 1 = the "
            "event took place) and is drawn on the time-series plots.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #5B6B73; background: transparent;")
        outer.addWidget(intro)

        # Toolbar: visibility + grouping + density + manage/add.
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.show_chk = QCheckBox("Show events on plots")
        self.show_chk.setChecked(events_store.manager.visible)
        self.show_chk.toggled.connect(events_store.manager.set_visible)
        self.show_chk.setStyleSheet("background: transparent;")
        bar.addWidget(self.show_chk)
        bar.addStretch(1)
        bar.addWidget(self._toolbar_label("Group"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(["None", "Category", "Year"])
        self.group_combo.currentTextChanged.connect(self._refresh)
        bar.addWidget(self.group_combo)
        bar.addWidget(self._toolbar_label("Density"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["Comfortable", "Compact"])
        self.density_combo.currentTextChanged.connect(self._refresh)
        bar.addWidget(self.density_combo)
        self.cat_btn = QPushButton("Manage categories…")
        self.cat_btn.clicked.connect(self._manage_categories)
        self.add_btn = QPushButton("Add event…")
        self.add_btn.clicked.connect(self._add_event)
        bar.addWidget(self.cat_btn)
        bar.addWidget(self.add_btn)
        outer.addLayout(bar)

        # Filter field.
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter events by name or category…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._refresh)
        outer.addWidget(self.filter_edit)

        # Scrollable board of cards (transparent so the board colour shows behind).
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { background: transparent;"
                                  " border: none; }")
        self.scroll.viewport().setStyleSheet("background: transparent;")
        outer.addWidget(self.scroll, 1)

        events_store.manager.changed.connect(self._refresh)
        events_store.manager.categories_changed.connect(self._refresh)
        self._refresh()
        return root

    @staticmethod
    def _toolbar_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #78909C; background: transparent;")
        return lbl

    # --- project save / restore (inputs only) -------------------------
    def save_state(self) -> dict:
        return {"group": self.group_combo.currentText(),
                "density": self.density_combo.currentText()}

    def restore_state(self, state: dict) -> None:
        g = state.get("group")
        if g:
            self.group_combo.setCurrentText(g)
        d = state.get("density")
        if d:
            self.density_combo.setCurrentText(d)

    # --- data flow -----------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._refresh()  # the span bars depend on the loaded record's range

    def _span(self):
        """The dataset's (start, end) for seeding the pickers, or a sane default."""
        if self._df is not None and len(self._df.index):
            return self._df.index.min(), self._df.index.max()
        now = pd.Timestamp.now().normalize()
        return now, now

    def _data_span(self):
        """The loaded record's (start, end) for the card span bars, or None."""
        if self._df is not None and len(self._df.index):
            return (pd.Timestamp(self._df.index.min()),
                    pd.Timestamp(self._df.index.max()))
        return None

    # --- actions -------------------------------------------------------
    def _add_event(self) -> None:
        start, end = self._span()
        dlg = AddEventDialog(start, end, parent=self.widget())
        if dlg.exec():
            events_store.manager.add(dlg.make_event())

    def _edit(self, index: int) -> None:
        if not (0 <= index < len(events_store.manager.events)):
            return
        start, end = self._span()
        existing = events_store.manager.events[index]
        dlg = AddEventDialog(start, end, event=existing, parent=self.widget())
        if dlg.exec():
            events_store.manager.replace(index, dlg.make_event())

    def _delete(self, index: int) -> None:
        events_store.manager.remove(index)

    def _focus(self, index: int) -> None:
        if 0 <= index < len(events_store.manager.events):
            ev = events_store.manager.events[index]
            events_store.manager.request_focus(ev.start, ev.end)

    def _duplicate(self, index: int) -> None:
        events_store.manager.duplicate(index)

    def _shift(self, index: int, delta) -> None:
        events_store.manager.shift(index, delta)

    def _manage_categories(self) -> None:
        CategoryDialog(self.widget()).exec()

    # --- rendering -----------------------------------------------------
    def _flow_host(self) -> tuple[QWidget, FlowLayout]:
        """A transparent host whose FlowLayout reflows its cards; height-for-width
        enabled so it sizes correctly nested in the vertical board layout."""
        host = QWidget()
        host.setStyleSheet("background: transparent;")
        sp = host.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        host.setSizePolicy(sp)
        flow = FlowLayout(host, margin=2, hspacing=14, vspacing=14)
        return host, flow

    def _section_header(self, title: str, count: int) -> QLabel:
        lbl = QLabel(f"{theme.manager.label_text(title)}  ·  {count}")
        f = theme.manager.tracked_font(lbl.font())
        f.setBold(True)
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1.0))
        lbl.setFont(f)
        lbl.setStyleSheet("color: #607D8B; background: transparent;"
                          " padding-top: 4px;")
        return lbl

    def _sections(self, idxs: list[int]) -> list[tuple[str | None, list[int]]]:
        """Partition date-sorted indices into (title, indices) groups per the
        current Group mode (title None = a single unlabelled section)."""
        evs = events_store.manager.events
        mode = self.group_combo.currentText()
        if mode == "Category":
            buckets: dict[str, list[int]] = {}
            for i in idxs:
                key = (evs[i].category or "").strip() or "uncategorized"
                buckets.setdefault(key, []).append(i)
            order = [c for c in events_store.manager.categories if c in buckets]
            order += sorted(k for k in buckets
                            if k not in events_store.manager.categories
                            and k != "uncategorized")
            if "uncategorized" in buckets:
                order.append("uncategorized")
            return [(k, buckets[k]) for k in order]
        if mode == "Year":
            buckets = {}
            for i in idxs:
                buckets.setdefault(evs[i].start.year, []).append(i)
            return [(str(y), buckets[y]) for y in sorted(buckets)]
        return [(None, idxs)]

    def _refresh(self) -> None:
        # Keep the checkbox in sync if visibility changed elsewhere.
        if self.show_chk.isChecked() != events_store.manager.visible:
            self.show_chk.blockSignals(True)
            self.show_chk.setChecked(events_store.manager.visible)
            self.show_chk.blockSignals(False)

        evs = events_store.manager.events
        cats = events_store.manager.categories
        span = self._data_span()
        now = pd.Timestamp.now()
        compact = self.density_combo.currentText() == "Compact"

        # Date-sorted, then filtered by the search text.
        order = sorted(range(len(evs)), key=lambda i: evs[i].start)
        text = self.filter_edit.text().strip().lower()
        if text:
            order = [i for i in order
                     if text in evs[i].name.lower()
                     or text in (evs[i].category or "").lower()]

        # Build a fresh board (setWidget deletes the previous container + cards).
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        sp = container.sizePolicy()
        sp.setHeightForWidth(True)
        container.setSizePolicy(sp)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        grouped = self.group_combo.currentText() != "None"
        last_flow: FlowLayout | None = None
        for sec_title, idxs in self._sections(order):
            if sec_title is not None:
                vbox.addWidget(self._section_header(sec_title, len(idxs)))
            host, flow = self._flow_host()
            for idx in idxs:
                color = evs[idx].resolved_color(idx, colors=cats)
                flow.addWidget(self._make_card(idx, color, span, now, compact))
            vbox.addWidget(host)
            last_flow = flow

        add_card = _AddCard(self._add_event, compact=compact)
        if grouped or last_flow is None:
            host, flow = self._flow_host()
            flow.addWidget(add_card)
            vbox.addWidget(host)
        else:
            last_flow.addWidget(add_card)
        vbox.addStretch(1)
        self.scroll.setWidget(container)

    def _make_card(self, idx: int, color: str, span, now,
                   compact: bool) -> _EventCard:
        ev = events_store.manager.events[idx]
        return _EventCard(
            ev, color, span, now, compact=compact,
            on_edit=lambda i=idx: self._edit(i),
            on_delete=lambda i=idx: self._delete(i),
            on_focus=lambda i=idx: self._focus(i),
            on_duplicate=lambda i=idx: self._duplicate(i),
            on_shift=lambda d, i=idx: self._shift(i, d))
