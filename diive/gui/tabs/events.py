"""
GUI.TABS.EVENTS: EVENTS MANAGER
===============================

Central place to manage *events* (fertilization, harvest, grazing, management
interventions) and decide whether they show on the plots. A table lists every
event (name, category, type, start, end/duration); buttons add / edit / delete
them; a single **Show events on plots** checkbox is the master visibility toggle.

The tab edits the app-wide ``events.manager`` (the live event list + visibility
flag). The main window reacts to ``events.manager.changed`` to (re)build each
event's 0/1 flag column and the Overview redraws its overlays — so this tab holds
no data or plotting logic itself.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diive.gui import events as events_store
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.add_event_dialog import AddEventDialog


class EventsTab(DiiveTab):
    """List + add/edit/delete events; toggle their visibility on the plots."""

    title = "Events"

    def build(self) -> QWidget:
        self._df = None
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QLabel(theme.manager.label_text("Events"))
        tf = theme.manager.tracked_font(title.font())
        tf.setBold(True)
        tf.setPointSizeF(tf.pointSizeF() + 1.0)
        title.setFont(tf)
        outer.addWidget(title)

        intro = QLabel(
            "Mark when something happened — fertilization, harvest, grazing, a "
            "management step. Each event is stored as a 0/1 column (1 = the event "
            "took place) and drawn on the time-series plots: a line for an instant, "
            "a shaded band for a period.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #607D8B;")
        outer.addWidget(intro)

        # Master visibility toggle.
        self.show_chk = QCheckBox("Show events on plots")
        self.show_chk.setChecked(events_store.manager.visible)
        self.show_chk.toggled.connect(events_store.manager.set_visible)
        outer.addWidget(self.show_chk)

        # Event table.
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Category", "Type", "Start", "End / duration"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(lambda *_: self._edit_selected())
        outer.addWidget(self.table, 1)

        # Buttons.
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add event…")
        self.add_btn.clicked.connect(self._add_event)
        self.edit_btn = QPushButton("Edit…")
        self.edit_btn.clicked.connect(self._edit_selected)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.del_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        events_store.manager.changed.connect(self._refresh)
        self._refresh()
        return root

    # --- data flow -----------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df

    def _span(self):
        """The dataset's (start, end) for seeding the pickers, or a sane default."""
        if self._df is not None and len(self._df.index):
            return self._df.index.min(), self._df.index.max()
        now = pd.Timestamp.now().normalize()
        return now, now

    # --- actions -------------------------------------------------------
    def _add_event(self) -> None:
        start, end = self._span()
        dlg = AddEventDialog(start, end, parent=self.widget())
        if dlg.exec():
            events_store.manager.add(dlg.make_event())

    def _selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _edit_selected(self) -> None:
        row = self._selected_row()
        if row < 0 or row >= len(events_store.manager.events):
            return
        start, end = self._span()
        existing = events_store.manager.events[row]
        dlg = AddEventDialog(start, end, event=existing, parent=self.widget())
        if dlg.exec():
            events_store.manager.replace(row, dlg.make_event())

    def _delete_selected(self) -> None:
        row = self._selected_row()
        if row >= 0:
            events_store.manager.remove(row)

    # --- rendering -----------------------------------------------------
    def _refresh(self) -> None:
        # Keep the checkbox in sync if visibility changed elsewhere.
        if self.show_chk.isChecked() != events_store.manager.visible:
            self.show_chk.blockSignals(True)
            self.show_chk.setChecked(events_store.manager.visible)
            self.show_chk.blockSignals(False)

        evs = events_store.manager.events
        self.table.setRowCount(len(evs))
        for i, ev in enumerate(evs):
            if ev.is_range:
                kind = "Period"
                end_txt = f"{ev.end:%Y-%m-%d %H:%M}  ({_fmt_duration(ev.duration)})"
            else:
                kind = "Instant"
                end_txt = "—"
            cells = [ev.name, ev.category or "—", kind,
                     f"{ev.start:%Y-%m-%d %H:%M}", end_txt]
            for c, text in enumerate(cells):
                self.table.setItem(i, c, QTableWidgetItem(text))
            self.table.item(i, 0).setToolTip(ev.description or ev.name)
            # Colour the category cell to match the colour the event plots in.
            colour = ev.resolved_color(i)
            self.table.item(i, 1).setBackground(QColor(colour))
            self.table.item(i, 1).setForeground(
                Qt.GlobalColor.black if QColor(colour).lightnessF() > 0.55
                else Qt.GlobalColor.white)


def _fmt_duration(td: pd.Timedelta) -> str:
    """Compact human-readable duration (days/hours)."""
    total_h = td.total_seconds() / 3600.0
    if total_h >= 48:
        return f"{total_h / 24:.1f} d"
    if total_h >= 1:
        return f"{total_h:.1f} h"
    return f"{td.total_seconds() / 60:.0f} min"
