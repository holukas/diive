"""
GUI.WIDGETS.ADD_EVENT_DIALOG: ADD / EDIT AN EVENT
=================================================

A dialog to define an *event* — something that happened at a point in time or over
a period (fertilization, harvest, grazing, a management intervention). The user
picks a name and category and one of three timing modes:

- **Single date/time** — an instant (one calendar pick).
- **From / To** — an explicit start and end.
- **Start + duration** — a start plus a length (days / hours), end derived.

On accept, :meth:`event` returns a library :class:`diive.events.Event`. The dialog
only collects values; the event model and its 0/1-flag / overlay maths live in the
library.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QDateTime
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from diive.events import CATEGORY_COLORS, Event

_MODE_INSTANT, _MODE_RANGE, _MODE_DURATION = 0, 1, 2


def _to_qdatetime(ts: pd.Timestamp) -> QDateTime:
    return QDateTime.fromString(ts.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss")


def _from_qdatetime(qdt: QDateTime) -> pd.Timestamp:
    return pd.Timestamp(qdt.toString("yyyy-MM-dd HH:mm:ss"))


class AddEventDialog(QDialog):
    """Collect an event definition.

    Args:
        data_start / data_end: The dataset span; seeds and clamps the pickers so an
            event lands within the data.
        event: An existing event to edit (the dialog pre-fills from it); ``None``
            for a new event.
    """

    def __init__(self, data_start: pd.Timestamp, data_end: pd.Timestamp,
                 event: Event | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit event" if event else "Add event")
        self.setMinimumWidth(380)
        self._data_start = pd.Timestamp(data_start)
        self._data_end = pd.Timestamp(data_end)
        self._color: str | None = event.color if event else None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Data range: {self._data_start:%Y-%m-%d %H:%M} "
            f"to {self._data_end:%Y-%m-%d %H:%M}"))

        form = QFormLayout()
        self.name_edit = QLineEdit(event.name if event else "")
        self.name_edit.setPlaceholderText("e.g. Fertilization")
        form.addRow("Name", self.name_edit)

        self.category_edit = QComboBox()
        self.category_edit.setEditable(True)
        self.category_edit.addItems([""] + sorted(CATEGORY_COLORS))
        if event and event.category:
            self.category_edit.setCurrentText(event.category)
        self.category_edit.currentTextChanged.connect(self._sync_color_swatch)
        form.addRow("Category", self.category_edit)
        layout.addLayout(form)

        # --- timing mode ---
        mode_row = QHBoxLayout()
        self.rb_instant = QRadioButton("Single date/time")
        self.rb_range = QRadioButton("From / To")
        self.rb_duration = QRadioButton("Start + duration")
        for rb in (self.rb_instant, self.rb_range, self.rb_duration):
            mode_row.addWidget(rb)
        layout.addLayout(mode_row)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_instant_page())
        self.stack.addWidget(self._build_range_page())
        self.stack.addWidget(self._build_duration_page())
        layout.addWidget(self.stack)
        self.rb_instant.toggled.connect(lambda on: on and self.stack.setCurrentIndex(0))
        self.rb_range.toggled.connect(lambda on: on and self.stack.setCurrentIndex(1))
        self.rb_duration.toggled.connect(lambda on: on and self.stack.setCurrentIndex(2))

        # --- colour ---
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Colour"))
        self.color_btn = QPushButton("Auto (by category)")
        self.color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_btn, 1)
        self.reset_color_btn = QPushButton("Auto")
        self.reset_color_btn.clicked.connect(self._reset_color)
        color_row.addWidget(self.reset_color_btn)
        layout.addLayout(color_row)

        # --- description ---
        self.desc_edit = QLineEdit(event.description if event else "")
        self.desc_edit.setPlaceholderText("Optional note")
        dform = QFormLayout()
        dform.addRow("Description", self.desc_edit)
        layout.addLayout(dform)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._seed_mode(event)
        self._sync_color_swatch()

    # --- timing pages -------------------------------------------------
    def _make_edit(self, value: pd.Timestamp) -> QDateTimeEdit:
        edit = QDateTimeEdit()
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        edit.setMinimumDateTime(_to_qdatetime(self._data_start))
        edit.setMaximumDateTime(_to_qdatetime(self._data_end))
        edit.setDateTime(_to_qdatetime(value))
        return edit

    def _build_instant_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.instant_edit = self._make_edit(self._data_start)
        form.addRow("Date / time", self.instant_edit)
        return page

    def _build_range_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.from_edit = self._make_edit(self._data_start)
        self.to_edit = self._make_edit(self._data_start)
        form.addRow("From", self.from_edit)
        form.addRow("To", self.to_edit)
        return page

    def _build_duration_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.start_edit = self._make_edit(self._data_start)
        dur_row = QHBoxLayout()
        self.dur_value = QDoubleSpinBox()
        self.dur_value.setRange(0.0, 100000.0)
        self.dur_value.setValue(1.0)
        self.dur_value.setDecimals(2)
        self.dur_unit = QComboBox()
        self.dur_unit.addItems(["days", "hours"])
        dur_row.addWidget(self.dur_value, 1)
        dur_row.addWidget(self.dur_unit)
        dur_wrap = QWidget()
        dur_wrap.setLayout(dur_row)
        form.addRow("Start", self.start_edit)
        form.addRow("Duration", dur_wrap)
        return page

    def _seed_mode(self, event: Event | None) -> None:
        """Pick the initial timing mode (and pre-fill it) from an edited event."""
        if event is None:
            self.rb_instant.setChecked(True)
            return
        if event.is_range:
            self.rb_range.setChecked(True)
            self.from_edit.setDateTime(_to_qdatetime(event.start))
            self.to_edit.setDateTime(_to_qdatetime(event.end))
        else:
            self.rb_instant.setChecked(True)
            self.instant_edit.setDateTime(_to_qdatetime(event.start))

    # --- colour -------------------------------------------------------
    def _pick_color(self) -> None:
        initial = QColor(self._color) if self._color else QColor("#42A5F5")
        chosen = QColorDialog.getColor(initial, self, "Event colour")
        if chosen.isValid():
            self._color = chosen.name()
            self._sync_color_swatch()

    def _reset_color(self) -> None:
        self._color = None
        self._sync_color_swatch()

    def _sync_color_swatch(self, *_args) -> None:
        """Reflect the effective colour on the button (explicit, else by category)."""
        if self._color:
            shown, label = self._color, self._color
        else:
            cat = self.category_edit.currentText().strip().lower()
            shown = CATEGORY_COLORS.get(cat, "#90A4AE")
            label = "Auto (by category)"
        text = "white" if QColor(shown).lightnessF() < 0.55 else "black"
        self.color_btn.setText(label)
        self.color_btn.setStyleSheet(
            f"QPushButton {{ background: {shown}; color: {text};"
            f" border: 1px solid #B0BEC5; border-radius: 4px; padding: 4px; }}")

    # --- result -------------------------------------------------------
    def _on_accept(self) -> None:
        if not self.name_edit.text().strip():
            self.name_edit.setFocus()
            return  # name is required (it labels the plot + the flag column)
        if self.stack.currentIndex() == _MODE_RANGE \
                and self.from_edit.dateTime() > self.to_edit.dateTime():
            self.from_edit, self.to_edit = self.to_edit, self.from_edit
        self.accept()

    def make_event(self) -> Event:
        """The defined :class:`diive.events.Event` (call after the dialog is accepted).

        Named ``make_event`` (not ``event``) to avoid overriding ``QDialog.event``.
        """
        name = self.name_edit.text().strip()
        category = self.category_edit.currentText().strip()
        description = self.desc_edit.text().strip()
        mode = self.stack.currentIndex()
        if mode == _MODE_INSTANT:
            start = _from_qdatetime(self.instant_edit.dateTime())
            end = None
        elif mode == _MODE_RANGE:
            start = _from_qdatetime(self.from_edit.dateTime())
            end = _from_qdatetime(self.to_edit.dateTime())
        else:  # start + duration
            start = _from_qdatetime(self.start_edit.dateTime())
            unit = "D" if self.dur_unit.currentText() == "days" else "h"
            end = start + pd.Timedelta(self.dur_value.value(), unit=unit)
        return Event(name=name, start=start, end=end, category=category,
                     description=description, color=self._color)
