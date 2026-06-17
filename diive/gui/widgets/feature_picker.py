"""
GUI.WIDGETS.FEATURE_PICKER: SEARCHABLE MULTI-SELECT COLUMN PICKER
================================================================

A compact, user-friendly replacement for a ctrl-click multi-select list: a
**filter field + a checkable list** with a live selected-count and All / None
buttons that act on the *filtered* set (type ``TA`` → All → checks every TA
column). Checkboxes make the selection explicit and sticky — it survives
filtering and isn't lost on a stray click.

Used by the flux-chain L4.1 page to pick Random-Forest / XGBoost predictor
columns. GUI-only presentation: emits ``changed``; the host reads
:meth:`selected`.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FeaturePicker(QWidget):
    """Filter box + checkable column list with a selected-count and All / None."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None, max_height: int = 150) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        head = QHBoxLayout()
        self._count = QLabel("0 selected")
        self._count.setStyleSheet("color: #6B7780; font-size: 11px;")
        head.addWidget(self._count)
        head.addStretch(1)
        for label, slot in (("All", self._select_all), ("None", self._select_none)):
            b = QPushButton(label)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            head.addWidget(b)
        v.addLayout(head)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("filter features…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        v.addWidget(self._filter)

        self._list = QListWidget()
        self._list.setMaximumHeight(max_height)
        self._list.itemChanged.connect(self._on_item_changed)
        v.addWidget(self._list)

    # --- public API ---
    def set_columns(self, cols, keep_selection: bool = True) -> None:
        """Repopulate with ``cols``; keep currently-checked names checked by default."""
        prev = set(self.selected()) if keep_selection else set()
        self._list.blockSignals(True)
        self._list.clear()
        for c in cols:
            item = QListWidgetItem(str(c))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if str(c) in prev else Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._apply_filter()
        self._update_count()

    def selected(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]

    def set_selected(self, names) -> None:
        wanted = {str(n) for n in (names or [])}
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        self._list.blockSignals(False)
        self._update_count()
        self.changed.emit()

    def count(self) -> int:
        return self._list.count()

    # --- internals ---
    def _select_all(self) -> None:
        self._set_visible_checked(True)

    def _select_none(self) -> None:
        self._set_visible_checked(False)

    def _set_visible_checked(self, on: bool) -> None:
        # Act on the filtered (visible) rows only, so "filter + All" is a fast
        # way to add a whole family of columns.
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._list.blockSignals(False)
        self._update_count()
        self.changed.emit()

    def _apply_filter(self, *_) -> None:
        q = self._filter.text().strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(q) and q not in item.text().lower())

    def _on_item_changed(self, *_) -> None:
        self._update_count()
        self.changed.emit()

    def _update_count(self) -> None:
        self._count.setText(f"{len(self.selected())} selected")
