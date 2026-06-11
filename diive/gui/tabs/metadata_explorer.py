"""
GUI.TABS.METADATA_EXPLORER: VARIABLE METADATA EXPLORER
======================================================

"Where does the current version of this variable come from?" Pick a variable on
the left and see its metadata on the right: its origin (original / modified /
derived) and parent variable(s), its tags (editable — add/remove, toggle
favorite), and the ordered provenance timeline of operations that produced it.

The model is the library's :mod:`diive.core.metadata` (held app-wide in
``metadata_store.manager``); this tab only renders it and wires the tag edits —
no domain logic of its own (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import (
    DERIVED,
    FAVORITE,
    MAX_DESCRIPTION_WORDS,
    MODIFIED,
    ORIGINAL,
)
from diive.gui import metadata_store, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.variable_panel import VariablePanel

#: Background tint per origin (matches the "modified/derived/original" wording).
_ORIGIN_COLORS = {
    ORIGINAL: "#90A4AE",   # blue-grey 300 — untouched
    MODIFIED: "#FB8C00",   # orange 600 — transformed copy
    DERIVED: "#8E24AA",    # purple 600 — computed from a parent
}
_MUTED = "#90A4AE"


def _clear_layout(layout) -> None:
    """Remove and delete every widget/sub-layout from `layout`."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


class MetadataExplorerTab(DiiveTab):
    """Inspect and edit per-variable tags + provenance."""

    title = "Metadata explorer"

    def build(self) -> QWidget:
        self._df = None
        self._current: str | None = None
        self._desc_edit: QPlainTextEdit | None = None
        self._desc_save: QPushButton | None = None
        self._desc_var: str | None = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel(clearable=True)
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        self.varpanel.clearRequested.connect(self._clear_one)
        splitter.addWidget(self.varpanel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._detail_host = QWidget()
        self._detail_layout = QVBoxLayout(self._detail_host)
        self._detail_layout.setContentsMargins(16, 16, 16, 16)
        self._detail_layout.setSpacing(10)
        scroll.setWidget(self._detail_host)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        # Footer: a dataset-wide destructive action (confirmed before it runs).
        footer = QHBoxLayout()
        footer.setContentsMargins(8, 4, 8, 4)
        footer.addStretch(1)
        self._clear_btn = QPushButton("Clear all tags & notes…")
        self._clear_btn.setToolTip(
            "Remove every favorite, custom tag, and note for the current dataset. "
            "Auto-assigned tags, origin, and history are kept.")
        self._clear_btn.clicked.connect(self._clear_all)
        footer.addWidget(self._clear_btn)
        outer.addLayout(footer)

        # Keep the detail view live as operations/tags change in other tabs.
        metadata_store.manager.changed.connect(self._refresh_detail)
        self._refresh_detail()
        return root

    def _clear_all(self) -> None:
        """Wipe all user tags + notes for the current dataset, after confirming."""
        if QMessageBox.question(
                self._detail_host, "Clear all tags & notes",
                "Remove every favorite, custom tag, and note for the current "
                "dataset?\n\nAuto-assigned tags, origin, and processing history "
                "are kept.") != QMessageBox.StandardButton.Yes:
            return
        # The `changed` signal rebuilds the detail, which would otherwise flush
        # the open note editor back into the just-cleared store — unbind it first.
        self._desc_var = None
        metadata_store.manager.clear_user_data()

    def _clear_one(self, name: str) -> None:
        """Right-click "Remove all tags & note" for a single variable."""
        # If the cleared variable is the one on screen, unbind its note editor so
        # the rebuild doesn't flush the old text back over the cleared note.
        if name == self._desc_var:
            self._desc_var = None
        metadata_store.manager.clear_variable_user_data(name)

    # --- data flow -----------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        if self._current not in df.columns:
            self._current = str(df.columns[0]) if len(df.columns) else None
        if self._current is not None:
            self.varpanel.set_panels([self._current])
        self._refresh_detail()

    def select_variable(self, name: str) -> None:
        """Public entry point — focus a variable (e.g. from another tab's
        'Edit metadata…' right-click)."""
        self._select(name)

    def _select(self, name: str) -> None:
        if not name:
            return
        self._current = name
        self.varpanel.set_panels([name])
        self._refresh_detail()

    # --- detail view ---------------------------------------------------
    def _refresh_detail(self) -> None:
        # Persist any pending note before the rebuild tears down its editor (a
        # rebuild is triggered by `changed`, e.g. when a tag is added/toggled).
        self._flush_description()
        _clear_layout(self._detail_layout)
        self._desc_edit = None
        self._desc_var = None
        name = self._current
        if name is None:
            self._detail_layout.addWidget(
                QLabel("Select a variable to see its metadata."))
            self._detail_layout.addStretch(1)
            return

        md = metadata_store.manager.store.get(name)

        title = QLabel(str(name))
        f = title.font(); f.setPointSizeF(f.pointSizeF() + 4); f.setBold(True)
        title.setFont(f)
        title.setWordWrap(True)
        self._detail_layout.addWidget(title)

        # Origin badge + parents.
        row = QHBoxLayout()
        badge = QLabel(md.origin.upper())
        color = _ORIGIN_COLORS.get(md.origin, _MUTED)
        badge.setStyleSheet(
            f"background:{color}; color:white; border-radius:6px; "
            f"padding:2px 8px; font-weight:bold;")
        row.addWidget(badge)
        if md.parents:
            row.addWidget(QLabel(f"from  {', '.join(md.parents)}"))
        row.addStretch(1)
        self._detail_layout.addLayout(row)

        self._detail_layout.addWidget(self._tags_section(name, md))
        self._detail_layout.addWidget(self._description_section(name, md))
        self._detail_layout.addWidget(self._provenance_section(md))
        self._detail_layout.addStretch(1)

    def _tags_section(self, name: str, md) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._heading("TAGS"))

        fav = FAVORITE in md.tags
        fav_btn = QPushButton("★ Favorite" if fav else "☆ Favorite")
        fav_btn.setCheckable(True)
        fav_btn.setChecked(fav)
        fav_btn.clicked.connect(
            lambda: metadata_store.manager.toggle_user_tag(name, FAVORITE))
        v.addWidget(fav_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Tag chips: user tags carry a removable "✕"; function tags are shown
        # quietly (not user-removable). The origin tag itself is implicit above.
        chips = QHBoxLayout()
        chips.setSpacing(6)
        shown = sorted(t for t in md.tags if t not in (ORIGINAL, FAVORITE))
        for tag in shown:
            chips.addWidget(self._chip(name, tag, removable=md.is_user_tag(tag)))
        chips.addStretch(1)
        v.addLayout(chips)

        add_row = QHBoxLayout()
        self._tag_input = QLineEdit()
        self._tag_input.setPlaceholderText("add a tag…")
        self._tag_input.returnPressed.connect(lambda: self._add_tag(name))
        add_row.addWidget(self._tag_input)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self._add_tag(name))
        add_row.addWidget(add_btn)
        v.addLayout(add_row)

        # Clear this one variable's user tags + note (only when it has any).
        if md.user_tags() or md.description:
            clear_btn = QPushButton("Clear this variable's tags & note")
            clear_btn.setFlat(True)
            clear_btn.setStyleSheet(f"color:{_MUTED}; text-align:left;")
            clear_btn.clicked.connect(lambda: self._clear_one(name))
            v.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        return box

    def _chip(self, name: str, tag: str, *, removable: bool) -> QWidget:
        # User tags get an auto-assigned colour (stable per tag name); auto
        # (function-set) tags stay a quiet neutral so they recede.
        if removable:
            bg, fg = theme.tag_color(tag)
        else:
            bg, fg = "#ECEFF1", "#37474F"
        chip = QFrame()
        chip.setStyleSheet(f"background:{bg}; border-radius:9px;")
        lay = QHBoxLayout(chip)
        lay.setContentsMargins(8, 2, 6 if removable else 8, 2)
        lay.setSpacing(4)
        label = QLabel(tag)
        label.setStyleSheet(f"color:{fg}; background:transparent;")
        lay.addWidget(label)
        if removable:
            x = QPushButton("✕")
            x.setFlat(True)
            x.setFixedSize(16, 16)
            x.setStyleSheet(f"color:{fg}; border:none; background:transparent;")
            x.clicked.connect(
                lambda: metadata_store.manager.remove_user_tag(name, tag))
            lay.addWidget(x)
        return chip

    def _description_section(self, name: str, md) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.addWidget(self._heading("NOTE"))
        head.addStretch(1)
        self._desc_count = QLabel()
        head.addWidget(self._desc_count)
        v.addLayout(head)

        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlainText(md.description)
        self._desc_edit.setPlaceholderText(
            f"Describe this variable (max {MAX_DESCRIPTION_WORDS} words)…")
        self._desc_edit.setFixedHeight(90)
        # setPlainText above ran before this connect, so it won't fire here:
        # the button starts disabled (no unsaved edits) and enables on real typing.
        self._desc_edit.textChanged.connect(self._on_desc_changed)
        self._desc_var = name
        v.addWidget(self._desc_edit)

        self._desc_save = QPushButton("Save note")
        self._desc_save.setEnabled(False)  # nothing to save until the text changes
        self._desc_save.clicked.connect(self._save_description)
        v.addWidget(self._desc_save, alignment=Qt.AlignmentFlag.AlignLeft)
        self._update_desc_count()
        return box

    def _on_desc_changed(self) -> None:
        """Re-enable the save button (with its default label) on any edit."""
        self._update_desc_count()
        if self._desc_save is not None:
            self._desc_save.setText("Save note")
            self._desc_save.setEnabled(True)

    def _update_desc_count(self) -> None:
        if self._desc_edit is None:
            return
        n = len(self._desc_edit.toPlainText().split())
        over = n > MAX_DESCRIPTION_WORDS
        self._desc_count.setText(f"{n} / {MAX_DESCRIPTION_WORDS} words")
        self._desc_count.setStyleSheet(
            "color:#E53935;" if over else f"color:{_MUTED};")  # red when over cap

    def _flush_description(self) -> None:
        """Persist the editor's text to the store (no `changed` emit, so the
        editor isn't torn down mid-edit). Called before any rebuild."""
        edit, var = self._desc_edit, self._desc_var
        if edit is None or var is None:
            return
        try:
            text = edit.toPlainText()
        except RuntimeError:  # underlying C++ widget already deleted
            return
        metadata_store.manager.store.set_description(var, text)

    def _save_description(self) -> None:
        self._flush_description()
        # Reflect any word-cap truncation back into the editor (this fires
        # textChanged -> _on_desc_changed, which re-enables the button)...
        stored = metadata_store.manager.store.get(self._desc_var).description
        if self._desc_edit is not None and self._desc_edit.toPlainText() != stored:
            self._desc_edit.setPlainText(stored)
        self._update_desc_count()
        # ...so set the saved/disabled state last.
        if self._desc_save is not None:
            self._desc_save.setText("Saved ✓")
            self._desc_save.setEnabled(False)

    def _provenance_section(self, md) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._heading("HISTORY"))
        if not md.provenance:
            note = QLabel("No operations recorded — this is an original variable."
                          if md.origin == ORIGINAL else "No operations recorded.")
            note.setStyleSheet(f"color:{_MUTED};")
            v.addWidget(note)
            return box
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for i, entry in enumerate(md.provenance, start=1):
            parts = [f"{i}.  {entry.describe()}"]
            if entry.parent:
                parts.append(f"from {entry.parent}")
            if entry.timestamp:
                parts.append(entry.timestamp)
            lst.addItem("   ·   ".join(parts))
        lst.setMaximumHeight(180)
        v.addWidget(lst)
        return box

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{theme.manager.tokens.get('MUTED_FG', _MUTED)}; "
            "font-weight:bold; letter-spacing:1px;")
        return label

    def _add_tag(self, name: str) -> None:
        text = self._tag_input.text().strip()
        if not text:
            return
        # add_user_tag fires `changed` -> _refresh_detail rebuilds the detail
        # (and a fresh, empty tag field). Re-focus it so several tags can be
        # typed in quick succession (Enter, type, Enter, ...).
        metadata_store.manager.add_user_tag(name, text)
        self._tag_input.setFocus()
