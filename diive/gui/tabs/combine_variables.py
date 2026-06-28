"""
GUI.TABS.COMBINE_VARIABLES: COMBINE TWO VARIABLES INTO A NEW ONE
================================================================

Drag a variable from the shared list onto **heatmap 1** and another onto
**heatmap 2**, pick how to combine them (multiply / add / subtract / divide,
optionally only where both have data), and **heatmap 3** previews the resulting
variable. Name it and add it to the dataset.

Three independent date/time heatmaps sit side by side: the two sources are drop
targets (heatmap 3 is read-only output). All combination maths is the library's
:func:`diive.variables.combine_variables`; this tab only collects the two source
variables + the method, previews the heatmaps, and emits the new column with
DERIVED provenance (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle

_C_MUTED = "#6B7780"

#: Heatmap chrome font sizes, matching the Overview tab's compact panels.
_TITLE_FONTSIZE = 10
_FONT_SIZE = 9

#: Combine methods offered in the dropdown: (library method key, menu label,
#: status phrase template using {a}/{b} placeholders for the two variables).
_METHODS = [
    ("multiply", "Multiply  (a × b)", "{a} × {b}"),
    ("add", "Add  (a + b)", "{a} + {b}"),
    ("subtract", "Subtract  (a − b)", "{a} − {b}"),
    ("divide", "Divide  (a ÷ b)", "{a} ÷ {b}"),
    ("fillgaps", "Fill gaps of a with b", "gaps of {a} filled with {b}"),
]


class _HeatmapSlot(QFrame):
    """A titled date/time-heatmap panel; optionally a drop target for a variable.

    Source slots (``accepts_drop=True``) accept a variable name dragged from the
    list and emit :attr:`dropped`. The output slot is read-only. Presentation
    only — the owning tab decides what to plot.
    """

    dropped = Signal(str)

    def __init__(self, title: str, accepts_drop: bool = False) -> None:
        super().__init__()
        self._title = title
        self._accepts = accepts_drop
        self.setObjectName("heatmapslot")
        self.setFrameShape(QFrame.Shape.NoFrame)
        if accepts_drop:
            self.setAcceptDrops(True)
        self._apply_style(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self.header = QLabel(self._header_text(None))
        self.header.setWordWrap(True)
        self.header.setStyleSheet("background: transparent;")
        lay.addWidget(self.header)

        self.canvas = MplCanvas()
        lay.addWidget(self.canvas, stretch=1)
        self._show_placeholder()

    # --- presentation --------------------------------------------------
    def _header_text(self, var: str | None) -> str:
        if var:
            return f"<b>{self._title}</b> <span style='color:{_C_MUTED}'>· {var}</span>"
        hint = "drag a variable here" if self._accepts else "combined result"
        return f"<b>{self._title}</b> <span style='color:{_C_MUTED}'>({hint})</span>"

    def set_header(self, var: str | None) -> None:
        self.header.setText(self._header_text(var))

    def _apply_style(self, highlight: bool) -> None:
        border = theme.manager.tokens.get("ACCENT", "#3A4D5C") if highlight \
            else theme.manager.tokens.get("BORDER", "#E6E6E3")
        width = 2 if highlight else 1
        style = "dashed" if self._accepts else "solid"
        self.setStyleSheet(
            f"QFrame#heatmapslot {{ border: {width}px {style} {border}; "
            f"border-radius: 8px; }}")

    def _show_placeholder(self) -> None:
        ax = self.canvas.new_axes(1)[0]
        msg = ("Drag a variable from the list\nonto this panel"
               if self._accepts else "Combined variable\nappears here")
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=_C_MUTED,
                transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        self.canvas.draw()

    def plot_series(self, series: pd.Series) -> None:
        """Render ``series`` as a date/time heatmap, or a message on failure."""
        ax = self.canvas.new_axes(1)[0]
        try:
            # Compact chrome (matching the Overview panels) and an auto-precision
            # colorbar so labels carry only as many decimals as the range needs.
            dv.plotting.HeatmapDateTime(series).plot(
                ax=ax, fig=self.canvas.fig,
                format_style=dv.plotting.FormatStyle(
                    title_fontsize=_TITLE_FONTSIZE, axlabel_fontsize=_FONT_SIZE,
                    ticks_fontsize=_FONT_SIZE),
                cb_digits_after_comma="auto", cb_labelsize=_FONT_SIZE)
        except Exception as err:  # non-datetime index, all-NaN, ... — show why
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes, fontsize=9)
            ax.axis("off")
        self.canvas.draw()
        self.canvas.reset_history()

    def clear_plot(self) -> None:
        self._show_placeholder()

    # --- drag & drop ---------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if self._accepts and event.mimeData().hasText():
            event.acceptProposedAction()
            self._apply_style(True)
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event) -> None:
        self._apply_style(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self._apply_style(False)
        if self._accepts and event.mimeData().hasText():
            self.dropped.emit(event.mimeData().text())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class _Signals(QObject):
    """Qt signal host (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)


class CombineVariablesTab(DiiveTab):
    """Combine two dragged-in variables into a new one, previewed as heatmaps."""

    title = "Combine variables"
    intro = ("Drag a variable onto heatmap 1 and another onto heatmap 2, choose "
             "how to combine them, then add the result (heatmap 3) as a new "
             "variable.")

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        #: Assigned source variables, by slot number (1, 2).
        self._vars: dict[int, str | None] = {1: None, 2: None}
        #: The most recent combined series (None until both sources are set).
        self._combined: pd.Series | None = None
        #: The last auto-suggested name, so user edits aren't overwritten.
        self._auto_name = ""

        self._sig = _Signals()
        #: Exposed bound signal the main window connects to (merges the column).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive snippet reproducing this combination.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        # "Add to dataset" sits left-aligned in its own row just below the header
        # (mirrors the plotting tab's Update button). Its label carries the new
        # variable name so it's clear what gets added.
        self.add_btn = QPushButton("Add to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.setToolTip("Append the combined variable as a new column.")
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        action_row = QHBoxLayout()
        action_row.setContentsMargins(10, 0, 10, 4)
        action_row.addWidget(self.add_btn)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: the shared variable list (draggable — drag onto a heatmap).
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.addWidget(list_header("Variable", "drag onto heatmap 1 or 2"))
        self.varpanel = VariablePanel(draggable=True)
        self.varpanel.list.setToolTip(
            "Drag a variable onto heatmap 1 or 2 to set the operands.")
        llay.addWidget(self.varpanel, stretch=1)
        splitter.addWidget(left)

        # Right: controls strip + the three heatmaps side by side.
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(4)
        rlay.addLayout(self._build_controls())
        self.status = QLabel(self.intro)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {_C_MUTED}; padding: 0 2px;")
        rlay.addWidget(self.status)

        heatmaps = QHBoxLayout()
        heatmaps.setSpacing(8)
        self.slot1 = _HeatmapSlot("Heatmap 1", accepts_drop=True)
        self.slot2 = _HeatmapSlot("Heatmap 2", accepts_drop=True)
        self.slot3 = _HeatmapSlot("Heatmap 3", accepts_drop=False)
        self.slot1.dropped.connect(lambda name: self._assign(1, name))
        self.slot2.dropped.connect(lambda name: self._assign(2, name))
        for slot in (self.slot1, self.slot2, self.slot3):
            heatmaps.addWidget(slot, stretch=1)
        rlay.addLayout(heatmaps, stretch=1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lock_panel_handle(splitter)
        layout.addWidget(splitter)
        outer.addWidget(body, stretch=1)
        return root

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(8)

        row.addWidget(QLabel("Combine:"))
        self.method = QComboBox()
        for key, label, _phrase in _METHODS:
            self.method.addItem(label, key)
        self.method.setToolTip("How to combine heatmap 1 (a) and heatmap 2 (b).")
        self.method.currentIndexChanged.connect(self._recombine)
        row.addWidget(self.method)

        self.overlap_cb = QCheckBox("Keep overlapping data points only")
        self.overlap_cb.setChecked(True)
        self.overlap_cb.setToolTip(
            "Checked: keep a result only where BOTH variables have a value. "
            "Unchecked: a missing value is treated as the operation's identity "
            "(0 for add/subtract, 1 for multiply/divide), so one-sided records "
            "survive.")
        self.overlap_cb.toggled.connect(self._recombine)
        row.addWidget(self.overlap_cb)

        row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("new variable name")
        self.name_edit.setToolTip("Name for the combined variable.")
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.name_edit.setMaximumWidth(260)
        row.addWidget(self.name_edit)

        row.addStretch(1)
        return row

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        cols = set(str(c) for c in df.columns)
        # Drop any assigned source that no longer exists, then re-render.
        for slot_no in (1, 2):
            var = self._vars[slot_no]
            if var is not None and var not in cols:
                self._vars[slot_no] = None
                self._slot(slot_no).clear_plot()
                self._slot(slot_no).set_header(None)
            elif var is not None:
                self._slot(slot_no).plot_series(df[var])
        self._recombine()

    def _slot(self, slot_no: int) -> _HeatmapSlot:
        return self.slot1 if slot_no == 1 else self.slot2

    # --- assignment ----------------------------------------------------
    def _assign(self, slot_no: int, name: str) -> None:
        if self._df is None or name not in self._df.columns:
            return
        self._vars[slot_no] = name
        slot = self._slot(slot_no)
        slot.set_header(name)
        slot.plot_series(self._df[name])
        self._mark_selected()
        self._recombine()

    def _mark_selected(self) -> None:
        """Number the assigned variables in the list (1 = heatmap 1, 2 = heatmap 2)."""
        panels = [v for v in (self._vars[1], self._vars[2]) if v]
        self.varpanel.set_panels(panels)

    # --- combine -------------------------------------------------------
    def _recombine(self, *_args) -> None:
        """Recompute and preview the combined variable when both sources are set."""
        v1, v2 = self._vars[1], self._vars[2]
        method = self.method.currentData()
        # "Keep overlapping only" is meaningless for gap-filling (always a union).
        self.overlap_cb.setEnabled(method != "fillgaps")
        if self._df is None or not v1 or not v2:
            self._combined = None
            self.slot3.clear_plot()
            self.slot3.set_header(None)
            self.add_btn.setEnabled(False)
            self._update_add_label()
            if self._df is not None and (v1 or v2):
                self.status.setText("Assign a variable to both heatmap 1 and 2.")
            return

        keep_overlap = self.overlap_cb.isChecked()
        self._refresh_auto_name(v1, v2, method)
        try:
            combined = dv.variables.combine_variables(
                self._df[v1], self._df[v2], method=method,
                keep_overlap_only=keep_overlap, name=self._out_name())
        except Exception as err:
            self._combined = None
            self.add_btn.setEnabled(False)
            self.status.setText(f"Cannot combine: {err}")
            return

        self._combined = combined
        self.slot3.set_header(self._out_name())
        self.slot3.plot_series(combined)
        n = int(combined.notna().sum())
        phrase = {k: p for k, _label, p in _METHODS}[method].format(a=v1, b=v2)
        self.status.setText(
            f"{phrase} -> {n:,} values. Name it and add it to the dataset.")
        self.add_btn.setEnabled(n > 0 and bool(self._out_name()))
        self._update_add_label()

    def _refresh_auto_name(self, v1: str, v2: str, method: str) -> None:
        """Update the suggested name, unless the user typed their own."""
        suggested = f"{v1}_{method.upper()}_{v2}".replace(".", "_")
        current = self.name_edit.text().strip()
        if current == "" or current == self._auto_name:
            self.name_edit.blockSignals(True)
            self.name_edit.setText(suggested)
            self.name_edit.blockSignals(False)
        self._auto_name = suggested

    def _on_name_changed(self, _text: str) -> None:
        # Live-update the output header + add-button label/enablement to the new name.
        self._update_add_label()
        if self._combined is not None:
            name = self._out_name()
            self._combined.name = name
            self.slot3.set_header(name or None)
            self.add_btn.setEnabled(int(self._combined.notna().sum()) > 0 and bool(name))

    def _update_add_label(self) -> None:
        """Label the Add button with the new variable name (or a generic fallback)."""
        name = self._out_name()
        self.add_btn.setText(f"Add {name} to dataset" if name else "Add to dataset")

    def _out_name(self) -> str:
        return self.name_edit.text().strip()

    def _python_code(self) -> str | None:
        """Runnable snippet reproducing the current combination (Copy Python).

        Returns None (button no-op) until both sources are assigned."""
        v1, v2 = self._vars[1], self._vars[2]
        if not v1 or not v2:
            self.status.setText("Assign a variable to both heatmaps first to copy the code.")
            return None
        return dv.variables.combine_variables_to_code(
            str(v1), str(v2), method=self.method.currentData(),
            keep_overlap_only=self.overlap_cb.isChecked(),
            name=self._out_name() or None)

    # --- emit ----------------------------------------------------------
    def _add(self) -> None:
        if self._combined is None or not self._out_name():
            return
        name = self._out_name()
        out = self._combined.copy()
        out.name = name
        result = pd.DataFrame({name: out})
        result.attrs[ATTRS_KEY] = {
            name: provenance_attr(
                origin=DERIVED, parent=str(self._vars[1]), operation=self.title,
                params={"other": str(self._vars[2]),
                        "method": self.method.currentData(),
                        "keep_overlap_only": self.overlap_cb.isChecked()},
                tags=["combined"]),
        }
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {name} to the variable list.")
        self.add_btn.setEnabled(False)

    # --- state ---------------------------------------------------------
    def save_state(self) -> dict:
        return {"var1": self._vars[1], "var2": self._vars[2],
                "method": self.method.currentData(),
                "keep_overlap": self.overlap_cb.isChecked(),
                "name": self.name_edit.text()}

    def restore_state(self, state: dict) -> None:
        idx = self.method.findData(state.get("method"))
        if idx >= 0:
            self.method.setCurrentIndex(idx)
        self.overlap_cb.setChecked(bool(state.get("keep_overlap", True)))
        for slot_no, key in ((1, "var1"), (2, "var2")):
            var = state.get(key)
            if var and self._df is not None and var in self._df.columns:
                self._assign(slot_no, var)
        name = state.get("name")
        if name:
            self.name_edit.setText(name)
