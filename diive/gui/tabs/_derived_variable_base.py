"""
GUI.TABS._DERIVED_VARIABLE_BASE: SHARED BASE FOR DERIVED-VARIABLE TABS
=====================================================================

Common machinery for tabs that compute a single **derived variable** from a
handful of existing columns via one library function — the thermodynamic /
radiation family (VPD from air temperature + relative humidity, evapotranspiration
from latent heat, dry-air density, ...). Each such calculation is a fast,
closed-form formula.

Layout matches the other result-producing tabs: a draggable variable list on the
left, a **Settings** column in the middle (input fields you can drag a variable
onto — or pick from the dropdown — plus an output name and a **Calculate**
button), and a preview column on the right that stacks one date/time heatmap per
input plus one for the result. Nothing is computed until **Calculate** is
pressed; the input heatmaps update as soon as a field is set.

A concrete tab subclasses :class:`BaseDerivedVariableTab` and supplies only the
method-specific bits: which input columns it needs (:attr:`inputs`), how to call
the library function (:meth:`_compute`), and how to render the reproducible
snippet (:meth:`_code`). All maths is library work (``dv.variables.*``); this
base only collects inputs, previews, and emits the new column with DERIVED
provenance (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.column_picker import ColumnPicker, NONE_ITEM
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel

_C_MUTED = "#6B7780"
#: Compact heatmap chrome, matching the Overview / Combine-variables panels.
_TITLE_FONTSIZE = 10
_FONT_SIZE = 9


class _HeatmapPanel(QFrame):
    """A titled, display-only date/time-heatmap panel (one input or the result).

    Presentation only — the owning tab decides what to plot. Mirrors the
    Combine-variables slot look without the drop target (dropping targets the
    input fields now, not the preview)."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self.setObjectName("heatmapslot")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QFrame#heatmapslot { border: 1px solid "
            + theme.manager.tokens.get("BORDER", "#E6E6E3")
            + "; border-radius: 8px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setStyleSheet("background: transparent;")
        lay.addWidget(self.header)
        self.canvas = MplCanvas()
        lay.addWidget(self.canvas, stretch=1)
        self.set_header(None)
        self._show_placeholder()

    def set_header(self, var: str | None) -> None:
        if var:
            self.header.setText(
                f"<b>{self._title}</b> <span style='color:{_C_MUTED}'>· {var}</span>")
        else:
            self.header.setText(
                f"<b>{self._title}</b> <span style='color:{_C_MUTED}'>(not set)</span>")

    def _show_placeholder(self, msg: str = "") -> None:
        ax = self.canvas.new_axes(1)[0]
        ax.text(0.5, 0.5, msg or "preview appears here", ha="center",
                va="center", color=_C_MUTED, transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        self.canvas.draw()

    def clear_plot(self, msg: str = "") -> None:
        self._show_placeholder(msg)

    def plot_series(self, series: pd.Series) -> None:
        """Render ``series`` as a date/time heatmap, or a message on failure."""
        ax = self.canvas.new_axes(1)[0]
        try:
            dv.plotting.HeatmapDateTime(series).plot(
                ax=ax, fig=self.canvas.fig,
                format_style=dv.plotting.FormatStyle(
                    title_fontsize=_TITLE_FONTSIZE, axlabel_fontsize=_FONT_SIZE,
                    ticks_fontsize=_FONT_SIZE),
                cb_digits_after_comma="auto", cb_labelsize=_FONT_SIZE)
        except Exception as err:  # non-datetime index, all-NaN, ...
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes, fontsize=9)
            ax.axis("off")
        self.canvas.draw()
        self.canvas.reset_history()


class _Signals(QObject):
    """Qt signal host (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)


class BaseDerivedVariableTab(DiiveTab):
    """Shared base for single-formula derived-variable tabs.

    Subclasses set the class attributes below and implement :meth:`_compute`
    and :meth:`_code`."""

    #: Tab title (DiiveTab requirement) — set by the subclass.
    title = "Derived variable"
    #: One-line description shown at the top of the settings column.
    intro = "Calculate a derived variable from existing columns."
    #: Input columns the formula needs — :class:`ColumnPicker` specs (see that
    #: widget's docstring: keys ``key`` / ``label`` / ``needle`` / opt. ``tip``).
    #: Each input also gets its own preview heatmap; ``short`` (optional) titles
    #: that heatmap (defaults to ``label``).
    inputs: list[dict] = []
    #: Suggested output name, seeded into the name field (user-editable).
    default_name = "derived"
    #: Physical unit of the result, shown in the status line.
    out_unit = ""
    #: Provenance tags attached to the emitted column.
    method_tags: list[str] = ["derived"]

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        #: The most recent computed series (None until Calculate succeeds).
        self._result: pd.Series | None = None

        self._sig = _Signals()
        #: Exposed bound signal the main window connects to (merges the column).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive snippet reproducing this calculation.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        # Body: variable list | settings | stacked preview heatmaps.
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)

        # Left: the shared variable list (draggable — drag onto an input field).
        left = QVBoxLayout()
        left.addWidget(list_header("Variable", "drag onto a field"))
        self.varpanel = VariablePanel(draggable=True)
        self.varpanel.list.setToolTip(
            "Drag a variable onto an input field in the Settings column.")
        left.addWidget(self.varpanel, stretch=1)
        layout.addLayout(left)

        # Middle: settings (inputs + name + Calculate/Add).
        mid = self._build_settings()
        mid.setFixedWidth(300)
        layout.addWidget(mid)

        # Right: status on top, then the heatmaps in a row (one per input, then
        # the result) — left to right.
        right = QVBoxLayout()
        right.setSpacing(6)
        self.status = QLabel(self.intro)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {_C_MUTED}; padding: 0 2px;")
        right.addWidget(self.status)
        maps = QHBoxLayout()
        maps.setSpacing(8)
        self._input_panels: dict[str, _HeatmapPanel] = {}
        for spec in self.inputs:
            panel = _HeatmapPanel(spec.get("short", spec["label"]))
            self._input_panels[spec["key"]] = panel
            maps.addWidget(panel, stretch=1)
        self._result_panel = _HeatmapPanel(self.default_name)
        maps.addWidget(self._result_panel, stretch=1)
        right.addLayout(maps, stretch=1)
        layout.addLayout(right, stretch=1)

        outer.addWidget(body, stretch=1)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        v.addWidget(list_header("Settings", "set inputs & calculate"))

        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        self.picker = ColumnPicker(self.inputs, title="Input columns")
        self.picker.changed.connect(self._on_field_changed)
        v.addWidget(self.picker)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self.default_name)
        self.name_edit.setPlaceholderText("new variable name")
        self.name_edit.setToolTip("Name for the derived variable.")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit, stretch=1)
        v.addLayout(name_row)

        # Let a subclass add any extra widgets (rare — e.g. a unit toggle).
        self._add_extra_controls(v)

        # Calculate + Add, anchored below the settings (like the other tabs).
        self.calc_btn = QPushButton("Calculate")
        theme.set_button_role(self.calc_btn, "confirm")
        self.calc_btn.setToolTip("Compute the derived variable from the inputs.")
        self.calc_btn.clicked.connect(self._calculate)
        v.addWidget(self.calc_btn)

        self.add_btn = QPushButton(f"Add {self.default_name} to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.setToolTip("Append the derived variable as a new column.")
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        v.addWidget(self.add_btn)

        v.addStretch(1)
        return panel

    def _add_extra_controls(self, layout: QVBoxLayout) -> None:
        """Add method-specific widgets to the settings column (subclass hook)."""

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._result = None
        self.add_btn.setEnabled(False)
        self.varpanel.set_variables(df.columns, created)
        self.picker.seed(df.columns)  # blocks its own signals; no _on_field_changed
        self._render_inputs()
        self._result_panel.set_header(self._out_name() or None)
        self._result_panel.clear_plot("press Calculate")
        self.status.setText(self.intro)

    # --- fields --------------------------------------------------------
    def _valid_picks(self) -> dict[str, str] | None:
        """The current picks if every required input maps to a real column, else
        None. (No optional inputs in the derived-variable family so far.)"""
        if self._df is None:
            return None
        cols = {str(c) for c in self._df.columns}
        picks = self.picker.picks()
        for spec in self.inputs:
            val = picks[spec["key"]]
            if not val or val == NONE_ITEM or val not in cols:
                return None
        return picks

    def _render_inputs(self) -> None:
        """Preview each input column's heatmap; highlight the picks in the list."""
        if self._df is None:
            return
        cols = {str(c) for c in self._df.columns}
        picks = self.picker.picks()
        chosen = []
        for spec in self.inputs:
            panel = self._input_panels[spec["key"]]
            col = picks[spec["key"]]
            if col and col != NONE_ITEM and col in cols:
                panel.set_header(col)
                panel.plot_series(self._df[col])
                chosen.append(col)
            else:
                panel.set_header(None)
                panel.clear_plot()
        self.varpanel.set_panels(chosen)

    def _on_field_changed(self) -> None:
        """A field changed: refresh input previews and stale the result (the user
        must press Calculate again — nothing is computed automatically)."""
        self._render_inputs()
        self._result = None
        self.add_btn.setEnabled(False)
        self._result_panel.set_header(self._out_name() or None)
        self._result_panel.clear_plot("press Calculate")

    # --- calculate -----------------------------------------------------
    def _calculate(self) -> None:
        picks = self._valid_picks()
        if picks is None:
            self.status.setText("Pick valid input columns first.")
            return
        try:
            series = self._compute(self._df, picks)
        except Exception as err:  # let the user see why, don't crash the tab
            self._result = None
            self.add_btn.setEnabled(False)
            self.status.setText(f"Cannot calculate: {err}")
            self._result_panel.clear_plot("calculation failed")
            return

        name = self._out_name()
        self._result = series.rename(name) if name else series
        n = int(series.notna().sum())
        unit = f" {self.out_unit}" if self.out_unit else ""
        self.status.setText(
            f"{n:,} values{unit}. Name it and add it to the dataset.")
        self._result_panel.set_header(name or None)
        self._result_panel.plot_series(self._result)
        self.add_btn.setEnabled(n > 0 and bool(name))

    def _compute(self, df: pd.DataFrame, picks: dict[str, str]) -> pd.Series:
        """Call the library function and return the derived series (subclass hook)."""
        raise NotImplementedError

    def _code(self, picks: dict[str, str], name: str | None) -> str:
        """Return a runnable snippet reproducing the calculation (subclass hook)."""
        raise NotImplementedError

    # --- name ----------------------------------------------------------
    def _out_name(self) -> str:
        return self.name_edit.text().strip()

    def _on_name_changed(self, _text: str) -> None:
        name = self._out_name()
        self.add_btn.setText(
            f"Add {name} to dataset" if name else "Add to dataset")
        self._result_panel.set_header(name or None)
        if self._result is not None:
            self._result.name = name
            self.add_btn.setEnabled(
                int(self._result.notna().sum()) > 0 and bool(name))

    # --- python code ---------------------------------------------------
    def _python_code(self) -> str | None:
        """Runnable snippet reproducing the current calculation (Copy Python).

        Returns None (button no-op) until every input is validly assigned."""
        picks = self._valid_picks()
        if picks is None:
            self.status.setText("Pick valid input columns first to copy its code.")
            return None
        return self._code(picks, self._out_name() or None)

    # --- emit ----------------------------------------------------------
    def _add(self) -> None:
        if self._result is None or not self._out_name():
            return
        name = self._out_name()
        out = self._result.copy()
        out.name = name
        picks = self.picker.picks()
        parent = picks[self.inputs[0]["key"]] if self.inputs else None
        result = pd.DataFrame({name: out})
        result.attrs[ATTRS_KEY] = {
            name: provenance_attr(
                origin=DERIVED, parent=str(parent) if parent else None,
                operation=self.title, params={k: str(v) for k, v in picks.items()},
                tags=list(self.method_tags)),
        }
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {name} to the variable list.")
        self.add_btn.setEnabled(False)

    # --- state ---------------------------------------------------------
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self.picker.combos()),
                "name": self.name_edit.text()}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self.picker.combos(), state.get("controls"))
        self.picker.refresh_availability()
        name = state.get("name")
        if name:
            self.name_edit.setText(name)
        self._render_inputs()
