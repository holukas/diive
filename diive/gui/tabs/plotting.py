"""
GUI.TABS.PLOTTING: INTERACTIVE PLOTTING TAB
===========================================

Two-column plotting tab: a list of variables on the left, the plot area on the
right. On startup the bundled example dataset is loaded, the variable list is
populated from its columns, and NEE is selected and rendered as a date x
time-of-day heatmap.

Selection model:
- Plain click  -> reset to a single panel showing the clicked variable.
- Ctrl + click -> append another panel to the right (up to `_MAX_PANELS`),
  for side-by-side comparison.

Selected variables are highlighted in the list and numbered with their panel
position (left to right). All panels share one date y-axis.

diive's plot classes use the two-phase `__init__(data)` / `plot(ax=...)`
pattern, so each panel is just `HeatmapDateTime(series).plot(ax=..., fig=...)`.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLineEdit,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_delegate import NAME_ROLE, PANEL_ROLE, VariableDelegate
from diive.gui.widgets.variable_list import VariableList

#: Column selected on startup -- gap-filled (continuous) NEE from the bundled
#: CH-DAV example dataset.
_DEFAULT_VAR = "NEE_CUT_REF_f"

#: Maximum number of side-by-side panels (further Ctrl+clicks are ignored).
_MAX_PANELS = 5


class PlottingTab(DiiveTab):
    """Pick variables on the left, render one or more heatmaps on the right."""

    title = "Plotting"

    def build(self) -> QWidget:
        self._df = None
        self._panels: list[str] = []

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left column: a filter field above the variable list. Width is
        # user-resizable via the splitter handle.
        left = QWidget()
        left.setMinimumWidth(160)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText("Filter variables...")
        self.search.textChanged.connect(self._filter_list)

        # Variable list. Plain click resets; Ctrl+click toggles a panel. The
        # variable name is stored in NAME_ROLE so the delegate can render a
        # panel-order prefix and pills without affecting click handling.
        self.var_list = VariableList()
        # A custom delegate paints rows (highlight + NEE pill); disable Qt's
        # own selection highlight, and enable mouse tracking for hover.
        self.var_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.var_list.setItemDelegate(VariableDelegate(self.var_list))
        self.var_list.setMouseTracking(True)
        # Names are elided by the delegate; never scroll horizontally (it would
        # push the right-aligned pills out of view).
        self.var_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.var_list.setWordWrap(False)
        self.var_list.selected.connect(self._on_selected)

        left_layout.addWidget(self.search)
        left_layout.addWidget(self.var_list, stretch=1)

        # Right: embedded matplotlib canvas.
        self.canvas = MplCanvas()

        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)   # list keeps its width
        splitter.setStretchFactor(1, 1)   # canvas takes extra space
        splitter.setSizes([260, 840])
        layout.addWidget(splitter)
        return root

    def on_data_loaded(self, df) -> None:
        """Populate the variable list from a newly loaded dataset and render."""
        self._df = df
        self._panels = []
        self.search.clear()
        self.var_list.clear()
        for col in df.columns:
            item = QListWidgetItem(str(col))
            item.setData(NAME_ROLE, str(col))
            item.setData(PANEL_ROLE, 0)
            self.var_list.addItem(item)
        self._select_default()

    def _filter_list(self, text: str) -> None:
        """Hide variables not matching `text`.

        Matches as a subsequence over the normalized (lowercased,
        separator-stripped) name, so the typed characters need only appear in
        order -- e.g. 'gpp16' matches 'GPP_CUT_16_f'.
        """
        needle = _normalize(text)
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            name = _normalize(item.data(NAME_ROLE) or "")
            item.setHidden(not _is_subsequence(needle, name))

    def _select_default(self) -> None:
        """Highlight and render the startup variable in a single panel."""
        cols = [str(c) for c in self._df.columns]
        if _DEFAULT_VAR in cols:
            row = cols.index(_DEFAULT_VAR)
        elif cols:
            row = 0
        else:
            return
        self.var_list.setCurrentRow(row)
        self._panels = [cols[row]]
        self._render()

    def _on_selected(self, name: str, additive: bool) -> None:
        if not name:
            return
        if additive:
            # Ctrl+click toggles a panel: remove if already shown, else append.
            if name in self._panels:
                self._panels.remove(name)
            elif len(self._panels) < _MAX_PANELS:
                self._panels.append(name)
            else:
                return  # cap reached -- ignore further panels
        else:
            self._panels = [name]
        self._render()

    def _render(self) -> None:
        """Render one panel per entry in `self._panels`, left to right.

        All variables share the same date index and time-of-day axis, so the
        panels share both x and y; the date labels are kept only on the
        leftmost panel.
        """
        if not self._panels:
            # All panels toggled off -- show a blank canvas.
            self.canvas.new_axes(1)
            self.canvas.draw()
            self._mark_selected()
            return
        axes = self.canvas.new_axes(len(self._panels), sharex=True, sharey=True)
        for ax, name in zip(axes, self._panels):
            self._draw_one(ax, name)
        # Drop the redundant date axis from every panel but the first.
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        self.canvas.draw()
        self._mark_selected()

    def _draw_one(self, ax, name: str) -> None:
        """Draw one variable into `ax`, or an explanatory message on failure.

        Columns that cannot be heatmapped (non-numeric, all-NaN) show a message
        instead of raising, so the variable list stays usable.
        """
        series = self._df[name]
        try:
            hm = dv.plotting.HeatmapDateTime(series)
            hm.plot(
                ax=ax, fig=self.canvas.fig, title=name,
                cb_digits_after_comma='auto',
            )
        except Exception as err:
            ax.text(
                0.5, 0.5, f"Cannot plot '{name}':\n{err}",
                ha="center", va="center", wrap=True, transform=ax.transAxes,
            )

    def _mark_selected(self) -> None:
        """Tag each list entry with its panel position for the delegate.

        Order 1 = primary panel, 2+ = additional panels, 0 = not shown. The
        delegate maps these to colours and the panel-number prefix.
        """
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            name = item.data(NAME_ROLE)
            order = self._panels.index(name) + 1 if name in self._panels else 0
            item.setData(PANEL_ROLE, order)
        self.var_list.viewport().update()


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric chars for separator-insensitive search."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_subsequence(needle: str, hay: str) -> bool:
    """True if every char of `needle` appears in `hay` in order (gaps allowed)."""
    it = iter(hay)
    return all(c in it for c in needle)
