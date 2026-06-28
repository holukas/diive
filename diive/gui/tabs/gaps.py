"""
GUI.TABS.GAPS: GAP & COVERAGE DASHBOARD
=======================================

A diagnostics tab for missing-data patterns: pick a variable on the left, and
the right shows KPI stat cards, a two-panel "gap map" (daily availability
heatmap + gap-spike timeline), and a sortable table of the longest gaps.

The gap map is *clickable* both ways:
- Click a row in the gap table -> the gap is highlighted on the timeline.
- Click the timeline -> the nearest gap is found and highlighted, and its row
  in the table is selected.

All detection, statistics and plotting are the library's (`dv.analysis.GapStats`
— `.summary`, `.long_gaps`, `.gap_at()`, and the per-`ax` `plot_*` methods). This
tab only arranges those into widgets and wires the click interactions; it
implements no gap logic of its own (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui.tabs._explorer_base import SingleVariableExplorerTab
from diive.gui.tabs.overview import _fmt
from diive.gui.widgets.mpl_canvas import MplCanvas

_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

#: Highlight overlay colours (selection on the timeline "gap map").
_HL_FILL = "#1E88E5"
_HL_RING = "#0D47A1"


def _dur(value) -> str:
    """Format a gap duration (Timedelta / NaT) compactly."""
    return str(value) if pd.notna(value) else "n/a"


class GapDashboardTab(SingleVariableExplorerTab):
    """Gap & coverage dashboard for the selected variable."""

    title = "Gaps & coverage"

    def _init_state(self) -> None:
        self._gs = None                 # current GapStats (library)
        self._long_gaps = None          # DataFrame shown in the table
        self._timeline_ax = None
        self._highlight = []            # overlay artists to clear on reselect
        self._syncing = False           # guard table<->plot selection echo

    def _build_right(self) -> QWidget:
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        rl.addWidget(self._build_stats_strip())
        rl.addWidget(self._build_controls())

        # Canvas (gap map) over the gap table, user-resizable.
        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.canvas = MplCanvas()
        self.canvas.fig.canvas.mpl_connect("button_press_event", self._on_click)
        vsplit.addWidget(self.canvas)
        vsplit.addWidget(self._build_table())
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 0)
        vsplit.setSizes([460, 200])
        rl.addWidget(vsplit, stretch=1)
        return right

    # --- sub-widgets ---------------------------------------------------
    def _build_controls(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addWidget(QLabel("Long gap ≥ (records)"))
        self.threshold = QSpinBox()
        self.threshold.setRange(1, 1_000_000)
        self.threshold.setValue(48)  # one day at 30-min resolution
        self.threshold.setToolTip(
            "Gaps with at least this many missing records are listed as 'long "
            "gaps' (GapStats.long_gap_records).")
        self.threshold.valueChanged.connect(self._on_threshold)
        lay.addWidget(self.threshold)
        lay.addStretch(1)
        return bar

    def _build_table(self) -> QWidget:
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Records", "Duration"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for c in (1, 2, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        return self.table

    # --- data flow -----------------------------------------------------
    def save_state(self) -> dict:
        return {"var": self._target, "threshold": self.threshold.value()}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import set_widget_value
        set_widget_value(self.threshold, state.get("threshold"))
        v = state.get("var")
        if v and self._df is not None and v in self._df.columns:
            self._on_select(v)

    def _default_variable(self, df) -> str | None:
        # Default to the variable with the most missing values — the most useful
        # starting point for a gap dashboard. Fall back to the first column when
        # the dataset has no gaps at all.
        if df.shape[1] == 0:
            return None
        missing = df.isna().sum()
        return str(missing.idxmax()) if missing.max() > 0 else str(df.columns[0])

    def _on_threshold(self, _value: int) -> None:
        self._recompute()

    def _compute(self) -> None:
        # All gap logic is the library's; the tab only reads results back.
        series = self._df[self._target]
        self._gs = dv.analysis.GapStats(series, long_gap_records=self.threshold.value())
        self._fill_stats()
        self._fill_table()
        self._render_figure()

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None or self._target is None or self._target not in self._df.columns:
            return None
        from diive.analysis.gapfinder import gapstats_to_code
        return gapstats_to_code(self._target, long_gap_records=self.threshold.value())

    def _fill_stats(self) -> None:
        s = self._gs.summary
        wm = s["worst_month"]
        cards = [
            ("Missing", f"{s['missing_pct']:.1f}%"),
            ("Gap periods", _fmt(s["n_gaps"])),
            ("Long gaps", _fmt(s["n_long_gaps"])),
            ("Longest", f"{_fmt(s['longest_gap_records'])} rec"),
            ("Longest dur.", _dur(s["longest_gap_duration"])),
            ("Worst month", _MONTHS[wm] if wm else "—"),
        ]
        self._set_stat_cards(cards)

    def _fill_table(self) -> None:
        lg = self._gs.long_gaps.reset_index(drop=True)
        self._long_gaps = lg
        self._syncing = True  # block selection echo while repopulating
        try:
            self.table.clearSelection()
            self.table.setRowCount(len(lg))
            for r, (_, row) in enumerate(lg.iterrows()):
                cells = [
                    str(r + 1),
                    row["GAP_START"].strftime("%Y-%m-%d %H:%M"),
                    row["GAP_END"].strftime("%Y-%m-%d %H:%M"),
                    f"{int(row['GAP_LENGTH']):,}",
                    _dur(row["GAP_DURATION"]),
                ]
                for c, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if c in (0, 3):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                              | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, c, item)
        finally:
            self._syncing = False

    def _render_figure(self) -> None:
        self.canvas.reset_layout()
        fig = self.canvas.fig
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1])
        ax_hm = fig.add_subplot(gs[0])
        ax_tl = fig.add_subplot(gs[1])
        try:
            self._gs.plot_availability_heatmap(ax=ax_hm)
            self._gs.plot_gap_spike_timeline(ax=ax_tl)
        except Exception as err:
            ax_tl.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                       wrap=True, transform=ax_tl.transAxes)
        self._timeline_ax = ax_tl
        self._highlight = []  # axes were recreated; old artists are gone
        self.canvas.draw()

    # --- clickable gap map ---------------------------------------------
    def _on_row_selected(self) -> None:
        if self._syncing or self._long_gaps is None:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._long_gaps):
            self._highlight_gap(self._long_gaps.iloc[r])

    def _on_click(self, event) -> None:
        if (self._gs is None or self._timeline_ax is None
                or event.inaxes is not self._timeline_ax or event.xdata is None):
            return
        gap = self._gs.gap_at(mdates.num2date(event.xdata))  # library lookup
        if gap is None:
            return
        self._highlight_gap(gap)
        self._select_table_row_for(gap)

    def _highlight_gap(self, gap) -> None:
        """Draw the selection overlay for `gap` on the timeline."""
        ax = self._timeline_ax
        if ax is None or gap is None:
            return
        for art in self._highlight:
            art.remove()
        x0 = mdates.date2num(gap["GAP_START"])
        x1 = mdates.date2num(gap["GAP_END"])
        length = float(gap["GAP_LENGTH"])
        span = ax.axvspan(x0, x1, color=_HL_FILL, alpha=0.25, zorder=6)
        # A ring at (start, length) marks short gaps too thin to see as a span.
        ring, = ax.plot([x0], [length], marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor=_HL_RING, markeredgewidth=2.0, zorder=7)
        self._highlight = [span, ring]
        self.canvas.draw_idle()

    def _select_table_row_for(self, gap) -> None:
        """Select the table row matching `gap` (if it's a listed long gap)."""
        if self._long_gaps is None or self._long_gaps.empty:
            return
        match = self._long_gaps.index[self._long_gaps["GAP_START"] == gap["GAP_START"]]
        if len(match):
            self._syncing = True
            try:
                self.table.selectRow(int(match[0]))
            finally:
                self._syncing = False
