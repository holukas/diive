"""
GUI.TABS.OVERVIEW: SELECTED-VARIABLE OVERVIEW
=============================================

The first tab, shown when a dataset is loaded. Pick a variable on the left; the
right shows a multi-panel figure and a full-width strip of KPI-style stat cards
(`dv.sstats`) along the bottom. Figure panels are easy to extend (`_PANELS`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel

_DEFAULT_VAR = "NEE_CUT_REF_f"

# Overview figure layout (2 rows x 3 cols): time series spans the top-left, the
# cumulative and diel-cycle panels sit below it, and the heatmap fills the right
# column over the full height. Add panels by extending _PANELS.
# Each entry: (gridspec row-slice, gridspec col-slice, plot type).
_PANELS = [
    ((0, slice(0, 2)), "Time series"),
    ((1, 0), "Cumulative"),
    ((1, 1), "Daily average"),
    ((slice(0, 2), 2), "Heatmap (date/time)"),
]


def _fmt(value) -> str:
    """Format a statistic value compactly (ints plain, floats to 4 sig figs)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:.4g}"


class _StatCard(QFrame):
    """A small KPI-style card: stat name on top, value below."""

    def __init__(self, name: str, value: str) -> None:
        super().__init__()
        self.setObjectName("statcard")
        self.setFixedHeight(64)
        self.setMinimumWidth(92)
        self.setStyleSheet(
            "QFrame#statcard { background: #FFFFFF; border: 1px solid #E0E4E7;"
            " border-radius: 8px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(11, 7, 11, 7)
        lay.setSpacing(1)

        name_lbl = QLabel(name)
        nf = name_lbl.font()
        nf.setPointSizeF(max(7.0, nf.pointSizeF() - 1.5))
        nf.setBold(True)
        name_lbl.setFont(nf)
        name_lbl.setStyleSheet("color: #90A4AE; background: transparent;")

        value_lbl = QLabel(value)
        vf = value_lbl.font()
        vf.setPointSizeF(vf.pointSizeF() + 4.0)
        vf.setBold(True)
        value_lbl.setFont(vf)
        value_lbl.setStyleSheet("color: #263238; background: transparent;")
        value_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        lay.addWidget(name_lbl)
        lay.addWidget(value_lbl)
        lay.addStretch(1)


class OverviewTab(DiiveTab):
    """Stats + multi-panel figure for the selected variable."""

    title = "Overview"

    def build(self) -> QWidget:
        self._df = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top: variable list + figure.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)
        self.canvas = MplCanvas()
        splitter.addWidget(self.varpanel)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # Bottom: full-width "dashboard strip" of stat cards.
        self.stats_strip = QScrollArea()
        self.stats_strip.setWidgetResizable(True)
        self.stats_strip.setFixedHeight(92)
        self.stats_strip.setFrameShape(QFrame.Shape.NoFrame)
        self.stats_strip.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.stats_strip.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_bg = theme.manager.tokens["LIST_BG"]
        border = theme.manager.tokens["BORDER"]
        self.stats_strip.setStyleSheet(
            f"QScrollArea {{ background: {list_bg}; border-top: 1px solid {border}; }}")
        host = QWidget()
        host.setStyleSheet(f"background: {list_bg};")
        self.stats_layout = QHBoxLayout(host)
        self.stats_layout.setContentsMargins(10, 8, 10, 8)
        self.stats_layout.setSpacing(8)
        self.stats_layout.addStretch(1)
        self.stats_strip.setWidget(host)
        outer.addWidget(self.stats_strip)
        return root

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        cols = [str(c) for c in df.columns]
        default = _DEFAULT_VAR if _DEFAULT_VAR in cols else (cols[0] if cols else None)
        if default is not None:
            self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self.varpanel.set_panels([name])  # highlight the selected variable
        series = self._df[name]

        def _render() -> None:
            self._fill_stats(series)
            self._render_figure(series, name)

        self.varpanel.run_with_loading(name, _render)

    def _fill_stats(self, series) -> None:
        # Clear existing cards (keep the trailing stretch).
        while self.stats_layout.count() > 1:
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        try:
            values = dv.sstats(series).iloc[:, 0]
        except Exception:
            return
        for i, (stat, value) in enumerate(values.items()):
            self.stats_layout.insertWidget(i, _StatCard(str(stat), _fmt(value)))

    def _render_figure(self, series, name: str) -> None:
        fig = self.canvas.fig
        fig.clear()
        gs = fig.add_gridspec(2, 3)
        for (rows, cols), plot_type in _PANELS:
            ax = fig.add_subplot(gs[rows, cols])
            self._draw_panel(ax, series, plot_type)
        fig.suptitle(name)
        self.canvas.draw()

    def _draw_panel(self, ax, series, plot_type: str) -> None:
        try:
            if plot_type == "Time series":
                dv.plotting.TimeSeries(series).plot(ax=ax, title="Time series")
            elif plot_type == "Cumulative":
                dv.plotting.Cumulative(df=series.to_frame()).plot(
                    ax=ax, showplot=False)
            elif plot_type == "Daily average":
                dv.plotting.DielCycle(series).plot(ax=ax, title="Mean diel cycle")
            elif plot_type == "Heatmap (date/time)":
                dv.plotting.HeatmapDateTime(series).plot(
                    ax=ax, fig=self.canvas.fig, title="Heatmap",
                    cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
