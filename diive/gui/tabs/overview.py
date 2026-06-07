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

# Overview figure layout (2 rows x 4 cols): the time series spans the top-left
# three columns; below it sit the cumulative, mean-diel-cycle and daily-mean
# panels; the heatmap fills the right column over the full height. Add panels by
# extending _PANELS. Each entry: (gridspec row-slice, gridspec col-slice, type).
_PANELS = [
    ((0, slice(0, 3)), "Time series"),
    ((1, 0), "Cumulative"),
    ((1, 1), "Diel cycle"),
    ((1, 2), "Daily mean"),
    ((slice(0, 2), 3), "Heatmap (date/time)"),
]

# Short, uniform panel headers (the variable name lives in the figure suptitle).
_PANEL_TITLES = {
    "Time series": "Time series",
    "Cumulative": "Cumulative",
    "Diel cycle": "Diel cycle",
    "Daily mean": "Daily mean ± SD",
    "Heatmap (date/time)": "Heatmap",
}
_TITLE_FONTSIZE = 10
_TITLE_COLOR = "#37474F"  # blue-grey 800 — fallback header colour (heatmap)

# One size for every tick number, axis label, and in-plot annotation across all
# panels, so the overview reads cleanly despite the plot classes' own defaults.
_FONT_SIZE = 9

# Modern, mutually distinct line colours so each panel reads at a glance: the
# time series stays Material blue; the diel cycle and daily mean differ from it.
_DIEL_COLOR = "#7E57C2"   # deep purple 400 — mean diel cycle
_DAILY_COLOR = "#26A69A"  # teal 400 — daily mean (line + SD band)


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
        # Clear + re-enable constrained layout (canvas.draw() freezes it after,
        # so zoom/pan don't reflow the panels).
        self.canvas.reset_layout()
        gs = fig.add_gridspec(2, 4)
        for (rows, cols), plot_type in _PANELS:
            ax = fig.add_subplot(gs[rows, cols])
            self._draw_panel(ax, series, plot_type)
            if plot_type != "Heatmap (date/time)":
                # The selected variable is known, so the value axis needs no
                # label (the heatmap's y-axis is Date and keeps its label).
                ax.set_ylabel("")
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            # Header tinted to match the panel's line, for quick visual pairing.
            ax.set_title(_PANEL_TITLES.get(plot_type, plot_type),
                         fontsize=_TITLE_FONTSIZE, fontweight=500,
                         color=self._line_color(ax))
        # One uniform font size for every number, axis label, and in-plot text
        # (including the heatmap colourbar), overriding the plot classes' own
        # sizes for a clean, consistent look.
        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=_FONT_SIZE)
            ax.xaxis.label.set_size(_FONT_SIZE)
            ax.yaxis.label.set_size(_FONT_SIZE)
            for txt in ax.texts:
                txt.set_fontsize(_FONT_SIZE)
        fig.suptitle(name)
        self.canvas.draw()

    @staticmethod
    def _line_color(ax) -> str:
        """Colour of the panel's main data line (the zorder-99 line the diive
        plot classes draw), used to tint the header. Falls back to the neutral
        title colour for panels without a line (the heatmap)."""
        for line in ax.get_lines():
            if line.get_zorder() == 99:
                return line.get_color()
        return _TITLE_COLOR

    def _draw_panel(self, ax, series, plot_type: str) -> None:
        try:
            if plot_type == "Time series":
                dv.plotting.TimeSeries(series).plot(ax=ax)
            elif plot_type == "Cumulative":
                dv.plotting.Cumulative(df=series.to_frame()).plot(
                    ax=ax, showplot=False, show_title=False)
            elif plot_type == "Diel cycle":
                dv.plotting.DielCycle(series).plot(
                    ax=ax, color=_DIEL_COLOR, show_legend=False)
            elif plot_type == "Daily mean":
                # Daily mean ± SD over the (possibly subselected) range.
                daily = dv.times.resample_to_daily_agg(series, agg="mean")
                sd = dv.times.resample_to_daily_agg(series, agg="std")
                ax.fill_between(daily.index, (daily - sd).to_numpy(),
                                (daily + sd).to_numpy(), color=_DAILY_COLOR,
                                alpha=0.2, edgecolor="none", zorder=0)
                dv.plotting.TimeSeries(daily).plot(ax=ax, color=_DAILY_COLOR)
            elif plot_type == "Heatmap (date/time)":
                dv.plotting.HeatmapDateTime(series).plot(
                    ax=ax, fig=self.canvas.fig, cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
