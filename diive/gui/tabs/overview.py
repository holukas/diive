"""
GUI.TABS.OVERVIEW: SELECTED-VARIABLE OVERVIEW
=============================================

The first tab, shown when a dataset is loaded. Pick a variable on the left; the
right shows a multi-panel figure and a full-width strip of KPI-style stat cards
(`dv.sstats`) along the bottom. Figure panels are easy to extend (`_PANELS`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd
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
from diive.core.dfun.stats import SSTATS_DESCRIPTIONS
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

# Panels whose x-axis is the datetime index: linked via a shared x-axis so
# zooming/panning one zooms all of them to the same time period. The diel cycle
# (x = hour of day) and the heatmap (x = time of day, date on the y-axis) live in
# different domains and are intentionally left unlinked.
_DATETIME_X_PANELS = {"Time series", "Cumulative", "Daily mean"}

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

# Refined, mutually distinct line colours so each panel reads at a glance and
# looks professional (the bright Material blue read as garish).
_TS_COLOR = "#546E7A"     # blue-grey 600 — time series (refined, professional)
_DAILY_COLOR = "#26A69A"  # teal 400 — daily mean (line + SD band)
# The diel cycle now draws one auto-coloured line per month (no single colour).
_ZERO_COLOR = "#90A4AE"   # blue-grey 300 — zero reference line
_BADGE_BG = "#1565C0"     # elegant deep blue — variable-name badge background


def _fmt(value) -> str:
    """Format a statistic value compactly (ints plain, floats to 4 sig figs)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:.4g}"


class _StatItem(QWidget):
    """A borderless metric: a tiny tracked label above a bold value.

    No card chrome — items sit on the strip separated by hairlines, reading as
    one clean metrics ribbon rather than a row of boxes.
    """

    def __init__(self, name: str, value: str, tip: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        name_lbl = QLabel(theme.manager.label_text(name))
        nf = theme.manager.tracked_font(name_lbl.font())
        nf.setPointSizeF(max(6.5, nf.pointSizeF() - 2.0))
        nf.setBold(True)
        name_lbl.setFont(nf)
        name_lbl.setStyleSheet("color: #90A4AE; background: transparent;")

        value_lbl = QLabel(value)
        vf = value_lbl.font()
        vf.setPointSizeF(vf.pointSizeF() + 2.0)
        vf.setBold(True)
        value_lbl.setFont(vf)
        value_lbl.setStyleSheet("color: #263238; background: transparent;")
        value_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        lay.addWidget(name_lbl)
        lay.addWidget(value_lbl)

        # Explain the metric on hover (set on the children too, so the tooltip
        # shows wherever the cursor lands within the item).
        if tip:
            full = f"{name}: {tip}"
            for w in (self, name_lbl, value_lbl):
                w.setToolTip(full)


def _stat_separator() -> QFrame:
    """A short vertical hairline between metrics."""
    line = QFrame()
    line.setFixedSize(1, 30)
    line.setStyleSheet("background: #E6E6E3;")
    return line


class _StatCard(QFrame):
    """A compact KPI-style card (used by the Gaps/Drivers/Seasonal tabs)."""

    def __init__(self, name: str, value: str) -> None:
        super().__init__()
        self.setObjectName("statcard")
        self.setFixedHeight(44)
        self.setMinimumWidth(74)
        self.setStyleSheet(
            "QFrame#statcard { background: #FFFFFF; border: 1px solid #E0E4E7;"
            " border-radius: 7px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 4, 9, 4)
        lay.setSpacing(0)

        name_lbl = QLabel(theme.manager.label_text(name))
        nf = theme.manager.tracked_font(name_lbl.font())
        nf.setPointSizeF(max(6.5, nf.pointSizeF() - 2.0))
        nf.setBold(True)
        name_lbl.setFont(nf)
        name_lbl.setStyleSheet("color: #90A4AE; background: transparent;")

        value_lbl = QLabel(value)
        vf = value_lbl.font()
        vf.setPointSizeF(vf.pointSizeF() + 1.0)
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
        # Live zoom-sync state (set per render; see _render_figure / _on_zoom).
        self._zoom_series = None
        self._diel_ax = None
        self._heatmap_ax = None
        self._heatmap_ylim = None
        self._syncing_zoom = False

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
        self.stats_strip.setFixedHeight(54)
        self.stats_strip.setFrameShape(QFrame.Shape.NoFrame)
        self.stats_strip.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.stats_strip.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        border = theme.manager.tokens["BORDER"]
        self.stats_strip.setStyleSheet(
            f"QScrollArea {{ background: #FFFFFF; border-top: 1px solid {border}; }}")
        host = QWidget()
        host.setStyleSheet("background: #FFFFFF;")
        self.stats_layout = QHBoxLayout(host)
        self.stats_layout.setContentsMargins(14, 0, 14, 0)
        self.stats_layout.setSpacing(14)
        self.stats_layout.addStretch(1)
        self.stats_strip.setWidget(host)
        outer.addWidget(self.stats_strip)
        return root

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._created = created or set()
        self.varpanel.set_variables(df.columns, created)
        cols = [str(c) for c in df.columns]
        default = _DEFAULT_VAR if _DEFAULT_VAR in cols else (cols[0] if cols else None)
        if default is not None:
            self._on_select(default)

    def show_variable_subset(self, var_names: list) -> None:
        """Restrict the variable list to `var_names` (from the Select-variables
        tab). Uses the library's `dv.keep_vars` to validate/order the subset;
        the full dataset (`self._df`) is untouched, only the list is filtered."""
        if self._df is None or not var_names:
            return
        subset = dv.keep_vars(self._df, [v for v in var_names if v in self._df.columns])
        names = [str(c) for c in subset.columns]
        if not names:
            return
        self.varpanel.set_variables(names, getattr(self, "_created", set()))
        self._on_select(names[0])

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
            values = dv.sstats(series).iloc[:, 0]  # all stats (ribbon scrolls)
        except Exception:
            return
        at = 0  # insert before the trailing stretch, hairline-separated
        for i, (stat, value) in enumerate(values.items()):
            if i > 0:
                self.stats_layout.insertWidget(at, _stat_separator())
                at += 1
            tip = SSTATS_DESCRIPTIONS.get(str(stat), "")
            self.stats_layout.insertWidget(at, _StatItem(str(stat), _fmt(value), tip))
            at += 1

    def _render_figure(self, series, name: str) -> None:
        fig = self.canvas.fig
        # Clear + re-enable constrained layout (canvas.draw() freezes it after,
        # so zoom/pan don't reflow the panels).
        self.canvas.reset_layout()
        # Pack the panels tighter (less whitespace between them, esp. the three
        # lower panels) while keeping room for tick labels.
        engine = fig.get_layout_engine()
        if engine is not None:
            try:
                engine.set(w_pad=0.015, h_pad=0.02, wspace=0.0, hspace=0.03)
            except (AttributeError, TypeError):
                pass
        gs = fig.add_gridspec(2, 4)
        # Full series for the current range; the diel cycle re-slices it on zoom.
        self._zoom_series = series
        panel_axes: dict[str, object] = {}
        shared_x_ax = None  # first datetime panel; the rest share its x-axis
        for (rows, cols), plot_type in _PANELS:
            kwargs = {}
            if plot_type in _DATETIME_X_PANELS and shared_x_ax is not None:
                kwargs["sharex"] = shared_x_ax
            ax = fig.add_subplot(gs[rows, cols], **kwargs)
            if plot_type in _DATETIME_X_PANELS and shared_x_ax is None:
                shared_x_ax = ax
            panel_axes[plot_type] = ax
            self._draw_panel(ax, series, plot_type)
            self._style_panel(ax, plot_type)
        # One uniform font size for every number, axis label, and in-plot text
        # (including the heatmap colourbar), overriding the plot classes' own
        # sizes for a clean, consistent look.
        for ax in fig.axes:
            self._panel_fonts(ax)
        # Variable name as an elegant badge inside the time-series panel's
        # top-left corner (bold white on deep blue) — saves the figure-level
        # title strip. Added after the per-axes font pass so it keeps its size;
        # fig.clear() drops it on re-render so it never accumulates.
        ts_ax = panel_axes.get("Time series")
        if ts_ax is not None:
            ts_ax.text(0.012, 0.95, name, transform=ts_ax.transAxes,
                       ha="left", va="top", fontsize=12, fontweight="bold",
                       color="white", zorder=100,
                       bbox=dict(boxstyle="round,pad=0.35", facecolor=_BADGE_BG,
                                 edgecolor="none"))

        # Live zoom sync: the three datetime panels follow each other via sharex;
        # the diel cycle (recomputed on the visible window) and the heatmap
        # (clipped to the visible date range) don't share that axis, so update
        # them on every xlim change.
        self._diel_ax = panel_axes.get("Diel cycle")
        self._heatmap_ax = panel_axes.get("Heatmap (date/time)")
        # Heatmap y-axis data range (Date), so zoom can clamp to it.
        self._heatmap_ylim = (
            self._heatmap_ax.get_ylim() if self._heatmap_ax is not None else None)
        if shared_x_ax is not None:
            shared_x_ax.callbacks.connect("xlim_changed", self._on_zoom)
        self.canvas.draw()

    def _style_panel(self, ax, plot_type: str) -> None:
        """Per-panel styling shared by the initial render and zoom re-draws."""
        if plot_type != "Heatmap (date/time)":
            # The selected variable is known, so the value axis needs no label
            # (the heatmap's y-axis is Date and keeps its label).
            ax.set_ylabel("")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        # Header tinted to match the panel's line, for quick visual pairing.
        ax.set_title(_PANEL_TITLES.get(plot_type, plot_type),
                     fontsize=_TITLE_FONTSIZE, fontweight=500,
                     color=self._line_color(ax))

    @staticmethod
    def _panel_fonts(ax) -> None:
        ax.tick_params(axis="both", labelsize=_FONT_SIZE)
        ax.xaxis.label.set_size(_FONT_SIZE)
        ax.yaxis.label.set_size(_FONT_SIZE)
        for txt in ax.texts:
            txt.set_fontsize(_FONT_SIZE)

    def _on_zoom(self, shared_ax) -> None:
        """React to a zoom/pan of the shared datetime x-axis.

        (1) Recompute the diel cycle from only the data in the visible window.
        (2) Clip the heatmap to the same date range — its date axis is the y-axis
            (same matplotlib date-number units as the line panels' x-axis), so
            the hour-of-day x-axis is deliberately left untouched.
        """
        if self._zoom_series is None or self._syncing_zoom:
            return
        x0, x1 = shared_ax.get_xlim()
        lo, hi = min(x0, x1), max(x0, x1)
        self._syncing_zoom = True
        try:
            if self._heatmap_ax is not None and self._heatmap_ylim is not None:
                # Clamp to the heatmap's own date span so zooming past the data
                # doesn't add empty margins.
                ylo = max(lo, self._heatmap_ylim[0])
                yhi = min(hi, self._heatmap_ylim[1])
                if yhi > ylo:
                    self._heatmap_ax.set_ylim(ylo, yhi)
            if self._diel_ax is not None:
                start = pd.Timestamp(mdates.num2date(lo)).tz_localize(None)
                end = pd.Timestamp(mdates.num2date(hi)).tz_localize(None)
                sub = dv.times.keep_daterange(self._zoom_series, start=start, end=end)
                self._diel_ax.clear()
                self._draw_panel(self._diel_ax, sub, "Diel cycle")
                self._style_panel(self._diel_ax, "Diel cycle")
                self._panel_fonts(self._diel_ax)
        finally:
            self._syncing_zoom = False
        # Repaint without re-freezing the layout (draw() would flip the layout
        # engine and could abort an in-progress resize re-solve).
        self.canvas.draw_idle()

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
                dv.plotting.TimeSeries(series).plot(
                    ax=ax, color=_TS_COLOR, linewidth=1.4)
            elif plot_type == "Cumulative":
                dv.plotting.Cumulative(df=series.to_frame()).plot(
                    ax=ax, showplot=False, show_title=False, fill=True)
            elif plot_type == "Diel cycle":
                # One auto-coloured line per month (seasonal diel pattern).
                dv.plotting.DielCycle(series).plot(
                    ax=ax, each_month=True, show_legend=False, linewidth=1.1)
                ax.axhline(0, color=_ZERO_COLOR, linestyle="--", linewidth=1.0,
                           alpha=0.6, zorder=1)
            elif plot_type == "Daily mean":
                # Daily mean ± SD over the (possibly subselected) range.
                daily = dv.times.resample_to_daily_agg(series, agg="mean")
                sd = dv.times.resample_to_daily_agg(series, agg="std")
                ax.fill_between(daily.index, (daily - sd).to_numpy(),
                                (daily + sd).to_numpy(), color=_DAILY_COLOR,
                                alpha=0.2, edgecolor="none", zorder=0)
                dv.plotting.TimeSeries(daily).plot(ax=ax, color=_DAILY_COLOR, linewidth=1.4)
                ax.axhline(0, color=_ZERO_COLOR, linestyle="--", linewidth=1.0,
                           alpha=0.6, zorder=0)
            elif plot_type == "Heatmap (date/time)":
                dv.plotting.HeatmapDateTime(series).plot(
                    ax=ax, fig=self.canvas.fig, cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
