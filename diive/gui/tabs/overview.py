"""
GUI.TABS.OVERVIEW: SELECTED-VARIABLE OVERVIEW
=============================================

The first tab, shown when a dataset is loaded. Pick a variable on the left
(full-height list); the right column shows a multi-panel figure with a strip of
KPI-style stat cards (`dv.sstats`) directly below it. Figure panels are easy to
extend (`_PANELS`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from html import escape

import matplotlib.dates as mdates
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import events as events_store
from diive.gui import metadata_store
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle

_DEFAULT_VAR = "NEE_CUT_REF_f"


def _is_flag(name: str) -> bool:
    """A QC/outlier flag column (``FLAG_..._TEST`` / ``..._QCF``) — not a series
    worth auto-plotting; the Overview prefers the cleaned data column instead."""
    return str(name).startswith("FLAG_")

# Overview figure layout (2 rows x 4 cols): the time series spans the top-left
# three columns with the date/time heatmap top-right; below them sit the
# cumulative, diel-cycle, daily-mean and histogram panels. Add panels by
# extending _PANELS. Each entry: (gridspec row-slice, gridspec col-slice, type).
_PANELS = [
    ((0, slice(0, 3)), "Time series"),
    ((0, 3), "Heatmap (date/time)"),
    ((1, 0), "Cumulative"),
    ((1, 1), "Diel cycle"),
    ((1, 2), "Daily mean"),
    ((1, 3), "Histogram"),
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
    "Histogram": "Distribution",
    "Heatmap (date/time)": "Heatmap",
}
_TITLE_FONTSIZE = 10

# One size for every tick number, axis label, and in-plot annotation across all
# panels, so the overview reads cleanly despite the plot classes' own defaults.
_FONT_SIZE = 9

# diive overview tick/spine standard, applied uniformly to every panel (the plot
# classes' own tick/spine styles vary): ticks point inward, thin and short
# (matching the cumulative panel); all four spines shown at the same thin width;
# no grid (the zero reference line is enough).
_TICK_LENGTH = 4
_LINE_WIDTH = 0.8

# Refined, mutually distinct line colours so each panel reads at a glance and
# looks professional (the bright Material blue read as garish).
_TS_COLOR = "#546E7A"     # blue-grey 600 — time series (refined, professional)
_DAILY_COLOR = "#26A69A"  # teal 400 — daily mean (line + SD band)
# The diel cycle now draws one auto-coloured line per month (no single colour).
_ZERO_COLOR = "#90A4AE"   # blue-grey 300 — zero reference line


def _fmt(value) -> str:
    """Format a statistic value compactly (ints plain, floats to 4 sig figs)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:.4g}"


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


def _human_timedelta(td) -> str:
    """A compact, unit-light duration label (e.g. '3.2 years', '18 days', '6 h')."""
    try:
        secs = float(pd.Timedelta(td).total_seconds())
    except (TypeError, ValueError):
        return "—"
    if secs <= 0:
        return "—"
    days = secs / 86400.0
    if days >= 365:
        return f"{days / 365.25:.1f} years"
    if days >= 1:
        return f"{days:.0f} days"
    hours = secs / 3600.0
    if hours >= 1:
        return f"{hours:.0f} h"
    return f"{secs / 60:.0f} min"


def _human_step(index) -> str:
    """Median sampling step of a datetime index, compactly (e.g. '30 min')."""
    try:
        secs = float(pd.Timedelta(index.to_series().diff().median()).total_seconds())
    except (TypeError, ValueError, AttributeError):
        return ""
    if secs <= 0:
        return ""
    if secs >= 86400:
        return f"{secs / 86400:.0f} d"
    if secs >= 3600:
        return f"{secs / 3600:.0f} h"
    return f"{secs / 60:.0f} min"


# Origin -> (chip background, chip text). Authoritative provenance from the
# metadata store (not name-guessed), so it gets a confident badge.
_ORIGIN_COLORS = {
    "original": ("#E3F2FD", "#1565C0"),
    "modified": ("#FFF3E0", "#E65100"),
    "derived": ("#F3E5F5", "#6A1B9A"),
}


def _chip_qss(bg: str, fg: str) -> str:
    """Stylesheet for a small rounded pill (origin badge / tag chip)."""
    return (f"QLabel {{ background: {bg}; color: {fg}; border-radius: 7px;"
            f" padding: 2px 9px; font-size: 11px; font-weight: 600; }}")


def _chip(text: str, bg: str, fg: str) -> QLabel:
    """A small rounded pill (origin badge / tag chip)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(_chip_qss(bg, fg))
    return lbl


def _history_notes_html(meta) -> str:
    """Rich-text tooltip: the user note + provenance history (or '' if neither).

    Mirrors the variable-list tooltip's history formatting (``ProvenanceEntry.
    describe()`` + timestamp) so the two read identically.
    """
    if meta is None:
        return ""
    rows = []
    note = (getattr(meta, "description", "") or "").strip()
    if note:
        rows.append(f"<span style='color:#90A4AE'>note:</span> <i>{escape(note)}</i>")
    prov = getattr(meta, "provenance", None) or []
    if prov:
        steps = "".join(
            f"<li>{escape(p.describe())}"
            + (f" <span style='color:#90A4AE'>· {escape(p.timestamp)}</span>"
               if p.timestamp else "")
            + "</li>"
            for p in prov)
        rows.append("<span style='color:#90A4AE'>history:</span>"
                    f"<ol style='margin:2px 0 0 -22px'>{steps}</ol>")
    return "<br>".join(rows)


def _clear_layout(layout) -> None:
    """Recursively remove and delete every item in a layout.

    Widgets are unparented immediately (``setParent(None)``) before
    ``deleteLater``: ``deleteLater`` alone keeps them as children until a later
    event-loop pass, so the old content lingers/flickers under the new content.
    """
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())
            item.layout().deleteLater()


class _MetricSlot(QWidget):
    """A persistent metric (tiny tracked label over a bold value), updated in
    place — built once, never torn down, so it can't flicker."""

    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        tip_qss = theme.manager.tooltip_qss()
        self._name = QLabel()
        nf = theme.manager.tracked_font(self._name.font())
        nf.setPointSizeF(max(6.5, nf.pointSizeF() - 2.0))
        nf.setBold(True)
        self._name.setFont(nf)
        self._name.setStyleSheet(
            "QLabel { color: #90A4AE; background: transparent; }" + tip_qss)
        self._value = QLabel()
        vf = self._value.font()
        vf.setPointSizeF(vf.pointSizeF() + 2.0)
        vf.setBold(True)
        self._value.setFont(vf)
        self._value.setStyleSheet(
            "QLabel { color: #263238; background: transparent; }" + tip_qss)
        self._value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._name)
        lay.addWidget(self._value)

    def update_metric(self, name: str, value: str, tip: str) -> None:
        self._name.setText(theme.manager.label_text(name))
        self._value.setText(value)
        full = f"{name}: {tip}" if tip else ""
        for w in (self, self._name, self._value):
            w.setToolTip(full)


class _HeroBand(QFrame):
    """Identity + full-stats header for the selected variable.

    Identity line: name + origin badge + user tags + a history/notes hover field
    (authoritative metadata, not name-guessed). Below it, all summary stats laid
    out in logical rows (record extent & availability; centre & spread;
    distribution from min to max) — this is the only stats surface (there is no
    bottom ribbon). Values come from `dv.sstats` (plus `count_gaps`); deliberately
    unit-free and length-agnostic, each computed defensively (a failure shows '—'
    rather than blanking the row).

    Built **once** with persistent widgets — `set_variable` only updates their
    text/visibility, never tears them down — so switching variables can't flicker.
    The only per-call rebuild is the tag chips (variable count), confined to a
    small sub-container. Vertical size policy is Fixed: the structure is constant,
    so the band keeps a constant height and never resizes the canvas below.
    """

    #: Metric slots grouped into rows (logical order). Each entry is
    #: (label, tooltip); `_metric_values` returns a {label: value} map. A single
    #: row: record extent & availability, then centre/spread/total, then the
    #: distribution low -> high.
    _STAT_ROWS = [
        [
            ("STARTDATE", "First timestamp in the record"),
            ("ENDDATE", "Last timestamp in the record"),
            ("PERIOD", "Time span of the record"),
            ("NOV", "Number of values (records)"),
            ("MISSING", "Number of missing records"),
            ("COVERAGE", "Share of records present (non-missing)"),
            ("GAPS", "Number of gaps (consecutive missing runs)"),
            ("MEAN ± SD", "Mean ± standard deviation"),
            ("VAR", "Variance"),
            ("CV", "Coefficient of variation (SD / mean)"),
            ("SUM", "Sum of the available values"),
            ("MIN", "Minimum value"),
            ("P01", "1st percentile"),
            ("MEDIAN", "Median (50th percentile)"),
            ("P99", "99th percentile"),
            ("MAX", "Maximum value"),
        ],
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("heroband")
        border = theme.manager.tokens["BORDER"]
        self.setStyleSheet(
            f"QFrame#heroband {{ background: #FFFFFF; border: 1px solid {border};"
            f" border-radius: 10px; }}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 11, 16, 11)
        lay.setSpacing(8)

        # --- identity row (persistent widgets) -------------------------------
        idrow = QHBoxLayout()
        idrow.setSpacing(8)
        self._name = QLabel()
        nf = self._name.font()
        nf.setPointSizeF(nf.pointSizeF() + 5.0)
        nf.setBold(True)
        self._name.setFont(nf)
        self._name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        idrow.addWidget(self._name)

        self._origin = QLabel()
        idrow.addWidget(self._origin)

        # Tag chips vary in number, so they live in their own small container
        # that's the only thing rebuilt per selection.
        self._tags_box = QWidget()
        self._tags_lay = QHBoxLayout(self._tags_box)
        self._tags_lay.setContentsMargins(0, 0, 0, 0)
        self._tags_lay.setSpacing(6)
        idrow.addWidget(self._tags_box)

        self._hist = QLabel("history & notes")
        self._hist.setStyleSheet(
            _chip_qss("#ECEFF1", "#546E7A") + theme.manager.tooltip_qss())
        self._hist.setCursor(Qt.CursorShape.WhatsThisCursor)
        idrow.addWidget(self._hist)
        idrow.addStretch(1)
        lay.addLayout(idrow)

        # --- stat rows (persistent slots, keyed by label) --------------------
        self._slots: dict[str, _MetricSlot] = {}
        for row in self._STAT_ROWS:
            rowlay = QHBoxLayout()
            rowlay.setSpacing(14)
            for i, (label, _tip) in enumerate(row):
                if i > 0:
                    rowlay.addWidget(_stat_separator())
                slot = _MetricSlot()
                self._slots[label] = slot
                rowlay.addWidget(slot)
            rowlay.addStretch(1)
            lay.addLayout(rowlay)

    def set_variable(self, name: str, series) -> None:
        meta = metadata_store.manager.store.peek(name)  # None if untracked
        self._name.setText(name)

        origin = getattr(meta, "origin", None)
        if origin:
            bg, fg = _ORIGIN_COLORS.get(origin, ("#ECEFF1", "#37474F"))
            self._origin.setText(origin.upper())
            self._origin.setStyleSheet(_chip_qss(bg, fg))
            self._origin.show()
        else:
            self._origin.hide()

        # Tags: rebuild only the small tag container (updates suppressed on it
        # alone so the rest of the band is untouched).
        self._tags_box.setUpdatesEnabled(False)
        _clear_layout(self._tags_lay)
        if meta is not None:
            for tag in sorted(meta.user_tags()):
                cbg, cfg = theme.tag_color(tag)
                self._tags_lay.addWidget(_chip(tag, cbg, cfg))
        self._tags_lay.activate()
        self._tags_box.setUpdatesEnabled(True)

        hist_html = _history_notes_html(meta)
        if hist_html:
            self._hist.setToolTip(hist_html)
            self._hist.show()
        else:
            self._hist.hide()

        values = self._metric_values(series)
        for row in self._STAT_ROWS:
            for label, tip in row:
                self._slots[label].update_metric(label, values.get(label, "—"), tip)

    @staticmethod
    def _metric_values(series) -> dict[str, str]:
        """{label: value string} for every stat slot.

        Numeric stats come from `dv.sstats` (the single canonical computation);
        coverage/gaps are derived directly. Everything is defensive — a value
        that can't be computed is '—', so an odd series never blanks the band."""
        try:
            s = dv.sstats(series).iloc[:, 0]  # Series: stat label -> value
        except Exception:
            s = None

        def g(key: str) -> str:
            if s is None:
                return "—"
            try:
                return _fmt(s[key])
            except Exception:
                return "—"

        out: dict[str, str] = {}
        # Timestamps + span straight from the index (clean, dtype-independent).
        idx = series.index
        try:
            out["STARTDATE"] = pd.Timestamp(idx.min()).strftime("%Y-%m-%d %H:%M")
            out["ENDDATE"] = pd.Timestamp(idx.max()).strftime("%Y-%m-%d %H:%M")
            out["PERIOD"] = _human_timedelta(idx.max() - idx.min())
        except Exception:
            out["STARTDATE"] = out["ENDDATE"] = out["PERIOD"] = "—"
        # Coverage / missing / gaps derived directly (robust on odd series).
        try:
            n = len(series)
            miss = int(series.isna().sum())
            out["NOV"] = f"{n:,}"
            out["MISSING"] = f"{miss:,}"
            out["COVERAGE"] = f"{100.0 * (n - miss) / n if n else 0.0:.0f}%"
        except Exception:
            out["NOV"] = out["MISSING"] = out["COVERAGE"] = "—"
        try:
            out["GAPS"] = f"{int(dv.analysis.count_gaps(series)):,}"
        except Exception:
            out["GAPS"] = "—"
        # Combined mean ± SD.
        if s is not None:
            try:
                out["MEAN ± SD"] = f"{_fmt(s['MEAN'])} ± {_fmt(s['SD'])}"
            except Exception:
                out["MEAN ± SD"] = "—"
        else:
            out["MEAN ± SD"] = "—"
        # The rest pulled straight from sstats.
        for label in ("VAR", "CV", "SUM", "MIN", "MAX", "MEDIAN", "P01", "P99"):
            out[label] = g(label)
        return out


class OverviewTab(DiiveTab):
    """Stats + multi-panel figure for the selected variable."""

    title = "Overview"

    def build(self) -> QWidget:
        self._df = None
        self._current = None   # selected variable (for project save/restore)
        # Live zoom-sync state (set per render; see _render_figure / _on_zoom).
        self._zoom_series = None
        self._diel_ax = None
        self._hist_ax = None
        self._heatmap_ax = None
        self._heatmap_ylim = None
        self._syncing_zoom = False

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Variable list (left, full height) | right column (figure over stats).
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)

        # Right column: the hero band (identity + all stats) on top, the figure
        # below. There is no separate bottom stats ribbon — every stat lives in
        # the hero band.
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        # Small gap so the figure (and its panel titles) isn't flush against the
        # hero band above it.
        right_lay.setSpacing(10)
        self.hero = _HeroBand()
        right_lay.addWidget(self.hero)
        self.canvas = MplCanvas()
        # matplotlib hides the time-series x tick labels because it shares its
        # x-axis with the panels below it; re-reveal them after every draw.
        self.canvas.fig.canvas.mpl_connect("draw_event", self._reveal_ts_xlabels)
        right_lay.addWidget(self.canvas, stretch=1)

        splitter.addWidget(self.varpanel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # The variable panel is fixed-width, so its splitter handle can't resize
        # anything — lock it so it doesn't show a misleading ↔ resize cursor.
        lock_panel_handle(splitter)
        outer.addWidget(splitter, stretch=1)
        return root

    def refresh_events(self) -> None:
        """Redraw the current variable so event overlays update (called by the
        main window on a visibility-only toggle — column add/edit/delete already
        re-render through the normal data push).

        Renders the figure directly (not via ``run_with_loading``) so it stays
        synchronous — the overlay change is cheap and a deferred busy indicator is
        unnecessary here."""
        if self._current is None or self._df is None \
                or self._current not in self._df.columns:
            return
        self._render_figure(self._df[self._current], self._current)

    def save_state(self) -> dict:
        return {"current": self._current}

    def restore_state(self, state: dict) -> None:
        cur = state.get("current")
        if cur and self._df is not None and cur in self.varpanel.names():
            self._on_select(cur)

    def on_data_loaded(self, df, created: set | None = None) -> None:
        created = created or set()
        cols = [str(c) for c in df.columns]
        # Columns created since the last push (e.g. a freshly added outlier-
        # cleaned column + its flag), so the new feature can be auto-selected.
        prev_created = getattr(self, "_created", set())
        new_cols = [c for c in cols if c in created and c not in prev_created]
        self._df = df
        self._created = created
        # A leftover fuzzy-filter would hide a freshly added variable that doesn't
        # match it (the reported "new var not visible"); clear it so the new
        # feature actually shows. set_variables re-applies whatever text remains.
        if new_cols:
            self.varpanel.clear_filter()
        self.varpanel.set_variables(cols, created)
        # Selection priority: a freshly added feature (so it's plotted and
        # obviously visible — flags are skipped in favour of the cleaned series),
        # then the surviving current selection, then a default.
        # Auto-select a freshly added *non-flag* column (the cleaned series).
        # If every new column is a flag, select none here and fall through to
        # the surviving selection / default — plotting a bare 0/2 flag is not
        # useful and was never the intent.
        non_flags = [c for c in new_cols if not _is_flag(c)]
        feature = non_flags[0] if non_flags else None
        if feature in cols:
            self.varpanel.scroll_to(feature)  # bring the appended row into view
            self._on_select(feature)
        elif self._current in cols:
            self._on_select(self._current)
        elif cols:
            default = _DEFAULT_VAR if _DEFAULT_VAR in cols else cols[0]
            self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self._current = name
        self.varpanel.set_panels([name])  # highlight the selected variable
        series = self._df[name]

        def _render() -> None:
            # The hero (identity + all stats) updates its persistent widgets in
            # place, then the figure renders.
            self.hero.set_variable(name, series)
            self._render_figure(series, name)

        self.varpanel.run_with_loading(name, _render)

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
        # sizes for a clean, consistent look. The variable name lives in the hero
        # band above the figure, so no in-figure name badge is drawn.
        for ax in fig.axes:
            self._panel_fonts(ax)

        # Event overlays: vertical line / shaded span on the datetime panels and a
        # horizontal line / band on the heatmap (date is on its y-axis). Drawn
        # after the font pass so the labels keep their size; they persist through
        # zoom (they're axes artists, not data-window-dependent).
        self._overlay_events(panel_axes)

        # Live zoom sync: the three datetime panels follow each other via sharex;
        # the diel cycle (recomputed on the visible window) and the heatmap
        # (clipped to the visible date range) don't share that axis, so update
        # them on every xlim change.
        self._diel_ax = panel_axes.get("Diel cycle")
        self._hist_ax = panel_axes.get("Histogram")
        self._heatmap_ax = panel_axes.get("Heatmap (date/time)")
        # Heatmap y-axis data range (Date), so zoom can clamp to it.
        self._heatmap_ylim = (
            self._heatmap_ax.get_ylim() if self._heatmap_ax is not None else None)
        self._shared_x_ax = shared_x_ax  # for focus_on() navigation
        if shared_x_ax is not None:
            shared_x_ax.callbacks.connect("xlim_changed", self._on_zoom)
        self.canvas.draw()

    def focus_on(self, start, end) -> None:
        """Zoom the linked datetime panels onto ``[start, end]`` (an event window).

        ``end`` may be ``None`` (instant) — a small symmetric window is opened
        around the instant. The existing ``xlim_changed`` sync then recomputes the
        diel cycle and clips the heatmap to match. No-op if nothing is plotted."""
        ax = getattr(self, "_shared_x_ax", None)
        if ax is None:
            return
        lo = mdates.date2num(pd.Timestamp(start))
        hi = mdates.date2num(pd.Timestamp(end)) if end is not None else lo
        span = hi - lo
        pad = span * 0.5 if span > 0 else 5.0  # ±5 days around an instant
        ax.set_xlim(lo - pad, hi + pad)
        self.canvas.draw_idle()

    def _reveal_ts_xlabels(self, _event) -> None:
        """Re-show the time-series x tick labels after a draw.

        The time series shares its x-axis with the panels below it, so matplotlib
        (treating it as a non-bottom shared subplot) hides its tick labels and
        re-hides them whenever the ticks regenerate (zoom/pan). Re-reveal them so
        the main plot stays dated. A same-view redraw keeps the ticks, so the
        follow-up draw settles; the visibility guard prevents a redraw loop."""
        ax = getattr(self, "_shared_x_ax", None)
        if ax is None:
            return
        changed = False
        for tick in ax.xaxis.get_major_ticks():
            if not tick.label1.get_visible():
                tick.label1.set_visible(True)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _overlay_events(self, panel_axes: dict) -> None:
        """Draw the configured events onto the datetime panels + heatmap."""
        if not events_store.manager.visible:
            return
        evs = events_store.manager.events
        if not evs:
            return
        cats = events_store.manager.categories  # user category palette overrides
        for ptype in ("Time series", "Cumulative", "Daily mean"):
            ax = panel_axes.get(ptype)
            if ax is not None:
                # Label only on the time series to avoid repeating the tags in
                # every small panel.
                dv.events.overlay_events(
                    ax, evs, axis="x", show_labels=(ptype == "Time series"),
                    colors=cats)
        hm = panel_axes.get("Heatmap (date/time)")
        if hm is not None:
            dv.events.overlay_events(hm, evs, axis="y", show_labels=False,
                                     colors=cats)

    def _style_panel(self, ax, plot_type: str) -> None:
        """Per-panel styling shared by the initial render and zoom re-draws."""
        if plot_type != "Heatmap (date/time)":
            # The selected variable is known, so the value axis needs no label
            # (the heatmap's y-axis is Date and keeps its label).
            ax.set_ylabel("")
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        # Panel header in the Qt-header style used elsewhere (e.g. the gap-filling
        # tab): tracked uppercase, bold, monochrome ink. matplotlib has no
        # letter-spacing, so the tracking is approximated by the uppercasing alone.
        ax.set_title(theme.manager.label_text(_PANEL_TITLES.get(plot_type, plot_type)),
                     fontsize=_TITLE_FONTSIZE, fontweight="bold",
                     color=theme.manager.tokens["INK"])

    @staticmethod
    def _panel_fonts(ax) -> None:
        # Uniform fonts + the diive overview tick/spine standard (the plot
        # classes' own tick/spine/grid styles otherwise vary panel to panel).
        ax.tick_params(axis="both", labelsize=_FONT_SIZE, direction="in",
                       width=_LINE_WIDTH, length=_TICK_LENGTH)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(_LINE_WIDTH)
        ax.xaxis.label.set_size(_FONT_SIZE)
        ax.yaxis.label.set_size(_FONT_SIZE)
        for txt in ax.texts:
            txt.set_fontsize(_FONT_SIZE)

    def _on_zoom(self, shared_ax) -> None:
        """React to a zoom/pan of the shared datetime x-axis.

        (1) Recompute the diel cycle and the histogram from only the data in the
            visible window (their x-axes aren't datetime, so they don't follow the
            shared zoom automatically).
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
            # The diel cycle and histogram both summarise the visible window, so
            # recompute them on the zoomed sub-range.
            if self._diel_ax is not None or self._hist_ax is not None:
                start = pd.Timestamp(mdates.num2date(lo)).tz_localize(None)
                end = pd.Timestamp(mdates.num2date(hi)).tz_localize(None)
                sub = dv.times.keep_daterange(self._zoom_series, start=start, end=end)
                for ax, ptype in ((self._diel_ax, "Diel cycle"),
                                  (self._hist_ax, "Histogram")):
                    if ax is None:
                        continue
                    ax.clear()
                    self._draw_panel(ax, sub, ptype)
                    self._style_panel(ax, ptype)
                    self._panel_fonts(ax)
        finally:
            self._syncing_zoom = False
        # Repaint without re-freezing the layout (draw() would flip the layout
        # engine and could abort an in-progress resize re-solve).
        self.canvas.draw_idle()

    def _draw_panel(self, ax, series, plot_type: str) -> None:
        try:
            if plot_type == "Time series":
                dv.plotting.TimeSeries(series).plot(
                    ax=ax, color=_TS_COLOR, linewidth=1.4)
                # Zero reference line only when the data straddles zero (e.g.
                # fluxes) — pointless for all-positive variables far from zero.
                smin, smax = series.min(), series.max()
                if pd.notna(smin) and pd.notna(smax) and smin < 0 < smax:
                    ax.axhline(0, color=_ZERO_COLOR, linestyle="--",
                               linewidth=1.0, alpha=0.6, zorder=1)
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
            elif plot_type == "Histogram":
                # Compact distribution: drop the z-score twiny axis, counts and
                # info box so it reads cleanly at panel size.
                dv.plotting.HistogramPlot(series).plot(
                    ax=ax, show_title=False, show_zscores=False,
                    show_zscore_values=False, show_info=False,
                    show_counts=False, show_grid=False, highlight_peak=True)
            elif plot_type == "Heatmap (date/time)":
                dv.plotting.HeatmapDateTime(series).plot(
                    ax=ax, fig=self.canvas.fig, cb_digits_after_comma="auto")
        except Exception as err:
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
