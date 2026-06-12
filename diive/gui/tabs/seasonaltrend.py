"""
GUI.TABS.SEASONALTREND: SEASONAL-TREND & ANOMALY EXPLORER
=========================================================

"Is this variable changing over the years, and which years were anomalous?"
Pick a variable; the tab decomposes its daily-mean series into trend / seasonal
/ residual (STL, classical or harmonic) and — in a second view — shows each
year's anomaly relative to a reference period.

All the maths is the library's: `dv.times.resample_to_daily_agg` to build the
daily series, `dv.analysis.SeasonalTrendDecomposition` for the decomposition,
and `dv.plotting.LongtermAnomaliesYear` for the anomaly bars. This tab only
collects the options, lays out the panels, and renders — no statistics of its
own (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.plotting.bar import LongtermAnomaliesYear
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import _StatCard, _fmt
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle

#: Strong seasonal cycle + clear warming trend make this a good default demo.
_DEFAULT_VAR = "Tair_f"
#: Daily data -> the meaningful cycle is annual (365 observations).
_PERIOD_DAYS = 365

_VIEW_DECOMP = "Decomposition"
_VIEW_ANOM = "Yearly anomalies"

# Per-component colours (Material Design), matching diive conventions.
_C_OBSERVED = "#455A64"  # blue-grey 700
_C_TREND = "#E53935"     # red 600
_C_SEASONAL = "#43A047"  # green 600
_C_RESIDUAL = "#90A4AE"  # blue-grey 300


class SeasonalTrendTab(DiiveTab):
    """Decompose a variable and inspect its long-term anomalies."""

    title = "Seasonal-trend & anomalies"

    def build(self) -> QWidget:
        self._df = None
        self._target = None
        self._decomp = None      # dict: observed/trend/seasonal/residual + strength
        self._decomp_error = None
        self._yearly = None      # one value per year (for the anomaly view)
        self._loading_ctrls = False  # guard programmatic control updates

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_stats_strip())
        rl.addWidget(self._build_controls())
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)

        splitter.addWidget(self.varpanel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lock_panel_handle(splitter)  # fixed-width list → no misleading ↔ cursor
        outer.addWidget(splitter)
        return root

    # --- sub-widgets ---------------------------------------------------
    def _build_stats_strip(self) -> QWidget:
        strip = QScrollArea()
        strip.setWidgetResizable(True)
        strip.setFixedHeight(92)
        strip.setFrameShape(QFrame.Shape.NoFrame)
        strip.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_bg = theme.manager.tokens["LIST_BG"]
        border = theme.manager.tokens["BORDER"]
        strip.setStyleSheet(
            f"QScrollArea {{ background: {list_bg}; border-bottom: 1px solid {border}; }}")
        host = QWidget()
        host.setStyleSheet(f"background: {list_bg};")
        self.stats_layout = QHBoxLayout(host)
        self.stats_layout.setContentsMargins(10, 8, 10, 8)
        self.stats_layout.setSpacing(8)
        self.stats_layout.addStretch(1)
        strip.setWidget(host)
        return strip

    def _build_controls(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)

        lay.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems(["STL", "Classical", "Harmonic"])
        self.method.setToolTip("STL = robust Loess (default); Classical = moving "
                               "average; Harmonic = Fourier.")
        lay.addWidget(self.method)

        self.robust = QCheckBox("Robust")
        self.robust.setToolTip("STL only: down-weight outliers (slower).")
        lay.addWidget(self.robust)

        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self._update_current)
        lay.addWidget(self.update_btn)

        lay.addSpacing(16)
        lay.addWidget(QLabel("View"))
        self.view = QComboBox()
        self.view.addItems([_VIEW_DECOMP, _VIEW_ANOM])
        self.view.currentTextChanged.connect(self._on_view_changed)
        lay.addWidget(self.view)

        lay.addSpacing(8)
        lay.addWidget(QLabel("Reference"))
        self.ref_start = QSpinBox()
        self.ref_end = QSpinBox()
        for sp in (self.ref_start, self.ref_end):
            sp.setRange(1900, 2200)
            sp.setToolTip("Reference period for the yearly-anomaly view.")
            sp.valueChanged.connect(self._on_ref_changed)
        lay.addWidget(self.ref_start)
        lay.addWidget(QLabel("to"))
        lay.addWidget(self.ref_end)
        lay.addStretch(1)
        return bar

    # --- data flow -----------------------------------------------------
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "controls": save_controls(
                    {"method": self.method, "robust": self.robust,
                     "view": self.view, "ref_start": self.ref_start,
                     "ref_end": self.ref_end})}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        # `state.get("controls") or state` tolerates older projects that saved
        # the control values flat at the top level.
        restore_controls({"method": self.method, "robust": self.robust,
                          "view": self.view, "ref_start": self.ref_start,
                          "ref_end": self.ref_end},
                         state.get("controls") or state)
        t = state.get("target")
        if t and self._df is not None and t in self._df.columns:
            self._on_select(t)

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        cols = [str(c) for c in df.columns]
        numeric = df.select_dtypes(include="number").columns.tolist()
        if _DEFAULT_VAR in cols and _DEFAULT_VAR in numeric:
            default = _DEFAULT_VAR
        elif numeric:
            default = str(numeric[0])
        else:
            return
        self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self.varpanel.run_with_loading(name, self._compute)

    def _update_current(self) -> None:
        if self._target is not None and self._df is not None:
            self.varpanel.run_with_loading(self._target, self._compute)

    def _on_view_changed(self, _text: str) -> None:
        if self._target is not None:
            self._render()

    def _on_ref_changed(self, _value: int) -> None:
        # Cheap: only the anomaly view depends on the reference period.
        if not self._loading_ctrls and self._yearly is not None \
                and self.view.currentText() == _VIEW_ANOM:
            self._render()

    def _compute(self) -> None:
        # All maths is the library's; the tab only reads results back.
        series = self._df[self._target]
        daily = dv.times.resample_to_daily_agg(series, agg="mean").dropna()
        method = self.method.currentText().lower()
        jump = max(1, round(_PERIOD_DAYS / 30))  # speed up STL Loess
        # Annual decomposition needs at least two cycles of daily data; on a
        # short record it cannot run -> keep the (independent) anomaly view alive
        # and show a message in the decomposition view.
        self._decomp = None
        self._decomp_error = None
        if len(daily) < 2 * _PERIOD_DAYS:
            self._decomp_error = (
                f"Need ~2 years of data for an annual decomposition "
                f"(have {len(daily)} days).")
        else:
            try:
                std = dv.analysis.SeasonalTrendDecomposition(
                    daily, method=method, seasonal_period=_PERIOD_DAYS,
                    robust=self.robust.isChecked(), seasonal_jump=jump, trend_jump=jump)
                self._decomp = {
                    "observed": daily,
                    "trend": std.trend,
                    "seasonal": std.seasonal,
                    "residual": std.residual,
                    "strength": std.seasonality_strength,
                }
            except Exception as err:
                self._decomp_error = str(err)

        # Yearly means drive the anomaly view; seed the reference period to the
        # full record on (re)compute.
        yearly = series.resample("YE").mean()
        yearly.index = yearly.index.year
        yearly = yearly.dropna()
        yearly.name = self._target
        self._yearly = yearly
        if len(yearly):
            self._loading_ctrls = True
            lo, hi = int(yearly.index.min()), int(yearly.index.max())
            for sp in (self.ref_start, self.ref_end):
                sp.setRange(lo, hi)
            self.ref_start.setValue(lo)
            self.ref_end.setValue(hi)
            self._loading_ctrls = False

        self._fill_stats()
        self._render()

    def _fill_stats(self) -> None:
        while self.stats_layout.count() > 1:  # keep the trailing stretch
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        d = self._decomp
        trend = d["trend"].dropna() if d else pd.Series(dtype=float)
        change = (trend.iloc[-1] - trend.iloc[0]) if len(trend) else None
        n_years = self._yearly.index.nunique() if self._yearly is not None else 0
        cards = [
            ("Variable", self._target or "—"),
            ("Method", self.method.currentText()),
            ("Seasonality", f"{d['strength']:.2f}" if d else "—"),
            ("Trend change", f"{change:+.2f}" if change is not None else "—"),
            ("Years", _fmt(n_years)),
            ("Period", f"{_PERIOD_DAYS} d"),
        ]
        for i, (name, value) in enumerate(cards):
            self.stats_layout.insertWidget(i, _StatCard(name, value))

    # --- rendering -----------------------------------------------------
    def _render(self) -> None:
        if self._target is None:
            return
        if self.view.currentText() == _VIEW_ANOM:
            self._render_anomalies()
        else:
            self._render_decomposition()

    def _render_decomposition(self) -> None:
        d = self._decomp
        if d is None:
            ax = self.canvas.new_axes(1)[0]
            ax.text(0.5, 0.5, self._decomp_error or "No decomposition",
                    ha="center", va="center", wrap=True, transform=ax.transAxes)
            self.canvas.draw()
            return
        axes = self.canvas.new_axes(4, orientation="vertical", sharex=True)
        panels = [
            ("Observed", d["observed"], _C_OBSERVED),
            ("Trend", d["trend"], _C_TREND),
            ("Seasonal", d["seasonal"], _C_SEASONAL),
            ("Residual", d["residual"], _C_RESIDUAL),
        ]
        try:
            for ax, (label, comp, color) in zip(axes, panels):
                ax.plot(comp.index, comp.to_numpy(), color=color, linewidth=1.0)
                ax.set_ylabel(label, fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, axis="y", alpha=0.2, linewidth=0.6)
            # Trend overlaid on the observed panel for context.
            axes[0].plot(d["trend"].index, d["trend"].to_numpy(),
                         color=_C_TREND, linewidth=1.4, alpha=0.9)
            axes[0].set_title(f"{self._target} — seasonal-trend decomposition", fontsize=10)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
        except Exception as err:
            axes[0].text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                         wrap=True, transform=axes[0].transAxes)
        self.canvas.draw()

    def _render_anomalies(self) -> None:
        ax = self.canvas.new_axes(1)[0]
        if self._yearly is None or self._yearly.empty:
            ax.text(0.5, 0.5, "No yearly data", ha="center", va="center",
                    transform=ax.transAxes)
            self.canvas.draw()
            return
        try:
            LongtermAnomaliesYear(
                series=self._yearly,
                reference_start_year=self.ref_start.value(),
                reference_end_year=self.ref_end.value(),
                series_label=self._target,
            ).plot(ax=ax, title=f"{self._target} — yearly anomaly vs. "
                                f"{self.ref_start.value()}–{self.ref_end.value()} mean")
        except Exception as err:
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()
