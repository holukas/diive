"""
GUI.TABS.TIMELAG: TIME-LAG ANALYSIS (EC concentration ↔ wind pairing)
=====================================================================

"What is the optimal time lag for each gas?" Eddy-covariance gas analysers sit
downstream of the sonic anemometer, so a gas signal arrives a fraction of a
second after the wind. This tab analyses the distribution of measured lags
(`*_TLAG_ACTUAL` columns), detects the peak lag and its range via gradient-based
edge detection, and formats an EddyPro-ready search window.

All computation + plotting is the library's `dv.flux.TimeLagAnalysis`; this tab
only collects parameters, picks the gas, and embeds the 4-panel figure in its
canvas (strict GUI<->library separation). Needs Level-0 data with TLAG columns
(e.g. `load_exampledata_parquet_tlag_vars_level0`) — the **Load example TLAG
data** button loads that bundled dataset locally so the feature is usable even
when the active dataset has no lag columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar

#: Suffix marking the measured-lag columns the analysis consumes.
_TLAG_SUFFIX = "_TLAG_ACTUAL"

_EXPLANATION = (
    "<b>Time-lag analysis.</b> A gas analyser is downstream of the sonic "
    "anemometer, so each gas signal lags the wind by a fraction of a second. "
    "This view builds a histogram of the measured lags for a gas, marks the "
    "<b>peak</b> lag (black), the <b>detected range</b> around it (teal), an "
    "<b>EddyPro</b>-ready search window expanded by one 0.05&nbsp;s step (orange), "
    "and your <b>reference</b> acceptable window (purple). Use the EddyPro range as "
    "the covariance-maximisation search window for that gas."
)


class TimeLagAnalysisTab(DiiveTab):
    """Time-lag distribution analysis for EC gas channels."""

    title = "Time lag"

    def build(self) -> QWidget:
        self._df = None          # active dataset (app-pushed or example)
        self._gases: list[str] = []

        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root_lay.addLayout(build_titlebar(self.title))  # shared tab header
        body = QWidget()
        row = QHBoxLayout(body)

        # Left: fixed-width, scrollable config column.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.addWidget(self._build_explanation())
        col.addWidget(self._gas_group())
        col.addWidget(self._params_group())
        self.run_btn = QPushButton("Analyze && plot")
        self.run_btn.clicked.connect(self._analyze)
        col.addWidget(self.run_btn)
        self.example_btn = QPushButton("Load example TLAG data")
        self.example_btn.clicked.connect(self._load_example)
        col.addWidget(self.example_btn)
        col.addStretch(1)
        scroll.setWidget(inner)
        row.addWidget(scroll)

        # Right: embedded figure + results readout.
        right = QVBoxLayout()
        self.canvas = MplCanvas()
        # plot_gas() lays out its own 4-panel gridspec with explicit margins;
        # keep the canvas from re-flowing it (like the ridgeline tab).
        self.canvas.auto_layout = False
        self.canvas.fig.set_layout_engine("none")
        right.addWidget(self.canvas, stretch=1)
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setMaximumHeight(120)
        right.addWidget(self.results)
        row.addLayout(right, stretch=1)

        root_lay.addWidget(body, stretch=1)
        self._update_availability()
        return root

    # --- sub-widgets ---------------------------------------------------
    def _build_explanation(self) -> QWidget:
        lbl = QLabel(_EXPLANATION)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        list_bg = theme.manager.tokens["LIST_BG"]
        border = theme.manager.tokens["BORDER"]
        lbl.setStyleSheet(
            f"QLabel {{ background: {list_bg}; border: 1px solid {border};"
            f" border-radius: 6px; padding: 8px 10px; color: #37474F; }}")
        return lbl

    def _gas_group(self) -> QGroupBox:
        box = QGroupBox("Gas")
        form = QFormLayout(box)
        self.gas = QComboBox()
        self.gas.setToolTip("Gas channel to analyse (from *_TLAG_ACTUAL columns).")
        form.addRow("Channel", self.gas)
        return box

    def _params_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)

        self.fringe_low = self._spin(5, 0, 100)
        self.fringe_low.setToolTip(
            "Lower fringe bin to exclude (edge bins accumulate non-physical lags).")
        self.fringe_high = self._spin(10, 0, 100)
        self.fringe_high.setToolTip("Upper fringe bin to exclude.")
        fringe = QHBoxLayout()
        fringe.setContentsMargins(0, 0, 0, 0)
        fringe.addWidget(self.fringe_low)
        fringe.addWidget(self.fringe_high)
        fringe_w = QWidget()
        fringe_w.setLayout(fringe)
        form.addRow("Ignore fringe bins", fringe_w)

        self.lag_window_min = self._dspin(0.05, -10, 10, 2, 0.05)
        self.lag_window_min.setToolTip("Lower bound of the reference acceptable lag window (s).")
        form.addRow("Window min (s)", self.lag_window_min)
        self.lag_window_max = self._dspin(1.00, -10, 10, 2, 0.05)
        self.lag_window_max.setToolTip("Upper bound of the reference acceptable lag window (s).")
        form.addRow("Window max (s)", self.lag_window_max)

        self.hist_startbin = self._spin(0, -100, 100)
        self.hist_startbin.setToolTip("First histogram bin to display/analyse.")
        form.addRow("Histogram start bin", self.hist_startbin)
        self.hist_endbin = self._spin(10, -100, 100)
        self.hist_endbin.setToolTip("Last histogram bin to display/analyse.")
        form.addRow("Histogram end bin", self.hist_endbin)

        self.gradient_threshold = self._dspin(0.15, 0.0, 1.0, 2, 0.01)
        self.gradient_threshold.setToolTip(
            "Edge-detection sensitivity (lower = stricter, narrower peak range).")
        form.addRow("Gradient threshold", self.gradient_threshold)

        self.zoom_before = self._dspin(0.5, 0.0, 10.0, 2, 0.1)
        self.zoom_before.setToolTip("Zoom offset before the peak (s).")
        self.zoom_after = self._dspin(0.8, 0.0, 10.0, 2, 0.1)
        self.zoom_after.setToolTip("Zoom offset after the peak (s).")
        zoom = QHBoxLayout()
        zoom.setContentsMargins(0, 0, 0, 0)
        zoom.addWidget(self.zoom_before)
        zoom.addWidget(self.zoom_after)
        zoom_w = QWidget()
        zoom_w.setLayout(zoom)
        form.addRow("Zoom margin (s)", zoom_w)
        return box

    @staticmethod
    def _spin(value, lo, hi) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(value)
        return sp

    @staticmethod
    def _dspin(value, lo, hi, decimals, step) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(value)
        return sp

    # --- data flow -----------------------------------------------------
    @staticmethod
    def _detect_gases(df) -> list[str]:
        return [str(c)[:-len(_TLAG_SUFFIX)] for c in df.columns
                if str(c).endswith(_TLAG_SUFFIX)]

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._refresh_gases()
        self._update_availability()

    def _refresh_gases(self) -> None:
        gases = self._detect_gases(self._df) if self._df is not None else []
        self._gases = gases
        current = self.gas.currentText()
        self.gas.blockSignals(True)
        self.gas.clear()
        self.gas.addItems(gases)
        if current in gases:
            self.gas.setCurrentText(current)
        self.gas.blockSignals(False)

    def _update_availability(self) -> None:
        has = bool(self._gases)
        self.run_btn.setEnabled(has)
        if not has:
            self.results.setPlainText(
                "No *_TLAG_ACTUAL columns in the active dataset.\n"
                "Click 'Load example TLAG data' to explore the feature.")

    def _load_example(self) -> None:
        try:
            from diive.configs.exampledata import (
                load_exampledata_parquet_tlag_vars_level0,
            )
            self._df = load_exampledata_parquet_tlag_vars_level0()
        except Exception as err:
            self.results.setPlainText(f"Could not load example data:\n{err}")
            return
        self._refresh_gases()
        self._update_availability()
        if self._gases:
            self._analyze()

    # --- analysis (library work) ---------------------------------------
    def _analyze(self) -> None:
        gas = self.gas.currentText()
        if not gas or self._df is None:
            return
        # Fresh instance each run so parameter edits aren't masked by its cache.
        analysis = dv.flux.TimeLagAnalysis(
            df=self._df,
            ignore_fringe_bins=[self.fringe_low.value(), self.fringe_high.value()],
            lag_window_min=self.lag_window_min.value(),
            lag_window_max=self.lag_window_max.value(),
            histogram_startbin=self.hist_startbin.value(),
            histogram_endbin=self.hist_endbin.value(),
            gradient_threshold=self.gradient_threshold.value(),
            zoom_margin=[self.zoom_before.value(), self.zoom_after.value()],
        )
        try:
            res = analysis.analyze_gas(gas)
            self.canvas.fig.set_layout_engine("none")
            analysis.plot_gas(gas, fig=self.canvas.fig, show=False)
        except Exception as err:
            self.results.setPlainText(f"Analysis failed for {gas}:\n{err}")
            return
        self.results.setPlainText(
            f"{gas}  ·  {res['first_date']} – {res['last_date']}\n"
            f"Peak lag:       {res['peak']:.3f} s\n"
            f"Detected range: {res['peak_min']:.3f} – {res['peak_max']:.3f} s\n"
            f"EddyPro input:  {res['eddypro_min']:.3f} – {res['eddypro_max']:.3f} s")
        self.canvas.draw()

    # --- project save/restore ------------------------------------------
    def _controls(self) -> dict:
        return {"gas": self.gas, "fringe_low": self.fringe_low,
                "fringe_high": self.fringe_high,
                "lag_window_min": self.lag_window_min,
                "lag_window_max": self.lag_window_max,
                "hist_startbin": self.hist_startbin,
                "hist_endbin": self.hist_endbin,
                "gradient_threshold": self.gradient_threshold,
                "zoom_before": self.zoom_before, "zoom_after": self.zoom_after}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls") or {})
