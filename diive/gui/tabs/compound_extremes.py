"""
GUI.TABS.COMPOUND_EXTREMES: COMPOUND-EXTREME DETECTION TAB
==========================================================

Analyze ▸ Compound extremes. Classify months or days into compound-extreme
categories from two drivers' z-scores (``dv.analysis.CompoundExtremes``) and
render the quadrant scatter (``dv.plotting.CompoundExtremesPlot``, after Wang
et al. Fig. 2): high-VPD = *air* extreme, low-SWC = *soil* extreme, both =
*compound*.

Two role-based variable combos (var1 / var2) with availability markers, the
classification parameters, a Run, and a Copy Python button. All computation is
library work; the tab only collects inputs, runs, and renders (strict
GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.analysis.compoundextremes import CompoundExtremes, compound_extremes_to_code
from diive.variables import auto_pick_column
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar

_C_MUTED = "#6B7780"

#: standardize_by combo label -> library value.
_STD_MODES = {
    "Deseasonalized (per month / day)": "season",
    "Whole-record": "record",
}


class CompoundExtremesTab(DiiveTab):
    """Classify and visualize compound extremes from two drivers' z-scores."""

    title = "Compound extremes"
    intro = ("Classify months or days into none / air / soil / compound extremes "
             "from two drivers' standardized anomalies (z-scores). High VPD marks "
             "an atmospheric-dryness extreme, low soil water content a soil-dryness "
             "extreme, both together a compound extreme.")

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._ce: CompoundExtremes | None = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.run_btn = QPushButton("Run")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        self.copy_btn = CopyPythonButton(self._python_code)
        outer.addLayout(build_titlebar(self.title, self.copy_btn, self.run_btn))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.status = QLabel(self.intro)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        # Headline period counts, filled after a run.
        self.hero = HeroBand("COMPOUND", "#FFF3E0", "#E65100")
        rl.addWidget(self.hero)
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)
        return root

    def _build_controls(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(320)
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        # Two role-based variable combos with availability markers.
        cols_box = QGroupBox("Input variables")
        cf = QFormLayout(cols_box)
        self.var1_combo = QComboBox()
        self.var1_combo.setToolTip("First driver, e.g. VPD (vapour pressure deficit).")
        self.var2_combo = QComboBox()
        self.var2_combo.setToolTip("Second driver, e.g. SWC (soil water content).")
        self._avail1 = QLabel("")
        self._avail2 = QLabel("")
        for combo, avail, label in ((self.var1_combo, self._avail1, "Variable 1 (e.g. VPD)"),
                                    (self.var2_combo, self._avail2, "Variable 2 (e.g. SWC)")):
            avail.setStyleSheet("font-size: 11px;")
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(combo, stretch=1)
            rh.addWidget(avail)
            cf.addRow(label, row)
            combo.currentTextChanged.connect(self._refresh_availability)
        v.addWidget(cols_box)

        # Extreme directions + threshold.
        ext_box = QGroupBox("Extremes")
        ef = QFormLayout(ext_box)
        self.dir1_combo = QComboBox()
        self.dir1_combo.addItems(["high", "low"])
        self.dir1_combo.setToolTip("Which tail of variable 1 counts as extreme "
                                   "(high = positive z-score, e.g. high VPD).")
        self.dir2_combo = QComboBox()
        self.dir2_combo.addItems(["high", "low"])
        self.dir2_combo.setCurrentText("low")
        self.dir2_combo.setToolTip("Which tail of variable 2 counts as extreme "
                                   "(low = negative z-score, e.g. low soil water).")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.5, 6.0)
        self.threshold.setSingleStep(0.5)
        self.threshold.setDecimals(1)
        self.threshold.setValue(2.0)
        self.threshold.setToolTip("z-score magnitude that marks an extreme (applies "
                                  "to both variables).")
        ef.addRow("Variable 1 extreme", self.dir1_combo)
        ef.addRow("Variable 2 extreme", self.dir2_combo)
        ef.addRow("Threshold (sigma)", self.threshold)
        v.addWidget(ext_box)

        # Resolution + standardization.
        std_box = QGroupBox("Aggregation & standardization")
        sf = QFormLayout(std_box)
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(["monthly", "daily"])
        self.agg_combo.setToolTip("Temporal resolution to classify.")
        self.std_combo = QComboBox()
        self.std_combo.addItems(list(_STD_MODES.keys()))
        self.std_combo.setToolTip(
            "Deseasonalized: standardize each period against the same calendar "
            "month / day-of-year across years (removes the seasonal cycle, the "
            "standard choice). Whole-record: one mean/std over the whole record.")
        sf.addRow("Resolution", self.agg_combo)
        sf.addRow("Standardize by", self.std_combo)
        v.addWidget(std_box)

        # Category labels.
        lab_box = QGroupBox("Category labels")
        lf = QFormLayout(lab_box)
        self.label1 = QLineEdit("Air")
        self.label2 = QLineEdit("Soil")
        lf.addRow("Variable 1 only", self.label1)
        lf.addRow("Variable 2 only", self.label2)
        v.addWidget(lab_box)

        v.addStretch(1)
        return host

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        cols = [str(c) for c in df.select_dtypes(include="number").columns]
        for combo, needles in ((self.var1_combo, ["VPD"]),
                               (self.var2_combo, ["SWC", "SOIL", "SWP"])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            if cur and cur in cols:
                combo.setCurrentText(cur)
            else:
                for needle in needles:
                    guess = auto_pick_column(cols, needle)
                    if guess:
                        combo.setCurrentText(guess)
                        break
            combo.blockSignals(False)
        self._refresh_availability()

    def _refresh_availability(self, *_) -> None:
        if self._df is None:
            return
        cols = {str(c) for c in self._df.columns}
        for combo, avail in ((self.var1_combo, self._avail1), (self.var2_combo, self._avail2)):
            txt = combo.currentText()
            if not txt:
                avail.setText("")
            elif txt in cols:
                avail.setText("✓")
                avail.setStyleSheet("color: #2E7D32; font-size: 11px;")
            else:
                avail.setText("✗")
                avail.setStyleSheet("color: #C62828; font-size: 11px;")

    # --- state ---------------------------------------------------------
    def _state_controls(self) -> dict:
        return {"var1": self.var1_combo, "var2": self.var2_combo,
                "dir1": self.dir1_combo, "dir2": self.dir2_combo,
                "threshold": self.threshold, "agg": self.agg_combo,
                "std": self.std_combo, "label1": self.label1, "label2": self.label2}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._state_controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._state_controls(), state.get("controls"))
        self._refresh_availability()

    # --- run -----------------------------------------------------------
    def _params(self) -> dict:
        return dict(
            var1=self.var1_combo.currentText(),
            var2=self.var2_combo.currentText(),
            var1_extreme=self.dir1_combo.currentText(),
            var2_extreme=self.dir2_combo.currentText(),
            threshold=float(self.threshold.value()),
            agg=self.agg_combo.currentText(),
            standardize_by=_STD_MODES[self.std_combo.currentText()],
            var1_label=self.label1.text() or "Air",
            var2_label=self.label2.text() or "Soil",
        )

    def _run(self) -> None:
        if self._df is None:
            return
        self.hero.clear()  # drop stale totals; refilled on success
        p = self._params()
        cols = {str(c) for c in self._df.columns}
        if p["var1"] not in cols or p["var2"] not in cols:
            self.status.setText("Pick two valid variables.")
            return
        if p["var1"] == p["var2"]:
            self.status.setText("Variable 1 and variable 2 must differ.")
            return
        try:
            ce = CompoundExtremes(
                var1=self._df[p["var1"]], var2=self._df[p["var2"]],
                agg=p["agg"], var1_extreme=p["var1_extreme"],
                var2_extreme=p["var2_extreme"], threshold=p["threshold"],
                standardize_by=p["standardize_by"],
                var1_label=p["var1_label"], var2_label=p["var2_label"])
        except Exception as exc:  # surface library validation to the user
            self.status.setText(f"Failed: {exc}")
            return
        self._ce = ce
        self._render(ce)

    def _render(self, ce: CompoundExtremes) -> None:
        from diive.core.plotting.compoundextremes import CompoundExtremesPlot
        counts = ce.counts
        summary = ", ".join(f"{label}: {int(n)}" for label, n in counts.items())
        self.status.setText(f"{len(ce.results)} {ce.agg} periods — {summary}.")
        metrics = [(f"{ce.agg.upper()} PERIODS", f"{len(ce.results):,}",
                    "Total classified periods")]
        metrics += [(str(label).upper(), f"{int(n):,}", f"{label} periods")
                    for label, n in counts.items()]
        self.hero.set_metrics(metrics)
        ax = self.canvas.new_axes(1)[0]
        CompoundExtremesPlot.from_compound_extremes(ce).plot(ax=ax)
        self.canvas.draw()

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None:
            return None
        p = self._params()
        cols = {str(c) for c in self._df.columns}
        if p["var1"] not in cols or p["var2"] not in cols or p["var1"] == p["var2"]:
            return None
        return compound_extremes_to_code(
            var1=p["var1"], var2=p["var2"], agg=p["agg"],
            var1_extreme=p["var1_extreme"], var2_extreme=p["var2_extreme"],
            threshold=p["threshold"], standardize_by=p["standardize_by"],
            var1_label=p["var1_label"], var2_label=p["var2_label"])
