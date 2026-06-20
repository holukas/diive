"""
GUI.TABS.GAPFILLING_MDS: MDS GAP-FILLING
========================================

Gap-fill one flux variable with Marginal Distribution Sampling
(``dv.gapfilling.FluxMDS``, Reichstein et al. 2005). Unlike the ML gap-fillers,
MDS is not a trained regressor: it fills gaps by matching similar meteorological
conditions on three fixed drivers — short-wave incoming radiation (SWIN), air
temperature (TA) and vapour pressure deficit (VPD) — within similarity
tolerances, assigning each fill a quality level. So this tab has no free feature
picker, no SHAP and no held-out test score; instead it offers a fixed
three-driver picker + tolerances, and a Results page focused on the
per-quality-level breakdown.

It uses the shared tab primitives — ``tab_chrome`` (title bar + list header),
``WorkerRunner`` (off-thread run), ``SubTabs`` (Model/Results) — rather than the
ML ``MlGapFillingTab`` template, since MDS shares the *chrome* but not the ML
flow. All computation is library work; strict GUI<->library separation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gapfilling.codegen import mds_gapfill_to_code
from diive.gapfilling.mds import FluxMDS
from diive.gui import theme
from diive.gui.tabs._partitioning_base import _auto_pick
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mds_results import MdsResultsPanel
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.sub_tabs import SubTabs
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel
from diive.gui.widgets.worker import WorkerRunner

_C_MUTED = "#6B7780"

#: Heatmap tick / axis-label / colorbar font size — matches the Overview tab.
_HM_FONT = 9

#: Driver specs: combo id, form label, auto-pick needles (UPPERCASE), tooltip.
_DRIVERS = [
    {"key": "swin", "label": "SWIN (W m⁻²)", "needles": ["SW_IN", "SWIN", "RG"],
     "tip": "Short-wave incoming radiation driver, in W m-2."},
    {"key": "ta", "label": "TA (°C)", "needles": ["TA", "TAIR"],
     "tip": "Air-temperature driver, in degrees Celsius."},
    {"key": "vpd", "label": "VPD (kPa)", "needles": ["VPD"],
     "tip": "Vapour-pressure-deficit driver, in kPa (not hPa — divide EddyPro hPa by 10)."},
]


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)
    #: (permille 0-1000, quality, n_remaining) from the worker thread — a queued
    #: connection marshals it safely back to the GUI thread. Permille (not raw
    #: level counts) carries the sub-level progress within a long quality level.
    progress = Signal(int, int, int)


class MdsGapFillingTab(DiiveTab):
    """MDS gap-filling tab: target + three fixed drivers + similarity tolerances."""

    title = "MDS gap-filling"
    method_name = "MDS"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._created: set = set()
        self._target: str = ""
        self._result_df: pd.DataFrame | None = None  # columns to emit on "Add"

        self._sig = _Signals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created
        self._sig.progress.connect(self._on_progress)
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this gap-filling run.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        self.run_btn = QPushButton("Run gap-filling")
        self.run_btn.setToolTip("Fill the target's gaps with MDS similarity matching.")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        self.add_btn = QPushButton("Add results to dataset")
        self.add_btn.setToolTip("Append the gap-filled series and its flag to the variable list.")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")

        self.subtabs = SubTabs()
        self.subtabs.add_page("Model", self._build_model_page())
        self.subtabs.add_page("Results", self._build_results_page())
        self.subtabs.add_corner_separator()
        self.subtabs.add_corner_widget(self.run_btn)
        self.subtabs.add_corner_widget(self.add_btn)
        outer.addWidget(self.subtabs, stretch=1)
        return root

    _list_header = staticmethod(list_header)

    # --- pages ---------------------------------------------------------
    def _build_model_page(self) -> QWidget:
        page = QWidget()
        body = QHBoxLayout(page)
        body.setContentsMargins(10, 4, 10, 4)
        body.setSpacing(0)
        body.addWidget(self._build_inputs())
        body.addWidget(self._build_settings())
        body.addWidget(self._build_results(), stretch=1)
        return page

    def _build_results_page(self) -> QWidget:
        self.results_panel = MdsResultsPanel()
        return self.results_panel

    def _build_inputs(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        intro = QLabel(
            "Click a variable in 'Target (flux)' to set the gap-fill target. MDS "
            "fills gaps from three meteorological drivers — pick the SWIN, TA and "
            "VPD columns on the right.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        row = QHBoxLayout()
        tcol = QVBoxLayout()
        tcol.addWidget(self._list_header("Target (flux)", "click to set target"))
        self.target_list = VariablePanel()
        self.target_list.list.setToolTip("Click a variable to set it as the gap-fill target.")
        self.target_list.selected.connect(lambda name, _c: self._set_target(name))
        tcol.addWidget(self.target_list, stretch=1)
        row.addLayout(tcol)
        row.addWidget(self._build_driver_box(), stretch=1)
        v.addLayout(row, stretch=1)

        self.target_label = QLabel("Target: (none)")
        self.target_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.target_label)
        return host

    def _build_driver_box(self) -> QWidget:
        box = QGroupBox("Meteorological drivers")
        form = QFormLayout(box)
        self._combos: dict[str, QComboBox] = {}
        self._avail: dict[str, QLabel] = {}
        for spec in _DRIVERS:
            combo = QComboBox()
            combo.setToolTip(spec["tip"])
            combo.currentTextChanged.connect(self._refresh_availability)
            mark = QLabel("")
            mark.setToolTip("Whether the chosen column is present in the dataset.")
            cell = QWidget()
            ch = QHBoxLayout(cell)
            ch.setContentsMargins(0, 0, 0, 0)
            ch.setSpacing(6)
            ch.addWidget(combo, stretch=1)
            ch.addWidget(mark)
            form.addRow(spec["label"], cell)
            self._combos[spec["key"]] = combo
            self._avail[spec["key"]] = mark
        return box

    def _build_settings(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(320)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(self._build_tol_box())
        v.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        return host

    def _build_tol_box(self) -> QGroupBox:
        box = QGroupBox("Similarity tolerances")
        f = QFormLayout(box)
        self.swin_tol_low = QSpinBox(); self.swin_tol_low.setRange(0, 1000)
        self.swin_tol_low.setValue(20)
        self.swin_tol_low.setToolTip(
            "SWIN tolerance at LOW radiation (W m-2). A complete record matches the "
            "gap when its SWIN is within this band — tighter low-light matching.")
        f.addRow("SWIN tol (low)", self.swin_tol_low)
        self.swin_tol_high = QSpinBox(); self.swin_tol_high.setRange(0, 1000)
        self.swin_tol_high.setValue(50)
        self.swin_tol_high.setToolTip(
            "SWIN tolerance at HIGH radiation (W m-2). The band widens with "
            "radiation between the low and high values.")
        f.addRow("SWIN tol (high)", self.swin_tol_high)
        self.ta_tol = QDoubleSpinBox(); self.ta_tol.setRange(0.1, 50.0)
        self.ta_tol.setDecimals(1); self.ta_tol.setSingleStep(0.5); self.ta_tol.setValue(2.5)
        self.ta_tol.setToolTip("Air-temperature match tolerance, in degrees Celsius.")
        f.addRow("TA tol (°C)", self.ta_tol)
        self.vpd_tol = QDoubleSpinBox(); self.vpd_tol.setRange(0.05, 50.0)
        self.vpd_tol.setDecimals(2); self.vpd_tol.setSingleStep(0.1); self.vpd_tol.setValue(0.5)
        self.vpd_tol.setToolTip("VPD match tolerance, in kPa.")
        f.addRow("VPD tol (kPa)", self.vpd_tol)
        self.avg_min_n_vals = QSpinBox(); self.avg_min_n_vals.setRange(0, 1000)
        self.avg_min_n_vals.setValue(2)
        self.avg_min_n_vals.setToolTip(
            "Minimum number of matching records required to compute a gap-fill "
            "average. ONEFlux uses 2; larger values are stricter (a gap with "
            "fewer matches falls through to a looser stage).")
        f.addRow("Min samples for mean", self.avg_min_n_vals)
        self.sym_mean = QCheckBox("Symmetric mean (Vekuri 2023)")
        self.sym_mean.setChecked(False)
        self.sym_mean.setToolTip(
            "Use the Vekuri (2023) symmetric mean for the SWIN-driven methods "
            "(splits similar samples by radiation above/below the target and "
            "averages the two halves) instead of the plain mean. Off by default "
            "(standard ONEFlux).")
        f.addRow("", self.sym_mean)
        return box

    def _build_results(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.status = QLabel("Set a target and the three drivers, then run gap-filling.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        v.addWidget(self.status)
        # Progress over the MDS quality levels — hidden until a run starts.
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(16)
        self.progress.setVisible(False)
        pwrap = QWidget()
        pl = QVBoxLayout(pwrap)
        pl.setContentsMargins(10, 0, 10, 4)
        pl.addWidget(self.progress)
        v.addWidget(pwrap)
        v.addWidget(self._build_heatmaps(), stretch=1)
        return host

    def _build_heatmaps(self) -> QWidget:
        col = QWidget()
        cv = QVBoxLayout(col)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        headers = QHBoxLayout()
        headers.setContentsMargins(0, 0, 0, 0)
        for text in ("Observed (with gaps)", "MDS gap-filled"):
            lbl = self._panel_header(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            headers.addWidget(lbl, stretch=1)
        cv.addLayout(headers)
        self.canvas = MplCanvas()
        cv.addWidget(self.canvas, stretch=1)
        return col

    @staticmethod
    def _panel_header(text: str) -> QLabel:
        lbl = QLabel(theme.manager.label_text(text))
        f = theme.manager.tracked_font(lbl.font())
        f.setBold(True)
        lbl.setFont(f)
        return lbl

    # --- data / inputs -------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._created = set(created or set())
        self._all_cols = [str(c) for c in df.columns]
        self._result_df = None
        self.add_btn.setEnabled(False)

        self.target_list.set_variables(self._all_cols, self._created)
        if self._target not in self._all_cols:
            self._target = ""
        self._refresh_inputs()
        self.target_list.set_panels([self._target] if self._target else [])
        self._update_target_label()
        self._update_status()

    def _set_target(self, name: str) -> None:
        if not name or name not in self._all_cols:
            return
        self._target = name
        self.target_list.set_panels([name])
        self._update_target_label()
        self._update_status()

    def _update_target_label(self) -> None:
        self.target_label.setText(f"Target: {self._target or '(none)'}")

    def _refresh_inputs(self) -> None:
        """Repopulate the driver combos from the current columns; keep the user's
        pick if it survives, else auto-seed by name (skipping FLAG_* columns)."""
        cols = self._all_cols
        for spec in _DRIVERS:
            combo = self._combos[spec["key"]]
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(cols)
            if cur in cols:
                combo.setCurrentText(cur)
            else:
                guess = ""
                for needle in spec["needles"]:  # try each naming alternative
                    # Prefer a gap-filled driver ('_f'); never pick a flag column.
                    guess = _auto_pick(cols, needle, prefer="_F", avoid="FLAG")
                    if guess:
                        break
                if guess:
                    combo.setCurrentText(guess)
            combo.blockSignals(False)
        self._refresh_availability()

    def _refresh_availability(self, *_) -> None:
        for key, combo in self._combos.items():
            present = combo.currentText() in self._all_cols
            mark = self._avail[key]
            mark.setText("✓" if present else "✗")
            mark.setStyleSheet(f"color: {'#2E7D32' if present else '#C62828'}; font-weight: bold;")
        self._update_status()

    def _driver_names(self) -> dict:
        return {k: c.currentText() for k, c in self._combos.items()}

    def _update_status(self, *_) -> None:
        if getattr(self, "_runner", None) is not None and self._runner.is_running:
            return
        if getattr(self, "status", None) is None:
            return
        if not self._target:
            self.status.setText("Click a variable in 'Target (flux)' to set the gap-fill target.")
            return
        drivers = self._driver_names()
        missing = [k.upper() for k, v in drivers.items() if v not in self._all_cols]
        if missing:
            self.status.setText(
                f"Target: {self._target}. Pick valid columns for: {', '.join(missing)}.")
        else:
            self.status.setText(
                f"Target: {self._target} — drivers SWIN={drivers['swin']}, "
                f"TA={drivers['ta']}, VPD={drivers['vpd']}. Run gap-filling.")

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {**self._combos,
                "swin_tol_low": self.swin_tol_low, "swin_tol_high": self.swin_tol_high,
                "ta_tol": self.ta_tol, "vpd_tol": self.vpd_tol,
                "avg_min_n_vals": self.avg_min_n_vals, "sym_mean": self.sym_mean}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target, "controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        tgt = state.get("target")
        if tgt in self._all_cols:
            self._set_target(tgt)
        self._update_status()

    # --- run -----------------------------------------------------------
    def _method_kwargs(self) -> dict:
        return {
            "swin_tol": [self.swin_tol_low.value(), self.swin_tol_high.value()],
            "ta_tol": self.ta_tol.value(),
            "vpd_tol": self.vpd_tol.value(),
            "avg_min_n_vals": self.avg_min_n_vals.value(),
            "sym_mean": self.sym_mean.isChecked(),
        }

    def _python_code(self) -> str | None:
        target = self._target
        drivers = self._driver_names()
        if not target or any(v not in self._all_cols for v in drivers.values()):
            return None
        return mds_gapfill_to_code(
            target, drivers["swin"], drivers["ta"], drivers["vpd"], self._method_kwargs())

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        target = self._target
        if not target:
            self.status.setText("Set a target (flux) variable first.")
            return
        drivers = self._driver_names()
        missing = [k.upper() for k, v in drivers.items() if v not in self._all_cols]
        if missing:
            self.status.setText(f"Pick valid driver columns for: {', '.join(missing)}.")
            return
        swin, ta, vpd = drivers["swin"], drivers["ta"], drivers["vpd"]
        if len({target, swin, ta, vpd}) < 4:
            self.status.setText("Target and the three drivers must be four distinct columns.")
            return
        kwargs = self._method_kwargs()
        work = self._df[[target, swin, ta, vpd]].copy()
        self._set_running(True)
        # Indeterminate (busy) until the first quality level reports in.
        self.progress.setRange(0, 0)
        self.progress.setFormat("Preparing…")
        self.progress.setVisible(True)
        self.status.setText("Gap-filling… (MDS similarity matching — this can take a while)")
        self._runner.run(self._compute_payload, work, target, swin, ta, vpd, kwargs)

    def _compute_payload(self, work, target, swin, ta, vpd, kwargs):
        """Gap-fill off-thread; return the tuple consumed by :meth:`_on_done`.
        Raises on error (the runner forwards to :meth:`_on_failed`)."""
        # verbose=2 (PROGRESS): stream the library's MDS report into the Log tab.
        model = FluxMDS(df=work, flux=target, swin=swin, ta=ta, vpd=vpd,
                        verbose=2, **kwargs)
        # Marshal the worker-thread progress to the GUI via a queued signal. The
        # callback reports gaps filled / total gaps, so the permille (and thus the
        # bar percentage) is gap-based, not quality-level-based.
        model.run(progress_callback=lambda filled, total, quality, n_filled, remaining:
                  self._sig.progress.emit(
                      int(round(1000 * filled / total)) if total else 0, quality, remaining))
        gapfilled = model.get_gapfilled_target()
        flag = model.get_flag()
        out = pd.DataFrame({str(gapfilled.name): gapfilled, str(flag.name): flag})
        out.attrs[ATTRS_KEY] = {
            str(gapfilled.name): provenance_attr(
                origin=DERIVED, parent=str(target), operation=self.title,
                tags=["gapfilling", "mds"]),
            str(flag.name): provenance_attr(
                origin=DERIVED, parent=str(target), operation=self.title,
                tags=["gapfilling", "flag"]),
        }
        return out, work[target], gapfilled, model, target

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    def _on_progress(self, permille: int, quality: int, remaining: int) -> None:
        """Update the progress bar as the MDS quality levels are worked (queued
        from the worker thread). ``permille`` (0-1000) carries the within-level
        progress, so the bar moves continuously through the slow early levels."""
        self.progress.setRange(0, 1000)
        self.progress.setValue(max(0, min(1000, permille)))
        self.progress.setFormat(
            f"Quality level {quality}  ·  {remaining:,} gaps remaining  ·  {permille / 10:.0f}%")

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        out, observed, gapfilled, model, target = payload
        self._set_running(False)
        self.progress.setVisible(False)
        self._result_df = out
        n_filled = int((out[str(gapfilled.name)].notna() & observed.isna()).sum())
        self.status.setText(
            f"Done — filled {n_filled:,} gaps. 'Add' appends {', '.join(out.columns)}.")
        self.add_btn.setEnabled(True)
        self._plot(observed, gapfilled)
        self.results_panel.update(model, target)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.progress.setVisible(False)
        self.status.setText(f"Failed: {msg}")
        ax = self.canvas.new_axes(1)[0]
        ax.text(0.5, 0.5, "Gap-filling failed", ha="center", va="center",
                transform=ax.transAxes)
        self.canvas.draw()
        self.results_panel.reset(f"Gap-filling failed: {msg}")

    def _plot(self, observed, gapfilled) -> None:
        """Side-by-side date/time heatmaps: observed (with gaps) vs gap-filled,
        sharing one value scale + one colorbar (only the right panel draws it)."""
        ax_obs, ax_gf = self.canvas.new_axes(2, sharey=True)
        try:
            both = pd.concat([observed.dropna(), gapfilled.dropna()])
            vmin = float(np.nanpercentile(both, 1)) if len(both) else None
            vmax = float(np.nanpercentile(both, 99)) if len(both) else None
            opts = {"vmin": vmin, "vmax": vmax, "cb_labelsize": _HM_FONT}
            hm_style = dv.plotting.FormatStyle(ticks_fontsize=_HM_FONT, axlabel_fontsize=_HM_FONT)
            dv.plotting.HeatmapDateTime(series=observed).plot(
                ax=ax_obs, format_style=hm_style, show_colormap=False, **opts)
            dv.plotting.HeatmapDateTime(series=gapfilled).plot(ax=ax_gf, format_style=hm_style, **opts)
            ax_obs.set_title("")
            ax_gf.set_title("")
            ax_gf.set_ylabel("")
            ax_gf.tick_params(labelleft=False)
        except Exception as err:  # plotting must never crash the tab
            ax_obs.text(0.5, 0.5, f"Plot failed: {err}", ha="center", va="center",
                        transform=ax_obs.transAxes, fontsize=8)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {', '.join(result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
