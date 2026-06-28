"""
GUI.TABS.UNCERTAINTY_RANDUNC: RANDOM UNCERTAINTY (PAS20)
=======================================================

Estimate the random measurement uncertainty of a flux with the hierarchical
4-method PAS20 approach (Pastorello et al. 2020), a faithful ONEFlux port —
``dv.flux.RandomUncertaintyPAS20``. The tab collects the measured + gap-filled
flux and the three similarity drivers (TA, VPD, SW_IN), runs the library class
on a worker thread (with a progress bar over the four methods), previews the
flux ± uncertainty band, the cumulative uncertainty bounds and the
measured-flux vs uncertainty relationship, and emits the ``{flux}_RANDUNC``
column.

Layout mirrors the data-correction tabs (e.g. *Set to value*): a fixed-width
**Settings** column (header -> description -> input/option controls -> Run ->
status -> Add) beside a method **hero** band over the preview. It is a
standalone ``DiiveTab`` sharing the common chrome (title bar, ``WorkerRunner``,
the column-picker idiom, the hero helpers) rather than subclassing the
partitioning base — the result is a single uncertainty column, not GPP/RECO.
All computation is library work (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.flux import RandomUncertaintyPAS20
from diive.flux.lowres.codegen import randunc_to_code
from diive.gui import theme
from diive.gui.tabs._partitioning_base import _auto_pick
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import _MetricSlot, _chip, _stat_separator
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.worker import WorkerRunner

_C_FLUX = "#455A64"   # blue-grey 700 — the flux line
_C_UNC = "#E53935"    # red 600       — the uncertainty band / bounds
_C_MUTED = "#6B7780"

#: Input specs: combo id, form label, auto-pick needle(s), prefer/avoid, tooltip.
_INPUTS = [
    {"key": "flux", "label": "Flux (measured)", "needle": "NEE",
     "prefer": "ORIG", "avoid": "_F",
     "tip": "Measured flux to estimate the random uncertainty for (e.g. NEE, µmol m⁻² s⁻¹)."},
    {"key": "flux_f", "label": "Flux (gap-filled)", "needle": "NEE",
     "prefer": "_F",
     "tip": "Gap-filled flux — used for the cumulative uncertainty propagation."},
    {"key": "ta", "label": "TA (°C)", "needle": ["TA", "TAIR"],
     "prefer": "_F",
     "tip": "Air-temperature similarity driver, in degrees Celsius."},
    {"key": "vpd", "label": "VPD (kPa)", "needle": "VPD",
     "prefer": "_F",
     "tip": "Vapour-pressure-deficit similarity driver (unit set by the toggle below)."},
    {"key": "swin", "label": "SW_IN (W m⁻²)", "needle": ["SW_IN", "RG"],
     "prefer": "_F", "avoid": "POT",
     "tip": "Short-wave incoming radiation similarity driver, in W m⁻²."},
]


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)
    #: (permille 0-1000, phase 1-4) from the worker thread; a queued connection
    #: marshals it safely back to the GUI thread.
    progress = Signal(int, int)


class RandomUncertaintyTab(DiiveTab):
    """Random uncertainty (PAS20): measured + gap-filled flux + three drivers."""

    title = "Random uncertainty (PAS20)"
    intro = ("Estimate the random measurement uncertainty of a flux with the "
             "hierarchical 4-method PAS20 approach (Pastorello et al. 2020, "
             "ONEFlux port). Emits a {flux}_RANDUNC column.")
    #: Hero chip (method identity), mirroring the correction / gap-filling tabs.
    method_chip_label = "RANDOM UNCERTAINTY"
    method_chip_bg = "#FFEBEE"
    method_chip_fg = "#C62828"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._result_df: pd.DataFrame | None = None  # column to emit on "Add"
        self._combos: dict[str, QComboBox] = {}
        self._avail: dict[str, QLabel] = {}

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

        # Title bar (correction-/gap-filling-tab style): bold title with Copy
        # Python at the far-right edge.
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this uncertainty estimate.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        # Body: settings column | hero + preview (mirrors the correction tabs).
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.addWidget(self._build_settings())
        layout.addWidget(self._build_right(), stretch=1)
        outer.addWidget(body, stretch=1)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(320)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        # Settings header + description below it.
        outer.addWidget(list_header("Settings", "set inputs & run"))
        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(intro)

        # Input columns.
        cols_box = QGroupBox("Input columns")
        cf = QFormLayout(cols_box)
        for spec in _INPUTS:
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
            cf.addRow(spec["label"], cell)
            self._combos[spec["key"]] = combo
            self._avail[spec["key"]] = mark
        outer.addWidget(cols_box)

        # Options.
        opt_box = QGroupBox("Options")
        of = QVBoxLayout(opt_box)
        self.vpd_in_kpa = QCheckBox("VPD is in kPa (diive convention)")
        self.vpd_in_kpa.setChecked(True)
        self.vpd_in_kpa.setToolTip(
            "Checked: the VPD column is in kPa and converted to hPa internally for "
            "the ONEFlux 5-hPa similarity tolerance. Uncheck if it is already in hPa.")
        of.addWidget(self.vpd_in_kpa)
        outer.addWidget(opt_box)

        note = QLabel(
            "Similarity tolerances and window sizes follow the ONEFlux reference "
            "(TA ±2.5 °C, VPD ±5 hPa, SW_IN-dependent; ±7 d / ±1 h then ±14 d) and "
            "are not adjustable here.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-style: italic;")
        outer.addWidget(note)

        # Buttons below the settings: Run, then status + progress, then Add.
        self.run_btn = QPushButton("Run uncertainty")
        self.run_btn.setToolTip("Estimate the random uncertainty with the 4-method PAS20 cascade.")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        self.status = QLabel("Pick the input columns, then run.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(16)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        self.add_btn = QPushButton("Add result to dataset")
        self.add_btn.setToolTip("Append the {flux}_RANDUNC column to the variable list.")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        return panel

    def _build_right(self) -> QWidget:
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self._build_hero())
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)
        return right

    def _build_hero(self) -> QWidget:
        """A slim band: the method chip on the left, a stats strip on the right
        (filled in after a run via :meth:`_hero_metrics`)."""
        hero = QFrame()
        hero.setStyleSheet("QFrame { background: #FAFAFA; border-radius: 8px; }")
        h = QHBoxLayout(hero)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(12)
        h.addWidget(_chip(self.method_chip_label, self.method_chip_bg, self.method_chip_fg))
        h.addStretch(1)
        self._metrics_host = QWidget()
        self._metrics_lay = QHBoxLayout(self._metrics_host)
        self._metrics_lay.setContentsMargins(0, 0, 0, 0)
        self._metrics_lay.setSpacing(12)
        h.addWidget(self._metrics_host)
        return hero

    def _clear_hero(self) -> None:
        while self._metrics_lay.count():
            w = self._metrics_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _update_hero(self, metrics: list) -> None:
        self._clear_hero()
        for i, (name, value, tip) in enumerate(metrics):
            if i > 0:
                self._metrics_lay.addWidget(_stat_separator())
            slot = _MetricSlot()
            slot.update_metric(name, value, tip)
            self._metrics_lay.addWidget(slot)

    def _hero_metrics(self, randunc) -> list:
        unc = randunc.randunc_series
        cum = randunc.randunc_results_cumulatives["UNC_CUMULATIVE"]
        cum_last = float(cum.iloc[-1]) if len(cum) and pd.notna(cum.iloc[-1]) else float("nan")
        return [
            ("MEAN ±σ", f"{unc.mean():.3f}", "Mean per-record random uncertainty"),
            ("MEDIAN ±σ", f"{unc.median():.3f}", "Median per-record random uncertainty"),
            ("RECORDS", f"{int(unc.notna().sum()):,}",
             "Records with a random-uncertainty estimate"),
            ("CUMULATIVE ±σ", f"{cum_last:.1f}",
             "Final cumulative (quadrature) uncertainty of the summed flux"),
        ]

    # --- data / inputs -------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._all_cols = [str(c) for c in df.columns]
        self._result_df = None
        self.add_btn.setEnabled(False)
        self._clear_hero()
        for spec in _INPUTS:
            combo = self._combos[spec["key"]]
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._all_cols)
            if cur in self._all_cols:
                combo.setCurrentText(cur)
            else:
                needles = spec["needle"]
                if isinstance(needles, str):
                    needles = [needles]
                for needle in needles:
                    guess = _auto_pick(self._all_cols, needle,
                                       prefer=spec.get("prefer"), avoid=spec.get("avoid"))
                    if guess:
                        combo.setCurrentText(guess)
                        break
            combo.blockSignals(False)
        self._refresh_availability()

    def _refresh_availability(self, *_) -> None:
        if self._df is None:
            return
        for key, combo in self._combos.items():
            present = combo.currentText() in self._all_cols
            mark = self._avail[key]
            mark.setText("✓" if present else "✗")
            mark.setStyleSheet(f"color: {'#2E7D32' if present else '#C62828'}; font-weight: bold;")

    def _picks(self) -> dict[str, str]:
        return {k: c.currentText() for k, c in self._combos.items()}

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {**self._combos, "vpd_in_kpa": self.vpd_in_kpa}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        self._refresh_availability()

    # --- run -----------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None:
            return None
        picks = self._picks()
        cols = set(self._all_cols)
        if any(picks[s["key"]] not in cols for s in _INPUTS):
            self.status.setText("Pick valid input columns first to copy the code.")
            return None
        return randunc_to_code(
            fluxcol=picks["flux"], fluxgapfilledcol=picks["flux_f"],
            tacol=picks["ta"], vpdcol=picks["vpd"], swincol=picks["swin"],
            vpd_in_kpa=self.vpd_in_kpa.isChecked())

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        picks = self._picks()
        cols = set(self._all_cols)
        missing = [s["label"] for s in _INPUTS if picks[s["key"]] not in cols]
        if missing:
            self.status.setText("Pick valid columns for: " + ", ".join(missing))
            return
        if len(set(picks.values())) < len(picks):
            self.status.setText("The five input columns must be distinct.")
            return

        work = self._df[[picks["flux"], picks["flux_f"], picks["ta"],
                         picks["vpd"], picks["swin"]]].copy()
        self._set_running(True)
        self._clear_hero()
        self.progress.setRange(0, 0)  # busy until the first method reports in
        self.progress.setFormat("Preparing…")
        self.progress.setVisible(True)
        self.status.setText("Estimating random uncertainty… (per-record cascade — this can take a while)")
        self._runner.run(self._compute_payload, work, picks, self.vpd_in_kpa.isChecked())

    def _compute_payload(self, work, picks, vpd_in_kpa):
        """Run the library class off-thread; return the tuple consumed by
        :meth:`_on_done`. Raises on error (the runner forwards to _on_failed)."""
        randunc = RandomUncertaintyPAS20(
            df=work, fluxcol=picks["flux"], fluxgapfilledcol=picks["flux_f"],
            tacol=picks["ta"], vpdcol=picks["vpd"], swincol=picks["swin"],
            vpd_in_kpa=vpd_in_kpa)
        randunc.run(progress_callback=lambda phase, n_phases, done, total:
                    self._sig.progress.emit(
                        int(round(1000 * ((phase - 1) + done / max(total, 1)) / n_phases)),
                        phase))
        randunc_col = randunc.randunccol
        out = pd.DataFrame({randunc_col: randunc.randunc_series})
        out.attrs[ATTRS_KEY] = {
            randunc_col: provenance_attr(
                origin=DERIVED, parent=str(picks["flux"]), operation=self.title,
                tags=["uncertainty", "randunc"]),
        }
        return out, randunc

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    def _on_progress(self, permille: int, phase: int) -> None:
        self.progress.setRange(0, 1000)
        self.progress.setValue(max(0, min(1000, permille)))
        self.progress.setFormat(f"Method {phase} of 4  ·  {permille / 10:.0f}%")

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        out, randunc = payload
        self._set_running(False)
        self.progress.setVisible(False)
        self._result_df = out
        randunc_col = out.columns[0]
        results = randunc.randunc_results
        counts = [int(results[f"WINDOW_N_VALS_METHOD{m}"].notna().sum()) for m in (1, 2, 3, 4)]
        self._update_hero(self._hero_metrics(randunc))
        self.status.setText(
            f"Done. Method 1/2/3/4 records: "
            f"{counts[0]:,} / {counts[1]:,} / {counts[2]:,} / {counts[3]:,}. "
            f"'Add' appends {randunc_col}.")
        self.add_btn.setEnabled(True)
        self._plot(randunc)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.progress.setVisible(False)
        self._clear_hero()
        self.status.setText(f"Failed: {msg}")
        ax = self.canvas.new_axes(1)[0]
        ax.text(0.5, 0.5, "Uncertainty estimation failed", ha="center", va="center",
                transform=ax.transAxes)
        self.canvas.draw()

    def _plot(self, randunc) -> None:
        """Three panels: the flux ± uncertainty band (daily means) spanning the
        whole top row, with the cumulative flux + its propagated bounds (bottom
        left) and the measured-flux vs random-uncertainty relationship (bottom
        right — the Hollinger & Richardson 2005 scaling: random uncertainty grows
        with flux magnitude)."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        gs = fig.add_gridspec(2, 2)
        ax_band = fig.add_subplot(gs[0, :])
        ax_cum = fig.add_subplot(gs[1, 0], sharex=ax_band)
        ax_sc = fig.add_subplot(gs[1, 1])
        try:
            res = randunc.randunc_results
            flux = res[randunc.fluxcol]
            unc = res[randunc.randunccol]
            d_flux = flux.resample("1D").mean()
            d_unc = unc.resample("1D").mean()
            ax_band.fill_between(d_flux.index, (d_flux - d_unc).to_numpy(),
                                 (d_flux + d_unc).to_numpy(), color=_C_UNC, alpha=0.25,
                                 label="± random uncertainty")
            ax_band.plot(d_flux.index, d_flux.to_numpy(), color=_C_FLUX, lw=1.0,
                         label=f"{randunc.fluxcol} (daily mean)")
            ax_band.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
            ax_band.set_ylabel("umol m-2 s-1")
            ax_band.set_title("Flux ± random uncertainty (daily means)", fontsize=9)
            ax_band.legend(frameon=False, fontsize=8)

            cum = randunc.randunc_results_cumulatives
            gf = cum[randunc.fluxgapfilledcol]
            ax_cum.fill_between(cum.index, cum["FLUX-UNC"].to_numpy(),
                                cum["FLUX+UNC"].to_numpy(), color=_C_UNC, alpha=0.25,
                                label="cumulative ± σ")
            ax_cum.plot(gf.index, gf.to_numpy(), color=_C_FLUX, lw=1.2,
                        label=f"Σ {randunc.fluxgapfilledcol}")
            ax_cum.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
            ax_cum.set_ylabel("cumulative")
            ax_cum.set_title("Cumulative uncertainty propagation", fontsize=9)
            ax_cum.legend(frameon=False, fontsize=8)

            # Measured flux (x) vs random uncertainty (y): the heteroscedastic
            # flux-magnitude relationship (uncertainty rises with |flux|).
            # Restricted to method-1 records (a direct standard deviation of >= 6
            # similar measured fluxes). Methods 2-4 assign the *median* of other
            # records' uncertainties, which repeats across records and would show
            # as horizontal streaks here rather than the genuine relationship
            # (matches the library showplot_random_uncertainty).
            sc_df = pd.DataFrame({"x": flux, "y": unc})
            sc_df = sc_df.loc[res["WINDOW_N_VALS_METHOD1"] >= 6].dropna()
            ax_sc.scatter(sc_df["x"].to_numpy(), sc_df["y"].to_numpy(), s=5,
                          color=_C_UNC, alpha=0.3, edgecolors="none")
            ax_sc.axvline(0, color="#B0BEC5", lw=0.6, zorder=0)
            ax_sc.set_xlabel(f"{randunc.fluxcol} (measured)")
            ax_sc.set_ylabel("random uncertainty (±σ)")
            ax_sc.set_title("Uncertainty vs flux (method 1, direct SD)", fontsize=9)
        except Exception as err:  # plotting must never crash the tab
            ax_band.text(0.5, 0.5, f"Plot failed: {err}", ha="center", va="center",
                         transform=ax_band.transAxes, fontsize=8)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        self.featuresCreated.emit(self._result_df)
        self.status.setText(
            f"Added {', '.join(self._result_df.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
