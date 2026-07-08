"""
GUI.TABS.UNCERTAINTY_JOINTUNC: JOINT UNCERTAINTY (PAS20)
=======================================================

Combine the per-record **random measurement uncertainty** (e.g. ``{flux}_RANDUNC``
from the Random uncertainty tab) with the **scenario-ensemble uncertainty** of the
flux-partitioning/filtering percentile scenarios in quadrature — the faithful
ONEFlux ``compute_join`` (Pastorello et al. 2020), ``dv.flux.JointUncertaintyPAS20``:

    JOINTUNC = sqrt( RANDUNC^2 + ((scenario_upper - scenario_lower) / divisor)^2 )

For NEE the scenario ensemble is the USTAR-threshold percentile scenarios (16th/84th,
divisor 2); for the energy fluxes LE/H it is the energy-balance-correction
percentiles (25th/75th, IQR divisor 1.349). The tab collects the random-uncertainty
column, the lower/upper scenario fluxes and the gap-filled flux, picks the divisor
via a percentile-convention selector, previews the flux ± joint band, the random-vs-
scenario decomposition and the cumulative bounds, and emits the ``{base}_JOINTUNC``
column.

Layout mirrors the Random uncertainty tab: a fixed-width **Settings** column beside
a method **hero** band over the preview. All computation is library work (strict
GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.flux import JointUncertaintyPAS20
from diive.flux.lowres.codegen import jointunc_to_code
from diive.flux.lowres.uncertainty import JOINT_DIVISOR_1SIGMA, JOINT_DIVISOR_IQR
from diive.gui import theme
from diive.variables import auto_pick_column
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.worker import WorkerRunner

_C_FLUX = "#455A64"      # blue-grey 700 — the flux line
_C_JOINT = "#5E35B1"     # deep-purple 600 — the joint uncertainty band
_C_RANDOM = "#E53935"    # red 600 — the random component
_C_SCEN = "#1E88E5"      # blue 600 — the scenario (USTAR) component
_C_MUTED = "#6B7780"

#: Percentile-convention presets -> (label, divisor, lower-needle, upper-needle).
_DIVISOR_PRESETS = [
    ("NEE — USTAR scenarios (16th / 84th ÷ 2)", JOINT_DIVISOR_1SIGMA, "_16", "_84"),
    ("Energy flux LE/H (25th / 75th ÷ IQR 1.349)", JOINT_DIVISOR_IQR, "_25", "_75"),
]

#: Input specs: combo id, form label, auto-pick needle(s), prefer/avoid, tooltip.
_INPUTS = [
    {"key": "randunc", "label": "Random uncertainty", "needle": "RANDUNC",
     "tip": "Per-record random uncertainty column (e.g. NEE_CUT_REF_RANDUNC), "
            "produced by the Random uncertainty (PAS20) tab."},
    {"key": "lower", "label": "Scenario lower", "needle": "_16",
     "tip": "Lower-percentile scenario flux: 16th for NEE (e.g. NEE_CUT_16), "
            "25th for the LE/H energy-balance correction."},
    {"key": "upper", "label": "Scenario upper", "needle": "_84",
     "tip": "Upper-percentile scenario flux: 84th for NEE (e.g. NEE_CUT_84), "
            "75th for the LE/H energy-balance correction."},
    {"key": "flux_f", "label": "Flux (gap-filled)", "needle": "NEE",
     "prefer": "REF", "avoid": "ORIG",
     "tip": "Gap-filled flux — the central line the joint-uncertainty band "
            "brackets and the basis of the cumulative propagation."},
]


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)


class JointUncertaintyTab(DiiveTab):
    """Joint uncertainty (PAS20): random uncertainty + scenario percentile spread."""

    title = "Joint uncertainty (PAS20)"
    intro = ("Combine the random measurement uncertainty with the USTAR-filtering "
             "(or energy-balance) scenario spread in quadrature — the ONEFlux "
             "joint uncertainty (Pastorello et al. 2020). Emits a {base}_JOINTUNC "
             "column.")
    method_chip_label = "JOINT UNCERTAINTY"
    method_chip_bg = "#EDE7F6"
    method_chip_fg = "#4527A0"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._result_df: pd.DataFrame | None = None
        self._combos: dict[str, QComboBox] = {}
        self._avail: dict[str, QLabel] = {}

        self._sig = _Signals()
        self.featuresCreated = self._sig.features_created
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this joint-uncertainty estimate.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.addWidget(self._build_settings())
        layout.addWidget(self._build_right(), stretch=1)
        outer.addWidget(body, stretch=1)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(330)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        outer.addWidget(list_header("Settings", "set inputs & run"))
        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(intro)

        # Percentile convention -> divisor + scenario auto-pick needles.
        conv_box = QGroupBox("Scenario percentiles")
        cv = QVBoxLayout(conv_box)
        self.divisor_combo = QComboBox()
        for label, _div, _lo, _hi in _DIVISOR_PRESETS:
            self.divisor_combo.addItem(label)
        self.divisor_combo.setToolTip(
            "Which percentile pair the scenario fluxes represent. The range is "
            "converted to a 1-sigma-equivalent: 16th/84th bracket ±1σ (÷2, NEE); "
            "25th/75th are the interquartile range (÷1.349, energy fluxes).")
        self.divisor_combo.currentIndexChanged.connect(self._on_convention_changed)
        cv.addWidget(self.divisor_combo)
        outer.addWidget(conv_box)

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

        note = QLabel(
            "JOINTUNC = √(RANDUNC² + ((upper − lower) / divisor)²) per half-hour, "
            "the faithful ONEFlux compute_join. Run the Random uncertainty (PAS20) "
            "tab first to produce the RANDUNC column.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED}; font-style: italic;")
        outer.addWidget(note)

        self.run_btn = QPushButton("Run joint uncertainty")
        self.run_btn.setToolTip("Combine random + scenario uncertainty in quadrature.")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        self.status = QLabel("Pick the input columns, then run.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.add_btn = QPushButton("Add result to dataset")
        self.add_btn.setToolTip("Append the {base}_JOINTUNC column to the variable list.")
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
        self._hero = HeroBand(self.method_chip_label, self.method_chip_bg,
                              self.method_chip_fg)
        return self._hero

    def _clear_hero(self) -> None:
        self._hero.clear()

    def _update_hero(self, metrics: list) -> None:
        self._hero.set_metrics(metrics)

    def _hero_metrics(self, jointunc) -> list:
        res = jointunc.jointunc_results
        joint = jointunc.jointunc_series
        rand = res[jointunc.randunccol]
        scen = res[jointunc.scenarionunccol]
        cum = jointunc.jointunc_results_cumulatives["UNC_CUMULATIVE"]
        cum_last = float(cum.iloc[-1]) if len(cum) and pd.notna(cum.iloc[-1]) else float("nan")
        return [
            ("MEAN JOINT", f"{joint.mean():.3f}", "Mean per-record joint uncertainty"),
            ("RANDOM TERM", f"{rand.mean():.3f}", "Mean random-measurement component"),
            ("SCENARIO TERM", f"{scen.mean():.3f}",
             "Mean scenario (USTAR/EBC) component, as a 1σ-equivalent"),
            ("CUMULATIVE ±σ", f"{cum_last:.1f}",
             "Final cumulative joint uncertainty of the summed flux"),
        ]

    # --- data / inputs -------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._all_cols = [str(c) for c in df.columns]
        self._result_df = None
        self.add_btn.setEnabled(False)
        self._clear_hero()
        self._reseed_combos()

    def _scenario_prefer(self) -> str | None:
        """Bias scenario auto-picks toward the random-uncertainty column's flux
        (e.g. a NEE_CUT_REF_RANDUNC pick prefers NEE_CUT_16 over GPP_CUT_16)."""
        randunc = self._combos["randunc"].currentText()
        token = randunc.split("_", 1)[0].upper()
        return token or None

    def _reseed_combos(self) -> None:
        """(Re)populate every combo, auto-picking by the current convention's needles."""
        if self._df is None:
            return
        _label, _div, lo_needle, hi_needle = _DIVISOR_PRESETS[self.divisor_combo.currentIndex()]
        for spec in _INPUTS:
            combo = self._combos[spec["key"]]
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self._all_cols)
            if cur in self._all_cols:
                combo.setCurrentText(cur)
            else:
                # The scenario needles follow the chosen percentile convention and
                # prefer the random-uncertainty column's flux base.
                if spec["key"] in ("lower", "upper"):
                    needles = [lo_needle if spec["key"] == "lower" else hi_needle]
                    prefer = self._scenario_prefer()
                else:
                    needles = spec["needle"]
                    prefer = spec.get("prefer")
                if isinstance(needles, str):
                    needles = [needles]
                for needle in needles:
                    guess = auto_pick_column(self._all_cols, needle,
                                       prefer=prefer, avoid=spec.get("avoid"))
                    if guess:
                        combo.setCurrentText(guess)
                        break
            combo.blockSignals(False)
        self._refresh_availability()

    def _on_convention_changed(self, *_) -> None:
        # Re-pick only the scenario columns for the new percentile convention,
        # leaving any user-chosen randunc/flux picks intact.
        if self._df is None:
            return
        _label, _div, lo_needle, hi_needle = _DIVISOR_PRESETS[self.divisor_combo.currentIndex()]
        prefer = self._scenario_prefer()
        for key, needle in (("lower", lo_needle), ("upper", hi_needle)):
            guess = auto_pick_column(self._all_cols, needle, prefer=prefer)
            if guess:
                self._combos[key].setCurrentText(guess)
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

    def _divisor(self) -> float:
        return _DIVISOR_PRESETS[self.divisor_combo.currentIndex()][1]

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        return {**self._combos, "divisor_combo": self.divisor_combo}

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
        return jointunc_to_code(
            randunccol=picks["randunc"], scenario_lower_col=picks["lower"],
            scenario_upper_col=picks["upper"], fluxgapfilledcol=picks["flux_f"],
            divisor=self._divisor())

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
            self.status.setText("The four input columns must be distinct.")
            return

        work = self._df[[picks["randunc"], picks["lower"], picks["upper"],
                         picks["flux_f"]]].copy()
        self._set_running(True)
        self._clear_hero()
        self.status.setText("Computing joint uncertainty…")
        self._runner.run(self._compute_payload, work, picks, self._divisor())

    def _compute_payload(self, work, picks, divisor):
        """Run the library class off-thread; return the tuple for :meth:`_on_done`."""
        jointunc = JointUncertaintyPAS20(
            df=work, randunccol=picks["randunc"],
            scenario_lower_col=picks["lower"], scenario_upper_col=picks["upper"],
            fluxgapfilledcol=picks["flux_f"], divisor=divisor)
        jointunc.run()
        joint_col = jointunc.jointunccol
        out = pd.DataFrame({joint_col: jointunc.jointunc_series})
        out.attrs[ATTRS_KEY] = {
            joint_col: provenance_attr(
                origin=DERIVED, parent=str(picks["randunc"]), operation=self.title,
                tags=["uncertainty", "jointunc"]),
        }
        return out, jointunc

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        out, jointunc = payload
        self._set_running(False)
        self._result_df = out
        joint_col = out.columns[0]
        n = int(jointunc.jointunc_series.notna().sum())
        self._update_hero(self._hero_metrics(jointunc))
        self.status.setText(f"Done. {n:,} records. 'Add' appends {joint_col}.")
        self.add_btn.setEnabled(True)
        self._plot(jointunc)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self._clear_hero()
        self.status.setText(f"Failed: {msg}")
        self.canvas.show_message("Joint-uncertainty estimation failed")

    def _plot(self, jointunc) -> None:
        """Three panels: the flux ± joint-uncertainty band (daily means) spanning
        the top row, the random-vs-scenario component decomposition (bottom left)
        and the cumulative joint bounds (bottom right)."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        gs = fig.add_gridspec(2, 2)
        ax_band = fig.add_subplot(gs[0, :])
        ax_decomp = fig.add_subplot(gs[1, 0], sharex=ax_band)
        ax_cum = fig.add_subplot(gs[1, 1], sharex=ax_band)
        try:
            res = jointunc.jointunc_results
            joint = jointunc.jointunc_series
            flux = jointunc.df[jointunc.fluxgapfilledcol]
            d_flux = flux.resample("1D").mean()
            d_joint = joint.resample("1D").mean()
            ax_band.fill_between(d_flux.index, (d_flux - d_joint).to_numpy(),
                                 (d_flux + d_joint).to_numpy(), color=_C_JOINT, alpha=0.25,
                                 label="± joint uncertainty")
            ax_band.plot(d_flux.index, d_flux.to_numpy(), color=_C_FLUX, lw=1.0,
                         label=f"{jointunc.fluxgapfilledcol} (daily mean)")
            ax_band.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
            ax_band.set_ylabel("umol m-2 s-1")
            ax_band.set_title("Flux ± joint uncertainty (daily means)", fontsize=9)
            ax_band.legend(frameon=False, fontsize=8)

            # Decomposition: random vs scenario component (daily means). Shows how
            # much each error source contributes to the joint total.
            d_rand = res[jointunc.randunccol].resample("1D").mean()
            d_scen = res[jointunc.scenarionunccol].resample("1D").mean()
            ax_decomp.plot(d_rand.index, d_rand.to_numpy(), color=_C_RANDOM, lw=1.0,
                           label="random")
            ax_decomp.plot(d_scen.index, d_scen.to_numpy(), color=_C_SCEN, lw=1.0,
                           label="scenario (USTAR/EBC)")
            ax_decomp.plot(d_joint.index, d_joint.to_numpy(), color=_C_JOINT, lw=1.2,
                           label="joint")
            ax_decomp.set_ylabel("± uncertainty")
            ax_decomp.set_title("Component decomposition (daily means)", fontsize=9)
            ax_decomp.legend(frameon=False, fontsize=8)

            cum = jointunc.jointunc_results_cumulatives
            gf = cum[jointunc.fluxgapfilledcol]
            ax_cum.fill_between(cum.index, cum["FLUX-UNC"].to_numpy(),
                                cum["FLUX+UNC"].to_numpy(), color=_C_JOINT, alpha=0.25,
                                label="cumulative ± σ")
            ax_cum.plot(gf.index, gf.to_numpy(), color=_C_FLUX, lw=1.2,
                        label=f"Σ {jointunc.fluxgapfilledcol}")
            ax_cum.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
            ax_cum.set_ylabel("cumulative")
            ax_cum.set_title("Cumulative joint propagation", fontsize=9)
            ax_cum.legend(frameon=False, fontsize=8)
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
