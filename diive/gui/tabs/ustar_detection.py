"""
GUI.TABS.USTAR_DETECTION: USTAR THRESHOLD DETECTION
===================================================

Standalone friction-velocity (u*) threshold detection. Pick the NEE / TA /
USTAR / SW_IN columns and a method (ONEFlux moving point + change point, one of
the two, or REddyProc), then run a bootstrap over the years giving VUT
(variable, per-year) and CUT (constant, from all years) thresholds. Without the
bootstrap, the tab runs one seasonal detection with the ONEFlux moving point
method (per-season + annual threshold).

All detection is the library's `UstarThresholdDetection` /
`UstarMovingPointDetection`. This tab only collects columns + parameters, runs
them on a worker thread, and lays the numeric results into a table and a small
diagnostic plot (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diive.flux.lowres.ustar_detection import ONEFLUX_CUT_MIN_YEARS, UstarThresholdDetection
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner
from diive.variables import auto_pick_column

#: (label, ``UstarThresholdDetection`` mode), default first. Shared with the
#: flux chain's Level 3.3, so both store the mode value, not the label.
USTAR_METHOD_ITEMS = (("ONEFlux: moving point + change point", "oneflux"),
                      ("ONEFlux: moving point only", "oneflux_mp"),
                      ("ONEFlux: change point only", "oneflux_cp"),
                      ("REddyProc", "reddyproc"))

#: Modes that run the moving point method (and use its stratification).
USTAR_MP_MODES = ("oneflux", "oneflux_mp")

#: Short method names for the hero chip and status lines.
USTAR_METHOD_SHORT = {"oneflux": "ONEFlux MP + CP", "oneflux_mp": "ONEFlux MP",
                 "oneflux_cp": "ONEFlux CP", "reddyproc": "REddyProc"}

#: Default percentiles per mode (as the library uses them).
_DEFAULT_PERCENTILES = {"oneflux": "16, 50, 84", "oneflux_mp": "16, 50, 84",
                        "oneflux_cp": "16, 50, 84", "reddyproc": "5, 50, 95"}

USTAR_METHOD_TIP = (
    "How the u* threshold is detected.\n\n"
    "ONEFlux: moving point + change point (default): the method of FLUXNET2015 "
    "and ICOS. For each year, the moving point method (Papale et al. 2006) and "
    "the change point method (Barr et al. 2013) each give 100 bootstrap "
    "thresholds. Both sets are pooled.\n\n"
    "ONEFlux: moving point only / change point only: the same with one of the "
    "two methods.\n\n"
    "REddyProc: the moving point variant of REddyProc 1.3.4 (Wutzler et al. "
    "2018). Its seasons can cross the year boundary. 200 bootstrap thresholds "
    "per year.")

#: What VUT and CUT mean, shared by both tabs.
VUT_CUT_TIP = (
    "VUT (variable): one threshold per year. The ONEFlux methods pool the "
    "thresholds of the year and of the years before and after it. REddyProc uses "
    "the thresholds of the year only.\n"
    "CUT (constant): one threshold from the thresholds of all years.")


def make_ustar_method_combo() -> QComboBox:
    """A combo with the u* detection methods, ONEFlux (MP + CP) selected."""
    combo = QComboBox()
    for label, mode in USTAR_METHOD_ITEMS:
        combo.addItem(label, mode)
    combo.setToolTip(USTAR_METHOD_TIP)
    return combo


def set_ustar_method(combo: QComboBox, mode: str | None) -> None:
    """Select ``mode``; an unknown value or None leaves the combo unchanged."""
    i = combo.findData(mode) if mode else -1
    if i >= 0:
        combo.setCurrentIndex(i)


class UstarDetectionTab(DiiveTab):
    """Detect the USTAR threshold (ONEFlux or REddyProc)."""

    title = "USTAR detection"

    def build(self) -> QWidget:
        self._df = None
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.run_btn = QPushButton("Run detection")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        outer.addLayout(build_titlebar("USTAR threshold detection", self.run_btn))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.status = QLabel("Pick columns and run a detection.")
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        # Headline threshold(s), filled after a run — the one load-bearing number.
        self.hero = HeroBand("USTAR", "#E3F2FD", "#1565C0")
        rl.addWidget(self.hero)
        # Busy bar while detection / the multi-year bootstrap runs.
        self.progress = ProgressBar()
        rl.addWidget(self.progress)
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.addWidget(self._build_table())
        self.canvas = MplCanvas()
        hsplit.addWidget(self.canvas)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([360, 560])
        rl.addWidget(hsplit, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)
        return root

    # --- controls ------------------------------------------------------
    def _build_controls(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(300)
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        cols_box = QGroupBox("Columns")
        cf = QFormLayout(cols_box)
        self.nee_col = QComboBox()
        self.ta_col = QComboBox()
        self.ustar_col = QComboBox()
        self.swin_col = QComboBox()
        cf.addRow("NEE / flux", self.nee_col)
        cf.addRow("Air temperature", self.ta_col)
        cf.addRow("USTAR", self.ustar_col)
        cf.addRow("Shortwave in", self.swin_col)
        v.addWidget(cols_box)

        meth_box = QGroupBox("Method")
        mf = QVBoxLayout(meth_box)
        self.method = make_ustar_method_combo()
        self.method.currentIndexChanged.connect(self._sync_method)
        mf.addWidget(self.method)
        v.addWidget(meth_box)

        self.par_box = QGroupBox("Stratification (moving point)")
        self.par_box.setToolTip("Used by the moving point method only.")
        pf = QFormLayout(self.par_box)
        self.ta_classes = QSpinBox()
        self.ta_classes.setRange(2, 30)
        self.ta_classes.setValue(7)
        pf.addRow("TA classes", self.ta_classes)
        self.ustar_classes = QSpinBox()
        self.ustar_classes.setRange(4, 60)
        self.ustar_classes.setValue(20)
        pf.addRow("USTAR classes", self.ustar_classes)
        self.fw = QSpinBox()
        self.fw.setRange(1, 5)
        self.fw.setValue(2)
        self.fw.setToolTip("Forward-mode order (2 = Fw2, the ONEFlux/REddyProc default).")
        pf.addRow("Forward-mode n", self.fw)
        note = QLabel("Seasons: calendar quarters (ONEFlux default).")
        note.setStyleSheet("color: #6B7780; font-size: 11px;")
        note.setWordWrap(True)
        pf.addRow(note)
        v.addWidget(self.par_box)

        self.boot_chk = QCheckBox("Bootstrap over years (VUT + CUT)")
        self.boot_chk.setChecked(True)
        self.boot_chk.setToolTip(
            "Bootstrap the detection for each year, then report:\n" + VUT_CUT_TIP
            + "\n\nUnticked: one seasonal detection with the ONEFlux moving point "
              "method (a threshold per season and the annual value).")
        self.boot_chk.toggled.connect(self._sync_boot)
        v.addWidget(self.boot_chk)
        # The single detection exists for the moving point method only.
        self.single_note = QLabel("Without bootstrap: one seasonal detection with "
                                  "the ONEFlux moving point method.")
        self.single_note.setStyleSheet("color: #6B7780; font-size: 11px;")
        self.single_note.setWordWrap(True)
        v.addWidget(self.single_note)

        self.boot_box = QGroupBox("Bootstrap settings")
        bf = QFormLayout(self.boot_box)
        self.n_iter = QSpinBox()
        self.n_iter.setRange(0, 10000)
        self.n_iter.setValue(0)
        self.n_iter.setSpecialValueText("Default")
        self.n_iter.setToolTip("Bootstrap draws per year and method. Default: 100 "
                               "for ONEFlux, 200 for REddyProc.")
        bf.addRow("Iterations", self.n_iter)
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 256)
        self.n_jobs.setValue(1)
        self.n_jobs.setToolTip("Parallel workers (1 = sequential, -1 = all CPUs).")
        bf.addRow("Parallel workers", self.n_jobs)
        self.percentiles = QLineEdit("")
        self.percentiles.setToolTip("Percentiles of the bootstrap thresholds. Empty: "
                                    "16, 50, 84 for ONEFlux, 5, 50, 95 for REddyProc.")
        bf.addRow("Percentiles", self.percentiles)
        v.addWidget(self.boot_box)

        v.addStretch(1)
        # Bootstrap choice of the user, kept while a method without the single
        # detection forces the bootstrap on.
        self._boot_pref = True
        self._sync_method()
        return host

    def _method(self) -> str:
        return self.method.currentData()

    def _sync_method(self, *_) -> None:
        """Enable only the controls the selected method uses."""
        mode = self._method()
        has_mp = mode in USTAR_MP_MODES
        self.par_box.setEnabled(has_mp)
        if has_mp:
            if not self.boot_chk.isEnabled():
                self.boot_chk.setEnabled(True)
                self.boot_chk.setChecked(self._boot_pref)
        elif self.boot_chk.isEnabled():
            self._boot_pref = self.boot_chk.isChecked()
            self.boot_chk.setChecked(True)
            self.boot_chk.setEnabled(False)
        self.percentiles.setPlaceholderText(f"default: {_DEFAULT_PERCENTILES[mode]}")
        self._sync_boot()

    def _sync_boot(self, *_) -> None:
        boot = self.boot_chk.isChecked()
        self.boot_box.setVisible(boot)
        self.single_note.setVisible(not boot)
        self.run_btn.setText("Run bootstrap" if boot else "Run detection")

    def _build_table(self) -> QWidget:
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        return self.table

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        cols = [str(c) for c in df.columns]
        for combo in (self.nee_col, self.ta_col, self.ustar_col, self.swin_col):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(cols)
            if cur in cols:
                combo.setCurrentText(cur)
        # Auto-pick sensible defaults (mirrors the detector's own auto-detect).
        nee = auto_pick_column(cols, "NEE", prefer="QCF") or auto_pick_column(cols, "NEE")
        if nee:
            self.nee_col.setCurrentText(nee)
        ta = auto_pick_column(cols, "TA_") or auto_pick_column(cols, "TA")
        if ta:
            self.ta_col.setCurrentText(ta)
        us = auto_pick_column(cols, "USTAR")
        if us:
            self.ustar_col.setCurrentText(us)
        sw = auto_pick_column(cols, "SW_IN", avoid="POT")
        if sw:
            self.swin_col.setCurrentText(sw)

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        controls = save_controls(self._controls())
        controls["method"] = self._method()  # the value, not the label
        controls["boot_pref"] = self._boot_pref
        return {"controls": controls}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        saved = state.get("controls") or {}
        # Projects saved before the method choice ran the moving point method.
        set_ustar_method(self.method, saved.get("method", "oneflux_mp" if saved else None))
        restore_controls(self._controls(), saved)
        self._boot_pref = bool(saved.get("boot_pref", self.boot_chk.isChecked()))
        self._sync_method()

    def _controls(self) -> dict:
        return {"nee_col": self.nee_col, "ta_col": self.ta_col,
                "ustar_col": self.ustar_col, "swin_col": self.swin_col,
                "ta_classes": self.ta_classes, "ustar_classes": self.ustar_classes,
                "fw": self.fw, "boot_chk": self.boot_chk, "n_iter": self.n_iter,
                "n_jobs": self.n_jobs, "percentiles": self.percentiles}

    # --- run -----------------------------------------------------------
    def _percentiles(self) -> tuple[int, ...] | None:
        """Parsed percentiles, None for the method default (empty field), or
        an empty tuple on bad input."""
        text = self.percentiles.text().strip()
        if not text:
            return None
        out = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                p = int(round(float(tok)))
            except ValueError:
                return ()
            if 1 <= p <= 99:
                out.append(p)
        return tuple(out)

    def _mp_kwargs(self) -> dict:
        """Moving point stratification, as ``UstarMovingPointDetection`` takes it."""
        return dict(ta_classes_count=self.ta_classes.value(),
                    ustar_classes_count=self.ustar_classes.value(),
                    forward_mode_n=self.fw.value())

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        cols = {str(c) for c in self._df.columns}
        picks = {"nee_col": self.nee_col.currentText(),
                 "ta_col": self.ta_col.currentText(),
                 "ustar_col": self.ustar_col.currentText(),
                 "swin_col": self.swin_col.currentText()}
        missing = [k for k, v in picks.items() if not v or v not in cols]
        if missing:
            self.status.setText("Pick valid columns for: "
                                + ", ".join(m.replace("_col", "") for m in missing))
            return
        mode = self._method()
        bootstrap = self.boot_chk.isChecked() or mode not in USTAR_MP_MODES
        if bootstrap:
            pcts = self._percentiles()
            if pcts == ():
                self.status.setText("Enter at least one percentile (1-99), or "
                                    "leave the field empty for the default.")
                return
            boot_cfg = dict(mode=mode, n_iter=self.n_iter.value() or None,
                            n_jobs=self.n_jobs.value(), percentiles=pcts)
        else:
            boot_cfg = None

        self._set_running(True)
        self.progress.start_busy("Running bootstrap…" if bootstrap else "Detecting…")
        self.status.setText("Running bootstrap…" if bootstrap else "Detecting…")
        self._runner.run(self._compute_payload, self._df, picks, self._mp_kwargs(),
                         boot_cfg)

    @staticmethod
    def _compute_payload(df, picks, mp_kwargs, boot_cfg):
        """Run the library detector off-thread; return ``(kind, data)``.

        ``boot_cfg`` None runs one seasonal moving point detection. Otherwise it
        holds ``mode``, ``n_iter``, ``n_jobs`` and ``percentiles`` for
        ``UstarThresholdDetection``.
        """
        if boot_cfg is None:
            det = UstarMovingPointDetection(df, verbose=0, **picks, **mp_kwargs)
            res = det.detect()
            annual = det.get_annual_thresholds().get("threshold")
            return "single", (res.copy(), annual)
        mode = boot_cfg["mode"]
        det = UstarThresholdDetection(
            df, **picks, mode=mode, n_iter=boot_cfg.get("n_iter"),
            n_jobs=boot_cfg.get("n_jobs", 1), percentiles=boot_cfg.get("percentiles"),
            mp_kwargs=mp_kwargs if mode in USTAR_MP_MODES else None, verbose=0)
        annual = det.run()
        cut = det.get_cut_threshold()
        # Per year: how many bootstrap thresholds each method gave, and why a
        # method gave none (library failures_).
        counts = {y: {m: (int(np.isfinite(s).sum()), len(s)) for m, s in d.items()}
                  for y, d in det.samples_.items()}
        failures = {y: dict(d) for y, d in det.failures_.items() if d}
        info = dict(mode=mode, n_iter=det.n_iter, counts=counts, failures=failures)
        return "bootstrap", (annual.copy(), cut, info)

    def _worker(self, df, picks, mp_kwargs, boot_cfg) -> None:
        """Synchronous compute + dispatch — tests call this directly; the GUI runs
        :meth:`_compute_payload` on the worker thread instead."""
        try:
            self._on_done(self._compute_payload(df, picks, mp_kwargs, boot_cfg))
        except Exception as err:
            self._on_failed(str(err))

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        self._set_running(False)
        self.progress.finish()
        kind, data = payload
        if kind == "single":
            self._show_single(*data)
        else:
            self._show_bootstrap(*data)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.progress.finish()
        self.hero.clear()
        self.status.setText(f"Failed: {msg}")
        self.canvas.show_message("Detection failed")

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)

    def _fill_table(self, headers: list[str], rows: list[tuple[str, ...]],
                    tips: list[str] | None = None) -> None:
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if tips and tips[r]:
                    item.setToolTip(tips[r])
                self.table.setItem(r, c, item)

    def _show_single(self, res: pd.DataFrame, annual) -> None:
        rows = []
        for idx in res.index:
            val = res.loc[idx, "threshold"]
            txt = f"{val:.4f}" if (val is not None and np.isfinite(val)
                                   and val != 10.0) else "—"
            rows.append((str(idx), txt))
        ann_txt = (f"{annual:.4f}" if annual is not None and np.isfinite(annual)
                   else "—")
        rows.append(("Annual (max)", ann_txt))
        self._fill_table(["Season", "u* (m s⁻¹)"], rows)
        self.status.setText(
            f"Seasonal detection (ONEFlux moving point) complete. Annual threshold "
            f"(max across seasons): {ann_txt} m s⁻¹.")
        self.hero.set_chip(USTAR_METHOD_SHORT["oneflux_mp"])
        self.hero.set_metrics([
            ("ANNUAL u*", f"{ann_txt} m/s", "Annual threshold (max across seasons)"),
            ("SEASONS", f"{len(res.index)}", "Number of seasons detected"),
        ])
        self._plot_single(res, annual)

    @staticmethod
    def _year_note(counts: dict, failures: dict) -> str:
        """'MP 100/100, CP none' for one year."""
        parts = [f"{m.upper() if m != 'reddyproc' else 'REddyProc'} {n}/{t}"
                 for m, (n, t) in counts.items()]
        parts += [f"{m.upper()} none" for m in failures]
        return ", ".join(parts)

    def _show_bootstrap(self, annual: pd.DataFrame, cut: dict, info: dict) -> None:
        mode = info["mode"]
        counts, failures = info["counts"], info["failures"]
        pcols = list(annual.columns)
        headers = ["Year"] + pcols + ["Thresholds"]
        rows, tips = [], []
        for year in annual.index:
            vals = [f"{annual.loc[year, c]:.4f}"
                    if np.isfinite(annual.loc[year, c]) else "—" for c in pcols]
            note = self._year_note(counts.get(year, {}), failures.get(year, {}))
            rows.append((str(year), *vals, note))
            tips.append("\n".join(f"{m.upper()}: {why}"
                                  for m, why in failures.get(year, {}).items()))
        cut_vals = [f"{cut[c]:.4f}" if np.isfinite(cut[c]) else "—" for c in pcols]
        rows.append(("CUT (constant)", *cut_vals, "all years"))
        tips.append("")
        self._fill_table(headers, rows, tips)
        cut_str = "  ".join(f"{k}={v}" for k, v in zip(pcols, cut_vals, strict=False))
        vut_txt = ("pools the thresholds of the year and its neighbours"
                   if mode != "reddyproc" else "uses the thresholds of the year")
        msg = (f"Bootstrap complete ({USTAR_METHOD_SHORT[mode]}, {info['n_iter']} "
               f"draws per year and method). Rows per year = VUT (each year {vut_txt}). "
               f"Last row = CUT (all years): {cut_str}.")
        fail_lines = [f"{y} {m.upper()}: {why}" for y, d in sorted(failures.items())
                      for m, why in d.items()]
        if fail_lines:
            msg += "\nNo thresholds for: " + ". ".join(fail_lines) + "."
        if mode != "reddyproc" and len(annual.index) < ONEFLUX_CUT_MIN_YEARS:
            msg += (f"\nONEFlux itself reports a CUT only for {ONEFLUX_CUT_MIN_YEARS} "
                    f"or more years.")
        self.status.setText(msg)
        mid = "p50" if "p50" in pcols else pcols[len(pcols) // 2]
        cut_mid = f"{cut[mid]:.4f}" if np.isfinite(cut[mid]) else "—"
        self.hero.set_chip(USTAR_METHOD_SHORT[mode])
        self.hero.set_metrics([
            (f"CUT {mid}", f"{cut_mid} m/s", "Constant threshold from all years"),
            ("YEARS", f"{len(annual.index)}", "Years with a per-year (VUT) row"),
        ])
        self._plot_bootstrap(annual, cut)

    # --- plots (presentation only; the numbers are the library's) ------
    def _plot_single(self, res: pd.DataFrame, annual) -> None:
        ax = self.canvas.new_axes(1)[0]
        vals = res["threshold"].to_numpy(dtype=float)
        labels = [str(i) for i in res.index]
        valid = np.isfinite(vals) & (vals != 10.0)
        ax.bar(np.arange(len(vals))[valid], vals[valid], color="#2196F3", width=0.6)
        if annual is not None and np.isfinite(annual):
            ax.axhline(annual, color="#F44336", ls="--",
                       label=f"annual (max) = {annual:.3f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("u* threshold (m/s)")
        ax.set_title("Seasonal u* thresholds (moving point)")
        self.canvas.draw()

    def _plot_bootstrap(self, annual: pd.DataFrame, cut: dict) -> None:
        ax = self.canvas.new_axes(1)[0]
        years = list(annual.index)
        x = np.arange(len(years))
        pcols = list(annual.columns)
        # Use p50 as the centre; the lowest/highest requested percentile as whiskers.
        mid = "p50" if "p50" in pcols else pcols[len(pcols) // 2]
        lo, hi = pcols[0], pcols[-1]
        center = annual[mid].to_numpy(dtype=float)
        low = annual[lo].to_numpy(dtype=float)
        high = annual[hi].to_numpy(dtype=float)
        yerr = np.vstack([np.clip(center - low, 0, None),
                          np.clip(high - center, 0, None)])
        ax.errorbar(x, center, yerr=yerr, fmt="o", color="#2196F3", capsize=4,
                    label=f"{mid} ({lo}–{hi})")
        if mid in cut and np.isfinite(cut[mid]):
            ax.axhline(cut[mid], color="#F44336", ls="--",
                       label=f"CUT {mid} = {cut[mid]:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=8)
        ax.set_ylabel("u* threshold (m/s)")
        ax.set_title("VUT (per-year) vs CUT (constant) u* thresholds")
        ax.legend(frameon=False, fontsize=8)
        self.canvas.draw()
