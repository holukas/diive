"""
GUI.TABS.USTAR_DETECTION: USTAR THRESHOLD DETECTION
===================================================

Standalone friction-velocity (u*) threshold detection using the ONEFlux moving
point method (Papale et al. 2006). Pick the NEE / TA / USTAR / SW_IN columns and
run either a single seasonal detection (per-season + annual threshold) or a
multi-year bootstrap giving VUT (variable, per-year) and CUT (constant, pooled
across years) thresholds.

All detection is the library's `UstarMovingPointDetection` /
`UstarBootstrapThresholds`; this tab only collects columns + parameters, runs
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

from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner
from diive.variables import auto_pick_column


class UstarDetectionTab(DiiveTab):
    """Detect the USTAR threshold (moving point, Papale 2006)."""

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

        par_box = QGroupBox("Stratification")
        pf = QFormLayout(par_box)
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
        v.addWidget(par_box)

        self.boot_chk = QCheckBox("Multi-year bootstrap (VUT + CUT)")
        self.boot_chk.setToolTip(
            "Run a 3-year sliding-window bootstrap:\n"
            "• VUT (variable) — per-year p16/p50/p84 thresholds (smoothed over a "
            "3-year window for stability).\n"
            "• CUT (constant) — one threshold pooled across all years.")
        self.boot_chk.toggled.connect(self._sync_boot)
        v.addWidget(self.boot_chk)

        self.boot_box = QGroupBox("Bootstrap settings")
        bf = QFormLayout(self.boot_box)
        self.n_iter = QSpinBox()
        self.n_iter.setRange(1, 10000)
        self.n_iter.setValue(100)
        bf.addRow("Iterations / window", self.n_iter)
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(-1, 256)
        self.n_jobs.setValue(1)
        self.n_jobs.setToolTip("Parallel workers (1 = sequential, -1 = all CPUs).")
        bf.addRow("Parallel workers", self.n_jobs)
        self.percentiles = QLineEdit("16, 50, 84")
        bf.addRow("Percentiles", self.percentiles)
        v.addWidget(self.boot_box)

        v.addStretch(1)
        self._sync_boot()
        return host

    def _sync_boot(self, *_) -> None:
        self.boot_box.setVisible(self.boot_chk.isChecked())
        self.run_btn.setText("Run bootstrap" if self.boot_chk.isChecked()
                             else "Run detection")

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
        return {"controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        self._sync_boot()

    def _controls(self) -> dict:
        return {"nee_col": self.nee_col, "ta_col": self.ta_col,
                "ustar_col": self.ustar_col, "swin_col": self.swin_col,
                "ta_classes": self.ta_classes, "ustar_classes": self.ustar_classes,
                "fw": self.fw, "boot_chk": self.boot_chk, "n_iter": self.n_iter,
                "n_jobs": self.n_jobs, "percentiles": self.percentiles}

    # --- run -----------------------------------------------------------
    def _percentiles(self) -> tuple[int, ...]:
        out = []
        for tok in self.percentiles.text().replace(";", ",").split(","):
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
        kwargs = dict(
            picks,
            ta_classes_count=self.ta_classes.value(),
            ustar_classes_count=self.ustar_classes.value(),
            forward_mode_n=self.fw.value(),
        )
        bootstrap = self.boot_chk.isChecked()
        if bootstrap:
            pcts = self._percentiles()
            if not pcts:
                self.status.setText("Enter at least one percentile (1–99).")
                return
            boot_cfg = dict(n_iter=self.n_iter.value(), n_jobs=self.n_jobs.value(),
                            percentiles=pcts)
        else:
            boot_cfg = None

        self._set_running(True)
        self.progress.start_busy("Running bootstrap…" if bootstrap else "Detecting…")
        self.status.setText("Running bootstrap…" if bootstrap else "Detecting…")
        self._runner.run(self._compute_payload, self._df, kwargs, boot_cfg)

    def _compute_payload(self, df, kwargs, boot_cfg):
        """Run the library detector(s) off-thread; return ``(mode, data)``."""
        if boot_cfg is None:
            det = UstarMovingPointDetection(df, verbose=0, **kwargs)
            res = det.detect()
            annual = det.get_annual_thresholds().get("threshold")
            return "single", (res.copy(), annual)
        boot = UstarBootstrapThresholds(
            df, detector_class=UstarMovingPointDetection,
            detector_kwargs=kwargs, verbose=0, **boot_cfg)
        annual = boot.run()
        cut = boot.get_cut_threshold()
        return "bootstrap", (annual.copy(), cut)

    def _worker(self, df, kwargs, boot_cfg) -> None:
        """Synchronous compute + dispatch — tests call this directly; the GUI runs
        :meth:`_compute_payload` on the worker thread instead."""
        try:
            self._on_done(self._compute_payload(df, kwargs, boot_cfg))
        except Exception as err:
            self._on_failed(str(err))

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        self._set_running(False)
        self.progress.finish()
        mode, data = payload
        if mode == "single":
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

    def _fill_table(self, headers: list[str], rows: list[tuple[str, str]]) -> None:
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                if c > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
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
            f"Seasonal detection complete. Annual threshold (max across seasons): "
            f"{ann_txt} m s⁻¹.")
        self.hero.set_metrics([
            ("ANNUAL u*", f"{ann_txt} m/s", "Annual threshold (max across seasons)"),
            ("SEASONS", f"{len(res.index)}", "Number of seasons detected"),
        ])
        self._plot_single(res, annual)

    def _show_bootstrap(self, annual: pd.DataFrame, cut: dict) -> None:
        pcols = list(annual.columns)
        headers = ["Year"] + pcols
        rows = []
        for year in annual.index:
            vals = [f"{annual.loc[year, c]:.4f}"
                    if np.isfinite(annual.loc[year, c]) else "—" for c in pcols]
            rows.append((str(year), *vals))
        cut_vals = [f"{cut[c]:.4f}" if np.isfinite(cut[c]) else "—" for c in pcols]
        rows.append(("CUT (constant)", *cut_vals))
        self._fill_table(headers, rows)
        cut_str = "  ".join(f"{k}={v}" for k, v in zip(pcols, cut_vals))
        self.status.setText(
            f"Bootstrap complete. Rows per year = VUT (variable, per-year, smoothed "
            f"over a 3-year window); last row = CUT (constant, pooled across years): "
            f"{cut_str}.")
        mid = "p50" if "p50" in pcols else pcols[len(pcols) // 2]
        cut_mid = f"{cut[mid]:.4f}" if np.isfinite(cut[mid]) else "—"
        self.hero.set_metrics([
            (f"CUT {mid}", f"{cut_mid} m/s", "Constant threshold pooled across years"),
            ("YEARS", f"{len(annual.index)}", "Years with a per-year (VUT) threshold"),
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
