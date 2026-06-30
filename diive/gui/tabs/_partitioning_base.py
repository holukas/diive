"""
GUI.TABS._PARTITIONING_BASE: SHARED BASE FOR NEE-PARTITIONING TABS
=================================================================

Common machinery for the GUI's NEE -> GPP + RECO partitioning tabs (the four
faithful ports: nighttime ONEFlux / REddyProc, daytime ONEFlux / REddyProc):
a per-input column picker (one combo per required series, auto-seeded to the
standard name with an availability marker), the site coordinates (latitude /
longitude / UTC offset, seeded from Project settings), an optional VPD-unit
toggle, a worker thread that runs the library partitioner, a result plot
(measured NEE + partitioned GPP + RECO), and "Add results to dataset".

A concrete tab subclasses :class:`BasePartitioningTab` and supplies only the
method-specific bits: which input columns it needs, which coordinates, the
result column names to plot, and how to construct the library partitioner. All
partitioning is library work (``dv.flux.partition_nee_*``); this base only
collects the inputs, runs the class on a worker thread, previews the result,
and emits the new columns (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.flux.partitioning.codegen import partitioning_to_code
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand, _fmt
from diive.gui.widgets.column_picker import ColumnPicker
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner

_C_NEE = "#90A4AE"    # blue-grey 300 — measured NEE (recedes)
_C_RECO = "#E53935"   # red 600       — ecosystem respiration
_C_GPP = "#43A047"    # green 600     — gross primary production
_C_MUTED = "#6B7780"  # secondary/help text


class _Signals(QObject):
    """Qt signal (DiiveTab is a plain ABC, not a QObject). Run done/failed
    plumbing lives in :class:`WorkerRunner`."""
    features_created = Signal(object)


class BasePartitioningTab(DiiveTab):
    """Shared base for NEE-partitioning tabs; subclasses supply the method bits.

    Subclasses set the class attributes below and implement
    :meth:`_build_partitioner` (and optionally :meth:`_extra_kwargs`)."""

    #: Tab title (DiiveTab requirement) — set by the subclass.
    title = "NEE partitioning"
    #: One-line description shown above the controls.
    intro = "Partition measured NEE into GPP and RECO."
    #: Input columns the method needs. Each entry is a dict with keys:
    #: ``key`` (combo id), ``label`` (form label), ``needle`` (auto-pick
    #: substring) and optional ``prefer`` / ``avoid`` / ``tip``.
    inputs: list[dict] = []
    #: Which site coordinates the method requires.
    needs_lat = False
    needs_lon = False
    needs_utc = False
    #: Show the "VPD is in kPa" toggle (daytime methods take VPD).
    has_vpd_unit = False
    #: Result columns to plot as RECO / GPP, plus the suffix shown in the status.
    reco_col = "RECO"
    gpp_col = "GPP"
    #: Suffix that tags this method's output columns (e.g. "NT_OF"); used only
    #: for the status line.
    method_suffix = ""

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df = None
        self._results: pd.DataFrame | None = None
        self._nee_key = self.inputs[0]["key"] if self.inputs else "nee"
        self._sig = _Signals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.run_btn = QPushButton("Run partitioning")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        self.add_btn = QPushButton("Add results to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this partitioning.")
        # Only the export action sits in the header; Run/Add live below the
        # settings column (see _build_controls), matching the correction,
        # outlier and uncertainty tabs' settings->preview layout.
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

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
        # Headline carbon-flux totals, filled after a run (the green flux family).
        self.hero = HeroBand(self.method_suffix, "#E8F5E9", "#2E7D32")
        rl.addWidget(self.hero)
        # Busy bar while the (per-year, single opaque call) partitioning runs —
        # the honest indeterminate signal that the worker is alive.
        self.progress = ProgressBar()
        rl.addWidget(self.progress)
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

        site.manager.changed.connect(self._on_site_changed)
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

        self.picker = ColumnPicker(self.inputs, title="Input columns")
        v.addWidget(self.picker)

        if self.needs_lat or self.needs_lon or self.needs_utc:
            site_box = QGroupBox("Site coordinates")
            sf = QFormLayout(site_box)
            if self.needs_lat:
                self.lat = QDoubleSpinBox()
                self.lat.setRange(-90.0, 90.0)
                self.lat.setDecimals(4)
                self.lat.setToolTip("Site latitude in decimal degrees (north positive).")
                sf.addRow("Latitude", self.lat)
            else:
                self.lat = None
            if self.needs_lon:
                self.lon = QDoubleSpinBox()
                self.lon.setRange(-180.0, 180.0)
                self.lon.setDecimals(4)
                self.lon.setToolTip("Site longitude in decimal degrees (east positive).")
                sf.addRow("Longitude", self.lon)
            else:
                self.lon = None
            if self.needs_utc:
                self.utc = QSpinBox()
                self.utc.setRange(-12, 14)
                self.utc.setToolTip("UTC offset (hours) of the timestamps.")
                sf.addRow("UTC offset (h)", self.utc)
            else:
                self.utc = None
            note = QLabel("Defaults from Settings ▸ Project settings.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {_C_MUTED};")
            sf.addRow(note)
            v.addWidget(site_box)
            self._seed_site()
        else:
            self.lat = self.lon = self.utc = None

        if self.has_vpd_unit:
            opt_box = QGroupBox("Options")
            of = QVBoxLayout(opt_box)
            self.vpd_kpa_cb = QCheckBox("VPD is in kPa (diive convention)")
            self.vpd_kpa_cb.setChecked(True)
            self.vpd_kpa_cb.setToolTip(
                "Checked: the VPD column is in kPa and converted to hPa internally "
                "(the unit the Lasslop light-response curve expects). Uncheck if the "
                "column is already in hPa.")
            of.addWidget(self.vpd_kpa_cb)
            v.addWidget(opt_box)
        else:
            self.vpd_kpa_cb = None

        # Let the subclass add any extra widgets (rare).
        self._add_extra_controls(v)

        # Run/Add anchored below the settings (not in the header) — same as the
        # correction/outlier/uncertainty tabs.
        v.addWidget(self.run_btn)
        v.addWidget(self.add_btn)

        v.addStretch(1)
        return host

    def _add_extra_controls(self, layout: QVBoxLayout) -> None:
        """Add method-specific widgets to the control column (subclass hook)."""

    # --- site ----------------------------------------------------------
    def _seed_site(self) -> None:
        m = site.manager
        if not m.configured:
            return
        if self.lat is not None:
            self.lat.setValue(m.latitude)
        if self.lon is not None:
            self.lon.setValue(m.longitude)
        if self.utc is not None:
            self.utc.setValue(m.utc_offset)

    def _on_site_changed(self) -> None:
        self._seed_site()

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._results = None
        self.add_btn.setEnabled(False)
        self.picker.seed(df.columns)

    # --- state ---------------------------------------------------------
    def _state_controls(self) -> dict:
        controls = self.picker.combos()
        if self.lat is not None:
            controls["lat"] = self.lat
        if self.lon is not None:
            controls["lon"] = self.lon
        if self.utc is not None:
            controls["utc"] = self.utc
        if self.vpd_kpa_cb is not None:
            controls["vpd_kpa"] = self.vpd_kpa_cb
        return controls

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._state_controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._state_controls(), state.get("controls"))
        self.picker.refresh_availability()

    # --- run -----------------------------------------------------------
    def _picks(self) -> dict[str, str]:
        return self.picker.picks()

    def _python_code(self) -> str | None:
        """Runnable snippet reproducing this partitioning, or None when the inputs
        are incomplete (button no-op, mirroring :meth:`_run`'s validation)."""
        if self._df is None:
            self.status.setText("Load data first to copy its code.")
            return None
        cols = {str(c) for c in self._df.columns}
        picks = self._picks()
        # Drop optional inputs left as "(none)"; require every other pick to be valid.
        clean: dict[str, str] = {}
        for spec in self.inputs:
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            if not val or val not in cols:
                self.status.setText("Pick valid input columns first to copy its code.")
                return None
            clean[spec["key"]] = val
        lat = self.lat.value() if self.lat is not None else None
        lon = self.lon.value() if self.lon is not None else None
        utc = self.utc.value() if self.utc is not None else None
        vpd_in_kpa = self.vpd_kpa_cb.isChecked() if self.vpd_kpa_cb is not None else True
        return partitioning_to_code(
            self.method_suffix, clean, lat=lat, lon=lon, utc_offset=utc,
            vpd_in_kpa=vpd_in_kpa)

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        cols = {str(c) for c in self._df.columns}
        picks = self._picks()
        missing = []
        for spec in self.inputs:
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            if not val or val not in cols:
                missing.append(spec["label"])
        if missing:
            self.status.setText("Pick valid columns for: " + ", ".join(missing))
            return

        series_map: dict[str, pd.Series] = {}
        for spec in self.inputs:
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            series_map[spec["key"]] = self._df[val]

        coords = {}
        if self.lat is not None:
            coords["lat"] = self.lat.value()
        if self.lon is not None:
            coords["lon"] = self.lon.value()
        if self.utc is not None:
            coords["utc_offset"] = self.utc.value()
        vpd_in_kpa = self.vpd_kpa_cb.isChecked() if self.vpd_kpa_cb is not None else True

        self._set_running(True)
        self.progress.start_busy("Partitioning… (this can take a while per year)")
        self.status.setText("Partitioning… (this can take a while per year)")
        self._runner.run(self._compute_payload, series_map, coords, vpd_in_kpa,
                         picks[self._nee_key])

    def _compute_payload(self, series_map, coords, vpd_in_kpa, nee_name):
        """Run the library partitioner off-thread; return the results DataFrame.
        Raises on error (the runner forwards to :meth:`_on_failed`)."""
        part = self._build_partitioner(series_map, coords, vpd_in_kpa)
        part.run()
        results = part.results.copy()
        # Provenance: every output column is derived from the measured NEE.
        attrs = {}
        for col in results.columns:
            attrs[str(col)] = provenance_attr(
                origin=DERIVED, parent=str(nee_name), operation=self.title,
                tags=["partitioning", self.method_suffix.lower()])
        results.attrs[ATTRS_KEY] = attrs
        return results

    def _build_partitioner(self, series_map: dict, coords: dict, vpd_in_kpa: bool):
        """Construct the (not-yet-run) library partitioner (subclass hook)."""
        raise NotImplementedError

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    # --- results -------------------------------------------------------
    def _on_done(self, results: pd.DataFrame) -> None:
        self._set_running(False)
        self.progress.finish()
        self._results = results
        n_reco = int(results[self.reco_col].notna().sum()) if self.reco_col in results else 0
        n_gpp = int(results[self.gpp_col].notna().sum()) if self.gpp_col in results else 0
        self.status.setText(
            f"Done. {n_reco} RECO and {n_gpp} GPP values. "
            f"'Add' appends {len(results.columns)} {self.method_suffix} columns "
            f"({', '.join(str(c) for c in results.columns)}).")
        self._update_hero(results, n_reco, n_gpp)
        self.add_btn.setEnabled(True)
        self._plot(results)

    def _update_hero(self, results: pd.DataFrame, n_reco: int, n_gpp: int) -> None:
        """Headline cumulative carbon totals (same units as the cumulative plot)."""
        metrics = []
        if self.reco_col in results:
            metrics.append(("Σ RECO", _fmt(results[self.reco_col].sum()),
                            f"Sum of {self.reco_col} over the record"))
        if self.gpp_col in results:
            metrics.append(("Σ GPP", _fmt(results[self.gpp_col].sum()),
                            f"Sum of {self.gpp_col} over the record"))
        nee = self._df[self._picks()[self._nee_key]] if self._df is not None else None
        if nee is not None:
            metrics.append(("Σ NEE", _fmt(nee.sum()), "Sum of the measured NEE"))
        metrics.append(("RECO N", f"{n_reco:,}", "Number of RECO values produced"))
        metrics.append(("GPP N", f"{n_gpp:,}", "Number of GPP values produced"))
        self.hero.set_metrics(metrics)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.progress.finish()
        self.hero.clear()
        self.status.setText(f"Failed: {msg}")
        self.canvas.show_message("Partitioning failed")

    def _plot(self, results: pd.DataFrame) -> None:
        """Two stacked panels: measured NEE + GPP/RECO time series (top) and a
        cumulative-sum comparison (bottom). Daily means keep a year readable."""
        ax_ts, ax_cum = self.canvas.new_axes(2, orientation="vertical", sharex=True)
        nee = self._df[self._picks()[self._nee_key]] if self._df is not None else None
        reco = results.get(self.reco_col)
        gpp = results.get(self.gpp_col)

        def _daily(s):
            return s.resample("1D").mean() if s is not None else None

        if nee is not None:
            d = _daily(nee)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_NEE, lw=0.8, alpha=0.9,
                       label="NEE (measured, daily mean)")
        if reco is not None:
            d = _daily(reco)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_RECO, lw=1.0,
                       label=f"{self.reco_col} (daily mean)")
        if gpp is not None:
            d = _daily(gpp)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_GPP, lw=1.0,
                       label=f"{self.gpp_col} (daily mean)")
        ax_ts.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
        ax_ts.set_ylabel("umol m-2 s-1")
        ax_ts.set_title(f"{self.title} — daily means", fontsize=9)
        ax_ts.legend(frameon=False, fontsize=8)

        # Cumulative sums (only over records where both are present).
        if reco is not None:
            c = reco.fillna(0).cumsum()
            ax_cum.plot(c.index, c.to_numpy(), color=_C_RECO, lw=1.0,
                        label=f"Σ {self.reco_col}")
        if gpp is not None:
            c = gpp.fillna(0).cumsum()
            ax_cum.plot(c.index, c.to_numpy(), color=_C_GPP, lw=1.0,
                        label=f"Σ {self.gpp_col}")
        ax_cum.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
        ax_cum.set_ylabel("cumulative")
        ax_cum.set_title("Cumulative sum", fontsize=9)
        ax_cum.legend(frameon=False, fontsize=8)
        self.canvas.draw()

    def _add(self) -> None:
        if self._results is None or self._results.empty:
            return
        self.featuresCreated.emit(self._results)
        self.status.setText(
            f"Added {', '.join(str(c) for c in self._results.columns)} to the "
            f"variable list.")
        self.add_btn.setEnabled(False)
