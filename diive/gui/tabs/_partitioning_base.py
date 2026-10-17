"""
GUI.TABS._PARTITIONING_BASE: SHARED BASE FOR NEE-PARTITIONING TABS
=================================================================

Common machinery for the GUI's NEE -> GPP + RECO partitioning tab: a
per-input column picker (one combo per series, auto-seeded to the standard
name with an availability marker), the site coordinates (latitude / longitude
seeded from Project settings, the UTC offset shown read-only from Project
settings), an optional VPD-unit toggle, a worker thread that runs the library
partitioner, a result plot (measured NEE + partitioned GPP + RECO), and "Add
results to dataset".

A concrete tab subclasses :class:`BasePartitioningTab` and supplies the
method-specific bits: which input columns and coordinates it needs, the result
column names to plot, and how to construct the library partitioner. A tab that
switches between variants (``tabs/partitioning.py``) returns these from
properties and calls :meth:`_apply_variant` after a switch: every control is
built once and the ones the current variant does not need are hidden. All
partitioning is library work (``dv.flux.PartitioningNEE``). This base only
collects the inputs, runs the class on a worker thread, previews the result,
and emits the new columns (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand, _fmt
from diive.gui.widgets.column_picker import ColumnPicker
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.project_offset import ProjectUtcOffset
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

    Subclasses set the class attributes below (or override them with
    properties) and implement :meth:`_build_partitioner` and
    :meth:`_python_code_for`."""

    #: Tab title (DiiveTab requirement) — set by the subclass.
    title = "NEE partitioning"
    #: One-line description shown above the controls.
    intro = "Partition measured NEE into GPP and RECO."
    #: Every input column the tab can ask for. Each entry is a dict with keys:
    #: ``key`` (combo id), ``label`` (form label), ``needle`` (auto-pick
    #: substring) and optional ``prefer`` / ``avoid`` / ``optional`` / ``tip``.
    #: :meth:`_active_inputs` says which of them the current variant uses.
    inputs: list[dict] = []
    #: Which site coordinates the method requires.
    needs_lat = False
    needs_lon = False
    needs_utc = False

    def _active_inputs(self) -> list[dict]:
        """The input specs the current variant uses (default: all)."""
        return list(self.inputs)

    def _variant(self) -> dict:
        """Plain-data description of the current variant, handed to the worker
        (so :meth:`_compute_payload` never reads a widget off-thread)."""
        return {}

    @property
    def needs_coords(self) -> bool:
        """True if this method reads any site coordinate."""
        return bool(self.needs_lat or self.needs_lon or self.needs_utc)

    def _coords_missing(self) -> bool:
        """Refuse to run without a configured site, and say why.

        The lat/lon spin boxes default to 0/0 and the project UTC offset to 0,
        and `_seed_site` leaves them there when the site is unset — so an unguarded run would silently
        partition at (0, 0) on UTC and return plausible-looking GPP/RECO. Mirrors
        the guard in the outlier and correction tabs.
        """
        if not self.needs_coords or site.manager.configured:
            return False
        self.status.setText(
            "This partitioning needs the site location for its day/night split, "
            "but no coordinates are configured. Set latitude, longitude and UTC "
            "offset in Settings -> Project settings first (running now would "
            "silently partition at (0, 0) at UTC and corrupt the result).")
        return True
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
        # RECO/GPP column names of the run behind self._results (the variant
        # can change while a result is on screen).
        self._result_cols = (self.reco_col, self.gpp_col)
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
        self._apply_variant()
        return root

    def _build_controls(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(320)
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        self._intro_lbl = QLabel(self.intro)
        self._intro_lbl.setWordWrap(True)
        self._intro_lbl.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(self._intro_lbl)

        # Variant switches (method / mode), above the inputs they decide.
        self._add_variant_controls(v)

        self.picker = ColumnPicker(self.inputs, title="Input columns")
        v.addWidget(self.picker)

        # Every coordinate row is built. _apply_variant hides the unused ones.
        self._site_box = QGroupBox("Site coordinates")
        self._site_form = sf = QFormLayout(self._site_box)
        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(4)
        self.lat.setToolTip("Site latitude in decimal degrees (north positive).")
        sf.addRow("Latitude", self.lat)
        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(4)
        self.lon.setToolTip("Site longitude in decimal degrees (east positive).")
        sf.addRow("Longitude", self.lon)
        # Read-only: the offset has one home, Project settings.
        self.utc = ProjectUtcOffset()
        sf.addRow("UTC offset", self.utc)
        note = QLabel("Defaults from Settings ▸ Project settings.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_C_MUTED};")
        sf.addRow(note)
        v.addWidget(self._site_box)
        self._seed_site()

        self._opt_box = QGroupBox("Options")
        of = QVBoxLayout(self._opt_box)
        self.vpd_kpa_cb = QCheckBox("VPD is in kPa (diive convention)")
        self.vpd_kpa_cb.setChecked(True)
        self.vpd_kpa_cb.setToolTip(
            "Checked: the VPD column is in kPa and converted to hPa internally "
            "(the unit the Lasslop light-response curve expects). Uncheck if the "
            "column is already in hPa.")
        of.addWidget(self.vpd_kpa_cb)
        v.addWidget(self._opt_box)

        # Run/Add anchored below the settings (not in the header) — same as the
        # correction/outlier/uncertainty tabs.
        v.addWidget(self.run_btn)
        v.addWidget(self.add_btn)

        v.addStretch(1)
        return host

    def _add_variant_controls(self, layout: QVBoxLayout) -> None:
        """Add variant switches above the input columns (subclass hook)."""

    def _apply_variant(self) -> None:
        """Show only the inputs, coordinates and options the variant needs."""
        self.picker.set_visible_keys(s["key"] for s in self._active_inputs())
        self._site_form.setRowVisible(self.lat, self.needs_lat)
        self._site_form.setRowVisible(self.lon, self.needs_lon)
        self._site_form.setRowVisible(self.utc, self.needs_utc)
        self._site_box.setVisible(self.needs_coords)
        self._opt_box.setVisible(self.has_vpd_unit)
        self._intro_lbl.setText(self.intro)
        self.hero.set_chip(self.method_suffix)

    # --- site ----------------------------------------------------------
    def _seed_site(self) -> None:
        m = site.manager
        if not m.configured:
            return
        self.lat.setValue(m.latitude)
        self.lon.setValue(m.longitude)

    def _on_site_changed(self) -> None:
        self._seed_site()
        # A result computed before the change used the old coordinates / UTC
        # offset: keep the preview but don't let it be added.
        if self._results is not None and self.needs_coords:
            self.add_btn.setEnabled(False)
            self.status.setText("Project settings changed after this run. Run the "
                                "partitioning again to use them.")

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._results = None
        self.add_btn.setEnabled(False)
        self.picker.seed(df.columns)

    # --- state ---------------------------------------------------------
    def _state_controls(self) -> dict:
        # Every combo, also the hidden ones, so a pick survives a variant switch.
        controls = self.picker.combos()
        controls["lat"] = self.lat
        controls["lon"] = self.lon
        # No "utc": the offset comes from Project settings, so a value saved by
        # an older project is ignored on restore.
        controls["vpd_kpa"] = self.vpd_kpa_cb
        return controls

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._state_controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls, unrestored_message
        unrestored = restore_controls(self._state_controls(), state.get("controls"))
        self.picker.refresh_availability()
        # A hidden input does not matter for this variant: report only the rest.
        hidden = {s["key"] for s in self.inputs} - {s["key"] for s in self._active_inputs()}
        unrestored = [k for k in unrestored if k not in hidden]
        if unrestored:
            labels = {s["key"]: s["label"] for s in self.inputs}
            self.status.setText(unrestored_message(unrestored, labels))

    # --- run -----------------------------------------------------------
    def _picks(self) -> dict[str, str]:
        return self.picker.picks()

    def _coords(self) -> dict:
        """The site coordinates the current variant needs, as library kwargs."""
        coords = {}
        if self.needs_lat:
            coords["lat"] = self.lat.value()
        if self.needs_lon:
            coords["lon"] = self.lon.value()
        if self.needs_utc:
            coords["utc_offset"] = self.utc.value()
        return coords

    def _python_code(self) -> str | None:
        """Runnable snippet reproducing this partitioning, or None when the inputs
        are incomplete (button no-op, mirroring :meth:`_run`'s validation)."""
        if self._df is None:
            self.status.setText("Load data first to copy its code.")
            return None
        if self._coords_missing():
            return None  # the snippet would carry lat=0.0, lon=0.0
        cols = {str(c) for c in self._df.columns}
        picks = self._picks()
        # Drop optional inputs left as "(none)"; require every other pick to be valid.
        clean: dict[str, str] = {}
        for spec in self._active_inputs():
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            if not val or val not in cols:
                self.status.setText("Pick valid input columns first to copy its code.")
                return None
            clean[spec["key"]] = val
        return self._python_code_for(clean, self._coords(),
                                     self.vpd_kpa_cb.isChecked())

    def _python_code_for(self, picks: dict, coords: dict, vpd_in_kpa: bool) -> str:
        """Render the snippet from validated picks (subclass hook)."""
        raise NotImplementedError

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        if self._coords_missing():
            return
        cols = {str(c) for c in self._df.columns}
        picks = self._picks()
        missing = []
        for spec in self._active_inputs():
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            if not val or val not in cols:
                missing.append(spec["label"])
        if missing:
            self.status.setText("Pick valid columns for: " + ", ".join(missing))
            return

        series_map: dict[str, pd.Series] = {}
        for spec in self._active_inputs():
            val = picks[spec["key"]]
            if spec.get("optional") and val == "(none)":
                continue
            series_map[spec["key"]] = self._df[val]

        coords = self._coords()
        vpd_in_kpa = self.vpd_kpa_cb.isChecked()
        variant = self._variant()

        self._set_running(True)
        self._result_cols = (self.reco_col, self.gpp_col)
        self.progress.start_busy("Partitioning… (this can take a while per year)")
        self.status.setText("Partitioning… (this can take a while per year)")
        self._runner.run(self._compute_payload, series_map, coords, vpd_in_kpa,
                         picks[self._nee_key], variant)

    def _compute_payload(self, series_map, coords, vpd_in_kpa, nee_name, variant):
        """Run the library partitioner off-thread and return the results DataFrame.

        Pure: everything it needs arrives as arguments. Raises on error (the
        runner forwards to :meth:`_on_failed`)."""
        part = self._build_partitioner(series_map, coords, vpd_in_kpa, variant)
        part.run()
        results = part.results.copy()
        # Provenance: every output column is derived from the measured NEE.
        suffix = variant.get("suffix", self.method_suffix)
        attrs = {}
        for col in results.columns:
            attrs[str(col)] = provenance_attr(
                origin=DERIVED, parent=str(nee_name),
                operation=variant.get("operation", self.title),
                tags=["partitioning", suffix.lower()])
        results.attrs[ATTRS_KEY] = attrs
        return results

    def _build_partitioner(self, series_map: dict, coords: dict, vpd_in_kpa: bool,
                           variant: dict):
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
        reco_col, gpp_col = self._result_cols
        n_reco = int(results[reco_col].notna().sum()) if reco_col in results else 0
        n_gpp = int(results[gpp_col].notna().sum()) if gpp_col in results else 0
        suffix = reco_col.removeprefix("RECO_")
        self.status.setText(
            f"Done. {n_reco} RECO and {n_gpp} GPP values. "
            f"'Add' appends {len(results.columns)} {suffix} columns "
            f"({', '.join(str(c) for c in results.columns)}).")
        self._update_hero(results, n_reco, n_gpp)
        self.add_btn.setEnabled(True)
        self._plot(results)

    def _update_hero(self, results: pd.DataFrame, n_reco: int, n_gpp: int) -> None:
        """Headline cumulative carbon totals (same units as the cumulative plot)."""
        metrics = []
        reco_col, gpp_col = self._result_cols
        if reco_col in results:
            metrics.append(("Σ RECO", _fmt(results[reco_col].sum()),
                            f"Sum of {reco_col} over the record"))
        if gpp_col in results:
            metrics.append(("Σ GPP", _fmt(results[gpp_col].sum()),
                            f"Sum of {gpp_col} over the record"))
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
        reco_col, gpp_col = self._result_cols
        reco = results.get(reco_col)
        gpp = results.get(gpp_col)

        def _daily(s):
            return s.resample("1D").mean() if s is not None else None

        if nee is not None:
            d = _daily(nee)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_NEE, lw=0.8, alpha=0.9,
                       label="NEE (measured, daily mean)")
        if reco is not None:
            d = _daily(reco)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_RECO, lw=1.0,
                       label=f"{reco_col} (daily mean)")
        if gpp is not None:
            d = _daily(gpp)
            ax_ts.plot(d.index, d.to_numpy(), color=_C_GPP, lw=1.0,
                       label=f"{gpp_col} (daily mean)")
        ax_ts.axhline(0, color="#B0BEC5", lw=0.6, zorder=0)
        ax_ts.set_ylabel("umol m-2 s-1")
        ax_ts.set_title(f"{self.title}, {reco_col.removeprefix('RECO_')} — daily means",
                        fontsize=9)
        ax_ts.legend(frameon=False, fontsize=8)

        # Cumulative sums (only over records where both are present).
        if reco is not None:
            c = reco.fillna(0).cumsum()
            ax_cum.plot(c.index, c.to_numpy(), color=_C_RECO, lw=1.0,
                        label=f"Σ {reco_col}")
        if gpp is not None:
            c = gpp.fillna(0).cumsum()
            ax_cum.plot(c.index, c.to_numpy(), color=_C_GPP, lw=1.0,
                        label=f"Σ {gpp_col}")
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
