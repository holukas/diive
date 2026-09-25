"""
GUI.TABS.METEO_SCREENING: SCREEN & RESAMPLE HIGH-RES DATABASE METEO
==================================================================

The full screening experience (:class:`~diive.gui.tabs._screening_base.ScreeningTabBase`)
applied to a high-resolution meteo field handed over from the Database explorer,
**plus resampling**. It is feature-identical to the Stepwise screening tab —
editable outlier-test cards, corrections, QCF, the live preview, Copy Python —
and adds a **Resample** page; on Add it emits the screened (and corrected) series
resampled to the chosen resolution (default: the working dataset's detected
resolution, else 30min), ready to merge into the working dataset.

The screening runs through the library class
:class:`~diive.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb`, which
handles database downloads with one or more time resolutions (all records kept,
time-weighted resampling, day/night at TIMESTAMP_MIDDLE). How the tab differs from
the plain screening tab, all via the base's seams:
- **Data source**: the staged ``data_detailed`` builds a ``StepwiseMeteoScreeningDb``
  on a worker thread. A failed build (too few records, jittered timestamps, ...)
  shows in the status line and leaves the tab empty. ``self._df`` holds the
  field on the library's screening grid (TIMESTAMP_MIDDLE) for the variable list;
  the database tags are kept aside and re-attached on emit.
- **Chain, QCF, corrections**: each run screens a copy of the loaded instance
  (``start_outlier_detection`` -> steps + ``addflag`` ->
  ``finalize_outlier_detection``); corrections go through ``set_corrections``.
  Plots use the class's display helpers, so coarse records show in full.
- **Site coordinates** are passed at run time (project latitude, longitude and
  UTC offset), so edits in Project settings apply to the next run; results
  computed with the old coordinates are cleared. The Database explorer downloads
  in the project's UTC offset too, so the staged timestamps match it. While the
  offset is not set, the status line says so (0 is used).
- **Copy Python**: ``meteoscreening_to_code`` renders a ``StepwiseMeteoScreeningDb``
  script (download, steps, QCF, corrections, resampling).
- **Extra inspector page**: ``Resample`` (target resolution + aggregation + min counts).
- **Emit**: ``resample()`` to the target resolution (TIMESTAMP_END), converted
  END -> MIDDLE (so it aligns onto diive's TIMESTAMP_MIDDLE index), with a
  collision rename, an overlap guard, and the InfluxDB origin + all db tags in
  the variable's history.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import copy
import types

import pandas as pd
from pandas.tseries.frequencies import to_offset
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.core.times.times import DetectFrequency, convert_series_timestamp_to_middle
from diive.gui import db, site
from diive.gui.tabs._screening_base import ScreeningTabBase
from diive.gui.widgets.project_offset import (
    NOT_SET_WARNING,
    format_utc_offset,
    project_utc_offset,
    project_utc_offset_is_set,
)
from diive.gui.widgets.worker import LatestRunner
from diive.preprocessing.qaqc.codegen import meteoscreening_to_code
from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb
from diive.qaqc import detect_measurement

_C_MUTED = "#6B7780"
#: Common target resolutions offered in the resample picker (editable).
_FREQS = ["30min", "10min", "15min", "1h"]


def _load_screening(frame, field: str, site_name: str, coords: dict) -> dict:
    """Build the library screening for *field* on a worker thread; pure (no Qt).

    Returns the loaded instance (never screened or corrected: each run copies
    it), a working copy for corrections before any run, and what the status
    line and the history need."""
    mscr = StepwiseMeteoScreeningDb(data_detailed={field: frame}, fields=field,
                                    site=site_name, **coords)
    return {"field": field, "mscr": mscr, "work": copy.deepcopy(mscr),
            "first_end": frame.index.min(), "last_end": frame.index.max(),
            # resample() rewrites 'freq' and 'data_version': keep the download's.
            "tags": dict(mscr.tags[field]),
            "records": mscr.records_count(field), "resolutions": mscr.resolutions(field)}


class MeteoScreeningTab(ScreeningTabBase):
    """Full screening + resampling for a high-res field from the database."""

    title = "Meteo screening (database)"
    add_button_label = "Add resampled to dataset"
    provenance_op = "Imported from InfluxDB & screened"

    def build(self) -> QWidget:
        # DB-specific state (the base is otherwise self._df / self._var-centric).
        self._tags: dict = {}          # {field: {tag: value}} from the download
        self._db_meta: dict = {}        # bucket / measurement / data_version / utc_offset
        self._dataset_index = None      # working dataset index (overlap check)
        self._dataset_columns: set = set()  # working dataset columns (collision check)
        self._target_freq: str | None = None  # main-df resolution = resample target
        self._source_freq: str | None = None  # downloaded data resolution
        self._utc_offset = None         # UTC offset the staged data were downloaded in
        self._mscr = None               # loaded StepwiseMeteoScreeningDb (read-only here)
        self._mscr_work = None          # its copy for corrections before any run
        self._resolutions: list = []    # time resolution(s) of the download
        self._n_records: int | None = None  # records in the download (not grid slots)
        self._loading_field: str | None = None
        self._site_name: str = ""       # site name the class is built with
        self._db_range: tuple | None = None  # download (start, stop) for Copy Python
        # Coordinates the current results were computed with (see _on_site_changed).
        self._site_coords: dict = self._coords()
        # Builds the library class off the GUI thread; the newest download wins.
        self._runner = LatestRunner()
        self._runner.done.connect(self._on_loaded)
        self._runner.failed.connect(self._on_load_failed)
        root = super().build()
        # Bound method, not a lambda: site.manager is a singleton.
        site.manager.changed.connect(self._on_site_changed)
        return root

    # --- extra inspector page: Resample ---
    def _inspector_pages(self) -> list:
        pages = super()._inspector_pages()
        pages.append(("Resample", self._build_resample_page, self._INSPECTOR_W))
        return pages

    def _build_resample_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        box = QGroupBox("Resample (after screening)")
        f = QFormLayout(box)
        self._freq = QComboBox()
        self._freq.setEditable(True)
        self._freq.addItems(_FREQS)
        self._freq.setToolTip("Target resolution, e.g. 30min (default), 10min, 1h.")
        self._agg = QComboBox()
        self._agg.addItems(["mean", "sum"])
        self._agg.setToolTip("mean for most meteo, sum for precipitation.")
        self._mincounts = QDoubleSpinBox()
        self._mincounts.setRange(0.0, 1.0)
        self._mincounts.setSingleStep(0.05)
        self._mincounts.setValue(0.9)
        self._mincounts.setToolTip(
            "Minimum fraction of each target period that the kept records must "
            "cover (each record counts the time it covers), else the resampled "
            "value is missing. The minimum is counted in slots of the "
            "high-resolution grid and rounded down; when it is below three slots, "
            "one covered slot is enough (e.g. 10-min data resampled to 30 min "
            "need only one record per period).")
        f.addRow("Resolution", self._freq)
        f.addRow("Aggregation", self._agg)
        f.addRow("Min. records (frac)", self._mincounts)
        for w in (self._freq, self._agg):
            w.currentTextChanged.connect(self._on_resample_changed)
        self._mincounts.valueChanged.connect(self._on_resample_changed)
        v.addWidget(box)
        hint = QLabel(
            "The screened (and corrected) series is resampled to this resolution, "
            "then added on TIMESTAMP_MIDDLE so it aligns with the working dataset. "
            "The default target is the working dataset's resolution; if the "
            "downloaded data already match it, no resampling is done.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(hint)
        self._resample_info = QLabel("")
        self._resample_info.setWordWrap(True)
        v.addWidget(self._resample_info)
        v.addStretch(1)
        return page

    @staticmethod
    def _freq_str(index) -> str | None:
        """Normalised resolution of *index* (e.g. '30min', '1h'), or None."""
        if index is None or len(index) < 2:
            return None
        try:
            detected = DetectFrequency(index).freq
        except Exception:
            return None
        if not detected:
            return None
        off = to_offset(detected)
        return f"{off.n}{off.name}"

    @staticmethod
    def _same_freq(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        try:
            return to_offset(a) == to_offset(b)
        except Exception:
            return a == b

    def _agg_hint(self) -> str:
        """Flag a plain arithmetic mean on a circular/bounded variable (non-blocking)."""
        if self._agg.currentText() != "mean":
            return ""
        meas = detect_measurement(self._var) if self._var else None
        if meas == "WD":
            return ("  NOTE: wind direction is circular; a plain mean is misleading "
                    "across the 0/360 deg wrap.")
        if meas == "RH":
            return ("  NOTE: relative humidity is bounded (0-100%); a plain mean can "
                    "be misleading.")
        return ""

    def _resolution_text(self) -> str:
        """The download's resolution(s), e.g. '10min' or '10min and 1min'."""
        if not self._resolutions:
            return "unknown resolution"
        return " and ".join(self._resolutions)

    def _update_resample_info(self) -> None:
        """Show source vs target resolution and whether resampling will happen."""
        src = self._resolution_text() if self._resolutions else "?"
        tgt = self._target_freq or "?"
        selected = self._freq.currentText().strip()
        if self._same_freq(self._source_freq, selected):
            verdict = "-> already at target resolution, no resampling needed."
        else:
            verdict = f"-> resample to {selected} ({self._agg.currentText()})."
        self._resample_info.setText(
            f"Source: {src}  ·  dataset target: {tgt}  {verdict}{self._agg_hint()}")

    def _on_resample_changed(self, *_) -> None:
        """Resample settings changed: refresh the info + rebuild the column."""
        self._update_resample_info()
        self._build_result()
        self.add_btn.setEnabled(self._result_df is not None and not self._result_df.empty)

    # --- staged data intake (from the Database explorer) ---
    def load_staged(self, payload: dict) -> None:
        """Take a download handed over from the Database explorer. The previous
        field is cleared at once; the library class is built on a worker and
        :meth:`_on_loaded` (or :meth:`_on_load_failed`) fills the tab."""
        self._clear_loaded()
        data_detailed = payload.get("data_detailed") or {}
        field = payload.get("field")
        frame = data_detailed.get(field)
        if frame is None or field is None or field not in frame.columns:
            self._runner.cancel()  # a slower earlier download must not land now
            self.status.setText("No data received from the Database explorer.")
            return
        self._db_meta = {k: payload.get(k) for k in
                         ("bucket", "measurement", "data_version", "utc_offset")}
        self._utc_offset = payload.get("utc_offset")
        self._loading_field = field
        self._site_name = (str(frame["site"].dropna().iloc[0])
                           if "site" in frame.columns and frame["site"].notna().any()
                           else str(payload.get("bucket") or ""))
        self._site_coords = self._coords()
        self.status.setText(f"Loading {field} ({len(frame)} rows from the database) ...")
        self._runner.submit(_load_screening, frame, field, self._site_name,
                            self._site_coords)

    def _clear_loaded(self) -> None:
        """Forget the loaded field and everything computed from it, so a new or
        failed download never shows the previous field's data."""
        self._run_id += 1  # a chain run still in flight belongs to the old field
        self._mscr = self._mscr_work = None
        self._df = None
        self._var = None
        self._tags = {}
        self._resolutions = []
        self._n_records = None
        self._source_freq = None
        self._db_range = None
        self._payload = None
        self._corrected = None
        self._result_df = None
        self._selected_step = -1
        self.varpanel.set_variables([], None)
        self.qcf_label.setText("QCF: run to compute.")
        self.report_text.clear()
        self.report_copy_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self._rebuild_cards()
        self._refresh_preview()
        self._update_resample_info()

    def _on_loaded(self, res: dict) -> None:
        field = res["field"]
        mscr = res["mscr"]
        self._mscr = mscr
        self._mscr_work = res["work"]
        self._loading_field = None
        self._tags = {field: res["tags"]}
        self._resolutions = list(res["resolutions"])
        self._n_records = res["records"]
        self._source_freq = self._resolutions[0] if len(self._resolutions) == 1 else None
        # The download range for Copy Python: the payload has no request range, so
        # use the records' END timestamps; stop is excluded, so add the finest step.
        finest = min((pd.Timedelta(to_offset(r)) for r in self._resolutions),
                     default=pd.Timedelta(0))
        self._db_range = (f"{res['first_end']:%Y-%m-%d %H:%M:%S}",
                          f"{res['last_end'] + finest:%Y-%m-%d %H:%M:%S}")
        # The field on the library's screening grid (TIMESTAMP_MIDDLE): the
        # variable list and the base's selection work on it.
        self._df = pd.DataFrame({field: mscr.series_hires_orig[field]})

        # Default the resample target to the working dataset's resolution (so
        # matching data won't be resampled).
        if self._target_freq:
            self._freq.setCurrentText(self._target_freq)
        self.varpanel.set_variables(self._df.columns, None)
        meas = detect_measurement(field)
        self._agg.setCurrentText("sum" if meas == "PREC" else "mean")
        self._select(field)  # base: show raw, rebuild cards, detect measurement
        self._update_resample_info()
        same = self._same_freq(self._source_freq, self._freq.currentText().strip())
        note = ("already at the dataset resolution, so no resampling is needed"
                if same else f"will be resampled to {self._freq.currentText().strip()}")
        self.status.setText(
            f"Loaded {field} from the database ({self._n_records} records at "
            f"{self._resolution_text()}, {self._tz_label()}); {note}. "
            f"Add screening steps, then add to the dataset.{self._tz_warning()}")

    def _on_load_failed(self, msg: str) -> None:
        field = self._loading_field or "the field"
        self._loading_field = None
        self.status.setText(f"Could not load {field} for screening: {msg}")

    def _tz_label(self) -> str:
        """The timezone the staged data are in, e.g. 'UTC+01:00'."""
        if self._utc_offset is None:
            return "timezone unknown"
        return format_utc_offset(self._utc_offset)

    def _tz_warning(self) -> str:
        """Status suffix while Project settings have no UTC offset, or after
        the offset was changed there since the download (the staged timestamps
        keep the offset they were downloaded in)."""
        if not project_utc_offset_is_set():
            return f"  {NOT_SET_WARNING}"
        if self._utc_offset is not None and self._utc_offset != project_utc_offset():
            return (f"  The project's UTC offset changed after the download "
                    f"({self._tz_label()}): send the field again from the "
                    f"Database explorer.")
        return ""

    def on_data_loaded(self, df, created=None) -> None:
        # The screening source is the staged DB field, NOT the working dataset;
        # keep the dataset's index/columns (overlap + collision checks) and its
        # resolution (the resample target the downloaded data is compared to).
        self._dataset_index = df.index if df is not None else None
        self._dataset_columns = set(map(str, df.columns)) if df is not None else set()
        self._target_freq = self._freq_str(df.index) if df is not None else None
        self.corrections_panel.set_coords_available(site.manager.configured)

    # --- site coordinates: read at run time, so Project settings edits apply ---
    def _coords(self) -> dict:
        """The project's site coordinates and UTC offset (0 when not set)."""
        coords = ScreeningTabBase._coords()
        coords["utc_offset"] = project_utc_offset()
        return coords

    @staticmethod
    def _apply_coords(mscr, coords: dict) -> None:
        """Give *mscr* the current coordinates. The class reads them when a step
        needs them, so nothing has to be rebuilt."""
        mscr.site_lat = coords["site_lat"]
        mscr.site_lon = coords["site_lon"]
        mscr.utc_offset = coords["utc_offset"]

    def _on_site_changed(self) -> None:
        """Project settings saved: the next run and correction use the new
        coordinates. Results computed with the old ones are cleared and the run
        buttons marked pending, as when the variable is selected again."""
        self.corrections_panel.set_coords_available(site.manager.configured)
        coords = self._coords()
        if coords == self._site_coords:
            return  # e.g. only the site name or notes changed
        self._site_coords = coords
        if self._mscr is None or self._var is None:
            return
        if self._payload is None and self._corrected is None:
            return
        self._show_variable(self._var, redetect_measurement=False)
        self.status.setText(
            "Site coordinates changed: click Run outliers / Run corrections to "
            f"screen {self._var} with them.{self._tz_warning()}")

    # --- screening through the library class (the base's seams) ---
    def _current_mscr(self):
        """The instance the corrections and the emit work on: the last run's,
        else the working copy of the loaded one (no run yet)."""
        if self._payload is not None and "mscr" in self._payload:
            return self._payload["mscr"]
        return self._mscr_work

    def _chain_input(self):
        # The loaded instance; the worker screens a copy of it.
        return self._mscr

    @staticmethod
    def _run_chain(mscr, var, steps, coords, configured) -> dict:
        mscr = copy.deepcopy(mscr)
        MeteoScreeningTab._apply_coords(mscr, coords)
        mscr.start_outlier_detection()
        sod = mscr.outlier_detection[var]
        removed, bounds = ScreeningTabBase._run_steps(mscr, sod, steps)
        # Snapshot for the preview: later corrections rebase the detector's series.
        det = types.SimpleNamespace(series_hires_orig=sod.series_hires_orig.copy(),
                                    series_hires_cleaned=sod.series_hires_cleaned.copy(),
                                    flags=sod.flags.copy())
        mscr.finalize_outlier_detection()
        qcf = mscr.outlier_detection_qcf[var]
        _, report = qcf.screening_report()
        return {"var": var, "detector": det, "removed": removed, "bounds": bounds,
                "qcf": qcf, "report": report, "mscr": mscr}

    def _on_done(self, payload: dict) -> None:
        super()._on_done(payload)
        # The run split day/night with the offset: say so when it is not set
        # (unless the base already did).
        warning = self._tz_warning()
        if (payload.get("run_id") == self._run_id and warning
                and warning.strip() not in self.status.text()):
            self.status.setText(self.status.text() + warning)

    def _compute_corrected(self, corrs: list[dict]):
        mscr = self._current_mscr()
        if mscr is None or self._var is None:
            return None
        self._apply_coords(mscr, self._coords())
        try:
            mscr.set_corrections(corrs)
        except Exception:
            # Emit the uncorrected series after a failed correction, as the
            # base does, not a partly corrected one.
            mscr.set_corrections([])
            raise
        if not corrs:
            return None
        return mscr.series_hires_cleaned[self._var]

    def _raw_series(self):
        if self._mscr is None or self._var is None:
            return None
        return self._mscr.series_hires_orig[self._var]

    def _on_grid(self, data):
        """*data* on the screening grid, which the display helpers expect (a
        detection band may cover only part of it)."""
        grid = self._mscr.series_hires_orig[self._var].index
        return data if data.index.equals(grid) else data.reindex(grid)

    def _display_lines(self, data):
        if self._mscr is None or self._var is None:
            return data
        return self._mscr.display_lines(self._on_grid(data), field=self._var)

    def _display_heatmap(self, data):
        if self._mscr is None or self._var is None:
            return data
        return self._mscr.display_heatmap(self._on_grid(data), field=self._var)

    # --- Copy Python: the library renders the StepwiseMeteoScreeningDb script ---
    def _code_provider(self) -> str | None:
        enabled = self._enabled_steps()
        if not enabled or self._mscr is None or self._var is None:
            return None
        download = None
        if self._db_meta.get("bucket") and self._db_range:
            download = {"bucket": self._db_meta["bucket"],
                        "measurement": self._db_meta.get("measurement"),
                        "data_version": self._db_meta.get("data_version"),
                        "start": self._db_range[0], "stop": self._db_range[1],
                        "dirconf": db.manager.dirconf or None}
        return meteoscreening_to_code(
            enabled, field=self._var, site=self._site_name, **self._coords(),
            download=download, corrections=self.corrections_panel.corrections(),
            to_freqstr=self._freq.currentText().strip() or "30min",
            agg=self._agg.currentText(), mincounts_perc=self._mincounts.value())

    # --- emit: resample the screened series, END -> MIDDLE, with tags ---
    def _emit_frame(self):
        mscr = self._current_mscr()
        if mscr is None or self._var is None:
            return None
        field = self._var
        if mscr.series_hires_cleaned[field].dropna().empty:
            return None

        freq = self._freq.currentText().strip() or "30min"
        try:
            mscr.resample(to_freqstr=freq, agg=self._agg.currentText(),
                          mincounts_perc=self._mincounts.value())
        except Exception as err:
            self.status.setText(f"Resampling failed: {err}")
            return None
        resampled_end = mscr.resampled_detailed[field][field]
        if resampled_end.dropna().empty:
            return None
        # END -> MIDDLE so the column aligns onto diive's TIMESTAMP_MIDDLE index.
        series_mid = convert_series_timestamp_to_middle(resampled_end)

        name = self._unique_name(field)
        series_mid.name = name

        db_tags = self._tags.get(field, {})
        units = db_tags.get("units", "")
        data_version = db_tags.get("data_version", "")
        history = {
            "source": "InfluxDB",
            "bucket": self._db_meta.get("bucket"),
            "measurement": self._db_meta.get("measurement"),
            "db_field": field,
            "utc_offset": self._utc_offset,
            "timezone": self._tz_label(),
            "resample_freq": freq,
            "resample_agg": self._agg.currentText(),
            **db_tags,
        }
        pills = [t for t in ["from-database", "meteo-screened", units, data_version] if t]
        # Only the screened + corrected + resampled value goes into the dataset —
        # the QCF flag is deliberately not emitted here.
        out = pd.DataFrame({name: series_mid})
        out.attrs[ATTRS_KEY] = {
            name: provenance_attr(
                origin=DERIVED, parent=None, operation=self.provenance_op,
                params=history, tags=pills),
        }
        return out

    def _unique_name(self, field: str) -> str:
        """A column name not already in the working dataset (numeric suffix)."""
        taken = self._dataset_columns
        if field not in taken:
            return field
        i = 1
        while f"{field}_{i}" in taken:
            i += 1
        return f"{field}_{i}"

    # --- add: overlap guard + meaningful status ---
    def _add_to_dataset(self) -> None:
        if self._pending_guard():
            return
        if self._result_df is None or self._result_df.empty:
            return
        name = str(self._result_df.columns[0])
        series = self._result_df[name]
        n_total = int(series.count())

        overlap = None
        if self._dataset_index is not None and len(self._dataset_index):
            overlap = int(series.index.isin(self._dataset_index).sum())
            if overlap == 0:
                ds = self._dataset_index
                self.status.setText(
                    f"Not added: the resampled data "
                    f"({series.index.min():%Y-%m-%d %H:%M} to "
                    f"{series.index.max():%Y-%m-%d %H:%M}, {self._freq.currentText().strip()}) "
                    f"does not overlap the working dataset "
                    f"({ds.min():%Y-%m-%d %H:%M} to {ds.max():%Y-%m-%d %H:%M}). "
                    f"Download the meteo for the same period as your data.")
                return

        self.featuresCreated.emit(self._result_df)
        msg = f"Added {name} ({n_total} records) to the dataset"
        if name != self._var:
            msg = f"Added {name} (renamed from existing {self._var}, {n_total} records)"
        if overlap is not None and overlap < n_total:
            msg += (f"; {overlap} align with the dataset's timestamps, "
                    f"the rest fall outside its range (NaN)")
        self.status.setText(msg + ".")
        self.add_btn.setEnabled(False)
