"""
GUI.TABS.METEO_SCREENING: SCREEN & RESAMPLE HIGH-RES DATABASE METEO
==================================================================

Takes a high-resolution meteo field handed over from the Database explorer,
quality-screens it at native resolution, resamples it to 30MIN (or any other
resolution), and adds the screened + resampled column to the working dataset.

All the algorithm work is the library's
:class:`~diive.qaqc.StepwiseMeteoScreeningDb` (the same engine the meteoscreening
notebook workflow uses): it runs the outlier tests + corrections + QCF on the
``data_detailed`` dict the DB download produces, then ``resample()`` to a
tag-stamped 30MIN series. This tab only collects settings, runs the engine on a
worker thread, previews the result, and emits the column.

Timestamp convention: the database stores TIMESTAMP_END and the engine screens /
resamples in END; diive's working frame is TIMESTAMP_MIDDLE, so the resampled
series is converted END -> MIDDLE (which also aligns it onto the 30MIN flux
index) before it is emitted.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diive.core.io.db.influx.common import TAGS
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.core.times.times import convert_series_timestamp_to_middle
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.worker import WorkerRunner
from diive.qaqc import detect_measurement

_C_MUTED = "#6B7780"
#: Common target resolutions offered in the resample picker (editable).
_FREQS = ["30min", "10min", "15min", "1h"]


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject)."""
    features_created = Signal(object)


def _screen_payload(data_detailed: dict, field: str, site_name: str,
                    lat: float, lon: float, utc_offset: int,
                    tests: list, corrections: list, qcf: dict,
                    resample_cfg: dict) -> dict:
    """Pure worker: run the full screen -> QCF -> resample pipeline.

    Returns the original / cleaned high-res series (for the preview) and the
    resampled, tag-carrying DataFrame (TIMESTAMP_END) for the emit / upload.
    """
    from diive.qaqc import StepwiseMeteoScreeningDb  # heavy; import in worker

    mscr = StepwiseMeteoScreeningDb(
        data_detailed=data_detailed, fields=[field], site=site_name,
        site_lat=lat, site_lon=lon, utc_offset=utc_offset)

    # Outlier tests, each committed with addflag() (engine order matters).
    mscr.start_outlier_detection()
    for t in tests:
        getattr(mscr, t["method"])(**t["kwargs"])
        mscr.addflag()
    # Missing-values flag is added directly (no addflag), then QCF is aggregated.
    mscr.flag_missingvals_test()
    mscr.finalize_outlier_detection(
        daytime_accept_qcf_below=qcf["daytime"],
        nighttime_accept_qcf_below=qcf["nighttime"])

    # Corrections run after outlier detection.
    for method in corrections:
        getattr(mscr, method)()

    mscr.resample(to_freqstr=resample_cfg["freq"], agg=resample_cfg["agg"],
                  mincounts_perc=resample_cfg["mincounts"])

    return {
        "orig": mscr.series_hires_orig[field],
        "cleaned": mscr.series_hires_cleaned[field],
        "resampled_detailed": mscr.resampled_detailed[field],
        "field": field,
    }


class MeteoScreeningTab(DiiveTab):
    """Screen + resample a high-res meteo field from the database, then add it
    (30MIN) to the working dataset."""

    title = "Meteo screening (database)"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._payload: dict | None = None       # staged data from the explorer
        self._field: str | None = None
        self._result: dict | None = None         # last screening result
        self._dataset_index = None               # working dataset's index (for overlap check)
        self._dataset_columns: set = set()        # working dataset's columns (for name collisions)

        self._sig = _Signals()
        self.featuresCreated = self._sig.features_created
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(build_titlebar(self.title))

        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.addWidget(self._build_settings())
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, stretch=1)
        outer.addWidget(body, stretch=1)
        return root

    def _build_settings(self) -> QWidget:
        # Scrollable settings column (many groups).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(list_header("Screening", "screen & resample"))

        self._field_lbl = QLabel("No field loaded. Send one from the Database explorer.")
        self._field_lbl.setWordWrap(True)
        self._field_lbl.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(self._field_lbl)

        # --- outlier tests (checkable groups; only ticked ones run) ---
        self._tests = self._build_test_groups()
        for box in self._tests.values():
            outer.addWidget(box)

        # --- corrections (shown per measurement; filled on load) ---
        self._corr_box = QGroupBox("Corrections")
        self._corr_form = QVBoxLayout(self._corr_box)
        self._corr_checks: dict[str, QWidget] = {}
        outer.addWidget(self._corr_box)
        self._corr_box.setVisible(False)

        # --- QCF thresholds ---
        qcf_box = QGroupBox("Quality flag (QCF)")
        qf = QFormLayout(qcf_box)
        self._qcf_day = self._int_spin(0, 2, 2)
        self._qcf_night = self._int_spin(0, 2, 2)
        qf.addRow("Accept daytime QCF below", self._qcf_day)
        qf.addRow("Accept nighttime QCF below", self._qcf_night)
        outer.addWidget(qcf_box)

        # --- resample ---
        rs_box = QGroupBox("Resample")
        rf = QFormLayout(rs_box)
        self._freq = QComboBox()
        self._freq.setEditable(True)
        self._freq.addItems(_FREQS)
        self._freq.setToolTip("Target resolution, e.g. 30min (default), 10min, 1h.")
        self._agg = QComboBox()
        self._agg.addItems(["mean", "sum"])
        self._agg.setToolTip("Aggregation: mean for most meteo, sum for precipitation.")
        self._mincounts = QDoubleSpinBox()
        self._mincounts.setRange(0.0, 1.0)
        self._mincounts.setSingleStep(0.05)
        self._mincounts.setValue(0.25)
        self._mincounts.setToolTip(
            "Minimum fraction of records an interval must have to be aggregated "
            "(else the resampled value is missing).")
        rf.addRow("Resolution", self._freq)
        rf.addRow("Aggregation", self._agg)
        rf.addRow("Min. records (frac)", self._mincounts)
        outer.addWidget(rs_box)

        # --- run + progress + add ---
        self._run_btn = QPushButton("Screen && resample")
        theme.set_button_role(self._run_btn, "confirm")
        self._run_btn.clicked.connect(self._run)
        self._run_btn.setEnabled(False)
        outer.addWidget(self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(self._status)

        self._progress = ProgressBar()
        outer.addWidget(self._progress)

        self._add_btn = QPushButton("Add resampled to dataset")
        self._add_btn.setToolTip(
            "Convert the resampled series to TIMESTAMP_MIDDLE and merge it into "
            "the working dataset (aligned on the 30MIN index).")
        self._add_btn.clicked.connect(self._add)
        self._add_btn.setEnabled(False)
        outer.addWidget(self._add_btn)
        outer.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    # --- test group construction ---------------------------------------

    def _build_test_groups(self) -> dict[str, QGroupBox]:
        """One checkable group per outlier test, holding its key parameters.

        Each group exposes ``._spec()`` returning {method, kwargs} when ticked.
        """
        groups: dict[str, QGroupBox] = {}

        def group(title: str, checked: bool = False) -> tuple[QGroupBox, QFormLayout]:
            box = QGroupBox(title)
            box.setCheckable(True)
            box.setChecked(checked)
            form = QFormLayout(box)
            return box, form

        # Absolute limits
        box, form = group("Absolute limits")
        self._abslim_min = self._float_spin(-1e6, 1e6, -50.0)
        self._abslim_max = self._float_spin(-1e6, 1e6, 50.0)
        form.addRow("Min", self._abslim_min)
        form.addRow("Max", self._abslim_max)
        box._spec = lambda: {"method": "flag_outliers_abslim_test",
                             "kwargs": {"minval": self._abslim_min.value(),
                                        "maxval": self._abslim_max.value()}}
        groups["abslim"] = box

        # z-score
        box, form = group("Z-score")
        self._zscore_thres = self._float_spin(0.5, 100.0, 4.0)
        form.addRow("Threshold", self._zscore_thres)
        box._spec = lambda: {"method": "flag_outliers_zscore_test",
                             "kwargs": {"thres_zscore": self._zscore_thres.value()}}
        groups["zscore"] = box

        # z-score rolling
        box, form = group("Z-score (rolling)")
        self._zr_thres = self._float_spin(0.5, 100.0, 4.0)
        self._zr_win = self._int_spin(0, 100000, 0)
        self._zr_win.setToolTip("Window size in records; 0 = automatic.")
        form.addRow("Threshold", self._zr_thres)
        form.addRow("Window (0=auto)", self._zr_win)
        box._spec = lambda: {"method": "flag_outliers_zscore_rolling_test",
                             "kwargs": {"thres_zscore": self._zr_thres.value(),
                                        "winsize": self._zr_win.value() or None}}
        groups["zscore_rolling"] = box

        # Hampel
        box, form = group("Hampel")
        self._hampel_win = self._int_spin(3, 100000, 13)
        self._hampel_sigma = self._float_spin(0.5, 100.0, 5.5)
        form.addRow("Window length", self._hampel_win)
        form.addRow("n sigma", self._hampel_sigma)
        box._spec = lambda: {"method": "flag_outliers_hampel_test",
                             "kwargs": {"window_length": self._hampel_win.value(),
                                        "n_sigma": self._hampel_sigma.value()}}
        groups["hampel"] = box

        # Local SD
        box, form = group("Local SD")
        self._localsd_nsd = self._float_spin(0.5, 100.0, 7.0)
        form.addRow("n SD", self._localsd_nsd)
        box._spec = lambda: {"method": "flag_outliers_localsd_test",
                             "kwargs": {"n_sd": self._localsd_nsd.value()}}
        groups["localsd"] = box

        # Increments z-score
        box, form = group("Increments z-score")
        self._incr_thres = self._float_spin(0.5, 1000.0, 30.0)
        form.addRow("Threshold", self._incr_thres)
        box._spec = lambda: {"method": "flag_outliers_increments_zcore_test",
                             "kwargs": {"thres_zscore": self._incr_thres.value()}}
        groups["increments"] = box

        return groups

    @staticmethod
    def _float_spin(lo: float, hi: float, val: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(2)
        s.setValue(val)
        return s

    @staticmethod
    def _int_spin(lo: int, hi: int, val: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # --- staged data intake --------------------------------------------

    def load_staged(self, payload: dict) -> None:
        """Receive a high-res field (data_detailed + meta) from the explorer."""
        self._payload = payload
        self._field = payload.get("field")
        data_detailed = payload.get("data_detailed") or {}
        frame = data_detailed.get(self._field)
        if frame is None or self._field is None:
            self._field_lbl.setText("No data received.")
            self._run_btn.setEnabled(False)
            return

        series = frame[self._field]
        units = self._tag_value(frame, "units")
        n = int(series.count())
        meas = detect_measurement(self._field)
        self._field_lbl.setText(
            f"<b>{self._field}</b>  ({units or 'units?'})<br>"
            f"{payload.get('measurement', '?')} / {payload.get('data_version', '?')} "
            f"&middot; {n} records")

        # Sensible aggregation default: sum for precipitation, mean otherwise.
        self._agg.setCurrentText("sum" if meas == "PREC" else "mean")
        self._configure_corrections(meas)

        self._result = None
        self._add_btn.setEnabled(False)
        self._run_btn.setEnabled(True)
        self._status.setText("Configure the screening steps, then run.")
        self._draw_preview(orig=series)

    def _configure_corrections(self, measurement: str | None) -> None:
        """Show only the offset corrections that apply to this measurement."""
        from PySide6.QtWidgets import QCheckBox
        while self._corr_form.count():
            item = self._corr_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._corr_checks.clear()

        applicable = []
        if measurement in ("SW", "PPFD"):
            applicable.append(("correction_remove_nighttime_zero_offset",
                               "Remove nighttime zero offset"))
        if measurement == "RH":
            applicable.append(("correction_remove_relativehumidity_offset",
                               "Remove relative humidity offset (>100%)"))
        for method, label in applicable:
            chk = QCheckBox(label)
            self._corr_form.addWidget(chk)
            self._corr_checks[method] = chk
        self._corr_box.setVisible(bool(applicable))

    @staticmethod
    def _tag_value(frame: pd.DataFrame, tag: str) -> str:
        if tag in frame.columns and len(frame[tag]):
            return str(frame[tag].iloc[0])
        return ""

    def on_data_loaded(self, df, created=None) -> None:
        # This tab screens staged DB data, not the working dataset, but it keeps
        # the dataset's index (overlap check) and columns (name-collision check).
        self._dataset_index = df.index if df is not None else None
        self._dataset_columns = set(map(str, df.columns)) if df is not None else set()

    # --- run ------------------------------------------------------------

    def _run(self) -> None:
        if self._payload is None or self._field is None or self._runner.is_running:
            return
        tests = [box._spec() for box in self._tests.values() if box.isChecked()]
        corrections = [m for m, chk in self._corr_checks.items() if chk.isChecked()]
        qcf = {"daytime": self._qcf_day.value(), "nighttime": self._qcf_night.value()}
        resample_cfg = {"freq": self._freq.currentText().strip() or "30min",
                        "agg": self._agg.currentText(),
                        "mincounts": self._mincounts.value()}

        self._run_btn.setEnabled(False)
        self._add_btn.setEnabled(False)
        self._progress.start_busy("Screening & resampling…")
        self._status.setText("Screening at native resolution, then resampling…")
        self._runner.run(
            _screen_payload,
            {self._field: self._payload["data_detailed"][self._field]},
            self._field, site.manager.name or "site",
            float(site.manager.latitude), float(site.manager.longitude),
            int(self._payload.get("utc_offset", site.manager.utc_offset or 1)),
            tests, corrections, qcf, resample_cfg)

    def _on_done(self, result: dict) -> None:
        self._progress.finish()
        self._run_btn.setEnabled(True)
        self._result = result
        resampled = result["resampled_detailed"]
        n = int(resampled[result["field"]].count())
        self._draw_preview(orig=result["orig"], cleaned=result["cleaned"],
                           resampled=resampled[result["field"]])
        self._add_btn.setEnabled(n > 0)
        self._status.setText(
            f"Screened & resampled to {n} records "
            f"({self._freq.currentText().strip()}). Review, then add to dataset.")

    def _on_failed(self, err: str) -> None:
        self._progress.finish()
        self._run_btn.setEnabled(True)
        self._status.setText(f"Screening failed: {err}")

    # --- preview --------------------------------------------------------

    def _draw_preview(self, orig=None, cleaned=None, resampled=None) -> None:
        ax = self.canvas.new_axes(1)[0]
        if orig is not None:
            ax.plot(orig.index, orig.to_numpy(), color="#B0BEC5", lw=0.6,
                    label="high-res (original)")
        if cleaned is not None:
            ax.plot(cleaned.index, cleaned.to_numpy(), color="#2196F3", lw=0.6,
                    label="high-res (screened)")
        if resampled is not None:
            ax.plot(resampled.index, resampled.to_numpy(), color="#F44336", lw=1.0,
                    marker="o", markersize=2, label="resampled")
        ax.set_title(self._field or "", fontsize=10)
        ax.set_xlabel("Time (TIMESTAMP_END, database timezone)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        self.canvas.draw()
        self.canvas.reset_history()

    # --- add to dataset -------------------------------------------------

    def _add(self) -> None:
        if self._result is None:
            return
        field = self._result["field"]
        resampled = self._result["resampled_detailed"]
        series_end = resampled[field].copy()
        if series_end.dropna().empty:
            self._status.setText("Nothing to add — the resampled series is empty.")
            return
        # END -> MIDDLE so the column aligns onto diive's TIMESTAMP_MIDDLE index.
        series_end.index.name = "TIMESTAMP_END"
        series_mid = convert_series_timestamp_to_middle(series_end)
        series_mid.name = field

        # Guard against a non-overlapping merge: if the resampled timestamps fall
        # entirely outside the working dataset's range (e.g. recent DB data vs a
        # 2022 dataset), the merge would add an all-NaN column. Refuse and explain
        # rather than silently producing one.
        n_total = int(series_mid.count())
        overlap = None
        if self._dataset_index is not None and len(self._dataset_index):
            overlap = int(series_mid.index.isin(self._dataset_index).sum())
            if overlap == 0:
                ds = self._dataset_index
                self._status.setText(
                    f"Not added: the resampled data "
                    f"({series_mid.index.min():%Y-%m-%d} to {series_mid.index.max():%Y-%m-%d}, "
                    f"{self._freq.currentText().strip()}) does not overlap the working "
                    f"dataset ({ds.min():%Y-%m-%d} to {ds.max():%Y-%m-%d}). Download the "
                    f"meteo for the same period (and resolution) as your data.")
                return

        # Rename with a numeric suffix if a column of this name already exists,
        # so adding meteo never overwrites an existing variable.
        name = self._unique_name(field)
        series_mid.name = name
        out = pd.DataFrame({name: series_mid})

        # Provenance: record the full database origin + every tag in the history
        # (params), and a few identifying tag pills. The original DB field name is
        # kept even if the column was renamed.
        db_tags = {t: self._tag_value(resampled, t) for t in TAGS
                   if self._tag_value(resampled, t)}
        units = db_tags.get("units", "")
        data_version = db_tags.get("data_version", "")
        history = {
            "source": "InfluxDB",
            "bucket": self._payload.get("bucket") if self._payload else "",
            "measurement": self._payload.get("measurement") if self._payload else "",
            "db_field": field,
            "resample_freq": self._freq.currentText().strip(),
            "resample_agg": self._agg.currentText(),
            **db_tags,
        }
        # A few meaningful pills; the complete tag set lives in the history above
        # (adding all ~15 DB tags as pills would clutter the variable list).
        pills = [t for t in ["from-database", "meteo-screened", units, data_version] if t]
        out.attrs[ATTRS_KEY] = {
            name: provenance_attr(
                origin=DERIVED, parent=None,
                operation="Imported from InfluxDB & screened",
                params=history, tags=pills),
        }
        self.featuresCreated.emit(out)
        msg = f"Added {name} ({n_total} records) to the dataset"
        if name != field:
            msg = f"Added {name} (renamed from existing {field}, {n_total} records)"
        if overlap is not None and overlap < n_total:
            msg += (f"; {overlap} align with the dataset's timestamps, "
                    f"the rest fall outside its range (NaN)")
        self._status.setText(msg + ".")
        self._add_btn.setEnabled(False)

    def _unique_name(self, field: str) -> str:
        """A column name not already in the working dataset (numeric suffix)."""
        if field not in self._dataset_columns:
            return field
        i = 1
        while f"{field}_{i}" in self._dataset_columns:
            i += 1
        return f"{field}_{i}"
