"""
GUI.TABS.DATABASE_EXPLORER: BROWSE THE CONNECTED INFLUXDB
========================================================

A read-only browser for the connected InfluxDB's schema. Drill down: pick a
**bucket** -> pick a **data version** -> see the **measurements** stored under
that version -> pick one for its **fields** -> pick a field for a **field
overview** (all of its tags, incl. units, plus the first / last record
timestamps). Mirrors InfluxDB's bucket -> measurement -> field hierarchy, with
the ``data_version`` tag diive stores alongside each variable folded in as a
filter step.

Needs an active connection (see the Database connection tab); the backend lives
in ``diive.gui.db.manager``. All schema queries are the backend's job
(``diive.core.io.db``); this tab only renders lists and
runs each query on a worker thread so the UI never blocks. It refreshes whenever
the connection changes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import time

import pandas as pd
from PySide6.QtCore import QDateTime, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.gui import db, site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner

#: Datetime format used by the start/end pickers and passed to the download.
_DT_FORMAT = "yyyy-MM-dd HH:mm:ss"
#: Target chunk length for progressive downloads; the plot grows per chunk.
_CHUNK_DAYS = 14
_MAX_CHUNKS = 14


class _Signals(QObject):
    """Qt signals (DiiveTab is a plain ABC, not a QObject).

    ``chunk`` carries (accumulated_series, done_chunks, total_chunks) from the
    download worker thread; a queued connection marshals it to the GUI thread so
    the plot + progress bar can update as each chunk arrives.
    """
    chunk = Signal(object, int, int)


def _column(title: str) -> tuple[QWidget, QLabel, QListWidget]:
    """A titled list column: bold header (with a count slot) over a list."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    header = QLabel(f"<b>{title}</b>")
    lst = QListWidget()
    lay.addWidget(header)
    lay.addWidget(lst, stretch=1)
    return box, header, lst


class DatabaseExplorerTab(DiiveTab):
    """Browse buckets -> (data versions) -> measurements -> fields -> overview."""

    title = "Database explorer"

    def build(self) -> QWidget:
        self._bucket: str | None = None
        self._data_version: str | None = None
        self._measurement: str | None = None
        self._field: str | None = None
        self._dl_t0: float = 0.0  # download start time (set on each download)
        # The selected field's record span, kept in UTC (the DB's native time);
        # the pickers show this shifted into the chosen UTC offset.
        self._first_utc: QDateTime | None = None
        self._last_utc: QDateTime | None = None
        # The working dataset's (MIDDLE) index, for "Match dataset time range".
        self._dataset_index = None
        # Cache of the last detailed download so Download & plot and Send to
        # screening don't re-download the same request.
        self._cached_detailed: dict | None = None
        self._cache_key: tuple | None = None
        self._pending_key: tuple | None = None

        # One runner per stage so a slow query in one column can't block another.
        self._buckets_runner = WorkerRunner()
        self._buckets_runner.done.connect(self._on_buckets)
        self._buckets_runner.failed.connect(lambda e: self._fail("buckets", e))
        self._versions_runner = WorkerRunner()
        self._versions_runner.done.connect(self._on_versions)
        self._versions_runner.failed.connect(lambda e: self._fail("data versions", e))
        self._meas_runner = WorkerRunner()
        self._meas_runner.done.connect(self._on_measurements)
        self._meas_runner.failed.connect(lambda e: self._fail("measurements", e))
        self._fields_runner = WorkerRunner()
        self._fields_runner.done.connect(self._on_fields)
        self._fields_runner.failed.connect(lambda e: self._fail("fields", e))
        self._overview_runner = WorkerRunner()
        self._overview_runner.done.connect(self._on_overview)
        self._overview_runner.failed.connect(lambda e: self._fail("field overview", e))
        self._download_runner = WorkerRunner()
        self._download_runner.done.connect(self._on_downloaded)
        self._download_runner.failed.connect(self._on_download_failed)
        self._detailed_runner = WorkerRunner()
        self._detailed_runner.done.connect(self._on_detailed_downloaded)
        self._detailed_runner.failed.connect(self._on_detailed_failed)
        self._sig = _Signals()
        self._sig.chunk.connect(self._on_chunk)

        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._load_buckets)
        root_lay.addLayout(build_titlebar(self.title, self._refresh_btn))

        self._status = QLabel()
        self._status.setStyleSheet("color: #6B7780;")
        self._status.setContentsMargins(12, 0, 12, 4)
        root_lay.addWidget(self._status)

        body = QWidget()
        cols = QHBoxLayout(body)
        cols.setContentsMargins(12, 8, 12, 12)
        cols.setSpacing(12)
        bucket_col, self._bucket_hdr, self._bucket_list = _column("Buckets")
        ver_col, self._ver_hdr, self._ver_list = _column("Data version")
        meas_col, self._meas_hdr, self._meas_list = _column("Measurements")
        field_col, self._field_hdr, self._field_list = _column("Fields")
        ov_col, self._ov_hdr, self._ov_list = _column("Field overview")
        # The overview is read-only (key: value lines), not a pick list.
        self._ov_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._ov_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bucket_list.itemSelectionChanged.connect(self._on_bucket_pick)
        self._ver_list.itemSelectionChanged.connect(self._on_version_pick)
        self._meas_list.itemSelectionChanged.connect(self._on_measurement_pick)
        self._field_list.itemSelectionChanged.connect(self._on_field_pick)
        for col in (bucket_col, ver_col, meas_col, field_col, ov_col):
            cols.addWidget(col, stretch=1)
        cols.addWidget(self._build_download_col(), stretch=1)

        # Drill-down columns on top, the downloaded time-series plot below;
        # a splitter lets the user trade height between them.
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(body)
        self.canvas = MplCanvas()
        self.canvas.setVisible(False)  # shown after the first download
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root_lay.addWidget(split, stretch=1)

        # Refresh whenever the connection changes (connect / disconnect).
        db.manager.changed.connect(self._on_connection_changed)
        self._on_connection_changed()
        return root

    def _build_download_col(self) -> QWidget:
        """The 6th column: start / end pickers, a Download button + progress bar.

        Disabled until a field with records is selected; the pickers are then
        seeded to the field's last month of data."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(QLabel("<b>Download &amp; plot</b>"))

        self._dl_group = QGroupBox("Time range")
        form = QFormLayout(self._dl_group)
        # UTC offset of the start/end below AND of the returned timestamps. The
        # DB stores everything in UTC; the download converts to this offset. It
        # MUST match the working dataset's timezone or the merged data are shifted,
        # so it defaults to the project's configured timezone (else CET = +1).
        self._utc_offset = QDoubleSpinBox()
        self._utc_offset.setRange(-12.0, 14.0)
        self._utc_offset.setSingleStep(1.0)
        self._utc_offset.setDecimals(1)
        self._utc_offset.setValue(
            float(site.manager.utc_offset) if site.manager.configured else 1.0)
        self._utc_offset.setToolTip(
            "Timezone (offset to UTC, hours) the start/end are given in and the "
            "downloaded data are returned in. The database stores all data in UTC; "
            "set this to your dataset's timezone (default = the project timezone, "
            "or 1 = CET). A wrong offset silently shifts the merged data.")
        self._utc_offset.valueChanged.connect(self._on_offset_changed)
        self._start_edit = QDateTimeEdit()
        self._start_edit.setDisplayFormat(_DT_FORMAT)
        self._start_edit.setCalendarPopup(True)
        self._end_edit = QDateTimeEdit()
        self._end_edit.setDisplayFormat(_DT_FORMAT)
        self._end_edit.setCalendarPopup(True)
        form.addRow("UTC offset (h)", self._utc_offset)
        form.addRow("Start", self._start_edit)
        form.addRow("End", self._end_edit)
        self._match_btn = QPushButton("Match dataset time range")
        self._match_btn.setToolTip(
            "Set the high-res END range so that, after screening and resampling to "
            "the dataset's resolution, the TIMESTAMP_MIDDLE output covers the working "
            "dataset's time range (shifted by half a period for the END↔MIDDLE "
            "convention). Assumes the dataset is in the selected UTC offset.")
        self._match_btn.clicked.connect(self._match_dataset_range)
        form.addRow("", self._match_btn)
        lay.addWidget(self._dl_group)

        # Visible (not just tooltip) timezone note — the database stores UTC, so
        # the offset must match the dataset's timezone or the merge is shifted.
        self._tz_note = QLabel()
        self._tz_note.setWordWrap(True)
        self._tz_note.setStyleSheet("color: #6B7780; font-size: 11px;")
        self._utc_offset.valueChanged.connect(self._update_tz_note)
        lay.addWidget(self._tz_note)
        self._update_tz_note()

        self._download_btn = QPushButton("Download && plot")
        self._download_btn.setToolTip(
            "Download the selected field for the chosen range and plot it. "
            "High-resolution data can take a while.")
        theme.set_button_role(self._download_btn, "confirm")
        self._download_btn.clicked.connect(self._on_download)
        lay.addWidget(self._download_btn)

        self._download_bar = ProgressBar()
        lay.addWidget(self._download_bar)

        self._screen_btn = QPushButton("Send to Meteo screening  →")
        self._screen_btn.setToolTip(
            "Download this field (with its database tags) for the chosen range "
            "and hand it to the Meteo screening tab for screening + resampling.")
        self._screen_btn.clicked.connect(self._on_send_to_screening)
        lay.addWidget(self._screen_btn)
        lay.addStretch(1)

        self._dl_group.setEnabled(False)
        self._download_btn.setEnabled(False)
        self._screen_btn.setEnabled(False)
        self._match_btn.setEnabled(False)
        return box

    # --- connection state ---

    def _on_connection_changed(self) -> None:
        connected = db.manager.connected and db.manager.backend is not None
        self._refresh_btn.setEnabled(connected)
        if not connected:
            self._clear_from("bucket")
            self._status.setText("Not connected. Use the Database connection tab first.")
            return
        self._load_buckets()

    def _clear_from(self, level: str) -> None:
        """Clear the given column and everything to its right, resetting headers."""
        order = ["bucket", "version", "measurement", "field", "overview"]
        lists = {
            "bucket": (self._bucket_list, self._bucket_hdr, "Buckets"),
            "version": (self._ver_list, self._ver_hdr, "Data version"),
            "measurement": (self._meas_list, self._meas_hdr, "Measurements"),
            "field": (self._field_list, self._field_hdr, "Fields"),
            "overview": (self._ov_list, self._ov_hdr, "Field overview"),
        }
        idx = order.index(level)
        for name in order[idx:]:
            lst, hdr, title = lists[name]
            lst.clear()
            hdr.setText(f"<b>{title}</b>")
        if idx <= 0:
            self._bucket = None
        if idx <= 1:
            self._data_version = None
        if idx <= 2:
            self._measurement = None
        if idx <= 3:
            self._field = None
        # The overview always clears with (or before) anything to its right, so
        # the download controls go inactive until a field is re-selected.
        self._set_download_enabled(False)
        self._invalidate_cache()  # selection changed -> stale download

    def _set_download_enabled(self, enabled: bool) -> None:
        self._dl_group.setEnabled(enabled)
        busy = self._download_runner.is_running or self._detailed_runner.is_running
        self._download_btn.setEnabled(enabled and not busy)
        self._screen_btn.setEnabled(enabled and not busy)
        has_dataset = self._dataset_index is not None and len(self._dataset_index) > 0
        self._match_btn.setEnabled(enabled and has_dataset)

    def on_data_loaded(self, df, created=None) -> None:
        # Keep the working dataset's index so "Match dataset time range" can map
        # the (MIDDLE) dataset span onto a high-res (END) download range.
        self._dataset_index = df.index if df is not None else None

    def _match_dataset_range(self) -> None:
        """Set the high-res END pickers to cover the working dataset's MIDDLE range.

        The dataset's averaging periods span absolute time [M0 - half, M1 + half]
        (half = dataset resolution / 2). The high-res END download must cover that
        span, so start = M0 - half and stop = M1 + half + resolution (a margin so
        the last period is fully included; extra resampled bins fall outside the
        dataset and are dropped on merge).
        """
        idx = self._dataset_index
        if idx is None or len(idx) < 2:
            self._status.setText("No dataset loaded to match.")
            return
        res = idx.to_series().diff().median()
        if res is None or pd.isna(res):
            res = pd.Timedelta("30min")
        half = res / 2
        m0, m1 = idx.min(), idx.max()
        start = (m0 - half).strftime("%Y-%m-%d %H:%M:%S")
        stop = (m1 + half + res).strftime("%Y-%m-%d %H:%M:%S")
        self._start_edit.setDateTime(QDateTime.fromString(start, _DT_FORMAT))
        self._end_edit.setDateTime(QDateTime.fromString(stop, _DT_FORMAT))

        # Note if the field's available range clamped the request.
        clamped = (self._start_edit.dateTime().toString(_DT_FORMAT) != start
                   or self._end_edit.dateTime().toString(_DT_FORMAT) != stop)
        note = (" (clamped to the field's available range)" if clamped else "")
        self._status.setText(
            f"Range set to cover the dataset ({m0:%Y-%m-%d %H:%M} to {m1:%Y-%m-%d %H:%M}, "
            f"MIDDLE) as TIMESTAMP_END +/- half of {res}{note}.")

    # --- buckets ---

    def _load_buckets(self) -> None:
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(f"Loading buckets from {backend.describe()}...")
        self._clear_from("bucket")
        self._buckets_runner.run(backend.list_buckets)

    def _on_buckets(self, buckets: list) -> None:
        self._bucket_hdr.setText(f"<b>Buckets</b> ({len(buckets)})")
        self._bucket_list.addItems(buckets)
        self._status.setText(
            f"{len(buckets)} buckets. Select one to see its data versions.")

    def _on_bucket_pick(self) -> None:
        items = self._bucket_list.selectedItems()
        if not items:
            return
        self._bucket = items[0].text()
        self._clear_from("version")
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(f"Loading data versions in {self._bucket}...")
        self._versions_runner.run(backend.list_data_versions, self._bucket)

    # --- data versions (filter the rest of the chain) ---

    def _on_versions(self, versions: list) -> None:
        self._ver_hdr.setText(f"<b>Data version</b> ({len(versions)})")
        self._ver_list.addItems(versions)
        self._status.setText(
            f"{len(versions)} data versions in {self._bucket}. "
            f"Select one to see its measurements.")

    def _on_version_pick(self) -> None:
        items = self._ver_list.selectedItems()
        if not items or self._bucket is None:
            return
        self._data_version = items[0].text()
        self._clear_from("measurement")
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(
            f"Loading measurements in {self._bucket} ({self._data_version})...")
        self._meas_runner.run(
            backend.list_measurements, self._bucket, self._data_version)

    # --- measurements ---

    def _on_measurements(self, measurements: list) -> None:
        self._meas_hdr.setText(f"<b>Measurements</b> ({len(measurements)})")
        self._meas_list.addItems(measurements)
        self._status.setText(
            f"{len(measurements)} measurements in {self._bucket} "
            f"({self._data_version}). Select one to see its fields.")

    def _on_measurement_pick(self) -> None:
        items = self._meas_list.selectedItems()
        if not items or self._bucket is None:
            return
        self._measurement = items[0].text()
        self._clear_from("field")
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(f"Loading fields in {self._measurement}...")
        self._fields_runner.run(
            backend.list_fields, self._bucket, self._measurement, self._data_version)

    # --- fields ---

    def _on_fields(self, fields: list) -> None:
        self._field_hdr.setText(f"<b>Fields</b> ({len(fields)})")
        self._field_list.addItems(fields)
        self._status.setText(
            f"{len(fields)} fields in {self._measurement}. Select one for its overview.")

    def _on_field_pick(self) -> None:
        items = self._field_list.selectedItems()
        if not items or self._bucket is None or self._measurement is None:
            return
        self._field = items[0].text()
        self._ov_list.clear()
        self._ov_hdr.setText("<b>Field overview</b>")
        self._set_download_enabled(False)  # re-enabled once the overview loads
        self._invalidate_cache()  # new field -> previous download no longer applies
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(f"Loading overview for {self._field}...")
        self._overview_runner.run(
            backend.field_overview, self._bucket, self._measurement, self._field,
            self._data_version)

    # --- field overview (all tags + first/last record) ---

    def _on_overview(self, overview: dict) -> None:
        """Render the selected field's tags (incl. units) and record span."""
        self._ov_list.clear()
        tags = overview.get("tags", {})

        lines = [f"field: {overview.get('field', '')}"]
        for tag, values in tags.items():
            shown = ", ".join(v for v in values if v) or "(empty)"
            lines.append(f"{tag}: {shown}")
        lines.append(f"first record: {overview.get('first') or '-'} (UTC)")
        lines.append(f"last record: {overview.get('last') or '-'} (UTC)")

        n_series = overview.get("n_series", 0)
        if n_series == 0:
            self._ov_hdr.setText("<b>Field overview</b>")
            self._ov_list.addItem("(no records found)")
            self._status.setText(f"No records found for {self._field}.")
            return

        self._ov_hdr.setText(f"<b>Field overview</b> ({len(tags)} tags)")
        self._ov_list.addItems(lines)
        # n_series > 1 means some tag(s) changed over the field's lifetime.
        extra = f" ({n_series} tag-sets over time)" if n_series > 1 else ""
        self._status.setText(
            f"Overview for {self._field}: {overview.get('first') or '-'} to "
            f"{overview.get('last') or '-'} (UTC){extra}.")

        self._seed_download_range(overview.get("first"), overview.get("last"))

    # --- download & plot ---

    def _seed_download_range(self, first: str | None, last: str | None) -> None:
        """Store the field's (UTC) record span and seed the pickers to its last
        month of data in the chosen offset; enable the download controls."""
        first_dt = QDateTime.fromString(first or "", _DT_FORMAT)
        last_dt = QDateTime.fromString(last or "", _DT_FORMAT)
        if not first_dt.isValid() or not last_dt.isValid():
            self._first_utc = self._last_utc = None
            self._set_download_enabled(False)
            return
        self._first_utc = first_dt
        self._last_utc = last_dt
        self._apply_offset_to_pickers()
        self._set_download_enabled(True)

    def _offset_secs(self) -> int:
        """Selected UTC offset in seconds (rounded to the minute)."""
        return int(round(self._utc_offset.value() * 3600))

    def _offset_label(self) -> str:
        """The offset as an ISO-style label, e.g. ``'UTC+01:00'``."""
        total = self._offset_secs()
        sign = "+" if total >= 0 else "-"
        hours, mins = divmod(abs(total) // 60, 60)
        return f"UTC{sign}{hours:02d}:{mins:02d}"

    def _update_tz_note(self, *_) -> None:
        """Keep the visible timezone note in sync with the chosen offset."""
        note = (f"All times here are <b>{self._offset_label()}</b>. The database "
                f"stores UTC; this offset must match your dataset's timezone.")
        if site.manager.configured:
            same = float(self._utc_offset.value()) == float(site.manager.utc_offset)
            note += (" Default is the project timezone"
                     + (" (matches)." if same else " — currently differs!"))
        else:
            note += " Default 1 = CET."
        self._tz_note.setText(note)

    def _apply_offset_to_pickers(self) -> None:
        """Shift the (UTC) record span into the chosen offset and default the
        selection to the last month of data."""
        if self._first_utc is None or self._last_utc is None:
            return
        secs = self._offset_secs()
        first_local = self._first_utc.addSecs(secs)
        last_local = self._last_utc.addSecs(secs)
        for edit in (self._start_edit, self._end_edit):
            edit.setDateTimeRange(first_local, last_local)
        # Default selection: the last month of available data (clamped to first).
        self._start_edit.setDateTime(max(last_local.addMonths(-1), first_local))
        self._end_edit.setDateTime(last_local)

    def _on_offset_changed(self) -> None:
        """Re-seed the pickers when the timezone changes (the displayed range and
        the last-month default are both offset-dependent)."""
        self._apply_offset_to_pickers()

    def _request_key(self, start: str, stop: str) -> tuple:
        """Identifies a download so it can be reused (selection + range + offset)."""
        return (self._bucket, self._measurement, self._field, self._data_version,
                start, stop, self._utc_offset.value())

    def _invalidate_cache(self) -> None:
        self._cached_detailed = None
        self._cache_key = None

    def _field_series(self, data_detailed: dict | None):
        """The field's data column from a data_detailed dict (or None)."""
        if not data_detailed or self._field not in data_detailed:
            return None
        frame = data_detailed[self._field]
        return frame[self._field] if self._field in frame.columns else None

    def _begin_download(self, runner, fn, start: str, stop: str) -> None:
        """Shared launch for both download paths (chunked, with live plot)."""
        n = self._n_chunks(start, stop)
        self._pending_key = self._request_key(start, stop)
        self._download_btn.setEnabled(False)
        self._screen_btn.setEnabled(False)
        self._download_bar.set_progress(0, f"Downloading {self._field}… 0/{n}")
        self._dl_t0 = time.monotonic()
        cb = lambda acc, done, total: self._sig.chunk.emit(acc, done, total)
        runner.run(fn, self._bucket, self._measurement, self._field,
                   start, stop, self._data_version, self._utc_offset.value(), n, cb)

    def _on_download(self) -> None:
        backend = db.manager.backend
        if backend is None or self._download_runner.is_running:
            return
        if self._field is None or self._measurement is None or self._bucket is None:
            return
        start = self._start_edit.dateTime().toString(_DT_FORMAT)
        stop = self._end_edit.dateTime().toString(_DT_FORMAT)
        if self._start_edit.dateTime() >= self._end_edit.dateTime():
            self._status.setText("Start must be before end.")
            return
        # Reuse an already-downloaded identical request instead of fetching again.
        if self._cached_detailed is not None and self._cache_key == self._request_key(start, stop):
            series = self._field_series(self._cached_detailed)
            if series is not None and not series.empty:
                self._plot_series(series)
                self._status.setText(f"Plotted {self._field} from already-downloaded data.")
                return
        self._status.setText(
            f"Downloading {self._field} from {start} to {stop} ({self._offset_label()})…")
        self._begin_download(self._download_runner, backend.download_detailed_chunked,
                             start, stop)

    def _on_downloaded(self, data_detailed: dict) -> None:
        self._download_bar.finish()
        self._set_download_enabled(True)
        self._cached_detailed = data_detailed
        self._cache_key = self._pending_key
        took = self._fmt_duration(time.monotonic() - self._dl_t0)
        series = self._field_series(data_detailed)
        if series is None or series.empty:
            self._status.setText(
                f"No data returned for {self._field} in that range (took {took}).")
            return
        self._plot_series(series)
        self._status.setText(
            f"Downloaded {len(series)} records for {self._field} in {took}.")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Human-readable elapsed time, e.g. '0.8 s' or '1 min 5 s'."""
        if seconds < 60:
            return f"{seconds:.1f} s"
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes} min {secs} s"

    def _on_download_failed(self, err: str) -> None:
        self._download_bar.finish()
        self._set_download_enabled(True)
        took = self._fmt_duration(time.monotonic() - self._dl_t0)
        self._status.setText(f"Download failed after {took}: {err}")

    # --- hand off to the Meteo screening tab ---

    def _on_send_to_screening(self) -> None:
        backend = db.manager.backend
        if backend is None or self._detailed_runner.is_running:
            return
        if self._field is None or self._measurement is None or self._bucket is None:
            return
        start = self._start_edit.dateTime().toString(_DT_FORMAT)
        stop = self._end_edit.dateTime().toString(_DT_FORMAT)
        if self._start_edit.dateTime() >= self._end_edit.dateTime():
            self._status.setText("Start must be before end.")
            return
        # Reuse an already-downloaded identical request instead of fetching again.
        if self._cached_detailed is not None and self._cache_key == self._request_key(start, stop):
            self._hand_off(self._cached_detailed)
            self._status.setText(
                f"Sent {self._field} to Meteo screening (reused downloaded data).")
            return
        self._status.setText(f"Downloading {self._field} for screening…")
        self._begin_download(self._detailed_runner, backend.download_detailed_chunked,
                             start, stop)

    def _hand_off(self, data_detailed: dict) -> None:
        """Send a downloaded field (data_detailed) to the Meteo screening tab."""
        db.manager.request_screening({
            "data_detailed": data_detailed,
            "field": self._field,
            "measurement": self._measurement,
            "bucket": self._bucket,
            "data_version": self._data_version,
            "utc_offset": self._utc_offset.value(),
        })

    @staticmethod
    def _n_chunks(start: str, stop: str) -> int:
        """Number of download chunks for progressive loading (~_CHUNK_DAYS each)."""
        span_days = (pd.Timestamp(stop) - pd.Timestamp(start)) / pd.Timedelta(days=1)
        return max(1, min(_MAX_CHUNKS, round(span_days / _CHUNK_DAYS)))

    def _on_chunk(self, accumulated, done: int, total: int) -> None:
        """A download chunk arrived: advance the bar and grow the plot."""
        permille = int(done / total * 1000) if total else 1000
        self._download_bar.set_progress(permille, f"Downloading {self._field}… {done}/{total}")
        if accumulated is not None and len(accumulated):
            self._plot_series(accumulated)

    def _on_detailed_downloaded(self, data_detailed: dict) -> None:
        self._download_bar.finish()
        self._set_download_enabled(True)
        self._cached_detailed = data_detailed
        self._cache_key = self._pending_key
        took = self._fmt_duration(time.monotonic() - self._dl_t0)
        if not data_detailed or self._field not in data_detailed:
            self._status.setText(f"No data returned for {self._field} in that range (took {took}).")
            return
        series = self._field_series(data_detailed)
        if series is not None and not series.empty:
            self._plot_series(series)
        self._hand_off(data_detailed)
        self._status.setText(
            f"Sent {self._field} to Meteo screening (downloaded in {took}).")

    def _on_detailed_failed(self, err: str) -> None:
        self._download_bar.finish()
        self._set_download_enabled(True)
        self._status.setText(f"Download for screening failed: {err}")

    def _plot_series(self, series) -> None:
        """Draw a simple time series of the downloaded field."""
        units = self._field_units()
        ax = self.canvas.new_axes(1)[0]
        ax.plot(series.index, series.to_numpy(), color="#2196F3", lw=0.8)
        ax.set_title(self._field, fontsize=10)
        ax.set_xlabel(f"Time ({self._offset_label()})")
        ax.set_ylabel(f"{self._field} [{units}]" if units else self._field)
        ax.grid(True, alpha=0.3)
        self.canvas.setVisible(True)
        self.canvas.draw()
        self.canvas.reset_history()

    def _field_units(self) -> str:
        """Units string shown in the overview list (best-effort, for the y-label)."""
        for i in range(self._ov_list.count()):
            text = self._ov_list.item(i).text()
            if text.startswith("units:"):
                return text.split(":", 1)[1].strip()
        return ""

    def _fail(self, stage: str, err: str) -> None:
        self._status.setText(f"Failed to load {stage}: {err}")
