"""
GUI.TABS.DATABASE_EXPLORER: BROWSE THE CONNECTED INFLUXDB
========================================================

A read-only browser for the connected InfluxDB's schema. Drill down: pick a
**bucket** -> pick a **data version** -> see the **measurements** stored under
that version -> pick one for its **fields** -> pick a field for its **units**.
Mirrors InfluxDB's bucket -> measurement -> field hierarchy, with the
``data_version`` and ``units`` tags diive stores alongside each variable folded
in as filter / leaf steps.

Needs an active connection (see the Database connection tab); the backend lives
in ``diive.gui.db.manager``. All schema queries are the backend's job
(``diive.core.io.db`` over ``dbc-influxdb``); this tab only renders lists and
runs each query on a worker thread so the UI never blocks. It refreshes whenever
the connection changes.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.gui import db
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner


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
    """Browse buckets -> (data versions) -> measurements -> fields -> units."""

    title = "Database explorer"

    def build(self) -> QWidget:
        self._bucket: str | None = None
        self._data_version: str | None = None
        self._measurement: str | None = None
        self._field: str | None = None

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
        self._units_runner = WorkerRunner()
        self._units_runner.done.connect(self._on_units)
        self._units_runner.failed.connect(lambda e: self._fail("units", e))

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
        unit_col, self._unit_hdr, self._unit_list = _column("Units")
        self._bucket_list.itemSelectionChanged.connect(self._on_bucket_pick)
        self._ver_list.itemSelectionChanged.connect(self._on_version_pick)
        self._meas_list.itemSelectionChanged.connect(self._on_measurement_pick)
        self._field_list.itemSelectionChanged.connect(self._on_field_pick)
        for col in (bucket_col, ver_col, meas_col, field_col, unit_col):
            cols.addWidget(col, stretch=1)
        root_lay.addWidget(body, stretch=1)

        # Refresh whenever the connection changes (connect / disconnect).
        db.manager.changed.connect(self._on_connection_changed)
        self._on_connection_changed()
        return root

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
        order = ["bucket", "version", "measurement", "field", "units"]
        lists = {
            "bucket": (self._bucket_list, self._bucket_hdr, "Buckets"),
            "version": (self._ver_list, self._ver_hdr, "Data version"),
            "measurement": (self._meas_list, self._meas_hdr, "Measurements"),
            "field": (self._field_list, self._field_hdr, "Fields"),
            "units": (self._unit_list, self._unit_hdr, "Units"),
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
            f"{len(fields)} fields in {self._measurement}. Select one to see its units.")

    def _on_field_pick(self) -> None:
        items = self._field_list.selectedItems()
        if not items or self._bucket is None or self._measurement is None:
            return
        self._field = items[0].text()
        self._unit_list.clear()
        self._unit_hdr.setText("<b>Units</b>")
        backend = db.manager.backend
        if backend is None:
            return
        self._status.setText(f"Loading units for {self._field}...")
        self._units_runner.run(
            backend.list_units, self._bucket, self._measurement, self._field,
            self._data_version)

    # --- units ---

    def _on_units(self, units: list) -> None:
        self._unit_hdr.setText(f"<b>Units</b> ({len(units)})")
        self._unit_list.addItems(units)
        shown = ", ".join(units) if units else "(none recorded)"
        self._status.setText(f"Units for {self._field}: {shown}")

    def _fail(self, stage: str, err: str) -> None:
        self._status.setText(f"Failed to load {stage}: {err}")
