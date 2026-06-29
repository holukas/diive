"""
GUI.TABS.DATABASE: DATABASE CONNECTION
======================================

Point diive at a database and test the connection. The single app-wide backend
built here (a :class:`~diive.core.io.db.base.DatabaseBackend`) is stored in
``diive.gui.db.manager`` and reused by the Database explorer (and future
download / upload tabs).

InfluxDB is the only backend today: it is configured by a ``dbc-influxdb`` config
directory whose ``<dir>_secret`` sibling carries the url/org/token. This tab
holds no secrets — it only remembers the directory *path* and shows the resolved
url/org after a successful connect.

Strict GUI<->library separation: all database work lives in the library
(``diive.core.io.db``) and ``dbc-influxdb``; this tab only collects the path,
fires the connection test on a worker thread, and shows the result. Without the
optional ``db`` extra it shows install instructions instead of failing.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diive.gui import db
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import WorkerRunner

#: Width cap for the form column so fields/buttons stay readable instead of
#: stretching across a wide window (matches ProjectSettingsTab).
_COL_WIDTH = 480


class DatabaseConnectionTab(DiiveTab):
    """Collect the InfluxDB config-directory path and test the connection,
    storing the live backend in ``db.manager`` for the explorer / I/O tabs."""

    title = "Database connection"

    def build(self) -> QWidget:
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root_lay.addLayout(build_titlebar(self.title))

        body = QWidget()
        outer = QHBoxLayout(body)
        outer.setContentsMargins(24, 24, 24, 24)

        column = QWidget()
        column.setMaximumWidth(_COL_WIDTH)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)

        if not db.influxdb_available():  # optional 'db' extra not installed
            msg = QLabel(db.INSTALL_HINT)
            msg.setWordWrap(True)
            col.addWidget(msg)
            col.addStretch(1)
            outer.addWidget(column)
            outer.addStretch(1)
            root_lay.addWidget(body, stretch=1)
            return root

        intro = QLabel(
            "Connect diive to an InfluxDB through a dbc-influxdb config "
            "directory. Credentials (url, org, token) are read from the "
            "'<dir>_secret' sibling folder — diive only remembers the directory "
            "path, never the token. Use 'Test connection' to verify the server "
            "is reachable; the connection is then reused for explore / "
            "download / upload.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        col.addWidget(intro)

        box = QGroupBox("Connection")
        form = QFormLayout(box)

        self.path = QLineEdit(db.manager.dirconf)
        self.path.setPlaceholderText(r"e.g. F:\...\configs")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.addWidget(self.path, stretch=1)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        form.addRow("Config directory", path_widget)

        # Resolved after a successful connect (display-only; never the token).
        backend = db.manager.backend
        self.url_lbl = QLabel(getattr(backend, "url", "") or "-")
        self.org_lbl = QLabel(getattr(backend, "org", "") or "-")
        form.addRow("URL", self.url_lbl)
        form.addRow("Org", self.org_lbl)
        col.addWidget(box)

        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self._test)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addWidget(self.test_btn)
        btn_row.addStretch(1)
        col.addLayout(btn_row)

        self.status = QLabel(
            "Connected." if db.manager.connected else "Not connected.")
        self.status.setWordWrap(True)
        col.addWidget(self.status)
        col.addStretch(1)

        outer.addWidget(column)
        outer.addStretch(1)
        root_lay.addWidget(body, stretch=1)

        # The connect does network I/O — run it off the GUI thread.
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_ok)
        self._runner.failed.connect(self._on_fail)
        return root

    def _browse(self) -> None:
        start = db.manager.dirconf or ""
        chosen = QFileDialog.getExistingDirectory(
            self.widget(), "Select dbc-influxdb config directory", start)
        if chosen:
            self.path.setText(chosen)

    def _test(self) -> None:
        path = self.path.text().strip()
        if not path:
            self.status.setText("Enter a config directory first.")
            return
        db.manager.set_dirconf(path)
        self.status.setText("Connecting...")
        self.test_btn.setEnabled(False)
        self._runner.run(db.manager.connect)

    def _on_ok(self, backend) -> None:
        self.url_lbl.setText(getattr(backend, "url", "") or "-")
        self.org_lbl.setText(getattr(backend, "org", "") or "-")
        self.status.setText("Connected. Ready to explore / download / upload.")
        self.test_btn.setEnabled(True)
        db.manager.mark_connected()  # fires changed -> header pill + explorer

    def _on_fail(self, err: str) -> None:
        db.manager.disconnect()
        self.status.setText(f"Connection failed: {err}")
        self.test_btn.setEnabled(True)
