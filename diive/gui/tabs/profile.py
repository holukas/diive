"""
GUI.TABS.PROFILE: DATA PROFILING
================================

A whole-dataset profiling tab — the "what did I just load?" overview. The top
strip shows dataset-level facts (rows, variables, overall missing %, duplicate
timestamps/rows, inferred frequency, time span, memory); below it a sortable
table profiles every variable at once: dtype, valid count, missing count/%,
number of gaps, unique values, zeros, whether it's constant, and the numeric
summaries (mean/SD/min/median/max).

All profiling is the library's `dv.analysis.profile_dataframe` /
`dv.analysis.dataframe_overview`; this tab only arranges the results into
widgets and adds the missing-% colour tint (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.drivers import _NumItem
from diive.gui.tabs.overview import _StatCard, _fmt
from diive.gui.widgets.tab_chrome import build_titlebar

#: Table columns: (header, profile-key, numeric?, align-right?).
_COLUMNS = [
    ("Variable", "VARIABLE", False, False),
    ("Type", "DTYPE", False, False),
    ("Count", "COUNT", True, True),
    ("Missing", "MISSING", True, True),
    ("Missing %", "MISSING_PERC", True, True),
    ("Gaps", "N_GAPS", True, True),
    ("Unique", "N_UNIQUE", True, True),
    ("Zeros", "N_ZEROS", True, True),
    ("Const", "CONSTANT", False, True),
    ("Mean", "MEAN", True, True),
    ("SD", "SD", True, True),
    ("Min", "MIN", True, True),
    ("Median", "MEDIAN", True, True),
    ("Max", "MAX", True, True),
]


def _human_bytes(n: int) -> str:
    """Format a byte count compactly (KB/MB/GB, binary)."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class _ProfileTable(QTableWidget):
    """Profiling table that always fills its width without over-spreading.

    The old layout stretched a single column (Variable), which ballooned on a
    wide window while the numbers clustered to the right — reading as stretched.
    Here every column is sized snugly to its content; the leftover width is then
    handed mostly to the flexible text column (Variable) and only a little to the
    numeric columns, so the table fills the viewport with the numbers staying
    compact and no column dominating. When the content is wider than the viewport
    everything scales down to fit (no horizontal scroll). Recomputed on resize."""

    _COL_PADDING = 14            # snug breathing room on each column's content
    _TEXT_COL = 0               # the flexible (Variable) column
    _TEXT_SHARE = 0.55          # fraction of leftover width given to the text column

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.distribute_columns()

    def distribute_columns(self) -> None:
        n = self.columnCount()
        if n == 0 or self.rowCount() == 0:
            return
        hdr = self.horizontalHeader()
        content = [max(self.sizeHintForColumn(c), hdr.sectionSizeHint(c))
                   + self._COL_PADDING for c in range(n)]
        base = sum(content)
        avail = self.viewport().width()
        if base >= avail:                       # too wide: scale down to fit
            scale = avail / base
            widths = [w * scale for w in content]
        else:                                   # fill: text col takes the lion's share
            extra = avail - base
            widths = list(content)
            widths[self._TEXT_COL] += extra * self._TEXT_SHARE
            rest = extra * (1.0 - self._TEXT_SHARE) / max(1, n - 1)
            for c in range(n):
                if c != self._TEXT_COL:
                    widths[c] += rest
        for c, w in enumerate(widths):
            self.setColumnWidth(c, int(round(w)))


class ProfileTab(DiiveTab):
    """Per-variable profiling table for the whole loaded dataset."""

    title = "Data profile"

    def build(self) -> QWidget:
        self._df = None
        self._profile = None   # DataFrame from profile_dataframe

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addLayout(build_titlebar(self.title))  # shared tab header (no codegen)
        outer.addWidget(self._build_stats_strip())
        outer.addWidget(self._build_filter_bar())
        outer.addWidget(self._build_table(), stretch=1)
        return root

    # --- sub-widgets ---------------------------------------------------
    def _build_stats_strip(self) -> QWidget:
        strip = QScrollArea()
        strip.setWidgetResizable(True)
        strip.setFixedHeight(92)
        strip.setFrameShape(QFrame.Shape.NoFrame)
        strip.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_bg = theme.manager.tokens["LIST_BG"]
        border = theme.manager.tokens["BORDER"]
        strip.setStyleSheet(
            f"QScrollArea {{ background: {list_bg}; border-bottom: 1px solid {border}; }}")
        host = QWidget()
        host.setStyleSheet(f"background: {list_bg};")
        self.stats_layout = QHBoxLayout(host)
        self.stats_layout.setContentsMargins(10, 8, 10, 8)
        self.stats_layout.setSpacing(8)
        self.stats_layout.addStretch(1)
        strip.setWidget(host)
        return strip

    def _build_filter_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addWidget(QLabel("Filter variables"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("substring match (case-insensitive)…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self.filter, stretch=1)
        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet("color: #90A4AE;")
        lay.addWidget(self.count_lbl)
        return bar

    def _build_table(self) -> QWidget:
        self.table = _ProfileTable()
        self.table.setColumnCount(len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        # Proportional fill (see _ProfileTable): Interactive sections whose widths
        # this tab recomputes, instead of a single ballooning stretch column.
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        hdr.setHighlightSections(False)
        # Beauty: zebra rows, a clean header band, comfortable rows, and soft
        # row separators instead of a hard grid. (Style only the widget/header,
        # never ``::item`` — that would disable the per-cell missing-% tint and
        # the red 'constant' foreground.)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        t = theme.manager.tokens
        self.table.setStyleSheet(
            f"QTableWidget {{ background: {t['CANVAS']};"
            f" alternate-background-color: {t['LIST_BG']};"
            f" border: none; outline: none; }}"
            f"QHeaderView::section {{ background: {t['CANVAS']}; color: {t['INK']};"
            f" padding: 7px 10px; border: none;"
            f" border-bottom: 1px solid {t['ACCENT']}; font-weight: 600; }}")
        return self.table

    # --- data flow -----------------------------------------------------
    def save_state(self) -> dict:
        return {"filter": self.filter.text()}

    def restore_state(self, state: dict) -> None:
        self.filter.setText(state.get("filter", "") or "")

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        if df is None or df.shape[1] == 0:
            self._profile = None
            self.table.setRowCount(0)
            self._fill_stats(None)
            return
        # Profiling is the library's; the tab only reads the results back.
        self._profile = dv.analysis.profile_dataframe(df)
        self._fill_stats(dv.analysis.dataframe_overview(df))
        self._fill_table()
        self._apply_filter()

    def _fill_stats(self, ov: dict | None) -> None:
        while self.stats_layout.count() > 1:  # keep the trailing stretch
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if ov is None:
            return
        span = "—"
        if ov["start"] is not None and ov["end"] is not None:
            span = f"{ov['start']:%Y-%m-%d} → {ov['end']:%Y-%m-%d}"
        cards = [
            ("Records", _fmt(ov["n_rows"])),
            ("Variables", _fmt(ov["n_cols"])),
            ("Missing cells", f"{ov['missing_perc']:.1f}%"),
            ("Dup. timestamps", _fmt(ov["duplicate_timestamps"])),
            ("Dup. rows", _fmt(ov["duplicate_rows"])),
            ("Frequency", ov["freq"] or "—"),
            ("Time span", span),
            ("Memory", _human_bytes(ov["memory_bytes"])),
        ]
        for i, (name, value) in enumerate(cards):
            self.stats_layout.insertWidget(i, _StatCard(name, value))

    def _fill_table(self) -> None:
        prof = self._profile
        self.table.setSortingEnabled(False)
        try:
            self.table.setRowCount(0 if prof is None else len(prof))
            if prof is None:
                return
            for r, (_, row) in enumerate(prof.iterrows()):
                for c, (_, key, numeric, right) in enumerate(_COLUMNS):
                    item = self._make_cell(key, row[key], numeric)
                    if right:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                              | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, c, item)
        finally:
            self.table.setSortingEnabled(True)
        self.table.distribute_columns()  # fill width once content is in

    def _make_cell(self, key: str, value, numeric: bool) -> QTableWidgetItem:
        if key == "CONSTANT":
            # Flag constant columns — a common "this variable is useless" signal.
            item = QTableWidgetItem("yes" if value else "")
            if value:
                item.setForeground(QColor("#E53935"))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return item
        if numeric:
            v = float(value)
            if pd.isna(v):
                return _NumItem(float("-inf"), "—")
            is_int = key in ("COUNT", "MISSING", "N_GAPS", "N_UNIQUE", "N_ZEROS")
            text = f"{int(v):,}" if is_int else (
                f"{v:.2f}%" if key == "MISSING_PERC" else _fmt(v))
            item = _NumItem(v, text)
            if key == "MISSING_PERC" and v > 0:
                # Red tint scaling with missingness, so gappy variables pop.
                tint = QColor("#E53935")
                tint.setAlphaF(0.08 + 0.42 * min(v, 100.0) / 100.0)
                item.setBackground(tint)
            return item
        return QTableWidgetItem(str(value))

    def _apply_filter(self) -> None:
        needle = self.filter.text().strip().lower()
        shown = 0
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            name = item.text().lower() if item else ""
            hide = bool(needle) and needle not in name
            self.table.setRowHidden(r, hide)
            if not hide:
                shown += 1
        total = self.table.rowCount()
        self.count_lbl.setText(
            f"{shown} of {total} variables" if needle else f"{total} variables")
