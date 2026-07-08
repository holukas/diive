"""
GUI.TABS.DRIVERS: DRIVER EXPLORER
=================================

"What relates to this variable, and at what lag?" Pick a target on the left and
get every other variable ranked by how strongly it correlates with it (Pearson
or Spearman), optionally scanning a lead/lag window so storage / advection /
phenology relationships surface. Click a ranked driver to see the target-vs-
driver scatter (at that driver's best lag).

All ranking is the library's `dv.analysis.rank_drivers`; the scatter is
`dv.plotting.ScatterXY`. This tab only collects the target/method/lag, lays the
results into widgets, and wires the click-to-scatter interaction — no statistics
of its own (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui.tabs._explorer_base import SingleVariableExplorerTab
from diive.gui.tabs.overview import _fmt
from diive.gui.widgets.mpl_canvas import MplCanvas

_DRIVER_ROLE = Qt.ItemDataRole.UserRole + 1


class _NumItem(QTableWidgetItem):
    """Table cell that sorts by a stored numeric value, not its display text."""

    def __init__(self, value: float, text: str) -> None:
        super().__init__(text)
        self._v = value

    def __lt__(self, other) -> bool:
        try:
            return self._v < other._v
        except AttributeError:
            return super().__lt__(other)


class DriverExplorerTab(SingleVariableExplorerTab):
    """Rank drivers of a target variable; click one to see the scatter."""

    title = "Driver explorer"
    #: A continuous flux makes the ranking informative.
    default_var = "NEE_CUT_REF_f"

    def _init_state(self) -> None:
        self._ranked = None    # DataFrame from rank_drivers
        self._filling = False  # guard table-selection echo during repopulation

    def _build_right(self) -> QWidget:
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_stats_strip())
        rl.addWidget(self._build_controls())

        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.addWidget(self._build_table())
        self.canvas = MplCanvas()
        hsplit.addWidget(self.canvas)
        hsplit.setStretchFactor(0, 0)
        hsplit.setStretchFactor(1, 1)
        hsplit.setSizes([380, 560])
        rl.addWidget(hsplit, stretch=1)
        return right

    # --- sub-widgets ---------------------------------------------------
    def _build_controls(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems(["Pearson", "Spearman"])
        self.method.setToolTip("Pearson = linear; Spearman = rank/monotonic.")
        lay.addWidget(self.method)
        lay.addSpacing(12)
        lay.addWidget(QLabel("Max lag (records)"))
        self.max_lag = QSpinBox()
        self.max_lag.setRange(0, 240)
        self.max_lag.setValue(0)
        self.max_lag.setToolTip(
            "Scan lags ±N records and report each driver's strongest lag "
            "(0 = contemporaneous only). Positive lag = driver leads the target.")
        lay.addWidget(self.max_lag)
        lay.addSpacing(12)
        self.rank_btn = QPushButton("Rank drivers")
        self.rank_btn.clicked.connect(self._recompute)
        lay.addWidget(self.rank_btn)
        lay.addStretch(1)
        return bar

    def _build_table(self) -> QWidget:
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Driver", "r", "Lag", "N"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_driver_selected)
        return self.table

    # --- data flow -----------------------------------------------------
    def save_state(self) -> dict:
        return {"target": self._target, "method": self.method.currentText(),
                "max_lag": self.max_lag.value()}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import set_widget_value
        set_widget_value(self.method, state.get("method"))
        set_widget_value(self.max_lag, state.get("max_lag"))
        t = state.get("target")
        if t and self._df is not None and t in self._df.columns:
            self._on_select(t)

    def _compute(self) -> None:
        # Ranking is the library's; the tab only reads the result back.
        method = self.method.currentText().lower()
        self._ranked = dv.analysis.rank_drivers(
            self._df, target=self._target, method=method, max_lag=self.max_lag.value())
        self._fill_stats()
        self._fill_table()

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        if self._df is None or self._target is None or self._target not in self._df.columns:
            return None
        from diive.analysis.correlation import rank_drivers_to_code
        return rank_drivers_to_code(
            self._target, method=self.method.currentText().lower(),
            max_lag=self.max_lag.value())

    def _fill_stats(self) -> None:
        res = self._ranked
        top_name = res.iloc[0]["DRIVER"] if (res is not None and not res.empty) else "—"
        top_r = f"{res.iloc[0]['CORR']:+.3f}" if (res is not None and not res.empty) else "—"
        target_cov = 100.0 * self._df[self._target].notna().mean() if self._target else 0.0
        cards = [
            ("Target", self._target or "—"),
            ("Top driver", top_name),
            ("Top r", top_r),
            ("Drivers ranked", _fmt(0 if res is None else len(res))),
            ("Method", self.method.currentText()),
            ("Target coverage", f"{target_cov:.0f}%"),
        ]
        self._set_stat_cards(cards)

    def _fill_table(self) -> None:
        res = self._ranked
        self._filling = True
        self.table.setSortingEnabled(False)
        try:
            self.table.clearSelection()
            self.table.setRowCount(0 if res is None else len(res))
            if res is not None:
                for r, (_, row) in enumerate(res.iterrows()):
                    corr = float(row["CORR"])
                    name_item = QTableWidgetItem(str(row["DRIVER"]))
                    name_item.setData(_DRIVER_ROLE, str(row["DRIVER"]))
                    r_item = _NumItem(corr, f"{corr:+.3f}")
                    # Tint by strength so strong drivers pop (green +, red -).
                    tint = QColor("#43A047") if corr >= 0 else QColor("#E53935")
                    tint.setAlphaF(0.10 + 0.30 * abs(corr))
                    r_item.setBackground(tint)
                    r_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    lag_item = _NumItem(int(row["BEST_LAG"]), str(int(row["BEST_LAG"])))
                    lag_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    n_item = _NumItem(int(row["N"]), f"{int(row['N']):,}")
                    n_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, 0, name_item)
                    self.table.setItem(r, 1, r_item)
                    self.table.setItem(r, 2, lag_item)
                    self.table.setItem(r, 3, n_item)
        finally:
            self.table.setSortingEnabled(True)
            self._filling = False
        if res is not None and len(res):
            self.table.selectRow(0)  # shows the top driver's scatter
        else:
            self._render_scatter(None)

    # --- click-to-scatter ----------------------------------------------
    def _on_driver_selected(self) -> None:
        if self._filling or self._ranked is None:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        driver = item.data(_DRIVER_ROLE) if item else None
        if not driver:
            return
        match = self._ranked[self._ranked["DRIVER"] == driver]
        if match.empty:
            return
        self._render_scatter(match.iloc[0])

    def _render_scatter(self, row) -> None:
        if row is None:
            self.canvas.show_message("No drivers to show")
            return
        ax = self.canvas.new_axes(1)[0]
        driver = str(row["DRIVER"])
        lag = int(row["BEST_LAG"])
        try:
            x = self._df[driver].shift(lag) if lag else self._df[driver]
            y = self._df[self._target]
            sub = pd.concat([x.rename(driver), y.rename(self._target)], axis=1).dropna()
            title = f"{self._target} vs. {driver}" + (f"  (lag {lag:+d} rec.)" if lag else "")
            dv.plotting.ScatterXY(x=sub[driver], y=sub[self._target]).plot(
                ax=ax, format_style=dv.plotting.FormatStyle(
                    title=title, xlabel=driver, ylabel=self._target))
            self.canvas.draw()
        except Exception as err:
            self.canvas.show_message(f"Cannot plot:\n{err}")
