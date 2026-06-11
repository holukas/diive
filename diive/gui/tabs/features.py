"""
GUI.TABS.FEATURES: FEATURE ENGINEERING TAB
==========================================

Build engineered features with `diive.core.ml.feature_engineer.FeatureEngineer`:
move variables from the available list (left) into the selected-features list,
configure the engineering stages, and run. Created features can then be added to
the app's variable list, where they are tagged with a "NEW" pill.

All computation is library work (`FeatureEngineer`); this tab only collects the
inputs and runs it (on a worker thread to keep the UI responsive).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.core.ml.feature_engineer import FeatureEngineer
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.variable_panel import VariablePanel

#: Placeholder target column (FeatureEngineer requires one; excluded from output).
_TARGET = "__fe_target__"


def _parse_ints(text: str) -> list[int]:
    return [int(p) for p in text.split(",") if p.strip()]


def _parse_strs(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


class _FeatureSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""
    run_done = Signal(object)
    run_failed = Signal(str)
    features_created = Signal(object)


class FeatureEngineerTab(DiiveTab):
    """Select variables, configure stages, run the FeatureEngineer."""

    title = "Features"

    def build(self) -> QWidget:
        self._df = None
        self._created_df = None
        self._sig = _FeatureSignals()
        #: Exposed bound signal the main window connects to.
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QHBoxLayout(root)

        # Fixed-width columns packed left (trailing stretch), so the controls
        # stay compact and readable instead of stretching across the wide window.
        # --- Available variables (left): shared variable list. ---
        avail_col = QWidget()
        avail_col.setFixedWidth(self.available_width())
        avail_box = QVBoxLayout(avail_col)
        avail_box.setContentsMargins(0, 0, 0, 0)
        avail_box.addWidget(self._caption("Available variables", "click to add"))
        self.available = VariablePanel()
        self.available.selected.connect(lambda name, _ctrl: self._add_feature(name))
        avail_box.addWidget(self.available)
        layout.addWidget(avail_col)

        # --- Selected features (middle): double-click to remove ---
        sel_col = QWidget()
        sel_col.setFixedWidth(220)
        sel_box = QVBoxLayout(sel_col)
        sel_box.setContentsMargins(0, 0, 0, 0)
        sel_box.addWidget(self._caption("Selected features", "double-click to remove"))
        self.selected = QListWidget()
        self.selected.itemDoubleClicked.connect(self._remove_feature)
        sel_box.addWidget(self.selected)
        layout.addWidget(sel_col)

        # --- Settings + run (right) ---
        settings = self._build_settings()
        settings.setFixedWidth(380)
        layout.addWidget(settings)
        layout.addStretch(1)

        self._sig.run_done.connect(self._on_run_done)
        self._sig.run_failed.connect(self._on_run_failed)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form_host = QVBoxLayout(inner)

        # Lag
        self.lag_cb = QCheckBox("Lag features")
        self.lag_min = QSpinBox(); self.lag_min.setRange(-100, 0); self.lag_min.setValue(-1)
        self.lag_max = QSpinBox(); self.lag_max.setRange(0, 100); self.lag_max.setValue(1)
        self.lag_step = QSpinBox(); self.lag_step.setRange(1, 100); self.lag_step.setValue(1)
        form_host.addWidget(self._group("Lag", self.lag_cb, [
            ("Min lag", self.lag_min), ("Max lag", self.lag_max), ("Step size", self.lag_step)]))

        # Rolling
        self.roll_cb = QCheckBox("Rolling statistics")
        self.roll_windows = QLineEdit("12,24")
        self.roll_stats = QLineEdit()
        self.roll_stats.setPlaceholderText("optional, e.g. median,min,max")
        form_host.addWidget(self._group("Rolling", self.roll_cb, [
            ("Windows", self.roll_windows), ("Extra stats", self.roll_stats)]))

        # Differencing
        self.diff_cb = QCheckBox("Differencing")
        self.diff_orders = QLineEdit("1")
        form_host.addWidget(self._group("Differencing", self.diff_cb, [
            ("Orders", self.diff_orders)]))

        # EMA
        self.ema_cb = QCheckBox("Exponential moving average")
        self.ema_spans = QLineEdit("6,24")
        form_host.addWidget(self._group("EMA", self.ema_cb, [("Spans", self.ema_spans)]))

        # Polynomial
        self.poly_cb = QCheckBox("Polynomial expansion")
        self.poly_deg = QSpinBox(); self.poly_deg.setRange(2, 5); self.poly_deg.setValue(2)
        form_host.addWidget(self._group("Polynomial", self.poly_cb, [("Degree", self.poly_deg)]))

        # STL
        self.stl_cb = QCheckBox("STL decomposition")
        self.stl_period = QSpinBox(); self.stl_period.setRange(0, 100000); self.stl_period.setValue(0)
        form_host.addWidget(self._group("STL", self.stl_cb, [("Seasonal period (0=auto)", self.stl_period)]))

        # Timestamp / record number (no extra inputs)
        self.ts_cb = QCheckBox("Timestamp features (year, season, hour, ...)")
        self.rec_cb = QCheckBox("Continuous record number")
        misc = QGroupBox("Other")
        misc_l = QVBoxLayout(misc)
        misc_l.addWidget(self.ts_cb)
        misc_l.addWidget(self.rec_cb)
        form_host.addWidget(misc)

        form_host.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        self.run_btn = QPushButton("Run feature engineering")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        # Explicit list of the features created by the last run (they also show
        # up in the variable list, but this is a clear, named confirmation).
        self.created_label = QLabel("Newly created features")
        outer.addWidget(self.created_label)
        self.created_list = QListWidget()
        self.created_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.created_list.setMaximumHeight(160)
        outer.addWidget(self.created_list)

        self.add_btn = QPushButton("Add features to variable list")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        outer.addWidget(self.add_btn)

        return panel

    @staticmethod
    def available_width() -> int:
        """Fixed width of the available-variables column (matches the var list)."""
        return theme.manager.list_width

    @staticmethod
    def _caption(title: str, hint: str) -> QLabel:
        """A compact column caption: bold title + a smaller grey hint."""
        label = QLabel(f"<b>{title}</b> <span style='color:#90A4AE'>({hint})</span>")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _group(title: str, checkbox: QCheckBox, rows: list) -> QGroupBox:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        v.addWidget(checkbox)
        form = QFormLayout()
        for label, widget in rows:
            form.addRow(label, widget)
        v.addLayout(form)
        return box

    # --- list moves ---
    def _add_feature(self, name: str) -> None:
        if not name:
            return
        self.available.remove_name(name)
        self.selected.addItem(name)

    def _remove_feature(self, item) -> None:
        if item is None:
            return
        name = item.text()
        self.selected.takeItem(self.selected.row(item))
        self.available.add_name(name)

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._created_df = None
        self.selected.clear()
        self.available.set_variables(df.columns, created)
        self.add_btn.setEnabled(False)
        self.status.setText("")
        self.created_list.clear()
        self.created_label.setText("Newly created features")

    # --- run ---
    def _run(self) -> None:
        if self._df is None:
            return
        names = [self.selected.item(i).text() for i in range(self.selected.count())]
        # Timestamp features and the record number derive from the index alone;
        # the other stages transform selected variables and so need at least one.
        per_var = any(cb.isChecked() for cb in (
            self.lag_cb, self.roll_cb, self.diff_cb,
            self.ema_cb, self.poly_cb, self.stl_cb))
        index_only = self.ts_cb.isChecked() or self.rec_cb.isChecked()
        if not per_var and not index_only:
            self.status.setText("Enable at least one feature stage first.")
            return
        if per_var and not names:
            self.status.setText("Select at least one variable for the chosen stages.")
            return
        try:
            kwargs = self._collect_settings()
        except ValueError as err:
            self.status.setText(f"Invalid settings: {err}")
            return
        # No selected variables (timestamp/record only) -> keep just the index.
        work = (self._df[names].copy() if names else self._df[[]].copy())
        work[_TARGET] = 0.0
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.created_list.clear()
        self.status.setText("Running feature engineering...")
        threading.Thread(
            target=self._run_worker, args=(work, kwargs), daemon=True).start()

    def _collect_settings(self) -> dict:
        kwargs: dict = {"target_col": _TARGET, "verbose": 2}
        if self.lag_cb.isChecked():
            kwargs["features_lag"] = [self.lag_min.value(), self.lag_max.value()]
            kwargs["features_lag_stepsize"] = self.lag_step.value()
        if self.roll_cb.isChecked():
            kwargs["features_rolling"] = _parse_ints(self.roll_windows.text())
            stats = _parse_strs(self.roll_stats.text())
            if stats:
                kwargs["features_rolling_stats"] = stats
        if self.diff_cb.isChecked():
            kwargs["features_diff"] = _parse_ints(self.diff_orders.text())
        if self.ema_cb.isChecked():
            kwargs["features_ema"] = _parse_ints(self.ema_spans.text())
        if self.poly_cb.isChecked():
            kwargs["features_poly_degree"] = self.poly_deg.value()
        if self.stl_cb.isChecked():
            kwargs["features_stl"] = True
            if self.stl_period.value() > 0:
                kwargs["features_stl_seasonal_period"] = self.stl_period.value()
        kwargs["vectorize_timestamps"] = self.ts_cb.isChecked()
        kwargs["add_continuous_record_number"] = self.rec_cb.isChecked()
        return kwargs

    def _run_worker(self, work, kwargs: dict) -> None:
        try:
            out = FeatureEngineer(**kwargs).fit_transform(work)
            inputs = set(work.columns)
            new_cols = [c for c in out.columns if c not in inputs]
            new_df = out[new_cols].copy()
            # Provenance: each engineered column is derived; best-effort link to
            # the source variable whose name is embedded in the feature name.
            src_cols = [str(c) for c in work.columns if c != _TARGET]
            new_df.attrs[ATTRS_KEY] = {
                col: provenance_attr(
                    origin=DERIVED,
                    parent=next((s for s in src_cols if s in str(col)), None),
                    operation="Feature engineering", tags=["feature"])
                for col in new_df.columns}
        except Exception as err:
            self._sig.run_failed.emit(str(err))
            return
        self._sig.run_done.emit(new_df)

    def _on_run_done(self, new_df) -> None:
        self._created_df = new_df
        n = len(new_df.columns)
        self.run_btn.setEnabled(True)
        self.created_list.clear()
        if n:
            self.created_list.addItems([str(c) for c in new_df.columns])
            self.created_label.setText(f"Newly created features ({n})")
            self.status.setText(f"Created {n} features. Click 'Add' to use them.")
            self.add_btn.setEnabled(True)
        else:
            self.created_label.setText("Newly created features")
            self.status.setText("No features created (enable at least one stage).")

    def _on_run_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(f"Failed: {msg}")

    def _add(self) -> None:
        if self._created_df is None or self._created_df.empty:
            return
        # Emitting triggers a data re-push that runs this tab's on_data_loaded
        # (resetting _created_df), so capture what we need first.
        created = self._created_df
        n = len(created.columns)
        self.featuresCreated.emit(created)
        self.status.setText(f"Added {n} features to the variable list.")
        self.add_btn.setEnabled(False)
