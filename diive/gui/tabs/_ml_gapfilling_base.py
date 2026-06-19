"""
GUI.TABS._ML_GAPFILLING_BASE: TEMPLATE FOR ML GAP-FILLING TABS
==============================================================

The shared layout, flow and results plumbing for the machine-learning
gap-filling tabs (XGBoost, Random Forest, ...). A concrete tab subclasses
``MlGapFillingTab`` and supplies only the method-specific bits:

  * the library model class (``_model_class``),
  * the hyperparameter widgets (``_build_model_box``) and the kwargs they
    produce (``_method_kwargs``),
  * the save/restore control map (``_method_controls``),
  * the reproducible-script renderer (``_codegen``),
  * a few labels (``title``, ``method_name``, the hero chip).

Everything else — the Model/Results sub-tabs, the three-list target/feature
picker, the performance hero, the observed-vs-gap-filled heatmaps, the SHAP
table, the full Results dashboard, the worker thread and the emit/merge flow —
is identical across methods and lives here once. All computation is library work
(the ``*TS`` gap-filling classes + ``dv.plotting``); this template only collects
inputs, runs them off-thread, previews, and emits. Strict GUI<->library separation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
# Reuse the Overview's hero building blocks (presentation-only helpers).
from diive.gui.tabs.overview import _MetricSlot, _chip_qss, _stat_separator
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.dual_variable_picker import DualVariablePicker
from diive.gui.widgets.gapfill_results import GapFillResultsPanel
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.sub_tabs import SubTabs
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel
from diive.gui.widgets.worker import WorkerRunner

_C_MUTED = "#6B7780"

#: Heatmap tick / axis-label / colorbar font size — matches the Overview tab.
_HM_FONT = 9

#: Role on the value cell holding the raw SHAP importance (for the bar delegate).
_SHAP_ROLE = Qt.ItemDataRole.UserRole + 1


class _ShapBarDelegate(QStyledItemDelegate):
    """Paints the SHAP-value cell as a faint horizontal bar (proportional to the
    importance) behind a right-aligned value — a mini bar chart inside the table."""

    _BAR = QColor("#BBD7F0")   # light blue fill
    _TEXT = QColor("#263238")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.maxval = 1.0  # set by the tab when the table is (re)filled

    def paint(self, painter, option, index) -> None:
        val = index.data(_SHAP_ROLE)
        rect = option.rect
        painter.save()
        painter.fillRect(rect, QColor("#FFFFFF"))
        if isinstance(val, (int, float)) and self.maxval > 0:
            frac = max(0.0, min(1.0, float(val) / self.maxval))
            bw = int((rect.width() - 10) * frac)
            if bw > 0:
                bar = QRect(rect.left() + 4, rect.top() + 4, bw, rect.height() - 8)
                painter.setBrush(self._BAR)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(bar, 3, 3)
        painter.setPen(self._TEXT)
        painter.drawText(rect.adjusted(6, 0, -8, 0),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                         index.data(Qt.ItemDataRole.DisplayRole) or "")
        painter.restore()


class _Signals(QObject):
    """Qt signal (DiiveTab is a plain ABC, not a QObject). Run done/failed
    plumbing lives in :class:`WorkerRunner`."""
    features_created = Signal(object)


class _PerformanceHero(QFrame):
    """Hero band above the plots showing the gap-filling model's performance.

    Same visual pattern as the Overview's `_HeroBand` (identity row + a row of
    persistent `_MetricSlot`s with hairline separators), but the metrics are the
    held-out (test-set) model scores from gap-filling, not variable statistics.
    Built once; `set_metrics` updates the slot text in place (no flicker). The
    method chip (label + colours) is supplied by the concrete tab."""

    #: (label, tooltip) per metric slot, in display order.
    _METRICS = [
        ("R²", "Coefficient of determination on the held-out test set (1 = perfect)."),
        ("RMSE", "Root mean squared error on the test set, in target units."),
        ("MAE", "Mean absolute error on the test set, in target units."),
        ("MAPE", "Mean absolute percentage error on the test set."),
        ("MAXE", "Maximum absolute error on the test set, in target units."),
        ("FILLED", "Number of gap records filled by the model."),
        ("FALLBACK", "Of the filled records, how many used the low-quality "
                     "fallback model (timestamp features only, because some "
                     "predictor was missing). A high count means weak fills."),
        ("FEATURES", "Number of feature variables used to train the model."),
    ]

    def __init__(self, method_label: str, chip_bg: str, chip_fg: str) -> None:
        super().__init__()
        self.setObjectName("perfhero")
        border = theme.manager.tokens["BORDER"]
        self.setStyleSheet(
            f"QFrame#perfhero {{ background: #FFFFFF; border: 1px solid {border};"
            f" border-radius: 10px; }}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 11, 16, 11)
        lay.setSpacing(8)

        # --- identity row ---
        idrow = QHBoxLayout()
        idrow.setSpacing(8)
        title = QLabel("Gap-filling performance")
        tf = title.font()
        tf.setPointSizeF(tf.pointSizeF() + 3.0)
        tf.setBold(True)
        title.setFont(tf)
        idrow.addWidget(title)
        self._target = QLabel()
        idrow.addWidget(self._target)
        self._method = QLabel(method_label)
        self._method.setStyleSheet(_chip_qss(chip_bg, chip_fg))
        idrow.addWidget(self._method)
        # Make explicit that the fit/error metrics are held-out (not in-sample).
        test_chip = QLabel("HELD-OUT TEST")
        test_chip.setStyleSheet(_chip_qss("#E8F5E9", "#2E7D32"))
        test_chip.setToolTip(
            "R² / RMSE / MAE / MAPE / MAXE are computed on a held-out test split "
            "(default 25% of complete records), not on the training data.\n\n"
            "The split is RANDOM (scattered in time), not a temporal block — this is "
            "the right test for gap-filling: real gaps are short and scattered among "
            "observed data, so a random hold-out reproduces the actual task of "
            "interpolating an isolated timestamp from present neighbours. A temporal/"
            "block split would instead measure extrapolation skill (relevant for "
            "forecasting or driver analysis), which understates gap-filling performance.")
        idrow.addWidget(test_chip)
        idrow.addStretch(1)
        lay.addLayout(idrow)

        # --- metric row (persistent slots) ---
        self._slots: dict[str, _MetricSlot] = {}
        rowlay = QHBoxLayout()
        rowlay.setSpacing(14)
        for i, (label, _tip) in enumerate(self._METRICS):
            if i > 0:
                rowlay.addWidget(_stat_separator())
            slot = _MetricSlot()
            self._slots[label] = slot
            rowlay.addWidget(slot)
        rowlay.addStretch(1)
        lay.addLayout(rowlay)

        self.reset()

    def reset(self, target: str = "") -> None:
        """Blank the metrics (placeholder dashes); show the target if known."""
        if target:
            self._target.setText(target)
            self._target.setStyleSheet(_chip_qss("#E3F2FD", "#1565C0"))
            self._target.show()
        else:
            self._target.hide()
        for label, tip in self._METRICS:
            self._slots[label].update_metric(label, "—", tip)

    def set_metrics(self, target: str, n_features: int, n_filled: int,
                    n_fallback: int, scores: dict) -> None:
        self._target.setText(target)
        self._target.setStyleSheet(_chip_qss("#E3F2FD", "#1565C0"))
        self._target.show()

        def g(key: str) -> str:
            v = scores.get(key)
            return f"{v:.3g}" if isinstance(v, (int, float)) else "—"

        # Fallback as a share of the filled records, so its weight is clear.
        fb = f"{n_fallback:,}"
        if n_filled > 0:
            fb += f" ({100.0 * n_fallback / n_filled:.0f}%)"
        values = {
            "R²": g("r2"), "RMSE": g("rmse"), "MAE": g("mae"),
            "MAPE": g("mape"), "MAXE": g("maxe"),
            "FILLED": f"{n_filled:,}", "FALLBACK": fb,
            "FEATURES": f"{n_features}",
        }
        for label, tip in self._METRICS:
            self._slots[label].update_metric(label, values.get(label, "—"), tip)


class MlGapFillingTab(DiiveTab):
    """Template tab for the ML gap-fillers. Subclass and fill the method hooks.

    Pick a target + features in three lists, configure the model, gap-fill on a
    worker thread, preview (hero + heatmaps + SHAP + Results dashboard), emit the
    gap-filled + flag columns. Subclasses override the small method-specific
    surface; the whole layout/flow below is shared."""

    # --- method hooks (override in subclasses) -------------------------
    title = "ML gap-filling"
    #: Human method name, used in tooltips/status/heatmap header (e.g. "XGBoost").
    method_name = "model"
    #: Hero chip text + colours (e.g. "XGBOOST", lilac).
    method_chip_label = "MODEL"
    method_chip_bg = "#ECEFF1"
    method_chip_fg = "#37474F"

    #: Minimum number of distinct years in the record for the long-term option to
    #: be selectable ("spans more than 3 years" -> at least 4 distinct years).
    longterm_min_years = 4

    def _model_class(self):
        """Return the library gap-filling class (e.g. ``XGBoostTS``)."""
        raise NotImplementedError

    def _longterm_model_class(self):
        """Return the per-year long-term gap-filling class (e.g.
        ``LongTermGapFillingXGBoostTS``), or None if the method has no long-term
        variant (then the long-term option is never shown)."""
        return None

    def _longterm_codegen(self, target: str, features: list[str], kwargs: dict,
                          reduce: bool, shap_factor: float) -> str:
        """Return a runnable diive snippet reproducing a long-term run.

        Subclasses that support a long-term variant must override this."""
        raise NotImplementedError

    def _build_model_box(self) -> QGroupBox:
        """Return a QGroupBox of method hyperparameter widgets (stored on self)."""
        raise NotImplementedError

    def _method_kwargs(self) -> dict:
        """Return the model constructor kwargs from the hyperparameter widgets
        (without ``input_df``/``target_col``/``verbose``)."""
        raise NotImplementedError

    def _method_controls(self) -> dict:
        """Return {name: widget} of the method controls, for save/restore."""
        raise NotImplementedError

    def _codegen(self, target: str, features: list[str], kwargs: dict,
                 reduce: bool, shap_factor: float) -> str:
        """Return a runnable diive snippet reproducing this run."""
        raise NotImplementedError

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._all_cols: list[str] = []
        self._created: set = set()
        self._target: str = ""
        self._result_df: pd.DataFrame | None = None  # columns to emit on "Add"
        self._n_years = 0  # distinct years in the current record (gates long-term)

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

        # Title bar (the Run/Add buttons sit in the sub-tab row, not here; Copy
        # Python sits at the far-right edge of the header row).
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this gap-filling run.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        # Action buttons (built here, mounted into the sub-tab row so they sit
        # next to the page pills and stay prominent).
        self.run_btn = QPushButton("Run gap-filling")
        self.run_btn.setToolTip(
            f"Train {self.method_name} on the selected features and fill the target's gaps.")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        self.add_btn = QPushButton("Add results to dataset")
        self.add_btn.setToolTip("Append the gap-filled series and its flag to the variable list.")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")

        # Sub-tabs split the configuration+output (Model) from the Results
        # dashboard — the standardized layout for output-heavy tabs. The action
        # buttons live in the tab row (corner), always visible.
        self.subtabs = SubTabs()
        self.subtabs.add_page("Model", self._build_model_page())
        self.subtabs.add_page("Results", self._build_results_page())
        self.subtabs.add_corner_separator()
        self.subtabs.add_corner_widget(self.run_btn)
        self.subtabs.add_corner_widget(self.add_btn)
        outer.addWidget(self.subtabs, stretch=1)
        return root

    def _build_model_page(self) -> QWidget:
        """Model page = the full gap-filling workspace: inputs | settings | the
        results region (hero + heatmaps + SHAP). A plain HBox (not a splitter)
        honours the fixed widths and hands the rest to the results region."""
        page = QWidget()
        body = QHBoxLayout(page)
        body.setContentsMargins(10, 4, 10, 4)
        body.setSpacing(0)
        body.addWidget(self._build_inputs())
        body.addWidget(self._build_settings())
        body.addWidget(self._build_results(), stretch=1)
        return page

    def _build_results_page(self) -> QWidget:
        """Results page — a card dashboard with the full score/config/quality
        tables and extra diagnostic + temporal plots (populated after a run)."""
        self.results_panel = GapFillResultsPanel()
        return self.results_panel

    def _build_inputs(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        intro = QLabel(
            "Click a variable in 'Target' to set the gap-fill target. Click a "
            "variable in 'Available features' to use it as a model feature; click "
            "one in 'Selected features' to remove it.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        row = QHBoxLayout()

        # --- target list (far left) ---
        tcol = QVBoxLayout()
        tcol.addWidget(self._list_header("Target", "click to set target"))
        self.target_list = VariablePanel()
        self.target_list.list.setToolTip("Click a variable to set it as the gap-fill target.")
        self.target_list.selected.connect(lambda name, _c: self._set_target(name))
        tcol.addWidget(self.target_list, stretch=1)
        row.addLayout(tcol)

        # --- available features (middle) | selected features (right) ---
        self.picker = DualVariablePicker(
            available_title="Available features", selected_title="Selected features",
            available_hint="click to use", selected_hint="click to remove")
        self.picker.changed.connect(self._update_status)
        row.addWidget(self.picker, stretch=1)

        v.addLayout(row, stretch=1)

        self.target_label = QLabel("Target: (none)")
        self.target_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.target_label)
        return host

    def _build_settings(self) -> QWidget:
        host = QWidget()
        host.setFixedWidth(320)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)  # drop the faint outer border
        inner = QWidget()
        v = QVBoxLayout(inner)

        lt_box = self._build_longterm_box()         # long-term per-year mode (optional)
        if lt_box is not None:
            v.addWidget(lt_box)
        v.addWidget(self._build_model_box())       # method-specific hyperparameters
        v.addWidget(self._build_reduction_box())   # shared SHAP feature reduction
        v.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        return host

    def _build_longterm_box(self) -> QGroupBox | None:
        """Optional 'Long-term' mode box. Only built when the method has a
        long-term variant; the checkbox is enabled only when the record spans
        enough years (set in ``on_data_loaded``)."""
        self.longterm_cb = None
        if self._longterm_model_class() is None:
            return None
        box = QGroupBox("Long-term mode")
        lv = QVBoxLayout(box)
        self.longterm_cb = QCheckBox("Build one model per year (neighbouring years)")
        self.longterm_cb.setToolTip(
            "Gap-fill each calendar year with its own model, trained on that year "
            "plus its two closest neighbouring years (e.g. 2016 is filled from "
            "2015-2017). This tracks slow inter-annual change better than one model "
            "for the whole record.\n\n"
            f"Available only when the record spans more than "
            f"{self.longterm_min_years - 1} years.")
        self.longterm_cb.toggled.connect(self._on_longterm_toggled)
        lv.addWidget(self.longterm_cb)
        self.longterm_hint = QLabel()
        self.longterm_hint.setWordWrap(True)
        self.longterm_hint.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        lv.addWidget(self.longterm_hint)
        return box

    def _longterm_enabled(self) -> bool:
        """True if the long-term checkbox exists, is enabled, and is checked."""
        cb = getattr(self, "longterm_cb", None)
        return bool(cb is not None and cb.isEnabled() and cb.isChecked())

    def _on_longterm_toggled(self, *_) -> None:
        self._update_status()

    def _build_reduction_box(self) -> QGroupBox:
        """The shared SHAP feature-reduction controls (every ML method has them)."""
        red_box = QGroupBox("Feature reduction")
        rv = QVBoxLayout(red_box)
        self.reduce_cb = QCheckBox("Reduce features (SHAP importance)")
        self.reduce_cb.setToolTip(
            "Before the final model, drop features whose SHAP importance is below a "
            "random-baseline threshold.")
        rv.addWidget(self.reduce_cb)
        rf = QFormLayout()
        self.shap_factor = QDoubleSpinBox()
        self.shap_factor.setRange(0.0, 5.0); self.shap_factor.setDecimals(2)
        self.shap_factor.setSingleStep(0.1); self.shap_factor.setValue(0.5)
        self.shap_factor.setToolTip(
            "Higher = stricter reduction (keeps fewer features). Threshold = random "
            "importance + factor × random SD.")
        rf.addRow("SHAP threshold factor", self.shap_factor)
        rv.addLayout(rf)
        return red_box

    def _build_results(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.status = QLabel(
            "Set a target and at least one feature, then run gap-filling.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        v.addWidget(self.status)
        # Hero band with the model's performance metrics — spans the full width
        # of the right region (over both the heatmaps and the SHAP panel).
        self.hero = _PerformanceHero(
            self.method_chip_label, self.method_chip_bg, self.method_chip_fg)
        hwrap = QWidget()
        hl = QVBoxLayout(hwrap)
        hl.setContentsMargins(10, 0, 10, 6)
        hl.addWidget(self.hero)
        v.addWidget(hwrap)
        # Below the hero: heatmaps (stretch) | SHAP panel at the far-right edge.
        # A plain HBox (not a splitter) honours the SHAP panel's fixed width
        # exactly and hands all remaining width to the heatmaps, so SHAP sits
        # flush against the right edge with no gap.
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(0)
        bottom.addWidget(self._build_heatmaps(), stretch=1)
        bottom.addWidget(self._build_right_panels())
        v.addLayout(bottom, stretch=1)
        return host

    def _build_heatmaps(self) -> QWidget:
        """Heatmap canvas with Qt panel headers (one per panel) above it, styled
        like the SHAP header so the matplotlib titles can be dropped."""
        col = QWidget()
        cv = QVBoxLayout(col)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        headers = QHBoxLayout()
        headers.setContentsMargins(0, 0, 0, 0)
        for text in ("Observed (with gaps)", f"{self.method_name} gap-filled"):
            lbl = self._panel_header(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            headers.addWidget(lbl, stretch=1)
        cv.addLayout(headers)
        self.canvas = MplCanvas()
        cv.addWidget(self.canvas, stretch=1)
        return col

    @staticmethod
    def _panel_header(text: str) -> QLabel:
        """A panel header styled like the SHAP header (tracked uppercase, bold)."""
        lbl = QLabel(theme.manager.label_text(text))
        f = theme.manager.tracked_font(lbl.font())
        f.setBold(True)
        lbl.setFont(f)
        return lbl

    def _build_right_panels(self) -> QWidget:
        """Narrow far-right column: a SHAP feature-importance table. Fixed width so
        the heatmaps absorb the rest and this column sits flush against the right
        edge (a maximumWidth would fight the layout and leave a gap)."""
        host = QWidget()
        host.setFixedWidth(250)
        v = QVBoxLayout(host)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        header = self._panel_header("Feature importance (SHAP)")
        header.setToolTip(
            "Mean |SHAP value| per feature of the final gap-filling model — how "
            "much each driver moves the model's prediction. Computed over all "
            "complete observations (not just the held-out test set, unlike the "
            "performance scores above).")
        v.addWidget(header)
        self.shap_caption = QLabel("Final model, all complete observations")
        self.shap_caption.setWordWrap(True)
        self.shap_caption.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
        v.addWidget(self.shap_caption)

        self.shap_table = QTableWidget(0, 2)
        self.shap_table.setHorizontalHeaderLabels(["Feature", "mean |SHAP|"])
        self.shap_table.verticalHeader().setVisible(False)
        self.shap_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.shap_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.shap_table.setShowGrid(False)
        self.shap_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shap_table.setWordWrap(False)
        hh = self.shap_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.shap_table.setColumnWidth(1, 96)
        self._shap_delegate = _ShapBarDelegate(self.shap_table)
        self.shap_table.setItemDelegateForColumn(1, self._shap_delegate)
        self.shap_table.setStyleSheet(
            "QTableWidget { background: #FFFFFF; border: 1px solid "
            f"{theme.manager.tokens['BORDER']}; border-radius: 8px; }}"
            "QTableWidget::item { padding: 3px 6px; color: #263238; }"
            "QHeaderView::section { background: #F5F6F7; color: #6B7780; "
            "border: none; padding: 5px 6px; font-weight: 600; }")
        v.addWidget(self.shap_table, stretch=1)
        return host

    _list_header = staticmethod(list_header)

    # --- data ----------------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._created = set(created or set())
        self._all_cols = [str(c) for c in df.columns]
        self._result_df = None
        # Keep any rendered results visible across data pushes (e.g. our own
        # 'Add results' merge); only the Add button gates on a fresh run.
        self.add_btn.setEnabled(False)

        self._update_longterm_availability(df)
        self.target_list.set_variables(self._all_cols, self._created)
        if self._target not in self._all_cols:
            self._target = ""
        self._refresh_features()
        self.target_list.set_panels([self._target] if self._target else [])
        self._update_target_label()
        self._update_status()

    def _update_longterm_availability(self, df) -> None:
        """Enable the long-term checkbox only when the record spans enough years."""
        cb = getattr(self, "longterm_cb", None)
        try:
            self._n_years = int(pd.DatetimeIndex(df.index).year.nunique())
        except Exception:
            self._n_years = 0
        if cb is None:
            return
        enough = self._n_years >= self.longterm_min_years
        cb.setEnabled(enough)
        if not enough:
            cb.setChecked(False)
            self.longterm_hint.setText(
                f"Record spans {self._n_years} year(s) — long-term needs more than "
                f"{self.longterm_min_years - 1}. One model is used for the whole record.")
        else:
            self.longterm_hint.setText(
                f"Record spans {self._n_years} years — one model per year, each "
                f"trained on the year plus its two closest neighbours.")

    def _set_target(self, name: str) -> None:
        if not name or name not in self._all_cols:
            return
        self._target = name
        self.target_list.set_panels([name])
        self._refresh_features()  # drop the target from the feature pool
        self._update_target_label()
        self._update_status()

    def _refresh_features(self) -> None:
        """Feature pool = all columns except the target (keeps current selection)."""
        pool = [c for c in self._all_cols if c != self._target]
        self.picker.set_variables(pool, self._created)

    def _update_target_label(self) -> None:
        self.target_label.setText(f"Target: {self._target or '(none)'}")

    def _feature_cols(self) -> list[str]:
        return [f for f in self.picker.selected_names() if f != self._target]

    def _update_status(self, *_) -> None:
        if self._runner.is_running:
            return
        feats = self._feature_cols()
        if not self._target:
            self.status.setText("Click a variable in 'Target' to set the gap-fill target.")
        elif not feats:
            self.status.setText(
                f"Target: {self._target}. Now click variables in 'Available features'.")
        else:
            mode = (f" — long-term ({self._n_years} per-year models)"
                    if self._longterm_enabled() else "")
            self.status.setText(
                f"Target: {self._target} — {len(feats)} feature(s) selected{mode}. "
                f"Run gap-filling.")

    # --- state ---------------------------------------------------------
    def _controls(self) -> dict:
        ctrls = {**self._method_controls(),
                 "reduce": self.reduce_cb, "shap_factor": self.shap_factor}
        if getattr(self, "longterm_cb", None) is not None:
            ctrls["longterm"] = self.longterm_cb
        return ctrls

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "features": self.picker.selected_names(),
                "controls": save_controls(self._controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._controls(), state.get("controls"))
        tgt = state.get("target")
        if tgt in self._all_cols:
            self._set_target(tgt)
        feats = [f for f in (state.get("features") or []) if f in self.picker.all_names()]
        self.picker.set_selected(feats)
        self._update_status()

    # --- run -----------------------------------------------------------
    def _python_code(self) -> str | None:
        """Reproducible diive script for the current target/features/settings
        (library codegen; returns None until a target + features are chosen)."""
        target = self._target
        feats = self._feature_cols()
        if not target or not feats:
            return None
        gen = self._longterm_codegen if self._longterm_enabled() else self._codegen
        return gen(
            target, feats, self._method_kwargs(),
            self.reduce_cb.isChecked(), self.shap_factor.value())

    def _run(self) -> None:
        if self._df is None or self._runner.is_running:
            return
        target = self._target
        feats = self._feature_cols()
        if not target:
            self.status.setText("Set a target variable first.")
            return
        if not feats:
            self.status.setText("Select at least one feature variable.")
            return
        kwargs = self._method_kwargs()
        reduce = self.reduce_cb.isChecked()
        shap_factor = self.shap_factor.value()
        longterm = self._longterm_enabled()

        work = self._df[[target] + feats].copy()
        self._set_running(True)
        extra = (f" — long-term, {self._n_years} per-year models"
                 if longterm else "")
        self.status.setText(
            f"Gap-filling… (training {self.method_name}{extra} — this can take a while)")
        self._runner.run(self._compute_payload, work, target, kwargs, reduce,
                         shap_factor, longterm)

    def _compute_payload(self, work, target, kwargs, reduce, shap_factor, longterm):
        """Train + gap-fill off-thread; return the result tuple consumed by
        :meth:`_on_done`. Raises on error (the runner forwards to
        :meth:`_on_failed`)."""
        if longterm:
            model, gapfilled, flag, scores = self._run_longterm(
                work, target, kwargs, reduce, shap_factor)
        else:
            model, gapfilled, flag, scores = self._run_single(
                work, target, kwargs, reduce, shap_factor)

        # Flag: 0 = observed, 1 = full-model fill, 2 = fallback fill.
        n_fallback = int((flag == 2).sum())

        out = pd.DataFrame({str(gapfilled.name): gapfilled,
                            str(flag.name): flag})
        attrs = {
            str(gapfilled.name): provenance_attr(
                origin=DERIVED, parent=str(target), operation=self.title,
                tags=["gapfilling", self.method_name.lower()]),
            str(flag.name): provenance_attr(
                origin=DERIVED, parent=str(target), operation=self.title,
                tags=["gapfilling", "flag"]),
        }
        out.attrs[ATTRS_KEY] = attrs
        n_features = work.shape[1] - 1  # everything except the target
        # Pass the reduction factor so the Results card can show the exact
        # threshold equation/line (None when reduction was off).
        factor = shap_factor if reduce else None
        return (out, work[target], gapfilled, scores, target, n_features,
                n_fallback, model, factor, longterm)

    def _run_single(self, work, target, kwargs, reduce, shap_factor):
        # verbose=2 (PROGRESS): surfaces the base class's rich, coloured
        # training/gap-filling report in the Log tab (which mirrors the console).
        model = self._model_class()(input_df=work, target_col=target,
                                    verbose=2, **kwargs)
        if reduce:
            model.reduce_features(shap_threshold_factor=shap_factor)
        model.run(showplot_scores=False, showplot_importance=False)
        scores = model.scores_traintest_ or model.scores_
        return model, model.get_gapfilled_target(), model.get_flag(), scores

    def _run_longterm(self, work, target, kwargs, reduce, shap_factor):
        """Long-term flow: one model per year (built from the year + its two
        neighbours), stitched back together. Hero scores are the per-year
        held-out scores averaged across years."""
        model = self._longterm_model_class()(input_df=work, target_col=target,
                                             verbose=2, **kwargs)
        model.run(reduce_features=reduce, shap_threshold_factor=shap_factor)
        scores = model.scores_traintest_overall_ or model.scores_overall_
        return model, model.get_gapfilled_target(), model.get_flag(), scores

    def _set_running(self, on: bool) -> None:
        self.run_btn.setEnabled(not on)
        if on:
            self.add_btn.setEnabled(False)

    # --- results -------------------------------------------------------
    def _on_done(self, payload) -> None:
        (out, observed, gapfilled, scores, target, n_features,
         n_fallback, model, factor, longterm) = payload
        self._set_running(False)
        self._result_df = out
        n_filled = int((out[str(gapfilled.name)].notna() & observed.isna()).sum())
        self.hero.set_metrics(target=target, n_features=n_features,
                              n_filled=n_filled, n_fallback=n_fallback, scores=scores)
        mode = " (long-term, per-year models)" if longterm else ""
        self.status.setText(f"Done{mode}. 'Add' appends {', '.join(out.columns)}.")
        self.add_btn.setEnabled(True)
        self.shap_caption.setText("Mean across per-year models" if longterm
                                  else "Final model, all complete observations")
        self._plot(observed, gapfilled)
        if longterm:
            self._fill_shap_table_longterm(model)
            self.results_panel.update_longterm(model, target,
                                                shap_threshold_factor=factor)
        else:
            self._fill_shap_table(model)
            self.results_panel.update(model, target, shap_threshold_factor=factor)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.hero.reset(self._target)
        self.status.setText(f"Failed: {msg}")
        ax = self.canvas.new_axes(1)[0]
        ax.text(0.5, 0.5, "Gap-filling failed", ha="center", va="center",
                transform=ax.transAxes)
        self.canvas.draw()
        self.results_panel.reset(f"Gap-filling failed: {msg}")

    def _fill_shap_table(self, model) -> None:
        """Fill the SHAP table from the model's feature importances (a DataFrame
        with a SHAP_IMPORTANCE column, index = feature). Sorted strongest-first;
        the bar delegate scales each value against the maximum."""
        table = self.shap_table
        table.setRowCount(0)
        try:
            fi = model.feature_importances_  # DataFrame: index=feature, SHAP_IMPORTANCE
            ser = fi["SHAP_IMPORTANCE"].sort_values(ascending=False)
            # Drop the random-baseline row if it slipped in (reduction artifact).
            ser = ser[[i for i in ser.index if i != getattr(model, "random_col", None)]]
        except Exception:
            return
        self._shap_delegate.maxval = float(ser.max()) if len(ser) else 1.0
        table.setRowCount(len(ser))
        for row, (name, val) in enumerate(ser.items()):
            name_item = QTableWidgetItem(str(name))
            name_item.setToolTip(str(name))  # full name (cells elide when narrow)
            val_item = QTableWidgetItem()
            val_item.setData(Qt.ItemDataRole.DisplayRole, f"{val:.4f}")
            val_item.setData(_SHAP_ROLE, float(val))
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, val_item)

    def _fill_shap_table_longterm(self, model) -> None:
        """Fill the SHAP table with each feature's mean importance across the
        per-year models (long-term has no single ``feature_importances_``)."""
        table = self.shap_table
        table.setRowCount(0)
        try:
            fi = model.feature_importance_per_year  # DataFrame: index=feature, cols=years
            ser = fi.mean(axis=1).sort_values(ascending=False).dropna()
        except Exception:
            return
        self._shap_delegate.maxval = float(ser.max()) if len(ser) else 1.0
        table.setRowCount(len(ser))
        for row, (name, val) in enumerate(ser.items()):
            name_item = QTableWidgetItem(str(name))
            name_item.setToolTip(f"{name} (mean |SHAP| across years)")
            val_item = QTableWidgetItem()
            val_item.setData(Qt.ItemDataRole.DisplayRole, f"{val:.4f}")
            val_item.setData(_SHAP_ROLE, float(val))
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, val_item)

    def _plot(self, observed, gapfilled) -> None:
        """Side-by-side date/time heatmaps: observed (with gaps) vs gap-filled.

        Both panels share one value scale, so only the right panel draws a
        colorbar and the left's y-axis is the only "Date" axis (the right shares
        it). Dropping the duplicate colorbar + y-axis lets the two heatmaps fill
        far more of the canvas width."""
        ax_obs, ax_gf = self.canvas.new_axes(2, sharey=True)
        try:
            # Shared colour scale (robust 1–99th pct over both series), so the
            # single colorbar on the right is valid for both panels. Fonts match
            # the Overview tab; titles come from the Qt headers above the canvas.
            both = pd.concat([observed.dropna(), gapfilled.dropna()])
            vmin = float(np.nanpercentile(both, 1)) if len(both) else None
            vmax = float(np.nanpercentile(both, 99)) if len(both) else None
            opts = {"vmin": vmin, "vmax": vmax, "cb_labelsize": _HM_FONT}
            hm_style = dv.plotting.FormatStyle(ticks_fontsize=_HM_FONT, axlabel_fontsize=_HM_FONT)
            dv.plotting.HeatmapDateTime(series=observed).plot(
                ax=ax_obs, format_style=hm_style, show_colormap=False, **opts)
            dv.plotting.HeatmapDateTime(series=gapfilled).plot(ax=ax_gf, format_style=hm_style, **opts)
            # Drop the library's auto title (series name) — the Qt headers above
            # the canvas label the panels instead.
            ax_obs.set_title("")
            ax_gf.set_title("")
            # Right panel shares the left's dates — drop its redundant y-axis.
            ax_gf.set_ylabel("")
            ax_gf.tick_params(labelleft=False)
        except Exception as err:  # plotting must never crash the tab
            ax_obs.text(0.5, 0.5, f"Plot failed: {err}", ha="center", va="center",
                        transform=ax_obs.transAxes, fontsize=8)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)
        self.status.setText(f"Added {', '.join(result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
