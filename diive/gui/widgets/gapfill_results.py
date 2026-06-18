"""
GUI.WIDGETS.GAPFILL_RESULTS: XGBOOST GAP-FILLING RESULTS DASHBOARD
=================================================================

The "Results" page of the XGBoost gap-filling tab — a scrollable, card-based
dashboard that surfaces, natively in Qt, everything the library's gap-filling
report writes to the console (full score tables, configuration, gap-fill quality
breakdown, feature importances) plus extra diagnostic and temporal plots
(predicted-vs-observed, SHAP importance, diel cycle, cumulative sum).

All numbers/plots come straight from the fitted library model
(``XGBoostTS`` / ``MlRegressorGapFillingBase``): ``scores_``/``scores_traintest_``,
``feature_importances_``, ``gapfilling_df_``, ``plot_feature_importances``, and the
``dv.plotting`` classes. This widget only arranges and styles them — strict
GUI<->library separation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.widgets.mpl_canvas import MplCanvas

_C_MUTED = "#6B7780"
_PLOT_FONT = 8

#: Verdict colours shared by the reduction table.
_C_ACCEPT = "#2E7D32"   # green  — feature kept
_C_REJECT = "#9AA0A6"   # grey   — feature dropped
_C_BENCH = "#B26A00"    # amber  — random benchmark row
_C_THRESHOLD = "#E04646"  # red    — threshold line

#: Score metrics in display order: (label, dict key, decimals/'g', tooltip).
_METRICS = [
    ("R²", "r2", "f3",
     "Coefficient of determination: the fraction of the target's variance the "
     "model reproduces. 1 = perfect, 0 = no better than predicting the mean, "
     "negative = worse than the mean."),
    ("RMSE", "rmse", "g",
     "Root mean squared error, in the target's units. Penalizes large errors "
     "more than small ones; sensitive to outliers."),
    ("MAE", "mae", "g",
     "Mean absolute error, in the target's units. The typical size of a "
     "prediction error, treating all errors equally."),
    ("MedAE", "medae", "g",
     "Median absolute error, in the target's units. Like MAE but uses the "
     "median, so it ignores a few extreme errors."),
    ("MAPE", "mape", "g",
     "Mean absolute percentage error: the average error relative to the "
     "observed value. Inflated when the target is near zero."),
    ("MAXE", "maxe", "g",
     "Maximum absolute error, in the target's units — the single worst "
     "prediction in the set."),
    ("MSE", "mse", "g",
     "Mean squared error, in the target's units squared. The square of RMSE; "
     "strongly dominated by the largest errors."),
]

#: Column header -> tooltip for the score table.
_SCORE_COL_TIPS = {
    "Held-out test":
        "Computed on a random hold-out split (default 25% of complete records) "
        "the model never saw during training — the honest estimate of "
        "gap-filling skill.",
    "In-sample":
        "Computed on all complete records, including those the model trained on. "
        "Optimistically biased (better than real gap-filling) — shown for "
        "reference, not as the true skill.",
}

#: Parameter -> tooltip for the configuration table.
_CONFIG_TIPS = {
    "regressor": "The gap-filling model class used.",
    "target": "The variable whose gaps were filled.",
    "test_size": "Fraction of complete records held out to score the model; the "
                 "rest is used for training.",
    "below_zero": "How predicted negative values were treated (keep / clip to "
                  "zero / set to NaN).",
    "n_estimators": "Number of boosting rounds (trees) in the model.",
    "max_depth": "Maximum depth of each tree — higher captures more feature "
                 "interactions but risks overfitting.",
    "learning_rate": "Step-size shrinkage per boosting round.",
    "random_state": "Reproducibility seed. A fixed value makes runs repeatable.",
    "early_stopping_rounds": "Stop adding trees when the validation score has "
                             "not improved for this many rounds.",
    "n_jobs": "Number of CPU cores used for training.",
}

#: Row label -> tooltip for the gap-fill quality table.
_QUALITY_TIPS = {
    "Records": "Total number of timestamps in the record.",
    "Observed": "Records that were already present (not gap-filled).",
    "Filled (full model)": "Gaps filled by the full model, using all selected "
                           "feature variables (flag = 1).",
    "Filled (fallback)": "Gaps filled by the low-quality fallback model "
                         "(timestamp features only, because a predictor was "
                         "missing). A high count means weak fills (flag = 2).",
    "Coverage after fill": "Share of all records that have a value after "
                           "gap-filling (observed + filled).",
}


def _fmt(val, kind: str) -> str:
    if not isinstance(val, (int, float)) or val != val:  # None / NaN
        return "—"
    return f"{val:.3f}" if kind == "f3" else f"{val:.4g}"


class _Card(QFrame):
    """A titled, rounded white panel — the dashboard's building block."""

    def __init__(self, title: str, subtitle: str = "", info: str = "") -> None:
        super().__init__()
        self.setObjectName("card")
        border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        self.setStyleSheet(
            f"QFrame#card {{ background: #FFFFFF; border: 1px solid {border};"
            f" border-radius: 12px; }}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(16, 13, 16, 14)
        self._v.setSpacing(8)

        head = QLabel(theme.manager.label_text(title))
        hf = theme.manager.tracked_font(head.font())
        hf.setBold(True)
        head.setFont(hf)
        if info:
            hrow = QHBoxLayout()
            hrow.setContentsMargins(0, 0, 0, 0)
            hrow.setSpacing(6)
            hrow.addWidget(head)
            hrow.addWidget(self._info_button(info))
            hrow.addStretch(1)
            self._v.addLayout(hrow)
        else:
            self._v.addWidget(head)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px;")
            self._v.addWidget(sub)

    def body(self) -> QVBoxLayout:
        return self._v

    @staticmethod
    def _info_button(text: str) -> QToolButton:
        """A small circular 'i' button; hover or click reveals the explanation."""
        btn = QToolButton()
        btn.setText("i")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(text)
        btn.setFixedSize(16, 16)
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        btn.setStyleSheet(
            "QToolButton { border: none; border-radius: 8px; background: "
            f"{accent}; color: white; font-style: italic; font-weight: 600; "
            "font-size: 11px; }"
            "QToolButton:hover { background: #2A3942; }"
            + theme.manager.tooltip_qss())
        # Clicking also surfaces the tooltip (so it's discoverable without hover).
        btn.clicked.connect(lambda: QToolTip.showText(QCursor.pos(), text, btn))
        return btn


class GapFillResultsPanel(QScrollArea):
    """Scrollable results dashboard; call ``update(model, target)`` after a run."""

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._host = QWidget()
        self.setWidget(self._host)
        self._reset_layout()
        self.reset("Run gap-filling to populate the results.")

    # --- layout scaffolding -------------------------------------------
    def _reset_layout(self) -> None:
        # Replace the host widget wholesale so a re-run starts from a clean slate
        # (clearing nested layouts/canvases item-by-item is error-prone).
        self._host = QWidget()
        self._root = QVBoxLayout(self._host)
        self._root.setContentsMargins(14, 12, 14, 16)
        self._root.setSpacing(14)
        self.setWidget(self._host)

    def reset(self, message: str) -> None:
        self._reset_layout()
        lbl = QLabel(message)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {_C_MUTED};")
        self._root.addWidget(lbl, stretch=1)

    # --- public update -------------------------------------------------
    def update(self, model, target: str,
               shap_threshold_factor: float | None = None) -> None:
        """Render the full dashboard from a fitted XGBoostTS model. Pass
        ``shap_threshold_factor`` (the k used in ``reduce_features``) so the
        reduction card can show the exact threshold equation + line."""
        self._reset_layout()
        try:
            gdf = model.gapfilling_df_
            observed = gdf[target]
            gapfilled = model.get_gapfilled_target()
            flag = model.get_flag()
        except Exception as err:
            self.reset(f"Could not read results: {err}")
            return

        # Top: all tables in one row — performance, settings, (reduction), quality.
        tables = QHBoxLayout()
        tables.setSpacing(14)
        tables.addWidget(self._scores_card(model), stretch=1)
        tables.addWidget(self._config_card(model, target), stretch=1)
        reduction = self._reduction_card(model, shap_threshold_factor)
        if reduction is not None:
            tables.addWidget(reduction, stretch=1)
        tables.addWidget(self._quality_card(observed, flag), stretch=1)
        self._root.addLayout(tables)

        # Below: all diagnostic + temporal plots in a single row.
        self._root.addLayout(self._plot_row([
            ("Predicted vs. observed", lambda ax: self._plot_scatter(ax, gdf, target)),
            ("Feature importance (SHAP)", lambda ax: self._plot_importance(ax, model)),
            ("Diel cycle (gap-filled, by month)", lambda ax: self._plot_diel(ax, gapfilled)),
            ("Cumulative sum (gap-filled)", lambda ax: self._plot_cumulative(ax, gapfilled)),
        ]))
        self._root.addStretch(1)

    # --- scores --------------------------------------------------------
    def _scores_card(self, model) -> _Card:
        card = _Card(
            "Model performance",
            "Held-out test = honest skill on a random hold-out (the gap-filling "
            "task). In-sample = fit on all complete records (optimistically biased).")
        tt = getattr(model, "scores_traintest_", None) or {}
        ins = getattr(model, "scores_", None) or {}
        table = self._make_table(["Metric", "Held-out test", "In-sample"], len(_METRICS),
                                  col_tips=["The error/fit metric.", *_SCORE_COL_TIPS.values()])
        for r, (label, key, kind, tip) in enumerate(_METRICS):
            self._set_cell(table, r, 0, label, bold=True, align_left=True, tooltip=tip)
            self._set_cell(table, r, 1, _fmt(tt.get(key), kind), tooltip=tip)
            self._set_cell(table, r, 2, _fmt(ins.get(key), kind), tooltip=tip)
        self._fit_table_height(table, len(_METRICS))
        card.body().addWidget(table)
        return card

    # --- configuration -------------------------------------------------
    def _config_card(self, model, target: str) -> _Card:
        card = _Card("Configuration", "The settings that produced this run (reproducibility).")
        rows: list[tuple[str, str]] = [
            ("regressor", type(model).__name__),
            ("target", str(target)),
            ("test_size", str(getattr(model, "test_size", "—"))),
        ]
        below = getattr(model, "below_zero", None)
        if below is not None:
            rows.append(("below_zero", str(below)))
        for k, v in (getattr(model, "kwargs", {}) or {}).items():
            rows.append((str(k), str(v)))
        table = self._make_table(["Parameter", "Value"], len(rows))
        for r, (k, v) in enumerate(rows):
            tip = _CONFIG_TIPS.get(k)
            self._set_cell(table, r, 0, k, bold=True, align_left=True, tooltip=tip)
            self._set_cell(table, r, 1, v, align_left=True, tooltip=tip)
        self._fit_table_height(table, len(rows))
        card.body().addWidget(table)
        return card

    # --- gap-fill quality ----------------------------------------------
    def _quality_card(self, observed: pd.Series, flag: pd.Series) -> _Card:
        card = _Card("Gap-fill quality", "How the filled records break down by model.")
        n_total = int(len(flag))
        n_obs = int((flag == 0).sum())
        n_full = int((flag == 1).sum())
        n_fallback = int((flag == 2).sum())
        n_filled = n_full + n_fallback
        cov = 100.0 * (n_obs + n_filled) / n_total if n_total else 0.0

        def pct(n: int) -> str:
            return f"{n:,}  ({100.0 * n / n_total:.1f}%)" if n_total else f"{n:,}"

        rows = [
            ("Records", f"{n_total:,}"),
            ("Observed", pct(n_obs)),
            ("Filled (full model)", pct(n_full)),
            ("Filled (fallback)", pct(n_fallback)),
            ("Coverage after fill", f"{cov:.1f}%"),
        ]
        table = self._make_table(["", ""], len(rows))
        table.horizontalHeader().setVisible(False)
        for r, (k, v) in enumerate(rows):
            tip = _QUALITY_TIPS.get(k)
            self._set_cell(table, r, 0, k, bold=True, align_left=True, tooltip=tip)
            self._set_cell(table, r, 1, v, align_left=True, tooltip=tip)
        self._fit_table_height(table, len(rows), header=False)
        card.body().addWidget(table)
        return card

    # --- feature reduction (only if it was run) ------------------------
    def _reduction_card(self, model, factor: float | None) -> _Card | None:
        try:
            red = getattr(model, "feature_importances_reduction_", None)
            accepted = set(getattr(model, "accepted_features_", []) or [])
            rejected = set(getattr(model, "rejected_features_", []) or [])
        except Exception:
            return None
        if red is None or red.empty or not (accepted or rejected):
            return None
        random_col = getattr(model, "random_col", ".RANDOM")

        # Exact threshold = random_importance + k * random_SD (library equation,
        # MlRegressorGapFillingBase._remove_rejected_features).
        threshold = None
        equation = (
            "A feature is KEPT when its mean |SHAP| importance exceeds a noise "
            "threshold, otherwise it is DROPPED before the final model.\n\n"
            "Threshold equation:\n"
            "    threshold = random + k × SD\n\n"
            "where 'random' is the mean |SHAP| of an added random benchmark "
            f"variable ('{random_col}'), 'SD' is that benchmark's SHAP standard "
            "deviation, and k is the SHAP threshold factor.")
        if factor is not None and random_col in red.index:
            r_imp = float(red.loc[random_col, "SHAP_IMPORTANCE"])
            r_sd = float(red.loc[random_col, "SHAP_SD"]) if "SHAP_SD" in red.columns else 0.0
            threshold = r_imp + factor * r_sd
            equation += (
                f"\n\nThis run:\n    threshold = {r_imp:.4f} + {factor:g} × "
                f"{r_sd:.4f} = {threshold:.4f}")

        card = _Card(
            "Feature reduction (SHAP)",
            "Importance vs. a random-baseline threshold; green = kept, "
            "grey = dropped, amber = the random benchmark.",
            info=equation)

        ser = red["SHAP_IMPORTANCE"].sort_values(ascending=False)
        # Sorted descending, so all accepted rows precede the threshold and all
        # dropped rows follow it — insert one marker row at that boundary.
        names = list(ser.index)
        boundary = sum(1 for n in names if n in accepted)
        n_rows = len(ser) + 1  # + the threshold marker row
        table = self._make_table(["Feature", "mean |SHAP|", "Verdict"], n_rows)

        r = 0
        for i, (name, val) in enumerate(ser.items()):
            if i == boundary:
                self._add_threshold_row(table, r, threshold)
                r += 1
            if name == random_col:
                color, verdict = _C_BENCH, "benchmark"
                tip = ("The random benchmark variable. Its importance is pure "
                       "noise and sets the floor the threshold is built from.")
            elif name in accepted:
                color, verdict = _C_ACCEPT, "kept"
                tip = "Kept: importance above the threshold."
            else:
                color, verdict = _C_REJECT, "dropped"
                tip = "Dropped: importance at or below the threshold."
            self._set_cell(table, r, 0, str(name), align_left=True, tooltip=tip, color=color)
            self._set_cell(table, r, 1, f"{val:.4f}", tooltip=tip, color=color)
            self._set_cell(table, r, 2, verdict, color=color, tooltip=tip)
            r += 1
        self._fit_table_height(table, n_rows, cap=16)
        card.body().addWidget(table)
        return card

    def _add_threshold_row(self, table: QTableWidget, r: int, threshold: float | None) -> None:
        """A full-width dashed red divider marking the keep/drop threshold."""
        table.setSpan(r, 0, 1, table.columnCount())
        text = (f"threshold = {threshold:.4f}" if threshold is not None
                else "keep / drop threshold")
        line = QLabel(text)
        line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line.setStyleSheet(
            f"QLabel {{ border-top: 1px dashed {_C_THRESHOLD}; color: {_C_THRESHOLD}; "
            "font-size: 10px; font-weight: 600; padding-top: 3px; "
            "background: transparent; }")
        table.setCellWidget(r, 0, line)

    # --- plots ---------------------------------------------------------
    def _plot_row(self, specs: list[tuple]) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(14)
        for title, draw in specs:
            card = _Card(title)
            canvas = MplCanvas(show_toolbar=False)
            canvas.setMinimumHeight(300)
            try:
                draw(canvas)
            except Exception as err:  # a plot must never break the dashboard
                ax = canvas.new_axes(1)[0]
                ax.text(0.5, 0.5, f"Plot failed:\n{err}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#B23B3B")
            canvas.draw()
            card.body().addWidget(canvas, stretch=1)
            row.addWidget(card, stretch=1)
        return row

    @staticmethod
    def _style() -> "dv.plotting.FormatStyle":
        return dv.plotting.FormatStyle(ticks_fontsize=_PLOT_FONT,
                                       axlabel_fontsize=_PLOT_FONT,
                                       title_fontsize=_PLOT_FONT + 1)

    def _plot_scatter(self, canvas, gdf: pd.DataFrame, target: str) -> None:
        obs = gdf[target].dropna()
        pred = gdf[".PREDICTIONS"].reindex(obs.index)
        valid = pred.notna()
        obs, pred = obs[valid], pred[valid]
        ax = canvas.new_axes(1)[0]
        dv.plotting.ScatterXY(x=obs, y=pred, nbins=0).plot(
            ax=ax, format_style=self._style(), show_colorbar=False,
            markersize=6, alpha=0.25)
        # 1:1 reference line across the shared data range.
        if len(obs):
            lo = float(min(obs.min(), pred.min()))
            hi = float(max(obs.max(), pred.max()))
            ax.plot([lo, hi], [lo, hi], color="#455A64", lw=1.0, ls="--", zorder=5)
        ax.set_xlabel("Observed", fontsize=_PLOT_FONT)
        ax.set_ylabel("Predicted", fontsize=_PLOT_FONT)
        ax.set_title("")

    def _plot_importance(self, canvas, model) -> None:
        ax = canvas.new_axes(1)[0]
        model.plot_feature_importances(ax=ax, max_features=12, title="")
        ax.tick_params(labelsize=_PLOT_FONT)
        ax.xaxis.label.set_size(_PLOT_FONT)
        ax.yaxis.label.set_size(_PLOT_FONT)

    def _plot_diel(self, canvas, gapfilled: pd.Series) -> None:
        ax = canvas.new_axes(1)[0]
        dv.plotting.DielCycle(gapfilled).plot(
            ax=ax, format_style=dv.plotting.FormatStyle(
                show_legend=False, ticks_fontsize=_PLOT_FONT, axlabel_fontsize=_PLOT_FONT),
            each_month=True, linewidth=1.1)
        ax.axhline(0, color="#90A4AE", linestyle="--", linewidth=0.9)
        ax.set_title("")

    def _plot_cumulative(self, canvas, gapfilled: pd.Series) -> None:
        ax = canvas.new_axes(1)[0]
        dv.plotting.Cumulative(df=gapfilled.to_frame()).plot(
            ax=ax, format_style=self._style(), showplot=False, show_title=False, fill=True)
        ax.set_title("")

    # --- table helpers -------------------------------------------------
    def _make_table(self, headers: list[str], rows: int,
                    col_tips: list[str] | None = None) -> QTableWidget:
        table = QTableWidget(rows, len(headers))
        table.setHorizontalHeaderLabels(headers)
        if col_tips:
            for c, tip in enumerate(col_tips):
                if c < len(headers) and tip:
                    table.horizontalHeaderItem(c).setToolTip(tip)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setShowGrid(False)
        table.setWordWrap(False)
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(headers)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        # NB: no `color:` on `::item` — a stylesheet colour there overrides
        # per-item setForeground (Qt gotcha), killing the green/grey/amber rows.
        # Append tooltip_qss so the table's own stylesheet doesn't detach its
        # tooltips to the dark, low-contrast system default (GUI gotcha).
        table.setStyleSheet(
            "QTableWidget { background: transparent; border: none; color: #263238; }"
            "QTableWidget::item { padding: 4px 8px; }"
            "QHeaderView::section { background: #F5F6F7; color: #6B7780; "
            "border: none; padding: 5px 8px; font-weight: 600; }"
            + theme.manager.tooltip_qss())
        return table

    @staticmethod
    def _set_cell(table: QTableWidget, r: int, c: int, text: str, *,
                  bold: bool = False, align_left: bool = False,
                  color: str | None = None, tooltip: str | None = None) -> None:
        item = QTableWidgetItem(text)
        item.setToolTip(tooltip if tooltip else text)
        if bold:
            f = item.font(); f.setBold(True); item.setFont(f)
        if color:
            item.setForeground(QColor(color))
        align = Qt.AlignmentFlag.AlignVCenter
        align |= (Qt.AlignmentFlag.AlignLeft if align_left else Qt.AlignmentFlag.AlignRight)
        item.setTextAlignment(int(align))
        table.setItem(r, c, item)

    @staticmethod
    def _fit_table_height(table: QTableWidget, rows: int, *, header: bool = True,
                          cap: int | None = None) -> None:
        """Size the table to its content so cards hug the rows (no inner scroll)."""
        row_h = 28
        shown = min(rows, cap) if cap else rows
        head_h = table.horizontalHeader().height() if header else 0
        if cap and rows > cap:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            h = head_h + shown * row_h + 4
        else:
            table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            h = head_h + rows * row_h + 4
        table.verticalHeader().setDefaultSectionSize(row_h)
        table.setFixedHeight(h)
