"""
GUI.WIDGETS.MDS_RESULTS: MDS GAP-FILLING RESULTS DASHBOARD
=========================================================

The "Results" page of the MDS gap-filling tab — a slimmed, card-based dashboard
tailored to ``dv.gapfilling.FluxMDS``. MDS is not an ML regressor, so it has no
held-out test split, no SHAP feature importances and no feature reduction; the
panel instead surfaces what MDS *does* produce: the configuration, the in-sample
fit scores, the per-quality-level gap-fill breakdown (``model.quality_breakdown()``)
and a quality-level bar plot, plus a predicted-vs-observed scatter and the
gap-filled cumulative sum.

All numbers/tables come straight from the fitted library model
(``scores_``, ``gapfilling_df_``, ``quality_breakdown``); this widget only arranges
and styles them — strict GUI<->library separation. The card/table styling is shared
with the ML dashboard (:mod:`gapfill_results`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.widgets.gapfill_results import _C_MUTED, _Card
from diive.gui.widgets.mpl_canvas import MplCanvas

_PLOT_FONT = 8

#: In-sample MDS score metrics in display order: (label, key, tooltip).
_METRICS = [
    ("R²", "r2", "Coefficient of determination on the complete (observed) records — "
                 "1 = perfect. In-sample only (MDS has no held-out test split)."),
    ("RMSE", "rmse", "Root mean squared error on the observed records, in target units."),
    ("MAE", "mae", "Mean absolute error on the observed records, in target units."),
    ("MedAE", "medae", "Median absolute error on the observed records, in target units."),
    ("MAPE", "mape", "Mean absolute percentage error (inflated near zero target)."),
    ("MAXE", "maxe", "Maximum absolute error on the observed records, in target units."),
]

#: Score interpretation note: MDS scores are computed by predicting the *observed*
#: records (which is all MDS can compare against — there is no withheld test set).
_SCORES_SUBTITLE = (
    "In-sample fit: MDS predictions vs. the observed values it could compare "
    "against. MDS has no held-out test split, so there is no separate test score.")


def _fmt(val, kind: str = "g") -> str:
    if not isinstance(val, (int, float)) or val != val:  # None / NaN
        return "—"
    return f"{val:.3f}" if kind == "f3" else f"{val:.4g}"


class MdsResultsPanel(QScrollArea):
    """Scrollable MDS results dashboard; call ``update(model, target)`` after a run."""

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
        # Replace the host widget wholesale so a re-run starts from a clean slate.
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
    def update(self, model, target: str) -> None:
        """Render the dashboard from a fitted FluxMDS model."""
        self._reset_layout()
        try:
            gdf = model.gapfilling_df_
            gapfilled = model.get_gapfilled_target()
            breakdown = model.quality_breakdown()
            scores = dict(model.scores_)
        except Exception as err:
            self.reset(f"Could not read results: {err}")
            return

        # Top: tables — configuration + in-sample scores hug their content on the
        # left (stretch 0), the quality breakdown takes the rest. Align the cards
        # to the top so shorter cards don't get stretched to the tallest's height.
        tables = QHBoxLayout()
        tables.setSpacing(14)
        tables.addWidget(self._config_card(model, target), 0, Qt.AlignmentFlag.AlignTop)
        tables.addWidget(self._scores_card(scores), 0, Qt.AlignmentFlag.AlignTop)
        tables.addWidget(self._quality_card(breakdown), 1)
        self._root.addLayout(tables)

        # Full-width: the gap-filled series coloured + markered by quality level.
        self._root.addWidget(self._timeseries_card(model))

        # Below: the quality-level plot + diagnostic/temporal plots.
        self._root.addLayout(self._plot_row([
            ("Gap fills by quality level", lambda c: self._plot_quality(c, breakdown)),
            ("Predicted vs. observed", lambda c: self._plot_scatter(c, gdf, target)),
            ("Cumulative sum (gap-filled)", lambda c: self._plot_cumulative(c, gapfilled)),
        ]))
        self._root.addStretch(1)

    # --- quality time series (full width) ------------------------------
    def _timeseries_card(self, model) -> _Card:
        card = _Card(
            "Gap-filled series by quality level",
            "Each point coloured + markered by its MDS quality level (0 = measured, "
            "higher = looser match); whiskers show the mean ± SD of the filled points.")
        # Keep the navigation toolbar here (pan / box-zoom / reset / save): this
        # full-width plot has room for it, and zooming into a period is useful.
        canvas = MplCanvas(show_toolbar=True)
        # Taller than the other plots: the per-quality legend sits below the axes,
        # plus the toolbar row at the bottom.
        canvas.setMinimumHeight(470)
        try:
            ax = canvas.new_axes(1)[0]
            model.plot_quality_timeseries(
                ax=ax, legend=True, ax_labels_fontsize=_PLOT_FONT,
                legend_textsize=_PLOT_FONT - 1)
            ax.set_title("")
        except Exception as err:  # a plot must never break the dashboard
            ax = canvas.new_axes(1)[0]
            ax.text(0.5, 0.5, f"Plot failed:\n{err}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#B23B3B")
        canvas.draw()
        card.body().addWidget(canvas)
        return card

    # --- configuration -------------------------------------------------
    def _config_card(self, model, target: str) -> _Card:
        card = _Card("Configuration", "The settings that produced this run (reproducibility).")
        rows = [
            ("method", "FluxMDS"),
            ("target (flux)", str(target)),
            ("SWIN driver", str(model.swin)),
            ("TA driver", str(model.ta)),
            ("VPD driver", str(model.vpd)),
            ("swin_tol", str(list(model.swin_tol))),
            ("ta_tol", f"{model.ta_tol} °C"),
            ("vpd_tol", f"{model.vpd_tol} kPa"),
            ("avg_min_n_vals", str(model.avg_min_n_vals)),
            ("sym_mean", str(model.sym_mean)),
        ]
        # No stretch: label + value hug their content and cluster left.
        table = self._make_table(["Parameter", "Value"], len(rows), stretch_col=None)
        for r, (k, v) in enumerate(rows):
            self._set_cell(table, r, 0, k, bold=True, align_left=True)
            self._set_cell(table, r, 1, v, align_left=True)
        self._fit_table_height(table, len(rows))
        card.body().addWidget(table)
        return self._compact_card(card, table)

    # --- scores --------------------------------------------------------
    def _scores_card(self, scores: dict) -> _Card:
        # Short subtitle so the narrow (content-hugging) card stays compact; the
        # full explanation lives in the column tooltip.
        card = _Card("Model performance", "In-sample fit (no held-out test split).")
        # No stretch: metric + value hug their content and cluster left.
        table = self._make_table(["Metric", "In-sample"], len(_METRICS),
                                 col_tips=["The error/fit metric.", _SCORES_SUBTITLE],
                                 stretch_col=None)
        for r, (label, key, tip) in enumerate(_METRICS):
            kind = "f3" if key == "r2" else "g"
            self._set_cell(table, r, 0, label, bold=True, align_left=True, tooltip=tip)
            self._set_cell(table, r, 1, _fmt(scores.get(key), kind), align_left=True, tooltip=tip)
        self._fit_table_height(table, len(_METRICS))
        card.body().addWidget(table)
        return self._compact_card(card, table)

    # --- quality breakdown ---------------------------------------------
    def _quality_card(self, breakdown: pd.DataFrame) -> _Card:
        card = _Card(
            "Gap-fill quality (MDS flags)",
            "How the records break down by gap-fill flag. Flag 0 = observed; "
            "non-zero = gap-filled (method*1000 + window), higher = looser "
            "meteorological match (Reichstein et al. 2005 / ONEFlux).")
        rows = len(breakdown)
        # Stretch the description column so Flag/Records/% stay compact on the
        # left instead of the flag number floating far from its header.
        table = self._make_table(["Flag", "Records", "%", "Match window"], rows,
                                  stretch_col=3)
        for r, row in enumerate(breakdown.itertuples(index=False)):
            obs = row.level == 0
            tip = "Observed (measured) records." if obs else \
                f"Gap-filled, flag {row.level}: {row.description}."
            self._set_cell(table, r, 0, str(row.level), bold=True, tooltip=tip,
                           color="#455A64" if obs else None)
            self._set_cell(table, r, 1, f"{row.count:,}", tooltip=tip)
            self._set_cell(table, r, 2, f"{row.pct:.1f}%", tooltip=tip)
            self._set_cell(table, r, 3, row.description, align_left=True, tooltip=tip)
        self._fit_table_height(table, rows, cap=14)
        card.body().addWidget(table)
        card.body().addStretch(1)  # top-pack content; no distributed vertical gaps
        return card

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

    def _plot_quality(self, canvas, breakdown: pd.DataFrame) -> None:
        """Bar chart of records per quality level — observed (grey) vs the
        gap-fill tiers (blue gradient, darker = looser match)."""
        ax = canvas.new_axes(1)[0]
        if breakdown.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        levels = breakdown["level"].tolist()
        counts = breakdown["count"].tolist()
        from matplotlib import colormaps
        blues = colormaps["Blues"]
        gap_levels = [lv for lv in levels if lv > 0]
        maxlv = max(gap_levels) if gap_levels else 1
        colors = []
        for lv in levels:
            if lv == 0:
                colors.append("#90A4AE")  # grey — observed
            else:
                # darker blue for higher (looser) levels.
                colors.append(blues(0.35 + 0.6 * (lv / maxlv)))
        positions = list(range(len(levels)))
        bars = ax.bar(positions, counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(lv) for lv in levels], fontsize=_PLOT_FONT)
        ax.set_xlabel("Gap-fill flag (0 = observed, higher = looser match)", fontsize=_PLOT_FONT)
        ax.set_ylabel("Records", fontsize=_PLOT_FONT)
        ax.tick_params(labelsize=_PLOT_FONT)
        # Value labels on top of each bar.
        for bar, cnt in zip(bars, counts):
            ax.annotate(f"{cnt:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=_PLOT_FONT - 1, color="#455A64",
                        xytext=(0, 1), textcoords="offset points")
        ax.margins(y=0.15)

    def _plot_scatter(self, canvas, gdf: pd.DataFrame, target: str) -> None:
        obs = gdf[target].dropna()
        pred = gdf[".PREDICTIONS"].reindex(obs.index)
        valid = pred.notna()
        obs, pred = obs[valid], pred[valid]
        ax = canvas.new_axes(1)[0]
        dv.plotting.ScatterXY(x=obs, y=pred, nbins=0).plot(
            ax=ax, format_style=self._style(), show_colorbar=False,
            markersize=6, alpha=0.25)
        if len(obs):
            lo = float(min(obs.min(), pred.min()))
            hi = float(max(obs.max(), pred.max()))
            ax.plot([lo, hi], [lo, hi], color="#455A64", lw=1.0, ls="--", zorder=5)
        ax.set_xlabel("Observed", fontsize=_PLOT_FONT)
        ax.set_ylabel("Predicted (MDS)", fontsize=_PLOT_FONT)
        ax.set_title("")

    def _plot_cumulative(self, canvas, gapfilled: pd.Series) -> None:
        ax = canvas.new_axes(1)[0]
        dv.plotting.Cumulative(df=gapfilled.to_frame()).plot(
            ax=ax, format_style=self._style(), showplot=False, show_title=False, fill=True)
        ax.set_title("")

    # --- table helpers (mirror gapfill_results) ------------------------
    def _make_table(self, headers: list[str], rows: int,
                    col_tips: list[str] | None = None,
                    stretch_col: int | None = 0) -> QTableWidget:
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
        # `stretch_col=None`: no column stretches, so every column hugs its content
        # and they cluster on the left (key->value tables — else the lone value
        # column gets shoved to the far card edge with a big gap). Otherwise that
        # one column absorbs the slack and the rest hug their content.
        hh.setStretchLastSection(False)
        for c in range(len(headers)):
            hh.setSectionResizeMode(
                c, QHeaderView.ResizeMode.Stretch if c == stretch_col
                else QHeaderView.ResizeMode.ResizeToContents)
        # Left-align the stretched column's header so it sits above its (left-
        # aligned) text rather than centred in the wide column.
        hdr = table.horizontalHeaderItem(stretch_col) if stretch_col is not None else None
        if hdr is not None:
            hdr.setTextAlignment(int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
        # NB: no `color:` on `::item` — that would override per-item setForeground
        # (Qt gotcha). Append tooltip_qss so the table keeps the light tooltip look.
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
        from PySide6.QtGui import QColor
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

    @staticmethod
    def _fit_table_width(table: QTableWidget) -> int:
        """Pin the table to the summed width of its (content-sized) columns so it
        hugs its data instead of stretching — the source of the wide empty gap in
        the key->value cards. Returns the pinned width."""
        table.resizeColumnsToContents()
        w = table.horizontalHeader().length() + table.frameWidth() * 2 + 2
        table.setFixedWidth(w)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return w

    def _compact_card(self, card: "_Card", table: QTableWidget) -> "_Card":
        """Make a key->value card hug its content: pin the table width, cap the
        card so it can't sprawl, and top-pack the content (no distributed vertical
        gaps). Returns the card."""
        w_table = self._fit_table_width(table)
        # The (uppercase, letter-spaced) header can be wider than a narrow table,
        # so size the card to whichever is wider — else the header clips.
        labels = card.findChildren(QLabel)
        header_w = max((lbl.sizeHint().width() for lbl in labels if lbl.font().bold()),
                       default=0)
        content_w = max(w_table, header_w + 4)  # +4: never clip the header's edge
        # Let the subtitle wrap within the content width instead of stretching the
        # card to its one-line length (the original sprawl).
        for lbl in labels:
            if not lbl.font().bold():
                lbl.setMaximumWidth(content_w)
        card.body().addStretch(1)                 # push content up; no mid-card gaps
        card.setMaximumWidth(content_w + 34)      # + the body's L/R margins (16+16)
        return card
