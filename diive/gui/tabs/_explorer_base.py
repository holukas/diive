"""
GUI.TABS._EXPLORER_BASE: TEMPLATE FOR SINGLE-VARIABLE ANALYSIS TABS
===================================================================

The shared skeleton for the analysis "explorer" tabs that pick ONE variable on
the left and compute/render a view of it on the right (Driver explorer, Gaps &
coverage, Seasonal trend & anomalies, Spectrogram, 3D surface). Before this each
tab re-derived the same left/right split, the same select -> busy-indicator ->
compute flow, the same default-variable pick, and (for several) the same stats
strip.

A concrete tab subclasses :class:`SingleVariableExplorerTab` and supplies only
the parts that differ:

  * ``_build_right()`` — the right-hand widget (its own controls + canvas/table),
  * ``_compute()`` — read the selected variable from ``self._df[self._target]``,
    call the library, store results, render (runs via ``run_with_loading``),
  * optionally a preferred default (``default_var`` / ``_default_variable``) and
    extra per-tab state (``_init_state``).

The stats strip (a horizontal band of :class:`_StatCard`s) is offered as opt-in
helpers — ``_build_stats_strip()`` builds it and ``_set_stat_cards()`` fills it —
for the tabs that show one. All computation is library work; this template only
collects the selection, defers the compute behind the busy indicator, and lays
out the panels (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import _StatCard
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle


class SingleVariableExplorerTab(DiiveTab):
    """Base for analysis tabs that explore one selected variable at a time."""

    #: Preferred default variable to select on load (else the first numeric column).
    default_var: str | None = None
    #: Only consider numeric columns for the default selection (and reject a
    #: non-numeric ``default_var``).
    default_numeric_only = True
    #: Optional bold header above the variable list (matching the outlier /
    #: correction tabs). When None (default) the list has no header.
    list_title: str | None = None
    #: Muted parenthetical hint next to ``list_title``.
    list_hint = "click to select"

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._target: str | None = None
        self._init_state()

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Shared tab title bar (matches every other result/analysis tab); a
        # Copy-Python button is added only when the subclass provides codegen.
        actions = []
        if type(self)._python_code is not SingleVariableExplorerTab._python_code:
            self.copy_btn = CopyPythonButton(self._python_code)
            self.copy_btn.setToolTip(
                "Copy a runnable diive script reproducing this view.")
            actions.append(self.copy_btn)
        outer.addLayout(build_titlebar(self.title, *actions))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)
        # Optional list header (the varpanel keeps its own fixed width, so the
        # wrapper column sizes to it); without a title the panel goes in bare.
        if self.list_title:
            left = QWidget()
            ll = QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            ll.setSpacing(6)
            ll.addWidget(list_header(self.list_title, self.list_hint))
            ll.addWidget(self.varpanel, stretch=1)
            splitter.addWidget(left)
        else:
            splitter.addWidget(self.varpanel)
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lock_panel_handle(splitter)  # fixed-width list → no misleading ↔ cursor
        outer.addWidget(splitter, stretch=1)  # body absorbs all extra vertical space
        return root

    # --- subclass hooks ------------------------------------------------
    def _python_code(self) -> str | None:
        """Runnable snippet reproducing this view (subclass hook). Returning a
        non-default override adds a Copy-Python button to the title bar; the
        string itself must come from a library codegen function (the GUI never
        builds the script — strict GUI<->library separation). Default None."""
        return None

    def _init_state(self) -> None:
        """Initialise extra per-tab state attributes (subclass hook). Runs at the
        start of ``build`` before any widgets are created."""

    def _build_right(self) -> QWidget:
        """Build and return the right-hand widget — the tab's own controls and
        canvas/table (subclass hook)."""
        raise NotImplementedError

    def _compute(self) -> None:
        """Read ``self._df[self._target]``, call the library, store results and
        render (subclass hook). Runs behind the variable-panel busy indicator."""
        raise NotImplementedError

    def _default_variable(self, df) -> str | None:
        """Variable to auto-select on load. Default: ``default_var`` if present
        (and numeric when ``default_numeric_only``), else the first numeric (or
        first) column. Override for a different heuristic (e.g. the gappiest)."""
        cols = [str(c) for c in df.columns]
        numeric = [str(c) for c in df.select_dtypes(include="number").columns]
        if self.default_var and self.default_var in cols and (
                not self.default_numeric_only or self.default_var in numeric):
            return self.default_var
        if self.default_numeric_only:
            return numeric[0] if numeric else None
        return cols[0] if cols else None

    # --- data flow -----------------------------------------------------
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        default = self._default_variable(df)
        if default is not None:
            self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self.varpanel.run_with_loading(name, self._compute)

    def _recompute(self) -> None:
        """Re-run :meth:`_compute` on the current target (for Update/Rank-style
        buttons whose settings apply on click rather than on selection)."""
        if self._target is not None and self._df is not None:
            self.varpanel.run_with_loading(self._target, self._compute)

    # --- stats strip (opt-in) ------------------------------------------
    def _build_stats_strip(self) -> QWidget:
        """A horizontal KPI band above the body; fill it with :meth:`_set_stat_cards`.
        Sets ``self.stats_layout`` (the host layout, keeping a trailing stretch)."""
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

    def _set_stat_cards(self, cards: list[tuple[str, str]]) -> None:
        """Rebuild the stats strip from ``[(label, value), ...]`` (keeps the
        trailing stretch)."""
        while self.stats_layout.count() > 1:
            item = self.stats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (name, value) in enumerate(cards):
            self.stats_layout.insertWidget(i, _StatCard(name, value))
