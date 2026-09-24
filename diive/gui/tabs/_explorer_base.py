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
  * how to compute and render, in one of two ways:

    - synchronous: ``_compute()`` reads ``self._df[self._target]``, calls the
      library, stores results and renders, all on the GUI thread (deferred one
      tick by ``run_with_loading`` so the busy cue paints first);
    - on a worker thread, for computations slow enough to freeze the window:
      ``_compute_request()`` snapshots the inputs on the GUI thread,
      ``_compute_payload(*request)`` (a pure staticmethod: no Qt, no
      matplotlib) runs the library on a :class:`LatestRunner`, and
      ``_render_payload(payload)`` draws the result on the GUI thread.
      Overriding ``_compute_payload`` opts in. Only the newest request's
      result is drawn; a failure goes to ``_render_error``.

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
import shiboken6
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
from diive.gui.widgets.worker import LatestRunner


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
    #: Make the variable list a drag source (drag a name onto a drop target,
    #: e.g. the X/Y/Z fields of the coordinate-surface tab). Off by default.
    list_draggable = False

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._df: pd.DataFrame | None = None
        self._target: str | None = None
        self._init_state()

        root = QWidget()
        self._root = root
        if self._uses_worker():
            self._runner = LatestRunner()
            self._runner.done.connect(self._on_payload)
            self._runner.failed.connect(self._on_payload_failed)
            self._runner.settled.connect(self._end_busy)
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
        self.varpanel = VariablePanel(draggable=self.list_draggable)
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
        render (subclass hook). Runs behind the variable-panel busy indicator.
        Not used by a tab that overrides ``_compute_payload``."""
        raise NotImplementedError

    def _compute_request(self) -> tuple:
        """Snapshot everything ``_compute_payload`` needs (subclass hook, worker
        path). Runs on the GUI thread, so it may read widgets; the worker must
        not. Returns the positional arguments for ``_compute_payload``."""
        raise NotImplementedError

    @staticmethod
    def _compute_payload(*request):
        """Run the library on a worker thread and return the result (subclass
        hook, worker path). Overriding it moves the tab's compute off the GUI
        thread. Must be pure: no Qt, no matplotlib, no ``self``."""
        raise NotImplementedError

    def _render_payload(self, payload) -> None:
        """Store and draw the newest result on the GUI thread (subclass hook,
        worker path)."""
        raise NotImplementedError

    def _render_error(self, message: str) -> None:
        """Show a worker failure on the canvas (worker path). A tab without a
        ``canvas`` overrides this."""
        self.canvas.show_message(f"Cannot compute:\n{message}")

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
        if self._uses_worker():
            # A result still computing belongs to the old frame: never draw it.
            self._runner.cancel()
        self.varpanel.set_variables(df.columns, created)
        default = self._default_variable(df)
        if default is not None:
            self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self._start_compute()

    def _recompute(self) -> None:
        """Re-run the compute on the current target (for Update/Rank-style
        buttons whose settings apply on click rather than on selection)."""
        if self._target is not None and self._df is not None:
            self._start_compute()

    # --- compute dispatch ----------------------------------------------
    def _uses_worker(self) -> bool:
        return (type(self)._compute_payload
                is not SingleVariableExplorerTab._compute_payload)

    def _computing(self) -> bool:
        """True while a worker compute is running or queued. The shown result
        is then about to be replaced (and may belong to another variable), so
        view-only controls skip re-rendering it."""
        return self._uses_worker() and self._runner.is_busy

    def _start_compute(self) -> None:
        if not self._uses_worker():
            self.varpanel.run_with_loading(self._target, self._compute)
            return
        self.varpanel.set_loading(self._target)
        # A widget cursor, not the app-wide override: the window stays usable
        # during the run, and a cursor on a deleted widget cannot get stuck.
        self._root.setCursor(Qt.CursorShape.BusyCursor)
        self._runner.submit(self._compute_payload, *self._compute_request())

    def _alive(self) -> bool:
        # A tab closed mid-run has its widgets deleted (`deleteLater`) while the
        # job's outcome is still queued for delivery.
        return shiboken6.isValid(self._root)

    def _end_busy(self) -> None:
        if self._alive():
            self.varpanel.clear_loading()
            self._root.unsetCursor()

    def _on_payload(self, payload) -> None:
        if self._alive():
            self._render_payload(payload)

    def _on_payload_failed(self, message: str) -> None:
        if self._alive():
            self._render_error(message)

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
