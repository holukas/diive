"""
GUI.TABS._CORRECTION_BASE: SHARED BASE FOR DATA-CORRECTION TABS
==============================================================

Common machinery for the GUI's data-correction tabs (radiation zero offset,
relative-humidity offset, set-to-threshold, set-to-value, set-exact-to-missing):
the variable list, a method **hero chip** (visually kin to the RF/XGB
gap-filling tabs), the two-panel preview (original on top, corrected on the
bottom), a worker thread that runs the library correction, "Add to dataset", and
"Copy Python".

A concrete tab subclasses :class:`BaseCorrectionTab` and fills in only the
correction-specific parts via a small hook surface: the ``corr_key`` (one of the
``CORR_*`` constants), the parameter widgets, and the kwargs they map to. Every
tab routes through the library's :func:`apply_corrections` /
:func:`corrections_to_code` — the single key -> function mapping — so the GUI
never re-encodes the correction logic and the copied script stays in lockstep.
Each correction is its own tab, so all corrections are independently available;
the measurement is a suggestion (shown as a hint), never a lock.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

import pandas as pd
from matplotlib.lines import Line2D
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, MODIFIED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import _chip, _MetricSlot, _stat_separator
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel
from diive.preprocessing.corrections.apply import apply_corrections
from diive.preprocessing.corrections.codegen import corrections_to_code

# --- Correction preview palette (shared with the outlier tabs' look) --------
_C_RAW = "#B0BEC5"       # blue-grey 200 — original line / faint reference
_C_CORRECTED = "#43A047"  # green 600    — corrected series (the result)
_C_REMOVED = "#E53935"   # red 600       — points the correction set to missing
_C_MUTED = "#6B7780"     # secondary/help text


class _CorrectionSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""
    run_done = Signal(object)
    run_failed = Signal(str)
    features_created = Signal(object)


class BaseCorrectionTab(DiiveTab):
    """Shared base for data-correction tabs; subclasses supply the method bits.

    Subclasses set the class attributes below and implement the ``_…`` hooks.
    The default hooks make a parameter-less correction (the offset corrections)
    work with no override beyond the class attributes.
    """

    #: Tab title (DiiveTab requirement) — set by the subclass.
    title = "Correction"
    #: One-line description shown above the settings and in the hero.
    intro = "Correct a variable and keep the original alongside a corrected copy."
    #: Title of the method-settings group box.
    settings_title = "Settings"
    #: Suffix for the corrected column name, e.g. "RADOFFSET" -> "{var}_RADOFFSET".
    method_suffix = "CORR"
    #: Correction key routed through ``apply_corrections`` (a ``CORR_*`` constant).
    corr_key: str = ""
    #: Hero chip (method identity), mirroring the RF/XGB gap-filling tabs.
    method_chip_label = "CORRECTION"
    method_chip_bg = "#ECEFF1"
    method_chip_fg = "#37474F"
    #: When True, show a site-coordinates box (needed by the radiation correction
    #: for its day/night split) and pass lat/lon/utc to the library.
    needs_coords = False
    #: Optional one-line hint naming the measurement(s) the correction targets
    #: (a suggestion, not a lock) — shown under the settings.
    suited_for: str | None = None

    # --- build ---
    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._result_df = None   # corrected column, pending "Add"
        self._last_payload = None
        self._sig = _CorrectionSignals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar (XGBoost-gap-filling-style header): a tracked, bold title with
        # Copy Python at the far-right edge.
        titlebar = QHBoxLayout()
        titlebar.setContentsMargins(10, 8, 10, 8)
        title = QLabel(theme.manager.label_text(self.title))
        title.setFont(theme.manager.tracked_font(point_delta=1.0))
        title.setStyleSheet("font-weight: bold;")
        titlebar.addWidget(title)
        titlebar.addStretch(1)
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this correction.")
        titlebar.addWidget(self.copy_btn)
        outer.addLayout(titlebar)

        # Body: target list | settings | hero + preview.
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)

        # Left: pick the variable to correct (the target).
        tcol = QVBoxLayout()
        tcol.addWidget(self._list_header("Target", "click to set target"))
        self.varpanel = VariablePanel()
        self.varpanel.list.setToolTip("Click a variable to set it as the correction target.")
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        tcol.addWidget(self.varpanel, stretch=1)
        layout.addLayout(tcol)

        # Middle: settings + run/add.
        mid = self._build_settings()
        mid.setFixedWidth(300)
        layout.addWidget(mid)

        # Right: hero chip over the original/corrected preview.
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self._build_hero())
        self.canvas = MplCanvas()
        rlay.addWidget(self.canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        outer.addWidget(body, stretch=1)

        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        if self.needs_coords:
            self._seed_site()
            site.manager.changed.connect(self._on_site_changed)
        return root

    @staticmethod
    def _list_header(title: str, hint: str) -> QLabel:
        """A bold list title with a muted parenthetical hint, matching the ML
        gap-filling tabs' 'Target (click to set target)' header."""
        label = QLabel(f"<b>{title}</b> <span style='color:#90A4AE'>({hint})</span>")
        label.setWordWrap(True)
        return label

    def _build_hero(self) -> QWidget:
        """A slim band: the method chip on the left, a stats strip on the right
        (filled in after a run via :meth:`_hero_metrics`). The description lives
        only in the settings panel — not repeated here."""
        hero = QFrame()
        hero.setStyleSheet("QFrame { background: #FAFAFA; border-radius: 8px; }")
        h = QHBoxLayout(hero)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(12)
        chip = _chip(self.method_chip_label, self.method_chip_bg, self.method_chip_fg)
        h.addWidget(chip)
        h.addStretch(1)
        # Metrics strip: rebuilt after each run from _hero_metrics(payload).
        self._metrics_host = QWidget()
        self._metrics_lay = QHBoxLayout(self._metrics_host)
        self._metrics_lay.setContentsMargins(0, 0, 0, 0)
        self._metrics_lay.setSpacing(12)
        h.addWidget(self._metrics_host)
        return hero

    def _clear_hero(self) -> None:
        while self._metrics_lay.count():
            w = self._metrics_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _update_hero(self, payload: dict) -> None:
        """Rebuild the hero stats strip from the subclass's metrics."""
        self._clear_hero()
        for i, (name, value, tip) in enumerate(self._hero_metrics(payload)):
            if i > 0:
                self._metrics_lay.addWidget(_stat_separator())
            slot = _MetricSlot()
            slot.update_metric(name, value, tip)
            self._metrics_lay.addWidget(slot)

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        # No top margin so the "Settings" header top-aligns with the "Target"
        # header (the target column's sub-layout has no top margin either).
        outer.setContentsMargins(0, 0, 0, 0)

        outer.addWidget(self._list_header("Settings", "set parameters & run"))

        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        outer.addWidget(intro)

        # Method settings (subclass rows). Omitted entirely when the subclass has
        # no parameters (the offset corrections) — an empty box looks broken.
        form_box = QGroupBox(self.settings_title)
        form = QFormLayout(form_box)
        before = form.rowCount()
        self._add_method_rows(form)
        if form.rowCount() > before:
            outer.addWidget(form_box)
        else:
            form_box.deleteLater()

        # Optional site coordinates (radiation day/night split).
        if self.needs_coords:
            coord_box = QGroupBox("Site coordinates")
            cf = QFormLayout(coord_box)
            self.lat = QDoubleSpinBox(); self.lat.setRange(-90.0, 90.0); self.lat.setDecimals(4)
            self.lat.setToolTip("Site latitude in decimal degrees (north positive); "
                                "used to split day from night.")
            self.lon = QDoubleSpinBox(); self.lon.setRange(-180.0, 180.0); self.lon.setDecimals(4)
            self.lon.setToolTip("Site longitude in decimal degrees (east positive); "
                                "used to split day from night.")
            self.utc = QSpinBox(); self.utc.setRange(-12, 14)
            self.utc.setToolTip("UTC offset (hours) of the timestamps; used to align "
                                "solar position for the day/night split.")
            cf.addRow("Latitude", self.lat)
            cf.addRow("Longitude", self.lon)
            cf.addRow("UTC offset (h)", self.utc)
            note = QLabel("Coordinates default from Settings ▸ Project settings.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {_C_MUTED};")
            cf.addRow(note)
            outer.addWidget(coord_box)
        else:
            self.lat = self.lon = self.utc = None

        if self.suited_for:
            hint = QLabel(self.suited_for)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {_C_MUTED}; font-style: italic;")
            outer.addWidget(hint)

        self.run_btn = QPushButton("Run correction")
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        self.status = QLabel("Select a variable on the left.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.add_btn = QPushButton("Add corrected to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        return panel

    # --- subclass hooks ---
    def _add_method_rows(self, form: QFormLayout) -> None:
        """Add the correction's parameter rows to ``form`` (subclass hook).
        Default: no parameters (the offset corrections)."""

    def _current_kwargs(self) -> dict:
        """Correction kwargs from the current control values (subclass hook).
        Default: no kwargs."""
        return {}

    def _validate(self, kwargs: dict) -> str | None:
        """Return an error message if the inputs can't produce a real correction
        (subclass hook), else None. Default: always valid."""
        return None

    def _method_controls(self) -> dict:
        """{name: widget} of the subclass's persistable controls (subclass hook)."""
        return {}

    # --- site coords ---
    def _seed_site(self) -> None:
        m = site.manager
        if not m.configured:
            return
        self.lat.setValue(m.latitude)
        self.lon.setValue(m.longitude)
        self.utc.setValue(m.utc_offset)

    def _on_site_changed(self) -> None:
        self._seed_site()

    def _coords(self) -> tuple[float | None, float | None, int | None]:
        if not self.needs_coords:
            return None, None, None
        return self.lat.value(), self.lon.value(), self.utc.value()

    # --- state (save/restore inputs with a project) ---
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        controls = dict(self._method_controls())
        if self.needs_coords:
            controls.update(lat=self.lat, lon=self.lon, utc=self.utc)
        return {"var": self._var, "controls": save_controls(controls)}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        controls = dict(self._method_controls())
        if self.needs_coords:
            controls.update(lat=self.lat, lon=self.lon, utc=self.utc)
        restore_controls(controls, state.get("controls"))
        var = state.get("var")
        if var and self._df is not None and var in self._df.columns:
            self._select(var)  # re-plots the raw series; correction is not auto-run

    # --- data ---
    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self._result_df = None
        self.varpanel.set_variables(df.columns, created)
        self.add_btn.setEnabled(False)
        if self._var is not None and self._var not in df.columns:
            self._var = None

    def _select(self, name: str) -> None:
        if not name or self._df is None:
            return
        self._var = name
        self._result_df = None
        self.varpanel.set_panels([name])
        self.add_btn.setEnabled(False)
        self._clear_hero()
        self.status.setText(f"Selected '{name}'. Set parameters and apply.")
        self.varpanel.run_with_loading(name, lambda: self._draw(self._df[name]))

    # --- run ---
    def _corrections(self, kwargs: dict) -> list[dict]:
        """The single-correction chain this tab applies."""
        return [{"key": self.corr_key, "kwargs": kwargs}]

    def _python_code(self) -> str | None:
        if self._var is None:
            self.status.setText("Select a variable first to copy its code.")
            return None
        kwargs = self._current_kwargs()
        lat, lon, utc = self._coords()
        out_var = f"{self._var}_corrected"
        snippet = corrections_to_code(
            self._corrections(kwargs),
            site_lat=lat if lat is not None else 0.0,
            site_lon=lon if lon is not None else 0.0,
            utc_offset=utc if utc is not None else 0,
            in_var=str(self._var), out_var=out_var)
        return snippet or None

    def _run(self) -> None:
        if self._df is None or self._var is None:
            self.status.setText("Select a variable on the left first.")
            return
        kwargs = self._current_kwargs()
        err = self._validate(kwargs)
        if err:
            self.status.setText(err)
            return
        series = self._df[self._var]
        lat, lon, utc = self._coords()
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.status.setText("Applying correction...")
        threading.Thread(
            target=self._worker, args=(series, kwargs, lat, lon, utc),
            daemon=True).start()

    def _apply(self, series, kwargs: dict, coords: tuple):
        """Compute the corrected series; return ``(corrected, extra)``.

        Default routes through the library ``apply_corrections`` with no extra
        diagnostics. Subclasses with richer output (e.g. the radiation tab)
        override this to run a diagnostics variant once and return a dict of
        intermediate series + stats consumed by :meth:`_hero_metrics` /
        :meth:`_render_result` (subclass hook)."""
        lat, lon, utc = coords
        corrected = apply_corrections(
            series, self._corrections(kwargs),
            lat=lat, lon=lon, utc_offset=utc, showplot=False)
        return corrected, {}

    def _worker(self, series, kwargs: dict, lat, lon, utc) -> None:
        try:
            corrected, extra = self._apply(series, kwargs, (lat, lon, utc))
            corrected = corrected.copy()
            corrected.name = f"{series.name}_{self.method_suffix}"
            result = pd.DataFrame({corrected.name: corrected})
            tag = self.method_suffix.lower()
            result.attrs[ATTRS_KEY] = {
                corrected.name: provenance_attr(
                    origin=MODIFIED, parent=str(series.name), operation=self.title,
                    params=kwargs, tags=[tag, "corrected"]),
            }
            n_changed = int(self._changed_mask(series, corrected).sum())
            payload = {"var": series.name, "corrected": corrected,
                       "result": result, "n_changed": n_changed, "extra": extra}
        except Exception as err:  # surface the library error to the user
            self._sig.run_failed.emit(str(err))
            return
        self._sig.run_done.emit(payload)

    # --- result rendering hooks (overridable) ---
    def _hero_metrics(self, payload: dict) -> list:
        """Return ``[(name, value, tooltip), ...]`` for the hero stats strip
        (subclass hook). Default: one 'records changed' slot."""
        return [("RECORDS CHANGED", f"{payload['n_changed']:,}",
                 "Records altered by the correction")]

    def _status_text(self, payload: dict) -> str:
        """Status-line text after a run (subclass hook)."""
        n = payload["n_changed"]
        return (f"{n:,} records changed. 'Add' keeps {payload['corrected'].name} "
                f"(original '{payload['var']}' is unchanged).")

    def _render_result(self, payload: dict) -> None:
        """Draw the result preview (subclass hook). Default: the two-panel
        original/corrected view."""
        self._draw(self._df[payload["var"]], corrected=payload["corrected"])

    @staticmethod
    def _changed_mask(orig, corrected) -> pd.Series:
        """Records the correction altered, counting a value<->NaN flip as a change."""
        corrected = corrected.reindex(orig.index)
        both_nan = orig.isna() & corrected.isna()
        return (orig != corrected) & ~both_nan

    def _on_done(self, payload: dict) -> None:
        self.run_btn.setEnabled(True)
        self._result_df = payload["result"]
        self._last_payload = payload
        self._update_hero(payload)
        self._render_result(payload)
        self.status.setText(self._status_text(payload))
        self.add_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(f"Failed: {msg}")

    def _draw(self, series, corrected=None) -> None:
        """Two stacked panels: top = original (+ red X where the correction set
        points to missing), bottom = corrected over a faint original reference.
        Called on variable select (series only) and after a correction runs."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax_top = fig.add_subplot(2, 1, 1)
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)

        ax_top.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.6,
                    alpha=0.9, label="original", zorder=1)
        if corrected is not None:
            # Points present in the original but dropped to NaN by the correction.
            aligned = corrected.reindex(series.index)
            removed = series[series.notna() & aligned.isna()]
            if len(removed):
                ax_top.plot(removed.index, removed.to_numpy(), linestyle="none",
                            marker="x", color=_C_REMOVED, ms=6, markeredgewidth=1.4,
                            zorder=5, label=f"set to missing ({len(removed)})")
            handles, _ = ax_top.get_legend_handles_labels()
            if handles:
                ax_top.legend(loc="best", fontsize=8, framealpha=0.9)
        ax_top.set_title(f"{series.name} — original"
                         + (" + changes" if corrected is not None else ""), fontsize=9)

        if corrected is not None:
            # Faint original behind the corrected line so the shift/cap is visible.
            ax_bot.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.5,
                        alpha=0.5, zorder=1)
            ax_bot.plot(corrected.index, corrected.to_numpy(), color=_C_CORRECTED,
                        lw=0.8, zorder=2, label="corrected")
            ax_bot.legend(handles=[Line2D([], [], color=_C_RAW, label="original"),
                                   Line2D([], [], color=_C_CORRECTED, label="corrected")],
                          loc="best", fontsize=8, framealpha=0.9)
            ax_bot.set_title("corrected", fontsize=9)
        else:
            ax_bot.set_title("corrected (apply to preview)", fontsize=9)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)  # MainWindow merges into the dataset
        self.status.setText(
            f"Added {', '.join(str(c) for c in result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
