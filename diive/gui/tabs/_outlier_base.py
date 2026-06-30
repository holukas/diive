"""
GUI.TABS._OUTLIER_BASE: SHARED BASE FOR OUTLIER-DETECTION TABS
=============================================================

Common machinery for the GUI's outlier-detection tabs (Hampel, Local SD, …),
sharing the correction/gap-filling tabs' chrome (title bar with Copy Python,
``Target``/``Settings`` list headers, a method **hero chip** with a stats strip):
the variable list, the two-panel preview (original + outlier markers on top, the
cleaned series on the bottom), a worker thread that runs the library detector,
an iteration progress bar, live per-iteration preview, an optional detection-limit
overlay, day/night colouring, "Add to dataset", and "Copy Python".

A concrete tab subclasses :class:`BaseOutlierTab` and fills in only the
method-specific parts via hooks: the parameter widgets, the kwargs they map to,
how to build the detector, and the codegen function. All detection is library
work; this base only collects parameters, runs the detector, previews the
result, and emits the new columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diive.core.metadata import ATTRS_KEY, DERIVED, MODIFIED, provenance_attr
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.progress_bar import ProgressBar
from diive.gui.widgets.tab_chrome import build_titlebar, list_header
from diive.gui.widgets.variable_panel import VariablePanel
from diive.gui.widgets.worker import WorkerRunner

# --- Outlier preview palette ------------------------------------------------
# Colours used across the outlier tabs' preview plots, centralised here.
_C_RAW = "#B0BEC5"      # blue-grey 200 — raw/original line (recedes behind markers)
_C_CLEANED = "#43A047"  # green 600     — cleaned series (the good result)
_C_DAY = "#E53935"      # red 600       — daytime outliers + limits
_C_NIGHT = "#1E88E5"    # blue 600      — nighttime outliers + limits
_C_OUTLIER = "#E53935"  # red 600       — outliers when not separating day/night
_C_REMOVED = "#E53935"  # red 600       — points removed in the current pass (X)
_C_LIMIT = "#8E24AA"    # purple 600    — limits when not separating day/night
                        # (distinct from the blue-grey raw line + green/red series)
_C_MUTED = "#6B7780"    # secondary/help text


class _OutlierSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject). The
    run done/failed plumbing lives in :class:`WorkerRunner`; this carries only
    the live-progress and feature-emit signals."""
    features_created = Signal(object)
    # (iteration, n_outliers, cleaned_series, bounds) — progress bar + live plot;
    # bounds is (lower, upper) data-unit Series for this iteration, or None.
    progress = Signal(int, int, object, object)


class BaseOutlierTab(DiiveTab):
    """Shared base for outlier-detection tabs; subclasses supply the method bits.

    Subclasses set the class attributes below and implement the ``_…`` hooks
    marked "subclass hook"."""

    #: Tab title (DiiveTab requirement) — set by the subclass.
    title = "Outlier method"
    #: One-line description shown above the settings.
    intro = "Detect outliers and keep the original, a cleaned copy, and the flag."
    #: Title of the method-settings group box.
    settings_title = "Settings"
    #: Suffix for the cleaned column name, e.g. "HAMPEL" -> "{var}_HAMPEL".
    method_suffix = "OUTLIER"
    #: Hero chip (method identity), mirroring the correction / gap-filling tabs.
    method_chip_label = "OUTLIERS"
    method_chip_bg = "#FFEBEE"
    method_chip_fg = "#C62828"
    #: Whether the method supports daytime/nighttime separation. When False, the
    #: day/night box is omitted (e.g. rolling/increment z-score have no day/night).
    supports_daynight = True
    #: Whether the "Repeat until no more outliers" checkbox is shown. When False,
    #: it is omitted (e.g. manual removal flags fixed timestamps — repeating would
    #: re-flag the same records and never converge, so the library ignores repeat).
    supports_repeat = True
    #: Label of the run button (manual removal isn't really "detection").
    run_label = "Detect outliers"
    #: When set (e.g. "rolling mean"), the detection band's centre line is drawn
    #: alongside the limits and labelled with this text. Only meaningful for methods
    #: whose band centre is informative on its own (e.g. the rolling z-score).
    band_center_label: str | None = None

    # --- build ---
    def build(self) -> QWidget:
        self._df = None
        self._var: str | None = None
        self._result_df = None  # cleaned + flag columns, pending "Add"
        self._ax_top = None     # original panel (live: progressive outlier markers)
        self._ax_bot = None     # cleaned panel, kept for live per-iteration redraws
        self._prev_cleaned = None  # last iteration's cleaned series (this-pass removals)
        self._orig_series = None   # the series being detected (for cumulative outliers)
        self._live_is_daytime = None  # daytime mask for live colouring (set per run)
        self._bounds_history = []  # per-iteration (lower, upper) bands, accumulated
        self._last_payload = None   # last completed run (for re-render on toggle)
        # Progress-bar bookkeeping; (re)set in _run, but seeded here so the progress
        # slot is safe even when _worker is driven directly (e.g. in tests).
        self._first_n = None
        self._n_iter = 0
        self._sig = _OutlierSignals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created
        self._runner = WorkerRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Title bar (correction/gap-filling-style header): a tracked, bold title with
        # Copy Python at the far-right edge.
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setToolTip(
            "Copy a runnable diive script reproducing this outlier detection.")
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        # Body: target list | settings | hero + preview.
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(10, 4, 10, 4)

        # Left: pick the variable to clean (the target).
        tcol = QVBoxLayout()
        tcol.addWidget(self._list_header("Target", "click to set target"))
        self.varpanel = VariablePanel()
        self.varpanel.list.setToolTip("Click a variable to set it as the detection target.")
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        tcol.addWidget(self.varpanel, stretch=1)
        layout.addLayout(tcol)

        # Middle: settings + run/add.
        mid = self._build_settings()
        mid.setFixedWidth(290)
        layout.addWidget(mid)

        # Right: hero chip over the original/outliers + cleaned preview.
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self._build_hero())
        self.canvas = MplCanvas()
        rlay.addWidget(self.canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        outer.addWidget(body, stretch=1)

        self._sig.progress.connect(self._on_progress)
        # Seed the day/night coords from Site details and stay in sync with edits.
        self._seed_site()
        site.manager.changed.connect(self._on_site_changed)
        return root

    _list_header = staticmethod(list_header)

    def _build_hero(self) -> QWidget:
        """A slim band: the method chip on the left, a stats strip on the right
        (filled in after a run via :meth:`_hero_metrics`). The description lives
        only in the settings panel — not repeated here."""
        self._hero = HeroBand(self.method_chip_label, self.method_chip_bg,
                              self.method_chip_fg)
        return self._hero

    def _clear_hero(self) -> None:
        self._hero.clear()

    def _update_hero(self, payload: dict) -> None:
        """Rebuild the hero stats strip from the subclass's metrics."""
        self._hero.set_metrics(self._hero_metrics(payload))

    def _hero_metrics(self, payload: dict) -> list:
        """Return ``[(name, value, tooltip), ...]`` for the hero stats strip
        (overridable). Default: outliers + iterations, plus the day/night split
        when separation was used."""
        metrics = [
            ("OUTLIERS", f"{payload['n_outliers']:,}", "Records flagged as outliers"),
            ("ITERATIONS", str(self._n_iter),
             "Detection passes that ran (repeat-until-clean)"),
        ]
        if payload["is_daytime"] is not None:
            n_day, n_night = self._daynight_counts(payload["flag"], payload["is_daytime"])
            metrics.append(("DAYTIME", f"{n_day:,}", "Outliers flagged in daytime records"))
            metrics.append(("NIGHTTIME", f"{n_night:,}", "Outliers flagged in nighttime records"))
        return metrics

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

        # Method settings (subclass rows) + shared "repeat".
        form_box = QGroupBox(self.settings_title)
        form = QFormLayout(form_box)
        self._add_method_rows(form)
        # Always created so `self.repeat_cb.isChecked()` is safe everywhere, but
        # only shown for methods where repeating is meaningful.
        self.repeat_cb = QCheckBox("Repeat until no more outliers")
        self.repeat_cb.setChecked(True)
        self.repeat_cb.setToolTip(
            "Re-run the filter until a pass finds no new outliers (removed points are "
            "excluded from the next pass). Off = a single pass only.")
        if self.supports_repeat:
            form.addRow(self.repeat_cb)
        outer.addWidget(form_box)

        # Preview options (presentation only; do not affect detection results).
        preview_box = QGroupBox("Preview")
        preview = QVBoxLayout(preview_box)
        self.live_cb = QCheckBox("Live preview (update plots each iteration)")
        self.live_cb.setChecked(True)
        self.live_cb.setToolTip(
            "Redraw both panels after every iteration (markers, removed points, and "
            "detection bands). Detection runs faster with this off — it then renders "
            "once at the end instead of every iteration.")
        preview.addWidget(self.live_cb)
        self.limits_cb = QCheckBox("Show limit lines")
        self.limits_cb.setChecked(False)
        self.limits_cb.setToolTip(
            "Overlay the upper/lower detection limits as faint dashed lines (red = "
            "daytime, blue = nighttime). Off by default to keep the plot clean.")
        self.limits_cb.toggled.connect(self._rerender_last)
        preview.addWidget(self.limits_cb)
        outer.addWidget(preview_box)

        # Optional day/night separation (needs site coordinates). The checkbox is
        # always created (so `self.daynight_cb.isChecked()` is safe everywhere), but
        # the box is only shown for methods that support it.
        self.daynight_cb = QCheckBox("Separate daytime / nighttime")
        self.daynight_cb.setToolTip(
            "Apply different thresholds to daytime and nighttime records (split by "
            "solar elevation from the site coordinates). Only changes the result when "
            "the two thresholds differ.")
        if self.supports_daynight:
            self.daynight_cb.toggled.connect(self._toggle_daynight)
            dn_box = QGroupBox("Day / night (optional)")
            dn = QFormLayout(dn_box)
            dn.addRow(self.daynight_cb)
            # Subclass per-period threshold rows (returns its widgets to enable/disable).
            threshold_widgets = tuple(self._add_daynight_threshold_rows(dn))
            self.lat = QDoubleSpinBox(); self.lat.setRange(-90.0, 90.0); self.lat.setDecimals(4)
            self.lat.setToolTip("Site latitude in decimal degrees (north positive); used "
                                "to split day from night.")
            self.lon = QDoubleSpinBox(); self.lon.setRange(-180.0, 180.0); self.lon.setDecimals(4)
            self.lon.setToolTip("Site longitude in decimal degrees (east positive); used "
                                "to split day from night.")
            self.utc = QSpinBox(); self.utc.setRange(-12, 14)
            self.utc.setToolTip("UTC offset (hours) of the timestamps; used to align solar "
                                "position for the day/night split.")
            self._dn_widgets = threshold_widgets + (self.lat, self.lon, self.utc)
            for w in self._dn_widgets:
                w.setEnabled(False)
            dn.addRow("Latitude", self.lat)
            dn.addRow("Longitude", self.lon)
            dn.addRow("UTC offset (h)", self.utc)
            dn_note = QLabel("Coordinates default from Settings ▸ Project settings.")
            dn_note.setWordWrap(True)
            dn_note.setStyleSheet(f"color: {_C_MUTED};")
            dn.addRow(dn_note)
            outer.addWidget(dn_box)
        else:
            self.lat = self.lon = self.utc = None
            self._dn_widgets = ()

        self.run_btn = QPushButton(self.run_label)
        theme.set_button_role(self.run_btn, "confirm")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        # Iteration progress (visible only while a repeat-until-clean run is going).
        self.progress = ProgressBar()
        outer.addWidget(self.progress)

        self.status = QLabel("Select a variable on the left.")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.add_btn = QPushButton("Add cleaned + flag to dataset")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(self._add)
        theme.set_button_role(self.add_btn, "confirm")
        outer.addWidget(self.add_btn)

        outer.addStretch(1)
        return panel

    # --- subclass hooks ---
    def _add_method_rows(self, form: QFormLayout) -> None:
        """Add the method-specific parameter rows to ``form`` (subclass hook)."""
        raise NotImplementedError

    def _add_daynight_threshold_rows(self, form: QFormLayout) -> tuple:
        """Add per-period threshold rows; return the widgets to enable/disable with
        the day/night toggle (subclass hook). May return an empty tuple."""
        return ()

    def _seed_daynight_thresholds(self) -> None:
        """Seed the per-period threshold widgets from the global ones when day/night
        is switched on (subclass hook). No-op by default."""

    def _current_kwargs(self) -> dict:
        """Detector kwargs from the current control values (subclass hook)."""
        raise NotImplementedError

    def _make_detector(self, series, kwargs: dict):
        """Construct the (not-yet-run) detector instance (subclass hook)."""
        raise NotImplementedError

    def _codegen(self, kwargs: dict, repeat: bool, var_name: str) -> str:
        """Render a reproducible snippet via the library codegen (subclass hook)."""
        raise NotImplementedError

    # --- day/night coords ---
    def _toggle_daynight(self, on: bool) -> None:
        for w in self._dn_widgets:
            w.setEnabled(on)
        if on:
            self._seed_daynight_thresholds()
            self._seed_site()

    def _seed_site(self) -> None:
        """Prefill lat/lon/UTC from the app-wide Site details, if configured."""
        if not self.supports_daynight:
            return
        m = site.manager
        if not m.configured:
            return
        self.lat.setValue(m.latitude)
        self.lon.setValue(m.longitude)
        self.utc.setValue(m.utc_offset)

    def _on_site_changed(self) -> None:
        # Keep the fields in sync if the site is edited while day/night is on.
        if self.daynight_cb.isChecked():
            self._seed_site()

    # --- state (save/restore inputs with a project) ---
    def _state_controls(self) -> dict:
        """Shared persistable controls; subclasses extend with their own params."""
        controls = {"repeat": self.repeat_cb, "live": self.live_cb,
                    "limits": self.limits_cb}
        if self.supports_daynight:
            controls.update(daynight=self.daynight_cb, lat=self.lat,
                            lon=self.lon, utc=self.utc)
        return controls

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"var": self._var, "controls": save_controls(self._state_controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        # Restore in control order: the day/night toggle (re-seeds per-period
        # fields from the global value) comes before those fields, so the saved
        # per-period values, applied after, win.
        restore_controls(self._state_controls(), state.get("controls"))
        var = state.get("var")
        if var and self._df is not None and var in self._df.columns:
            self._select(var)  # re-plots the raw series; detection is not auto-run

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
        self.status.setText(f"Selected '{name}'. Set parameters and detect outliers.")
        # Plot the raw series immediately (top panel); the cleaned panel fills in
        # after detection runs.
        self.varpanel.run_with_loading(name, lambda: self._draw(self._df[name]))

    # --- run ---
    def _python_code(self) -> str | None:
        """Reproducible snippet for the current selection; None if nothing picked."""
        if self._var is None:
            self.status.setText("Select a variable first to copy its code.")
            return None
        return self._codegen(self._current_kwargs(),
                             repeat=self.repeat_cb.isChecked(), var_name=self._var)

    def _run(self) -> None:
        if self._df is None or self._var is None:
            self.status.setText("Select a variable on the left first.")
            return
        if self.daynight_cb.isChecked() and not site.manager.configured:
            self.status.setText(
                "Daytime/nighttime separation needs the site location, but no "
                "coordinates are configured. Set latitude, longitude and UTC offset "
                "in Settings -> Project settings first, or untick 'Separate daytime "
                "/ nighttime' (running now would silently split at (0, 0) at UTC and "
                "corrupt the result).")
            return
        kwargs = self._current_kwargs()
        series = self._df[self._var]
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self._first_n = None  # outlier count of the first iteration (for the %)
        self._n_iter = 0      # iterations that ran (last seen via progress)
        self.progress.set_progress(0, "starting…")
        self.status.setText("Detecting outliers...")
        # For live preview: draw the base view now so both panels exist and can be
        # updated each iteration. Seed "previous cleaned" + "original" with the full
        # series so iteration 1's removals (this-pass and cumulative) are detectable.
        self._orig_series = series
        self._prev_cleaned = series
        self._bounds_history = []
        if self.live_cb.isChecked():
            self._draw(series)
        # Snapshot widget state on the GUI thread; the worker must not read live
        # Qt widgets from a background thread.
        self._runner.run(self._compute_payload, series, kwargs,
                         self.repeat_cb.isChecked(), self.daynight_cb.isChecked())

    def _compute_payload(self, series, kwargs: dict, repeat: bool,
                         separate: bool = False) -> dict:
        """Run the library detector off-thread and return the result payload.
        Emits ``progress`` per iteration (the GUI thread renders it); raises on
        error (the runner forwards to :meth:`_on_failed`)."""
        h = self._make_detector(series, kwargs)
        # Daytime mask is computed in __init__; expose it for live colouring
        # before run() starts firing progress callbacks.
        self._live_is_daytime = getattr(h, "is_daytime", None) if separate else None

        def cb(it, n, s):
            # Read the iteration's data-unit detection band off the detector.
            lo, hi = h.last_lower_bound, h.last_upper_bound
            bounds = (lo.copy(), hi.copy()) if lo is not None and hi is not None else None
            self._sig.progress.emit(it, n, s, bounds)

        h.run(repeat=repeat, progress_callback=cb)
        cleaned = h.filteredseries.copy()
        cleaned.name = f"{series.name}_{self.method_suffix}"
        flag = h.overall_flag.copy()  # name: FLAG_{var}_OUTLIER_{method}_TEST
        result = pd.DataFrame({cleaned.name: cleaned, flag.name: flag})
        # Provenance for the metadata store: the cleaned series is a modified
        # copy of the parent; the flag is derived from it.
        tag = self.method_suffix.lower()
        result.attrs[ATTRS_KEY] = {
            cleaned.name: provenance_attr(
                origin=MODIFIED, parent=str(series.name), operation=self.title,
                params=kwargs, tags=[tag, "outliers-removed"]),
            flag.name: provenance_attr(
                origin=DERIVED, parent=str(series.name),
                operation=f"{self.title} flag", params=kwargs,
                tags=["flag", tag]),
        }
        # Day/night split of the outliers (only when separation was on).
        is_daytime = getattr(h, "is_daytime", None) if separate else None
        return {
            "var": series.name, "cleaned": cleaned, "flag": flag,
            "result": result, "n_outliers": int((flag == 2).sum()),
            "is_daytime": is_daytime,
        }

    def _worker(self, series, kwargs: dict, repeat: bool,
                separate: bool = False) -> None:
        """Synchronous compute + dispatch — tests call this directly; the GUI runs
        :meth:`_compute_payload` on the worker thread instead."""
        try:
            self._on_done(self._compute_payload(series, kwargs, repeat, separate))
        except Exception as err:
            self._on_failed(str(err))

    def _on_progress(self, iteration: int, n_outliers: int, cleaned, bounds) -> None:
        """Fill the bar as repeated iterations remove outliers. Total iterations
        aren't known ahead of time, so progress is measured against the first
        iteration's outlier count (the most that ever get removed at once).
        Accumulates this iteration's detection band and live-updates the panels."""
        self._n_iter = iteration
        if n_outliers > 0 and self._first_n is None:
            self._first_n = n_outliers
        if n_outliers == 0:
            pct = 100
        elif self._first_n:
            pct = max(0, min(99, round((1 - n_outliers / self._first_n) * 100)))
        else:
            pct = 0
        self.progress.set_progress(pct * 10, f"iteration {iteration} — {n_outliers} outliers")
        # Keep every iteration's band so the final render can show them all, even
        # when live preview is off.
        if bounds is not None:
            self._bounds_history.append(bounds)
        if self.live_cb.isChecked():
            self._live_update(cleaned, iteration, bounds)

    def _live_update(self, cleaned, iteration: int, bounds) -> None:
        """Redraw both panels for this iteration. Top: original + the outliers found
        *so far* (cumulative, day/night-coloured) + every iteration's detection band
        (faint — they pile up as a tightening envelope). Bottom: the cleaned series +
        the points removed *this pass* as an X and *this pass's* band, both gone next
        pass since the panel is fully redrawn each time."""
        orig = self._orig_series
        if orig is None or self._ax_top is None or self._ax_bot is None:
            return

        # Cumulative outliers = present in original, absent in cleaned-so-far.
        aligned = cleaned.reindex(orig.index)
        outliers = orig[orig.notna() & aligned.isna()]
        # This-pass removals = present last iteration, gone now.
        prev = self._prev_cleaned
        this_pass = (prev[prev.notna() & cleaned.reindex(prev.index).isna()]
                     if prev is not None else orig.iloc[:0])
        self._prev_cleaned = cleaned

        show_limits = self.limits_cb.isChecked()

        ax_top = self._ax_top
        ax_top.clear()
        ax_top.plot(orig.index, orig.to_numpy(), color=_C_RAW, lw=0.5,
                    alpha=0.8, zorder=1)
        if show_limits:
            for lo, hi in self._bounds_history:
                self._plot_band(ax_top, lo, hi, alpha=0.45, lw=1.0)
        self._plot_outlier_markers(ax_top, outliers, self._live_is_daytime)
        self._finalize_legend(ax_top, show_limits)
        ax_top.set_title(f"{orig.name} — original + outliers (iter {iteration})",
                         fontsize=9)

        ax_bot = self._ax_bot
        ax_bot.clear()
        ax_bot.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.8, zorder=1)
        if show_limits and bounds is not None:  # only this pass's band, a touch stronger
            self._plot_band(ax_bot, bounds[0], bounds[1], alpha=0.85, lw=1.3)
        if len(this_pass):
            ax_bot.plot(this_pass.index, this_pass.to_numpy(), linestyle="none",
                        marker="x", color=_C_REMOVED, ms=7, markeredgewidth=1.5,
                        zorder=5, label=f"removed this pass ({len(this_pass)})")
        self._finalize_legend(ax_bot, show_limits)
        ax_bot.set_title(f"cleaned — after iteration {iteration}", fontsize=9)
        # draw_idle (not draw) so we don't re-freeze the constrained layout mid-run.
        self.canvas.draw_idle()

    # Upper limit = dashed, lower limit = dotted, so the two are told apart in the
    # legend even though they share a day/night colour. The band centre (the rolling
    # mean/median the band is built around) is a solid line.
    _UPPER_STYLE = "--"
    _LOWER_STYLE = ":"
    _CENTER_STYLE = "-"

    def _plot_band(self, ax, lower, upper, *, alpha: float, lw: float) -> None:
        """Draw a detection band (lower/upper limits) as faint lines, day/night
        coloured (red/blue) when a day mask is set, else neutral slate. Reindexed to
        the original index so the lines break at removed/missing points (no long
        interpolation across gaps). When ``band_center_label`` is set, also draws the
        band centre (its midpoint, e.g. the rolling mean) as a solid line — the band
        is symmetric (centre ± limit), so the midpoint is exactly that centre."""
        orig = self._orig_series
        idx = orig.index if orig is not None else lower.index
        is_daytime = self._live_is_daytime
        bounds = [(upper, self._UPPER_STYLE), (lower, self._LOWER_STYLE)]
        if self.band_center_label is not None:
            bounds.append(((lower + upper) / 2.0, self._CENTER_STYLE))
        for bound, style in bounds:
            b = bound.reindex(idx)
            if is_daytime is not None:
                dm = is_daytime.reindex(idx).fillna(False).astype(bool)
                ax.plot(idx, b.where(dm).to_numpy(), color=_C_DAY, ls=style,
                        lw=lw, alpha=alpha, zorder=2)
                ax.plot(idx, b.where(~dm).to_numpy(), color=_C_NIGHT, ls=style,
                        lw=lw, alpha=alpha, zorder=2)
            else:
                ax.plot(idx, b.to_numpy(), color=_C_LIMIT, ls=style,
                        lw=lw, alpha=alpha, zorder=2)

    def _limit_legend_handles(self) -> list:
        """Proxy legend entries for the detection-limit lines (the band lines aren't
        labelled directly — that would add one entry per iteration)."""
        if self._live_is_daytime is not None:
            handles = [
                Line2D([], [], color=_C_DAY, ls=self._UPPER_STYLE, label="upper limit (day)"),
                Line2D([], [], color=_C_DAY, ls=self._LOWER_STYLE, label="lower limit (day)"),
                Line2D([], [], color=_C_NIGHT, ls=self._UPPER_STYLE, label="upper limit (night)"),
                Line2D([], [], color=_C_NIGHT, ls=self._LOWER_STYLE, label="lower limit (night)"),
            ]
            if self.band_center_label is not None:
                handles += [
                    Line2D([], [], color=_C_DAY, ls=self._CENTER_STYLE,
                           label=f"{self.band_center_label} (day)"),
                    Line2D([], [], color=_C_NIGHT, ls=self._CENTER_STYLE,
                           label=f"{self.band_center_label} (night)"),
                ]
            return handles
        handles = [
            Line2D([], [], color=_C_LIMIT, ls=self._UPPER_STYLE, label="upper limit"),
            Line2D([], [], color=_C_LIMIT, ls=self._LOWER_STYLE, label="lower limit"),
        ]
        if self.band_center_label is not None:
            handles.append(Line2D([], [], color=_C_LIMIT, ls=self._CENTER_STYLE,
                                  label=self.band_center_label))
        return handles

    def _finalize_legend(self, ax, show_limits: bool) -> None:
        """Build the axis legend from its labelled artists, plus limit proxies."""
        handles, _ = ax.get_legend_handles_labels()
        if show_limits:
            handles = handles + self._limit_legend_handles()
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)

    @staticmethod
    def _plot_outlier_markers(ax, outliers, is_daytime) -> None:
        """Plot outlier markers on ``ax``: red (daytime) / blue (nighttime) when a
        day mask is given, otherwise plain red. Does not build the legend — the
        caller does (to also add limit entries)."""
        if is_daytime is not None:
            day_mask = is_daytime.reindex(outliers.index).fillna(False).astype(bool)
            day, night = outliers[day_mask], outliers[~day_mask]
            ax.plot(day.index, day.to_numpy(), linestyle="none", marker="o",
                    color=_C_DAY, ms=4, markeredgecolor="none", alpha=0.85,
                    zorder=6, label=f"daytime ({len(day)})")
            ax.plot(night.index, night.to_numpy(), linestyle="none", marker="o",
                    color=_C_NIGHT, ms=4, markeredgecolor="none", alpha=0.85,
                    zorder=5, label=f"nighttime ({len(night)})")
        else:
            ax.plot(outliers.index, outliers.to_numpy(), linestyle="none",
                    marker="o", color=_C_OUTLIER, ms=4, markeredgecolor="none",
                    alpha=0.85, zorder=5, label=f"outliers ({len(outliers)})")

    def _on_done(self, payload: dict) -> None:
        self.run_btn.setEnabled(True)
        self.progress.finish()
        self._result_df = payload["result"]
        self._last_payload = payload  # for re-render when the limit toggle changes
        self._update_hero(payload)
        self._draw(self._df[payload["var"]], flag=payload["flag"],
                   cleaned=payload["cleaned"], n_outliers=payload["n_outliers"],
                   is_daytime=payload["is_daytime"],
                   bounds_history=self._bounds_history)
        n = payload["n_outliers"]
        iters = self._n_iter
        split = ""
        if payload["is_daytime"] is not None:
            n_day, n_night = self._daynight_counts(payload["flag"], payload["is_daytime"])
            split = f" ({n_day} daytime, {n_night} nighttime)"
        self.status.setText(
            f"{n} outliers flagged{split} over {iters} iteration{'' if iters == 1 else 's'}. "
            f"'Add' keeps {payload['cleaned'].name} and the flag "
            f"(original '{payload['var']}' is unchanged).")
        self.add_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.progress.finish()
        self.status.setText(f"Failed: {msg}")

    def _rerender_last(self) -> None:
        """Re-draw the last completed run when the limit-lines toggle changes.
        Ignored while a run is in progress (the live view already reflects it) or
        before any run."""
        p = self._last_payload
        if (p is None or not self.run_btn.isEnabled()
                or self._df is None or p["var"] not in self._df.columns):
            return
        self._draw(self._df[p["var"]], flag=p["flag"], cleaned=p["cleaned"],
                   n_outliers=p["n_outliers"], is_daytime=p["is_daytime"],
                   bounds_history=self._bounds_history)

    @staticmethod
    def _daynight_counts(flag, is_daytime) -> tuple[int, int]:
        """(n_daytime, n_nighttime) outliers, splitting the flag by the day mask."""
        out_idx = flag[flag == 2].index
        day_mask = is_daytime.reindex(out_idx).fillna(False).astype(bool)
        n_day = int(day_mask.sum())
        return n_day, len(out_idx) - n_day

    def _draw(self, series, flag=None, cleaned=None, n_outliers: int = 0,
              is_daytime=None, bounds_history=None) -> None:
        """Two stacked panels: top = original (+ outlier markers once detected),
        bottom = the cleaned series. Called on variable select (series only) and
        after detection (with flag + cleaned). When day/night separation was used,
        the outlier markers are coloured red (daytime) / blue (nighttime).
        ``bounds_history`` (list of (lower, upper)) draws the per-iteration detection
        bands faintly in the top panel."""
        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax_top = fig.add_subplot(2, 1, 1)
        # Share the time axis only: the cleaned panel must autoscale its y-axis to
        # the outlier-free range (sharing y would keep it stretched by the spikes
        # the top panel still shows).
        ax_bot = fig.add_subplot(2, 1, 2, sharex=ax_top)
        self._ax_top = ax_top  # kept so _on_progress can live-update this panel
        self._ax_bot = ax_bot  # kept so _on_progress can live-update this panel

        show_limits = self.limits_cb.isChecked()
        # Light, thin original line so the outlier markers stand out on top of it
        # (the series is dense, so a dark line would bury them).
        ax_top.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.5,
                    alpha=0.8, label="original", zorder=1)
        if show_limits:
            for lo, hi in (bounds_history or []):
                self._plot_band(ax_top, lo, hi, alpha=0.45, lw=1.0)
        if flag is not None:
            self._plot_outlier_markers(ax_top, series[flag == 2], is_daytime)
            self._finalize_legend(ax_top, show_limits and bool(bounds_history))
        ax_top.set_title(f"{series.name} — original"
                         + (" + outliers" if flag is not None else ""), fontsize=9)

        if cleaned is not None:
            ax_bot.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.8,
                        zorder=1)
            ax_bot.set_title("cleaned (outliers removed)", fontsize=9)
        else:
            ax_bot.set_title("cleaned (run detection)", fontsize=9)
        self.canvas.draw()

    def _add(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        result = self._result_df
        self.featuresCreated.emit(result)  # MainWindow merges into the dataset
        self.status.setText(
            f"Added {', '.join(str(c) for c in result.columns)} to the variable list.")
        self.add_btn.setEnabled(False)
