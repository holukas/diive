"""
GUI.TABS.OUTLIERS: HAMPEL OUTLIER DETECTION TAB
===============================================

Run the library's Hampel filter (`dv.outliers.Hampel`) on a selected variable.
Keeps all three series: the **original** (the existing variable, untouched), the
**cleaned** series (`{var}_HAMPEL`, outliers set to NaN), and the **flag** that
produced it (`FLAG_{var}_OUTLIER_HAMPEL_TEST`, 0 = ok, 2 = outlier). "Add to
dataset" merges the cleaned + flag columns into the variable list (the same
mechanism the feature-engineering tab uses).

All detection is library work; this tab only collects parameters, runs Hampel on
a worker thread, previews the result, and emits the new columns.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading

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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import site, theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel
from diive.preprocessing.outlier_detection.codegen import hampel_to_code

# --- Hampel preview palette -------------------------------------------------
# Colours used across the tab's preview plots, centralised here.
_C_RAW = "#B0BEC5"      # blue-grey 200 — raw/original line (recedes behind markers)
_C_CLEANED = "#43A047"  # green 600     — cleaned series (the good result)
_C_DAY = "#E53935"      # red 600       — daytime outliers + limits
_C_NIGHT = "#1E88E5"    # blue 600      — nighttime outliers + limits
_C_OUTLIER = "#E53935"  # red 600       — outliers when not separating day/night
_C_REMOVED = "#E53935"  # red 600       — points removed in the current pass (X)
_C_LIMIT = "#546E7A"    # blue-grey 600 — limits when not separating day/night
_C_MUTED = "#6B7780"    # secondary/help text


class _OutlierSignals(QObject):
    """Qt signals for the tab (DiiveTab is a plain ABC, not a QObject)."""
    run_done = Signal(object)
    run_failed = Signal(str)
    features_created = Signal(object)
    # (iteration, n_outliers, cleaned_series, bounds) — progress bar + live plot;
    # bounds is (lower, upper) data-unit Series for this iteration, or None.
    progress = Signal(int, int, object, object)


class HampelOutlierTab(DiiveTab):
    """Detect outliers with the Hampel filter; keep original + cleaned + flag."""

    title = "Hampel filter"

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
        self._sig = _OutlierSignals()
        #: Exposed bound signal the main window connects to (merges the columns).
        self.featuresCreated = self._sig.features_created

        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: pick the variable to clean.
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(lambda name, _ctrl: self._select(name))
        layout.addWidget(self.varpanel)

        # Middle: Hampel settings + run/add.
        mid = self._build_settings()
        mid.setFixedWidth(290)
        layout.addWidget(mid)

        # Right: preview (original + outliers + cleaned).
        self.canvas = MplCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self._sig.run_done.connect(self._on_done)
        self._sig.run_failed.connect(self._on_failed)
        self._sig.progress.connect(self._on_progress)
        # Seed the day/night coords from Site details and stay in sync with edits.
        self._seed_site()
        site.manager.changed.connect(self._on_site_changed)
        return root

    def _build_settings(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        intro = QLabel("Detect spikes with the Hampel filter (median absolute "
                       "deviation). Keeps the original, a cleaned copy, and the flag.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #6B7780;")
        outer.addWidget(intro)

        form_box = QGroupBox("Hampel settings")
        form = QFormLayout(form_box)
        self.window = QSpinBox()
        self.window.setRange(3, 1_000_000)
        self.window.setValue(48 * 13)  # 13 days at 30-min sampling (Papale 2006)
        self.window.setToolTip(
            "Size of the sliding window (record count, centred on each point) over "
            "which the local median and MAD are computed. 624 = 13 days at 30-min "
            "sampling (Papale 2006); use 24*13 for hourly data.")
        form.addRow("Window (records)", self.window)
        self.n_sigma = QDoubleSpinBox()
        self.n_sigma.setRange(0.1, 50.0)
        self.n_sigma.setSingleStep(0.5)
        self.n_sigma.setValue(5.5)
        self.n_sigma.setToolTip(
            "Threshold width: a point is an outlier if it deviates from the local "
            "median by more than n_sigma × k × MAD. Lower = stricter (flags more). "
            "Used for all records unless day/night separation is on.")
        form.addRow("n sigma (global)", self.n_sigma)
        self.diff_cb = QCheckBox("Use double-differencing (Papale 2006)")
        self.diff_cb.setChecked(True)
        self.diff_cb.setToolTip(
            "Run the test on the double-differenced series d = 2·xₜ − (xₜ₋₁ + xₜ₊₁) "
            "instead of raw values. Removes trends and isolates short spikes "
            "(recommended for flux/meteo data). Off = test the raw values.")
        form.addRow(self.diff_cb)
        self.repeat_cb = QCheckBox("Repeat until no more outliers")
        self.repeat_cb.setChecked(True)
        self.repeat_cb.setToolTip(
            "Re-run the filter until a pass finds no new outliers (removed points are "
            "excluded from the next pass). Off = a single pass only.")
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

        # Optional day/night separation (needs site coordinates).
        self.daynight_cb = QCheckBox("Separate daytime / nighttime")
        self.daynight_cb.setToolTip(
            "Apply different thresholds to daytime and nighttime records (split by "
            "solar elevation from the site coordinates). Only changes the result when "
            "the two thresholds differ.")
        self.daynight_cb.toggled.connect(self._toggle_daynight)
        dn_box = QGroupBox("Day / night (optional)")
        dn = QFormLayout(dn_box)
        dn.addRow(self.daynight_cb)
        # Separate thresholds make day/night meaningful (with one shared sigma the
        # result is identical to no separation). Seeded from the global n sigma.
        self.n_sigma_dt = QDoubleSpinBox()
        self.n_sigma_dt.setRange(0.1, 50.0); self.n_sigma_dt.setSingleStep(0.5)
        self.n_sigma_dt.setValue(5.5)
        self.n_sigma_dt.setToolTip("Threshold (n sigma) applied to daytime records.")
        self.n_sigma_nt = QDoubleSpinBox()
        self.n_sigma_nt.setRange(0.1, 50.0); self.n_sigma_nt.setSingleStep(0.5)
        self.n_sigma_nt.setValue(5.5)
        self.n_sigma_nt.setToolTip("Threshold (n sigma) applied to nighttime records.")
        self.lat = QDoubleSpinBox(); self.lat.setRange(-90.0, 90.0); self.lat.setDecimals(4)
        self.lat.setToolTip("Site latitude in decimal degrees (north positive); used "
                            "to split day from night.")
        self.lon = QDoubleSpinBox(); self.lon.setRange(-180.0, 180.0); self.lon.setDecimals(4)
        self.lon.setToolTip("Site longitude in decimal degrees (east positive); used "
                            "to split day from night.")
        self.utc = QSpinBox(); self.utc.setRange(-12, 14)
        self.utc.setToolTip("UTC offset (hours) of the timestamps; used to align solar "
                            "position for the day/night split.")
        self._dn_widgets = (self.n_sigma_dt, self.n_sigma_nt, self.lat, self.lon, self.utc)
        for w in self._dn_widgets:
            w.setEnabled(False)
        dn.addRow("Daytime n sigma", self.n_sigma_dt)
        dn.addRow("Nighttime n sigma", self.n_sigma_nt)
        dn.addRow("Latitude", self.lat)
        dn.addRow("Longitude", self.lon)
        dn.addRow("UTC offset (h)", self.utc)
        dn_note = QLabel("Coordinates default from Settings ▸ Site details; the "
                         "global n sigma above is ignored while this is on.")
        dn_note.setWordWrap(True)
        dn_note.setStyleSheet("color: #6B7780;")
        dn.addRow(dn_note)
        outer.addWidget(dn_box)

        self.run_btn = QPushButton("Detect outliers")
        self.run_btn.clicked.connect(self._run)
        outer.addWidget(self.run_btn)

        # Iteration progress (visible only while a repeat-until-clean run is going).
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
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

        # Reproducible snippet (library codegen) — a quiet footer action, not a
        # primary control, so it sits at the bottom of the panel.
        self.copy_btn = CopyPythonButton(self._python_code)
        self.copy_btn.setFlat(True)
        self.copy_btn.setStyleSheet("color: #6B7780; text-align: left;")
        outer.addWidget(self.copy_btn)
        return panel

    def _toggle_daynight(self, on: bool) -> None:
        for w in self._dn_widgets:
            w.setEnabled(on)
        if on:
            # Seed the per-period thresholds from the global sigma so they start
            # sensible; the user can then make day and night differ.
            self.n_sigma_dt.setValue(self.n_sigma.value())
            self.n_sigma_nt.setValue(self.n_sigma.value())
            self._seed_site()

    def _seed_site(self) -> None:
        """Prefill lat/lon/UTC from the app-wide Site details, if configured."""
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
        self.status.setText(f"Selected '{name}'. Set parameters and detect outliers.")
        # Plot the raw series immediately (top panel); the cleaned panel fills in
        # after detection runs.
        self.varpanel.run_with_loading(name, lambda: self._draw(self._df[name]))

    # --- run ---
    def _current_kwargs(self) -> dict:
        """Hampel kwargs from the current control values (shared by run + codegen)."""
        kwargs = dict(
            window_length=self.window.value(),
            n_sigma=self.n_sigma.value(),
            use_differencing=self.diff_cb.isChecked(),
            separate_day_night=self.daynight_cb.isChecked(),
        )
        if self.daynight_cb.isChecked():
            kwargs.update(lat=self.lat.value(), lon=self.lon.value(),
                          utc_offset=self.utc.value(),
                          n_sigma_daytime=self.n_sigma_dt.value(),
                          n_sigma_nighttime=self.n_sigma_nt.value())
        return kwargs

    def _python_code(self) -> str | None:
        """Reproducible snippet for the current selection; None if nothing picked."""
        if self._var is None:
            self.status.setText("Select a variable first to copy its code.")
            return None
        return hampel_to_code(self._current_kwargs(),
                              repeat=self.repeat_cb.isChecked(), var_name=self._var)

    def _run(self) -> None:
        if self._df is None or self._var is None:
            self.status.setText("Select a variable on the left first.")
            return
        kwargs = self._current_kwargs()
        series = self._df[self._var]
        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self._first_n = None  # outlier count of the first iteration (for the %)
        self._n_iter = 0      # iterations that ran (last seen via progress)
        self.progress.setValue(0)
        self.progress.setFormat("starting…")
        self.progress.setVisible(True)
        self.status.setText("Detecting outliers...")
        # For live preview: draw the base view now so both panels exist and can be
        # updated each iteration. Seed "previous cleaned" + "original" with the full
        # series so iteration 1's removals (this-pass and cumulative) are detectable.
        self._orig_series = series
        self._prev_cleaned = series
        self._bounds_history = []
        if self.live_cb.isChecked():
            self._draw(series)
        threading.Thread(
            target=self._worker,
            args=(series, kwargs, self.repeat_cb.isChecked()),
            daemon=True).start()

    def _worker(self, series, kwargs: dict, repeat: bool) -> None:
        try:
            h = dv.outliers.Hampel(series=series, **kwargs)
            # Daytime mask is computed in __init__; expose it for live colouring
            # before run() starts firing progress callbacks.
            self._live_is_daytime = h.is_daytime if kwargs.get("separate_day_night") else None

            def cb(it, n, s):
                # Read the iteration's data-unit detection band off the detector.
                lo, hi = h.last_lower_bound, h.last_upper_bound
                bounds = (lo.copy(), hi.copy()) if lo is not None and hi is not None else None
                self._sig.progress.emit(it, n, s, bounds)

            h.run(repeat=repeat, progress_callback=cb)
            cleaned = h.filteredseries.copy()
            cleaned.name = f"{series.name}_HAMPEL"
            flag = h.overall_flag.copy()  # name: FLAG_{var}_OUTLIER_HAMPEL_TEST
            result = pd.DataFrame({cleaned.name: cleaned, flag.name: flag})
            # Day/night split of the outliers (only when separation was on).
            # h.is_daytime is the library-computed daytime mask over the index.
            is_daytime = h.is_daytime if kwargs.get("separate_day_night") else None
            payload = {
                "var": series.name, "cleaned": cleaned, "flag": flag,
                "result": result, "n_outliers": int((flag == 2).sum()),
                "is_daytime": is_daytime,
            }
        except Exception as err:  # surface the library error to the user
            self._sig.run_failed.emit(str(err))
            return
        self._sig.run_done.emit(payload)

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
        self.progress.setValue(pct)
        self.progress.setFormat(f"iteration {iteration} — {n_outliers} outliers")
        # Keep every iteration's band so the final render can show them all, even
        # when live preview is off.
        if bounds is not None:
            self._bounds_history.append(bounds)
        if self.live_cb.isChecked():
            self._live_update(cleaned, iteration, bounds)

    def _live_update(self, cleaned, iteration: int, bounds) -> None:
        """Redraw both panels for this iteration. Top: original + the outliers found
        *so far* (cumulative, day/night-coloured) + every iteration's detection band
        (faint, dashed — they pile up as a tightening envelope). Bottom: the cleaned
        series + the points removed *this pass* as red X and *this pass's* band, both
        gone next pass since the panel is fully redrawn each time."""
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
        # All bands so far, faint (so many iterations don't swamp the data).
        if show_limits:
            for lo, hi in self._bounds_history:
                self._plot_band(ax_top, lo, hi, alpha=0.18, lw=0.6)
        self._plot_outlier_markers(ax_top, outliers, self._live_is_daytime)
        self._finalize_legend(ax_top, show_limits)
        ax_top.set_title(f"{orig.name} — original + outliers (iter {iteration})",
                         fontsize=9)

        ax_bot = self._ax_bot
        ax_bot.clear()
        ax_bot.plot(cleaned.index, cleaned.to_numpy(), color=_C_CLEANED, lw=0.8, zorder=1)
        if show_limits and bounds is not None:  # only this pass's band, a touch stronger
            self._plot_band(ax_bot, bounds[0], bounds[1], alpha=0.5, lw=0.8)
        if len(this_pass):
            ax_bot.plot(this_pass.index, this_pass.to_numpy(), linestyle="none",
                        marker="x", color=_C_REMOVED, ms=7, markeredgewidth=1.5,
                        zorder=5, label=f"removed this pass ({len(this_pass)})")
        self._finalize_legend(ax_bot, show_limits)
        ax_bot.set_title(f"cleaned — after iteration {iteration}", fontsize=9)
        # draw_idle (not draw) so we don't re-freeze the constrained layout mid-run.
        self.canvas.draw_idle()

    # Upper limit = dashed, lower limit = dotted, so the two are told apart in the
    # legend even though they share a day/night colour.
    _UPPER_STYLE = "--"
    _LOWER_STYLE = ":"

    def _plot_band(self, ax, lower, upper, *, alpha: float, lw: float) -> None:
        """Draw a detection band (lower/lower limits) as faint lines, day/night
        coloured (red/blue) when a day mask is set, else neutral slate. Reindexed to
        the original index so the lines break at removed/missing points (no long
        interpolation across gaps)."""
        orig = self._orig_series
        idx = orig.index if orig is not None else lower.index
        is_daytime = self._live_is_daytime
        for bound, style in ((upper, self._UPPER_STYLE), (lower, self._LOWER_STYLE)):
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
            return [
                Line2D([], [], color=_C_DAY, ls=self._UPPER_STYLE, label="upper limit (day)"),
                Line2D([], [], color=_C_DAY, ls=self._LOWER_STYLE, label="lower limit (day)"),
                Line2D([], [], color=_C_NIGHT, ls=self._UPPER_STYLE, label="upper limit (night)"),
                Line2D([], [], color=_C_NIGHT, ls=self._LOWER_STYLE, label="lower limit (night)"),
            ]
        return [
            Line2D([], [], color=_C_LIMIT, ls=self._UPPER_STYLE, label="upper limit"),
            Line2D([], [], color=_C_LIMIT, ls=self._LOWER_STYLE, label="lower limit"),
        ]

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
        day mask is given, otherwise plain red. Shared by the live and final draws.
        Does not build the legend — the caller does (to also add limit entries)."""
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
        self.progress.setValue(100)
        self.progress.setFormat("done")
        self.progress.setVisible(False)
        self._result_df = payload["result"]
        self._last_payload = payload  # for re-render when the limit toggle changes
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
        self.progress.setVisible(False)
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
                self._plot_band(ax_top, lo, hi, alpha=0.18, lw=0.6)
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
