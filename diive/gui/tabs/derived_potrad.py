"""
GUI.TABS.DERIVED_POTRAD: POTENTIAL RADIATION FROM SITE COORDINATES
=================================================================

Compute potential shortwave-incoming radiation (``SW_IN_POT``, W m-2) from the
dataset's timestamps and the site location via :func:`diive.variables.potrad`
(the ONEFlux ``get_rpot`` port behind FLUXNET's ``SW_IN_POT``). A
:class:`~diive.gui.tabs._derived_variable_base.BaseDerivedVariableTab` subclass
that declares **no input columns**: the only inputs are the timestamp index and
the site coordinates, so the variable list and the input-column box are hidden
and a "Site coordinates" box takes their place (seeded from **Project settings**,
mirroring the correction tabs).

All maths lives in the library; this tab only collects the coordinates, previews
the result, and emits it with DERIVED provenance.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

import diive as dv
from diive.gui import site
from diive.gui.tabs._derived_variable_base import (
    FONT_SIZE,
    TITLE_FONTSIZE,
    BaseDerivedVariableTab,
)


class PotradTab(BaseDerivedVariableTab):
    """Potential radiation from the timestamps + site coordinates."""

    title = "Potential radiation"
    intro = ("Calculate potential shortwave-incoming radiation (SW_IN_POT, W m-2) "
             "from the record's timestamps and the site location. Needs no input "
             "columns - set the coordinates below.")
    inputs = []  # timestamps + coordinates only; no columns
    default_name = "SW_IN_POT"
    out_unit = "W m-2"
    method_tags = ["radiation", "potential_radiation"]

    # --- settings ------------------------------------------------------
    def _add_extra_controls(self, layout: QVBoxLayout) -> None:
        """Site coordinates, seeded from Project settings (and kept in sync)."""
        box = QGroupBox("Site coordinates")
        form = QFormLayout(box)
        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(4)
        self.lat.setToolTip("Site latitude in decimal degrees (north positive).")
        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(4)
        self.lon.setToolTip("Site longitude in decimal degrees (east positive).")
        self.utc = QSpinBox()
        self.utc.setRange(-12, 14)
        self.utc.setToolTip("UTC offset (hours) of the timestamps, e.g. 1 for "
                            "UTC+01:00. Local standard time - no daylight saving.")
        form.addRow("Latitude", self.lat)
        form.addRow("Longitude", self.lon)
        form.addRow("UTC offset", self.utc)
        layout.addWidget(box)

        self._seed_site()
        site.manager.changed.connect(self._seed_site)

    def _seed_site(self) -> None:
        m = site.manager
        if not m.configured:
            return
        self.lat.setValue(m.latitude)
        self.lon.setValue(m.longitude)
        self.utc.setValue(m.utc_offset)

    def _coords(self) -> tuple[float, float, int]:
        return self.lat.value(), self.lon.value(), self.utc.value()

    # --- library calls -------------------------------------------------
    def _compute(self, df: pd.DataFrame, picks: dict[str, str]) -> pd.Series:
        if not site.manager.configured:
            # Every value here is a function of the coordinates, so an unset site
            # would silently return the (0, 0) at UTC curve - plausible-looking
            # and wrong. Refuse rather than emit it.
            raise ValueError(
                "no site coordinates are configured. Set latitude, longitude and "
                "UTC offset in Settings -> Project settings first.")
        lat, lon, utc = self._coords()
        return dv.variables.potrad(df.index, lat=lat, lon=lon, utc_offset=utc)

    def _code(self, picks: dict[str, str], name: str | None) -> str:
        lat, lon, utc = self._coords()
        return dv.variables.potrad_to_code(
            lat=lat, lon=lon, utc_offset=utc, name=name)

    # --- preview -------------------------------------------------------
    def _plot_result(self, series: pd.Series) -> None:
        """Three views of the result, in the space the missing input panels free
        up: the per-month diel cycles and the full time series stacked on the
        left, the date/time heatmap spanning both rows on the right."""
        canvas = self._result_panel.canvas
        canvas.reset_layout()
        try:
            gs = canvas.fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])
            ax_diel = canvas.fig.add_subplot(gs[0, 0])
            ax_ts = canvas.fig.add_subplot(gs[1, 0])
            ax_heat = canvas.fig.add_subplot(gs[:, 1])  # right, spanning both rows
            self._plot_diel(ax_diel, series)
            self._plot_timeseries(ax_ts, series)
            self._plot_heatmap(ax_heat, canvas.fig, series)
        except Exception as err:  # let the user see why, don't crash the tab
            canvas.show_message(f"Cannot plot:\n{err}")
            return
        canvas.draw()
        canvas.reset_history()

    def _compact(self, **overrides) -> "dv.plotting.FormatStyle":
        """The panel-sized chrome these three previews share."""
        return dv.plotting.FormatStyle(
            title_fontsize=TITLE_FONTSIZE, axlabel_fontsize=FONT_SIZE,
            ticks_fontsize=FONT_SIZE, legend_fontsize=FONT_SIZE, **overrides)

    def _plot_diel(self, ax, series: pd.Series) -> None:
        # One auto-coloured curve per month: potential radiation's whole seasonal
        # signal is the changing day length + noon peak, which this reads off
        # directly. Legend columns as on the Overview's diel panel.
        n_months = series.dropna().index.month.nunique()
        dv.plotting.DielCycle(series).plot(
            ax=ax, format_style=self._compact(
                title="Diel cycle per month", show_legend=True,
                legend_ncol=3 if n_months > 8 else 2),
            each_month=True, linewidth=1.1)

    def _plot_timeseries(self, ax, series: pd.Series) -> None:
        # The full record at native resolution: every half-hour swings 0 -> peak,
        # so the line reads as a band whose envelope is the seasonal cycle.
        dv.plotting.TimeSeries(series).plot(
            ax=ax, format_style=self._compact(
                title="Full time series", show_legend=False),  # y-label already names it
            color="#FFC107", linewidth=0.4, alpha=0.9)

    def _plot_heatmap(self, ax, fig, series: pd.Series) -> None:
        dv.plotting.HeatmapDateTime(series).plot(
            ax=ax, fig=fig, format_style=self._compact(),
            cb_digits_after_comma="auto", cb_labelsize=FONT_SIZE)
