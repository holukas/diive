"""
GUI.TABS.DERIVED_POTRAD: POTENTIAL RADIATION FROM SITE COORDINATES
=================================================================

Compute potential shortwave-incoming radiation (``SW_IN_POT``, W m-2) from the
dataset's timestamps and the site location via :func:`diive.variables.potrad`
(the ONEFlux ``get_rpot`` port behind FLUXNET's ``SW_IN_POT``). A
:class:`~diive.gui.tabs._derived_variable_base.BaseDerivedVariableTab` subclass
that declares **no input columns**: the only inputs are the timestamp index and
the site coordinates, so the variable list and the input-column box are hidden
and a "Site coordinates" box takes their place. That box is a **read-only mirror
of Project settings** — the coordinates have exactly one home, and a per-tab
override would silently produce an ``SW_IN_POT`` curve for a different site than
the rest of the project is processed for.

All maths lives in the library; this tab only collects the coordinates, previews
the result, and emits it with DERIVED provenance.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

import diive as dv
from diive.gui import site, theme
from diive.gui.tabs._derived_variable_base import (
    _C_MUTED,
    FONT_SIZE,
    TITLE_FONTSIZE,
    BaseDerivedVariableTab,
)


class PotradTab(BaseDerivedVariableTab):
    """Potential radiation from the timestamps + site coordinates."""

    title = "Potential radiation"
    intro = ("Calculate potential shortwave-incoming radiation (SW_IN_POT, W m-2) "
             "from the record's timestamps and the site location. Needs no input "
             "columns - the coordinates come from Settings -> Project settings "
             "and are shown below read-only; change them there.")
    inputs = []  # timestamps + coordinates only; no columns
    default_name = "SW_IN_POT"
    out_unit = "W m-2"
    method_tags = ["radiation", "potential_radiation"]

    # --- settings ------------------------------------------------------
    def _add_extra_controls(self, layout: QVBoxLayout) -> None:
        """Site coordinates, mirrored read-only from Project settings.

        Display-only on purpose: every SW_IN_POT value is a function of these
        three numbers, so an editable copy here could silently produce a curve
        for a different site than the rest of the project runs on."""
        box = QGroupBox("Site coordinates")
        box.setToolTip("Read-only. These values come from Settings -> Project "
                       "settings; edit them there.")
        v = QVBoxLayout(box)

        note = QLabel("From <b>Project settings</b>. To change them, go to "
                      "Settings -> Project settings.")
        note.setWordWrap(True)
        note.setStyleSheet(f"QLabel {{ color: {_C_MUTED}; }}"
                           + theme.manager.tooltip_qss())
        v.addWidget(note)

        form = QFormLayout()
        self.lat = QDoubleSpinBox()
        self.lat.setRange(-90.0, 90.0)
        self.lat.setDecimals(4)
        self.lon = QDoubleSpinBox()
        self.lon.setRange(-180.0, 180.0)
        self.lon.setDecimals(4)
        self.utc = QSpinBox()
        self.utc.setRange(-12, 14)
        for w in (self.lat, self.lon, self.utc):
            # Disabled, not read-only: read-only still reads as an input field.
            w.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            w.setEnabled(False)
        form.addRow("Latitude", self.lat)
        form.addRow("Longitude", self.lon)
        form.addRow("UTC offset", self.utc)
        v.addLayout(form)

        # Says why the fields read 0/0/0 when no site has been set yet.
        self._coord_state = QLabel()
        self._coord_state.setWordWrap(True)
        v.addWidget(self._coord_state)

        layout.addWidget(box)

        self._seed_site()
        site.manager.changed.connect(self._seed_site)

    def _seed_site(self) -> None:
        m = site.manager
        if not m.configured:
            self._coord_state.setText(
                "<b>No site configured yet</b> - set latitude, longitude and "
                "UTC offset in Settings -> Project settings.")
            self._coord_state.setStyleSheet(
                f"QLabel {{ color: {theme.manager.tokens.get('DANGER_BG', '#E04646')}; }}"
                + theme.manager.tooltip_qss())
            self._coord_state.setVisible(True)
            return
        self.lat.setValue(m.latitude)
        self.lon.setValue(m.longitude)
        self.utc.setValue(m.utc_offset)
        self._coord_state.setVisible(False)

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
