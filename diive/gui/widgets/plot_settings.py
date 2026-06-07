"""
GUI.WIDGETS.PLOT_SETTINGS: LIVE PLOT PARAMETER PANEL
====================================================

A scrollable panel of controls that sits between the variable list and the
canvas in a plotting tab. Each control maps to one parameter of the underlying
diive plot class (`HeatmapDateTime.plot(...)` or `TimeSeries.plot(...)`); any
change emits `changed`, which the tab connects to a re-render for a live
preview.

This is GUI-only: the controls merely collect parameter values via `values()`;
all plotting is done by the library classes the tab calls. The plot-type
constants live here (not in `plotting.py`) so this module has no dependency on
the tab — `plotting.py` re-exports them.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: Plot-method identifiers; the tab dispatches on these and passes one to this
#: panel so it can build the matching set of controls.
HEATMAP = "Heatmap (date/time)"
HEATMAP_YEARMONTH = "Heatmap (year/month)"
TIMESERIES = "Time series"
RIDGELINE = "Ridgeline"

#: Period grouping for the ridgeline (one density ridge per group).
_RIDGELINE_HOW = ["monthly", "weekly", "yearly"]

#: Year/month aggregation methods offered in the dropdown.
_YEARMONTH_AGGS = ["mean", "median", "sum", "min", "max", "std"]

#: Curated colormaps offered in the heatmap dropdown (it stays editable, so any
#: valid matplotlib name can also be typed). Diverging first (the diive
#: default), then perceptually-uniform sequential, then a few classics.
_COLORMAPS = [
    "RdYlBu_r", "RdYlBu", "RdBu_r", "coolwarm", "Spectral", "Spectral_r",
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "YlOrRd", "YlGnBu", "Greys", "jet",
]

#: NaN-cell colour choices (also editable so any colour name/hex can be typed).
_BAD_COLORS = ["grey", "white", "black", "lightgrey", "#FAFAFA", "none"]


class PlotSettingsPanel(QScrollArea):
    """Live-editable plot parameters for one plot type.

    Builds the control set matching `plot_type` (heatmap or time series). Every
    control is wired so editing it emits `changed`; the tab re-renders on that
    signal. Current values are read back as a dict via `values()`.
    """

    changed = Signal()

    def __init__(self, plot_type: str) -> None:
        super().__init__()
        self._plot_type = plot_type
        self.setWidgetResizable(True)
        self.setFixedWidth(250)

        inner = QWidget()
        self._col = QVBoxLayout(inner)
        self._col.setContentsMargins(8, 8, 8, 8)

        if plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            self._build_heatmap(yearmonth=plot_type == HEATMAP_YEARMONTH)
        elif plot_type == TIMESERIES:
            self._build_timeseries()
        elif plot_type == RIDGELINE:
            self._build_ridgeline()

        self._col.addStretch(1)
        self.setWidget(inner)

    # --- heatmap controls (shared by date/time and year/month) ---
    def _build_heatmap(self, yearmonth: bool = False) -> None:
        self._yearmonth = yearmonth

        colors = QGroupBox("Colors")
        form = QFormLayout(colors)
        self.cmap = QComboBox()
        self.cmap.setEditable(True)
        # Year/month defaults to "auto" (RdYlBu for ranks, RdYlBu_r otherwise).
        self.cmap.addItems((["auto"] + _COLORMAPS) if yearmonth else _COLORMAPS)
        self.cmap.currentTextChanged.connect(self.changed)
        form.addRow("Colormap", self.cmap)

        self.vmin = QLineEdit()
        self.vmin.setPlaceholderText("auto")
        self.vmin.editingFinished.connect(self.changed)
        form.addRow("Min value", self.vmin)
        self.vmax = QLineEdit()
        self.vmax.setPlaceholderText("auto")
        self.vmax.editingFinished.connect(self.changed)
        form.addRow("Max value", self.vmax)

        self.color_bad = QComboBox()
        self.color_bad.setEditable(True)
        self.color_bad.addItems(_BAD_COLORS)
        self.color_bad.currentTextChanged.connect(self.changed)
        form.addRow("Missing color", self.color_bad)
        self._col.addWidget(colors)

        layout = QGroupBox("Layout")
        form = QFormLayout(layout)
        self.orientation = QComboBox()
        self.orientation.addItems(["vertical", "horizontal"])
        self.orientation.currentTextChanged.connect(self.changed)
        form.addRow("Orientation", self.orientation)
        if yearmonth:
            # Year/month aggregates each month across years; offer the method and
            # a rank transform (highest value per month -> rank 1).
            self.agg = QComboBox()
            self.agg.addItems(_YEARMONTH_AGGS)
            self.agg.currentTextChanged.connect(self.changed)
            form.addRow("Aggregation", self.agg)
            self.ranks = self._check("Show ranks", form)
        else:
            self.minticks = self._spin(3, 1, 50, form, "Date min ticks")
            self.maxticks = self._spin(10, 1, 100, form, "Date max ticks")
        self.show_less_xticklabels = self._check("Skip every 2nd x-label", form)
        self.show_grid = self._check("Show grid", form)
        self._col.addWidget(layout)

        cbar = QGroupBox("Colorbar")
        form = QFormLayout(cbar)
        self.show_colormap = self._check("Show colorbar", form, checked=True)
        self.zlabel = QLineEdit()
        self.zlabel.setPlaceholderText("(units)")
        self.zlabel.editingFinished.connect(self.changed)
        form.addRow("Label", self.zlabel)
        self.cb_digits = QComboBox()
        self.cb_digits.addItems(["auto", "0", "1", "2", "3", "4"])
        self.cb_digits.currentTextChanged.connect(self.changed)
        form.addRow("Decimals", self.cb_digits)
        self.cb_extend = QComboBox()
        self.cb_extend.addItems(["neither", "both", "min", "max"])
        self.cb_extend.currentTextChanged.connect(self.changed)
        form.addRow("Extend arrows", self.cb_extend)
        self._col.addWidget(cbar)

        vals = QGroupBox("Cell values")
        form = QFormLayout(vals)
        self.show_values = self._check("Overlay values", form)
        self.show_values_dec = self._spin(0, 0, 6, form, "Decimals")
        self.show_values_fontsize = self._fontspin(form, "Value font")
        self._col.addWidget(vals)

        fonts = QGroupBox("Fonts (0 = auto)")
        form = QFormLayout(fonts)
        self.axlabels_fontsize = self._fontspin(form, "Axis labels")
        self.ticks_labelsize = self._fontspin(form, "Tick labels")
        self.cb_labelsize = self._fontspin(form, "Colorbar labels")
        self._col.addWidget(fonts)

    # --- time-series controls ---
    def _build_timeseries(self) -> None:
        line = QGroupBox("Line")
        form = QFormLayout(line)
        self.linewidth = self._dspin(2.2, 0.2, 10.0, 0.2, 1, form, "Line width")
        self.alpha = self._dspin(0.95, 0.05, 1.0, 0.05, 2, form, "Opacity")
        self.marker = self._check("Show markers", form)
        self.drop_gaps = self._check("Drop gaps (connect)", form)
        self._col.addWidget(line)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.xlabel = QLineEdit()
        self.xlabel.setPlaceholderText("Date")
        self.xlabel.editingFinished.connect(self.changed)
        form.addRow("X label", self.xlabel)
        self.ylabel = QLineEdit()
        self.ylabel.setPlaceholderText("(variable name)")
        self.ylabel.editingFinished.connect(self.changed)
        form.addRow("Y label", self.ylabel)
        self.series_units = QLineEdit()
        self.series_units.setPlaceholderText("e.g. °C")
        self.series_units.editingFinished.connect(self.changed)
        form.addRow("Y units", self.series_units)
        self._col.addWidget(labels)

    # --- ridgeline controls ---
    def _build_ridgeline(self) -> None:
        grp = QGroupBox("Ridges")
        form = QFormLayout(grp)
        self.how = QComboBox()
        self.how.addItems(_RIDGELINE_HOW)
        self.how.currentTextChanged.connect(self.changed)
        form.addRow("Group by", self.how)
        self.hspace = self._dspin(-0.5, -1.0, 0.5, 0.1, 2, form, "Overlap")
        self.shade_percentile = self._dspin(0.5, 0.0, 1.0, 0.05, 2, form, "Shade percentile")
        self.bandwidth = self._dspin(0.0, 0.0, 100.0, 0.5, 2, form, "KDE bandwidth")
        self.bandwidth.setSpecialValueText("auto")  # 0 -> default KernelDensity
        self.show_mean_line = self._check("Show mean line", form)
        self.ascending = self._check("Ascending order", form)
        self._col.addWidget(grp)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.xlabel = QLineEdit()
        self.xlabel.setPlaceholderText("(variable name)")
        self.xlabel.editingFinished.connect(self.changed)
        form.addRow("X label", self.xlabel)
        self._col.addWidget(labels)

    # --- control factories (wire each to `changed`) ---
    def _spin(self, value, lo, hi, form, label) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(value)
        sp.valueChanged.connect(self.changed)
        form.addRow(label, sp)
        return sp

    def _dspin(self, value, lo, hi, step, decimals, form, label) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setValue(value)
        sp.valueChanged.connect(self.changed)
        form.addRow(label, sp)
        return sp

    def _fontspin(self, form, label) -> QDoubleSpinBox:
        """Font-size spinbox where the minimum (0) reads as 'auto' (-> None)."""
        sp = QDoubleSpinBox()
        sp.setRange(0.0, 40.0)
        sp.setSingleStep(1.0)
        sp.setDecimals(0)
        sp.setSpecialValueText("auto")  # shown at the minimum value (0)
        sp.setValue(0.0)
        sp.valueChanged.connect(self.changed)
        form.addRow(label, sp)
        return sp

    def _check(self, label, form, checked: bool = False) -> QCheckBox:
        cb = QCheckBox()
        cb.setChecked(checked)
        cb.toggled.connect(self.changed)
        form.addRow(label, cb)
        return cb

    # --- value readback ---
    @staticmethod
    def _float_or_none(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def values(self) -> dict:
        """Current settings as a dict keyed by the library plot() parameter."""
        if self._plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            digits = self.cb_digits.currentText()
            # Font-size spinboxes use 0 ("auto") to mean "library default" (None).
            def _font(sp):
                return sp.value() or None
            cmap_text = self.cmap.currentText().strip()
            opts = {
                "vmin": self._float_or_none(self.vmin.text()),
                "vmax": self._float_or_none(self.vmax.text()),
                "color_bad": self.color_bad.currentText().strip() or "grey",
                "ax_orientation": self.orientation.currentText(),
                "show_less_xticklabels": self.show_less_xticklabels.isChecked(),
                "show_grid": self.show_grid.isChecked(),
                "show_colormap": self.show_colormap.isChecked(),
                "zlabel": self.zlabel.text().strip() or None,
                "cb_digits_after_comma": "auto" if digits == "auto" else int(digits),
                "cb_extend": self.cb_extend.currentText(),
                "show_values": self.show_values.isChecked(),
                "show_values_n_dec_places": self.show_values_dec.value(),
                "show_values_fontsize": _font(self.show_values_fontsize),
                "axlabels_fontsize": _font(self.axlabels_fontsize),
                "ticks_labelsize": _font(self.ticks_labelsize),
                "cb_labelsize": _font(self.cb_labelsize),
            }
            if self._plot_type == HEATMAP_YEARMONTH:
                # "auto"/empty -> None so HeatmapYearMonth picks the rank-aware default.
                opts["cmap"] = None if cmap_text in ("", "auto") else cmap_text
                opts["agg"] = self.agg.currentText()
                opts["ranks"] = self.ranks.isChecked()
            else:
                opts["cmap"] = cmap_text or "RdYlBu_r"
                opts["minticks"] = self.minticks.value()
                opts["maxticks"] = self.maxticks.value()
            return opts
        if self._plot_type == RIDGELINE:
            bw = self.bandwidth.value()
            return {
                "how": self.how.currentText(),
                "hspace": self.hspace.value(),
                "shade_percentile": self.shade_percentile.value(),
                "show_mean_line": self.show_mean_line.isChecked(),
                "ascending": self.ascending.isChecked(),
                "xlabel": self.xlabel.text().strip() or None,
                "kd_kwargs": {"bandwidth": bw} if bw > 0 else None,
            }
        return {
            "linewidth": self.linewidth.value(),
            "alpha": self.alpha.value(),
            "marker": self.marker.isChecked(),
            "drop_gaps": self.drop_gaps.isChecked(),
            "xlabel": self.xlabel.text().strip() or None,
            "ylabel": self.ylabel.text().strip() or None,
            "series_units": self.series_units.text().strip() or None,
        }
