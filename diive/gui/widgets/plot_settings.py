"""
GUI.WIDGETS.PLOT_SETTINGS: LIVE PLOT PARAMETER PANEL
====================================================

A scrollable panel of controls that sits between the variable list and the
canvas in a plotting tab. Each control maps to one parameter of the underlying
diive plot class (`HeatmapDateTime.plot(...)` or `TimeSeries.plot(...)`). Editing
a control does not re-render on its own — the tab's "Update plot" button reads
`values()` and re-renders when clicked. (Controls still emit `changed`; it is
left available for callers but the plotting tab no longer renders on it.)

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
    QLabel,
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
DIELCYCLE = "Diel cycle"
CUMULATIVE_YEAR = "Cumulative year"
HEXBIN = "Hexbin"
SCATTER = "Scatter (XY)"

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
    control emits `changed` when edited (kept for callers that want it), but the
    plotting tab does not re-render on it — its "Update plot" button reads the
    current values back as a dict via `values()` instead.
    """

    changed = Signal()

    def __init__(self, plot_type: str) -> None:
        super().__init__()
        self._plot_type = plot_type
        self.setWidgetResizable(True)
        self.setFixedWidth(320)

        inner = QWidget()
        self._col = QVBoxLayout(inner)
        self._col.setContentsMargins(8, 8, 8, 8)

        if plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            self._build_heatmap(yearmonth=plot_type == HEATMAP_YEARMONTH)
        elif plot_type == TIMESERIES:
            self._build_timeseries()
        elif plot_type == RIDGELINE:
            self._build_ridgeline()
        elif plot_type == DIELCYCLE:
            self._build_dielcycle()
        elif plot_type == CUMULATIVE_YEAR:
            self._build_cumulative_year()
        elif plot_type == HEXBIN:
            self._build_hexbin()
        elif plot_type == SCATTER:
            self._build_scatter()

        self._col.addStretch(1)
        # Keep every form within the panel's fixed width: wrap a row's field
        # under its label when the two together are too wide, and let fields grow
        # to fill the available width. Without this, long labels/fields push the
        # content wider than the panel and the right edge (combo/spin arrows)
        # gets clipped.
        for form in inner.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.setWidget(inner)
        self._apply_tooltips()

    def _apply_tooltips(self) -> None:
        """Tooltip each control with its library plot() parameter docstring."""
        import diive as dv
        from diive.core.utils.docstrings import param_docs
        method = {
            HEATMAP: dv.plotting.HeatmapDateTime.plot,
            HEATMAP_YEARMONTH: dv.plotting.HeatmapYearMonth.plot,
            TIMESERIES: dv.plotting.TimeSeries.plot,
            RIDGELINE: dv.plotting.RidgeLinePlot.plot,
            DIELCYCLE: dv.plotting.DielCycle.plot,
            CUMULATIVE_YEAR: dv.plotting.CumulativeYear.__init__,
            HEXBIN: dv.plotting.HexbinPlot.plot,
            SCATTER: dv.plotting.ScatterXY.plot,
        }.get(self._plot_type)
        docs = param_docs(method) if method else {}

        if self._plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            pairs = [
                ("cmap", self.cmap), ("vmin", self.vmin), ("vmax", self.vmax),
                ("color_bad", self.color_bad), ("ax_orientation", self.orientation),
                ("show_less_xticklabels", self.show_less_xticklabels),
                ("show_grid", self.show_grid), ("show_colormap", self.show_colormap),
                ("zlabel", self.zlabel), ("cb_digits_after_comma", self.cb_digits),
                ("cb_extend", self.cb_extend), ("show_values", self.show_values),
                ("show_values_n_dec_places", self.show_values_dec),
                ("show_values_fontsize", self.show_values_fontsize),
                ("axlabels_fontsize", self.axlabels_fontsize),
                ("ticks_labelsize", self.ticks_labelsize), ("cb_labelsize", self.cb_labelsize),
            ]
            if self._plot_type == HEATMAP:
                pairs += [("minticks", self.minticks), ("maxticks", self.maxticks)]
            else:
                pairs += [("agg", self.agg), ("ranks", self.ranks)]
        elif self._plot_type == TIMESERIES:
            pairs = [("linewidth", self.linewidth), ("alpha", self.alpha),
                     ("marker", self.marker), ("markersize", self.markersize),
                     ("drop_gaps", self.drop_gaps), ("title", self.title),
                     ("xlabel", self.xlabel), ("ylabel", self.ylabel),
                     ("series_units", self.series_units)]
        elif self._plot_type == RIDGELINE:
            pairs = [("how", self.how), ("hspace", self.hspace),
                     ("shade_percentile", self.shade_percentile), ("kd_kwargs", self.bandwidth),
                     ("show_mean_line", self.show_mean_line), ("ascending", self.ascending),
                     ("xlabel", self.xlabel)]
        elif self._plot_type == DIELCYCLE:
            pairs = [("mean", self.dc_mean), ("std", self.dc_std),
                     ("each_month", self.dc_each_month), ("show_legend", self.dc_show_legend),
                     ("showgrid", self.dc_show_grid), ("legend_n_col", self.dc_legend_ncol),
                     ("legend_loc", self.dc_legend_loc),
                     ("ylabel", self.dc_ylabel), ("txt_ylabel_units", self.dc_units)]
        elif self._plot_type == CUMULATIVE_YEAR:
            pairs = [("show_reference", self.cy_show_reference),
                     ("highlight_year", self.cy_highlight), ("digits_after_comma", self.cy_digits),
                     ("series_units", self.cy_units), ("yearly_end_date", self.cy_yearly_end)]
        elif self._plot_type == HEXBIN:
            pairs = [
                ("cmap", self.cmap), ("vmin", self.vmin), ("vmax", self.vmax),
                ("color_bad", self.color_bad), ("zlabel", self.zlabel),
                ("xlabel", self.xlabel), ("ylabel", self.ylabel),
                ("cb_digits_after_comma", self.cb_digits), ("cb_extend", self.cb_extend),
                ("show_colormap", self.show_colormap), ("show_values", self.show_values),
                ("show_values_n_dec_places", self.show_values_dec),
                ("show_values_fontsize", self.show_values_fontsize),
                ("axlabels_fontsize", self.axlabels_fontsize),
                ("ticks_labelsize", self.ticks_labelsize), ("cb_labelsize", self.cb_labelsize),
            ]
            # gridsize/normalize_axes/mincnt are __init__ params, not in plot().
            init_docs = param_docs(dv.plotting.HexbinPlot.__init__)
            for param, widget in [("gridsize", self.gridsize),
                                  ("normalize_axes", self.normalize_axes),
                                  ("mincnt", self.mincnt)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == SCATTER:
            pairs = [
                ("cmap", self.sc_cmap), ("show_colorbar", self.sc_show_colorbar),
                ("markersize", self.sc_markersize), ("alpha", self.sc_alpha),
                ("vmin", self.sc_vmin), ("vmax", self.sc_vmax), ("title", self.sc_title),
                ("xlabel", self.sc_xlabel), ("ylabel", self.sc_ylabel),
                ("zlabel", self.sc_zlabel), ("xunits", self.sc_xunits),
                ("yunits", self.sc_yunits),
            ]
            # nbins/binagg are __init__ params, not in plot().
            init_docs = param_docs(dv.plotting.ScatterXY.__init__)
            for param, widget in [("nbins", self.sc_nbins), ("binagg", self.sc_binagg)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        else:
            pairs = []
        for param, widget in pairs:
            tip = docs.get(param)
            if tip:
                widget.setToolTip(tip)

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
        self.reverse_cmap = self._check("Reverse colormap", form)
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
        self.markersize = self._dspin(3.0, 0.5, 20.0, 0.5, 1, form, "Marker size")
        self.drop_gaps = self._check("Drop gaps (connect)", form)
        self._col.addWidget(line)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.title = QLineEdit()
        self.title.setPlaceholderText("(variable name)")
        self.title.editingFinished.connect(self.changed)
        form.addRow("Title", self.title)
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

        self._build_axes_group()

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

    # --- hexbin controls ---
    def _build_hexbin(self) -> None:
        # Read-only readout of the current X/Y/Z role assignment (the tab sets
        # these via set_xyz as the user clicks variables). The list highlight
        # numbers the same picks 1/2/3 = X/Y/Z.
        roles = QGroupBox("Variables")
        form = QFormLayout(roles)
        hint = QLabel("Click list in order →")
        hint.setStyleSheet("color: #90A4AE;")
        form.addRow(hint)
        self.x_role = QLabel("—")
        self.y_role = QLabel("—")
        self.z_role = QLabel("—")
        for lbl in (self.x_role, self.y_role, self.z_role):
            lbl.setStyleSheet("color: #455A64;")
        form.addRow("X (driver)", self.x_role)
        form.addRow("Y (driver)", self.y_role)
        form.addRow("Z (flux)", self.z_role)
        self._col.addWidget(roles)

        binning = QGroupBox("Binning")
        form = QFormLayout(binning)
        self.gridsize = self._spin(11, 2, 100, form, "Grid size")
        self.normalize_axes = self._check("Normalize (pctile)", form)
        self.mincnt = self._spin(0, 0, 1000, form, "Min count")
        self._col.addWidget(binning)

        colors = QGroupBox("Colors")
        form = QFormLayout(colors)
        self.cmap = QComboBox()
        self.cmap.setEditable(True)
        self.cmap.addItems(_COLORMAPS)
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
        self.reverse_cmap = self._check("Reverse colormap", form)
        self._col.addWidget(colors)

        cbar = QGroupBox("Colorbar")
        form = QFormLayout(cbar)
        self.show_colormap = self._check("Show colorbar", form, checked=True)
        self.zlabel = QLineEdit()
        self.zlabel.setPlaceholderText("(units)")
        self.zlabel.editingFinished.connect(self.changed)
        form.addRow("Label", self.zlabel)
        self.cb_digits = QComboBox()
        self.cb_digits.addItems(["0", "1", "2", "3", "4"])
        self.cb_digits.setCurrentText("2")
        self.cb_digits.currentTextChanged.connect(self.changed)
        form.addRow("Decimals", self.cb_digits)
        self.cb_extend = QComboBox()
        self.cb_extend.addItems(["neither", "both", "min", "max"])
        self.cb_extend.currentTextChanged.connect(self.changed)
        form.addRow("Extend arrows", self.cb_extend)
        self._col.addWidget(cbar)

        vals = QGroupBox("Bin values")
        form = QFormLayout(vals)
        self.show_values = self._check("Overlay values", form)
        self.show_values_dec = self._spin(0, 0, 6, form, "Decimals")
        self.show_values_fontsize = self._fontspin(form, "Value font")
        self._col.addWidget(vals)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.xlabel = QLineEdit()
        self.xlabel.setPlaceholderText("(X variable name)")
        self.xlabel.editingFinished.connect(self.changed)
        form.addRow("X label", self.xlabel)
        self.ylabel = QLineEdit()
        self.ylabel.setPlaceholderText("(Y variable name)")
        self.ylabel.editingFinished.connect(self.changed)
        form.addRow("Y label", self.ylabel)
        self._col.addWidget(labels)

        fonts = QGroupBox("Fonts (0 = auto)")
        form = QFormLayout(fonts)
        self.axlabels_fontsize = self._fontspin(form, "Axis labels")
        self.ticks_labelsize = self._fontspin(form, "Tick labels")
        self.cb_labelsize = self._fontspin(form, "Colorbar labels")
        self._col.addWidget(fonts)

    def set_years(self, years) -> None:
        """Populate the cumulative-year highlight dropdown from the data's years.

        Called by the tab when data loads (or the date range changes). Keeps the
        leading "none" entry and preserves the current selection if that year is
        still present.
        """
        if self._plot_type != CUMULATIVE_YEAR:
            return
        current = self.cy_highlight.currentText()
        self.cy_highlight.blockSignals(True)
        self.cy_highlight.clear()
        self.cy_highlight.addItem("none")
        for year in years:
            self.cy_highlight.addItem(str(year))
        idx = self.cy_highlight.findText(current)
        self.cy_highlight.setCurrentIndex(idx if idx >= 0 else 0)
        self.cy_highlight.blockSignals(False)

    def set_xyz(self, x: str | None, y: str | None, z: str | None) -> None:
        """Update the X/Y/Z role readout (hexbin / scatter)."""
        if self._plot_type not in (HEXBIN, SCATTER):
            return
        self.x_role.setText(x or "—")
        self.y_role.setText(y or "—")
        self.z_role.setText(z or "—")

    # --- scatter (XY) controls ---
    def _build_scatter(self) -> None:
        roles = QGroupBox("Variables")
        form = QFormLayout(roles)
        hint = QLabel("Click list in order →")
        hint.setStyleSheet("color: #90A4AE;")
        form.addRow(hint)
        self.x_role = QLabel("—")
        self.y_role = QLabel("—")
        self.z_role = QLabel("—")
        for lbl in (self.x_role, self.y_role, self.z_role):
            lbl.setStyleSheet("color: #455A64;")
        form.addRow("X", self.x_role)
        form.addRow("Y", self.y_role)
        form.addRow("Z (colour, optional)", self.z_role)
        self._col.addWidget(roles)

        binning = QGroupBox("Binning")
        form = QFormLayout(binning)
        self.sc_nbins = self._spin(0, 0, 100, form, "Bins (0 = raw)")
        self.sc_binagg = QComboBox()
        self.sc_binagg.addItems(["median", "mean"])
        self.sc_binagg.currentTextChanged.connect(self.changed)
        form.addRow("Bin aggregation", self.sc_binagg)
        self._col.addWidget(binning)

        points = QGroupBox("Points")
        form = QFormLayout(points)
        self.sc_markersize = self._dspin(40.0, 1.0, 300.0, 5.0, 0, form, "Marker size")
        self.sc_alpha = self._dspin(1.0, 0.05, 1.0, 0.05, 2, form, "Opacity")
        self._col.addWidget(points)

        colors = QGroupBox("Colour (Z)")
        form = QFormLayout(colors)
        self.sc_cmap = QComboBox()
        self.sc_cmap.setEditable(True)
        self.sc_cmap.addItems(["viridis", "plasma", "inferno", "magma", "cividis",
                               "coolwarm", "RdYlBu_r", "Spectral_r"])
        self.sc_cmap.currentTextChanged.connect(self.changed)
        form.addRow("Colormap", self.sc_cmap)
        self.sc_reverse_cmap = self._check("Reverse colormap", form)
        self.sc_show_colorbar = self._check("Show colorbar", form, checked=True)
        self.sc_vmin = QLineEdit()
        self.sc_vmin.setPlaceholderText("auto")
        self.sc_vmin.editingFinished.connect(self.changed)
        form.addRow("Z min", self.sc_vmin)
        self.sc_vmax = QLineEdit()
        self.sc_vmax.setPlaceholderText("auto")
        self.sc_vmax.editingFinished.connect(self.changed)
        form.addRow("Z max", self.sc_vmax)
        self._col.addWidget(colors)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.sc_title = self._lineedit("(Y vs. X)", form, "Title")
        self.sc_xlabel = self._lineedit("(X name)", form, "X label")
        self.sc_ylabel = self._lineedit("(Y name)", form, "Y label")
        self.sc_zlabel = self._lineedit("(Z name)", form, "Z label")
        self.sc_xunits = self._lineedit("e.g. °C", form, "X units")
        self.sc_yunits = self._lineedit("e.g. µmol m⁻²s⁻¹", form, "Y units")
        self._col.addWidget(labels)

        self._build_axes_group()

    def _lineedit(self, placeholder, form, label) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(self.changed)
        form.addRow(label, edit)
        return edit

    # --- diel-cycle controls ---
    def _build_dielcycle(self) -> None:
        grp = QGroupBox("Diel cycle")
        form = QFormLayout(grp)
        self.dc_mean = self._check("Show mean", form, checked=True)
        self.dc_std = self._check("Show ± SD band", form, checked=True)
        self.dc_each_month = self._check("One curve per month", form)
        self.dc_show_legend = self._check("Show legend", form, checked=True)
        self.dc_show_grid = self._check("Show grid", form, checked=True)
        self.dc_legend_ncol = self._spin(1, 1, 6, form, "Legend columns")
        self.dc_legend_loc = QComboBox()
        self.dc_legend_loc.addItems([
            "best", "upper right", "upper left", "lower left", "lower right",
            "right", "center left", "center right", "lower center",
            "upper center", "center",
        ])
        self.dc_legend_loc.currentTextChanged.connect(self.changed)
        form.addRow("Legend position", self.dc_legend_loc)
        self._col.addWidget(grp)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.dc_ylabel = QLineEdit()
        self.dc_ylabel.setPlaceholderText("(variable name)")
        self.dc_ylabel.editingFinished.connect(self.changed)
        form.addRow("Y label", self.dc_ylabel)
        self.dc_units = QLineEdit()
        self.dc_units.setPlaceholderText("e.g. °C")
        self.dc_units.editingFinished.connect(self.changed)
        form.addRow("Y units", self.dc_units)
        self._col.addWidget(labels)

        self._build_axes_group(yonly=True)

    # --- yearly-cumulative controls ---
    def _build_cumulative_year(self) -> None:
        grp = QGroupBox("Yearly cumulative")
        form = QFormLayout(grp)
        self.cy_show_reference = self._check("Show mean reference", form)
        # Populated from the data's years by set_years() when data loads; the
        # first entry ("none") means no highlight.
        self.cy_highlight = QComboBox()
        self.cy_highlight.addItem("none")
        self.cy_highlight.currentTextChanged.connect(self.changed)
        form.addRow("Highlight year", self.cy_highlight)
        self.cy_digits = self._spin(2, 0, 6, form, "Label decimals")
        self._col.addWidget(grp)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.cy_units = QLineEdit()
        self.cy_units.setPlaceholderText("e.g. gC m-2")
        self.cy_units.editingFinished.connect(self.changed)
        form.addRow("Units", self.cy_units)
        self.cy_yearly_end = QLineEdit()
        self.cy_yearly_end.setPlaceholderText("MM-DD (optional)")
        self.cy_yearly_end.editingFinished.connect(self.changed)
        form.addRow("Yearly end date", self.cy_yearly_end)
        self._col.addWidget(labels)

        self._build_axes_group()

    # --- shared "Axes" group (GUI-only: limits/scale/grid applied post-render) ---
    def _build_axes_group(self, yonly: bool = False) -> None:
        """Build the GUI-only Axes group (limits, log scale, invert, grid).

        These are not library plot() params — the tab applies them in a small
        post-render pass on the data axes. `yonly` omits the x controls (e.g. the
        diel cycle, whose x-axis is the fixed 0-24 hour range).
        """
        self._has_axes_group = True
        self._axes_has_x = not yonly
        grp = QGroupBox("Axes")
        form = QFormLayout(grp)
        if not yonly:
            self.ax_xmin = QLineEdit()
            self.ax_xmin.setPlaceholderText("auto")
            self.ax_xmin.editingFinished.connect(self.changed)
            form.addRow("X min", self.ax_xmin)
            self.ax_xmax = QLineEdit()
            self.ax_xmax.setPlaceholderText("auto")
            self.ax_xmax.editingFinished.connect(self.changed)
            form.addRow("X max", self.ax_xmax)
        self.ax_ymin = QLineEdit()
        self.ax_ymin.setPlaceholderText("auto")
        self.ax_ymin.editingFinished.connect(self.changed)
        form.addRow("Y min", self.ax_ymin)
        self.ax_ymax = QLineEdit()
        self.ax_ymax.setPlaceholderText("auto")
        self.ax_ymax.editingFinished.connect(self.changed)
        form.addRow("Y max", self.ax_ymax)
        if not yonly:
            self.ax_logx = self._check("Log X", form)
        self.ax_logy = self._check("Log Y", form)
        self.ax_invert_y = self._check("Invert Y", form)
        self.ax_grid = self._check("Show grid", form)
        self._col.addWidget(grp)

    def _axes_values(self) -> dict | None:
        """GUI-only axis settings, or None if this plot type has no Axes group."""
        if not getattr(self, "_has_axes_group", False):
            return None
        has_x = self._axes_has_x
        return {
            "xmin": self._float_or_none(self.ax_xmin.text()) if has_x else None,
            "xmax": self._float_or_none(self.ax_xmax.text()) if has_x else None,
            "ymin": self._float_or_none(self.ax_ymin.text()),
            "ymax": self._float_or_none(self.ax_ymax.text()),
            "logx": self.ax_logx.isChecked() if has_x else False,
            "logy": self.ax_logy.isChecked(),
            "invert_y": self.ax_invert_y.isChecked(),
            "grid": self.ax_grid.isChecked(),
        }

    @staticmethod
    def _reverse_cmap(cmap, reverse: bool):
        """Toggle a matplotlib colormap's `_r` suffix when `reverse` is set."""
        if not reverse or not cmap:
            return cmap
        return cmap[:-2] if cmap.endswith("_r") else cmap + "_r"

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
            reverse = self.reverse_cmap.isChecked()
            if self._plot_type == HEATMAP_YEARMONTH:
                # "auto"/empty -> None so HeatmapYearMonth picks the rank-aware default.
                cmap = None if cmap_text in ("", "auto") else cmap_text
                opts["cmap"] = self._reverse_cmap(cmap, reverse)
                opts["agg"] = self.agg.currentText()
                opts["ranks"] = self.ranks.isChecked()
            else:
                opts["cmap"] = self._reverse_cmap(cmap_text or "RdYlBu_r", reverse)
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
        if self._plot_type == DIELCYCLE:
            return {
                "mean": self.dc_mean.isChecked(),
                "std": self.dc_std.isChecked(),
                "each_month": self.dc_each_month.isChecked(),
                "show_legend": self.dc_show_legend.isChecked(),
                "showgrid": self.dc_show_grid.isChecked(),
                "legend_n_col": self.dc_legend_ncol.value(),
                "legend_loc": self.dc_legend_loc.currentText(),
                "ylabel": self.dc_ylabel.text().strip() or None,
                "txt_ylabel_units": self.dc_units.text().strip() or None,
                "_axes": self._axes_values(),
            }
        if self._plot_type == CUMULATIVE_YEAR:
            hy = self.cy_highlight.currentText()
            return {
                "show_reference": self.cy_show_reference.isChecked(),
                "highlight_year": int(hy) if hy.isdigit() else None,
                "digits_after_comma": self.cy_digits.value(),
                "series_units": self.cy_units.text().strip() or None,
                "yearly_end_date": self.cy_yearly_end.text().strip() or None,
                "_axes": self._axes_values(),
            }
        if self._plot_type == HEXBIN:
            def _font(sp):
                return sp.value() or None
            return {
                # __init__ params
                "gridsize": self.gridsize.value(),
                "normalize_axes": self.normalize_axes.isChecked(),
                "mincnt": self.mincnt.value(),
                # plot() styling params
                "cmap": self._reverse_cmap(self.cmap.currentText().strip() or "RdYlBu_r",
                                           self.reverse_cmap.isChecked()),
                "vmin": self._float_or_none(self.vmin.text()),
                "vmax": self._float_or_none(self.vmax.text()),
                "color_bad": self.color_bad.currentText().strip() or "grey",
                "zlabel": self.zlabel.text().strip() or None,
                "xlabel": self.xlabel.text().strip() or None,
                "ylabel": self.ylabel.text().strip() or None,
                "cb_digits_after_comma": int(self.cb_digits.currentText()),
                "cb_extend": self.cb_extend.currentText(),
                "show_colormap": self.show_colormap.isChecked(),
                "show_values": self.show_values.isChecked(),
                "show_values_n_dec_places": self.show_values_dec.value(),
                "show_values_fontsize": _font(self.show_values_fontsize),
                "axlabels_fontsize": _font(self.axlabels_fontsize),
                "ticks_labelsize": _font(self.ticks_labelsize),
                "cb_labelsize": _font(self.cb_labelsize),
            }
        if self._plot_type == SCATTER:
            return {
                "nbins": self.sc_nbins.value(),
                "binagg": self.sc_binagg.currentText(),
                "cmap": self._reverse_cmap(self.sc_cmap.currentText().strip() or "viridis",
                                           self.sc_reverse_cmap.isChecked()),
                "show_colorbar": self.sc_show_colorbar.isChecked(),
                "markersize": self.sc_markersize.value(),
                "alpha": self.sc_alpha.value(),
                "vmin": self._float_or_none(self.sc_vmin.text()),
                "vmax": self._float_or_none(self.sc_vmax.text()),
                "title": self.sc_title.text().strip() or None,
                "xlabel": self.sc_xlabel.text().strip() or None,
                "ylabel": self.sc_ylabel.text().strip() or None,
                "zlabel": self.sc_zlabel.text().strip() or None,
                "xunits": self.sc_xunits.text().strip() or None,
                "yunits": self.sc_yunits.text().strip() or None,
                "_axes": self._axes_values(),
            }
        return {
            "linewidth": self.linewidth.value(),
            "alpha": self.alpha.value(),
            "marker": self.marker.isChecked(),
            "markersize": self.markersize.value(),
            "drop_gaps": self.drop_gaps.isChecked(),
            "title": self.title.text().strip() or None,
            "xlabel": self.xlabel.text().strip() or None,
            "ylabel": self.ylabel.text().strip() or None,
            "series_units": self.series_units.text().strip() or None,
            "_axes": self._axes_values(),
        }
