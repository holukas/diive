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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: Sentinel entry meaning "no colour-by variable" in the colour-by dropdown.
_COLORBY_NONE = "(none)"


class _DropComboBox(QComboBox):
    """A non-editable combo that also accepts a variable name dropped onto it as
    plain text (drag a variable from the list into the field). If the dropped
    name is one of its items, it is selected."""

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        text = event.mimeData().text().strip()
        i = self.findText(text)
        if i >= 0:
            self.setCurrentIndex(i)
            event.acceptProposedAction()


#: Modern line-colour presets offered as one-click swatches (hex, name).
_LINE_COLOR_PRESETS = [
    ("#2196F3", "Blue"), ("#3F51B5", "Indigo"), ("#009688", "Teal"),
    ("#43A047", "Green"), ("#FB8C00", "Orange"), ("#FFB300", "Amber"),
    ("#E53935", "Red"), ("#8E24AA", "Purple"), ("#607D8B", "Blue-grey"),
    ("#37474F", "Ink"),
]

#: Plot-method identifiers; the tab dispatches on these and passes one to this
#: panel so it can build the matching set of controls.
HEATMAP = "Heatmap (date/time)"
HEATMAP_YEARMONTH = "Heatmap (year/month)"
HEATMAP_XYZ = "Heatmap (x/y/z)"
TIMESERIES = "Time series"
RIDGELINE = "Ridgeline"
DIELCYCLE = "Diel cycle"
CUMULATIVE_YEAR = "Cumulative year"
CUMULATIVE = "Cumulative"
HEXBIN = "Hexbin"
SCATTER = "Scatter (XY)"
HISTOGRAM = "Histogram"
SHIFTEDDIST = "Shifted distribution"
WINDROSE = "Wind rose"
TREERING = "Tree ring"
WATERFALL = "Waterfall"

#: Resampling period the waterfall aggregates each bar over (WaterfallPlot
#: `resample`). Editable, so any pandas offset alias can be typed.
_WATERFALL_FREQS = ["D", "W", "ME", "MS", "YE"]
#: Aggregation applied within each waterfall period (WaterfallPlot `agg`).
_WATERFALL_AGGS = ["sum", "mean", "median", "min", "max"]

#: Zone-label presets for the shifted distribution (5 labels, lowest->highest).
#: "Temperature" is the library default (None -> the class's own defaults); the
#: generic preset suits any variable. The display label maps to the label list
#: passed to ShiftedDistributionPlot.plot(zone_labels=...).
_SHIFTEDDIST_ZONE_PRESETS = {
    "Temperature (default)": None,
    "Generic (low/high)": ["Extremely low", "Low", "Normal", "High", "Extremely high"],
}

#: Tree-ring render styles: display label -> TreeRingPlot method selector.
_TREERING_STYLES = {"Filled rings": "filled", "Line traces": "line"}

#: Resampling frequency offered for the tree-ring angular resolution (one slot
#: per resampled step; finer = more slices per ring). Maps the combo to the
#: TreeRingPlot __init__ `resample_freq`.
_TREERING_FREQS = ["D", "h", "30min"]

#: Per-sector aggregations offered for the wind rose (match WindRosePlot's agg).
_WINDROSE_AGGS = ["mean", "median", "min", "max", "sum", "std", "count"]

#: Period grouping for the ridgeline (one density ridge per group).
_RIDGELINE_HOW = ["monthly", "weekly", "yearly"]

#: Diel-cycle central aggregations: display label -> DielCycle.plot `agg` value.
_DIELCYCLE_AGGS = {
    "Mean": "mean", "Median": "median", "Min": "min", "Max": "max",
    "25th percentile": "p25", "75th percentile": "p75",
}
#: Diel-cycle uncertainty bands: display label -> DielCycle.plot `band` value.
_DIELCYCLE_BANDS = {
    "± SD": "sd", "± SE": "se", "IQR (25-75%)": "iqr",
    "Min-Max": "minmax", "None": "none",
}
#: Diel-cycle per-month colour schemes: display label -> colormap (None = the
#: diive 12-month palette).
_DIELCYCLE_COLORSCHEMES = {
    "Months (default)": None, "Viridis": "viridis", "Plasma": "plasma",
    "Spectral": "Spectral", "Turbo": "turbo", "Twilight": "twilight",
    "Rainbow": "rainbow", "HSV": "hsv", "Cool-warm": "coolwarm",
}
#: Diel-cycle curve modes (per month vs a single overall curve).
_DIELCYCLE_PERMONTH = "One curve per month"
_DIELCYCLE_OVERALL = "One curve overall"

#: Year/month aggregation methods offered in the dropdown.
_YEARMONTH_AGGS = ["mean", "median", "sum", "min", "max", "std"]

#: x/y/z heatmap binning: GridAggregator binning strategies (custom needs
#: explicit edges, so it is omitted here) and aggregation functions.
_XYZ_BINNING_TYPES = ["quantiles", "equal_width"]
_XYZ_AGGFUNCS = ["mean", "median", "min", "max", "sum", "count"]

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
    #: Emitted when a scatter X/Y/Colour role picker changes (dropdown or drop).
    #: Unlike `changed`, the tab DOES re-render live on this (variable selection).
    xyz_changed = Signal()

    def __init__(self, plot_type: str) -> None:
        super().__init__()
        self._plot_type = plot_type
        self.setWidgetResizable(True)
        self.setFixedWidth(320)
        # Drop the scroll-area's outer frame so the settings sit borderless
        # (the inner group boxes carry their own framing).
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        inner = QWidget()
        self._col = QVBoxLayout(inner)
        self._col.setContentsMargins(8, 8, 8, 8)

        if plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            self._build_heatmap(yearmonth=plot_type == HEATMAP_YEARMONTH)
        elif plot_type == HEATMAP_XYZ:
            self._build_heatmap_xyz()
        elif plot_type == TIMESERIES:
            self._build_timeseries()
        elif plot_type == RIDGELINE:
            self._build_ridgeline()
        elif plot_type == DIELCYCLE:
            self._build_dielcycle()
        elif plot_type == CUMULATIVE_YEAR:
            self._build_cumulative_year()
        elif plot_type == CUMULATIVE:
            self._build_cumulative()
        elif plot_type == HEXBIN:
            self._build_hexbin()
        elif plot_type == SCATTER:
            self._build_scatter()
        elif plot_type == HISTOGRAM:
            self._build_histogram()
        elif plot_type == SHIFTEDDIST:
            self._build_shifted_distribution()
        elif plot_type == WINDROSE:
            self._build_windrose()
        elif plot_type == TREERING:
            self._build_treering()
        elif plot_type == WATERFALL:
            self._build_waterfall()

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

    #: Fallback one-line descriptions for the shared Format controls, used when
    #: the FormatStyle dataclass auto-__init__ exposes no parsable docstring.
    _FORMAT_TOOLTIPS = {
        "title": "Plot title (blank -> the variable name).",
        "xlabel": "X-axis label (blank -> auto).",
        "ylabel": "Y-axis label (blank -> auto).",
        "xunits": "Units appended to the x-axis label, e.g. (degC).",
        "yunits": "Units appended to the y-axis label, e.g. (degC).",
        "zlabel": "Colorbar / z-axis label.",
        "title_fontsize": "Title font size (0 = auto / theme default).",
        "axlabel_fontsize": "Axis-label font size (0 = auto / theme default).",
        "ticks_fontsize": "Tick-label font size (0 = auto / theme default).",
        "show_grid": "Draw the background grid.",
        "show_legend": "Draw a legend when the axes has labelled artists.",
        "legend_loc": "Legend location (matplotlib loc).",
        "legend_ncol": "Number of legend columns.",
    }

    def _apply_format_tooltips(self) -> None:
        """Tooltip the shared Format controls from the FormatStyle field docs."""
        import diive as dv
        from diive.core.utils.docstrings import param_docs
        docs = param_docs(dv.plotting.FormatStyle.__init__) or {}
        pairs = [
            ("fmt_title", "title"), ("fmt_xlabel", "xlabel"),
            ("fmt_ylabel", "ylabel"), ("fmt_xunits", "xunits"),
            ("fmt_yunits", "yunits"), ("fmt_zlabel", "zlabel"),
            ("fmt_title_fs", "title_fontsize"),
            ("fmt_axlabel_fs", "axlabel_fontsize"),
            ("fmt_ticks_fs", "ticks_fontsize"), ("fmt_grid", "show_grid"),
            ("fmt_legend", "show_legend"), ("fmt_legend_loc", "legend_loc"),
            ("fmt_legend_ncol", "legend_ncol"),
        ]
        for attr, field in pairs:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            tip = docs.get(field) or self._FORMAT_TOOLTIPS.get(field)
            if tip:
                widget.setToolTip(tip)

    def _apply_tooltips(self) -> None:
        """Tooltip each control with its library plot() parameter docstring."""
        import diive as dv
        from diive.core.utils.docstrings import param_docs
        self._apply_format_tooltips()
        method = {
            HEATMAP: dv.plotting.HeatmapDateTime.plot,
            HEATMAP_YEARMONTH: dv.plotting.HeatmapYearMonth.plot,
            HEATMAP_XYZ: dv.plotting.HeatmapXYZ.plot,
            TIMESERIES: dv.plotting.TimeSeries.plot,
            RIDGELINE: dv.plotting.RidgeLinePlot.plot,
            DIELCYCLE: dv.plotting.DielCycle.plot,
            CUMULATIVE_YEAR: dv.plotting.CumulativeYear.__init__,
            CUMULATIVE: dv.plotting.Cumulative.plot,
            HEXBIN: dv.plotting.HexbinPlot.plot,
            SCATTER: dv.plotting.ScatterXY.plot,
            HISTOGRAM: dv.plotting.HistogramPlot.plot,
            SHIFTEDDIST: dv.plotting.ShiftedDistributionPlot.plot,
            WINDROSE: dv.plotting.WindRosePlot.plot,
            TREERING: dv.plotting.TreeRingPlot.plot,
            WATERFALL: dv.plotting.WaterfallPlot.plot,
        }.get(self._plot_type)
        docs = param_docs(method) if method else {}

        if self._plot_type in (HEATMAP, HEATMAP_YEARMONTH):
            pairs = [
                ("cmap", self.cmap), ("vmin", self.vmin), ("vmax", self.vmax),
                ("color_bad", self.color_bad), ("ax_orientation", self.orientation),
                ("show_less_xticklabels", self.show_less_xticklabels),
                ("show_colormap", self.show_colormap),
                ("zlabel", self.zlabel), ("cb_digits_after_comma", self.cb_digits),
                ("cb_extend", self.cb_extend), ("show_values", self.show_values),
                ("show_values_n_dec_places", self.show_values_dec),
                ("show_values_fontsize", self.show_values_fontsize),
                ("cb_labelsize", self.cb_labelsize),
            ]
            if self._plot_type == HEATMAP:
                pairs += [("minticks", self.minticks), ("maxticks", self.maxticks)]
            else:
                pairs += [("agg", self.agg), ("ranks", self.ranks)]
        elif self._plot_type == TIMESERIES:
            pairs = [("linewidth", self.linewidth), ("alpha", self.alpha),
                     ("marker", self.marker), ("markersize", self.markersize),
                     ("drop_gaps", self.drop_gaps)]
        elif self._plot_type == RIDGELINE:
            pairs = [("how", self.how), ("hspace", self.hspace),
                     ("shade_percentile", self.shade_percentile), ("kd_kwargs", self.bandwidth),
                     ("show_mean_line", self.show_mean_line), ("ascending", self.ascending)]
        elif self._plot_type == DIELCYCLE:
            pairs = [("agg", self.dc_agg), ("band", self.dc_band),
                     ("each_month", self.dc_curves), ("cmap", self.dc_colorscheme),
                     ("marker", self.dc_marker), ("markersize", self.dc_markersize)]
        elif self._plot_type == CUMULATIVE_YEAR:
            pairs = [("show_reference", self.cy_show_reference),
                     ("highlight_year", self.cy_highlight), ("digits_after_comma", self.cy_digits),
                     ("series_units", self.cy_units), ("yearly_end_date", self.cy_yearly_end)]
        elif self._plot_type == CUMULATIVE:
            pairs = [("digits_after_comma", self.cu_digits),
                     ("show_title", self.cu_show_title), ("fill", self.cu_fill)]
            # units is an __init__ param, not in plot().
            tip = param_docs(dv.plotting.Cumulative.__init__).get("units")
            if tip:
                self.cu_units.setToolTip(tip)
        elif self._plot_type == HEXBIN:
            pairs = [
                ("cmap", self.cmap), ("vmin", self.vmin), ("vmax", self.vmax),
                ("color_bad", self.color_bad), ("zlabel", self.zlabel),
                ("cb_digits_after_comma", self.cb_digits), ("cb_extend", self.cb_extend),
                ("show_colormap", self.show_colormap), ("show_values", self.show_values),
                ("show_values_n_dec_places", self.show_values_dec),
                ("show_values_fontsize", self.show_values_fontsize),
                ("cb_labelsize", self.cb_labelsize),
            ]
            # gridsize/normalize_axes/mincnt are __init__ params, not in plot().
            init_docs = param_docs(dv.plotting.HexbinPlot.__init__)
            for param, widget in [("gridsize", self.gridsize),
                                  ("normalize_axes", self.normalize_axes),
                                  ("mincnt", self.mincnt)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == HEATMAP_XYZ:
            pairs = [
                ("cmap", self.cmap), ("vmin", self.vmin), ("vmax", self.vmax),
                ("color_bad", self.color_bad), ("zlabel", self.zlabel),
                ("cb_digits_after_comma", self.cb_digits), ("cb_extend", self.cb_extend),
                ("show_colormap", self.show_colormap), ("show_values", self.show_values),
                ("show_values_n_dec_places", self.show_values_dec),
                ("show_values_fontsize", self.show_values_fontsize),
                ("cb_labelsize", self.cb_labelsize),
            ]
            # binning_type/n_bins/aggfunc/min_n_vals_per_bin are GridAggregator
            # params (the raw x/y/z are binned before HeatmapXYZ plots them).
            init_docs = param_docs(dv.analysis.GridAggregator.__init__)
            for param, widget in [("binning_type", self.xyz_binning_type),
                                  ("n_bins", self.xyz_nbins),
                                  ("aggfunc", self.xyz_aggfunc),
                                  ("min_n_vals_per_bin", self.xyz_min_n)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == SCATTER:
            pairs = [
                ("cmap", self.sc_cmap), ("show_colorbar", self.sc_show_colorbar),
                ("markersize", self.sc_markersize), ("alpha", self.sc_alpha),
                ("vmin", self.sc_vmin), ("vmax", self.sc_vmax),
            ]
            # nbins/binagg are __init__ params, not in plot().
            init_docs = param_docs(dv.plotting.ScatterXY.__init__)
            for param, widget in [("nbins", self.sc_nbins), ("binagg", self.sc_binagg)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == HISTOGRAM:
            pairs = [
                ("highlight_peak", self.hist_peak), ("show_counts", self.hist_counts),
                ("show_info", self.hist_info), ("show_title", self.hist_title),
                ("show_zscores", self.hist_zscores),
                ("show_zscore_values", self.hist_zvalues),
            ]
            # n_bins is an __init__ param, not in plot().
            tip = param_docs(dv.plotting.HistogramPlot.__init__).get("n_bins")
            if tip:
                self.hist_nbins.setToolTip(tip)
        elif self._plot_type == SHIFTEDDIST:
            pairs = [
                ("ref_label", self.sd_ref_label), ("comp_label", self.sd_comp_label),
                ("show_legend", self.sd_show_legend),
                ("show_title", self.sd_show_title),
                ("show_xaxis", self.sd_show_xaxis),
                ("show_yaxis", self.sd_show_yaxis),
                ("zone_labels", self.sd_zones),
            ]
        elif self._plot_type == WINDROSE:
            pairs = [
                ("cmap", self.wr_cmap), ("color", self.wr_color),
                ("vmin", self.wr_vmin), ("vmax", self.wr_vmax),
                ("show_colorbar", self.wr_show_colorbar),
                ("cb_label", self.wr_cb_label),
                ("cb_digits_after_comma", self.wr_cb_digits),
                ("max_sector_labels", self.wr_max_labels),
            ]
            # agg/n_sectors/z_agg are __init__ params, not in plot().
            init_docs = param_docs(dv.plotting.WindRosePlot.__init__)
            for param, widget in [("agg", self.wr_agg), ("n_sectors", self.wr_nsectors),
                                  ("z_agg", self.wr_zagg)]:
                tip = init_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == TREERING:
            pairs = [
                ("cmap", self.tr_cmap), ("vmin", self.tr_vmin), ("vmax", self.tr_vmax),
                ("show_month_labels", self.tr_month_labels),
                ("show_month_lines", self.tr_month_lines),
                ("show_year_labels", self.tr_year_labels),
                ("show_year_separators", self.tr_year_separators),
                ("year_label_frequency", self.tr_year_freq),
                ("cb_label", self.tr_cb_label),
                ("cb_digits_after_comma", self.tr_cb_digits),
                ("cb_labelsize", self.tr_cb_labelsize),
            ]
            # resample_freq is an __init__ param; the line-only knobs are
            # documented on plot_line(), not plot().
            init_docs = param_docs(dv.plotting.TreeRingPlot.__init__)
            tip = init_docs.get("resample_freq")
            if tip:
                self.tr_resample.setToolTip(tip)
            line_docs = param_docs(dv.plotting.TreeRingPlot.plot_line)
            for param, widget in [("linewidth", self.tr_linewidth),
                                  ("alpha", self.tr_alpha),
                                  ("amplitude_scale", self.tr_amplitude),
                                  ("ring_width", self.tr_ring_width)]:
                tip = line_docs.get(param)
                if tip:
                    widget.setToolTip(tip)
        elif self._plot_type == WATERFALL:
            pairs = [
                ("digits_after_comma", self.wf_digits),
                ("color_uptake", self.wf_color_uptake),
                ("color_release", self.wf_color_release),
                ("bar_width", self.wf_bar_width),
                ("show_connectors", self.wf_connectors),
            ]
            # series_units/resample/agg/uptake_is_negative are __init__ params.
            init_docs = param_docs(dv.plotting.WaterfallPlot.__init__)
            for param, widget in [("series_units", self.wf_units),
                                  ("resample", self.wf_resample),
                                  ("agg", self.wf_agg),
                                  ("uptake_is_negative", self.wf_uptake_negative)]:
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
        self.cb_labelsize = self._fontspin(form, "Colorbar font")
        self._col.addWidget(cbar)

        vals = QGroupBox("Cell values")
        form = QFormLayout(vals)
        self.show_values = self._check("Overlay values", form)
        self.show_values_dec = self._spin(0, 0, 6, form, "Decimals")
        self.show_values_fontsize = self._fontspin(form, "Value font")
        self._col.addWidget(vals)

        self._build_format_group(fields=["title", "fonts", "show_grid"])

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

        # Colour (incl. the colour-by-variable dropdown) sits near the top, next
        # to the line settings, matching where the other plots place colour.
        self._build_color_group()

        self._build_format_group(fields=[
            "title", "xlabel", "ylabel", "yunits", "fonts",
            "show_grid", "show_legend"])

        self._build_axes_group()

    def _build_color_group(self) -> None:
        """A 'Line color' section: a hex field (the persisted source of truth), a
        row of modern preset swatches, and a system colour-dialog button. The
        swatches/dialog only set the hex field, so the value round-trips through
        it for save/restore (and per-subplot settings)."""
        grp = QGroupBox("Line color")
        form = QFormLayout(grp)

        self.line_color = QLineEdit("auto")
        self.line_color.setPlaceholderText("auto / #RRGGBB / name")
        self.line_color.setToolTip(
            "Line and marker colour. 'auto' uses the theme palette colour for "
            "this panel. Type a hex code (#2196F3) or a matplotlib colour name, "
            "click a swatch, or use Pick.")
        form.addRow("Color", self.line_color)

        swatches = QWidget()
        row = QHBoxLayout(swatches)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self._color_swatches: list[tuple[str, QPushButton]] = []
        for hexcode, name in _LINE_COLOR_PRESETS:
            b = QPushButton()
            b.setFixedSize(18, 18)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(name)
            b.setStyleSheet(
                f"QPushButton {{ background: {hexcode}; border: 1px solid #B0BEC5; "
                f"border-radius: 4px; }} "
                f"QPushButton:checked {{ border: 2px solid #263238; }}")
            b.clicked.connect(lambda _c=False, h=hexcode: self.line_color.setText(h))
            self._color_swatches.append((hexcode, b))
            row.addWidget(b)
        row.addStretch(1)
        pick = QPushButton("Pick…")
        pick.setToolTip("Choose a colour from the system colour dialog.")
        pick.clicked.connect(self._pick_line_color)
        row.addWidget(pick)
        form.addRow(swatches)  # spanning row -> not part of positional state

        self.line_color.textChanged.connect(self._sync_color_swatches)
        # Mark the plot dirty (enable Update plot) when the colour changes -- via
        # typing, a swatch, the Pick dialog, or a saved-state restore (all route
        # through the hex field's text). Without this the Update button stays
        # disabled after a colour edit and appears not to work.
        self.line_color.textChanged.connect(self.changed)
        self._sync_color_swatches(self.line_color.text())

        # Colour-by-variable: colour the line by another variable's value via a
        # colormap. Pick from the dropdown or drag a variable from the list onto
        # it. When set, the single colour above is ignored.
        self.color_by = _DropComboBox()
        self.color_by.addItem(_COLORBY_NONE)
        self.color_by.setToolTip(
            "Colour the line by another variable's value (via the colormap "
            "below). Pick one here, or drag a variable from the list onto this "
            "field. '(none)' uses the single colour above.")
        self.color_by.currentTextChanged.connect(self.changed)
        form.addRow("Color by", self.color_by)
        self.colorby_cmap = QComboBox()
        self.colorby_cmap.setEditable(True)
        self.colorby_cmap.addItems(_COLORMAPS)
        self.colorby_cmap.setCurrentText("RdYlBu_r")
        self.colorby_cmap.setToolTip("Colormap used to colour the line by the "
                                     "colour-by variable.")
        self.colorby_cmap.currentTextChanged.connect(self.changed)
        form.addRow("Color-by map", self.colorby_cmap)

        self._col.addWidget(grp)

    def set_colorby_options(self, names) -> None:
        """Populate the colour-by dropdown with the dataset's variable names,
        keeping the current pick if it still exists. No-op for non-time-series."""
        if not hasattr(self, "color_by"):
            return
        cur = self.color_by.currentText()
        self.color_by.blockSignals(True)
        self.color_by.clear()
        self.color_by.addItem(_COLORBY_NONE)
        self.color_by.addItems([str(n) for n in names])
        i = self.color_by.findText(cur)
        self.color_by.setCurrentIndex(i if i >= 0 else 0)
        self.color_by.blockSignals(False)

    def _pick_line_color(self) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog
        cur = QColor(self.line_color.text().strip())
        if not cur.isValid():
            cur = QColor("#2196F3")
        chosen = QColorDialog.getColor(cur, self, "Line color")
        if chosen.isValid():
            self.line_color.setText(chosen.name())  # '#rrggbb'

    def _sync_color_swatches(self, text: str) -> None:
        """Check the swatch matching the current hex (none for 'auto'/custom)."""
        norm = (text or "").strip().lower()
        for hexcode, b in self._color_swatches:
            b.setChecked(norm == hexcode.lower())

    def _line_color_value(self) -> str | None:
        """The chosen line colour, or None for 'auto' (theme palette fallback)."""
        text = self.line_color.text().strip()
        return None if (not text or text.lower() == "auto") else text

    def _color_by_value(self) -> str | None:
        """The chosen colour-by variable, or None when '(none)' is selected."""
        text = self.color_by.currentText().strip()
        return None if (not text or text == _COLORBY_NONE) else text

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

        self._build_format_group(fields=["title", "xlabel", "fonts"])

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
        self.cb_labelsize = self._fontspin(form, "Colorbar font")
        self._col.addWidget(cbar)

        vals = QGroupBox("Bin values")
        form = QFormLayout(vals)
        self.show_values = self._check("Overlay values", form)
        self.show_values_dec = self._spin(0, 0, 6, form, "Decimals")
        self.show_values_fontsize = self._fontspin(form, "Value font")
        self._col.addWidget(vals)

        self._build_format_group(fields=[
            "title", "xlabel", "ylabel", "fonts", "show_grid"])

    # --- x/y/z heatmap controls ---
    def _build_heatmap_xyz(self) -> None:
        # Read-only readout of the current X/Y/Z role assignment (same click-in-
        # order model as hexbin); the list highlight numbers the picks 1/2/3.
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
        form.addRow("Z (value)", self.z_role)
        self._col.addWidget(roles)

        # Binning: HeatmapXYZ needs pre-aggregated input (one z per x/y bin), so
        # the raw variables are binned and aggregated through GridAggregator.
        binning = QGroupBox("Binning")
        form = QFormLayout(binning)
        self.xyz_binning_type = QComboBox()
        self.xyz_binning_type.addItems(_XYZ_BINNING_TYPES)
        self.xyz_binning_type.currentTextChanged.connect(self.changed)
        form.addRow("Binning", self.xyz_binning_type)
        self.xyz_nbins = self._spin(10, 2, 100, form, "Number of bins")
        self.xyz_aggfunc = QComboBox()
        self.xyz_aggfunc.addItems(_XYZ_AGGFUNCS)
        self.xyz_aggfunc.currentTextChanged.connect(self.changed)
        form.addRow("Aggregation", self.xyz_aggfunc)
        self.xyz_min_n = self._spin(1, 1, 10000, form, "Min values/bin")
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
        self.cb_labelsize = self._fontspin(form, "Colorbar font")
        self._col.addWidget(cbar)

        vals = QGroupBox("Cell values")
        form = QFormLayout(vals)
        self.show_values = self._check("Overlay values", form)
        self.show_values_dec = self._spin(0, 0, 6, form, "Decimals")
        self.show_values_fontsize = self._fontspin(form, "Value font")
        self._col.addWidget(vals)

        self._build_format_group(fields=[
            "title", "xlabel", "ylabel", "fonts", "show_grid"])

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

    def _build_role_combos(self, labels) -> list:
        """Build the three role dropdowns (X/Y/Colour, value/dir/colour) shared by
        the scatter and wind-rose tabs: two required pickers and a final optional
        one (a leading '(none)'). Each is a drop target -- a variable dragged from
        the list onto it is assigned -- and a change re-routes through
        ``xyz_changed`` (the variable selection). Returns the three combos."""
        self._role_none_ok = [False, False, True]
        self._role_combos = []
        for none_ok in self._role_none_ok:
            combo = _DropComboBox()
            if none_ok:
                combo.addItem(_COLORBY_NONE)
            combo.currentTextChanged.connect(self.xyz_changed)
            self._role_combos.append(combo)
        return self._role_combos

    def set_xyz(self, x: str | None, y: str | None, z: str | None) -> None:
        """Reflect the X/Y/Z role assignment into the controls.

        Scatter / wind rose use dropdown pickers (set the current item without
        re-emitting); hexbin uses a passive click-in-order readout label.
        """
        if getattr(self, "_role_combos", None) is not None:
            for combo, name, none_ok in zip(self._role_combos, (x, y, z), self._role_none_ok):
                combo.blockSignals(True)
                i = combo.findText(name) if name else -1
                combo.setCurrentIndex(i if i >= 0 else (0 if none_ok else combo.currentIndex()))
                combo.blockSignals(False)
            return
        if self._plot_type not in (HEXBIN, HEATMAP_XYZ):
            return
        self.x_role.setText(x or "—")
        self.y_role.setText(y or "—")
        self.z_role.setText(z or "—")

    def set_xyz_options(self, names) -> None:
        """Populate the role dropdowns with the dataset's variable names, keeping
        each current pick if it still exists. No-op when there are no dropdowns."""
        if getattr(self, "_role_combos", None) is None:
            return
        names = [str(n) for n in names]
        for combo, none_ok in zip(self._role_combos, self._role_none_ok):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if none_ok:
                combo.addItem(_COLORBY_NONE)
            combo.addItems(names)
            i = combo.findText(cur)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.blockSignals(False)

    def xyz_values(self) -> list:
        """Current role picks as [x, y] (+ [colour] when the optional one is set)."""
        x, y, z = (c.currentText().strip() for c in self._role_combos)
        picks = [n for n in (x, y) if n]
        if z and z != _COLORBY_NONE:
            picks.append(z)
        return picks

    # --- scatter (XY) controls ---
    def _build_scatter(self) -> None:
        roles = QGroupBox("Variables")
        form = QFormLayout(roles)
        hint = QLabel("Drag a variable from the list onto a field, or pick one "
                      "from the dropdowns.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #90A4AE;")
        form.addRow(hint)
        # X and Y are required; Colour (Z) is optional. Shared role-dropdown
        # trio (drag a variable onto a field or pick it); see _build_role_combos.
        self.sc_x, self.sc_y, self.sc_z = self._build_role_combos(
            ["X", "Y", "Colour (optional)"])
        for label, combo in (("X", self.sc_x), ("Y", self.sc_y),
                             ("Colour (optional)", self.sc_z)):
            form.addRow(label, combo)
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

        self._build_format_group(fields=[
            "title", "xlabel", "ylabel", "xunits", "yunits", "zlabel",
            "fonts", "show_grid", "show_legend"])

        self._build_axes_group()

    # --- wind-rose controls ---
    def _build_windrose(self) -> None:
        roles = QGroupBox("Variables")
        form = QFormLayout(roles)
        hint = QLabel("Drag a variable from the list onto a field, or pick one "
                      "from the dropdowns.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #90A4AE;")
        form.addRow(hint)
        # Value + wind direction are required; colour is optional. Same role-
        # dropdown trio as the scatter tab (drag onto a field or pick).
        self.wr_value, self.wr_winddir, self.wr_zcol = self._build_role_combos(
            ["Value", "Wind direction", "Colour (optional)"])
        for label, combo in (("Value", self.wr_value),
                             ("Wind direction", self.wr_winddir),
                             ("Colour variable (optional)", self.wr_zcol)):
            form.addRow(label, combo)
        self._col.addWidget(roles)

        agg = QGroupBox("Aggregation")
        form = QFormLayout(agg)
        self.wr_agg = QComboBox()
        self.wr_agg.addItems(_WINDROSE_AGGS)
        self.wr_agg.currentTextChanged.connect(self.changed)
        form.addRow("Aggregate (bar)", self.wr_agg)
        self.wr_nsectors = self._spin(8, 2, 180, form, "Sectors")
        self.wr_zagg = QComboBox()
        self.wr_zagg.addItems(_WINDROSE_AGGS)
        self.wr_zagg.setToolTip("Aggregation for the optional colour variable.")
        self.wr_zagg.currentTextChanged.connect(self.changed)
        form.addRow("Aggregate (colour)", self.wr_zagg)
        self.wr_max_labels = self._spin(16, 2, 64, form, "Max sector labels")
        self._col.addWidget(agg)

        colors = QGroupBox("Colors")
        form = QFormLayout(colors)
        self.wr_cmap = QComboBox()
        self.wr_cmap.setEditable(True)
        self.wr_cmap.addItems(_COLORMAPS)
        self.wr_cmap.setCurrentText("RdYlBu_r")
        self.wr_cmap.currentTextChanged.connect(self.changed)
        form.addRow("Colormap", self.wr_cmap)
        self.wr_reverse_cmap = self._check("Reverse colormap", form)
        self.wr_color = QLineEdit()
        self.wr_color.setPlaceholderText("(use colormap)")
        self.wr_color.setToolTip("Single solid bar colour (hex or name). When set, "
                                 "overrides the colormap and hides the colorbar.")
        self.wr_color.editingFinished.connect(self.changed)
        form.addRow("Single color", self.wr_color)
        self.wr_vmin = QLineEdit()
        self.wr_vmin.setPlaceholderText("auto")
        self.wr_vmin.editingFinished.connect(self.changed)
        form.addRow("Color min", self.wr_vmin)
        self.wr_vmax = QLineEdit()
        self.wr_vmax.setPlaceholderText("auto")
        self.wr_vmax.editingFinished.connect(self.changed)
        form.addRow("Color max", self.wr_vmax)
        self._col.addWidget(colors)

        cbar = QGroupBox("Colorbar")
        form = QFormLayout(cbar)
        self.wr_show_colorbar = self._check("Show colorbar", form, checked=True)
        self.wr_cb_label = QLineEdit()
        self.wr_cb_label.setPlaceholderText("(auto)")
        self.wr_cb_label.editingFinished.connect(self.changed)
        form.addRow("Label", self.wr_cb_label)
        self.wr_cb_digits = QComboBox()
        self.wr_cb_digits.addItems(["auto", "0", "1", "2", "3", "4"])
        self.wr_cb_digits.currentTextChanged.connect(self.changed)
        form.addRow("Decimals", self.wr_cb_digits)
        self._col.addWidget(cbar)

        self._build_format_group(fields=["title", "fonts"])

    # --- tree-ring controls ---
    def _build_treering(self) -> None:
        data = QGroupBox("Data")
        form = QFormLayout(data)
        # resample_freq sets the angular resolution of each ring (one slot per
        # resampled step). Editable so any pandas offset alias can be typed.
        self.tr_resample = QComboBox()
        self.tr_resample.setEditable(True)
        self.tr_resample.addItems(_TREERING_FREQS)
        self.tr_resample.currentTextChanged.connect(self.changed)
        form.addRow("Resample", self.tr_resample)
        # Render style: filled colour mesh per ring vs one radial line trace per
        # year (TreeRingPlot.plot vs .plot_line).
        self.tr_style = QComboBox()
        self.tr_style.addItems(list(_TREERING_STYLES.keys()))
        self.tr_style.currentTextChanged.connect(self.changed)
        form.addRow("Render style", self.tr_style)
        self._col.addWidget(data)

        colors = QGroupBox("Colors")
        form = QFormLayout(colors)
        self.tr_cmap = QComboBox()
        self.tr_cmap.setEditable(True)
        self.tr_cmap.addItems(_COLORMAPS)
        self.tr_cmap.setCurrentText("RdBu_r")  # TreeRingPlot.plot default
        self.tr_cmap.currentTextChanged.connect(self.changed)
        form.addRow("Colormap", self.tr_cmap)
        self.tr_reverse_cmap = self._check("Reverse colormap", form)
        self.tr_vmin = QLineEdit()
        self.tr_vmin.setPlaceholderText("auto")
        self.tr_vmin.editingFinished.connect(self.changed)
        form.addRow("Min value", self.tr_vmin)
        self.tr_vmax = QLineEdit()
        self.tr_vmax.setPlaceholderText("auto")
        self.tr_vmax.editingFinished.connect(self.changed)
        form.addRow("Max value", self.tr_vmax)
        self._col.addWidget(colors)

        rings = QGroupBox("Rings")
        form = QFormLayout(rings)
        self.tr_month_labels = self._check("Month labels", form, checked=True)
        self.tr_month_lines = self._check("Month boundary lines", form)
        self.tr_year_labels = self._check("Year labels", form, checked=True)
        self.tr_year_separators = self._check("Year separators", form, checked=True)
        self.tr_year_freq = self._spin(10, 1, 100, form, "Year label every")
        self._col.addWidget(rings)

        # Line-trace style only: a radial wiggle whose amplitude/width and line
        # look are configurable. Ignored by the filled style.
        lines = QGroupBox("Line traces")
        form = QFormLayout(lines)
        note = QLabel("Applies to the 'Line traces' render style.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #90A4AE;")
        form.addRow(note)
        self.tr_linewidth = self._dspin(1.2, 0.2, 10.0, 0.2, 1, form, "Line width")
        self.tr_alpha = self._dspin(0.85, 0.05, 1.0, 0.05, 2, form, "Opacity")
        self.tr_amplitude = self._dspin(0.5, 0.0, 2.0, 0.1, 2, form, "Amplitude")
        self.tr_ring_width = self._dspin(0.35, 0.05, 2.0, 0.05, 2, form, "Ring width")
        self._col.addWidget(lines)

        cbar = QGroupBox("Colorbar")
        form = QFormLayout(cbar)
        self.tr_cb_label = QLineEdit()
        self.tr_cb_label.setPlaceholderText("(units)")
        self.tr_cb_label.editingFinished.connect(self.changed)
        form.addRow("Label", self.tr_cb_label)
        self.tr_cb_digits = self._spin(1, 0, 6, form, "Decimals")
        self.tr_cb_labelsize = self._fontspin(form, "Colorbar font")
        self._col.addWidget(cbar)

        # Tree ring is polar: only the title (+ title font) of the shared chrome
        # applies, like the wind rose.
        self._build_format_group(fields=["title", "fonts"])

    # --- histogram controls ---
    def _build_histogram(self) -> None:
        binning = QGroupBox("Bins")
        form = QFormLayout(binning)
        self.hist_nbins = self._spin(20, 2, 200, form, "Number of bins")
        self._col.addWidget(binning)

        disp = QGroupBox("Display")
        form = QFormLayout(disp)
        self.hist_peak = self._check("Highlight peak bin", form, checked=True)
        self.hist_counts = self._check("Show bar counts", form, checked=True)
        self.hist_info = self._check("Show info box", form, checked=True)
        self.hist_title = self._check("Show title", form, checked=True)
        self._col.addWidget(disp)

        zgrp = QGroupBox("z-scores")
        form = QFormLayout(zgrp)
        self.hist_zscores = self._check("Show z-score axis", form, checked=True)
        self.hist_zvalues = self._check("Show z-score values", form, checked=True)
        self._col.addWidget(zgrp)

        self._build_format_group(fields=["title", "xlabel", "fonts", "show_grid"])

    # --- shifted-distribution controls ---
    def _build_shifted_distribution(self) -> None:
        # Two date periods compared on one density axis. Each field is a partial
        # date string passed straight to the library's label slicing (a year like
        # "2018", or "2018-06" / "2018-06-15"). Seeded from the data's year range
        # by set_periods() when data loads.
        periods = QGroupBox("Periods")
        form = QFormLayout(periods)
        self.sd_ref_start = self._dateedit(form, "Reference start")
        self.sd_ref_end = self._dateedit(form, "Reference end")
        self.sd_comp_start = self._dateedit(form, "Comparison start")
        self.sd_comp_end = self._dateedit(form, "Comparison end")
        self._col.addWidget(periods)

        zones = QGroupBox("Zones")
        form = QFormLayout(zones)
        # Zone breakpoints come from the reference period's mean ±1σ/±3σ (in the
        # library); the panel only picks the 5 labels drawn above the curve.
        self.sd_zones = QComboBox()
        self.sd_zones.addItems(list(_SHIFTEDDIST_ZONE_PRESETS.keys()))
        self.sd_zones.currentTextChanged.connect(self.changed)
        form.addRow("Zone labels", self.sd_zones)
        self._col.addWidget(zones)

        disp = QGroupBox("Display")
        form = QFormLayout(disp)
        self.sd_ref_label = QLineEdit()
        self.sd_ref_label.setPlaceholderText("(auto: Reference …)")
        self.sd_ref_label.editingFinished.connect(self.changed)
        form.addRow("Reference label", self.sd_ref_label)
        self.sd_comp_label = QLineEdit()
        self.sd_comp_label.setPlaceholderText("(auto: Comparison …)")
        self.sd_comp_label.editingFinished.connect(self.changed)
        form.addRow("Comparison label", self.sd_comp_label)
        self.sd_show_legend = self._check("Show legend", form, checked=True)
        self.sd_show_title = self._check("Show title", form, checked=True)
        self.sd_show_xaxis = self._check("Show x-axis", form, checked=True)
        self.sd_show_yaxis = self._check("Show y-axis", form, checked=False)
        self._col.addWidget(disp)

        self._build_format_group(fields=["title", "xlabel", "fonts"])

    def _dateedit(self, form, label) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText("YYYY or YYYY-MM-DD")
        edit.editingFinished.connect(self.changed)
        form.addRow(label, edit)
        return edit

    def set_periods(self, years) -> None:
        """Seed the reference/comparison period fields from the data's years.

        Splits the available years in half (reference = earlier half, comparison
        = later half) so the tab shows a meaningful shift on open. Only fills
        empty fields, so a restored project / user edit is never clobbered.
        """
        if self._plot_type != SHIFTEDDIST:
            return
        years = sorted({int(y) for y in years})
        if not years:
            return
        mid = len(years) // 2
        if mid == 0:  # single year -> compare it to itself (no shift)
            ref = comp = (str(years[0]), str(years[0]))
        else:
            ref = (str(years[0]), str(years[mid - 1]))
            comp = (str(years[mid]), str(years[-1]))
        seeds = [(self.sd_ref_start, ref[0]), (self.sd_ref_end, ref[1]),
                 (self.sd_comp_start, comp[0]), (self.sd_comp_end, comp[1])]
        for edit, value in seeds:
            if not edit.text().strip():
                edit.setText(value)

    # --- diel-cycle controls ---
    def _build_dielcycle(self) -> None:
        grp = QGroupBox("Diel cycle")
        form = QFormLayout(grp)
        # Central aggregation drawn as the curve (default mean). Display labels
        # map to DielCycle.plot's `agg` values in values().
        self.dc_agg = QComboBox()
        self.dc_agg.addItems(list(_DIELCYCLE_AGGS.keys()))
        self.dc_agg.currentTextChanged.connect(self.changed)
        form.addRow("Aggregation", self.dc_agg)
        # Uncertainty band around the curve (default SD).
        self.dc_band = QComboBox()
        self.dc_band.addItems(list(_DIELCYCLE_BANDS.keys()))
        self.dc_band.setCurrentText("± SD")
        self.dc_band.currentTextChanged.connect(self.changed)
        form.addRow("Uncertainty band", self.dc_band)
        # One curve per month (seasonal) vs a single curve over all data.
        self.dc_curves = QComboBox()
        self.dc_curves.addItems([_DIELCYCLE_PERMONTH, _DIELCYCLE_OVERALL])
        self.dc_curves.currentTextChanged.connect(self.changed)
        form.addRow("Curves", self.dc_curves)
        # Per-month colour scheme (only matters in "per month" mode).
        self.dc_colorscheme = QComboBox()
        self.dc_colorscheme.addItems(list(_DIELCYCLE_COLORSCHEMES.keys()))
        self.dc_colorscheme.currentTextChanged.connect(self.changed)
        form.addRow("Color scheme", self.dc_colorscheme)
        # Optional markers at each time-of-day point.
        self.dc_marker = self._check("Show markers", form)
        self.dc_markersize = self._dspin(4.0, 0.5, 20.0, 0.5, 1, form, "Marker size")
        self._col.addWidget(grp)

        # Legend position only -- the column count is set automatically from the
        # number of months (see the diel-cycle render), and the legend is shown
        # on a single panel since all subplots share the same months.
        self._build_format_group(fields=[
            "title", "ylabel", "yunits", "fonts",
            "show_grid", "show_legend", "legend_loc"])

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

        self._build_format_group(fields=["fonts", "show_grid", "show_legend"])

        self._build_axes_group()

    # --- cumulative (whole-record running total) controls ---
    def _build_cumulative(self) -> None:
        grp = QGroupBox("Cumulative")
        form = QFormLayout(grp)
        self.cu_digits = self._spin(0, 0, 6, form, "Label decimals")
        self.cu_show_title = self._check("Show title", form, checked=True)
        self.cu_fill = self._check("Shade to zero", form)
        self._col.addWidget(grp)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.cu_units = QLineEdit()
        self.cu_units.setPlaceholderText("e.g. gC m-2")
        self.cu_units.editingFinished.connect(self.changed)
        form.addRow("Units", self.cu_units)
        self._col.addWidget(labels)

        self._build_format_group(fields=["fonts", "show_grid", "show_legend"])

        self._build_axes_group()

    # --- waterfall (cumulative budget) controls ---
    def _build_waterfall(self) -> None:
        agg = QGroupBox("Aggregation")
        form = QFormLayout(agg)
        # resample/agg aggregate the series to one bar per period; editable so any
        # pandas offset alias can be typed.
        self.wf_resample = QComboBox()
        self.wf_resample.setEditable(True)
        self.wf_resample.addItems(_WATERFALL_FREQS)
        self.wf_resample.currentTextChanged.connect(self.changed)
        form.addRow("Resample", self.wf_resample)
        self.wf_agg = QComboBox()
        self.wf_agg.addItems(_WATERFALL_AGGS)
        self.wf_agg.currentTextChanged.connect(self.changed)
        form.addRow("Aggregation", self.wf_agg)
        # Sign convention: with the default, negative = uptake (sink, blue).
        self.wf_uptake_negative = self._check(
            "Uptake is negative (NEE)", form, checked=True)
        self._col.addWidget(agg)

        bars = QGroupBox("Bars")
        form = QFormLayout(bars)
        self.wf_color_uptake = QLineEdit("#2196F3")
        self.wf_color_uptake.setPlaceholderText("#2196F3")
        self.wf_color_uptake.editingFinished.connect(self.changed)
        form.addRow("Uptake color", self.wf_color_uptake)
        self.wf_color_release = QLineEdit("#F44336")
        self.wf_color_release.setPlaceholderText("#F44336")
        self.wf_color_release.editingFinished.connect(self.changed)
        form.addRow("Release color", self.wf_color_release)
        # 0 -> "auto" (~80% of the median period spacing).
        self.wf_bar_width = self._dspin(0.0, 0.0, 365.0, 0.5, 1, form, "Bar width (days)")
        self.wf_bar_width.setSpecialValueText("auto")
        self.wf_connectors = self._check("Connector lines", form, checked=True)
        self._col.addWidget(bars)

        labels = QGroupBox("Labels")
        form = QFormLayout(labels)
        self.wf_units = QLineEdit()
        self.wf_units.setPlaceholderText("e.g. gC m-2")
        self.wf_units.editingFinished.connect(self.changed)
        form.addRow("Units", self.wf_units)
        self.wf_digits = self._spin(0, 0, 6, form, "Total decimals")
        self._col.addWidget(labels)

        self._build_format_group(fields=["title", "ylabel", "fonts", "show_grid"])

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
        # Grid is controlled by the Format group's "Show grid" (the single source
        # of truth) -- no duplicate here, which previously could only add a grid,
        # never remove it, so toggling it appeared not to work.
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
        }

    #: Matplotlib legend locations offered in the Format group's legend combo.
    _LEGEND_LOCS = [
        "best", "upper right", "upper left", "lower left", "lower right",
        "right", "center left", "center right", "lower center",
        "upper center", "center",
    ]

    def _build_format_group(self, fields) -> None:
        """Append the shared "Format" group, building only the named controls.

        One consistent chrome group (title/labels/units/fonts/grid/legend) for
        every plot type; `fields` selects the subset. Each widget gets a stable
        `fmt_` attribute so `values()`/state can read it, and all rows live in one
        QFormLayout so `_state_widgets()` picks them up automatically. The
        per-type values flow into a `dv.plotting.FormatStyle` via
        :meth:`_format_values`.
        """
        grp = QGroupBox("Format")
        form = QFormLayout(grp)

        def _line(attr, label, placeholder):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.editingFinished.connect(self.changed)
            form.addRow(label, edit)
            setattr(self, attr, edit)

        if "title" in fields:
            _line("fmt_title", "Title", "(auto: variable name)")
        if "xlabel" in fields:
            _line("fmt_xlabel", "X label", "(auto)")
        if "ylabel" in fields:
            _line("fmt_ylabel", "Y label", "(auto)")
        if "xunits" in fields:
            _line("fmt_xunits", "X units", "e.g. °C")
        if "yunits" in fields:
            _line("fmt_yunits", "Y units", "e.g. °C")
        if "zlabel" in fields:
            _line("fmt_zlabel", "Z label", "(colour label)")
        if "fonts" in fields:
            self.fmt_title_fs = self._fontspin(form, "Title font")
            self.fmt_axlabel_fs = self._fontspin(form, "Axis-label font")
            self.fmt_ticks_fs = self._fontspin(form, "Tick font")
        if "show_grid" in fields:
            grid_default = self._plot_type not in (
                HEATMAP, HEATMAP_YEARMONTH, HEATMAP_XYZ, HEXBIN)
            self.fmt_grid = self._check("Show grid", form, checked=grid_default)
        if "show_legend" in fields:
            self.fmt_legend = self._check("Show legend", form, checked=True)
        if "legend" in fields or "legend_loc" in fields:
            self.fmt_legend_loc = QComboBox()
            self.fmt_legend_loc.addItems(self._LEGEND_LOCS)
            self.fmt_legend_loc.currentTextChanged.connect(self.changed)
            form.addRow("Legend position", self.fmt_legend_loc)
        if "legend" in fields:
            # Manual column count (the diel cycle uses "legend_loc" instead and
            # sets columns automatically from the month count).
            self.fmt_legend_ncol = self._spin(1, 1, 6, form, "Legend columns")

        self._col.addWidget(grp)

    def _format_values(self) -> dict:
        """FormatStyle-kwargs dict for whichever fmt_ widgets exist."""
        out: dict = {}
        if hasattr(self, "fmt_title"):
            out["title"] = self.fmt_title.text().strip() or None
        if hasattr(self, "fmt_xlabel"):
            out["xlabel"] = self.fmt_xlabel.text().strip() or None
        if hasattr(self, "fmt_ylabel"):
            out["ylabel"] = self.fmt_ylabel.text().strip() or None
        if hasattr(self, "fmt_xunits"):
            out["xunits"] = self.fmt_xunits.text().strip() or None
        if hasattr(self, "fmt_yunits"):
            out["yunits"] = self.fmt_yunits.text().strip() or None
        if hasattr(self, "fmt_zlabel"):
            out["zlabel"] = self.fmt_zlabel.text().strip() or None
        if hasattr(self, "fmt_title_fs"):
            out["title_fontsize"] = self.fmt_title_fs.value() or None
            out["axlabel_fontsize"] = self.fmt_axlabel_fs.value() or None
            out["ticks_fontsize"] = self.fmt_ticks_fs.value() or None
        if hasattr(self, "fmt_grid"):
            out["show_grid"] = self.fmt_grid.isChecked()
        if hasattr(self, "fmt_legend"):
            out["show_legend"] = self.fmt_legend.isChecked()
        if hasattr(self, "fmt_legend_loc"):
            out["legend_loc"] = self.fmt_legend_loc.currentText()
        if hasattr(self, "fmt_legend_ncol"):
            out["legend_ncol"] = self.fmt_legend_ncol.value()
        return out

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

    # --- raw widget state (for project save/restore) ---
    def _state_widgets(self) -> list:
        """Every persistable control, in deterministic build order. Collected as
        the field widget of each form row (so spinbox/combo *internal* editors are
        excluded, and the order is stable across two panels of the same type)."""
        from PySide6.QtWidgets import QFormLayout
        widgets = []
        for form in self.widget().findChildren(QFormLayout):
            for r in range(form.rowCount()):
                item = form.itemAt(r, QFormLayout.ItemRole.FieldRole)
                if item is not None and item.widget() is not None:
                    widgets.append(item.widget())
        return widgets

    def state(self) -> list:
        """Raw values of all controls (round-trips, unlike the transformed
        :meth:`values`). Positional — restored against the same plot type."""
        from diive.gui.widgets.state_utils import widget_value
        return [widget_value(w) for w in self._state_widgets()]

    def apply_state(self, values) -> None:
        """Re-apply a snapshot from :meth:`state` onto the controls."""
        from diive.gui.widgets.state_utils import set_widget_value
        for w, v in zip(self._state_widgets(), values or []):
            set_widget_value(w, v)

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
                "show_colormap": self.show_colormap.isChecked(),
                "zlabel": self.zlabel.text().strip() or None,
                "cb_digits_after_comma": "auto" if digits == "auto" else int(digits),
                "cb_extend": self.cb_extend.currentText(),
                "show_values": self.show_values.isChecked(),
                "show_values_n_dec_places": self.show_values_dec.value(),
                "show_values_fontsize": _font(self.show_values_fontsize),
                "cb_labelsize": _font(self.cb_labelsize),
                "_format": self._format_values(),
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
                "kd_kwargs": {"bandwidth": bw} if bw > 0 else None,
                "_format": self._format_values(),
            }
        if self._plot_type == DIELCYCLE:
            return {
                "agg": _DIELCYCLE_AGGS[self.dc_agg.currentText()],
                "band": _DIELCYCLE_BANDS[self.dc_band.currentText()],
                "each_month": self.dc_curves.currentText() == _DIELCYCLE_PERMONTH,
                "cmap": _DIELCYCLE_COLORSCHEMES[self.dc_colorscheme.currentText()],
                "marker": self.dc_marker.isChecked(),
                "markersize": self.dc_markersize.value(),
                "_format": self._format_values(),
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
                "_format": self._format_values(),
                "_axes": self._axes_values(),
            }
        if self._plot_type == CUMULATIVE:
            return {
                "units": self.cu_units.text().strip() or None,
                "digits_after_comma": self.cu_digits.value(),
                "show_title": self.cu_show_title.isChecked(),
                "fill": self.cu_fill.isChecked(),
                "_format": self._format_values(),
                "_axes": self._axes_values(),
            }
        if self._plot_type == WATERFALL:
            return {
                # __init__ params
                "series_units": self.wf_units.text().strip() or None,
                "resample": self.wf_resample.currentText().strip() or "D",
                "agg": self.wf_agg.currentText(),
                "uptake_is_negative": self.wf_uptake_negative.isChecked(),
                # plot() params (bar_width 0 -> auto/None)
                "digits_after_comma": self.wf_digits.value(),
                "color_uptake": self.wf_color_uptake.text().strip() or "#2196F3",
                "color_release": self.wf_color_release.text().strip() or "#F44336",
                "bar_width": self.wf_bar_width.value() or None,
                "show_connectors": self.wf_connectors.isChecked(),
                "_format": self._format_values(),
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
                "cb_digits_after_comma": int(self.cb_digits.currentText()),
                "cb_extend": self.cb_extend.currentText(),
                "show_colormap": self.show_colormap.isChecked(),
                "show_values": self.show_values.isChecked(),
                "show_values_n_dec_places": self.show_values_dec.value(),
                "show_values_fontsize": _font(self.show_values_fontsize),
                "cb_labelsize": _font(self.cb_labelsize),
                "_format": self._format_values(),
            }
        if self._plot_type == HEATMAP_XYZ:
            def _font(sp):
                return sp.value() or None
            return {
                # GridAggregator (binning) params
                "binning_type": self.xyz_binning_type.currentText(),
                "n_bins": self.xyz_nbins.value(),
                "aggfunc": self.xyz_aggfunc.currentText(),
                "min_n_vals_per_bin": self.xyz_min_n.value(),
                # plot() styling params
                "cmap": self._reverse_cmap(self.cmap.currentText().strip() or "RdYlBu_r",
                                           self.reverse_cmap.isChecked()),
                "vmin": self._float_or_none(self.vmin.text()),
                "vmax": self._float_or_none(self.vmax.text()),
                "color_bad": self.color_bad.currentText().strip() or "grey",
                "zlabel": self.zlabel.text().strip() or None,
                "cb_digits_after_comma": int(self.cb_digits.currentText()),
                "cb_extend": self.cb_extend.currentText(),
                "show_colormap": self.show_colormap.isChecked(),
                "show_values": self.show_values.isChecked(),
                "show_values_n_dec_places": self.show_values_dec.value(),
                "show_values_fontsize": _font(self.show_values_fontsize),
                "cb_labelsize": _font(self.cb_labelsize),
                "_format": self._format_values(),
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
                "_format": self._format_values(),
                "_axes": self._axes_values(),
            }
        if self._plot_type == WINDROSE:
            digits = self.wr_cb_digits.currentText()
            return {
                "agg": self.wr_agg.currentText(),
                "n_sectors": self.wr_nsectors.value(),
                "z_agg": self.wr_zagg.currentText(),
                "max_sector_labels": self.wr_max_labels.value(),
                "cmap": self._reverse_cmap(self.wr_cmap.currentText().strip() or "RdYlBu_r",
                                           self.wr_reverse_cmap.isChecked()),
                "color": self.wr_color.text().strip() or None,
                "vmin": self._float_or_none(self.wr_vmin.text()),
                "vmax": self._float_or_none(self.wr_vmax.text()),
                "show_colorbar": self.wr_show_colorbar.isChecked(),
                "cb_label": self.wr_cb_label.text().strip() or None,
                "cb_digits_after_comma": None if digits == "auto" else int(digits),
                "_format": self._format_values(),
            }
        if self._plot_type == HISTOGRAM:
            return {
                "n_bins": self.hist_nbins.value(),
                "highlight_peak": self.hist_peak.isChecked(),
                "show_counts": self.hist_counts.isChecked(),
                "show_info": self.hist_info.isChecked(),
                "show_title": self.hist_title.isChecked(),
                "show_zscores": self.hist_zscores.isChecked(),
                "show_zscore_values": self.hist_zvalues.isChecked(),
                "_format": self._format_values(),
            }
        if self._plot_type == SHIFTEDDIST:
            def _p(edit):
                return edit.text().strip() or None
            return {
                "ref_period": (_p(self.sd_ref_start), _p(self.sd_ref_end)),
                "comp_period": (_p(self.sd_comp_start), _p(self.sd_comp_end)),
                "ref_label": self.sd_ref_label.text().strip() or None,
                "comp_label": self.sd_comp_label.text().strip() or None,
                "zone_labels": _SHIFTEDDIST_ZONE_PRESETS[self.sd_zones.currentText()],
                "show_legend": self.sd_show_legend.isChecked(),
                "show_title": self.sd_show_title.isChecked(),
                "show_xaxis": self.sd_show_xaxis.isChecked(),
                "show_yaxis": self.sd_show_yaxis.isChecked(),
                "_format": self._format_values(),
            }
        if self._plot_type == TREERING:
            return {
                # __init__ param
                "resample_freq": self.tr_resample.currentText().strip() or "D",
                # which renderer: 'filled' (plot) or 'line' (plot_line)
                "style": _TREERING_STYLES[self.tr_style.currentText()],
                # shared plot()/plot_line() params
                "cmap": self._reverse_cmap(self.tr_cmap.currentText().strip() or "RdBu_r",
                                           self.tr_reverse_cmap.isChecked()),
                "vmin": self._float_or_none(self.tr_vmin.text()),
                "vmax": self._float_or_none(self.tr_vmax.text()),
                "show_month_labels": self.tr_month_labels.isChecked(),
                "show_month_lines": self.tr_month_lines.isChecked(),
                "show_year_labels": self.tr_year_labels.isChecked(),
                "show_year_separators": self.tr_year_separators.isChecked(),
                "year_label_frequency": self.tr_year_freq.value(),
                "cb_label": self.tr_cb_label.text().strip() or None,
                "cb_digits_after_comma": self.tr_cb_digits.value(),
                "cb_labelsize": self.tr_cb_labelsize.value() or None,
                # line-trace-only params
                "linewidth": self.tr_linewidth.value(),
                "alpha": self.tr_alpha.value(),
                "amplitude_scale": self.tr_amplitude.value(),
                "ring_width": self.tr_ring_width.value(),
                "_format": self._format_values(),
            }
        return {
            "linewidth": self.linewidth.value(),
            "alpha": self.alpha.value(),
            "marker": self.marker.isChecked(),
            "markersize": self.markersize.value(),
            "drop_gaps": self.drop_gaps.isChecked(),
            "color": self._line_color_value(),
            "color_by": self._color_by_value(),
            "color_by_cmap": self.colorby_cmap.currentText().strip() or "RdYlBu_r",
            "_format": self._format_values(),
            "_axes": self._axes_values(),
        }
