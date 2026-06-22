"""
PLOTTING: TIME SERIES
=====================

Time series plots with optional colour-by-value and interactive Bokeh output.

Part of the diive library: https://github.com/holukas/diive
"""
import os
import tempfile
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from bokeh.layouts import column
from bokeh.models import BoxZoomTool, PanTool, ResetTool, WheelZoomTool, WheelPanTool, UndoTool, \
    RedoTool, SaveTool, HoverTool, BoxSelectTool, RangeTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.styles.format import FormatStyle

# Material Design palette, matching diive's plotting conventions (CLAUDE.md):
# blue 500 for lines, blue-grey shades for ink/gridlines/reference.
_COLOR_LINE = '#2196F3'  # blue 500 — the series line/markers
_COLOR_INK = '#455A64'   # blue-grey 700 — text, spines, ticks, legend edge
_COLOR_GRID = '#B0BEC5'  # blue-grey 200 — subtle gridlines
_COLOR_ZERO = '#90A4AE'  # blue-grey 300 — zero reference line


# output_notebook()


class TimeSeries:
    """
    Create time series plots with matplotlib (static) or Bokeh (interactive).

    Two-phase design: separate data preparation (__init__) from rendering (plot).
    Phase 1 creates the plotter with data; Phase 2 renders with styling options.

    Methods:
        plot : Render static matplotlib plot with styling options
        plot_interactive : Render interactive Bokeh plot with zoom, pan, selection tools

    Example:
        See `examples/visualization/plot_timeseries.py` for matplotlib examples.
        See `examples/visualization/plot_timeseries_interactive.py` for Bokeh examples.
        See `examples/visualization/plot_timeseries_rangetool.py` for the RangeTool overview.
    """

    def __init__(self,
                 series: Series,
                 drop_gaps: bool = False,
                 color_series: Series = None):
        """
        Prepare time series data for plotting.

        Args:
            series: Data to plot (pandas Series with datetime index)
            drop_gaps: If True, drop missing values before plotting. Default False
                so gaps remain visible — matplotlib and Bokeh break the line at
                NaNs rather than drawing a continuous line across missing periods,
                which would misrepresent data coverage.
            color_series: Optional second series whose values colour the line (and
                markers) via a colormap instead of a single colour — e.g. colour a
                flux line by air temperature. Aligned to `series`' index; segments
                are coloured by the mean of their endpoints' values. When given,
                `plot()`'s `cmap`/`color_vmin`/`color_vmax`/`show_colorbar`/
                `color_label` apply and the scalar `color` argument is ignored.

        See Also:
            plot : Render the time series with matplotlib styling options
        """
        self.series = series.copy()
        self.varname = series.name
        if drop_gaps:
            self.series = self.series.dropna()
        # Align the colour-by series to the (possibly gap-dropped) index.
        self.color_series = (color_series.reindex(self.series.index)
                             if color_series is not None else None)
        self.color_name = color_series.name if color_series is not None else None

    def plot_interactive(self, height: int = 600, width: int = 1200, save_to_file: bool = False):
        """
        Render interactive time series plot using Bokeh.

        Provides interactive tools for data exploration: hover tooltips, zoom, pan,
        selection, and export. Useful for exploring large time series datasets.

        Args:
            height: Plot height in pixels (default: 600)
            width: Plot width in pixels (default: 1200)
            save_to_file: Save plot to HTML file (default: False, display in browser only)

        Tools Available:
            - Hover: Display date and value on mouse over
            - Box Zoom: Click and drag to zoom to region
            - Reset: Reset to original view
            - Pan: Click and drag to move across plot
            - Box Select: Select points in a region
            - Wheel Zoom: Scroll to zoom in/out
            - Undo/Redo: Undo and redo zoom/pan operations
            - Save: Export plot as PNG image

        Example:
            >>> ts = dv.plotting.TimeSeries(series=data)
            >>> ts.plot_interactive(height=800, width=1600)  # Larger interactive plot
        """
        # Handle file output: temp file if not saving, named file if saving
        if save_to_file:
            output_file(filename=f"{self.series.name}_interactive.html", title=self.series.name)
        else:
            # Use temporary directory so file is automatically cleaned up
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"bokeh_{id(self)}.html")
            output_file(filename=temp_file, title=self.series.name)

        # Bokeh needs dataframe
        df = pd.DataFrame()
        df['date'] = self.series.index
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M")
        df['value'] = self.series.to_numpy()

        # Convert dataframe for bokeh
        source = ColumnDataSource(df)

        # Modern scientific Bokeh styling
        p = figure(height=height,
                   width=width,
                   title=f"{self.series.name}",
                   tools=[
                       HoverTool(
                           tooltips=[('Date', '@date{%F %T}'),
                                     ('Value', '@value{0,0.00}')],
                           formatters={'@date': 'datetime'},
                           mode='mouse'
                       ),
                       BoxZoomTool(),
                       ResetTool(),
                       PanTool(),
                       BoxSelectTool(),
                       WheelZoomTool(),
                       WheelPanTool(),
                       UndoTool(),
                       RedoTool(),
                       SaveTool()],
                   x_axis_type='datetime',
                   background_fill_color='#FAFAFA',
                   border_fill_color='white')

        # Modern line styling (publication-ready)
        p.line(x='date', y='value', line_width=2.0, source=source, color=_COLOR_LINE, alpha=0.95,
               legend_label=self.series.name)
        p.scatter(x='date', y='value', size=4, source=source, color=_COLOR_LINE, alpha=0.6)

        # Modern axis styling
        p.yaxis.axis_label = self.series.name
        p.xaxis.axis_label = self.series.index.name

        # Modern typography and aesthetics
        p.title.text_font_size = '13pt'
        p.title.text_color = _COLOR_INK
        p.axis.axis_label_text_font_size = '11pt'
        p.axis.axis_label_text_color = _COLOR_INK
        p.axis.major_label_text_font_size = '10pt'
        p.axis.major_label_text_color = _COLOR_INK

        # Gridlines
        p.grid.grid_line_alpha = 0.2
        p.grid.grid_line_color = _COLOR_GRID
        p.grid.grid_line_width = 0.5

        # Legend styling
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        p.legend.background_fill_alpha = 0.8
        p.legend.border_line_width = 0.5

        # # https://stackoverflow.com/questions/61340741/get-bokehs-selection-in-notebook
        # from bokeh.models import CustomJS
        # # make a custom javascript callback that exports the indices of the selected points to the Jupyter notebook
        # callback = CustomJS(args=dict(s=source),
        #                     code="""
        #                          console.log('Running CustomJS callback now.');
        #                          var indices = s.selected.indices;
        #                          var kernel = IPython.notebook.kernel;
        #                          kernel.execute("selected_indices = " + indices)
        #                          """)
        #
        # # set the callback to run when a selection geometry event occurs in the figure
        # p.js_on_event('selectiongeometry', callback)
        # selected_indices

        # # Add hover tooltip
        # hover = HoverTool(
        #     tooltips=[
        #         ('Date', '@date{%F %T}'),
        #         ('Value', '@value')
        #     ],
        #     formatters={
        #         '@date':'datetime'
        #     },
        #     # Display a tooltip whenever the cursor is vertically in line with a glyph
        #     mode='vline'
        # )

        # hover.formatters = {'@date': 'datetime'}
        # p.add_tools(hover)

        # HoverTool(tooltips=[('date', '@DateTime{%F}')],
        #           formatters={'@DateTime': 'datetime'})

        # Show plot
        show(p)

    def plot_rangetool(self, height: int = 300, width: int = 900, overview_height: int = 130,
                       init_range: float = 0.25, save_to_file: bool = False):
        """
        Render an interactive Bokeh plot with a RangeTool overview.

        Two linked panels: a detail plot (top) showing a slice of the series, and a
        smaller overview plot (bottom) showing the full series with a draggable
        RangeTool. Drag or resize the shaded selection on the overview to pan and
        zoom the detail plot. The detail panel's y-axis auto-scales to the data in
        the visible x-window, so the curve always fills the panel. Useful for
        navigating long time series without losing the overall context.

        See https://docs.bokeh.org/en/latest/docs/examples/interaction/tools/range_tool.html

        Args:
            height: Detail-panel height in pixels (default: 300).
            width: Plot width in pixels (default: 900).
            overview_height: Overview-panel height in pixels (default: 130).
            init_range: Fraction of the series shown in the detail panel initially,
                measured from the start (default: 0.25 = first quarter).
            save_to_file: Save to a named HTML file instead of a temp file
                (default: False, opens in browser only).

        Example:
            >>> ts = dv.plotting.TimeSeries(series=data)
            >>> ts.plot_rangetool(init_range=0.1)  # start zoomed to the first 10%
        """
        if save_to_file:
            output_file(filename=f"{self.series.name}_rangetool.html", title=self.series.name)
        else:
            temp_file = os.path.join(tempfile.gettempdir(), f"bokeh_rangetool_{id(self)}.html")
            output_file(filename=temp_file, title=self.series.name)

        df = pd.DataFrame({'date': pd.to_datetime(self.series.index), 'value': self.series.to_numpy()})
        source = ColumnDataSource(df)

        # Initial detail window: the first `init_range` fraction of the record.
        n = len(df)
        end_ix = max(1, min(n - 1, int(n * init_range)))
        x_start, x_end = df['date'].iloc[0], df['date'].iloc[end_ix]

        # Detail panel (top): pan/zoom along x; x-axis on top like the Bokeh example.
        # window_axis='x' auto-scales the y-axis to the data inside the visible
        # x-window, so the detail view always fills the panel as you pan/zoom.
        detail = figure(height=height, width=width, x_axis_type='datetime', x_axis_location='above',
                        window_axis='x', x_range=(x_start, x_end), tools='xpan,xwheel_zoom,reset',
                        toolbar_location='right', background_fill_color='#FAFAFA',
                        border_fill_color='white', title=f"{self.series.name}")
        detail.line('date', 'value', source=source, line_width=2.0, color=_COLOR_LINE, alpha=0.95)
        detail.yaxis.axis_label = self.series.name
        detail.xaxis.axis_label = self.series.index.name

        # Overview panel (bottom): the full series with its own (full-range) y-axis,
        # independent of the detail panel's auto-scaling, plus the RangeTool linked
        # to the detail's x-range.
        overview = figure(height=overview_height, width=width,
                          x_axis_type='datetime', y_axis_type=None, tools='',
                          toolbar_location=None, background_fill_color='#FAFAFA',
                          border_fill_color='white')
        overview.x_range.range_padding = 0
        overview.x_range.bounds = 'auto'
        range_tool = RangeTool(x_range=detail.x_range, start_gesture='pan')
        range_tool.overlay.fill_color = _COLOR_LINE
        range_tool.overlay.fill_alpha = 0.2
        overview.line('date', 'value', source=source, line_width=1.2, color=_COLOR_LINE, alpha=0.7)
        overview.ygrid.grid_line_color = None
        overview.add_tools(range_tool)

        # Shared Material Design styling.
        for p in (detail, overview):
            p.title.text_color = _COLOR_INK
            p.axis.axis_label_text_color = _COLOR_INK
            p.axis.major_label_text_color = _COLOR_INK
            p.grid.grid_line_color = _COLOR_GRID
            p.grid.grid_line_alpha = 0.3

        show(column(detail, overview))

    def _plot_colored_line(self, color_vals, cmap, color_vmin, color_vmax,
                           linewidth, alpha, marker, markersize,
                           show_colorbar, color_label):
        """Draw the series as a `LineCollection` coloured by `color_vals`.

        Each segment between consecutive samples is coloured by the mean of its
        endpoints' colour values; segments touching a NaN in the data (a gap) are
        dropped so the line breaks rather than bridging. A `LineCollection` does
        not autoscale the axes, so x/y limits are set explicitly.
        """
        x = mdates.date2num(self.series.index)
        y = self.series.to_numpy(dtype=float)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_c = (color_vals[:-1] + color_vals[1:]) / 2.0
        keep = ~(np.isnan(y[:-1]) | np.isnan(y[1:]))  # skip gap-touching segments
        segments, seg_c = segments[keep], seg_c[keep]

        vmin = color_vmin if color_vmin is not None else np.nanmin(color_vals)
        vmax = color_vmax if color_vmax is not None else np.nanmax(color_vals)
        norm = Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_c)
        lc.set_linewidth(linewidth)
        lc.set_alpha(alpha)
        lc.set_zorder(99)
        self.ax.add_collection(lc)

        if marker:
            self.ax.scatter(x, y, c=color_vals, cmap=cmap, norm=norm,
                            s=markersize ** 2, edgecolors='none', alpha=alpha, zorder=100)

        # LineCollection does not autoscale; set limits from the data.
        self.ax.set_xlim(np.nanmin(x), np.nanmax(x))
        finite = np.isfinite(y)
        if finite.any():
            ymin, ymax = float(np.min(y[finite])), float(np.max(y[finite]))
            pad = (ymax - ymin) * 0.05 or 1.0
            self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.xaxis_date()

        if show_colorbar:
            cb = self.ax.figure.colorbar(lc, ax=self.ax, pad=0.01)
            cb.set_label(color_label or self.color_name or '')

    def plot(self, ax=None, format_style: FormatStyle = None, color: str = None,
             linewidth: float = 2.2, alpha: float = 0.95, marker: bool = False, markersize: float = 3,
             cmap: str = 'viridis', color_vmin: float = None, color_vmax: float = None,
             show_colorbar: bool = True, color_label: str = None):
        """
        Render time series plot with matplotlib styling (Phase 2 of two-phase design).

        Chrome (title, labels, units, font sizes, colours, grid, legend) comes from
        a shared :class:`~diive.plotting.FormatStyle` so it looks and is configured
        the same way as every other diive plot. The data-rendering arguments
        (``color``/``linewidth``/``alpha``/``marker``/``markersize``) are specific to
        the line itself and stay here. Can be called multiple times on the same
        object to draw on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates a new figure and displays it.
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome
                (title, labels, units, fonts, colours, grid, legend). When None the
                diive house style is used.
            color: Line color (default: Material Design blue #2196F3). Ignored when
                a `color_series` was given (the line is then colormap-coloured).
            linewidth: Line width in points (default: 2.2).
            alpha: Line/marker opacity, 0-1 (default: 0.95).
            marker: If True, draw a point marker at each observation (default: False).
            markersize: Point marker size in points, used only when marker=True (default: 3).
            cmap: Colormap used when a `color_series` was given (default: 'viridis').
            color_vmin: Lower bound of the colour scale (default: the color_series min).
            color_vmax: Upper bound of the colour scale (default: the color_series max).
            show_colorbar: Draw a colorbar for the colour-by series (default: True).
            color_label: Colorbar label (default: the color_series name).

        Returns:
            The matplotlib axes the series was drawn on (the new axes when ax=None),
            so callers can apply further styling.

        Example:
            >>> ts = dv.plotting.TimeSeries(series=data)
            >>> ax = ts.plot(ax=ax1, color='#2196F3')  # Plot on axis
            >>> style = dv.plotting.FormatStyle(title='Custom', yunits='(°C)')
            >>> ts.plot(format_style=style)  # New figure with a shared style
        """
        # Chrome comes only from the caller-supplied style.
        style = format_style or FormatStyle()

        # Create axis
        if ax is not None:
            # If ax is given, plot directly to ax, no fig needed
            self.fig = None
            self.ax = ax
            self.showplot = False
        else:
            # If no ax is given, create fig and ax and then show the plot
            self.fig, self.ax = pf.create_ax()
            self.ax.xaxis.axis_date()
            self.showplot = True

        color = color if color else _COLOR_LINE

        # Colour-by-variable mode: draw the line as a colormap-coloured
        # LineCollection. Falls back to a plain line if the colour series has
        # fewer than two finite values to scale.
        color_vals = (self.color_series.to_numpy(dtype=float)
                      if self.color_series is not None else None)
        if color_vals is not None and np.isfinite(color_vals).sum() >= 2:
            self._plot_colored_line(color_vals, cmap, color_vmin, color_vmax,
                                    linewidth, alpha, marker, markersize,
                                    show_colorbar, color_label)
        else:
            # NaNs are kept (unless drop_gaps=True) so the line breaks at gaps
            # instead of bridging them.
            self.ax.plot(self.series.index,
                         self.series,
                         color=color, alpha=alpha,
                         linewidth=linewidth,
                         linestyle='-',
                         marker='o' if marker else None,
                         markersize=markersize if marker else 0,
                         markeredgecolor='none',
                         zorder=99, label=self.series.name)

        # Shared formatting layer: title/labels/units/fonts/grid/legend/zeroline.
        style.apply(ax=self.ax, default_title=self.series.name, default_xlabel='Date',
                    default_ylabel=self.varname, zeroline_data=self.series)

        # Nice date ticks (time-series specific, not part of the shared chrome).
        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        if self.showplot:
            self.fig.patch.set_facecolor('white')
            self.fig.tight_layout(pad=1.2)
            self.fig.show()

        return self.ax
