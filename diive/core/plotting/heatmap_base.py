"""
HEATMAP
=======

Base class and shared utilities for all heatmap plot types in diive.

``HeatmapBase`` is not used directly.  Subclasses (``HeatmapDateTime``,
``HeatmapYearMonth``, ``HeatmapXYZ``) call ``super().__init__()`` to inherit
figure / axis management, colormap handling, colorbar formatting, and the
``show_vals_in_plot`` overlay.  Each subclass is responsible for shaping its
data into ``self.x``, ``self.y``, and ``self.z`` arrays before calling
``plot_pcolormesh``.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from diive.core.io.files import verify_dir
from diive.core.plotting.plotfuncs import default_format, format_spines, hide_xaxis_yaxis, hide_ticks_and_ticklabels, \
    make_patch_spines_invisible
from diive.core.plotting.styles import LightTheme as theme
from diive.core.times.times import TimestampSanitizer, insert_timestamp


class HeatmapBase:
    """Base class for all heatmap plot types.

    Provides figure/axis creation, colormap configuration, colorbar formatting,
    NaN masking, and cell-value overlays.  Subclasses must populate ``self.x``,
    ``self.y``, and ``self.z`` before calling :meth:`plot_pcolormesh`.

    Subclasses:
        - :class:`HeatmapDateTime` — time series as date × time-of-day grid.
        - :class:`HeatmapYearMonth` — time series aggregated into a year × month grid.
        - :class:`HeatmapXYZ` — arbitrary 2-D scatter data binned into a grid.

    Do not instantiate this class directly.
    """

    def __init__(self,
                 fig=None,
                 figsize: tuple = None,
                 figdpi: int = 72,
                 ax=None,
                 ax_orientation: str = "vertical",
                 title: str = None,
                 vmin: float = None,
                 vmax: float = None,
                 cb_digits_after_comma: int = 2,
                 cb_labelsize: float = theme.AX_LABELS_FONTSIZE,
                 cb_extend: str = 'neither',
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE,
                 minticks: int = 3,
                 maxticks: int = 10,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'grey',
                 zlabel: str = None,
                 show_colormap: bool = True,
                 show_less_xticklabels: bool = False,
                 show_values: bool = False,
                 show_values_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 show_values_n_dec_places: int = 0,
                 show_grid: bool = False,
                 heatmaptype: str = None,
                 verbose: bool = False):
        """Stores all shared configuration and creates the figure/axes when needed.

        All parameters are optional keyword arguments that subclasses forward via
        ``**kwargs``.  Callers typically only need to set the parameters relevant
        to their use case; sensible defaults are provided for everything else.

        Args:
            fig: Existing :class:`matplotlib.figure.Figure` to draw on.
                 When *None* a new figure is created with the given ``figsize``
                 and ``figdpi``.  Ignored if ``ax`` is provided.
            figsize: ``(width, height)`` in inches for the new figure.
                     Only used when ``ax`` is *None*.
            figdpi: Resolution of the new figure in dots per inch.
                    Only used when ``ax`` is *None*.  Defaults to 72.
            ax: Existing :class:`matplotlib.axes.Axes` to draw into.
                When *None* a fresh figure and axes are created.  Pass an axes
                when composing multiple heatmaps in one figure.
            ax_orientation: Layout of the date/time axes.
                ``'vertical'`` (default) — date on y, time-of-day on x.
                ``'horizontal'`` — date on x, time-of-day on y.
            title: Plot title.  When *None* an auto-title is generated from the
                   series name and frequency string.  Pass ``title=""`` to suppress.
            vmin: Lower bound of the colour scale.  *None* = auto from data.
            vmax: Upper bound of the colour scale.  *None* = auto from data.
            cb_digits_after_comma: Decimal places shown on colorbar tick labels.
                Defaults to 2.
            cb_labelsize: Font size for colorbar tick labels.
            cb_extend: Colorbar extension arrows.  One of ``'neither'`` (default),
                       ``'both'``, ``'min'``, or ``'max'``.  Use ``'both'`` when
                       ``vmin``/``vmax`` clip the data range.
            axlabels_fontsize: Font size for x-axis and y-axis labels.
            ticks_labelsize: Font size for tick mark labels on both axes.
            minticks: Minimum number of major ticks on date axes.  Defaults to 3.
            maxticks: Maximum number of major ticks on date axes.  Defaults to 10.
            cmap: Matplotlib colormap name (e.g. ``'RdYlBu_r'``, ``'viridis'``).
            color_bad: Colour used to fill cells where the value is NaN.
                       Defaults to ``'grey'``.
            zlabel: Colorbar label describing what the colour encodes
                    (e.g. ``'°C'``, ``'µmol CO₂ m⁻² s⁻¹'``).
            show_colormap: Whether to render the colorbar.  Set to *False* to
                           suppress it (e.g. when composing a multi-panel figure
                           with a shared colorbar).  Defaults to *True*.
            show_less_xticklabels: Hide every second x-axis tick label to reduce
                                   crowding on dense time axes.  Defaults to *False*.
            show_values: Overlay each cell with its numeric z-value.
                         Only practical for coarse grids (year × month).
                         Defaults to *False*.
            show_values_fontsize: Font size for the cell-value overlay text.
            show_values_n_dec_places: Decimal places for the cell-value overlay.
                                      Defaults to 0.
            show_grid: Draw a grid on the axes.  Defaults to *False*.
            heatmaptype: Internal tag set by subclasses to select the correct
                         cell-centering logic in :meth:`show_vals_in_plot`.
                         Valid values: ``'yearmonth'``, ``'xyz'``,
                         ``'datetime'``, ``'datetime_horizontal'``.
                         Do not set this manually.
            verbose: Print progress and diagnostic messages.  Defaults to *False*.
        """

        self.verbose = verbose

        self.fig = fig
        self.figsize = figsize
        self.figdpi = figdpi
        self.ax = ax
        self.ax_orientation = ax_orientation

        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cb_digits_after_comma = cb_digits_after_comma
        self.cb_labelsize = cb_labelsize
        self.cb_extend = cb_extend
        self.color_bad = color_bad

        # Create fig and axis if no axis is given, otherwise use given axis
        if ax is None:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=self.figsize, dpi=self.figdpi)

        self.axlabels_fontsize = axlabels_fontsize
        self.ticks_labelsize = ticks_labelsize
        self.minticks = minticks
        self.maxticks = maxticks
        self.zlabel = zlabel

        self.show_less_xticklabels = show_less_xticklabels
        self.show_values = show_values
        self.showvalues_fontsize = show_values_fontsize
        self.showvalues_n_dec_places = show_values_n_dec_places
        self.heatmaptype = heatmaptype
        self.show_colormap = show_colormap
        self.show_grid = show_grid

        self.plotdf = None
        self.x = None
        self.y = None
        self.z = None

    def _setup_timestamp(self, series: pd.Series) -> pd.Series:
        """Sanitizes the time series index so it is ready for pivot-based heatmap plotting.

        Runs :class:`~diive.core.times.times.TimestampSanitizer` to convert the
        index to datetime, sort ascending, remove duplicate timestamps, and
        regularise gaps (fill missing timestamps with NaN rows).  Regularisation
        ensures the pivot table forms a complete rectangular grid without missing
        time slots — a requirement for ``pcolormesh``.

        After sanitization the index is forced to the ``TIMESTAMP_START``
        convention.  Using start-of-period timestamps means the x-axis tick for
        hour 1 appears *before* the cell that covers 01:00–01:59, which matches
        the natural reading direction of a heatmap.

        Args:
            series: Input pandas Series with a datetime-like index.  The index
                    must follow diive's timestamp naming convention
                    (``TIMESTAMP_START``, ``TIMESTAMP_MIDDLE``, or
                    ``TIMESTAMP_END``).

        Returns:
            pd.Series: Sanitized series with a ``TIMESTAMP_START`` DatetimeIndex
                       and a detected/inferred frequency set.
        """

        series = TimestampSanitizer(
            data=series,
            output_middle_timestamp=False,
            validate_naming=True,
            convert_to_datetime=True,
            sort_ascending=True,
            remove_duplicates=True,
            regularize=True,
            verbose=self.verbose
        ).get()

        # Heatmap works best with TIMESTAMP_START, convert timestamp index if needed
        # Working with a timestamp that shows the start of the averaging period means
        # that e.g. the ticks on the x-axis for hours is shown before the corresponding
        # data between e.g. 1:00 and 1:59.
        if series.index.name != 'TIMESTAMP_START':
            _series_df = insert_timestamp(data=series, convention='start', set_as_index=True)
            series = _series_df[series.name].copy()
        return series

    def get_ax(self):
        """Returns the Matplotlib Axes object where the plot was generated.

        Returns:
            matplotlib.axes.Axes: The Axes object containing the heatmap.
        """
        return self.ax

    def get_plot_data(self) -> pd.DataFrame:
        """Returns the pivot DataFrame used to plot the heatmap.

        This DataFrame contains the reshaped data (`self.plotdf`) that forms
        the grid of values for the heatmap.

        Returns:
            pd.DataFrame: The DataFrame used for plotting.
        """
        return self.plotdf

    def plot_pcolormesh(self, shading: str = None):
        """Renders ``self.x``, ``self.y``, ``self.z`` as a ``pcolormesh`` plot.

        Calls :meth:`set_cmap` first to copy the colormap and mask NaN values in
        ``z`` so they are rendered in ``color_bad`` instead of the nearest valid
        colour.  The masked array is passed directly to ``pcolormesh`` — *not*
        the original unmasked ``self.z`` — so missing cells are always visible.

        Args:
            shading: Shading algorithm passed to ``pcolormesh``.  ``'flat'``
                     assigns the cell colour from the value at the lower-left
                     corner; ``'auto'`` / *None* lets matplotlib choose based on
                     whether ``x``/``y`` are the same length as ``z`` or one
                     longer (boundary mode).  Defaults to *None*.

        Returns:
            matplotlib.collections.QuadMesh: The mesh object, which is needed
            by :meth:`format` to attach the colorbar.
        """
        cmap, z = self.set_cmap(cmap=self.cmap, color_bad=self.color_bad, z=self.z)
        p = self.ax.pcolormesh(self.x, self.y, z,
                               linewidths=1, cmap=cmap,
                               vmin=self.vmin, vmax=self.vmax,
                               shading=shading, zorder=99)
        return p

    def show(self):
        """Generates the heatmap plot and displays the figure.

        This method calls the `plot` method (which should be implemented by
        subclasses) to draw the heatmap and then uses `self.fig.show()`
        to display the generated Matplotlib figure.
        """
        self.plot()
        # plt.tight_layout()
        self.fig.show()

    @staticmethod
    def set_cmap(cmap, color_bad, z):
        """Prepares a colormap and masks invalid data values.

        The colormap is copied before mutation because calling ``set_bad`` on a
        shared global colormap would affect every other plot that uses the same
        colormap in the same session (see `matplotlib issue #17634
        <https://github.com/matplotlib/matplotlib/issues/17634>`_).

        Args:
            cmap (str): Matplotlib colormap name (e.g. ``'RdYlBu_r'``).
            color_bad (str): Colour string for NaN / masked cells
                             (e.g. ``'grey'``, ``'#cccccc'``).
            z (np.ndarray): 2-D array of heatmap values, may contain NaN.

        Returns:
            tuple:
                - **cmap** (*matplotlib.colors.Colormap*) — deep copy of the
                  requested colormap with the bad-value colour applied.
                - **z** (*np.ma.MaskedArray*) — input array with NaN and ±inf
                  entries masked so ``pcolormesh`` renders them in ``color_bad``.
        """
        cmap = copy.copy(plt.get_cmap(cmap))  # Needed for now, https://github.com/matplotlib/matplotlib/issues/17634
        cmap.set_bad(color=color_bad, alpha=1.)  # Set missing data to specific color
        z = np.ma.masked_invalid(z)  # Mask NaN as missing
        return cmap, z

    def export_borderless_heatmap(self, outpath: str):
        """Exports borderless heatmap images for use as textures or heightmaps in 3-D software.

        Generates four 300 dpi PNG files — two greyscale heightmaps (normal and
        reversed) and two colour textures (``jet`` normal and reversed).  Each
        image is rendered without any axes, ticks, labels, or borders so it can
        be loaded directly into Blender or similar tools.

        Output filenames follow the pattern
        ``heatmap_{plottype}_{series_name}.png`` where ``plottype`` is one of
        ``heightmap_bw``, ``heightmap_bw_r``, ``texture_color``,
        ``texture_color_r``.

        .. note::
            This method is experimental and may change in future versions.

        Args:
            outpath: Directory where the PNG files are saved.
                     Created automatically if it does not exist.
        """
        # https://www.youtube.com/watch?v=BXDSfrzR0zI

        # First, check if output directory exists
        verify_dir(path=outpath)

        # # todo? Smooth z with rolling mean
        # # _plot_df = self.plot_df.copy()
        # plot_df['z_rolling_median'] = plot_df['z'].copy()
        # vmin = plot_df['z_rolling_median'].quantile(0.01)
        # vmax = plot_df['z_rolling_median'].quantile(0.99)
        # # _plot_df['z_rolling_median'] = self.plot_df['z'].rolling(window=3, center=True, min_periods=2).mean()
        # # _plot_df['z_rolling_median'] = _plot_df['z_rolling_median'].rolling(window=12, center=True, min_periods=3).mean()

        plots_list = ['heightmap_bw', 'heightmap_bw_r', 'texture_color', 'texture_color_r']  # _r is reverse cmap

        cmap = None
        for plottype in plots_list:
            if plottype == 'heightmap_bw':
                cmap = 'Greys'
            elif plottype == 'heightmap_bw_r':
                cmap = 'Greys_r'
            elif plottype == 'texture_color':
                cmap = 'jet'
            elif plottype == 'texture_color_r':
                cmap = 'jet_r'

            # Figure without borders for rendering, b/w used as heightmap, colored used as texture
            fig_render_bw = plt.figure(figsize=(12, 12), frameon=False)  # frameon False = borderless
            ax_render_bw = fig_render_bw.add_axes((0, 0, 1, 1))  # left, bottom, width, height

            p = ax_render_bw.pcolormesh(self.x, self.y, self.z,
                                        linewidths=0, cmap=cmap, antialiased=False,
                                        vmin=self.vmin, vmax=self.vmax,
                                        shading='flat', zorder=98, edgecolor='none')

            # For heightmap and texture, hide all ticks, labels etc, only plot is needed
            make_patch_spines_invisible(ax=ax_render_bw)
            hide_ticks_and_ticklabels(ax=ax_render_bw)
            hide_xaxis_yaxis(ax=ax_render_bw)

            # todo? if 'normal' in plot:
            #     # For normal plot, output also colorbar
            #     cb = plt.colorbar(p, ax=ax_render_bw, format=f"%.{int(self.drp_digits_after_comma.currentText())}f")
            #     cb.ax.tick_params(labelsize=theme.FONTSIZE_LABELS_AXIS * 2)
            #     # cbytick_obj = plt.getp(cb.axes_dict, 'yticklabels')  # Set y tick label color
            #     # plt.setp(cbytick_obj, color='#999c9f', fontsize=12)

            # Save to file
            filename_out = f"heatmap_{plottype}_{self.series.name}.png"
            filepath_out = Path(outpath) / filename_out
            fig_render_bw.savefig(filepath_out,
                                  format='png',
                                  # bbox_inches='tight',
                                  # facecolor='w',
                                  # edgecolor='red',
                                  transparent=True,
                                  dpi=300)

            fig_render_bw.show()

    def show_vals_in_plot(self):
        """Overlays the numeric z-value on every heatmap cell.

        Iterates over all ``(i, j)`` cells of ``self.z`` and calls
        ``ax.text`` at the centre of each cell.  NaN cells are skipped
        (rendered as an empty string).

        Cell-centre coordinates are computed differently depending on
        ``self.heatmaptype`` because the axis dtypes differ:

        - ``'xyz'`` / ``'datetime'`` — plain numeric midpoint
          ``(a + b) / 2``.
        - ``'yearmonth'`` — integer offset ``x + 0.5`` (months/years
          are 1-indexed integers with unit cell width).
        - ``'datetime_horizontal'`` — numeric midpoint on the float-hour
          y-axis; ``timedelta`` midpoint on the ``datetime.date`` x-axis.

        Raises:
            NotImplementedError: If ``self.heatmaptype`` is not one of the
                supported values listed above.

        Note:
            Overlaying values on a full-resolution ``HeatmapDateTime`` with
            thousands of cells produces an unreadable result.  This method is
            most useful for coarse grids such as ``HeatmapYearMonth``.
        """
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                # Calculate the center coordinates for each heatmap type
                if self.heatmaptype == 'xyz':
                    x_center = (self.x[j] + self.x[j + 1]) / 2
                    y_center = (self.y[i] + self.y[i + 1]) / 2
                elif self.heatmaptype == 'yearmonth':
                    x_center = self.x[j] + 0.5
                    y_center = self.y[i] + 0.5
                elif self.heatmaptype == 'datetime':
                    # vertical: x-axis = float hours, y-axis = datetime.date
                    x_center = (self.x[j] + self.x[j + 1]) / 2
                    y_center = self.y[i] + (self.y[i + 1] - self.y[i]) // 2
                elif self.heatmaptype == 'datetime_horizontal':
                    # horizontal: x-axis = datetime.date, y-axis = float hours
                    x_center = self.x[j] + (self.x[j + 1] - self.x[j]) // 2
                    y_center = (self.y[i] + self.y[i + 1]) / 2
                else:
                    raise NotImplementedError(
                        f"show_vals_in_plot is not implemented for heatmaptype={self.heatmaptype!r}"
                    )

                val = self.z[i, j]
                if not np.isnan(val):  # Check if number, skips part if NaN
                    valstr = f"{val:.{self.showvalues_n_dec_places}f}"
                else:
                    valstr = ""
                self.ax.text(x_center, y_center, valstr,
                             ha='center', va='center', color='black', fontsize=self.showvalues_fontsize, zorder=100)

    def format(self, ax_xlabel_txt, ax_ylabel_txt, plot, shown_freq: str = None):
        """Applies title, colorbar, axis labels, tick formatting, and spine styling.

        Title logic:

        - If ``self.title`` was set explicitly it is used as-is.
        - If ``shown_freq`` is given and ``self.title`` is *None*, the title is
          auto-generated as ``"{series.name} ({shown_freq})"``.
        - If both are *None* / empty, no title is shown.

        Args:
            ax_xlabel_txt (str): Label for the x-axis.
            ax_ylabel_txt (str): Label for the y-axis.
            plot (matplotlib.collections.QuadMesh): Mesh object returned by
                :meth:`plot_pcolormesh`, used to create the colorbar.
            shown_freq (str, optional): Frequency or aggregation descriptor
                appended to the auto-generated title (e.g. ``'30min'``,
                ``'mean, MS'``).  Pass *None* to omit.
        """
        if shown_freq:
            title = self.title if self.title else f"{self.series.name} ({shown_freq})"
        else:
            title = self.title if self.title else ""
        self.ax.set_title(title, color='black', size=self.axlabels_fontsize)
        # Colorbar
        # Inside your class, assuming self.ax is an Axes object
        fig = self.ax.get_figure()
        if self.show_colormap:
            cb = fig.colorbar(plot, ax=self.ax, format=f"%.{int(self.cb_digits_after_comma)}f",
                              label=self.zlabel, extend=self.cb_extend)
            cb.set_label(label=self.zlabel, size=self.axlabels_fontsize, labelpad=20)
            cb.ax.tick_params(labelsize=self.cb_labelsize)
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ax_ylabel_txt,
                       ticks_direction='out', ticks_length=4, ticks_width=2,
                       ax_labels_fontsize=self.axlabels_fontsize, showgrid=self.show_grid,
                       ticks_labels_fontsize=self.ticks_labelsize)
        format_spines(ax=self.ax, color='black', lw=2)
        self.ax.tick_params(left=True, right=False, top=False, bottom=True)


def list_of_colormaps() -> list:
    """Return all colormap names registered in the current matplotlib session.

    Returns:
        list[str]: Sorted list of colormap name strings (e.g. ``['Accent',
        'Blues', …, 'viridis']``).  Both built-in and any user-registered
        colormaps are included.
    """
    return plt.colormaps()


if __name__ == '__main__':
    cmaps = list_of_colormaps()
    print(cmaps)
