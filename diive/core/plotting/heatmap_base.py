"""
HEATMAP
=======
"""
import copy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from diive.core.io.files import verify_dir
from diive.core.plotting.plotfuncs import default_format, format_spines, hide_xaxis_yaxis, hide_ticks_and_ticklabels, \
    make_patch_spines_invisible
from diive.core.plotting.styles import LightTheme as theme
from diive.core.times.times import TimestampSanitizer, insert_timestamp


class HeatmapBase:
    """
    Base class for generating various types of heatmap visualizations.

    This class provides the fundamental structure and common functionalities
    required for creating heatmaps, including figure and axis management,
    color mapping, handling of missing values, and basic plot formatting.
    It is designed to be extended by specific heatmap types (e.g., HeatmapDateTime,
    HeatmapYearMonth) that prepare their data in a specific way.
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
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE,
                 minticks: int = 3,
                 maxticks: int = 10,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'grey',
                 zlabel: str = None,
                 show_less_xticklabels: bool = False,
                 show_values: bool = False,
                 show_values_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 show_values_n_dec_places: int = 0,
                 heatmaptype: str = None,
                 verbose: bool = False):
        """
        Initialize the HeatmapBase class for generating heatmap visualizations.

        Args:
            fig: Matplotlib Figure object to plot on. If None, a new figure will be created.
            figsize: Tuple (width, height) in inches, specifying the size of the figure. Only used if `fig` is None.
            figdpi: The resolution of the figure in dots per inch (DPI). Only used if `fig` is None.
            ax: Matplotlib Axes object in which the heatmap is shown. If None, a new figure with an axis will be generated.
            ax_orientation: Orientation of the heatmap. Options are 'vertical' (default, y-axis is primary grouping, x-axis is secondary) or 'horizontal'.
            title: Text string to be shown at the top of the plot as its main title.
            vmin: Minimum value for the color scale of the heatmap. If None, it will be determined automatically from the data.
            vmax: Maximum value for the color scale of the heatmap. If None, it will be determined automatically from the data.
            cb_digits_after_comma: The number of digits after the comma to display in the colorbar legend.
            cb_labelsize: Font size for the labels on the colorbar.
            axlabels_fontsize: Font size for the x and y axis labels.
            ticks_labelsize: Font size for the tick labels on both axes.
            minticks: Minimum number of major ticks to display.
            maxticks: Maximum number of major ticks to display.
            cmap: Name of a Matplotlib colormap (e.g., 'viridis', 'plasma', 'RdYlBu_r'). This colormap will be used for the heatmap colors.
            color_bad: Color used to represent missing (NaN) values in the heatmap.
            zlabel: Label for the colorbar, typically describing what the color intensity represents (e.g., 'Temperature', 'Value').
            show_less_xticklabels: If True, attempts to reduce the number of x-axis tick labels for better readability, especially when dealing with a dense time series.
            show_values: If True, the actual numerical z-values will be overlaid on top of the heatmap cells.
            show_values_fontsize: Font size for the numerical values displayed on the heatmap cells if `show_values` is True.
            show_values_n_dec_places: Number of decimal places to format the displayed values if `show_values` is True. This parameter is only considered if `show_values` is set to True.
            heatmaptype: A string indicating the type of heatmap (e.g., 'yearmonth', 'xyz'). Used internally for specific formatting logic.
            verbose: If True, enables verbose output for debugging or informational messages during heatmap generation.
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
        self.color_bad = color_bad

        # Create fig and axis if no axis is given, otherwise use given axis
        if not ax:
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

        self.plotdf = None
        self.x = None
        self.y = None
        self.z = None

    def _setup_timestamp(self, series: pd.Series) -> pd.Series:
        """Sanitizes and prepares the time series index for heatmap plotting.

        This method ensures that the time series index is in a consistent
        and appropriate format for heatmap generation. It uses `TimestampSanitizer`
        to perform operations like converting to datetime, sorting, removing
        duplicates, and regularizing the time series. It also ensures the
        timestamp convention is 'TIMESTAMP_START', which is optimal for
        representing data within a given period on a heatmap.

        Args:
            series: The input pandas Series with a datetime-like index.

        Returns:
            pd.Series: The sanitized and prepared pandas Series with a
                       'TIMESTAMP_START' index.
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
        """Generates the core pcolormesh plot for the heatmap.

        This method sets up the colormap, handles missing values, and then
        uses `matplotlib.pyplot.pcolormesh` to draw the heatmap. It uses
        the `x`, `y`, and `z` arrays prepared by the subclass.

        Args:
            shading: The shading method for `pcolormesh`. Can be 'flat' or 'gouraud'.
                     Defaults to None, letting Matplotlib choose based on data.

        Returns:
            matplotlib.collections.QuadMesh: The QuadMesh object returned by `pcolormesh`.
        """
        cmap, z = self.set_cmap(cmap=self.cmap, color_bad=self.color_bad, z=self.z)
        # self.z = self.z.reshape(-1, 1)
        p = self.ax.pcolormesh(self.x, self.y, self.z,
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
        """Sets the colormap and handles missing values for plotting.

        This static method takes a colormap name, a color for bad/missing values,
        and the data array. It creates a copy of the colormap and sets the
        color for NaN values. It also masks invalid (NaN) values in the data
        array so that the colormap's 'bad' color is applied.

        Args:
            cmap (str): The name of the colormap to use.
            color_bad (str): The color string for missing values.
            z (np.ndarray): The 2D numpy array of data values.

        Returns:
            tuple: A tuple containing:
                   - `cmap`: The modified Matplotlib colormap object.
                   - `z`: The masked numpy array with invalid values masked.
        """
        cmap = copy.copy(plt.get_cmap(cmap))  # Needed for now, https://github.com/matplotlib/matplotlib/issues/17634
        cmap.set_bad(color=color_bad, alpha=1.)  # Set missing data to specific color
        z = np.ma.masked_invalid(z)  # Mask NaN as missing
        return cmap, z

    def export_borderless_heatmap(self, outpath: str):
        # TODO----------
        """Saves borderless heatmap plots (grayscale heightmap and color texture)
        for external rendering purposes, such as in Blender.

        This method generates different versions of the heatmap (grayscale for
        heightmaps and color for textures, with normal and reversed colormaps).
        It creates a new figure and axis for each export, ensuring no borders,
        ticks, or labels are present, making them suitable for direct use as
        image textures or height maps in 3D rendering software.

        Args:
            outpath: The directory path where the generated image files will be saved.
                     The directory will be created if it does not exist.
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
        filename = None
        for plottype in plots_list:
            # if plot == 'normal':
            #     cmap = 'jet'
            #     str = 'NORMAL'
            # if plot == 'normal_r':
            #     cmap = 'jet_r'
            #     str = 'NORMAL_REVERSE'
            if plottype == 'heightmap_bw':
                cmap = 'Greys'
                filename = 'HEIGHTMAP'
            elif plottype == 'heightmap_bw_r':
                cmap = 'Greys_r'
                filename = 'HEIGHTMAP_REVERSE'
            elif plottype == 'texture_color':
                cmap = 'jet'
                filename = 'TEXTURE'
            elif plottype == 'texture_color_r':
                cmap = 'jet_r'
                filename = 'TEXTURE_REVERSE'

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
        """Overlays numerical values onto the heatmap cells.

        This method iterates through each cell of the heatmap (`self.z` data)
        and places the corresponding numerical value as text within the cell.
        It calculates the center coordinates for each cell based on the
        `heatmaptype` and formats the value according to `show_values_n_dec_places`.
        NaN values are displayed as empty strings.
        """
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                # Calculate the center coordinates
                if self.heatmaptype == 'xyz':
                    x_center = (self.x[j] + self.x[j + 1]) / 2
                    y_center = (self.y[i] + self.y[i + 1]) / 2
                elif self.heatmaptype == 'yearmonth':
                    x_center = self.x[j] + 0.5
                    y_center = self.y[i] + 0.5
                else:
                    raise NotImplementedError

                val = self.z[i, j]
                if not np.isnan(val):  # Check if number, skips part if NaN
                    valstr = f"{val:.{self.showvalues_n_dec_places}f}"
                else:
                    valstr = ""
                self.ax.text(x_center, y_center, valstr,
                             ha='center', va='center', color='black', fontsize=self.showvalues_fontsize, zorder=100)

    def format(self, ax_xlabel_txt, ax_ylabel_txt, plot, shown_freq: str = None):
        """Applies general formatting to the heatmap plot.

        This method sets the plot title, configures the colorbar, and applies
        default axis and spine formatting.

        Args:
            ax_xlabel_txt (str): The label for the x-axis.
            ax_ylabel_txt (str): The label for the y-axis.
            plot (matplotlib.collections.QuadMesh): The QuadMesh object returned by `pcolormesh`.
            shown_freq (str, optional): A string indicating the frequency or aggregation
                                        method of the data, which will be appended to the
                                        title if provided. Defaults to None.
        """
        if shown_freq:
            title = self.title if self.title else f"{self.series.name} ({shown_freq})"
        else:
            title = self.title if self.title else ""
        self.ax.set_title(title, color='black', size=self.axlabels_fontsize)
        # Colorbar
        # Inside your class, assuming self.ax is an Axes object
        fig = self.ax.get_figure()
        cb = fig.colorbar(plot, ax=self.ax, format=f"%.{int(self.cb_digits_after_comma)}f",
                          label=self.zlabel)
        cb.set_label(label=self.zlabel, size=self.axlabels_fontsize, labelpad=20)
        cb.ax.tick_params(labelsize=self.cb_labelsize)
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ax_ylabel_txt,
                       ticks_direction='out', ticks_length=4, ticks_width=2,
                       ax_labels_fontsize=self.axlabels_fontsize,
                       ticks_labels_fontsize=self.ticks_labelsize)
        format_spines(ax=self.ax, color='black', lw=2)
        self.ax.tick_params(left=True, right=False, top=False, bottom=True)


def list_of_colormaps() -> list:
    """List of matplotlib colormaps."""
    return plt.colormaps()


if __name__ == '__main__':
    cmaps = list_of_colormaps()
    print(cmaps)
