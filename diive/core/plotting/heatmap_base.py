"""
HEATMAP
=======
"""
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

import diive.core.plotting.styles.LightTheme as theme
from diive.core.io.files import verify_dir
from diive.core.plotting.plotfuncs import default_format, format_spines, hide_xaxis_yaxis, hide_ticks_and_ticklabels, \
    make_patch_spines_invisible


class HeatmapBase:

    def __init__(self,
                 series: Series,
                 fig=None,
                 figsize: tuple = None,
                 ax=None,
                 ax_orientation: str = "vertical",
                 title: str = None,
                 vmin: float = None,
                 vmax: float = None,
                 cb_digits_after_comma: int = 2,
                 cb_labelsize: float = theme.AX_LABELS_FONTSIZE,
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE,
                 minyticks: int = 3,
                 maxyticks: int = 10,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'grey',
                 zlabel: str = None,
                 show_less_xticklabels: bool = False,
                 show_values: bool = False,
                 show_values_n_dec_places: int = 0,
                 verbose: bool = False):
        """Base class for heatmap plots using series with datetime index.

        Args:
            series: Series
            ax: Axis in which heatmap is shown. If *None*, a figure with axis will be generated.
            ax_orientation: Orientation of heatmap. Options: 'horizontal', 'vertical'
            title: Text shown at the top of the plot.
            vmin: Minimum value shown in plot
            vmax: Maximum value shown in plot
            cb_digits_after_comma: How many digits after the comma are shown in the colorbar legend.
            cmap: Matplotlib colormap
            color_bad: Color of missing values
            zlabel: Label for colorbar.
            show_values: Show z-values in heatmap.
            show_values_n_dec_places: Number of decimal places to show in heatmap.
                Only considered if *show_values* is True.

        """

        # Instance variables
        self.series = series.copy()
        self.series.name = self.series.name if self.series.name else "data"  # Time series must have a name
        self.fig = fig
        self.figsize = figsize
        self.ax = ax
        self.ax_orientation = ax_orientation

        self.verbose = verbose

        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cb_digits_after_comma = cb_digits_after_comma
        self.cb_labelsize = cb_labelsize
        self.color_bad = color_bad

        # Create fig and axis if no axis is given, otherwise use given axis
        if not ax:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=self.figsize)

        self.axlabels_fontsize = axlabels_fontsize
        self.ticks_labelsize = ticks_labelsize
        self.minyticks = minyticks
        self.maxyticks = maxyticks
        self.zlabel = zlabel

        self.show_less_xticklabels = show_less_xticklabels
        self.show_values = show_values
        self.showvalues_n_dec_places = show_values_n_dec_places

        self.plot_df = None
        self.x = None
        self.y = None
        self.z = None

    def get_ax(self):
        """Return axis in which plot was generated."""
        return self.ax

    def get_plot_data(self) -> pd.DataFrame:
        """Return pivot dataframe used to plot the heatmap."""
        return self.plot_df

    def show(self):
        """Generate plot and show figure."""
        self.plot()
        # plt.tight_layout()
        self.fig.show()

    def export_borderless_heatmap(self, outpath: str):
        # TODO----------
        """Save borderless plots (b/w heightmap and texture) for Blender render."""
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
        # Write "X" into z-value rectangle
        for i in range(self.z.shape[0]):
            for j in range(self.z.shape[1]):
                # Calculate the center coordinates
                x_center = (self.x[j] + self.x[j + 1]) / 2
                y_center = (self.y[i] + self.y[i + 1]) / 2
                self.ax.text(self.x[j] + 0.5, self.y[i] + 0.5, f"{self.z[i, j]:.{self.showvalues_n_dec_places}f}",
                             ha='center', va='center', color='black', fontsize=9, zorder=100)

    def _setup_plotdf(self) -> pd.DataFrame:
        """Create dataframe with values as z variable for colors"""
        plotdf = pd.DataFrame(index=self.series.index,
                              columns=['z'],
                              data=self.series.values)
        return plotdf

    @staticmethod
    def set_cmap(cmap, color_bad, z):
        """Set colormap and color of missing values"""
        # Colormap
        cmap = copy.copy(plt.get_cmap(cmap))  # Needed for now, https://github.com/matplotlib/matplotlib/issues/17634
        cmap.set_bad(color=color_bad, alpha=1.)  # Set missing data to specific color
        z = np.ma.masked_invalid(z)  # Mask NaN as missing
        return cmap, z

    def format(self, ax_xlabel_txt, ax_ylabel_txt, plot):
        title = self.title if self.title else f"{self.series.name} ({self.series.index.freqstr})"
        self.ax.set_title(title, color='black')
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
