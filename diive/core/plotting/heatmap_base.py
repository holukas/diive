"""
HEATMAP
=======
"""
import copy
from pathlib import Path

import matplotlib.gridspec as gridspec
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
                 ax=None,
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
                 figsize: tuple = (6, 10.7),
                 zlabel: str = "Value",
                 verbose: bool = False):
        """Base class for heatmap plots using series with datetime index.

        Args:
            series: Series
            ax: Axis in which heatmap is shown. If *None*, a figure with axis will be generated.
            title: Text shown at the top of the plot.
            vmin: Minimum value shown in plot
            vmax: Maximum value shown in plot
            cb_digits_after_comma: How many digits after the comma are shown in the colorbar legend.
            cmap: Matplotlib colormap
            color_bad: Color of missing values
            zlabel: Label for colorbar.

        """
        self.series = series.copy()
        self.verbose = verbose
        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cb_digits_after_comma = cb_digits_after_comma
        self.cb_labelsize = cb_labelsize
        self.color_bad = color_bad
        self.figsize = figsize
        self.ax = ax
        self.axlabels_fontsize = axlabels_fontsize
        self.fig = fig
        self.ticks_labelsize = ticks_labelsize
        self.minyticks = minyticks
        self.maxyticks = maxyticks
        self.zlabel = zlabel

        if self.verbose:
            print("Plotting heatmap  ...")

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
        plt.tight_layout()
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

    def setup(self):

        fig = self.fig

        if self.verbose:
            print("Preparing timestamp for heatmap plotting ...")

        # Create axis if none is given
        if not self.ax:
            fig, ax = self._create_ax()
        else:
            ax = self.ax

        # Create dataframe with values as z variable for colors
        plot_df = pd.DataFrame(index=self.series.index,
                               columns=['z'],
                               data=self.series.values)
        return plot_df, fig, ax

    def _create_ax(self):
        """Create figure and axis"""
        # Figure setup
        fig = plt.figure(facecolor='white', figsize=self.figsize)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        return fig, ax

    def transform_data(self, xaxis_vals, yaxis_vals):
        """Transform data for plotting"""

        plot_df = self.plot_df.copy()

        # Setup xy axes
        # xaxis_vals, yaxis_vals = self._set_xy_axes_type()
        plot_df['y_vals'] = yaxis_vals
        plot_df['x_vals'] = xaxis_vals
        plot_df.reset_index(drop=True, inplace=True)

        # Put needed data in new df_pivot, then pivot to bring it in needed shape
        plot_df_pivot = plot_df.pivot(index='y_vals', columns='x_vals', values='z')  ## new in pandas 23.4
        # plot_df_pivot = plot_df_pivot.append(plot_df_pivot.iloc[-1])
        x = plot_df_pivot.columns.values
        y = plot_df_pivot.index.values
        z = plot_df_pivot.values

        return x, y, z, plot_df_pivot

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
        cb = plt.colorbar(plot, ax=self.ax, format=f"%.{int(self.cb_digits_after_comma)}f",
                          label=self.zlabel)
        cb.set_label(label=self.zlabel, size=self.axlabels_fontsize)
        cb.ax.tick_params(labelsize=self.cb_labelsize)
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ax_ylabel_txt,
                       ticks_direction='out', ticks_length=4, ticks_width=2,
                       ax_labels_fontsize=self.axlabels_fontsize,
                       ticks_labels_fontsize=self.ticks_labelsize)
        format_spines(ax=self.ax, color='black', lw=2)
