from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.dfun.stats import q25, q75


class ScatterXY:
    def __init__(
            self,
            x: Series,
            y: Series,
            z: Series = None,
            nbins: int = 0,
            binagg: Literal['mean', 'median'] = 'median',
    ):
        """Scatter plot with optional third variable as color and bin aggregation.

        Visualize relationships between two variables (x, y) with optional color-coding
        by a third variable (z). Supports optional binning and aggregation for trend
        visualization with confidence intervals (median ± IQR or mean ± std).

        Args:
            x: Series for x-axis
            y: Series for y-axis
            z: Optional Series for color-coding scatter points
            nbins: Number of bins for x-axis aggregation (0 = no aggregation)
            binagg: Aggregation method for bins ('mean' or 'median', default: 'median')

        Features:
            - 2-variable scatter: Basic x vs y plot
            - 3-variable scatter: Color-code points by z variable with colorbar
            - Bin aggregation: Group data by x-axis bins and overlay trends
            - Confidence intervals: Show IQR (median) or std (mean) per bin

        Call `plot()` to render with styling options (labels, limits, title, colormap).

        See Also:
            examples/core/visualization/plot_scatter_xy.py — Scatter plot variations with 2D and 3D coloring
        """
        self.xname = x.name
        self.yname = y.name
        self.zname = z.name if z is not None else None
        self.nbins = nbins
        self.binagg = binagg
        self.fig = None
        self.ax = None

        # Prepare data
        df_list = [x, y]
        if z is not None:
            df_list.append(z)
        self.xy_df = pd.concat(df_list, axis=1)
        self.xy_df = self.xy_df.dropna()

        self.binagg = None if self.nbins == 0 else self.binagg

        if self.nbins > 0:
            self._databinning()

    def _databinning(self):
        group, bins = pd.qcut(self.xy_df[self.xname], q=self.nbins, retbins=True, duplicates='drop')
        groupcol = f'GROUP_{self.xname}'
        self.xy_df[groupcol] = group
        self.xy_df_binned = self.xy_df.groupby(groupcol).agg({'mean', 'median', 'std', 'count', q25, q75})

    def plot(
            self,
            ax: plt.Axes = None,
            xlabel: str = None,
            ylabel: str = None,
            zlabel: str = None,
            xunits: str = None,
            yunits: str = None,
            xlim: list = None,
            ylim: list or Literal['auto'] = None,
            cmap: str = 'viridis',
            show_colorbar: bool = True,
            title: str = None,
            markersize: float = 40,
            alpha: float = 1.0,
            vmin: float = None,
            vmax: float = None,
    ):
        """Generate plot with optional styling and formatting.

        Renders scatter plot with all styling parameters. Can be called multiple
        times with different parameters/axes to explore different views of the same data.

        Args:
            ax: Matplotlib axes to plot on (default: creates new figure if None)
            xlabel: X-axis label (default: series name)
            ylabel: Y-axis label (default: series name)
            zlabel: Colorbar label (default: z series name, ignored if no z)
            xunits: X-axis units (appended to xlabel)
            yunits: Y-axis units (appended to ylabel)
            xlim: X-axis limits as [min, max] (default: data min/max)
            ylim: Y-axis limits as [min, max] or 'auto' (default: data range)
            cmap: Colormap name for z variable (default: 'viridis')
                  Examples: 'plasma', 'viridis', 'coolwarm', 'RdYlBu'
            show_colorbar: Display colorbar if z provided (default: True)
            title: Plot title (default: "{yname} vs. {xname}")
            markersize: Scatter point area in points^2 (default: 40)
            alpha: Scatter point opacity, 0-1 (default: 1.0)
            vmin: Lower bound of the z colour scale (default: data minimum)
            vmax: Upper bound of the z colour scale (default: data maximum)

        Notes:
            - xlim always uses full data range (no quantile trimming)
            - ylim='auto' with nbins: uses binned data limits
            - Colorbar shown only if z is provided and show_colorbar=True
        """
        xlabel = xlabel if xlabel else self.xname
        ylabel = ylabel if ylabel else self.yname
        zlabel = zlabel if zlabel else self.zname
        xunits = xunits
        yunits = yunits
        xlim = xlim
        ylim = ylim
        cmap = cmap
        show_colorbar = show_colorbar
        title = title if title else f"{self.yname} vs. {self.xname}"

        if not ax:
            self.fig, self.ax = pf.create_ax(figsize=(8, 8))
            self._plot(self.ax, xlabel, ylabel, zlabel, xunits, yunits, xlim, ylim, cmap, show_colorbar, title,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)
            # Skip tight_layout if colorbar is present (incompatible with new layout engine)
            if self.zname is None or not show_colorbar:
                plt.tight_layout()
            self.fig.show()
        else:
            self.ax = ax
            self._plot(self.ax, xlabel, ylabel, zlabel, xunits, yunits, xlim, ylim, cmap, show_colorbar, title,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)

    def _plot(self, ax: plt.Axes, xlabel: str = None, ylabel: str = None, zlabel: str = None,
              xunits: str = None, yunits: str = None, xlim: list = None,
              ylim: list or str = None, cmap: str = 'viridis',
              show_colorbar: bool = True, title: str = None, nbins: int = 10,
              markersize: float = 40, alpha: float = 1.0, vmin: float = None, vmax: float = None):
        """Generate plot on axis"""
        nbins += 1  # To include zero

        # Scatter plot with optional color
        if self.zname is not None:
            scatter = ax.scatter(x=self.xy_df[self.xname],
                                 y=self.xy_df[self.yname],
                                 c=self.xy_df[self.zname],
                                 s=markersize,
                                 alpha=alpha,
                                 marker='o',
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 label=self.yname)
            if show_colorbar:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(zlabel if zlabel else self.zname, fontsize=12)
        else:
            ax.scatter(x=self.xy_df[self.xname],
                       y=self.xy_df[self.yname],
                       c='none',
                       s=markersize,
                       alpha=alpha,
                       marker='o',
                       edgecolors='#607D8B',
                       label=self.yname)

        if self.nbins > 0:

            _min = self.xy_df_binned[self.yname]['count'].min()
            _max = self.xy_df_binned[self.yname]['count'].max()
            ax.plot(self.xy_df_binned[self.xname][self.binagg],
                    self.xy_df_binned[self.yname][self.binagg],
                    c='r', ms=10, marker='o', lw=2,
                    # c='none', ms=80, marker='o', edgecolors='r', lw=2,
                    label=f"binned data ({self.binagg}, {_min}-{_max} values per bin)")

            if self.binagg == 'median':
                ax.fill_between(self.xy_df_binned[self.xname][self.binagg],
                                self.xy_df_binned[self.yname]['q25'],
                                self.xy_df_binned[self.yname]['q75'],
                                alpha=.2, zorder=10, color='red',
                                label="interquartile range")

            if self.binagg == 'mean':
                ax.errorbar(x=self.xy_df_binned[self.xname][self.binagg],
                            y=self.xy_df_binned[self.yname][self.binagg],
                            xerr=self.xy_df_binned[self.xname]['std'],
                            yerr=self.xy_df_binned[self.yname]['std'],
                            elinewidth=3, ecolor='red', alpha=.6, lw=0,
                            label="standard deviation")

        self._apply_format(ax, xlabel, ylabel, xunits, yunits, xlim, ylim, title)
        ax.locator_params(axis='x', nbins=nbins)
        ax.locator_params(axis='y', nbins=nbins)

    def _apply_format(self, ax: plt.Axes, xlabel: str = None, ylabel: str = None,
                      xunits: str = None, yunits: str = None,
                      xlim: list = None, ylim: list or str = None, title: str = None):

        if xlim:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = self.xy_df[self.xname].min()
            xmax = self.xy_df[self.xname].max()
        ax.set_xlim(xmin, xmax)

        if ylim == 'auto':
            if self.binagg == 'median':
                ymin = self.xy_df_binned[self.yname]['q25'].min()
                ymax = self.xy_df_binned[self.yname]['q75'].max()
            elif self.binagg == 'mean':
                _lowery = self.xy_df_binned[self.yname]['mean'].sub(self.xy_df_binned[self.yname]['std'])
                _uppery = self.xy_df_binned[self.yname]['mean'].add(self.xy_df_binned[self.yname]['std'])
                ymin = _lowery.min()
                ymax = _uppery.max()
            else:
                ymin = self.xy_df[self.yname].quantile(0.01)
                ymax = self.xy_df[self.yname].quantile(0.99)
        elif isinstance(ylim, list):
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = self.xy_df[self.yname].min()
            ymax = self.xy_df[self.yname].max()

        ax.set_ylim(ymin, ymax)

        pf.add_zeroline_y(ax=ax, data=self.xy_df[self.yname])

        pf.default_format(ax=ax,
                          ax_xlabel_txt=xlabel,
                          ax_ylabel_txt=ylabel,
                          txt_ylabel_units=yunits)

        pf.default_legend(ax=ax,
                          labelspacing=0.2,
                          ncol=1)

        ax.set_title(title, size=20)

        # pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        # if self.showplot:
        #     self.fig.suptitle(f"{self.title}", fontsize=theme.FIGHEADER_FONTSIZE)
        #     self.fig.tight_layout()
