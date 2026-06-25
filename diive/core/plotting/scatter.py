"""
PLOTTING: SCATTER XY
====================

Scatter plot of x vs y with optional z colour-coding and bin aggregation.

Part of the diive library: https://github.com/holukas/diive
"""
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.dfun.stats import q25, q75
from diive.core.plotting.styles.format import FormatStyle


class ScatterXY:
    """Scatter plot of x vs y with optional z colour-coding and bin aggregation. See :meth:`__init__`."""

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

        # Internal, guaranteed-unique column keys for data access. The display
        # names (xname/yname/zname) are kept only for labels: x, y and z may
        # share a name (e.g. colouring a variable by itself), which would make
        # pd.concat produce duplicate column labels and turn xy_df[name] into a
        # DataFrame instead of a Series.
        self._xc, self._yc, self._zc = "_x", "_y", "_z"

        # Prepare data
        df_list = [x.rename(self._xc), y.rename(self._yc)]
        if z is not None:
            df_list.append(z.rename(self._zc))
        self.xy_df = pd.concat(df_list, axis=1)
        self.xy_df = self.xy_df.dropna()

        self.binagg = None if self.nbins == 0 else self.binagg

        if self.nbins > 0:
            self._databinning()

    def _databinning(self):
        group, bins = pd.qcut(self.xy_df[self._xc], q=self.nbins, retbins=True, duplicates='drop')
        groupcol = f'GROUP_{self._xc}'
        self.xy_df[groupcol] = group
        self.xy_df_binned = self.xy_df.groupby(groupcol).agg({'mean', 'median', 'std', 'count', q25, q75})

    def plot(
            self,
            ax: plt.Axes = None,
            format_style: FormatStyle = None,
            xlim: list = None,
            ylim: list or Literal['auto'] = None,
            cmap: str = 'viridis',
            show_colorbar: bool = True,
            markersize: float = 40,
            alpha: float = 1.0,
            vmin: float = None,
            vmax: float = None,
    ):
        """Generate plot with optional styling and formatting.

        Renders scatter plot with all styling parameters. Can be called multiple
        times with different parameters/axes to explore different views of the same data.

        Chrome (title, labels, units, font sizes, colours, grid, legend, zero line)
        comes from a shared :class:`~diive.plotting.FormatStyle` so it looks and is
        configured the same way as every other diive plot. The data-rendering
        arguments (cmap/markersize/alpha/vmin/vmax/colorbar, axis limits) stay here.

        Args:
            ax: Matplotlib axes to plot on (default: creates new figure if None)
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome
                (title/xlabel/ylabel/zlabel/xunits/yunits). When None the diive house
                style is used. The colorbar label reads from ``format_style.zlabel``.
            xlim: X-axis limits as [min, max] (default: data min/max)
            ylim: Y-axis limits as [min, max] or 'auto' (default: data range)
            cmap: Colormap name for z variable (default: 'viridis')
                  Examples: 'plasma', 'viridis', 'coolwarm', 'RdYlBu'
            show_colorbar: Display colorbar if z provided (default: True)
            markersize: Scatter point area in points^2 (default: 40)
            alpha: Scatter point opacity, 0-1 (default: 1.0)
            vmin: Lower bound of the z colour scale (default: data minimum)
            vmax: Upper bound of the z colour scale (default: data maximum)

        Notes:
            - xlim always uses full data range (no quantile trimming)
            - ylim='auto' with nbins: uses binned data limits
            - Colorbar shown only if z is provided and show_colorbar=True
        """
        style = format_style or FormatStyle()

        # Colorbar label comes only from the style; fall back to the z series name.
        zlabel = style.zlabel if style.zlabel is not None else self.zname

        if not ax:
            self.fig, self.ax = pf.create_ax(figsize=(8, 8))
            self._plot(self.ax, style, zlabel, xlim, ylim, cmap, show_colorbar,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)
            # Skip tight_layout if colorbar is present (incompatible with new layout engine)
            if self.zname is None or not show_colorbar:
                plt.tight_layout()
            self.fig.show()
        else:
            self.ax = ax
            self._plot(self.ax, style, zlabel, xlim, ylim, cmap, show_colorbar,
                       markersize=markersize, alpha=alpha, vmin=vmin, vmax=vmax)

    def _plot(self, ax: plt.Axes, style: FormatStyle, zlabel: str = None,
              xlim: list = None, ylim: list or str = None, cmap: str = 'viridis',
              show_colorbar: bool = True, nbins: int = 10,
              markersize: float = 40, alpha: float = 1.0, vmin: float = None, vmax: float = None):
        """Generate plot on axis"""
        nbins += 1  # To include zero

        # Scatter plot with optional color
        if self.zname is not None:
            scatter = ax.scatter(x=self.xy_df[self._xc],
                                 y=self.xy_df[self._yc],
                                 c=self.xy_df[self._zc],
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
            ax.scatter(x=self.xy_df[self._xc],
                       y=self.xy_df[self._yc],
                       c='none',
                       s=markersize,
                       alpha=alpha,
                       marker='o',
                       edgecolors='#607D8B',
                       label=self.yname)

        if self.nbins > 0:

            _min = self.xy_df_binned[self._yc]['count'].min()
            _max = self.xy_df_binned[self._yc]['count'].max()
            ax.plot(self.xy_df_binned[self._xc][self.binagg],
                    self.xy_df_binned[self._yc][self.binagg],
                    c='r', ms=10, marker='o', lw=2,
                    # c='none', ms=80, marker='o', edgecolors='r', lw=2,
                    label=f"binned data ({self.binagg}, {_min}-{_max} values per bin)")

            if self.binagg == 'median':
                ax.fill_between(self.xy_df_binned[self._xc][self.binagg],
                                self.xy_df_binned[self._yc]['q25'],
                                self.xy_df_binned[self._yc]['q75'],
                                alpha=.2, zorder=10, color='red',
                                label="interquartile range")

            if self.binagg == 'mean':
                ax.errorbar(x=self.xy_df_binned[self._xc][self.binagg],
                            y=self.xy_df_binned[self._yc][self.binagg],
                            xerr=self.xy_df_binned[self._xc]['std'],
                            yerr=self.xy_df_binned[self._yc]['std'],
                            elinewidth=3, ecolor='red', alpha=.6, lw=0,
                            label="standard deviation")

        self._apply_format(ax, style, xlim, ylim)
        ax.locator_params(axis='x', nbins=nbins)
        ax.locator_params(axis='y', nbins=nbins)

    def _apply_format(self, ax: plt.Axes, style: FormatStyle,
                      xlim: list = None, ylim: list or str = None):

        if xlim:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = self.xy_df[self._xc].min()
            xmax = self.xy_df[self._xc].max()
        ax.set_xlim(xmin, xmax)

        if ylim == 'auto':
            if self.binagg == 'median':
                ymin = self.xy_df_binned[self._yc]['q25'].min()
                ymax = self.xy_df_binned[self._yc]['q75'].max()
            elif self.binagg == 'mean':
                _lowery = self.xy_df_binned[self._yc]['mean'].sub(self.xy_df_binned[self._yc]['std'])
                _uppery = self.xy_df_binned[self._yc]['mean'].add(self.xy_df_binned[self._yc]['std'])
                ymin = _lowery.min()
                ymax = _uppery.max()
            else:
                ymin = self.xy_df[self._yc].quantile(0.01)
                ymax = self.xy_df[self._yc].quantile(0.99)
        elif isinstance(ylim, list):
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = self.xy_df[self._yc].min()
            ymax = self.xy_df[self._yc].max()

        ax.set_ylim(ymin, ymax)

        # Shared formatting layer: title/labels/units/fonts/grid/legend/zeroline.
        style.apply(ax=ax, default_title=f"{self.yname} vs. {self.xname}",
                    default_xlabel=self.xname, default_ylabel=self.yname,
                    zeroline_data=self.xy_df[self._yc])


def scatter_to_code(xcol: str, ycol: str, zcol: str = None, *,
                    nbins: int = 0, binagg: str = 'median',
                    cmap: str = 'viridis', show_colorbar: bool = True,
                    markersize: float = 40, alpha: float = 1.0,
                    vmin: float = None, vmax: float = None,
                    format_kwargs: dict = None, df_name: str = 'df') -> str:
    """Return a runnable snippet that reproduces a :class:`ScatterXY` plot.

    Mirrors what the GUI's Scatter XY tab renders: the X/Y (and optional Z)
    columns of ``df_name``, the binning, and the data-render arguments. Only the
    non-default ``FormatStyle`` fields (from ``format_kwargs``) are emitted, so a
    plot left at the house style produces a clean call.
    """
    init = [f"    x={df_name}[{xcol!r}],", f"    y={df_name}[{ycol!r}],"]
    if zcol:
        init.append(f"    z={df_name}[{zcol!r}],")
    init += [f"    nbins={nbins!r},", f"    binagg={binagg!r},"]

    plot = ["    ax=ax,"]
    fmt = {k: v for k, v in (format_kwargs or {}).items() if v is not None}
    if fmt:
        args = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
        plot.append(f"    format_style=dv.plotting.FormatStyle({args}),")
    plot += [
        f"    cmap={cmap!r},",
        f"    show_colorbar={show_colorbar!r},",
        f"    markersize={markersize!r},",
        f"    alpha={alpha!r},",
        f"    vmin={vmin!r},",
        f"    vmax={vmax!r},",
    ]
    return (
        "import matplotlib.pyplot as plt\n"
        "import diive as dv\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "dv.plotting.ScatterXY(\n"
        + "\n".join(init) + "\n"
        ").plot(\n"
        + "\n".join(plot) + "\n"
        ")\n"
        "plt.show()\n"
    )
