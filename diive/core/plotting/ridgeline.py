"""
RIDGELINE: STACKED KERNEL DENSITIES PER YEAR, MONTH OR WEEK
===========================================================

Ridgeline plot of a time series: one kernel density curve per year, month or week.

References:
    https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    https://matplotlib.org/stable/users/explain/colors/colormaps.html

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
from sklearn.neighbors import KernelDensity

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.styles import LightTheme as theme
from diive.core.plotting.styles.LightTheme import adjust_color_lightness
from diive.core.plotting.styles.format import FormatStyle
from diive.core.utils.console import detail


class RidgeLinePlot:
    """Ridgeline plot: one kernel density curve per year, month or week, stacked vertically.

    The densities come from the kernel density estimator in scikit-learn, see here:
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        - https://scikit-learn.org/stable/modules/density.html#kernel-density

    See Also:
        examples/visualization/plot_ridgeline_basic.py — Ridge line plots with kernel density estimation
    """

    def __init__(self, series: pd.Series):
        """Set up a ridgeline plot for *series* (one stacked density per group/year). See :meth:`plot`.

        Missing values (NaN) and infinite values (+inf, -inf) are dropped up
        front: the kernel density estimator rejects both, so any gappy series
        would otherwise raise. Dropping here (rather than per group) also means a
        group left without a single finite value gets no ridge at all, instead of
        failing at fit time. Without any finite value, :meth:`plot` draws one
        empty axes with the note "no finite values".
        """
        self.series, self.n_infinite = pf.drop_infinite(series.dropna())

        self.xlim = None
        self.ylim = None

        self.ys = None
        self.ys_unique = None
        self.colors = None
        self.assigned_colors = {}
        self.hspace = None
        self.fig_width = None
        self.fig_height = None
        self.fig_dpi = None
        self.shade_percentile = None
        self.fig = None
        self.showplot = None
        self.ascending = False
        self.verbose = False
        self.kd_kwargs = None
        self.kde = None

    def get_fig(self):
        """Return matplotlib figure in which plot was generated."""
        return self.fig

    def _update_params(self, xlim: list, ylim: list, hspace: float,
                       fig_width: float, fig_height: float, shade_percentile: float,
                       show_mean_line: bool, fig_dpi: float, showplot: bool,
                       ascending: bool, style: FormatStyle, verbose: bool = False,
                       kd_kwargs: dict = None, fig=None):
        self.xlim = xlim
        self.ylim = ylim
        self.hspace = hspace
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.shade_percentile = shade_percentile
        self.show_mean_line = show_mean_line
        self.fig_dpi = fig_dpi
        self.showplot = showplot
        self.ascending = ascending
        self.style = style
        self.verbose = verbose
        self.kd_kwargs = kd_kwargs
        self._fig_external = fig
        return None

    def plot(self, xlim: list = None, ylim: list = None, hspace: float = -0.5,
             fig_width: float = 8, fig_height: float = 8,
             shade_percentile: float = 0.5, show_mean_line: bool = False,
             fig_dpi: float = 72, showplot: bool = True,
             ascending: bool = False, how: str = 'weekly', kd_kwargs: dict = None,
             fig=None, format_style: FormatStyle = None):
        """Render the ridgeline plot.

        Each group (year, month or week) gets one density curve. The curves are
        coloured from ``Spectral_r`` by the group mean, lowest mean first. The
        mean is printed at the right end of each curve.

        Args:
            xlim: X-axis limits ``[min, max]``, shared by all curves. None uses the
                data range.
            ylim: Y-axis (density) limits ``[min, max]``, shared by all curves. None
                uses 0 up to the highest density of all groups.
            hspace: Vertical space between the curves. Negative values make them
                overlap (default -0.5).
            fig_width: Width of a new figure in inches (default 8).
            fig_height: Height of a new figure in inches (default 8).
            shade_percentile: Quantile (0 to 1) of each group's values. The area
                above it is shaded slightly darker (default 0.5, the median).
            show_mean_line: Draw a vertical line at each group's mean (default False).
            fig_dpi: DPI of a new figure (default 72).
            showplot: Show the figure (default True).
            ascending: Order of the groups from top to bottom. False (default) puts
                the highest group label (e.g. the latest year) at the top.
            how: Grouping: ``'yearly'``, ``'monthly'`` or ``'weekly'`` (ISO week,
                default).
            kd_kwargs: Keyword arguments for scikit-learn's ``KernelDensity``, e.g.
                ``{'bandwidth': 0.5, 'kernel': 'gaussian'}``. None uses its defaults
                (Gaussian kernel, bandwidth 1.0).
            fig: Existing matplotlib ``Figure`` to draw into (cleared first). When
                given, the plot builds its stacked-density grid on it instead of
                creating a new figure -- e.g. for embedding in a GUI canvas. The
                ``fig_width``/``fig_height``/``fig_dpi`` args are then ignored.
                Pass ``showplot=False`` with this.
            format_style: A :class:`~diive.plotting.FormatStyle` describing the
                shared chrome. Only the parts that fit this stacked-density layout
                are honoured here: the bottom x-axis label (text/units, font, colour)
                and the figure title (text, font, colour). The per-ridge grid, spines,
                ticks and shading stay fixed.
        """
        style = format_style or FormatStyle()
        self._update_params(xlim=xlim, ylim=ylim, hspace=hspace,
                            fig_width=fig_width, fig_height=fig_height,
                            shade_percentile=shade_percentile, show_mean_line=show_mean_line,
                            fig_dpi=fig_dpi, showplot=showplot,
                            ascending=ascending, style=style, kd_kwargs=kd_kwargs, fig=fig)
        if self.series.empty:
            self._plot_empty()
            return
        self.ys, self.ys_unique = self._y_index(how=how)
        self.colors = iter(cm.Spectral_r(np.linspace(0, 1, len(self.ys_unique))))
        self.assigned_colors = self._assign_colors(how=how)
        self._plot()

    def _y_index(self, how: str):
        if how == 'yearly':
            y_index = self.series.index.year
        elif how == 'monthly':
            y_index = self.series.index.month
        elif how == 'weekly':
            y_index = self.series.index.isocalendar().week
        else:
            raise Exception(f"{how} not implemented.")
        y_index_unique = [x for x in np.unique(y_index)]
        if self.ascending:
            pass
        else:
            y_index_unique.reverse()
        return y_index, y_index_unique

    def _assign_colors(self, how: str):
        """Assign colors by the mean of each group (year, month or week), lowest mean first."""
        if how == 'yearly':
            aggs = self.series.groupby(self.series.index.year).agg('mean')
        elif how == 'monthly':
            aggs = self.series.groupby(self.series.index.month).agg('mean')
        elif how == 'weekly':
            aggs = self.series.groupby(self.series.index.isocalendar().week).agg('mean')
        else:
            raise Exception(f"{how} not implemented.")
        aggs = aggs.sort_values(ascending=True)
        # monthly_means_offset = monthly_means.add(abs(monthly_means.min()))
        # monthly_means_offset_norm = monthly_means_offset / monthly_means_offset.max()
        y_index_order = aggs.index
        y_index_colors = {}
        for m in y_index_order:
            # months_colors.append(next(self.colors))
            y_index_colors[m] = next(self.colors)
        return y_index_colors

    def _make_fig(self):
        # Draw into a caller-supplied figure (e.g. a GUI canvas) when given,
        # otherwise create a standalone one.
        if getattr(self, "_fig_external", None) is not None:
            self.fig = self._fig_external
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=(self.fig_width, self.fig_height),
                                  layout=None, dpi=self.fig_dpi)

    def _plot_empty(self):
        """Draw one empty axes with a note: no finite value is left to plot (L353)."""
        self._make_fig()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        pf.no_values_note(self.ax, f"{self.series.name}: no finite values")
        if self.showplot:
            self.fig.show()

    def _plot(self):
        self._make_fig()

        gs = (grid_spec.GridSpec(len(self.ys_unique), 1))
        # Set hspace up front (negative -> ridges overlap). Doing it here, before
        # the subplots are added, means each axis is positioned with the overlap
        # baked in. A later gs.update(hspace=...) only repositions axes of figures
        # registered with pyplot's Gcf, so it is a silent no-op for an embedded
        # GUI figure (created directly, not via plt.figure) -- the overlap would
        # never apply there.
        gs.update(wspace=0, hspace=self.hspace, left=0.09, right=0.97, top=0.95, bottom=0.07)

        # Create empty list for dynamic number of plots (rows)
        ax_objs = []

        # Detect min and max of x across all required aggs (e.g. across all years),
        # to have the same scaling for the x-axis in all plots
        if not self.xlim:
            x_check = []
            for y_current in self.ys_unique:
                locs_current = self.ys == y_current
                series = self.series[locs_current].copy()
                # Check x
                x1 = np.array(series)
                x_check.append(min(x1))
                x_check.append(max(x1))
            self.xlim = [min(x_check), max(x_check)]

        # Detect min and max of y across all required aggs (e.g. across all years),
        # to have the same scaling for the y-axis in all plots
        if not self.ylim:
            y_currents = []  # List of y IDs, e.g. months or years
            y_maxs = []  # List of y maxima
            for y_current in self.ys_unique:
                locs_current = self.ys == y_current
                series = self.series[locs_current].copy()
                x1 = np.array(series)
                # Kernel density
                if self.kd_kwargs:
                    self.kde = KernelDensity(**self.kd_kwargs)
                else:
                    self.kde = KernelDensity()
                # self.kde1 = KernelDensity(bandwidth=0.99, kernel='gaussian')
                self.kde.fit(x1[:, None])
                x_d = np.linspace(self.xlim[0], self.xlim[1], 1000)
                logprob1 = self.kde.score_samples(x_d[:, None])
                y = np.exp(logprob1)
                # y_check.append(min(y))
                y_maxs.append(max(y))
                y_currents.append(y_current)

            # Maximum value for y (probability from KDE)
            found_ymax = max(y_maxs)

            # Set y-axis to found maximum, should always start at zero
            self.ylim = [0, found_ymax]

            # Index of ymax in list of maxima
            index_of_max = y_maxs.index(max(y_maxs))

            # TODO Use index to get y ID (e.g. week) where max was found
            y_id = y_currents[index_of_max]


        # n_uniques = len(self.ys_unique)
        i = 0
        for y_current in self.ys_unique:
            if self.verbose:
                detail(f"Current y: {y_current}  Position in gridspec: {i + 1}", verbose=self.verbose)
            # Get data for current year or month or etc.
            locs_current = self.ys == y_current
            series = self.series[locs_current].copy()
            x1 = np.array(series)
            x1_mean = x1.mean()
            x1_percentile = series.quantile(self.shade_percentile)

            # Kernel density
            if self.kd_kwargs:
                self.kde = KernelDensity(**self.kd_kwargs)
            else:
                self.kde = KernelDensity()
            self.kde.fit(x1[:, None])
            x_d = np.linspace(self.xlim[0], self.xlim[1], 1000)
            logprob1 = self.kde.score_samples(x_d[:, None])

            # Create new axes object
            # ax_objs.insert(0, self.fig.add_subplot(gs[i:i + 1, 0:]))
            ax_objs.append(self.fig.add_subplot(gs[i:i + 1, 0:]))
            # self.ax = ax_objs[0]
            self.ax = ax_objs[-1]

            # Plot distribution
            y = np.exp(logprob1)
            darker_color = adjust_color_lightness(self.assigned_colors[y_current], amount=0.4)
            self.ax.fill_between(x_d, y, alpha=1, color=self.assigned_colors[y_current])
            self.ax.plot(x_d, y, color=darker_color, lw=1, alpha=1)  # Outline darker than fill

            # Set uniform x and y limits
            self.ax.set_xlim(self.xlim[0], self.xlim[1])
            self.ax.set_ylim(self.ylim[0], self.ylim[1])

            # Make axis background transparent
            rect = self.ax.patch
            rect.set_alpha(0)

            # Disable gridlines explicitly
            self.ax.grid(False)

            # Remove axis borders, axis ticks, and labels
            self.ax.set_yticklabels([])
            self.ax.set_yticks([])

            # Show x labels only for last axis
            if i == len(self.ys_unique) - 1:
                # Bottom x-axis label from the shared chrome: text (style.xlabel or
                # series name) plus units, with the style's resolved font/colour.
                xlabel = self.style.xlabel if self.style.xlabel is not None else self.series.name
                if xlabel and self.style.xunits:
                    xlabel = f"{xlabel} {self.style.xunits}"
                axlabel_fs = (self.style.axlabel_fontsize
                              if self.style.axlabel_fontsize is not None else theme.FONTSIZE_AXLABEL)
                text_color = (self.style.text_color
                              if self.style.text_color is not None else theme.COLOR_TEXT)
                self.ax.set_xlabel(xlabel or "", fontsize=axlabel_fs, color=text_color,
                                   fontweight=self.style.axlabel_fontweight)
            else:
                self.ax.set_xticklabels([])
                self.ax.set_xticks([])

            # Hide axis spines
            spines = ["top", "right", "left"]
            for s in spines:
                self.ax.spines[s].set_visible(False)

            # Hide bottom spine except for the last (bottom-most) axis
            if i == len(self.ys_unique) - 1:
                self.ax.spines["bottom"].set_visible(True)
            else:
                self.ax.spines["bottom"].set_visible(False)

            # Show year or month etc. on y axis
            self.ax.text(-0.02, 0.01, y_current, fontweight="normal", fontsize=14, ha="right",
                         transform=self.ax.get_yaxis_transform())

            # Show mean value
            self.ax.text(1, 0.01, f"{x1_mean:.1f}", fontsize=12, ha="right",
                         transform=self.ax.get_yaxis_transform(), color=darker_color)

            # Show mean line
            if self.show_mean_line:
                self.ax.axvline(x1_mean, ls="-", lw=1, color="black")

            # Shade the area above the shade_percentile quantile
            locs_percentile = np.where(x_d >= x1_percentile)
            x_d_percentile = x_d[locs_percentile]
            y_percentile = y[locs_percentile]
            slightly_darker_color = adjust_color_lightness(self.assigned_colors[y_current], amount=0.9)
            self.ax.fill_between(x_d_percentile, y_percentile, alpha=1, color=slightly_darker_color)

            i += 1

        # ax_objs.reverse()

        # Figure title from the shared chrome: text (style.title or series name)
        # with the style's resolved font/colour. style.title == "" suppresses it.
        title = self.style.title if self.style.title is not None else self.series.name
        if title:
            title_fs = (self.style.title_fontsize
                        if self.style.title_fontsize is not None else theme.FONTSIZE_TITLE)
            text_color = (self.style.text_color
                          if self.style.text_color is not None else theme.COLOR_TEXT)
            self.fig.suptitle(title, fontsize=title_fs, color=text_color,
                              fontweight=self.style.title_fontweight)
        # plt.tight_layout()
        if self.showplot:
            self.fig.show()
