"""
https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
https://matplotlib.org/stable/users/explain/colors/colormaps.html
"""

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
from sklearn.neighbors import KernelDensity

from diive.core.plotting.styles.LightTheme import adjust_color_lightness


class RidgeLinePlot:
    """
    RidgeLinePlot uses the kernel density estimator from scikit-learn, see here:
        - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        - https://scikit-learn.org/stable/modules/density.html#kernel-density
    """

    def __init__(self, series: pd.Series):
        self.series = series

        self.xlim = None
        self.ylim = None

        self.ys = None
        self.ys_unique = None
        self.colors = None
        self.assigned_colors = {}
        self.hspace = None
        self.xlabel = None
        self.fig_width = None
        self.fig_height = None
        self.fig_dpi = None
        self.shade_percentile = None
        self.fig_title = None
        self.fig = None
        self.showplot = None
        self.ascending = False
        self.verbose = False
        self.kd_kwargs = None
        self.kde = None

    def get_fig(self):
        """Return matplotlib figure in which plot was generated."""
        return self.fig

    def _update_params(self, xlim: list, ylim: list, hspace: float, xlabel: str,
                       fig_width: float, fig_height: float, shade_percentile: float,
                       show_mean_line: bool, fig_title: str, fig_dpi: float, showplot: bool,
                       ascending: bool, verbose: bool = False, kd_kwargs: dict = None):
        self.xlim = xlim
        self.ylim = ylim
        self.hspace = hspace
        self.xlabel = xlabel
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.shade_percentile = shade_percentile
        self.show_mean_line = show_mean_line
        self.fig_title = fig_title
        self.fig_dpi = fig_dpi
        self.showplot = showplot
        self.ascending = ascending
        self.verbose = verbose
        self.kd_kwargs = kd_kwargs
        return None

    def plot(self, xlim: list = None, ylim: list = None, hspace: float = -0.5, xlabel: str = None,
             fig_width: float = 8, fig_height: float = 8,
             shade_percentile: float = 0.5, show_mean_line: bool = False,
             fig_title: str = None, fig_dpi: float = 72, showplot: bool = True,
             ascending: bool = False, how: str = 'weekly', kd_kwargs: dict = None):
        self._update_params(xlim=xlim, ylim=ylim, hspace=hspace, xlabel=xlabel,
                            fig_width=fig_width, fig_height=fig_height,
                            shade_percentile=shade_percentile, show_mean_line=show_mean_line,
                            fig_title=fig_title, fig_dpi=fig_dpi, showplot=showplot,
                            ascending=ascending, kd_kwargs=kd_kwargs)
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
        """Assign colors depending on the monthly mean value."""
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

    def _plot(self):

        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height),
                              layout=None, dpi=self.fig_dpi)

        gs = (grid_spec.GridSpec(len(self.ys_unique), 1))
        gs.update(wspace=0, hspace=0, left=0.09, right=0.97, top=0.95, bottom=0.07)

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
                print(f"Current y: {y_current}  Position in gridspec: {i + 1}")
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

            # Remove axis borders, axis ticks, and labels
            self.ax.set_yticklabels([])
            self.ax.set_yticks([])

            # Show x labels only for last axis
            if i == len(self.ys_unique) - 1:
                self.xlabel = self.series.name if not self.xlabel else self.xlabel
                self.ax.set_xlabel(self.xlabel, fontsize=16)
                # self.ax.set_xlabel(self.xlabel, fontsize=16, fontweight="bold")
            else:
                self.ax.set_xticklabels([])
                self.ax.set_xticks([])

            # Hide axis spines
            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                self.ax.spines[s].set_visible(False)

            # Show year or month etc. on y axis
            self.ax.text(-0.02, 0.01, y_current, fontweight="normal", fontsize=14, ha="right",
                         transform=self.ax.get_yaxis_transform())

            # Show mean value
            self.ax.text(1, 0.01, f"{x1_mean:.1f}", fontsize=12, ha="right",
                         transform=self.ax.get_yaxis_transform(), color=darker_color)

            # Show mean line
            if self.show_mean_line:
                self.ax.axvline(x1_mean, ls="-", lw=1, color="black")

            # Show 99th percentile
            locs_percentile = np.where(x_d >= x1_percentile)
            x_d_percentile = x_d[locs_percentile]
            y_percentile = y[locs_percentile]
            slightly_darker_color = adjust_color_lightness(self.assigned_colors[y_current], amount=0.9)
            self.ax.fill_between(x_d_percentile, y_percentile, alpha=1, color=slightly_darker_color)

            i += 1

        # ax_objs.reverse()

        gs.update(hspace=self.hspace)
        if not self.fig_title:
            title = self.series.name
        else:
            title = self.fig_title
        self.fig.suptitle(title, fontsize=20)
        # plt.tight_layout()
        if self.showplot:
            self.fig.show()


def _example():
    from diive.configs.exampledata import load_exampledata_parquet
    import diive as dv
    df = load_exampledata_parquet()
    # yr = 2015
    locs = (df.index.year >= 2019) & (df.index.year <= 2019)
    df = df[locs].copy()
    series = df['Tair_f'].copy()

    rp = dv.ridgeline(series=series)
    rp.plot(
        how='weekly',
        kd_kwargs=None,  # params from scikit KernelDensity as dict
        xlim=None,  # min/max as list
        ylim=None,  # min/max as list
        hspace=-0.5,  # overlap between months
        xlabel=r'Air temperature (°C)',
        fig_width=8,
        fig_height=10,
        shade_percentile=0.5,
        show_mean_line=False,
        fig_title="Air temperatures per month (2019)",
        fig_dpi=72,
        showplot=True,
        ascending=False
    )
    # rp.per_month()
    # rp.per_year()
    # rp.per_week()

    print(rp.get_fig())
    print(rp.xlim)
    print(rp.ylim)

    rp.kde


if __name__ == '__main__':
    _example()
