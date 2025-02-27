"""
https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
"""

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm
from sklearn.neighbors import KernelDensity


class RidgePlotTS:

    def __init__(self, series: pd.Series):
        self.series = series

        self.xlim = None
        self.ylim = None

        # Per month plotting
        self.months = None
        self.months_unique = None
        self.colors = None
        self.months_colors = {}
        self.hspace = None

    def per_month(self, xlim: list, ylim: list, hspace: float):
        self.xlim = xlim
        self.ylim = ylim
        self.hspace = hspace
        self.months = self.series.index.month
        self.months_unique = [x for x in np.unique(self.months)]
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
        self.colors = iter(cm.Spectral_r(np.linspace(0, 1, len(self.months_unique))))
        self.months_colors = self._xxx()
        self._plot()

    def _xxx(self):
        monthly_means = self.series.groupby(self.series.index.month).agg('mean')
        # monthly_means = self.series.resample('MS').mean()
        monthly_means = monthly_means.sort_values(ascending=True)
        months_order = monthly_means.index
        months_colors = {}
        for m in months_order:
            # months_colors.append(next(self.colors))
            months_colors[m] = next(self.colors)
        return months_colors

    def adjust_lightness(self, color, amount=0.5):
        # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def _plot(self):
        gs = (grid_spec.GridSpec(len(self.months_unique), 1))
        fig = plt.figure(figsize=(8, 8))

        i = 0

        # Creating empty list
        ax_objs = []

        for month in self.months_unique:
            # month = self.months_unique[i]
            locs_month = self.months == month

            x1 = np.array(self.series[locs_month])
            x1_mean = x1.mean()
            x_d = np.linspace(self.xlim[0], self.xlim[1], 1000)

            kde1 = KernelDensity(bandwidth=0.99, kernel='gaussian')
            kde1.fit(x1[:, None])
            logprob1 = kde1.score_samples(x_d[:, None])

            # creating new axes object
            ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))
            ax = ax_objs[-1]

            # plotting the distribution
            y = np.exp(logprob1)
            color_line = self.adjust_lightness(self.months_colors[month], amount=0.4)
            ax.plot(x_d, y, color=color_line, lw=1, alpha=1)

            # c = next(self.colors)
            ax.fill_between(x_d, y, alpha=1, color=self.months_colors[month])

            # setting uniform x and y lims
            ax.set_xlim(self.xlim[0], self.xlim[1])
            ax.set_ylim(self.ylim[0], self.ylim[1])

            # make background transparent
            rect = ax.patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
            ax.set_yticklabels([])
            ax.set_yticks([])

            if i == len(self.months_unique) - 1:
                ax.set_xlabel("X", fontsize=16, fontweight="bold")
            else:
                ax.set_xticklabels([])
                ax.set_xticks([])

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax.spines[s].set_visible(False)

            adj_country = str(month).replace(" ", "\n")
            ax.text(-0.02, 0.01, adj_country, fontweight="bold", fontsize=14, ha="right",
                    transform=ax.get_yaxis_transform())

            # Mean value
            ax.text(1, 0.01, f"{x1_mean:.1f}", fontsize=12, ha="right",
                    transform=ax.get_yaxis_transform(), color=color_line)

            # ax.axvline(0, ls="-", lw=1, color="black")
            ax.axvline(x1_mean, ls="-", lw=1, color="black")

            i += 1

        gs.update(hspace=self.hspace)

        fig.suptitle(self.series.name, fontsize=20)

        plt.tight_layout()
        plt.show()


def _example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    yr = 2015
    locs = (df.index.year >= yr) & (df.index.year <= yr)
    df = df[locs].copy()
    series = df['Tair_f'].copy()

    rp = RidgePlotTS(series=series)
    rp.per_month(xlim=[-20, 30], ylim=[0, 0.14], hspace=-0.5)


if __name__ == '__main__':
    _example()
