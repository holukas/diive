import calendar

import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, format_spines, default_legend, add_zeroline_y
from diive.core.plotting.plotfuncs import set_fig
from diive.core.times.resampling import diel_cycle


class DielCycle:

    def __init__(self, series: Series):
        """Plot diel cycles of time series.

        Args:
            series: Time series with datetime index.
                The index must contain date and time info.

        Example:
            See `examples/visualization/dielcycle.py` for complete examples.
        """
        self.series = series

        self.var = self.series.name

        self.fig = None
        self.ax = None
        self.showplot = False
        self.title = None
        self.ylabel = None
        self.txt_ylabel_units = None
        self._diel_cycles_df = None
        self.showgrid = True
        self.show_xticklabels = True
        self.show_xlabel = True
        self.show_legend = True

    def get_data(self) -> DataFrame:
        return self.diel_cycles_df

    @property
    def diel_cycles_df(self):
        """Return dataframe containing diel cycle aggregates."""
        if not isinstance(self._diel_cycles_df, DataFrame):
            raise Exception(f'No diel cycles dataframe available. Please run .plot() first.')
        return self._diel_cycles_df

    def plot(self,
             ax: plt.Axes = None,
             title: str = None,
             color: str = None,
             txt_ylabel_units: str = None,
             mean: bool = True,
             std: bool = True,
             each_month: bool = False,
             legend_n_col: int = 1,
             ylim: list = None,
             ylabel: str = None,
             showgrid: bool = True,
             show_xticklabels: bool = True,
             show_xlabel: bool = True,
             show_legend: bool = True,
             **kwargs):

        self.title = title
        self.txt_ylabel_units = txt_ylabel_units
        self.ylabel = ylabel
        self.showgrid = showgrid
        self.show_xticklabels = show_xticklabels
        self.show_xlabel = show_xlabel
        self.show_legend = show_legend

        # Resample
        self._diel_cycles_df = diel_cycle(series=self.series,
                                          mean=mean,
                                          std=std,
                                          each_month=each_month)

        months = set(self.diel_cycles_df.index.get_level_values(0).tolist())

        self.fig, self.ax, self.showplot = set_fig(ax=ax)

        counter_plotted = -1
        n_months = len(months)
        alpha = 0.05 if n_months > 10 else 0.1
        auto_color = True if not color else False
        for counter, month in enumerate(months):

            means = self.diel_cycles_df.loc[month]['mean']
            if means.isnull().all():
                continue
            else:
                counter_plotted += 1

            means_add_sd = self.diel_cycles_df.loc[month]['mean+sd']
            means_sub_sd = self.diel_cycles_df.loc[month]['mean-sd']

            if auto_color:
                color = theme.colors_12_months()[counter_plotted]

            # monthstr = calendar.month_name[month] if each_month else 'Mittelwert'
            monthstr = calendar.month_abbr[month] if each_month else 'mean'

            # Convert datetime.time index to decimal hours for a clean numeric x-axis.
            # Using datetime.time objects directly creates a categorical axis whose
            # registered categories may not include exact hour boundaries (e.g. when
            # timestamps are at :15/:45 offsets), causing set_xticks to fail.
            x_decimal = [t.hour + t.minute / 60 + t.second / 3600 for t in means.index]
            means_numeric = means.copy()
            means_numeric.index = x_decimal
            means_numeric.plot(ax=self.ax, label=f'{monthstr}', color=color, zorder=99, lw=2, **kwargs)

            # label = "Mittelwert ± 1 Standardabweichung"
            label = None
            # label = "mean±1sd" if counter == 0 else ""
            self.ax.fill_between(x_decimal,
                                 means_add_sd.values,
                                 means_sub_sd.values,
                                 alpha=alpha, zorder=0, color=color, edgecolor='none',
                                 label=label)

        self._format(title=title, legend_n_col=legend_n_col, ylim=ylim)

        if self.showplot:
            self.fig.show()

    def _format(self, title, legend_n_col, ylim):

        if title:
            self.ax.set_title(title, color='black', fontsize=24)
        ax_xlabel_txt = "Time (hours of day)"
        ylabel = self.ylabel if self.ylabel else self.series.name
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ylabel,
                       txt_ylabel_units=self.txt_ylabel_units,
                       ticks_direction='in', ticks_length=8, ticks_width=2,
                       ax_labels_fontsize=20,
                       ticks_labels_fontsize=20,
                       showgrid=self.showgrid)
        if not self.show_xlabel:
            self.ax.set_xlabel("")
        format_spines(ax=self.ax, color='black', lw=1)
        if self.show_legend:
            default_legend(ax=self.ax, ncol=legend_n_col)
        add_zeroline_y(ax=self.ax, data=self.diel_cycles_df['mean'])
        self.ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        if not self.show_xticklabels:
            self.ax.set_xticklabels([])
        self.ax.set_xlim([0, 24])
        self.ax.set_ylim(ylim)
        if self.showplot:
            self.fig.tight_layout()
