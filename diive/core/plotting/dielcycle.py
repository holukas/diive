import calendar

import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import set_fig
from diive.core.plotting.styles.format import FormatStyle
from diive.core.times.resampling import diel_cycle


class DielCycle:

    def __init__(self, series: Series):
        """Plot diel cycles of time series.

        Args:
            series: Time series with datetime index.
                The index must contain date and time info.

        See Also:
            examples/visualization/dielcycle.py — Diurnal cycle analysis and visualization
        """
        self.series = series

        self.var = self.series.name

        self.fig = None
        self.ax = None
        self.showplot = False
        self._diel_cycles_df = None
        self.show_xticklabels = True
        self.show_xlabel = True

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
             format_style: FormatStyle = None,
             color: str = None,
             mean: bool = True,
             std: bool = True,
             each_month: bool = False,
             linewidth: float = 2,
             ylim: list = None,
             show_xticklabels: bool = True,
             show_xlabel: bool = True,
             **kwargs):
        """Plot the diel (24-hour) cycle, optionally one curve per month.

        Chrome (title, labels, units, font sizes, colours, grid, legend, zero
        line) comes from a shared :class:`~diive.plotting.FormatStyle`.

        Args:
            format_style: A :class:`~diive.plotting.FormatStyle` describing the
                chrome. When None the diive house style is used.
        """

        self.show_xticklabels = show_xticklabels
        self.show_xlabel = show_xlabel

        style = format_style or FormatStyle()

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
            means_numeric.plot(ax=self.ax, label=f'{monthstr}', color=color, zorder=99,
                                lw=linewidth, **kwargs)

            # label = "Mittelwert ± 1 Standardabweichung"
            label = None
            # label = "mean±1sd" if counter == 0 else ""
            self.ax.fill_between(x_decimal,
                                 means_add_sd.to_numpy(),
                                 means_sub_sd.to_numpy(),
                                 alpha=alpha, zorder=0, color=color, edgecolor='none',
                                 label=label)

        # Shared formatting layer: title/labels/units/fonts/grid/legend/zeroline.
        style.apply(ax=self.ax, default_xlabel="Time (hours of day)",
                    default_ylabel=self.series.name,
                    zeroline_data=self.diel_cycles_df['mean'])

        # show_xlabel=False blanks the (already-set) x-label.
        if not self.show_xlabel:
            self.ax.set_xlabel("")

        # Fixed 0-24h diel x-axis with hourly ticks (domain-specific, not chrome).
        self.ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        if not self.show_xticklabels:
            self.ax.set_xticklabels([])
        self.ax.set_xlim([0, 24])
        self.ax.set_ylim(ylim)

        if self.showplot:
            self.fig.tight_layout()
            self.fig.show()
