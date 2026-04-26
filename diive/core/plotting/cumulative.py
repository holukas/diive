from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.times.times import remove_after_date, keep_years, doy_cumulatives_per_year, doy_mean_cumulative


class CumulativeYear:
    """Plot yearly cumulative sums with reference mean and standard deviation.

    Visualizes cumulative sums for each year with optional reference period
    mean and ±1 standard deviation band for comparison.

    Example:
        See `examples/visualization/timeseries_and_cumulative.py` for complete examples
        including reference bands and year highlighting.
    """

    def __init__(self,
                 series: Series,
                 series_units: str = None,
                 yearly_end_date: str = None,
                 start_year: int = None,
                 end_year: int = None,
                 show_reference: bool = False,
                 excl_years_from_reference: list = None,
                 highlight_year: int = None,
                 highlight_year_color: str = '#F44336'):
        """

        Args:
            series: Data for cumulative
            ax: XXX
            yearly_end_date: Calculate cumulatives up to this date, given as string,
                e.g. "08-11" means that cumulatives for each year are calculated until
                    11 August of each year.
            cumulative_for:
                - "all" for cumulative sum over all years
                - "year" for yearly cumulative sums
        """

        self.series = series.copy()
        self.series_units = series_units
        self.varname = series.name
        self.start_year = start_year if start_year else self.series.index.year.min()
        self.end_year = end_year if end_year else self.series.index.year.max()
        self.yearly_end_date = yearly_end_date
        self.show_reference = show_reference
        self.excl_years_from_reference = excl_years_from_reference
        self.highlight_year = highlight_year
        self.highlight_year_color = highlight_year_color

        self.series = self.series.dropna()
        # self.series_full = self.series.copy()

        self.series = keep_years(data=self.series, start_year=self.start_year, end_year=self.end_year)

        if self.yearly_end_date:
            self.series = remove_after_date(data=self.series, yearly_end_date=yearly_end_date)

        self.uniq_years = self.series.index.year.unique()  # Unique years

        # self.series_long_df = calc_doy_timefraction(input_series=self.series)
        self.cumulatives_per_year_df = doy_cumulatives_per_year(series=self.series)

        # Remove data for DOY 366
        self.cumulatives_per_year_df = self.cumulatives_per_year_df.loc[self.cumulatives_per_year_df.index < 366]

        # Create axis
        self.fig, self.ax = pf.create_ax()
        self.ax.xaxis.axis_date()


    def _add_reference(self, digits_after_comma):
        # Calculate reference
        self.mean_doy_cumulative_df = doy_mean_cumulative(cumulatives_per_year_df=self.cumulatives_per_year_df,
                                                          excl_years_from_reference=self.excl_years_from_reference)

        # label = f"{year}: {cumulative_df[year].dropna().iloc[-1]:.2f}"
        mean_end = self.mean_doy_cumulative_df['MEAN_DOY_TIME'].iloc[-1]
        self.ax.plot(self.mean_doy_cumulative_df.index.values,
                     self.mean_doy_cumulative_df['MEAN_DOY_TIME'].values,
                     color='black', alpha=1,
                     ls='-', lw=theme.WIDTH_LINE_WIDER,
                     marker='', markeredgecolor='none', ms=0,
                     zorder=99, label=f'mean {mean_end:.{digits_after_comma}f}')
        # self.ax.fill_between(mean_cumulative_df.index.values,
        #                      mean_cumulative_df['MEAN+1.96_SD'].values,
        #                      mean_cumulative_df['MEAN-1.96_SD'].values,
        #                      alpha=.005, zorder=0, color='black', edgecolor='none',
        #                      label="XXX")
        self.ax.fill_between(self.mean_doy_cumulative_df.index.values,
                             self.mean_doy_cumulative_df['MEAN+SD'].values,
                             self.mean_doy_cumulative_df['MEAN-SD'].values,
                             alpha=.1, zorder=0, color='black', edgecolor='none',
                             label="mean±1sd")


    def _apply_format(self):
        title = f"Cumulatives per year ({self.uniq_years.min()}-{self.uniq_years.max()}), " \
                f"until DOY {int(self.cumulatives_per_year_df.index[-1])}"
        self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)

        ymin = self.cumulatives_per_year_df.min().min()
        ymax = self.cumulatives_per_year_df.max().max()
        self.ax.set_ylim(ymin, ymax)

        pf.add_zeroline_y(ax=self.ax, data=self.cumulatives_per_year_df)

        pf.default_format(ax=self.ax,
                          ax_xlabel_txt='Month',
                          ax_ylabel_txt=self.varname,
                          txt_ylabel_units=self.series_units)

        n_legend_cols = pf.n_legend_cols(n_legend_entries=len(self.uniq_years))
        pf.default_legend(ax=self.ax,
                          labelspacing=0.2,
                          ncol=n_legend_cols)

        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='month')

        # self.fig.tight_layout()


    def get(self):
        """Return axis"""
        return self.ax


    def plot(self, showplot: bool = True, digits_after_comma: int = 2):
        color_list = theme.colorwheel_36()  # get some colors

        # Plot yearly cumulatives
        for ix, year in enumerate(self.cumulatives_per_year_df.columns):
            label = f"{year}: {self.cumulatives_per_year_df[year].dropna().iloc[-1]:.{digits_after_comma}f}"
            lw = theme.WIDTH_LINE_WIDER if year == self.highlight_year else theme.WIDTH_LINE_DEFAULT
            color = self.highlight_year_color if year == self.highlight_year else color_list[ix]

            self.ax.plot(self.cumulatives_per_year_df.index,
                         self.cumulatives_per_year_df[year],
                         color=color, alpha=1,
                         ls='-', lw=lw,
                         marker='', markeredgecolor='none', ms=0,
                         zorder=99, label=label)

        # Show reference
        if self.show_reference:
            self._add_reference(digits_after_comma=digits_after_comma)

        self._apply_format()

        if showplot:
            self.fig.show()


class Cumulative:
    """Plot cumulative sums across all data.

    Visualizes cumulative sums for each column in the input DataFrame
    with values displayed at the end of each series.

    Args:
        df: Dataframe, cumulatives are plotted for each column.
        units: Units shown in plot, LaTeX format is supported.
        start_year: Start year of shown data.
        end_year: End year of shown data.

    Example:
        See `examples/visualization/timeseries_and_cumulative.py` for complete examples
        including multiple USTAR scenarios and year ranges.
    """

    def __init__(self,
                 df: DataFrame,
                 units: str = None,
                 start_year: int = None,
                 end_year: int = None):
        self.df = df
        self.units = units
        self.start_year = start_year if start_year else self.df.index.year.min()
        self.end_year = end_year if end_year else self.df.index.year.max()

        if self.start_year | self.end_year:
            self.df = keep_years(data=self.df, start_year=self.start_year, end_year=self.end_year)

        self.show_grid = True
        self.show_legend = True
        self.ylabel = None
        self.show_title = True

        self.cumulative = self.df.cumsum()

        # Create axis
        self.fig, self.ax = pf.create_ax()
        self.ax.xaxis.axis_date()

    def _apply_format(self):
        if self.show_title:
            title = f"Cumulatives ({self.cumulative.index.min()}-{self.cumulative.index.max()})"
            self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        ymin = self.cumulative.min().min()
        ymax = self.cumulative.max().max()
        ymax = ymax * 1.05 if ymax > 0 else ymax * 0.95
        ymin = ymin * 0.95 if ymin > 0 else ymin * 1.05
        self.ax.set_ylim(ymin, ymax)
        pf.add_zeroline_y(ax=self.ax, data=self.cumulative)
        ax_ylabel_txt = self.ylabel if self.ylabel else "Cumulative"
        pf.default_format(ax=self.ax,
                          ax_xlabel_txt="Date",
                          ax_ylabel_txt=ax_ylabel_txt,
                          txt_ylabel_units=self.units,
                          showgrid=self.show_grid)
        n_legend_cols = pf.n_legend_cols(n_legend_entries=len(self.cumulative.columns))
        if self.show_legend:
            pf.default_legend(ax=self.ax,
                              labelspacing=0.2,
                              ncol=n_legend_cols)
        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        # self.fig.tight_layout()

    def get_ax(self):
        """Return axis"""
        return self.ax

    def plot(self, showplot: bool = True,
             digits_after_comma: int = 0,
             show_grid: bool = True,
             show_legend: bool = True,
             ylabel: str = None,
             show_title: bool = True
             ):
        self.show_grid = show_grid
        self.show_legend = show_legend
        self.ylabel = ylabel
        self.show_title = show_title

        color_list = theme.colorwheel_36()  # get some colors

        # Plot yearly cumulatives
        for ix, col in enumerate(self.cumulative):
            series = self.cumulative[col]
            label = f"{col}: {series.dropna().iloc[-1]:.{digits_after_comma}f}"
            lw = theme.WIDTH_LINE_DEFAULT
            color = color_list[ix]

            self.ax.plot(series.index, series,
                         color=color, alpha=1,
                         ls='-', lw=lw,
                         marker='', markeredgecolor='none', ms=0,
                         zorder=99, label=label)

            x = series.index[-1]
            y = series.iloc[-1]

            self.ax.plot(x, y, color=color, alpha=1, marker="o", markeredgecolor=color, ms=10)

            self.ax.text(x, y, f"  {y:.{digits_after_comma}f}",
                         size=12, color=color,
                         backgroundcolor='none', alpha=1,
                         horizontalalignment='left', verticalalignment='center', zorder=99)

        self._apply_format()

        if showplot:
            self.fig.show()
