from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.times.times import remove_after_date, keep_years, doy_cumulatives_per_year, doy_mean_cumulative


class CumulativeYear:

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
        self.start_year = start_year
        self.end_year = end_year
        self.yearly_end_date = yearly_end_date
        self.show_reference = show_reference
        self.excl_years_from_reference = excl_years_from_reference
        self.highlight_year = highlight_year
        self.highlight_year_color = highlight_year_color

        self.series = self.series.dropna()
        # self.series_full = self.series.copy()

        if (self.start_year) | (self.end_year):
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

        self.fig.tight_layout()

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

    def __init__(self,
                 df: DataFrame,
                 units: str = None,
                 start_year: int = None,
                 end_year: int = None):
        """Plot cumulative sums across all data.

        Args:
            df: Dataframe, cumulatives are plotted for each column.
            units: Units shown in plot, LaTeX format is supported.
            start_year: Start year of shown data.
            end_year: End year of shown data.
        """
        self.df = df
        self.units = units
        self.start_year = start_year
        self.end_year = end_year

        if self.start_year | self.end_year:
            self.df = keep_years(data=self.df, start_year=self.start_year, end_year=self.end_year)

        self.cumulative = self.df.cumsum()

        # Create axis
        self.fig, self.ax = pf.create_ax()
        self.ax.xaxis.axis_date()

    def _apply_format(self):
        title = f"Cumulatives ({self.cumulative.index.min()}-{self.cumulative.index.max()})"
        self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        ymin = self.cumulative.min().min()
        ymax = self.cumulative.max().max()
        self.ax.set_ylim(ymin, ymax)
        pf.add_zeroline_y(ax=self.ax, data=self.cumulative)
        pf.default_format(ax=self.ax,
                          ax_xlabel_txt="Date",
                          ax_ylabel_txt="Cumulative",
                          txt_ylabel_units=self.units)
        n_legend_cols = pf.n_legend_cols(n_legend_entries=len(self.cumulative.columns))
        pf.default_legend(ax=self.ax,
                          labelspacing=0.2,
                          ncol=n_legend_cols)
        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        self.fig.tight_layout()

    def get_ax(self):
        """Return axis"""
        return self.ax

    def plot(self, showplot: bool = True, digits_after_comma: int = 0):
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


def example_cum_overall():
    # Test data
    from diive.configs.exampledata import load_exampledata_parquet
    df_orig = load_exampledata_parquet()
    df = df_orig[['NEE_CUT_16_f', 'NEE_CUT_REF_f', 'NEE_CUT_84_f']].copy()
    df = df.multiply(0.02161926)  # umol CO2 m-2 s-1 --> g C m-2 30min-1
    series_units = r'($\mathrm{gC\ m^{-2}}$)'
    Cumulative(
        df=df,
        units=series_units,
        start_year=2015,
        end_year=2019).plot()


def example_cum_year():
    # Test data
    from diive.configs.exampledata import load_exampledata_parquet
    df_orig = load_exampledata_parquet()

    df = df_orig.copy()

    # print(df_orig)

    # # Keep daytime values only
    # df = df.loc[df['PotRad_CUT_REF'] > 20, :].copy()

    # # Short code snippet to check numbers in the ICOS Fluxes Bulletin 1 (2022)
    # df = df.loc[(df.index.month >= 7) & (df.index.month <= 8)]
    # series = df['NEE_CUT_REF_f'].dropna()
    # series_longterm = series.loc[(series.index.year <= 2021)].copy()
    # series_longterm = series_longterm.multiply(0.02161926)  # -> g C m-2 30min-1
    # series_longterm = series_longterm.groupby(series_longterm.index.date).sum()  # daily sums
    # series_longterm.index = pd.to_datetime(series_longterm.index)
    # series_longterm = series_longterm.groupby(series_longterm.index.year).mean()  # yearly mean

    # series = df['NEE_CUT_REF_f'].dropna()
    # series = series.loc[(series.index.year >= 2015)&(series.index.year <= 2021)]
    # series = series.multiply(0.02161926)  # -> g C m-2 30min-1
    # series = series.groupby(series.index.date).sum()  # daily sums
    # series.index = pd.to_datetime(series.index)
    # series = series.groupby(series.index.year).mean()  # yearly mean

    series = df['NEE_CUT_REF_f'].copy()
    # series = df['NEE_CUT_REF_f'].copy()
    series = series.multiply(0.02161926)  # umol CO2 m-2 s-1 --> g C m-2 30min-1
    series_units = r'($\mathrm{gC\ m^{-2}}$)'
    # series_units = '(umol CO2 m-2 s-1)'
    # series = df['VPD_f'].copy()
    # series_units = '(hPa)'
    # series = df['Tair_f'].copy()
    # series_units = '(°C)'
    CumulativeYear(
        series=series,
        series_units=series_units,
        yearly_end_date=None,
        # yearly_end_date='08-11',
        start_year=2005,
        end_year=2020,
        show_reference=True,
        excl_years_from_reference=None,
        # excl_years_from_reference=[2022],
        # highlight_year=2022,
        highlight_year_color='#F44336').plot()


if __name__ == '__main__':
    example_cum_overall()
    # example_cum_year()
