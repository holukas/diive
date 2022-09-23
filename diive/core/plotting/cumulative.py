from pandas import Series

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

    def _add_reference(self):

        # Calculate reference
        self.mean_doy_cumulative_df = doy_mean_cumulative(cumulatives_per_year_df=self.cumulatives_per_year_df,
                                                          excl_years_from_reference=self.excl_years_from_reference)

        # label = f"{year}: {cumulative_df[year].dropna().iloc[-1]:.2f}"
        self.ax.plot_date(x=self.mean_doy_cumulative_df.index.values,
                          y=self.mean_doy_cumulative_df['MEAN_DOY_TIME'].values,
                          color='black', alpha=1,
                          ls='-', lw=theme.WIDTH_LINE_WIDER,
                          marker='', markeredgecolor='none', ms=0,
                          zorder=99, label='mean')
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
                          txt_xlabel='Month',
                          txt_ylabel=self.varname,
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

    def plot(self, showplot: bool = True):
        color_list = theme.colorwheel_36()  # get some colors

        # Plot yearly cumulatives
        for ix, year in enumerate(self.cumulatives_per_year_df.columns):
            label = f"{year}: {self.cumulatives_per_year_df[year].dropna().iloc[-1]:.2f}"
            lw = theme.WIDTH_LINE_WIDER if year == self.highlight_year else theme.WIDTH_LINE_DEFAULT
            color = self.highlight_year_color if year == self.highlight_year else color_list[ix]

            self.ax.plot_date(x=self.cumulatives_per_year_df.index,
                              y=self.cumulatives_per_year_df[year],
                              color=color, alpha=1,
                              ls='-', lw=lw,
                              marker='', markeredgecolor='none', ms=0,
                              zorder=99, label=label)

        # Show reference
        if self.show_reference:
            self._add_reference()

        self._apply_format()

        if showplot:
            self.fig.show()


def example():
    # # Test data
    # from diive.core.io.filereader import ReadFileType
    # loaddatafile = ReadFileType(
    #     filetype='DIIVE_CSV_30MIN',
    #     filepath=r"M:\Downloads\_temp\CH_LAE_FP2021_2004-2020_ID20210607205711.diive.csv",
    #     # filepath=r"L:\Dropbox\luhk_work\_current\fp2022\7-14__IRGA627572__addingQCF0\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv",
    #     data_nrows=None)
    # data_df, metadata_df = loaddatafile.get_filedata()
    #
    # from diive.core.io.files import save_as_pickle
    # filepath = save_as_pickle(
    #     outpath=r"M:\Downloads\_temp",
    #     # outpath=r'L:\Dropbox\luhk_work\_current\fp2022\7-14__IRGA627572__addingQCF0',
    #     filename='CH_LAE_FP2021_2004-2020_ID20210607205711.diive.csv',
    #     # filename='CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv',
    #     data=data_df)

    # Test data
    from diive.core.io.files import load_pickle
    df_orig = load_pickle(
        filepath=r"M:\Downloads\_temp\CH_LAE_FP2021_2004-2020_ID20210607205711.diive.csv.pickle"
        # filepath=r'L:\Dropbox\luhk_work\_current\fp2022\7-14__IRGA627572__addingQCF0\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv.pickle'
    )

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

    series = df['NEE_f'].copy()
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
    example()
