import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from pandas import Series


# TODO
# TODO
# TODO
# TODO
# TODO

class TimeSeries:
    def __init__(self,
                 series: Series,
                 ax=None,
                 series_units: str = None):
        """

        """
        self.series = series.copy()
        self.series_units = series_units
        self.varname = series.name

        self.series = self.series.dropna()

        # Create axis
        if ax:
            # If ax is given, plot directly to ax, no fig needed
            self.fig = None
            self.ax = ax
            self.showplot = False
        else:
            # If no ax is given, create fig and ax and then show the plot
            self.fig, self.ax = pf.create_ax()
            self.showplot = True

    def plot(self):
        color_list = theme.colorwheel_36()  # get some colors

        label = self.series.name

        self.ax.plot_date(x=self.series.index,
                          y=self.series,
                          color=color_list[0], alpha=1,
                          ls='-', lw=theme.WIDTH_LINE_DEFAULT,
                          marker='', markeredgecolor='none', ms=0,
                          zorder=99, label=label)

        self._apply_format()

        if self.showplot:
            self.fig.show()

    def _apply_format(self):

        # ymin = self.series.min()
        # ymax = self.series.max()
        # self.ax.set_ylim(ymin, ymax)

        pf.add_zeroline_y(ax=self.ax, data=self.series)

        pf.default_format(ax=self.ax,
                          txt_xlabel='Date',
                          txt_ylabel=self.varname,
                          txt_ylabel_units=self.series_units)

        pf.default_legend(ax=self.ax,
                          labelspacing=0.2,
                          ncol=1)

        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        if self.showplot:
            title = f"{self.series.name}"
            self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
            self.fig.tight_layout()


def example():
    # Test data
    from diive.core.io.filereader import ReadFileType
    loaddatafile = ReadFileType(
        filetype='DIIVE_CSV_30MIN',
        filepath=r"F:\Dropbox\luhk_work\20 - CODING\21 - DIIVE\diive-gui\src\main\resources\base\example_files\ExampleFile_DIIVE_CSV_30T.diive.csv",
        # filepath=r"F:\Dropbox\luhk_work\_current\fp2022\7-14__IRGA627572__addingQCF0\CH-DAV_FP2022.1_1997-2022.08_ID20220826234456_30MIN.diive.csv",
        data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()

    series_col = 'co2_flux'
    series = data_df[series_col].copy()
    series_units = metadata_df.loc[series_col]['UNITS']

    fig, ax = pf.create_ax()
    # series_units = r'($\mathrm{gC\ m^{-2}}$)'
    TimeSeries(ax=ax,
               series=series,
               series_units=series_units).plot()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    example()
