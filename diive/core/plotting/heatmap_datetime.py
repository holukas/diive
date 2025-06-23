"""
HEATMAP
=======

Kudos:
    - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/
    pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py

"""
import datetime

import numpy as np
from pandas import Series
from pandas.plotting import register_matplotlib_converters

import diive as dv
import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.times.times import TimestampSanitizer, insert_timestamp


class HeatmapDateTime(HeatmapBase):

    def __init__(self, series: Series, fig=None, ax=None, title: str = None, vmin: float = None, vmax: float = None,
                 cb_digits_after_comma: int = 2, cb_labelsize: float = theme.AX_LABELS_FONTSIZE,
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE, minyticks: int = 3, maxyticks: int = 10,
                 cmap: str = 'RdYlBu_r', color_bad: str = 'grey', zlabel: str = "Value",
                 figsize: tuple = (6, 10.7), verbose: bool = False):
        """Plot heatmap of time series data with date on y-axis and time on x-axis.

        Args:
            series: Series
            ax: Axis in which heatmap is shown. If *None*, a figure with axis will be generated.
            title: Text shown at the top of the plot.
            vmin: Minimum value shown in plot
            vmax: Maximum value shown in plot
            cb_digits_after_comma: How many digits after the comma are shown in the colorbar legend.
            cmap: Matplotlib colormap
            color_bad: Color of missing values

        """
        super().__init__(series, fig, ax, title, vmin, vmax, cb_digits_after_comma, cb_labelsize, axlabels_fontsize,
                         ticks_labelsize, minyticks, maxyticks, cmap, color_bad, figsize, zlabel, verbose)

        # Setup data for plotting
        self.series = self._setup_timestamp()
        self.plot_df, self.fig, self.ax = self.setup()
        xaxis_vals, yaxis_vals = self._set_xy_axes_type()
        self.x, self.y, self.z, self.plot_df = self.transform_data(xaxis_vals=xaxis_vals, yaxis_vals=yaxis_vals)
        self.x, self.y, self.z = self._set_bounds(x=self.x, y=self.y, z=self.z)

    def _setup_timestamp(self) -> Series:
        # Sanitize timestamp
        # TimestampSanitizer outputs TIMESTAMP_MIDDLE by default.
        series = self.series.copy()

        series = TimestampSanitizer(
            data=series,
            output_middle_timestamp=True,
            validate_naming=True,
            convert_to_datetime=True,
            sort_ascending=True,
            remove_duplicates=True,
            regularize=True,
            verbose=self.verbose
        ).get()

        # Heatmap works best with TIMESTAMP_START, convert timestamp index if needed
        # Working with a timestamp that shows the start of the averaging period means
        # that e.g. the ticks on the x-axis for hours is shown before the corresponding
        # data between e.g. 1:00 and 1:59.
        if series.index.name != 'TIMESTAMP_START':
            _series_df = insert_timestamp(data=series, convention='start', set_as_index=True)
            series = _series_df[series.name].copy()
        return series

    @staticmethod
    def _set_bounds(x, y, z):
        """Extend data for plotting

        x and y are bounds, so z should be the value *inside* those bounds.
        Therefore, extend x and y by one value (last_x, last_y).
        """

        # Add last entry for x (datetime)
        # x-axis shows hours 0, 1, 2 ... 23
        last_x = x[-1]
        last_x = last_x.replace(hour=23, minute=59)
        x = np.append(x, last_x)

        # Add last entry for y (datetime)
        # y-axis shows years
        last_y = y[-1]
        last_y = last_y + datetime.timedelta(days=1)
        y = np.append(y, last_y)

        return x, y, z

    def _set_xy_axes_type(self):
        xaxis_vals = self.plot_df.index.time
        yaxis_vals = self.plot_df.index.date
        register_matplotlib_converters()  # Needed for time plotting
        return xaxis_vals, yaxis_vals

    def plot(self):
        """Plot heatmap"""
        cmap, z = self.set_cmap(cmap=self.cmap, color_bad=self.color_bad, z=self.z)

        p = self.ax.pcolormesh(self.x, self.y, self.z,
                               linewidths=1, cmap=cmap,
                               vmin=self.vmin, vmax=self.vmax,
                               shading='flat', zorder=99)

        # import matplotlib.pyplot as plt
        # from matplotlib.colors import ListedColormap
        # import numpy as np
        #
        # mat = np.random.randint(0, 3, (5, 8))
        # cmap = ListedColormap(['lime', 'orange', 'tomato'])
        #
        # plt.pcolormesh(np.arange(-0.5, mat.shape[1]), np.arange(-0.5, mat.shape[0]), mat, cmap=cmap)
        #
        # color_dict = {0: 'normal', 1: 'high', 2: 'very\nhigh'}
        # for i in range(mat.shape[0]):
        #     for j in range(mat.shape[1]):
        #         plt.text(j, i, color_dict[mat[i, j]], ha='center', va='center')
        # plt.show()

        # Ticks
        ax_xlabel_txt = 'Time (hours)'
        ax_ylabel_txt = 'Date'
        self.ax.set_xticks(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        # # matplotlib's HourLocator did not work
        # nice_date_ticks(ax=self.ax, minticks=1, maxticks=24, which='x', locator='hour')
        # For the y-axis AutoDateLocator worked
        nice_date_ticks(ax=self.ax, minticks=self.minyticks, maxticks=self.maxyticks, which='y')

        # Format
        self.format(
            ax_xlabel_txt=ax_xlabel_txt,
            ax_ylabel_txt=ax_ylabel_txt,
            plot=p
        )


# @ConsoleOutputDecorator(spacing=False)
class HeatmapYearMonth(HeatmapBase):

    def __init__(self,
                 series: Series,
                 agg: str = 'mean',
                 ranks: bool = False,
                 cmap: str = None,
                 **kwargs):
        """Plot heatmap of time series data with year and month.

        Args:
            series: Series in monthly time resolution.
            **kwargs: Parameter for HeatmapBase.

        """
        super().__init__(series, **kwargs)

        self.agg = agg
        self.ranks = ranks

        if not cmap:
            self.cmap = 'RdYlBu' if ranks else 'RdYlBu_r'
        else:
            self.cmap = cmap

        # Set xylabels
        self.xlabel = 'Month' if self.ax_orientation == "vertical" else 'Year'
        self.ylabel = 'Year' if self.ax_orientation == "vertical" else 'Month'

        self._prepare_data()

    def _prepare_data(self):
        # Bring data into shape
        self.series = self._sanitize_original_timestamp()
        self.plotdf = dv.resample_to_monthly_agg_matrix(series=self.series, agg=self.agg, ranks=self.ranks)

        # Transpose in case of horizontal, to have months as index, years as columns
        if self.ax_orientation == "horizontal":
            self.plotdf = self.plotdf.transpose()

        x = self.plotdf.columns.values
        y = self.plotdf.index.values
        z = self.plotdf.values
        self.x, self.y, self.z = self._set_bounds(x=x, y=y, z=z)

    def _sanitize_original_timestamp(self) -> Series:
        # Sanitize timestamp
        # TimestampSanitizer outputs TIMESTAMP_MIDDLE by default.
        series = self.series.copy()

        series = TimestampSanitizer(
            data=series,
            output_middle_timestamp=True,
            validate_naming=True,
            convert_to_datetime=True,
            sort_ascending=True,
            remove_duplicates=True,
            regularize=True,
            verbose=self.verbose
        ).get()

        # Heatmap works best with TIMESTAMP_START, convert timestamp index if needed
        # Working with a timestamp that shows the start of the averaging period means
        # that e.g. the ticks on the x-axis for hours is shown before the corresponding
        # data between e.g. 1:00 and 1:59.
        if series.index.name != 'TIMESTAMP_START':
            _series_df = insert_timestamp(data=series, convention='start', set_as_index=True)
            series = _series_df[series.name].copy()
        return series

    @staticmethod
    def _set_bounds(x, y, z):
        """Extend data for plotting

        x and y are bounds, so z should be the value *inside* those bounds.
        Therefore, extend x and y by one value (last_x, last_y).
        """

        # Add last entry for x (int)
        # x-axis shows months or years
        last_x = x[-1] + 1
        x = np.append(x, last_x)

        # Add last entry for y (int)
        # y-axis shows months or years
        last_y = y[-1] + 1
        y = np.append(y, last_y)

        return x, y, z

    def plot(self):
        """Plot heatmap"""
        cmap, z = self.set_cmap(cmap=self.cmap, color_bad=self.color_bad, z=self.z)
        # Run._remove_cbar(ax=ax)
        p = self.ax.pcolormesh(self.x, self.y, self.z,
                               linewidths=1, cmap=cmap,
                               vmin=self.vmin, vmax=self.vmax,
                               shading='flat', zorder=99)

        if self.show_values:
            self.show_vals_in_plot()

        # Set ticks for months and years
        xtickpos = np.arange(self.x[0] + 0.5, self.x[-1] + 0.5, 1)
        self.ax.set_xticks(xtickpos)
        xticklabels = [int(t) for t in xtickpos]
        self.ax.set_xticklabels(xticklabels)
        ytickpos = np.arange(self.y[0] + 0.5, self.y[-1] + 0.5, 1)
        self.ax.set_yticks(ytickpos)
        yticklabels = [int(t) for t in ytickpos]
        self.ax.set_yticklabels(yticklabels)

        # Get all current x-axis tick labels
        if self.show_less_xticklabels:
            labels = self.ax.get_xticklabels()
            # Iterate through the labels and hide every second one
            for i, label in enumerate(labels):
                if i % 2 != 0:  # Check if the index is odd (to hide every second, starting from the second label)
                    label.set_visible(False)

        # # # nice_date_ticks(ax=self.ax, minticks=1, maxticks=24, which='y', locator='year')
        # # Use Locator and Formatter to label every year or hour on y-axis
        # locator = MultipleLocator(2)  # Set ticks every 1 unit
        # formatter = FormatStrFormatter('%d')  # Integer format
        # self.ax.yaxis.set_major_locator(locator)
        # self.ax.yaxis.set_major_formatter(formatter)

        # Format
        self.format(
            ax_xlabel_txt=self.xlabel,
            ax_ylabel_txt=self.ylabel,
            plot=p,
            shown_freq=f'{self.agg}, MS'
        )


def _example_heatmap_datetime():
    # import diive as dv
    # from diive.configs.exampledata import load_exampledata_parquet
    # df = load_exampledata_parquet()

    from diive.core.io.filereader import ReadFileType
    filepath = r"F:\Sync\luhk_work\40 - DATA\DATASETS\2025_FORESTS\1-downloads\ICOSETC_CH-Dav_ARCHIVE_L2\ICOSETC_CH-Dav_FLUXNET_HH_L2.csv"
    loaddatafile = ReadFileType(filetype='FLUXNET-FULLSET-HH-CSV-30MIN',
                                filepath=filepath,
                                data_nrows=None)
    df, metadata_df = loaddatafile.get_filedata()

    var = 'RECO_NT_CUT_50'
    series = df[var].copy()
    # series = series.resample('3h', label='left').mean()
    series = series.loc[series.index.year >= 2020]
    series.index.name = 'TIMESTAMP_START'
    series.name = var
    hm = dv.heatmapdatetime(series=series, title=None, vmin=-20, vmax=20)
    hm.show()
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


def _example_multiple_heatmap_yearmonth_ranks():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from diive.configs.exampledata import load_exampledata_parquet_long
    df = load_exampledata_parquet_long()
    series = df['Tair_f'].copy()

    # Figure
    fig = plt.figure(facecolor='white', figsize=(16, 9))

    # Gridspec for layout
    gs = gridspec.GridSpec(1, 3)  # rows, cols
    gs.update(wspace=0.5, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_min = fig.add_subplot(gs[0, 1])
    ax_max = fig.add_subplot(gs[0, 2])

    dv.heatmapyearmonth_ranks(ax=ax_mean, series=series, agg='mean').plot()
    dv.heatmapyearmonth_ranks(ax=ax_min, series=series, agg='min').plot()
    dv.heatmapyearmonth_ranks(ax=ax_max, series=series, agg='max').plot()

    ax_mean.set_title("Air temperature mean", color='black')
    ax_min.set_title("Air temperature min", color='black')
    ax_max.set_title("Air temperature max", color='black')

    ax_mean.tick_params(left=True, right=False, top=False, bottom=True,
                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    ax_min.tick_params(left=True, right=False, top=False, bottom=True,
                       labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    ax_max.tick_params(left=True, right=False, top=False, bottom=True,
                       labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    fig.show()


def _example_heatmap_yearmonth():
    import numpy as np
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['Tair_f'].copy()
    series.name = None  # For testing
    series.index.freq = None  # For testing
    series.iloc[100:120] = np.nan
    dv.heatmapyearmonth(series=series, ax_orientation="horizontal", ranks=True, show_values=True).show()
    dv.heatmapyearmonth(series=series, ax_orientation="vertical", ranks=False, show_values=True).show()
    # dv.heatmapyearmonth(series_monthly=series, cb_digits_after_comma=0, zlabel="degC",
    #                     ax_orientation="horizontal", figsize=(14, 10)).show()
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


def _example_multiple_heatmaps_yearmonth_horizontal():
    # Figure (top-to-bottom, horizontal plots)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    fig = plt.figure(facecolor='white', figsize=(16, 9), layout="constrained")
    gs = gridspec.GridSpec(2, 1, figure=fig)  # rows, cols
    # gs.update(wspace=0.5, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    zlabel = r'$\mathrm{\mu mol\ CO_2\ m^{-2}\ s^{-1}}$'
    settings = dict(ax_orientation='horizontal', zlabel=zlabel, cb_digits_after_comma=0,
                    show_values=True, show_values_n_dec_places=0)
    series = df['NEE_CUT_REF_f']
    dv.heatmapyearmonth(ax=ax1, series=series, agg='mean', **settings).plot()
    dv.heatmapyearmonth(ax=ax2, series=series, agg=np.ptp, **settings).plot()
    ax1.set_xlabel("")
    ax1.set_xticklabels("")
    ax1.set_title("")
    ax2.set_title("")
    fig.suptitle("NEE")
    fig.show()


def _example_colormaps():
    """Creates heatmaps with all available colormaps and exports them to files."""
    from heatmap_base import list_of_colormaps
    from diive.configs.exampledata import load_exampledata_parquet
    cmaps = list_of_colormaps()
    df = load_exampledata_parquet()
    series = df['Tair_f'].copy()
    # series = series.resample('1MS', label='left').mean()
    # series.index.name = 'TIMESTAMP_START'
    for cmap in cmaps:
        hm = dv.heatmapyearmonth(series=series, cb_digits_after_comma=0, zlabel="degC",
                                 ax_orientation="vertical", figsize=(14, 10), cmap=cmap, title=cmap)
        hm.show()
        outfile = rf"F:\Sync\luhk_work\20 - CODING\matplotlib_colormaps\{cmap}.png"
        hm.fig.savefig(outfile)
        print(f"Saved {outfile}")
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


if __name__ == '__main__':
    # _example_heatmap_yearmonth_ranks()
    # _example_heatmap_datetime()
    _example_multiple_heatmaps_yearmonth_horizontal()
    # _example_heatmap_yearmonth()

# # from diive.core.io.files import load_parquet
#     # f1 = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-cha_flux_product"
#     # f2 = r"dataset_ch-cha_flux_product\notebooks\80_FINALIZE"
#     # f3 = f"81.1_FLUXES_M15_MGMT_L4.2_NEE_GPP_RECO_LE_H_FN2O_FCH4.parquet"
#     # f = f"{f1}\\{f2}\\{f3}"
#     # df = load_parquet(f)
#     # # [print(c) for c in df.columns if "FN2O" in c];
#     # nee_monthly = df['NEE_L3.1_L3.3_CUT_50_QCF_gfRF'].resample('1MS', label='left').agg(
#     #     {'mean'})  # numpy's ptp gives the data range
#     # n2o_monthly = df['FN2O_L3.1_L3.3_CUT_50_QCF_gfRF'].resample('1MS', label='left').agg(
#     #     {'mean'})  # numpy's ptp gives the data range
