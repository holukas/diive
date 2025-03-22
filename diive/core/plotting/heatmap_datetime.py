"""
HEATMAP
=======

Kudos:
    - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py

"""
import datetime

import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pandas import Series
from pandas.plotting import register_matplotlib_converters

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
        # TimestampSanitizer outputs TIMESTAMP_MIDDLE.
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

    def __init__(self, series_monthly: Series, fig=None, ax=None, title: str = None, vmin: float = None,
                 vmax: float = None,
                 cb_digits_after_comma: int = 2, cb_labelsize: float = theme.AX_LABELS_FONTSIZE,
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE, minyticks: int = 3, maxyticks: int = 10,
                 cmap: str = 'RdYlBu_r', color_bad: str = 'grey', zlabel: str = "Value",
                 figsize: tuple = (6, 10.7), verbose: bool = False, show_values: bool = False,
                 show_values_n_dec_places: int = 0):
        """Plot heatmap of time series data with year on y-axis and month on x-axis.

        Args:
            series_monthly: Series in monthly time resolution.
            ax: Axis in which heatmap is shown. If *None*, a figure with axis will be generated.
            title: Text shown at the top of the plot.
            vmin: Minimum value shown in plot
            vmax: Maximum value shown in plot
            cb_digits_after_comma: How many digits after the comma are shown in the colorbar legend.
            cmap: Matplotlib colormap
            color_bad: Color of missing values
            show_values: Show z-values in plot.
            show_values_n_dec_places: Number of decimal places to show in heatmap.
                Only considered if *show_values* is True.

        """
        super().__init__(series_monthly, fig, ax, title, vmin, vmax, cb_digits_after_comma, cb_labelsize,
                         axlabels_fontsize, ticks_labelsize, minyticks, maxyticks, cmap, color_bad, figsize, zlabel,
                         show_values, show_values_n_dec_places, verbose)

        # Setup data for plotting
        self.series = self._setup_timestamp()
        self.plot_df, self.fig, self.ax = self.setup()
        xaxis_vals, yaxis_vals = self._set_xy_axes_type()
        self.x, self.y, self.z, self.plot_df = self.transform_data(xaxis_vals=xaxis_vals, yaxis_vals=yaxis_vals)
        self.x, self.y, self.z = self._set_bounds(x=self.x, y=self.y, z=self.z)

    def _setup_timestamp(self) -> Series:
        series = self.series.copy()
        series = TimestampSanitizer(
            data=series,
            output_middle_timestamp=False,  # Not needed for monthly time resolution
            validate_naming=False,  # Not needed for monthly time resolution
            convert_to_datetime=True,
            sort_ascending=True,
            remove_duplicates=True,
            regularize=True,
            verbose=self.verbose
        ).get()
        return series

    @staticmethod
    def _set_bounds(x, y, z):
        """Extend data for plotting

        x and y are bounds, so z should be the value *inside* those bounds.
        Therefore, extend x and y by one value (last_x, last_y).
        """

        # Add last entry for x (int)
        # x-axis shows months 1, 2, 3 ... 12
        last_x = x[-1] + 1
        x = np.append(x, last_x)

        # Add last entry for y (int)
        # y-axis shows years
        last_y = y[-1] + 1
        y = np.append(y, last_y)

        return x, y, z

    def _set_xy_axes_type(self):
        xaxis_vals = self.plot_df.index.month
        yaxis_vals = self.plot_df.index.year
        return xaxis_vals, yaxis_vals

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

        # Ticks
        ax_xlabel_txt = 'Month'
        ax_ylabel_txt = 'Year'
        tickpos = np.arange(1.5, 13.5, 1)
        self.ax.set_xticks(tickpos)
        ticklabels = [int(t) for t in tickpos]
        self.ax.set_xticklabels(ticklabels)

        # nice_date_ticks(ax=self.ax, minticks=1, maxticks=24, which='y', locator='year')
        # Use Locator and Formatter to show every year on y-axis
        locator = MultipleLocator(1)  # Set ticks every 1 unit
        formatter = FormatStrFormatter('%d')  # Integer format
        self.ax.yaxis.set_major_locator(locator)
        self.ax.yaxis.set_major_formatter(formatter)

        # Format
        self.format(
            ax_xlabel_txt=ax_xlabel_txt,
            ax_ylabel_txt=ax_ylabel_txt,
            plot=p,
        )


def _example_heatmap_datetime():
    import diive as dv
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['VPD_f'].copy()
    # series = series.resample('3h', label='left').mean()
    series = series.loc[series.index.year == 2021]
    series.index.name = 'TIMESTAMP_START'
    series.name = None
    hm = dv.heatmapdatetime(series=series, title="test")
    hm.show()
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


def _example_heatmap_yearmonth():
    import diive as dv
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['GPP_DT_CUT_REF'].copy()
    # series = df['Tair_f'].copy()

    series = series.resample('1MS', label='left').mean()
    # series = series.resample('1MS', label='left').agg(np.ptp)
    series.index.name = 'TIMESTAMP_START'

    hm = dv.heatmapyearmonth(
        series_monthly=series,
        title="Range per month",
        cb_digits_after_comma=0,
        show_values=True,
        show_values_n_dec_places=0
        # todo display_type='week_year'
        # todo display_type='year_doy'
    )

    hm.show()
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


if __name__ == '__main__':
    _example_heatmap_datetime()
    # _example_heatmap_yearmonth()
