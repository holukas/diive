"""
HEATMAP
=======

Kudos:
    - https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/
    pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py

"""
import datetime

import numpy as np
import pandas as pd
from pandas import Series
from pandas.plotting import register_matplotlib_converters

import diive as dv
from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.plotfuncs import nice_date_ticks


class HeatmapDateTime(HeatmapBase):
    """
    A class for plotting heatmaps of time series data with date on the y-axis
    and time on the x-axis, or vice-versa, depending on orientation.

    This class extends HeatmapBase to specifically handle time series data,
    transforming it into a format suitable for a 2D heatmap where one
    dimension represents dates and the other represents times of day.
    """

    def __init__(self,
                 series: Series,
                 **kwargs):
        """Initializes the HeatmapDateTime object.

        Args:
            series: A pandas Series with a datetime-like index, representing
                    the time series data to be plotted.
            **kwargs: Arbitrary keyword arguments passed to the HeatmapBase
                      constructor. These can include plotting parameters like
                      `ax_orientation`, `minticks`, `maxticks`, etc.
        """
        super().__init__(**kwargs)
        self.series = series.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepares the time series data for plotting as a heatmap.

        This method performs several steps:
        1. Ensures the time series has a name, defaulting to "data" if none exists.
        2. Sets up the series index as a proper timestamp using `_setup_timestamp`.
        3. Registers Matplotlib converters for datetime objects to ensure correct plotting.
        4. Transforms the series into a pandas DataFrame, extracting date and time components.
        5. Pivots the DataFrame to create the grid for the heatmap, with 'DATE' or 'TIME'
           as the index and the other as columns, based on `self.ax_orientation`.
        6. Calls `_set_bounds` to extend the x and y data for proper heatmap cell rendering.
        """
        self.series.name = self.series.name if self.series.name else "data"  # Time series must have a name
        self.series = self._setup_timestamp(series=self.series)

        # Needed for time plotting
        register_matplotlib_converters()

        # Data for plotting
        self.plotdf = pd.DataFrame(self.series)
        self.plotdf['DATE'] = self.plotdf.index.date
        self.plotdf['TIME'] = self.plotdf.index.time
        self.plotdf = self.plotdf.reset_index(drop=True, inplace=False)

        if self.ax_orientation == "vertical":
            self.plotdf = self.plotdf.pivot(index='DATE', columns='TIME', values=self.series.name)
        elif self.ax_orientation == "horizontal":
            self.plotdf = self.plotdf.pivot(index='TIME', columns='DATE', values=self.series.name)

        # Extend
        self.x, self.y, self.z = self._set_bounds()

    def _set_bounds(self):
        """Extends the x and y data arrays for accurate heatmap plotting.

        In a `pcolormesh` plot, `x` and `y` represent the boundaries of the cells,
        while `z` contains the values *inside* those cells. To ensure that the
        last row/column of data is fully visible and correctly bounded, this
        method extends the `x` and `y` arrays by adding one extra boundary point
        beyond the last data point.

        The extension is either by one day (for date axes) or by setting the time
        to 23:59 (for time axes), depending on the `ax_orientation`.

        Returns:
            tuple: A tuple containing the extended x, y, and z data arrays,
                   ready for `pcolormesh` plotting.
        """
        x = self.plotdf.columns.values
        y = self.plotdf.index.values
        z = self.plotdf.values

        # Add last entries for x and y
        last_x = x[-1]  # Last record for x
        last_y = y[-1]  # Last record for y

        if self.ax_orientation == "vertical":
            # x = TIME, y = DATE
            last_x = last_x.replace(hour=23, minute=59)  # x-axis shows hours 0, 1, 2 ... 23
            last_y = last_y + datetime.timedelta(days=1)  # y-axis shows dates

        elif self.ax_orientation == "horizontal":
            # x = DATE, y = TIME
            last_x = last_x + datetime.timedelta(days=1)  # x-axis shows dates
            last_y = last_y.replace(hour=23, minute=59)  # y-axis shows hours 0, 1, 2 ... 23

        x = np.append(x, last_x)
        y = np.append(y, last_y)

        return x, y, z

    @staticmethod
    def _set_ticks():
        """Defines the standard tick locations and labels for a 24-hour time axis.

        This static method provides a consistent set of major tick marks
        (every 3 hours) and their corresponding integer labels for time axes
        in the heatmap.

        Returns:
            tuple: A tuple containing two lists:
                   - `ticks_time`: A list of strings representing time points
                                   (e.g., '3:00', '6:00') that serve as tick locations.
                   - `ticklabels_time`: A list of integers representing the
                                        hour labels (e.g., 3, 6) for the ticks.
        """
        ticks_time = ['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00']
        ticklabels_time = [3, 6, 9, 12, 15, 18, 21]
        return ticks_time, ticklabels_time

    def plot(self):
        """Plots the heatmap from the prepared time series data.

        This method orchestrates the plotting process:
        1. Calls `plot_pcolormesh` (inherited from HeatmapBase) to render the
           core heatmap using the `x`, `y`, and `z` data.
        2. Retrieves standard time ticks and labels using `_set_ticks`.
        3. Configures the x and y axis labels and tick formats based on
           `self.ax_orientation`.
           - For time axes, it sets specific tick locations and uses the predefined
             time labels.
           - For date axes, it leverages `nice_date_ticks` for intelligent date formatting.
        4. Calls the `format` method (inherited from HeatmapBase) to apply
           general plot formatting, including axis labels and plot title.

        Raises:
            NotImplementedError: If an unsupported `ax_orientation` value is encountered.
        """
        p = self.plot_pcolormesh()
        ticks_time, ticklabels_time = self._set_ticks()

        if self.ax_orientation == "vertical":
            xlabel = 'Time (hours)'
            ylabel = 'Date'
            self.ax.set_xticks(ticks_time)
            self.ax.set_xticklabels(ticklabels_time)
            # # matplotlib's HourLocator did not work
            # nice_date_ticks(ax=self.ax, minticks=1, maxticks=24, which='x', locator='hour')
            # For the y-axis (DATE) AutoDateLocator worked
            nice_date_ticks(ax=self.ax, minticks=self.minticks, maxticks=self.maxticks, which='y')
        elif self.ax_orientation == "horizontal":
            xlabel = 'Date'
            ylabel = 'Time (hours)'
            self.ax.set_yticks(ticks_time)
            self.ax.set_yticklabels(ticklabels_time)
            nice_date_ticks(ax=self.ax, minticks=self.minticks, maxticks=self.maxticks, which='x')
        else:
            raise NotImplementedError

        # Format
        self.format(
            ax_xlabel_txt=xlabel,
            ax_ylabel_txt=ylabel,
            plot=p,
            shown_freq=self.series.index.freqstr
        )


# @ConsoleOutputDecorator(spacing=False)
class HeatmapYearMonth(HeatmapBase):
    """
    A class for plotting heatmaps of time series data aggregated by year and month.

    This class extends HeatmapBase to visualize monthly aggregated time series
    data, where one axis represents years and the other represents months.
    It supports different aggregation methods and the display of ranks instead of raw values.
    """

    def __init__(self,
                 series: Series,
                 agg: str = 'mean',
                 ranks: bool = False,
                 cmap: str = None,
                 **kwargs):
        """Initializes the HeatmapYearMonth object.

        Args:
            series: A pandas Series with a datetime-like index, representing
                    the time series data to be aggregated and plotted.
            agg: The aggregation method to apply to the monthly data (e.g., 'mean',
                 'sum', 'median', 'min', 'max'). Defaults to 'mean'.
            ranks: If True, the data will be converted to ranks before plotting.
                   Defaults to False.
            cmap: The colormap to use for the heatmap. If None, 'RdYlBu' is used
                  for ranks and 'RdYlBu_r' for raw values.
            **kwargs: Arbitrary keyword arguments passed to the HeatmapBase
                      constructor. These can include plotting parameters like
                      `ax_orientation`, `show_values`, etc.
        """
        super().__init__(heatmaptype='yearmonth', **kwargs)
        self.series = series.copy()
        self.agg = agg
        self.ranks = ranks

        if not cmap:
            self.cmap = 'RdYlBu' if ranks else 'RdYlBu_r'
        else:
            self.cmap = cmap

        self._prepare_data()

    def _prepare_data(self):
        """Prepares the time series data for plotting as a year-month heatmap.

        This method performs the following steps:
        1. Ensures the time series has a name, defaulting to "data" if none exists.
        2. Sets up the series index as a proper timestamp using `_setup_timestamp`.
        3. Aggregates the time series data into a monthly matrix using
           `dv.resample_to_monthly_agg_matrix`, applying the specified
           aggregation method (`self.agg`) and optionally converting to ranks.
        4. Transposes the resulting DataFrame if `self.ax_orientation` is "horizontal"
           to ensure months are the index and years are columns.
        5. Extracts the x (columns), y (index), and z (values) data from the
           prepared DataFrame.
        6. Calls `_set_bounds` to extend the x and y data for proper heatmap
           cell rendering, as `pcolormesh` uses bounds.
        """
        self.series.name = self.series.name if self.series.name else "data"  # Time series must have a name
        self.series = self._setup_timestamp(series=self.series)

        # Bring data into shape
        self.plotdf = dv.resample_to_monthly_agg_matrix(series=self.series, agg=self.agg, ranks=self.ranks)

        # Transpose in case of horizontal, to have months as index, years as columns
        if self.ax_orientation == "horizontal":
            self.plotdf = self.plotdf.transpose()

        x = self.plotdf.columns.values
        y = self.plotdf.index.values
        z = self.plotdf.values
        self.x, self.y, self.z = self._set_bounds(x=x, y=y, z=z)

    @staticmethod
    def _set_bounds(x, y, z):
        """Extends the x and y data arrays for accurate heatmap plotting.

        In a `pcolormesh` plot, `x` and `y` represent the boundaries of the cells,
        while `z` contains the values *inside* those cells. To ensure that the
        last row/column of data is fully visible and correctly bounded, this
        method extends the `x` and `y` arrays by adding one extra integer unit
        beyond the last data point. This is appropriate for discrete month/year axes.

        Args:
            x (np.ndarray): The array of column values (e.g., months or years).
            y (np.ndarray): The array of index values (e.g., years or months).
            z (np.ndarray): The 2D array of heatmap values.

        Returns:
            tuple: A tuple containing the extended x, y, and z data arrays,
                   ready for `pcolormesh` plotting.
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
        """Plots the heatmap of year-month aggregated time series data.

        This method orchestrates the plotting process:
        1. Calls `plot_pcolormesh` (inherited from HeatmapBase) to render the
           core heatmap using the `x`, `y`, and `z` data.
        2. If `self.show_values` is True, it calls `show_vals_in_plot` to
           overlay the numerical values on the heatmap cells.
        3. Sets the tick positions and labels for both x and y axes. Since the
           axes represent months or years (integers), ticks are centered within
           the cells and labeled with their integer values.
        4. Optionally hides every second x-axis tick label if `self.show_less_xticklabels` is True,
           to prevent overcrowding.
        5. Determines and sets the appropriate x and y axis labels based on
           `self.ax_orientation`.
        6. Calls the `format` method (inherited from HeatmapBase) to apply
           general plot formatting, including axis labels and a title reflecting
           the aggregation method and frequency.
        """
        p = self.plot_pcolormesh()

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

        # Set xylabels
        xlabel = 'Month' if self.ax_orientation == "vertical" else 'Year'
        ylabel = 'Year' if self.ax_orientation == "vertical" else 'Month'

        # Format
        self.format(
            ax_xlabel_txt=xlabel,
            ax_ylabel_txt=ylabel,
            plot=p,
            shown_freq=f'{self.agg}, MS'
        )


def _example_heatmap_datetime():
    import diive as dv
    df = dv.load_exampledata_parquet()

    # from diive.core.io.filereader import ReadFileType
    # filepath = r"F:\Sync\luhk_work\40 - DATA\DATASETS\2025_FORESTS\1-downloads\ICOSETC_CH-Dav_ARCHIVE_L2\ICOSETC_CH-Dav_FLUXNET_HH_L2.csv"
    # loaddatafile = ReadFileType(filetype='FLUXNET-FULLSET-HH-CSV-30MIN',
    #                             filepath=filepath,
    #                             data_nrows=None)
    # df, metadata_df = loaddatafile.get_filedata()

    var = 'NEE_CUT_REF_f'
    series = df[var].copy()
    # series = series.resample('3h', label='left').mean()  # For testing
    # series.index.name = "TIMESTAMP_START"  # For testing
    # locs = (series.index.date >= datetime.date(2022, 6, 1)) & (series.index.date <= datetime.date(2022, 6, 5))
    locs = series.index.year >= 2020
    series = series.loc[locs]
    series.iloc[100:120] = np.nan  # For testing
    series = series.dropna()  # For testing

    # hm = dv.heatmapdatetime(series=series, title=None, vmin=-10, vmax=10, ax_orientation="vertical")
    hm = dv.heatmapdatetime(series=series, title=None, vmin=-10, vmax=10, ax_orientation="horizontal")
    hm.show()
    # hm.export_borderless_heatmap(outpath=r"F:\TMP\heightmap_blender")
    # print(hm.get_ax())
    # print(hm.get_plot_data())


def _example_heatmap_yearmonth():
    df = dv.load_exampledata_parquet()
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
    df = dv.load_exampledata_parquet()
    fig = plt.figure(facecolor='white', figsize=(16, 9), layout="constrained", dpi=300)
    gs = gridspec.GridSpec(2, 1, figure=fig)  # rows, cols
    gs.update(wspace=0.5, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
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
    cmaps = list_of_colormaps()
    df = dv.load_exampledata_parquet()
    series = df['Tair_f'].copy()
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
    _example_heatmap_datetime()
    # _example_heatmap_yearmonth()
    # _example_multiple_heatmaps_yearmonth_horizontal()
    # _example_colormaps()

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
