"""
HEATMAP
=======
"""
import copy
import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from pandas.plotting import register_matplotlib_converters

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format, format_spines, nice_date_ticks
from diive.core.times.times import TimestampSanitizer
from diive.core.times.times import insert_timestamp


# @ConsoleOutputDecorator(spacing=False)
class HeatmapDateTime:

    def __init__(self,
                 series: Series,
                 fig=None,
                 ax=None,
                 title: str = None,
                 vmin: float = None,
                 vmax: float = None,
                 cb_digits_after_comma: int = 2,
                 cb_labelsize: float = theme.AX_LABELS_FONTSIZE,
                 axlabels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labelsize: float = theme.TICKS_LABELS_FONTSIZE,
                 minyticks: int = 3,
                 maxyticks: int = 10,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'grey',
                 display_type: str = 'time_date',
                 figsize: tuple = (6, 10.7),
                 verbose: bool = False):
        """
        Plot heatmap of time series data with date on y-axis and time on x-axis

        Args:
            series: Series
            ax: Axis in which heatmap is shown. If *None*, a figure with axis will be generated.
            title: Text shown at the top of the plot.
            vmin: Minimum value shown in plot
            vmax: Maximum value shown in plot
            cb_digits_after_comma: How many digits after the comma are shown in the colorbar legend.
            cmap: Matplotlib colormap
            color_bad: Color of missing values
            display_type: Only one option at the moment: 'time_date'.
                Planned: 'month_year', 'year_doy' and 'week_year'

        """
        self.series = series.copy()
        self.verbose = verbose

        if self.verbose:
            print("Plotting heatmap  ...")
        if self.verbose:
            print("Preparing timestamp for heatmap plotting ...")

        # Sanitize timestamp
        # TimestampSanitizer output TIMESTAMP_MIDDLE.
        self.series = TimestampSanitizer(data=self.series, verbose=self.verbose).get()

        # Heatmap works best with TIMESTAMP_START, convert timestamp index if needed
        # Working with a timestamp that shows the start of the averaging period means
        # that e.g. the ticks on the x-axis for hours is shown before the corresponding
        # data between e.g. 1:00 and 1:59.
        if self.series.index.name != 'TIMESTAMP_START':
            _series_df = insert_timestamp(data=self.series, convention='start', set_as_index=True)
            self.series = _series_df[self.series.name].copy()

        # # todo setting for different series time resolutions
        # if display_type == 'time_date':
        #     self.series = self.series.resample('30min').mean()
        # elif display_type == 'month_year':
        #     self.series = self.series.resample('1MS').mean()
        # elif display_type == 'week_year':
        #     self.series = self.series.resample('1WS').mean()

        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cb_digits_after_comma = cb_digits_after_comma
        self.cb_labelsize = cb_labelsize
        self.color_bad = color_bad
        self.display_type = display_type
        self.figsize = figsize
        self.ax = ax
        self.axlabels_fontsize = axlabels_fontsize
        self.fig = fig
        self.ticks_labelsize = ticks_labelsize
        self.minyticks = minyticks
        self.maxyticks = maxyticks

        # Create axis if none is given
        if not ax:
            self.fig, self.ax = self._create_ax()

        # Create dataframe with values as z variable for colors
        self.plot_df = pd.DataFrame(index=self.series.index,
                                    columns=['z'],
                                    data=self.series.values)

        # Transform data for plotting
        self.x, self.y, self.z = self._transform_data()

    def get_ax(self):
        """Return axis in which plot was generated"""
        return self.ax

    def show(self):
        """Generate plot and show figure"""
        self.plot()
        plt.tight_layout()
        self.fig.show()

    def _create_ax(self):
        """Create figure and axis"""
        # Figure setup
        fig = plt.figure(facecolor='white', figsize=self.figsize)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        return fig, ax

    def plot(self):
        """Plot heatmap"""
        cmap, z = self._set_cmap(cmap=self.cmap, color_bad=self.color_bad, z=self.z)
        # Run._remove_cbar(ax=ax)
        p = self.ax.pcolormesh(self.x, self.y, self.z,
                               linewidths=1, cmap=cmap,
                               vmin=self.vmin, vmax=self.vmax,
                               shading='flat', zorder=99)

        # Title
        title = self.title if self.title else f"{self.series.name} in {self.series.index.freqstr} time resolution"
        self.ax.set_title(title, color='black')
        # self.ax.set_title(self.title, color='black', size=theme.FONTSIZE_HEADER_AXIS)

        # Colorbar
        cb = plt.colorbar(p, ax=self.ax, format=f"%.{int(self.cb_digits_after_comma)}f")
        cb.ax.tick_params(labelsize=self.cb_labelsize)
        # cbytick_obj = plt.getp(cb.axes_dict, 'yticklabels')  # Set y tick label color
        # plt.setp(cbytick_obj, color='black', fontsize=FONTSIZE_HEADER_AXIS)

        # Ticks
        ax_xlabel_txt = ""
        ax_ylabel_txt = ""
        if self.display_type == 'time_date':
            ax_xlabel_txt = 'Time (hours)'
            ax_ylabel_txt = 'Date'

            # found_hours = []
            # for t in self.x:
            #     found_hours.append(t.hour)
            # uniq_found_hours = list(set(found_hours))
            # xticks = [f"{str(x).zfill(2)}:00" for x in uniq_found_hours]
            # xticklabels = [f"{x}" for x in uniq_found_hours]
            # self.ax.set_xticks(xticks)
            # self.ax.set_xticklabels(xticklabels)
            self.ax.set_xticks(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
            self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])

            # # matplotlib's HourLocator did not work
            # nice_date_ticks(ax=self.ax, minticks=1, maxticks=24, which='x', locator='hour')

            # For the y-axis AutoDateLocator worked
            nice_date_ticks(ax=self.ax, minticks=self.minyticks, maxticks=self.maxyticks, which='y')

        # TODO elif self.display_type == 'month_year':
        #     ticks = list(range(1, 13, 1))
        #     # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolormesh_levels.html#sphx-glr-gallery-images-contours-and-fields-pcolormesh-levels-py
        #     self.ax.set_xticks(ticks)
        #     self.ax.set_xticklabels([str(t) for t in ticks])
        #     ax_xlabel_txt = 'Month'
        #     ax_ylabel_txt = 'Year'

        # Format
        default_format(ax=self.ax, ax_xlabel_txt=ax_xlabel_txt, ax_ylabel_txt=ax_ylabel_txt,
                       ticks_direction='out', ticks_length=8, ticks_width=2,
                       ax_labels_fontsize=self.axlabels_fontsize,
                       ticks_labels_fontsize=self.ticks_labelsize)
        format_spines(ax=self.ax, color='black', lw=2)

    def _transform_data(self):
        """Transform data for plotting"""

        # Setup xy axes
        xaxis_vals, yaxis_vals = self._set_xy_axes_type()
        self.plot_df['y_vals'] = yaxis_vals
        self.plot_df['x_vals'] = xaxis_vals
        self.plot_df.reset_index(drop=True, inplace=True)

        # if display_type == 'month_year':
        #     plot_df = plot_df.resample('1D').mean()
        #     plot_df['y_vals'] = plot_df.index.dayofyear
        #     plot_df['x_vals'] = plot_df.index.year

        # Put needed data in new df_pivot, then pivot to bring it in needed shape
        plot_df_pivot = self.plot_df.pivot(index='y_vals', columns='x_vals', values='z')  ## new in pandas 23.4
        # plot_df_pivot = plot_df_pivot.append(plot_df_pivot.iloc[-1])
        x = plot_df_pivot.columns.values
        y = plot_df_pivot.index.values
        z = plot_df_pivot.values

        # x and y are bounds, so z should be the value *inside* those bounds.
        # Therefore, extend x and y by one value (last_x, last_y).

        if self.display_type == 'time_date':

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

        elif self.display_type == 'month_year':
            x = np.append(x, x[-1] + 1)
            # x = np.append(x[0], x)
            y = np.append(y, y[-1] + 1)
            # y = np.append(y[0], y)

        # if self.display_type == 'month_year':
        #     # x = [str(xx) for xx in x]
        #     # x needs to be extended by 1 value, otherwise the plot cuts off the last year
        #     x = np.append(x, 'end')

        return x, y, z

    def _set_xy_axes_type(self):
        if self.display_type == 'time_date':
            xaxis_vals = self.plot_df.index.time
            yaxis_vals = self.plot_df.index.date
            register_matplotlib_converters()  # Needed for time plotting
        elif self.display_type == 'month_year':
            xaxis_vals = self.plot_df.index.month
            yaxis_vals = self.plot_df.index.year
        else:
            xaxis_vals = self.plot_df.index.time  # Fallback
            yaxis_vals = self.plot_df.index.date
        return xaxis_vals, yaxis_vals

    def _set_cmap(self, cmap, color_bad, z):
        """Set colormap and color of missing values"""
        # Colormap
        cmap = copy.copy(plt.get_cmap(cmap))  # Needed for now, https://github.com/matplotlib/matplotlib/issues/17634
        cmap.set_bad(color=color_bad, alpha=1.)  # Set missing data to specific color
        z = np.ma.masked_invalid(z)  # Mask NaN as missing
        return cmap, z


def example():
    from diive.configs.exampledata import load_exampledata_parquet
    df = load_exampledata_parquet()
    series = df['Tair_f'].copy()

    # series = series.resample('2MS').mean()
    series = series.resample('4h', label='left').mean()
    series.index.name = 'TIMESTAMP_START'
    # series = series[series.index.month > 10].copy()
    # series = series[series.index.year > 2018].copy()

    hm = HeatmapDateTime(
        series=series,
        title="test",
        # vmin=-0,
        # vmax=1000,
        # todo display_type='month_year'
        display_type='time_date'
        # todo display_type='week_year'
        # todo display_type='year_doy'
    )

    hm.show()
    print(hm.get_ax())


if __name__ == '__main__':
    example()
