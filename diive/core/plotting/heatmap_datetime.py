"""
HEATMAP
=======
"""
import copy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from pandas.plotting import register_matplotlib_converters

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import nice_date_ticks, default_format, format_spines
from diive.core.times.times import TimestampSanitizer
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator(spacing=False)
class HeatmapDateTime:

    def __init__(self,
                 series: Series,
                 ax=None,
                 title: str = None,
                 vmin: float = None,
                 vmax: float = None,
                 cb_digits_after_comma: int = 2,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'white',
                 display_type: str = 'Time & Date'):
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
            display_type: Currently only option is 'Time & Date'

            Example notebook:
            diive/Plotting/Heatmap.ipynb
            in https://gitlab.ethz.ch/gl-notebooks/general-notebooks

        """
        print(f"Plotting heatmap  ...")

        self.series = series.copy()

        print("Preparing timestamp for heatmap plotting ...")
        self.series = TimestampSanitizer(data=self.series).get()

        self.title = title
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.cb_digits_after_comma = cb_digits_after_comma
        self.color_bad = color_bad
        self.display_type = display_type

        # Create axis if none is given
        if not ax:
            self.fig, self.ax = self._create_ax()

        # Create dataframe with values as z variable for colors
        self.plot_df = pd.DataFrame(index=self.series.index,
                                    columns=['z'],
                                    data=self.series.values)

        # # # This is not applicable, because the dataframe is pivoted:
        # # Add frequency property
        # # Creating a new dataframe with the series.index above
        # # loses the .freq property of the series, i.e., the dataframe
        # # does not have a defined frequency after this step and
        # # needs to be added manually.
        # self.plot_df = self.plot_df.asfreq(self.series.index.freq)

        # Transform data for plotting
        self.x, self.y, self.z = \
            self._transform_data()

    def get_ax(self):
        """Return axis in which plot was generated"""
        return self.ax

    def show(self):
        """Generate plot and show figure"""
        self.plot()
        self.fig.show()

    def _create_ax(self):
        """Create figure and axis"""
        # Figure setup
        fig = plt.figure(facecolor='white', figsize=(9, 16))
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
        cb.ax.tick_params(labelsize=theme.AXLABELS_FONTSIZE)
        # cbytick_obj = plt.getp(cb.axes_dict, 'yticklabels')  # Set y tick label color
        # plt.setp(cbytick_obj, color='black', fontsize=FONTSIZE_HEADER_AXIS)

        # Ticks
        self.ax.set_xticks(['3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00'])
        self.ax.set_xticklabels([3, 6, 9, 12, 15, 18, 21])
        nice_date_ticks(ax=self.ax, minticks=6, maxticks=8, which='y')

        default_format(ax=self.ax, txt_xlabel='Time (hours)', txt_ylabel='Date',
                       ticks_direction='out', ticks_length=8, ticks_width=2)
        format_spines(ax=self.ax, color='black', lw=2)

    def _transform_data(self):
        """Transform data for plotting"""

        # Setup xy axes
        xaxis_vals, yaxis_vals = self._set_xy_axes_type()
        self.plot_df['y_vals'] = yaxis_vals
        self.plot_df['x_vals'] = xaxis_vals
        self.plot_df.reset_index(drop=True, inplace=True)

        # if display_type == 'Year & Day of Year':
        #     plot_df = plot_df.resample('1D').mean()
        #     plot_df['y_vals'] = plot_df.index.dayofyear
        #     plot_df['x_vals'] = plot_df.index.year

        # Put needed data in new df_pivot, then pivot to bring it in needed shape
        plot_df_pivot = self.plot_df.pivot(index='y_vals', columns='x_vals', values='z')  ## new in pandas 23.4
        x = plot_df_pivot.columns.values
        y = plot_df_pivot.index.values
        z = plot_df_pivot.values

        if self.display_type == 'Time & Date':
            x = np.append(x, x[-1])
            y = np.append(y, y[-1])

        # if display_type == 'Year & Day of Year':
        #     x = np.append(x, 'end')
        #     # y = np.append(y, y[-1])

        # if self.display_type == 'Year & Day of Year':
        #     # x = [str(xx) for xx in x]
        #     # x needs to be extended by 1 value, otherwise the plot cuts off the last year
        #     x = np.append(x, 'end')

        return x, y, z

    def _set_xy_axes_type(self):
        if self.display_type == 'Time & Date':
            xaxis_vals = self.plot_df.index.time
            yaxis_vals = self.plot_df.index.date
            register_matplotlib_converters()  # Needed for time plotting
        elif self.display_type == 'Year & Day of Year':
            xaxis_vals = self.plot_df.index.year
            yaxis_vals = self.plot_df.index.date
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
    # Example with random data
    import pandas as pd
    import numpy as np
    date_rng = pd.date_range(start='2018-01-01 03:30', end='2018-01-05 00:00:00', freq='30T')
    df = pd.DataFrame(date_rng, columns=['TIMESTAMP_END'])
    df['DATA'] = np.random.randint(0, 10, size=(len(date_rng)))
    df = df.set_index('TIMESTAMP_END')
    df = df.asfreq('30T')
    df.head()
    series = df['DATA'].copy()
    hm = HeatmapDateTime(series=series, title="Example heatmap")
    hm.show()
    print(hm.get_ax())

    # # Example with download from database
    # from dbc_influxdb import dbcInflux
    #
    # # Download settings
    # BUCKET = 'ch-dav_raw'
    # MEASUREMENTS = ['TA']
    # FIELDS = ['TA_PRF_T1_35_1']
    # START = '2022-01-01 00:00:10'
    # STOP = '2022-01-15 00:00:10'
    # TIMEZONE_OFFSET_TO_UTC_HOURS = 1  # We need returned timestamps in CET (winter time), which is UTC + 1 hour
    # DATA_VERSION = 'raw'
    # DIRCONF = r'L:\Dropbox\luhk_work\20 - CODING\22 - POET\configs'
    # dbc = dbcInflux(dirconf=DIRCONF)
    #
    # # Download data
    # data_simple, data_detailed, assigned_measurements = \
    #     dbc.download(
    #         bucket=BUCKET,
    #         measurements=MEASUREMENTS,
    #         fields=FIELDS,
    #         start=START,
    #         stop=STOP,
    #         timezone_offset_to_utc_hours=TIMEZONE_OFFSET_TO_UTC_HOURS,
    #         data_version=DATA_VERSION
    #     )
    #
    # HeatmapDateTime(series=data_simple['TA_PRF_T1_35_1']).show()


if __name__ == '__main__':
    example()
