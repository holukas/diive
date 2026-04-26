"""
HEATMAP — Date/Time and Year/Month variants
============================================

Contains two public heatmap classes built on :class:`~diive.core.plotting.heatmap_base.HeatmapBase`:

- :class:`HeatmapDateTime` — plots a time series as a 2-D grid of *date × time-of-day*
  (or the transposed *time-of-day × date* in horizontal orientation).  The time axis
  is stored as float hours (0–24) so that ``pcolormesh`` can handle it natively.

- :class:`HeatmapYearMonth` — aggregates a time series to monthly resolution and
  plots it as a *year × month* (or *month × year*) grid.  Supports arbitrary
  aggregation functions and optional rank transformation.

Both classes are exposed as top-level convenience aliases in the ``diive``
namespace (``dv.heatmap_datetime`` and ``dv.heatmap_year_month``).

References:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html
"""
import datetime

import numpy as np
import pandas as pd
from pandas import Series

import diive as dv
from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.plotfuncs import nice_date_ticks


class HeatmapDateTime(HeatmapBase):
    """Heatmap of a time series arranged as a date × time-of-day grid.

    The series is pivoted so that every unique date forms one row and every
    unique time-of-day forms one column (or the transpose in ``'horizontal'``
    orientation).  Each cell color represents the measured value at that
    date/time combination; NaN cells are filled with ``color_bad``.

    The time-of-day axis is stored internally as **float hours** (e.g.
    ``0.5`` for 00:30, ``6.0`` for 06:00) because ``pcolormesh`` requires
    numeric coordinates.

    Top-level alias: ``dv.heatmap_datetime(series, ...)``

    Example:
        See `examples/visualization/heatmap_datetime.py` for complete examples
        including vertical/horizontal orientations.
    """

    def __init__(self,
                 series: Series,
                 **kwargs):
        """Creates the heatmap object and prepares the data grid.

        The ``heatmaptype`` parameter (used internally for cell-centering in
        :meth:`~diive.core.plotting.heatmap_base.HeatmapBase.show_vals_in_plot`)
        is set automatically to ``'datetime'`` or ``'datetime_horizontal'``
        based on ``ax_orientation`` and must not be passed in ``**kwargs``.

        Args:
            series: Pandas Series with a ``DatetimeIndex`` (or compatible
                    datetime-like index following diive's timestamp naming
                    convention).  Any temporal resolution is supported; the
                    series will be sanitized and regularised automatically.
            **kwargs: All keyword arguments accepted by
                      :class:`~diive.core.plotting.heatmap_base.HeatmapBase`,
                      e.g. ``figsize``, ``ax_orientation``, ``cmap``,
                      ``vmin``/``vmax``, ``zlabel``, ``verbose``.
        """
        _orientation = kwargs.get('ax_orientation', 'vertical')
        _heatmaptype = 'datetime' if _orientation == 'vertical' else 'datetime_horizontal'
        super().__init__(heatmaptype=_heatmaptype, **kwargs)
        self.series = series.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepares the time series data for plotting as a heatmap.

        This method performs several steps:
        1. Ensures the time series has a name, defaulting to "data" if none exists.
        2. Sets up the series index as a proper timestamp using `_setup_timestamp`.
        3. Transforms the series into a pandas DataFrame, extracting date and time components.
        4. Pivots the DataFrame to create the grid for the heatmap, with 'DATE' or 'TIME'
           as the index and the other as columns, based on `self.ax_orientation`.
        5. Calls `_set_bounds` to extend the x and y data for proper heatmap cell rendering.
        """
        self.series.name = self.series.name if self.series.name else "data"  # Time series must have a name
        self.series = self._setup_timestamp(series=self.series)

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

    @staticmethod
    def _time_to_hours(time_values) -> np.ndarray:
        """Converts an array of ``datetime.time`` objects to float hours since midnight.

        ``pcolormesh`` requires numeric coordinates; ``datetime.time`` objects are not
        natively supported by matplotlib's unit converters.  Converting to float hours
        (e.g. 00:30 → 0.5, 06:00 → 6.0) gives a clean numeric axis while preserving
        the intuitive hourly scale for tick labels.

        Args:
            time_values: Array-like of ``datetime.time`` objects.

        Returns:
            np.ndarray: Float array of hours since midnight.
        """
        return np.array([t.hour + t.minute / 60 + t.second / 3600 for t in time_values])

    def _set_bounds(self):
        """Extends the x and y data arrays for accurate heatmap plotting.

        In a ``pcolormesh`` plot ``x`` and ``y`` represent the **boundaries** of the
        cells while ``z`` contains the values *inside* those cells.  To ensure the
        last row/column of data is fully visible an extra boundary point is added
        beyond the last data point by repeating the uniform step size.

        The time axis (``datetime.time``) is converted to float hours so that
        matplotlib can handle the numeric coordinates natively.  The date axis
        (``datetime.date``) is left as-is; matplotlib's date converter handles it.

        Returns:
            tuple: Extended ``(x, y, z)`` arrays ready for ``pcolormesh``.
        """
        x = self.plotdf.columns.values
        y = self.plotdf.index.values
        z = self.plotdf.values

        if self.ax_orientation == "vertical":
            # x = TIME → convert to float hours; y = DATE (keep as datetime.date)
            x_hours = self._time_to_hours(x)
            step = (x_hours[1] - x_hours[0]) if len(x_hours) > 1 else 0.5
            x = np.append(x_hours, x_hours[-1] + step)
            last_y = y[-1] + datetime.timedelta(days=1)
            y = np.append(y, last_y)

        elif self.ax_orientation == "horizontal":
            # x = DATE (keep as datetime.date); y = TIME → convert to float hours
            last_x = x[-1] + datetime.timedelta(days=1)
            x = np.append(x, last_x)
            y_hours = self._time_to_hours(y)
            step = (y_hours[1] - y_hours[0]) if len(y_hours) > 1 else 0.5
            y = np.append(y_hours, y_hours[-1] + step)

        return x, y, z

    def _set_ticks(self):
        """Defines tick locations and labels for a 24-hour time axis.

        The tick interval adapts to the inferred data frequency so that ticks
        always fall on actual data boundaries.  Falls back to 3-hour ticks when
        the frequency cannot be determined.  Returns float hours (e.g. ``3.0``,
        ``6.0``) to match the numeric time axis produced by ``_set_bounds``.

        Returns:
            tuple: A tuple containing two lists:
                   - ``ticks_time``: Float hours at each major tick (e.g. ``[3.0, 6.0, …]``).
                   - ``ticklabels_time``: Integer hour labels for the ticks.
        """
        freq = self.series.index.freq
        if freq is not None:
            try:
                freq_minutes = int(pd.tseries.frequencies.to_offset(freq).nanos / 1e9 / 60)
            except Exception:
                freq_minutes = 30  # safe fallback
        else:
            freq_minutes = 30  # safe fallback

        # Pick the smallest interval (3h, 6h, 12h) that is a whole multiple of the data frequency
        tick_interval_hours = 3
        for candidate in [3, 6, 12]:
            if (candidate * 60) % freq_minutes == 0:
                tick_interval_hours = candidate
                break

        tick_hours = list(range(tick_interval_hours, 24, tick_interval_hours))
        ticks_time = [float(h) for h in tick_hours]  # float hours match the numeric time axis
        ticklabels_time = tick_hours
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

        if self.show_values:
            self.show_vals_in_plot()

        # Format — guard against freq being None (irregular or stripped series)
        _freq = self.series.index.freqstr if self.series.index.freq is not None else None
        self.format(
            ax_xlabel_txt=xlabel,
            ax_ylabel_txt=ylabel,
            plot=p,
            shown_freq=_freq
        )


# @ConsoleOutputDecorator(spacing=False)
class HeatmapYearMonth(HeatmapBase):
    """
    A class for plotting heatmaps of time series data aggregated by year and month.

    This class extends HeatmapBase to visualize monthly aggregated time series
    data, where one axis represents years and the other represents months.
    It supports different aggregation methods and the display of ranks instead of raw values.

    Example:
        See `examples/visualization/heatmap_datetime.py` for complete examples
        including rank transformation, multi-panel layouts, and colormap options.
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
