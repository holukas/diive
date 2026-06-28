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

from diive.core.times.resampling import resample_to_monthly_agg_matrix
from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.plotfuncs import nice_date_ticks
from diive.core.plotting.styles import LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle


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

    See Also:
        examples/visualization/heatmap_datetime.py — DateTime heatmap variations (vertical/horizontal)
        examples/gap_filling/interpolate.py — Heatmap in gap-filling context
        examples/gap_filling/randomforest_ts.py — Heatmap for model output visualization
    """

    def __init__(self,
                 series: Series,
                 ax_orientation: str = "vertical",
                 verbose: bool = False):
        """Prepare time series data for heatmap plotting (Phase 1 of two-phase design).

        Args:
            series: Pandas Series with a ``DatetimeIndex`` (or compatible
                    datetime-like index following diive's timestamp naming
                    convention).  Any temporal resolution is supported; the
                    series will be sanitized and regularised automatically.
            ax_orientation: Layout of the axes.
                ``'vertical'`` (default) — dates on y-axis, hours on x-axis.
                ``'horizontal'`` — dates on x-axis, hours on y-axis.
            verbose: Print progress and diagnostic messages.  Defaults to *False*.

        See Also:
            plot : Render the heatmap with matplotlib styling options
        """
        _heatmaptype = 'datetime' if ax_orientation == 'vertical' else 'datetime_horizontal'
        super().__init__(heatmaptype=_heatmaptype, verbose=verbose)

        self.series = series.copy()
        self.ax_orientation = ax_orientation
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
        x = self.plotdf.columns.to_numpy()
        y = self.plotdf.index.to_numpy()
        z = self.plotdf.to_numpy()

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

    def plot(self,
             ax=None,
             fig=None,
             figsize: tuple = None,
             figdpi: int = 72,
             ax_orientation: str = None,
             format_style: FormatStyle = None,
             vmin: float = None,
             vmax: float = None,
             cmap: str = 'RdYlBu_r',
             zlabel: str = None,
             cb_digits_after_comma: int | str = 2,
             cb_labelsize: float = None,
             cb_extend: str = 'neither',
             minticks: int = 3,
             maxticks: int = 10,
             color_bad: str = 'grey',
             show_colormap: bool = True,
             show_less_xticklabels: bool = False,
             show_values: bool = False,
             show_values_fontsize: float = None,
             show_values_n_dec_places: int = 0):
        """Render HeatmapDateTime with matplotlib styling (Phase 2 of two-phase design).

        All styling and presentation parameters go here. Can be called multiple times
        on the same HeatmapDateTime object to plot on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure and displays it
            fig: Existing matplotlib Figure. If None and ax is None, creates new figure
            figsize: Figure size as (width, height) in inches. Only used when ax is None
            figdpi: Figure DPI. Only used when ax is None. Defaults to 72
            ax_orientation: Layout of axes. If None, uses value from __init__()
                ``'vertical'`` — dates on y-axis, hours on x-axis (default)
                ``'horizontal'`` — dates on x-axis, hours on y-axis
            format_style: Shared chrome (title/labels/fonts/ticks/spines/grid) via
                :class:`~diive.core.plotting.styles.format.FormatStyle`. None = house
                style (grid off). The colorbar stays ``cb_*``/``zlabel``-controlled.
            vmin: Minimum color value (auto from data if None)
            vmax: Maximum color value (auto from data if None)
            cmap: Colormap name (default: 'RdYlBu_r')
            zlabel: Colorbar label (e.g., '°C', 'µmol m⁻²s⁻¹')
            cb_digits_after_comma: Decimal places on colorbar labels (default: 2)
            cb_labelsize: Font size for colorbar tick labels
            cb_extend: Colorbar extension arrows ('neither', 'both', 'min', 'max')
            minticks: Minimum major ticks on date axis (default: 3)
            maxticks: Maximum major ticks on date axis (default: 10)
            color_bad: Color for NaN cells (default: 'grey')
            show_colormap: Whether to show colorbar (default: True)
            show_less_xticklabels: Hide every second x-tick label (default: False)
            show_values: Overlay numeric values on cells (default: False)
            show_values_fontsize: Font size for value overlay text
            show_values_n_dec_places: Decimal places for value overlay (default: 0)

        Returns:
            None (displays plot if ax=None, otherwise renders on provided axes)
        """
        # Use provided ax_orientation or fall back to __init__ value
        if ax_orientation is None:
            ax_orientation = self.ax_orientation

        # Use theme defaults if not provided
        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if show_values_fontsize is None:
            show_values_fontsize = theme.AX_LABELS_FONTSIZE

        # Call parent plot() to create figure/axes and apply styling
        super().plot(
            ax=ax,
            fig=fig,
            figsize=figsize,
            figdpi=figdpi,
            ax_orientation=ax_orientation,
            format_style=format_style,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            zlabel=zlabel,
            cb_digits_after_comma=cb_digits_after_comma,
            cb_labelsize=cb_labelsize,
            cb_extend=cb_extend,
            minticks=minticks,
            maxticks=maxticks,
            color_bad=color_bad,
            show_colormap=show_colormap,
            show_less_xticklabels=show_less_xticklabels,
            show_values=show_values,
            show_values_fontsize=show_values_fontsize,
            show_values_n_dec_places=show_values_n_dec_places
        )

        # Domain-specific rendering (pcolormesh + formatting)
        p = self.plot_pcolormesh()
        ticks_time, ticklabels_time = self._set_ticks()

        if self.ax_orientation == "vertical":
            xlabel = 'Time (hours)'
            ylabel = 'Date'
            self.ax.set_xticks(ticks_time)
            self.ax.set_xticklabels(ticklabels_time)
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

        if self.showplot:
            self.fig.patch.set_facecolor('white')
            self.fig.show()


# @ConsoleOutputDecorator(spacing=False)
class HeatmapYearMonth(HeatmapBase):
    """
    A class for plotting heatmaps of time series data aggregated by year and month.

    This class extends HeatmapBase to visualize monthly aggregated time series
    data, where one axis represents years and the other represents months.
    It supports different aggregation methods and the display of ranks instead of raw values.

    See Also:
        examples/visualization/heatmap_datetime.py — Year/Month heatmap variations (rank, multi-panel, colormaps)
    """

    def __init__(self,
                 series: Series,
                 agg: str = 'mean',
                 ranks: bool = False,
                 ax_orientation: str = "vertical",
                 verbose: bool = False):
        """Prepare year-month aggregated data for heatmap plotting (Phase 1 of two-phase design).

        Args:
            series: Pandas Series with a datetime-like index representing the time series data.
            agg: Aggregation method ('mean', 'sum', 'median', 'min', 'max'). Defaults to 'mean'.
            ranks: Convert data to ranks before plotting. Defaults to False.
            ax_orientation: Layout of axes ('vertical' or 'horizontal'). Defaults to 'vertical'.
            verbose: Print progress and diagnostic messages.  Defaults to *False*.

        See Also:
            plot : Render the heatmap with matplotlib styling options
        """
        super().__init__(heatmaptype='yearmonth', verbose=verbose)
        self.series = series.copy()
        self.agg = agg
        self.ranks = ranks
        self.ax_orientation = ax_orientation

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
        self.plotdf = resample_to_monthly_agg_matrix(series=self.series, agg=self.agg, ranks=self.ranks)

        # Transpose in case of horizontal, to have months as index, years as columns
        if self.ax_orientation == "horizontal":
            self.plotdf = self.plotdf.transpose()

        x = self.plotdf.columns.to_numpy()
        y = self.plotdf.index.to_numpy()
        z = self.plotdf.to_numpy()
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

    def plot(self,
             ax=None,
             fig=None,
             figsize: tuple = None,
             figdpi: int = 72,
             ax_orientation: str = None,
             format_style: FormatStyle = None,
             vmin: float = None,
             vmax: float = None,
             cmap: str = None,
             zlabel: str = None,
             cb_digits_after_comma: int | str = 2,
             cb_labelsize: float = None,
             cb_extend: str = 'neither',
             minticks: int = 3,
             maxticks: int = 10,
             color_bad: str = 'grey',
             show_colormap: bool = True,
             show_less_xticklabels: bool = False,
             show_values: bool = False,
             show_values_fontsize: float = None,
             show_values_n_dec_places: int = 0):
        """Render HeatmapYearMonth with matplotlib styling (Phase 2 of two-phase design).

        All styling and presentation parameters go here. Can be called multiple times
        on the same HeatmapYearMonth object to plot on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure and displays it
            fig: Existing matplotlib Figure. If None and ax is None, creates new figure
            figsize: Figure size as (width, height) in inches. Only used when ax is None
            figdpi: Figure DPI. Only used when ax is None. Defaults to 72
            ax_orientation: Layout of axes. If None, uses value from __init__()
            format_style: Shared chrome (title/labels/fonts/ticks/spines/grid) via
                :class:`~diive.core.plotting.styles.format.FormatStyle`. None = house
                style (grid off). The colorbar stays ``cb_*``/``zlabel``-controlled.
            vmin: Minimum color value (auto from data if None)
            vmax: Maximum color value (auto from data if None)
            cmap: Colormap name (auto-selected if None: 'RdYlBu' for ranks, 'RdYlBu_r' for raw)
            zlabel: Colorbar label
            cb_digits_after_comma: Decimal places on colorbar labels (default: 2)
            cb_labelsize: Font size for colorbar tick labels
            cb_extend: Colorbar extension arrows ('neither', 'both', 'min', 'max')
            minticks: Minimum major ticks (default: 3)
            maxticks: Maximum major ticks (default: 10)
            color_bad: Color for NaN cells (default: 'grey')
            show_colormap: Whether to show colorbar (default: True)
            show_less_xticklabels: Hide every second x-tick label (default: False)
            show_values: Overlay numeric values on cells (default: False)
            show_values_fontsize: Font size for value overlay text
            show_values_n_dec_places: Decimal places for value overlay (default: 0)

        Returns:
            None (displays plot if ax=None, otherwise renders on provided axes)
        """
        # Use provided ax_orientation or fall back to __init__ value
        if ax_orientation is None:
            ax_orientation = self.ax_orientation

        # Auto-select colormap based on ranks if not provided
        if cmap is None:
            cmap = 'RdYlBu' if self.ranks else 'RdYlBu_r'

        # Use theme defaults if not provided
        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if show_values_fontsize is None:
            show_values_fontsize = theme.AX_LABELS_FONTSIZE

        # Call parent plot() to create figure/axes and apply styling
        super().plot(
            ax=ax,
            fig=fig,
            figsize=figsize,
            figdpi=figdpi,
            ax_orientation=ax_orientation,
            format_style=format_style,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            zlabel=zlabel,
            cb_digits_after_comma=cb_digits_after_comma,
            cb_labelsize=cb_labelsize,
            cb_extend=cb_extend,
            minticks=minticks,
            maxticks=maxticks,
            color_bad=color_bad,
            show_colormap=show_colormap,
            show_less_xticklabels=show_less_xticklabels,
            show_values=show_values,
            show_values_fontsize=show_values_fontsize,
            show_values_n_dec_places=show_values_n_dec_places
        )

        # Domain-specific rendering (pcolormesh + formatting)
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

        # Hide every second x-label if requested
        if self.show_less_xticklabels:
            labels = self.ax.get_xticklabels()
            for i, label in enumerate(labels):
                if i % 2 != 0:
                    label.set_visible(False)

        # Set axis labels based on orientation
        xlabel = 'Month' if self.ax_orientation == "vertical" else 'Year'
        ylabel = 'Year' if self.ax_orientation == "vertical" else 'Month'

        # Format
        self.format(
            ax_xlabel_txt=xlabel,
            ax_ylabel_txt=ylabel,
            plot=p,
            shown_freq=f'{self.agg}, MS'
        )

        if self.showplot:
            self.fig.patch.set_facecolor('white')
            self.fig.show()
