"""
TREERING: CIRCULAR TIME SERIES VISUALIZATION
=============================================

Displays annual time series data as concentric rings in a polar coordinate system.
Each ring represents one year, growing outward from the center. Color encodes
the data value (e.g., temperature anomaly). Months are arranged around the
circumference with January at the bottom and July at the top, mirroring the
natural climate calendar where summer occupies the top of the plot.

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import diive.core.plotting.styles.LightTheme as theme
from diive.core.utils.console import detail


class TreeRingPlot:
    """Circular spiral visualization of annual time series as concentric color-coded rings.

    Creates a polar plot where each year forms an annular ring. Color encodes a numeric
    value (e.g., temperature anomaly). Months are placed around the circle with January
    at the bottom (6 o'clock) and July at the top (12 o'clock), so the annual temperature
    cycle reads intuitively: cold winters at the bottom, warm summers at the top.

    Inner rings correspond to the earliest years in the data; outer rings to the most
    recent, giving a tree-ring-like growth pattern that shows trends at a glance.

    Top-level alias: ``dv.plot_treering(df, value_col, ...)``

    Example::

        import diive as dv
        df = dv.load_exampledata_parquet()
        tr = dv.plot_treering(df=df, value_col='Tair_f', resample_freq='D')
        tr.plot(cmap='RdBu_r', vmin=-20, vmax=20,
                title='Air temperature (2013-2022)',
                cb_label='Air temperature (deg C)')

    See Also:
        examples/visualization/plot_treering_temperature.py
    """

    def __init__(self,
                 df: pd.DataFrame,
                 value_col: str,
                 resample_freq: str = 'D',
                 verbose: bool = False):
        """Prepare time series data for tree-ring plotting (Phase 1 of two-phase design).

        Args:
            df: DataFrame with DatetimeIndex at any sub-annual frequency
            value_col: Column name with the numeric values to visualize
            resample_freq: Pandas offset alias controlling angular resolution of each ring.
                ``'D'`` (default) resamples to daily means (366 slots per ring).
                Finer aliases such as ``'h'`` or ``'30min'`` preserve more detail at the
                cost of more slots.  Pass ``None`` to use the raw data as-is; the slot
                count is then inferred from the median time step.
            verbose: Print progress messages (default False)

        Raises:
            ValueError: If df does not have a DatetimeIndex or is empty
            KeyError: If value_col is not found in df
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if value_col not in df.columns:
            raise KeyError(f"Column '{value_col}' not found in df")
        if df.empty:
            raise ValueError("df cannot be empty")

        self.df = df[[value_col]].copy()
        self.value_col = value_col
        self.resample_freq = resample_freq
        self.verbose = verbose

        self._prepare_data()

    def _prepare_data(self):
        """Build 2D grid (years x angular slots) of values for polar rendering."""

        # --- Resample if requested ---
        if self.resample_freq:
            data = self.df.resample(self.resample_freq).mean()
        else:
            data = self.df

        years = sorted(data.index.year.unique())
        self.years = years
        self.n_years = len(years)

        # --- Determine angular slot count from frequency ---
        # Each slot corresponds to a thin arc of the ring.  Daily data uses 366
        # slots (one per calendar day, covering leap years).  Finer frequencies
        # scale proportionally, giving narrower slices and more detail per ring.
        if self.resample_freq == 'D':
            n_slots = 366
        elif self.resample_freq:
            offset = pd.tseries.frequencies.to_offset(self.resample_freq)
            freq_days = offset.nanos / (86400 * 1e9)
            n_slots = int(np.ceil(366 / freq_days))
        else:
            # Auto-detect from median time delta of the raw data
            delta = data.index.to_series().diff().median()
            freq_days = delta.total_seconds() / 86400
            n_slots = int(np.ceil(366 / freq_days))

        self.n_days = n_slots  # attribute name kept for plot() compatibility
        self.grid = np.full((self.n_years, n_slots), np.nan)

        year_to_idx = {y: i for i, y in enumerate(years)}

        # Map each observation to (year_ring, angular_slot) via year fraction
        for ts, row in data.iterrows():
            yi = year_to_idx.get(ts.year)
            if yi is None:
                continue
            year_start = pd.Timestamp(ts.year, 1, 1)
            year_end = pd.Timestamp(ts.year + 1, 1, 1)
            frac = (ts - year_start).total_seconds() / (year_end - year_start).total_seconds()
            di = min(int(frac * n_slots), n_slots - 1)
            self.grid[yi, di] = row[self.value_col]

        if self.verbose:
            filled = int(np.sum(~np.isnan(self.grid)))
            total = self.n_years * n_slots
            detail(f"Grid: {self.n_years} years x {n_slots} slots, {filled}/{total} filled "
                   f"(freq={self.resample_freq})", verbose=self.verbose)

    def plot(self,
             ax=None,
             figsize: tuple = (10, 10),
             figdpi: int = 100,
             title: str = None,
             cmap: str = 'RdBu_r',
             vmin: float = None,
             vmax: float = None,
             show_month_labels: bool = True,
             show_month_lines: bool = False,
             show_year_labels: bool = True,
             show_year_separators: bool = True,
             year_label_frequency: int = 10,
             cb_label: str = None,
             cb_labelsize: float = None,
             cb_digits_after_comma: int = 1,
             month_label_fontsize: float = None,
             year_label_fontsize: float = None,
             title_fontsize: float = None):
        """Render TreeRingPlot on a polar axis (Phase 2 of two-phase design).

        All styling and presentation parameters live here. Can be called multiple times
        on the same object to produce differently styled outputs.

        Args:
            ax: Polar matplotlib axes. If None, a new figure is created
            figsize: Figure size in inches (default (10, 10))
            figdpi: Figure DPI (default 100)
            title: Plot title text
            cmap: Colormap name (default 'RdBu_r', blue=low, red=high)
            vmin: Minimum color scale value. If the data contains values below vmin,
                the colorbar is extended with a downward arrow automatically.
                Defaults to data minimum.
            vmax: Maximum color scale value. If the data contains values above vmax,
                the colorbar is extended with an upward arrow automatically.
                Defaults to data maximum.
            show_month_labels: Show month abbreviations around the outer ring (default True)
            show_month_lines: Draw thin radial lines at each month boundary (default False)
            show_year_labels: Show year numbers on rings (default True)
            show_year_separators: Draw thin white circles between year rings (default True)
            year_label_frequency: Interval between year labels, e.g. 10 = every decade (default 10)
            cb_label: Colorbar label text
            cb_labelsize: Font size for colorbar tick labels
            cb_digits_after_comma: Decimal places on colorbar tick labels (default 1)
            month_label_fontsize: Font size for month labels (default theme value)
            year_label_fontsize: Font size for year labels (default theme value)
            title_fontsize: Font size for the plot title (default theme value)
        """
        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if month_label_fontsize is None:
            month_label_fontsize = theme.AX_LABELS_FONTSIZE
        if year_label_fontsize is None:
            year_label_fontsize = theme.TICKS_LABELS_FONTSIZE - 4
        if title_fontsize is None:
            title_fontsize = theme.AX_LABELS_FONTSIZE

        flat = self.grid[~np.isnan(self.grid)]
        data_min = float(np.nanmin(flat))
        data_max = float(np.nanmax(flat))

        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max

        # --- Colorbar extension arrows ---
        # Extend automatically when vmin/vmax clip the actual data range, mirroring
        # the heatmap_base convention used across diive plotting classes.
        clipped_below = vmin > data_min
        clipped_above = vmax < data_max
        if clipped_below and clipped_above:
            cb_extend = 'both'
        elif clipped_below:
            cb_extend = 'min'
        elif clipped_above:
            cb_extend = 'max'
        else:
            cb_extend = 'neither'

        # Create polar figure when no ax is provided
        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi)
            ax = fig.add_subplot(111, projection='polar')
            self._fig = fig
        else:
            self._fig = ax.figure

        self.ax = ax

        # --- Angle layout ---
        # Clockwise progression: 0 rad = right (3 o'clock), direction=-1 makes angles
        # increase clockwise. Jan 1 placed at pi/2 (bottom, 6 o'clock) in CW mode.
        # Months flow: Jan(bottom) -> Apr(left) -> Jul(top) -> Oct(right) -> Jan.
        day_fractions = np.linspace(0, 1, self.n_days + 1)
        offset = np.pi / 2.0  # Jan 1 at bottom (CW mode)
        angles = offset + day_fractions * 2.0 * np.pi  # CW from Jan 1

        # --- Radius layout ---
        # Small inner hub keeps the plot readable; each year occupies one radial unit.
        inner_gap = 1.5
        radii = np.arange(self.n_years + 1, dtype=float) + inner_gap

        # --- Color mesh ---
        colormap = plt.get_cmap(cmap)
        colormap.set_bad(color='none')  # NaN -> transparent

        mesh = ax.pcolormesh(
            angles,
            radii,
            self.grid,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='flat',
            zorder=0
        )

        # --- Polar axis cosmetics ---
        ax.set_theta_zero_location('E')  # 0 rad = right (3 o'clock)
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rlim(0, radii[-1] + 0.5)
        ax.axis('off')  # hide default polar ticks/grid

        # --- Month boundary angles (start of each month, non-leap reference year) ---
        month_start_angles = []
        for m in range(1, 13):
            start_day = pd.Timestamp(2001, m, 1).dayofyear
            month_start_angles.append(offset + (start_day - 1) / 365.0 * 2.0 * np.pi)

        # --- Month lines ---
        # Thin radial spokes at each month boundary, from inner hub to outer edge.
        if show_month_lines:
            for angle in month_start_angles:
                ax.plot(
                    [angle, angle],
                    [inner_gap, radii[-1]],
                    color='white',
                    linewidth=0.8,
                    alpha=0.7,
                    zorder=2
                )

        # --- Year separators ---
        # Thin white concentric circles between rings make individual years easier to
        # distinguish, especially in datasets spanning many decades.
        if show_year_separators:
            theta_circle = np.linspace(0, 2.0 * np.pi, 360)
            for r in radii[1:-1]:  # skip innermost and outermost edges
                ax.plot(
                    theta_circle, np.full_like(theta_circle, r),
                    color='white', linewidth=0.5, alpha=0.6, zorder=1
                )

        # --- Month labels ---
        if show_month_labels:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            label_radius = radii[-1] + 2.8  # clear gap outside the outermost ring

            for m_idx, m_name in enumerate(month_names):
                mid = pd.Timestamp(2001, m_idx + 1, 15).dayofyear
                angle = offset + (mid - 1) / 365.0 * 2.0 * np.pi
                ax.text(
                    angle, label_radius, m_name,
                    ha='center', va='center',
                    fontsize=month_label_fontsize,
                    fontweight='bold',
                    color='#333333'
                )

        # --- Year labels ---
        # Place year text at Jan-1 angle (bottom), at the ring's midpoint radius.
        if show_year_labels:
            jan1_angle = offset  # 3pi/2 (bottom)
            for i, year in enumerate(self.years):
                if (year % year_label_frequency == 0) or (i == 0) or (i == self.n_years - 1):
                    r_mid = radii[i] + 0.5
                    ax.text(
                        jan1_angle, r_mid, str(year),
                        ha='center', va='center',
                        fontsize=year_label_fontsize,
                        color='black',
                        fontweight='bold',
                        zorder=5
                    )

        # --- Title ---
        if title:
            # Reserve top margin so suptitle stays within the saved figure boundary.
            self._fig.subplots_adjust(top=0.92)
            self._fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.97)

        # --- Colorbar ---
        # Shrink and shift the colorbar so it does not overlap month labels on the
        # right side (Apr/May area).
        fmt = mticker.FormatStrFormatter(f'%.{cb_digits_after_comma}f')
        cb = self._fig.colorbar(
            mesh,
            ax=ax,
            orientation='vertical',
            shrink=0.6,
            pad=0.12,
            extend=cb_extend,
            format=fmt
        )
        if cb_label:
            cb.set_label(cb_label, fontsize=cb_labelsize, rotation=90, labelpad=10)
        cb.ax.tick_params(labelsize=cb_labelsize)

    def plot_line(self,
                  ax=None,
                  figsize: tuple = (10, 10),
                  figdpi: int = 100,
                  title: str = None,
                  vmin: float = None,
                  vmax: float = None,
                  cmap: str = 'RdYlBu_r',
                  linewidth: float = 1.2,
                  alpha: float = 0.85,
                  amplitude_scale: float = 0.5,
                  ring_width: float = 0.35,
                  show_month_labels: bool = True,
                  show_month_lines: bool = False,
                  show_year_labels: bool = False,
                  show_year_separators: bool = False,
                  year_label_frequency: int = 1,
                  cb_label: str = None,
                  cb_labelsize: float = None,
                  cb_digits_after_comma: int = 1,
                  month_label_fontsize: float = None,
                  year_label_fontsize: float = None,
                  title_fontsize: float = None):
        """Render TreeRingPlot as concentric radial line traces colored by data value.

        Each year traces its own ring at a fixed base radius (oldest at center, newest
        at the outer edge). Within each ring the line wiggles radially by
        ``amplitude_scale`` around the base, encoding the data value. Segment color also
        encodes the data value via ``cmap``, so both position and color carry the same
        information — matching the reference spiral-plot style.

        All styling and presentation parameters live here. Can be called multiple times
        on the same object to produce differently styled outputs.

        Args:
            ax: Polar matplotlib axes. If None, a new figure is created.
            figsize: Figure size in inches (default (10, 10))
            figdpi: Figure DPI (default 100)
            title: Plot title text
            vmin: Low colour anchor (defaults to data minimum).
            vmax: High colour anchor (defaults to data maximum).
            cmap: Colormap applied to data values (default 'RdYlBu_r', blue=low, red=high).
            linewidth: Line width in points (default 0.8)
            alpha: Line opacity (default 0.85)
            amplitude_scale: Maximum radial displacement as a fraction of one ring width.
                0.5 makes lines reach ring boundaries; >0.5 causes overlap (default 0.5).
            ring_width: Radial width of each year's ring in data units (default 0.35).
                Smaller values pack rings closer together.
            show_month_labels: Show month abbreviations outside the outer ring (default True)
            show_month_lines: Draw thin radial spokes at each month boundary (default False)
            show_year_labels: Show year numbers at each ring's midpoint (default True)
            show_year_separators: Draw faint circles between year rings (default False)
            year_label_frequency: Interval between year labels, e.g. 1 = every year (default 1)
            cb_label: Colorbar label text
            cb_labelsize: Font size for colorbar tick labels
            cb_digits_after_comma: Decimal places on colorbar tick labels (default 1)
            month_label_fontsize: Font size for month labels (default theme value)
            year_label_fontsize: Font size for year labels (default theme value)
            title_fontsize: Font size for the plot title (default theme value)
        """
        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if month_label_fontsize is None:
            month_label_fontsize = theme.AX_LABELS_FONTSIZE
        if year_label_fontsize is None:
            year_label_fontsize = theme.TICKS_LABELS_FONTSIZE - 4
        if title_fontsize is None:
            title_fontsize = theme.AX_LABELS_FONTSIZE

        flat = self.grid[~np.isnan(self.grid)]
        data_min = float(np.nanmin(flat))
        data_max = float(np.nanmax(flat))
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max

        data_range = vmax - vmin if vmax != vmin else 1.0
        data_mid = (vmin + vmax) / 2.0

        clipped_below = vmin > data_min
        clipped_above = vmax < data_max
        if clipped_below and clipped_above:
            cb_extend = 'both'
        elif clipped_below:
            cb_extend = 'min'
        elif clipped_above:
            cb_extend = 'max'
        else:
            cb_extend = 'neither'

        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi)
            ax = fig.add_subplot(111, projection='polar')
            self._fig = fig
        else:
            self._fig = ax.figure
        self.ax = ax

        # Angle layout — Jan at bottom (6 o'clock), CW progression, same as plot().
        offset = np.pi / 2.0
        slot_fracs = np.linspace(0, 1, self.n_days, endpoint=False)
        theta = offset + slot_fracs * 2.0 * np.pi
        theta_closed = np.append(theta, theta[0])

        # Radius layout — inner_gap reserves white center; ring_width controls spacing.
        inner_gap = 1.5
        radii = np.arange(self.n_years + 1, dtype=float) * ring_width + inner_gap

        colormap = plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for i, year in enumerate(self.years):
            r_base = radii[i] + ring_width / 2.0  # midpoint of this year's ring
            vals = self.grid[i, :]
            # Wiggle ±(amplitude_scale * ring_width) around the ring's base radius.
            normalized = (vals - data_mid) / (data_range / 2.0) * amplitude_scale * ring_width
            r = r_base + normalized
            r_closed = np.append(r, r[0])
            vals_closed = np.append(vals, vals[0])

            # Segments in polar (theta, r) — polar axes transData handles the transform.
            points = np.array([theta_closed, r_closed]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color each segment by the midpoint data value of its two endpoints.
            seg_vals = (vals_closed[:-1] + vals_closed[1:]) / 2.0

            lc = LineCollection(segments, cmap=colormap, norm=norm,
                                linewidth=linewidth, alpha=alpha, zorder=3)
            lc.set_array(seg_vals)
            ax.add_collection(lc)

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rlim(0, radii[-1] + ring_width * 0.5)
        ax.axis('off')

        # Month boundary angles (non-leap reference year)
        month_start_angles = []
        for m in range(1, 13):
            start_day = pd.Timestamp(2001, m, 1).dayofyear
            month_start_angles.append(offset + (start_day - 1) / 365.0 * 2.0 * np.pi)

        if show_month_lines:
            for angle in month_start_angles:
                ax.plot([angle, angle], [inner_gap, radii[-1]],
                        color='#888888', linewidth=0.8, alpha=0.5, zorder=2)

        if show_year_separators:
            theta_circle = np.linspace(0, 2.0 * np.pi, 360)
            for r in radii[1:-1]:
                ax.plot(theta_circle, np.full_like(theta_circle, r),
                        color='#cccccc', linewidth=0.4, alpha=0.5, zorder=1)

        if show_month_labels:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            label_radius = radii[-1] + 2.8
            for m_idx, m_name in enumerate(month_names):
                mid = pd.Timestamp(2001, m_idx + 1, 15).dayofyear
                angle = offset + (mid - 1) / 365.0 * 2.0 * np.pi
                ax.text(angle, label_radius, m_name,
                        ha='center', va='center',
                        fontsize=month_label_fontsize, fontweight='bold',
                        color='#333333')

        if show_year_labels:
            jan1_angle = offset
            for i, year in enumerate(self.years):
                if (year % year_label_frequency == 0) or (i == 0) or (i == self.n_years - 1):
                    r_mid = radii[i] + ring_width / 2.0
                    ax.text(jan1_angle, r_mid, str(year),
                            ha='center', va='center',
                            fontsize=year_label_fontsize, color='black',
                            fontweight='bold', zorder=5)

        if title:
            self._fig.subplots_adjust(top=0.92)
            self._fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=0.97)

        # Colorbar — ScalarMappable since LineCollection has no standalone colorbar method.
        fmt = mticker.FormatStrFormatter(f'%.{cb_digits_after_comma}f')
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cb = self._fig.colorbar(
            sm,
            ax=ax,
            orientation='vertical',
            shrink=0.6,
            pad=0.12,
            extend=cb_extend,
            format=fmt
        )
        if cb_label:
            cb.set_label(cb_label, fontsize=cb_labelsize, rotation=90, labelpad=10)
        cb.ax.tick_params(labelsize=cb_labelsize)
