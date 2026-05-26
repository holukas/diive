"""
DETECT_TIMESTAMP_SHIFTS: TIMESTAMP SHIFT DETECTION VIA RADIATION PHASE ANALYSIS
=================================================================================

Tools for detecting clock/timestamp errors in meteorological time series by
comparing measured shortwave radiation against theoretical potential radiation.

Two detection approaches are implemented:

- **FFT phase shift** (``DetectTimestampShifts.fft_phase_shift``):
  Projects each day's radiation signal onto the 24-hour Fourier basis vector and
  computes the phase angle difference between measured and potential radiation.
  Fast and robust on clear days; reports shifts in minutes.

- **High-resolution cross-correlation** (``DetectTimestampShifts.crosscorr``):
  Upsamples data to 1-minute resolution via interpolation, then iterates through
  candidate lags to find the shift that maximises the Pearson correlation between
  measured and potential radiation.  Achieves 1-minute precision at the cost of
  higher compute time.

Supporting detection method:

- ``DetectTimestampShifts.noon_shift`` — simple peak-time comparison per day.

Plot methods:

- ``plot_fft_results`` — time series, histogram, polar and monthly boxplot.
- ``plot_crosscorr_results`` — scatter over time and shift histogram.
- ``plot_monthly_dielcycles`` — multi-year diel cycles per calendar month.
- ``plot_radiation_fingerprint`` — heatmap of intra-day radiation patterns.

Example
-------
``examples/preprocessing/qaqc/qaqc_detect_timestamp_shifts.py``

Part of the diive library: https://github.com/holukas/diive
"""

import calendar

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from scipy.signal import correlate as sp_correlate

from diive.core.times.resampling import diel_cycle
from diive.features.variables.potentialradiation import potrad


class DetectTimestampShifts:
    """Detect timestamp/clock errors by comparing measured vs potential radiation.

    Potential radiation is computed automatically when lat/lon are supplied and
    col_pot is absent from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DateTimeIndex containing at minimum the measured radiation
        column.  If col_pot is already present it is used directly; otherwise
        potential radiation is computed from lat, lon, and utc_offset.
    col_meas : str
        Column name of the measured shortwave radiation signal.
    col_pot : str
        Column name for potential radiation.
    lat : float, optional
        Site latitude in decimal degrees.  Required when col_pot is not in df.
    lon : float, optional
        Site longitude in decimal degrees.  Required when col_pot is not in df.
    utc_offset : int
        UTC offset in hours (e.g. 1 for CET).
    """

    def __init__(
            self,
            df: pd.DataFrame,
            col_meas: str,
            col_pot: str,
            lat: float = None,
            lon: float = None,
            utc_offset: int = 1,
    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex")

        cols = [col_meas] + ([col_pot] if col_pot in df.columns else [])
        self.df = df[cols].copy()
        self.col_meas = col_meas
        self.col_pot = col_pot
        self.lat = lat
        self.lon = lon
        self.utc_offset = utc_offset

        if col_pot not in self.df.columns:
            if lat is None or lon is None:
                raise ValueError("lat and lon are required when col_pot is not in df")
            self.df[col_pot] = potrad(
                timestamp_index=self.df.index,
                lat=lat,
                lon=lon,
                utc_offset=utc_offset,
            )

        self._fft_results: pd.DataFrame = None
        self._crosscorr_results: pd.DataFrame = None
        self._noon_shift_results: pd.Series = None

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def fft_phase_shift(self, min_clearness: float = 0.6) -> pd.DataFrame:
        """Detect daily time shift via FFT phase-angle comparison.

        Projects each day's signal onto the k=1 Fourier basis (24-hour cycle)
        and computes the phase difference between measured and potential radiation.
        Positive shift_minutes means the measured peak is earlier than potential.

        Math: Delta_t (min) = (Delta_phi (rad) / 2*pi) * 1440

        Parameters
        ----------
        min_clearness : float
            Minimum clearness index (measured / potential daily sum).  Days below
            this are too cloudy to yield a reliable phase estimate.

        Returns
        -------
        pd.DataFrame
            Indexed by date.  Columns: shift_minutes, amplitude_meas.
        """
        df = self.df

        freq = pd.infer_freq(df.index)
        freq = '1min' if freq == 'min' else freq
        if freq:
            dt_min = pd.to_timedelta(freq).total_seconds() / 60
        elif len(df.index) >= 2:
            # median diff is robust against logging gaps at the start of the index
            dt_min = df.index.to_series().diff().median().total_seconds() / 60
        else:
            raise ValueError("Cannot determine sampling frequency: index has fewer than 2 rows")

        points_per_day = int(1440 / dt_min)
        results = {}

        for date, group in df.groupby(df.index.normalize()):
            if len(group) < points_per_day * 0.9:
                results[date] = {'shift_minutes': np.nan, 'amplitude_meas': 0}
                continue

            # Interpolate short daytime gaps first (limit=4 ~ 2 h at 30-min resolution);
            # fill remaining NaN (nighttime) with zero so FFT operates on a full vector
            y_meas = group[self.col_meas].interpolate(method='linear', limit=4).fillna(0).to_numpy()
            y_pot = group[self.col_pot].fillna(0).to_numpy()

            pot_sum = np.sum(y_pot)
            if pot_sum <= 0 or np.sum(y_meas) / pot_sum < min_clearness:
                results[date] = {'shift_minutes': np.nan, 'amplitude_meas': 0}
                continue

            N = len(y_meas)
            n = np.arange(N)
            # k=1 basis vector: one complete cycle over the N samples of the day
            basis = np.exp(-1j * 2 * np.pi * n / N)

            X_meas = np.sum(y_meas * basis)
            X_pot = np.sum(y_pot * basis)

            delta_phi = np.angle(X_meas) - np.angle(X_pot)
            # Wrap to [-pi, pi] to find shortest path around the circle
            delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
            shift_minutes = (delta_phi / (2 * np.pi)) * 1440

            results[date] = {
                'shift_minutes': shift_minutes,
                'amplitude_meas': np.abs(X_meas),
            }

        self._fft_results = pd.DataFrame.from_dict(results, orient='index')
        return self._fft_results

    def crosscorr(
            self,
            max_shift_min: int = 120,
            upsample_freq: str = '1min',
            min_clearness_index: float = 0.5,
    ) -> pd.DataFrame:
        """Detect daily time shift via high-resolution cross-correlation.

        Upsamples each day to upsample_freq resolution, then searches the lag
        that maximises the Pearson correlation between measured and potential
        radiation curves.

        Parameters
        ----------
        max_shift_min : int
            Search window in minutes (symmetric, e.g. 120 searches +/-2 h).
        upsample_freq : str
            Pandas frequency string for upsampling (e.g. '1min').
        min_clearness_index : float
            Minimum daily clearness index to include a day.

        Returns
        -------
        pd.DataFrame
            Indexed by date.  Columns: shift_minutes, max_corr.
        """
        df = self.df
        results = {}

        for date, group in df.groupby(df.index.normalize()):
            pot_sum = group[self.col_pot].sum()
            meas_sum = group[self.col_meas].sum()

            if pot_sum < 100:
                results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
                continue

            if meas_sum / pot_sum < min_clearness_index:
                results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
                continue

            try:
                highres_idx = pd.date_range(group.index[0], group.index[-1], freq=upsample_freq)

                if (group[self.col_pot] > 0).sum() < 5:
                    continue

                # pchip for potential: sun moves smoothly, cubic interpolation is safe
                ts_pot_hr = (
                    group[self.col_pot]
                    .reindex(highres_idx)
                    .interpolate(method='pchip')
                    .fillna(0)
                )
                # linear for measured: clouds cause sharp edges; cubic causes ringing
                ts_meas_hr = (
                    group[self.col_meas]
                    .reindex(highres_idx)
                    .interpolate(method='linear')
                    .fillna(0)
                )

                # Restrict to daytime to avoid correlating noise against noise at night
                sun_up = ts_pot_hr > 10
                ts_pot_hr = ts_pot_hr[sun_up]
                ts_meas_hr = ts_meas_hr[sun_up]

                if len(ts_pot_hr) == 0:
                    continue

                pot_arr = ts_pot_hr.to_numpy()
                meas_arr = ts_meas_hr.to_numpy()

                # Zero-mean arrays so the cross-correlation is proportional to Pearson r
                pot_zm = pot_arr - pot_arr.mean()
                meas_zm = meas_arr - meas_arr.mean()

                # correlate(a, b) at lag L = sum a[n] * b[n-L]; peaks when shifting
                # meas by L aligns it with pot, matching the original sign convention
                corr_full = sp_correlate(pot_zm, meas_zm, mode='full')
                lags_full = np.arange(-(len(meas_zm) - 1), len(pot_zm))

                mask = (lags_full >= -max_shift_min) & (lags_full <= max_shift_min)
                lags_win = lags_full[mask]
                corr_win = corr_full[mask]

                best_idx = int(np.argmax(corr_win))
                best_lag = int(lags_win[best_idx])

                denom = np.std(pot_zm) * np.std(meas_zm) * len(pot_zm)
                best_corr = float(corr_win[best_idx] / denom) if denom > 0 else 0.0

                results[date] = {'shift_minutes': best_lag, 'max_corr': best_corr}

            except (ValueError, TypeError):
                results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}

        self._crosscorr_results = pd.DataFrame.from_dict(results, orient='index')
        return self._crosscorr_results

    def noon_shift(self, clearness_threshold: float = 0.7) -> pd.Series:
        """Detect time shift by comparing daily peak times of measured vs potential.

        Parameters
        ----------
        clearness_threshold : float
            Minimum clearness index (measured / potential daily sum) to include a day.

        Returns
        -------
        pd.Series
            Daily time shift in minutes (positive = measured peak is earlier).
            Index is DatetimeIndex of clear days only.
        """
        df = self.df
        daily_sums = df[[self.col_meas, self.col_pot]].resample('D').sum()
        clearness = daily_sums[self.col_meas] / daily_sums[self.col_pot]

        idx_max_meas = df[self.col_meas].groupby(df.index.normalize()).idxmax()
        idx_max_pot = df[self.col_pot].groupby(df.index.normalize()).idxmax()

        # pot - meas: positive when meas peaks earlier, matching FFT and crosscorr convention
        delta = (idx_max_pot - idx_max_meas).dt.total_seconds() / 60
        result = pd.Series(delta.values, index=idx_max_meas.index, name='time_shift_minutes')

        clear_mask = clearness[clearness > clearness_threshold].index
        self._noon_shift_results = result[result.index.isin(clear_mask)]
        return self._noon_shift_results

    # ------------------------------------------------------------------
    # Plot methods  (Phase 2: styling + rendering only)
    # ------------------------------------------------------------------

    def plot_fft_results(
            self,
            amplitude_threshold: float = 1000,
            rolling_window: int = 15,
            title: str = 'Phase Shift Detection (FFT)',
    ) -> tuple:
        """Four-panel overview of FFT phase-shift results.

        Panels: time series with rolling median, shift histogram, polar scatter,
        monthly boxplot.

        Parameters
        ----------
        amplitude_threshold : float
            Minimum measured FFT amplitude to consider a day valid.  Filters
            days where the radiation signal is too weak for a reliable phase estimate.
        rolling_window : int
            Window size in days for the rolling median trend line.
        title : str
            Figure suptitle.

        Returns
        -------
        tuple(fig, list of Axes)
        """
        if self._fft_results is None:
            raise RuntimeError("Call fft_phase_shift() before plotting.")

        valid = self._fft_results[self._fft_results['amplitude_meas'] > amplitude_threshold].copy()

        fig = plt.figure(figsize=(14, 10))
        gs = grid_spec.GridSpec(3, 2)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(valid.index, valid['shift_minutes'],
                    color='teal', alpha=0.6, s=15, label='Daily Phase Shift')
        rolling = valid['shift_minutes'].rolling(window=rolling_window, center=True).median()
        ax1.plot(valid.index, rolling, color='red', linewidth=2,
                 label=f'{rolling_window}-Day Rolling Median')
        ax1.set_ylabel('Time Shift (Minutes)')
        ax1.set_title(f'{title}\nPositive = Measured is EARLY (Leading) | Negative = Measured is LATE (Lagging)')
        ax1.axhline(0, color='k', linewidth=1)
        ax1.axhline(60, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(-60, color='gray', linestyle=':', alpha=0.5)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        try:
            sns.histplot(valid['shift_minutes'], bins=50, kde=True, ax=ax2, color='teal')
        except (ValueError, TypeError):
            ax2.hist(valid['shift_minutes'], bins=50, color='teal', alpha=0.7)
        ax2.set_xlabel('Shift (Minutes)')
        ax2.set_title('Distribution of Time Shifts')
        ax2.axvline(0, color='k')
        median_val = valid['shift_minutes'].median()
        ax2.axvline(median_val, color='red', linestyle='--')
        ax2.text(median_val, ax2.get_ylim()[1] * 0.9,
                 f' Median: {median_val:.2f} min', color='red')

        ax3 = fig.add_subplot(gs[1, 1], projection='polar')
        rads = (valid['shift_minutes'] / 1440) * 2 * np.pi
        ax3.scatter(rads, valid['amplitude_meas'], c='teal', alpha=0.3, s=10)
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_title('Phase Shift Polar Plot (0 = Perfect Sync)')
        ax3.set_thetamin(-45)
        ax3.set_thetamax(45)

        ax4 = fig.add_subplot(gs[2, :])
        valid = valid.assign(Month=valid.index.month)
        valid.boxplot(column='shift_minutes', by='Month', ax=ax4, grid=False,
                      patch_artist=True, boxprops=dict(facecolor='teal', alpha=0.5))
        ax4.set_title('Monthly Variability of Time Shift')
        ax4.set_ylabel('Shift (Minutes)')
        ax4.axhline(0, color='k', linewidth=1)
        fig.suptitle('')

        plt.tight_layout()
        return fig, fig.axes

    def plot_crosscorr_results(
            self,
            min_corr: float = 0.97,
            title: str = 'High-Resolution Time Shift Detection (1-min precision)',
    ) -> tuple:
        """Two-panel plot of cross-correlation shift results.

        Panels: scatter of shift over time (colour-coded by correlation strength),
        histogram of shift distribution.

        Parameters
        ----------
        min_corr : float
            Minimum correlation strength to include a day as confident.
        title : str
            Title for the upper subplot.

        Returns
        -------
        tuple(fig, list of Axes)
        """
        if self._crosscorr_results is None:
            raise RuntimeError("Call crosscorr() before plotting.")

        confident = self._crosscorr_results[self._crosscorr_results['max_corr'] > min_corr]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                       gridspec_kw={'height_ratios': [2, 1]})

        sc = ax1.scatter(confident.index, confident['shift_minutes'],
                         c=confident['max_corr'], cmap='viridis', s=15, alpha=0.8)
        ax1.set_ylabel('Time Shift (Minutes)')
        ax1.set_title(f'{title}\nPositive = Measured is Early | Negative = Measured is Late')
        ax1.axhline(0, c='k', lw=1)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax1, label='Correlation Strength')
        ax1.axhline(60, c='r', ls=':', alpha=0.4)
        if len(confident) > 0:
            ax1.text(confident.index[0], 62, 'DST (+60m)', color='red', fontsize=8)
        ax1.axhline(-60, c='r', ls=':', alpha=0.4)

        ax2.hist(confident['shift_minutes'], bins=range(-120, 120, 2),
                 color='tab:blue', alpha=0.7, edgecolor='k')
        ax2.set_xlabel('Shift Minutes')
        ax2.set_ylabel('Frequency (Days)')
        ax2.set_title('Distribution of Detected Shifts')
        ax2.axvline(0, c='k', lw=2)
        median_shift = confident['shift_minutes'].median()
        if not np.isnan(median_shift):
            ax2.axvline(median_shift, c='r', ls='--', lw=2)
            ax2.text(median_shift + 2, ax2.get_ylim()[1] * 0.9,
                     f'Median: {median_shift:.1f} min', color='r', fontweight='bold')

        plt.tight_layout()
        return fig, [ax1, ax2]

    def plot_noon_shift_results(
            self,
            rolling_window: int = 15,
            title: str = 'Noon Shift Detection (Peak Time)',
    ) -> tuple:
        """Three-panel plot of noon-shift results.

        Panels: scatter of daily shift over time with rolling median, shift
        distribution histogram, monthly boxplot showing seasonal patterns.

        Parameters
        ----------
        rolling_window : int
            Window size in days for the rolling median trend line.
        title : str
            Title for the upper subplot.

        Returns
        -------
        tuple(fig, list of Axes)
        """
        if self._noon_shift_results is None:
            raise RuntimeError("Call noon_shift() before plotting.")

        result = self._noon_shift_results.dropna()

        fig = plt.figure(figsize=(14, 9))
        gs = grid_spec.GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(result.index, result.values,
                    color='steelblue', alpha=0.5, s=15, label='Daily Peak Shift')
        rolling = result.rolling(window=rolling_window, center=True).median()
        ax1.plot(result.index, rolling, color='red', linewidth=2,
                 label=f'{rolling_window}-Day Rolling Median')
        ax1.axhline(0, color='k', linewidth=1)
        ax1.axhline(60, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(-60, color='gray', linestyle=':', alpha=0.5)
        ax1.set_ylabel('Time Shift (Minutes)')
        ax1.set_title(f'{title}\nPositive = Measured is EARLY (Leading) | Negative = Measured is LATE (Lagging)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(result.values, bins=40, color='steelblue', alpha=0.7, edgecolor='k')
        ax2.axvline(0, color='k', lw=2)
        median_val = result.median()
        if not np.isnan(median_val):
            ax2.axvline(median_val, color='red', linestyle='--', lw=2)
            ax2.text(median_val + 1, ax2.get_ylim()[1] * 0.9,
                     f'Median: {median_val:.1f} min', color='red', fontweight='bold')
        ax2.set_xlabel('Shift (Minutes)')
        ax2.set_ylabel('Frequency (Days)')
        ax2.set_title('Distribution of Noon Shifts')

        ax3 = fig.add_subplot(gs[1, 1])
        df_box = pd.DataFrame({'shift_minutes': result.values, 'Month': result.index.month})
        df_box.boxplot(column='shift_minutes', by='Month', ax=ax3, grid=False,
                       patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.5))
        ax3.axhline(0, color='k', linewidth=1)
        ax3.set_title('Monthly Variability')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Shift (Minutes)')
        fig.suptitle('')

        plt.tight_layout()
        return fig, [ax1, ax2, ax3]

    def plot_monthly_dielcycles(
            self,
            years: list = None,
            colors=None,
    ) -> tuple:
        """12-panel figure with multi-year mean diel cycles per calendar month.

        Each panel overlays one line per year for the measured variable plus the
        mean potential radiation curve.

        Parameters
        ----------
        years : list of int, optional
            Years to include.  Defaults to all years in the data.
        colors : array-like, optional
            One colour per year.  Defaults to Spectral_r colormap.

        Returns
        -------
        tuple(fig, list of Axes)
        """
        df = self.df
        if years is None:
            years = sorted(set(df.index.year))
        if colors is None:
            colors = cm.Spectral_r(np.linspace(0, 1, len(years)))

        months = sorted(set(df.index.month))

        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = grid_spec.GridSpec(3, 4)
        axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(4)]
        axmap = {m: axes[i] for i, m in enumerate(range(1, 13))}

        final_handles, final_labels = [], []

        for month in months:
            subset = df.loc[df.index.month == month]
            ax_m = axmap[month]
            ax_r = ax_m.twinx()

            swinpot_dc = diel_cycle(series=subset[self.col_pot], mean=True, std=True, each_month=False)
            means_pot = swinpot_dc['mean'].droplevel(level=0)
            means_pot.index = self._timedelta_to_hhmm(means_pot.index)
            means_pot.plot(ax=ax_r, label=self.col_pot, color='black', zorder=99, lw=2, ls='--')
            if month == 8:
                ax_r.set_ylabel(self.col_pot)

            for yix, year in enumerate(years):
                series_year = subset.loc[subset.index.year == year, self.col_meas]
                dc = diel_cycle(series=series_year, mean=True, std=True, each_month=False)
                means = dc['mean'].droplevel(level=0)
                means.index = self._timedelta_to_hhmm(means.index)
                means.plot(ax=ax_m, label=str(year), color=colors[yix], zorder=99, lw=2, alpha=0.6)

            if month == 5:
                ax_m.set_ylabel(self.col_meas)
            if month in [9, 10, 11, 12]:
                ax_m.set_xlabel('Time of Day')
            ax_m.set_title(calendar.month_abbr[month])

            if not final_handles:
                lines, labels = ax_m.get_legend_handles_labels()
                lines2, labels2 = ax_r.get_legend_handles_labels()
                final_handles = lines + lines2
                final_labels = labels + labels2

        fig.legend(final_handles, final_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.92), ncol=int(np.ceil(len(years) / 2)), frameon=False)
        fig.suptitle(f'Diel Cycles: {self.col_meas}', fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        return fig, list(axmap.values())

    def plot_radiation_fingerprint(
            self,
            year: int,
            ax: plt.Axes = None,
            vmin: float = None,
            vmax: float = None,
    ) -> tuple:
        """Heatmap of intra-day radiation values for a single year.

        Rows are calendar days, columns are time-of-day slots.

        Parameters
        ----------
        year : int
            Year to visualise.
        ax : matplotlib Axes, optional
            Target axes.  A new figure is created when None.
        vmin, vmax : float, optional
            Colour scale limits.  Defaults to data min/max.

        Returns
        -------
        tuple(fig, ax)
        """
        df_year = self.df.loc[self.df.index.year == year].copy()
        df_year['Date'] = df_year.index.date
        df_year['Time'] = df_year.index.time
        pivot = df_year.pivot(index='Date', columns='Time', values=self.col_meas)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 12))
        else:
            fig = ax.get_figure()

        im = ax.imshow(
            pivot, aspect='auto', cmap='inferno', origin='lower',
            vmin=vmin if vmin is not None else pivot.min().min(),
            vmax=vmax if vmax is not None else pivot.max().max(),
        )
        ax.set_title(f'Radiation Fingerprint - {year}')
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Day of Year')
        plt.colorbar(im, ax=ax, label='W/m2')
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timedelta_to_hhmm(index) -> list:
        """Convert a TimedeltaIndex to HH:MM strings for axis labelling."""
        td = pd.to_timedelta(index.astype(str))
        return (pd.to_datetime('today').normalize() + td).strftime('%H:%M').tolist()
