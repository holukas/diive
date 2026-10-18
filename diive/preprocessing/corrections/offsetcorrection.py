"""
OFFSET CORRECTION: MEASUREMENT OFFSET DETECTION AND REMOVAL
===========================================================

Detect and remove systematic offsets: a constant offset to a replicate sensor,
yearly wind direction offsets (histogram comparison with reference years), the
relative humidity offset above 100% and the nighttime zero offset of radiation.

Part of the diive library: https://github.com/holukas/diive
"""

import decimal
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import diive.core.dfun.frames as frames
from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.console import detail
from diive.variables import potrad, daytime_nighttime_flag_from_swinpot
from diive.variables.radiation import _noaa_solar_noon_minutes, _oneflux_daily_rpot


class MeasurementOffsetFromReplicate:
    """
    Detect and remove a constant additive offset between a measurement and a
    co-located replicate.

    Scans candidate offsets and selects the one that minimizes the summed
    absolute difference between the offset-shifted measurement and the
    replicate, then returns the corrected measurement (``measurement + offset``).
    A single global offset is applied — no per-year handling and no gain term.

    Example:
        See `examples/preprocessing/corrections/correction_measurement_offset_replicate.py`
    """

    def __init__(self,
                 measurement: Series,
                 replicate: Series,
                 offset_start: int = -100,
                 offset_end: int = 100,
                 offset_stepsize: float = 0.1,
                 verbose: int | bool | None = None):
        """Find the additive offset that best aligns *measurement* with *replicate*.

        For each offset in ``arange(offset_start, offset_end, offset_stepsize)``,
        the measurement is shifted by that offset and the summed absolute
        difference to the replicate is computed. The offset with the smallest
        summed absolute difference is selected and added to the measurement.

        Args:
            measurement: Time series to be corrected.
            replicate: Co-located reference series the measurement is aligned to.
            offset_start: Smallest offset to try (same units as the data).
            offset_end: Largest offset to try (exclusive).
            offset_stepsize: Step between candidate offsets.
            verbose: Verbosity for the console helpers. The line printed for each
                candidate offset appears at ``VERBOSE_DEBUG`` (3). None uses the
                level set with ``set_verbosity()``.
        """
        self.measurement = measurement
        self.replicate = replicate
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.offset_stepsize = offset_stepsize
        self.verbose = verbose

        d = decimal.Decimal(str(offset_stepsize))
        self.n_digits_after_comma = abs(d.as_tuple().exponent)

        # Find the best-fitting constant offset.
        self.offset = self._find_best_offset()

        # Apply it to the measurement.
        self.measurement_corrected = self._correct_measurement()

    def get_corrected_measurement(self):
        """Return the offset-corrected measurement series."""
        return self.measurement_corrected

    def get_offset(self):
        """Return the detected offset (from the replicate measurement)."""
        return self.offset

    def _correct_measurement(self):
        """Add the detected offset to the measurement."""
        measurement_corrected = self.measurement.add(self.offset)
        return measurement_corrected

    def _find_best_offset(self):
        """Scan candidate offsets and return the one minimizing the summed
        absolute difference between the offset-shifted measurement and the replicate."""
        m = self.measurement.copy()
        r = self.replicate.copy()
        offsets_df = pd.DataFrame(columns=['OFFSET', 'ABS_DIFF'])
        counter = 0
        for offset in np.arange(self.offset_start, self.offset_end, self.offset_stepsize):
            counter += 1
            m_shifted = m.copy()
            m_shifted += offset
            abs_diff = m_shifted.sub(r)
            abs_diff = abs_diff.abs()
            abs_diff = abs_diff.sum()
            abs_diff = float(abs_diff)
            offsets_df.loc[len(offsets_df)] = [offset, abs_diff]

            detail(f"#{counter}   trying with offset: {offset:.{self.n_digits_after_comma}f}   "
                   f"found absolute difference: {abs_diff}", verbose=self.verbose)

        offsets_df = offsets_df.sort_values(by='ABS_DIFF', ascending=True)
        min_ix = offsets_df['ABS_DIFF'] == offsets_df['ABS_DIFF'].min()
        offset = offsets_df.loc[min_ix]['OFFSET'].iloc[0]
        return offset

    # def showplots(self):
    #     """Plot absolute correlations for each year"""
    #     for key, val in self.shiftdict.items():
    #         shiftdf = val.set_index(keys='SHIFT', drop=True)
    #         shiftdf.plot()
    #         plt.show()


def remove_relativehumidity_offset(series: Series,
                                   showplot: bool = False) -> Series:
    """Correct relative humidity offset > 100%

    Works for relative humidity data where maximum values should not exceed 100%.

    For each day with values above 100%, the offset is the mean of these values
    minus 100. Days without values above 100% get the offset interpolated
    linearly between the nearest earlier and later days with such values.
    Before the first and after the last such day, the offset of that day is
    carried. A single day above 100% (e.g. two hours of fog) thus lowers every
    record of the series by its offset. This assumes that a sensor reading too
    high when saturated reads too high on every day. The result depends on the
    period screened. Without any value above 100% the offset is 0. After the
    offset is subtracted, values still above 100% are set to 100%.

    Args:
        series: Data for relative humidity variable that is corrected
        showplot: Show plot

    Returns:
        Corrected series

    Example:
        See `examples/preprocessing/corrections/correction_relativehumidity_offset.py`
    """

    # print(f"Removing RH offset from {series.name} ...")
    outname = series.name
    # Renamed copy, not an in-place rename: the label is only needed so the
    # quickplot panels distinguish input from output, and renaming the parameter
    # would leave the caller's own series called "input_data". Same pattern as
    # _nighttime_zero_offset below.
    work = series.rename("input_data")

    # Detect series data that exceeds 100% relative humidity
    _series_exceeds = work.loc[work > 100]
    _series_exceeds = _series_exceeds.rename("input_data_exceeds_100")
    exceeds_ix = _series_exceeds.index

    # Calculate daily mean of values > 100
    _daily_mean_above_100 = frames.aggregated_as_hires(aggregate_series=_series_exceeds,
                                                       to_freq='D',
                                                       to_agg='mean',
                                                       hires_timestamp=work.index,
                                                       interpolate_missing_vals=True)

    # Calculate and gap-fill offset values
    _offset = _daily_mean_above_100.sub(100)  # Offset is the difference to 100
    if _offset.dropna().empty:
        _offset.loc[:] = 0  # Set offset to zero
    _offset = _offset.interpolate().ffill().bfill()
    # _offset = _offset.interpolate().ffill().bfill()
    _offset = _offset.rename("offset")

    # Subtract offset from relative humidity (RH) column (series - offset)
    # Offset examples assuming measured RH is 130:
    #   In case of positive offset +30:
    #       (RH must not be larger than 100% but 130% were measured)
    #       130 - (+30) = 100
    # Corrected RH is most likely not *exactly* 100%, but closer to it.
    _series_corr = work.sub(_offset)

    # Set maximum to 100
    series_corr_max100 = _series_corr.copy()
    still_above_100_locs = series_corr_max100 > 100
    series_corr_max100.loc[still_above_100_locs] = 100
    series_corr_max100 = series_corr_max100.rename("corrected_by_offset_and_max100")

    # Plot
    if showplot:
        quickplot([work, _series_exceeds,
                   _daily_mean_above_100, _offset, _series_corr, series_corr_max100],
                  subplots=True,
                  showplot=showplot, hline=100,
                  title=f"Remove RH offset from {outname}")

    # Rename for output
    series_corr_max100 = series_corr_max100.rename(outname)

    return series_corr_max100


@dataclass
class NighttimeZeroOffsetResult:
    """Intermediate series + below-zero stats of the nighttime zero-offset
    correction, for inspection/plotting.

    ``corrected`` is identical to :func:`remove_nighttime_zero_offset`'s output.

    Attributes:
        input: Original (uncorrected) series.
        offset: Daily nighttime-mean offset broadcast to the hires timestamps
            (days without nighttime data filled with the median daily offset,
            0 when there is no nighttime data at all).
        corrected_by_offset: ``input - offset``, before nighttime is forced to
            zero and remaining negatives are clamped.
        corrected: Final corrected series (nighttime forced to zero, remaining
            negative values clamped to zero when ``clamp_negatives``).
        nighttime_flag: 1 where nighttime, 0 where daytime.
        n_below_zero_before: Records below zero in ``input``.
        n_below_zero_before_night: Nighttime records below zero in ``input``.
        n_below_zero_after: Records below zero in ``corrected`` (expected 0 when
            ``clamp_negatives``).
        n_below_zero_after_night: Nighttime records below zero in ``corrected``
            (expected 0).
        n_night: Number of nighttime records.
    """
    input: Series
    offset: Series
    corrected_by_offset: Series
    corrected: Series
    nighttime_flag: Series
    n_below_zero_before: int
    n_below_zero_before_night: int
    n_below_zero_after: int
    n_below_zero_after_night: int
    n_night: int


def _midnight_shift_zeros(swinpot: Series, nighttime: Series, lat: float, lon: float,
                          utc_offset: int, period_min: int | None = None,
                          nighttime_threshold: float = 0.001,
                          period_sec: Series | None = None) -> Series:
    """Night records that are night only because of the zeros ``potrad`` leaves at midnight.

    ``potrad`` (faithful ONEFlux ``get_rpot``) shifts each calendar day's curve
    onto solar noon and sets the minutes vacated by the shift to 0. These minutes
    sit at the start of the calendar day (solar noon after 12:00) or at its end
    (before 12:00). Where the sun is up then, as in polar day, a record that lies
    in them gets too low a SW_IN_POT and can fall below *nighttime_threshold*.

    For each night record this function finds the vacated minutes in its
    averaging window and adds the curve the sun follows there: the neighbouring
    day's curve, continued across midnight. A record is returned if that sum
    reaches *nighttime_threshold*. Away from polar day the sun is down at
    midnight, the added curve is 0 and nothing changes.

    The window of a record is the one ``potrad`` averages: ``period_min`` minutes
    centred on the timestamp. Without ``period_min`` (a *swinpot* given by the
    caller), each record's period is its value in *period_sec* (seconds, e.g.
    meteo screening's FREQ_AUTO_SEC), else its spacing to the nearer
    neighbouring record. The spacing is wrong for the last coarse record before
    a finer block, so pass *period_sec* where the records have several time
    resolutions. A record with a period longer than the grid of *swinpot* is
    taken to end half a grid step after its timestamp, as on meteo screening's
    grids with several time resolutions. Records with missing *swinpot* are not
    records.

    Returns:
        Boolean Series on the index of *swinpot*, True for such records.
    """
    found = pd.Series(False, index=swinpot.index)
    valid = swinpot.notna().to_numpy()
    candidate = valid & (nighttime.to_numpy() == 1)
    if not candidate.any():
        return found
    minute_ns = 60_000_000_000
    ns = swinpot.index.as_unit('ns').asi8
    if period_min is None:
        period = np.zeros(len(ns), dtype=np.int64)
        if valid.sum() >= 2:
            spacing = np.diff(ns[valid])
            period[valid] = np.minimum(np.r_[spacing[0], spacing], np.r_[spacing, spacing[-1]])
        if period_sec is not None:
            own = period_sec.reindex(swinpot.index).to_numpy(dtype=float) * 1e9
            given = np.isfinite(own) & (own > 0)
            period = np.where(given, np.round(np.where(given, own, 0)), period).astype(np.int64)
        # Records without a period (a single record and no period_sec) are skipped
        candidate &= period > 0
        if not candidate.any():
            return found
        grid = np.diff(ns).min() if len(ns) > 1 else period[candidate][0]
        middle = ns + (grid - period) // 2
        period_minutes = np.maximum(1, np.round(period / minute_ns)).astype(np.int64)
    else:
        middle = ns
        period_minutes = np.full(len(ns), period_min, dtype=np.int64)

    rows = np.flatnonzero(candidate)
    middle, period_minutes = middle[rows], period_minutes[rows]
    # Window start on potrad's 1-minute grid, rounded half up as in potrad
    starts = middle - period_minutes * (minute_ns // 2)
    epoch = pd.Timestamp(int(starts.min()), tz=swinpot.index.tz).normalize() - pd.Timedelta(days=1)
    start_off = (starts - epoch.as_unit('ns').value + minute_ns // 2) // minute_ns
    n_days = int((start_off.max() + period_minutes.max()) // 1440) + 2
    dates = pd.date_range(epoch, periods=n_days, freq='D')
    # potrad's own sign convention (positive west), see _oneflux_window_mean
    shifts = np.array([_noaa_solar_noon_minutes(d.year, d.month, d.day, -lon, -utc_offset) - 720
                       for d in dates], dtype=np.int64)
    day_rows = np.array([d.dayofyear - 1 for d in dates], dtype=np.int64)

    added = np.zeros(len(rows))
    for k in range(int(period_minutes.max())):
        minute = start_off + k
        day = minute // 1440
        i = (minute % 1440) - shifts[day]
        vacated = ((i < 0) | (i >= 1440)) & (k < period_minutes)
        if not vacated.any():
            continue
        # The vacated minute on the neighbouring day's curve, only where vacated
        sel = np.flatnonzero(vacated)
        day, i = day[sel], i[sel]
        day = np.where(i < 0, day - 1, np.where(i >= 1440, day + 1, day))
        i = np.where(i < 0, i + 1440, np.where(i >= 1440, i - 1440, i))
        added[sel] += _oneflux_daily_rpot(lat, day_rows[day] + i / 1440.0 + 1.0, i / 60.0)
    true_swinpot = swinpot.to_numpy()[rows] + added / period_minutes
    found.iloc[rows[true_swinpot >= nighttime_threshold]] = True
    return found


def _nighttime_zero_offset(series: Series,
                           lat: float,
                           lon: float,
                           utc_offset: int,
                           clamp_negatives: bool = True,
                           swin_pot: Series | None = None,
                           swin_pot_period_sec: Series | None = None) -> NighttimeZeroOffsetResult:
    """Compute the nighttime zero-offset correction and all intermediate series.

    Shared by :func:`remove_nighttime_zero_offset` (returns only the final series)
    and :func:`nighttime_zero_offset_diagnostics` (returns the full result). Does
    not mutate *series*. When ``clamp_negatives`` is False, daytime negatives that
    remain after the offset is removed are kept instead of being clamped to zero
    (nighttime is forced to zero either way). *swin_pot*, when given, replaces
    the potential radiation computed from *lat*, *lon* and *utc_offset*. It must
    cover the index of *series* and may hold further records, which only help to
    find each record's period (see :func:`_midnight_shift_zeros`).
    *swin_pot_period_sec* gives the period of the records in *swin_pot*.
    """
    outname = series.name
    work = series.rename("input_data")

    # Detect nighttime
    period_min = None
    if swin_pot is None:
        swin_pot = potrad(timestamp_index=work.index, lat=lat, lon=lon, utc_offset=utc_offset)
        # The averaging period potrad infers
        period_min = max(1, int(round(pd.Series(work.index).diff().median().total_seconds() / 60)))
    elif not work.index.isin(swin_pot.index).all():
        raise ValueError("swin_pot must cover every timestamp of series.")
    _, nighttimeflag = daytime_nighttime_flag_from_swinpot(swinpot=swin_pot,
                                                           nighttime_threshold=0.001)
    # Where the sun is up at midnight, potrad's midnight zeros are daytime: neither
    # zeroed nor used as that day's offset (L244, L266)
    artifact = _midnight_shift_zeros(swinpot=swin_pot, nighttime=nighttimeflag, lat=lat,
                                     lon=lon, utc_offset=utc_offset, period_min=period_min,
                                     nighttime_threshold=0.001, period_sec=swin_pot_period_sec)
    nighttimeflag = nighttimeflag.mask(artifact, 0.0).reindex(work.index)
    nighttime_ix = nighttimeflag == 1

    # Calculate offset as the daily nighttime mean, broadcast to hires; days
    # without nighttime data fall back to the median daily offset.
    series_nighttime = work.loc[nighttime_ix]
    _offset = frames.aggregated_as_hires(aggregate_series=series_nighttime,
                                         to_freq='D',
                                         to_agg='mean',
                                         hires_timestamp=work.index,
                                         interpolate_missing_vals=True)
    _offset = _offset.fillna(_offset.median()).rename("offset")
    # No nighttime value at all (e.g. a period of polar day): nothing to derive
    # an offset from, so none is removed. Without this every record became NaN.
    _offset = _offset.fillna(0.0)

    # Subtract offset from the variable (value - offset). Offset examples
    # assuming a measured value of 120: negative offset -8 -> 120 - (-8) = 128;
    # positive offset +8 -> 120 - (+8) = 112.
    _series_corr = work.sub(_offset).rename("corrected_by_offset")

    # Force nighttime to zero, then optionally clamp remaining (daytime) negatives.
    final = _series_corr.copy()
    final.loc[nighttime_ix] = 0
    if clamp_negatives:
        final.loc[final < 0] = 0

    return NighttimeZeroOffsetResult(
        input=work.rename(outname),
        offset=_offset,
        corrected_by_offset=_series_corr,
        corrected=final.rename(outname),
        nighttime_flag=nighttimeflag,
        n_below_zero_before=int((work < 0).sum()),
        n_below_zero_before_night=int((work[nighttime_ix] < 0).sum()),
        n_below_zero_after=int((final < 0).sum()),
        n_below_zero_after_night=int((final[nighttime_ix] < 0).sum()),
        n_night=int(nighttime_ix.sum()),
    )


def remove_nighttime_zero_offset(series: Series,
                                 lat: float,
                                 lon: float,
                                 utc_offset: int,
                                 showplot: bool = False,
                                 clamp_negatives: bool = True,
                                 swin_pot: Series | None = None,
                                 swin_pot_period_sec: Series | None = None) -> Series:
    """Remove a nighttime zero-offset from a variable that should read zero at night

    General-purpose: works for any variable expected to be zero during nighttime
    (e.g. shortwave radiation, PPFD). A nighttime flag is calculated and used to
    detect nighttime data. Then, for each day, the mean of that day's nighttime
    values is the offset by which all of the day's records are corrected (days
    without nighttime data use the median of all daily offsets, and with no
    nighttime value at all the offset is 0). Finally, after the offset is
    applied, nighttime values are set to zero and (when ``clamp_negatives``) any
    remaining negative values are clamped to zero, enforcing the variable's
    physical floor of zero.

    Nighttime is where potential radiation is below 0.001 W m-2. ``potrad`` gives
    0 for the minutes its solar-noon shift vacates at the start or end of each
    calendar day. Where the sun is up then (polar day, or a short night next to
    midnight), a record in these minutes counts as daytime if the sun's curve
    across midnight brings its potential radiation to 0.001 W m-2 or more. It is
    then not zeroed and gives no offset. A polar day has no nighttime data and
    gets the median offset, or 0 if no day in the series has nighttime data. The
    offset cannot be estimated in polar day.

    Args:
        series: Data for the variable that is corrected
        lat: Latitude of the location where data were recorded
        lon: Longitude of the location where data were recorded
        utc_offset: UTC offset of the timestamp index of *series*, e.g. 1 for
            UTC+01:00. Used to find nighttime; the returned Series keeps the
            index of *series*.
        showplot: Show plot
        clamp_negatives: If True (default), set remaining negative values (daytime
            included) to zero after the offset is removed, enforcing a physical
            floor of zero. If False, keep those negatives (nighttime is still
            forced to zero).
        swin_pot: Potential radiation (W m-2) aligned to the index of *series*,
            used for the day/night split instead of computing it from *lat*,
            *lon* and *utc_offset*. These still give the midnight zeros
            described above. Pass it when the potential radiation must be
            computed differently, e.g. per time resolution.
            It must contain every timestamp of *series* and may contain more.
            Timestamps are record middles. Each record's period is its value
            in *swin_pot_period_sec*, else its spacing to the nearer
            neighbouring record. A record with a period longer than the grid
            of *swin_pot* is taken to end half a grid step after its
            timestamp, as on meteo screening's grids with several time
            resolutions. NaN means not a record. None (default) computes it
            with ``potrad``.
        swin_pot_period_sec: Seconds each record of *swin_pot* averages, on
            its index (e.g. meteo screening's FREQ_AUTO_SEC). Only used with
            *swin_pot*, to find the midnight zeros described above. NaN or a
            missing timestamp falls back to the spacing to the nearer
            neighbouring record. That spacing is too short for the last
            coarse record before a finer block, so pass this when the records
            have several time resolutions. None (default) uses the spacing.

    Returns:
        Corrected series

    Example:
        See `examples/preprocessing/corrections/correction_radiation_offset.py`
    """
    result = _nighttime_zero_offset(series, lat=lat, lon=lon, utc_offset=utc_offset,
                                    clamp_negatives=clamp_negatives, swin_pot=swin_pot,
                                    swin_pot_period_sec=swin_pot_period_sec)

    if showplot:
        # Named per stage: the raw and the final series otherwise share the variable
        # name, which makes the panels ambiguous about which stage they show.
        varname = result.input.name
        quickplot([result.input.rename(f"{varname} (measured)"),
                   result.corrected_by_offset.rename("after offset subtraction"),
                   result.corrected.rename("after nighttime zeroing + clamping"),
                   result.offset.rename("daily offset")],
                  subplots=True,
                  showplot=showplot,
                  title=f"Removing nighttime zero-offset from {varname}")

    return result.corrected


def nighttime_zero_offset_diagnostics(series: Series,
                                      lat: float,
                                      lon: float,
                                      utc_offset: int,
                                      clamp_negatives: bool = True,
                                      swin_pot: Series | None = None,
                                      swin_pot_period_sec: Series | None = None) -> NighttimeZeroOffsetResult:
    """Nighttime zero-offset correction with every intermediate series exposed.

    Same computation as :func:`remove_nighttime_zero_offset` (``result.corrected``
    is identical), but additionally returns the daily offset, the offset-subtracted
    series, the nighttime flag, and the count of below-zero records before/after
    the correction (overall and nighttime) — for inspection and plotting.

    Args:
        series: Data for the variable that is corrected.
        lat: Latitude of the location where data were recorded.
        lon: Longitude of the location where data were recorded.
        utc_offset: UTC offset of the timestamp index, e.g. 1 for UTC+01:00.
        clamp_negatives: If True (default), clamp remaining negatives to zero (see
            :func:`remove_nighttime_zero_offset`). With False, ``n_below_zero_after``
            reflects the daytime negatives that were kept.
        swin_pot: Potential radiation on the index of *series*, used instead of
            computing it (see :func:`remove_nighttime_zero_offset`).
        swin_pot_period_sec: Seconds each record of *swin_pot* averages (see
            :func:`remove_nighttime_zero_offset`).

    Returns:
        A :class:`NighttimeZeroOffsetResult`.
    """
    return _nighttime_zero_offset(series, lat=lat, lon=lon, utc_offset=utc_offset,
                                  clamp_negatives=clamp_negatives, swin_pot=swin_pot,
                                  swin_pot_period_sec=swin_pot_period_sec)


class WindDirOffset:
    """
    Compare yearly wind direction histograms to reference, detect
    offset in comparison to reference and correct wind directions
    for offset per year

    Example:
        See `examples/preprocessing/corrections/correction_winddir_offset.py`
    """

    def __init__(self,
                 winddir: Series,
                 hist_ref_years: list,
                 offset_start: int = -100,
                 offset_end: int = 100,
                 hist_n_bins: int = 360,
                 verbose: int | bool | None = None):
        """
        Build histogram of wind directions for each year and compare to reference
        histogram built from data in reference years

        (1) Build reference histogram of wind directions from reference years
        (2) For each year:
            2a: Add constant offset to wind directions, starting with *offset_start*
            2b: Build histogram of wind directions
            2c: Calculate absolute correlation between 2b and reference
            2d: Continue with next offset, ending with *offset_end*
            2e: Detect offset that yielded maximum absolute correlation with reference

        Args:
            winddir: Time series of wind directions in degrees
            hist_ref_years: List of years for building reference histogram, e.g. "[2021, 2022]"
            offset_start: Minimum offset in degrees to shift winddir
            offset_end: Maximum offset in degrees to shift winddir
            hist_n_bins: Number of bins for building histograms. The bins always
                span the full compass circle [0, 360), so the default 360 gives
                one bin per degree and e.g. 36 gives one per 10 degrees.
            verbose: Verbosity for the console helpers. The line printed for each
                year appears at ``VERBOSE_DEBUG`` (3). None uses the level set
                with ``set_verbosity()``.
        """
        self.winddir = winddir
        self.hist_ref_years = hist_ref_years
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.hist_n_bins = hist_n_bins
        self.verbose = verbose

        # Wind directions shifted by offset that yielded maximum absolute
        # correlation with reference
        self.winddir_shifted = self.winddir.copy()

        # Unique years
        self.uniq_years = list(self.winddir.index.year.unique())

        # Reference histogram
        self.ref_results = self._reference_histogram()

        # Find offset per year and store results in dict
        self.shiftdict = self._calc_histogram_correlations()

        # Detect offset per year
        self.yearlyoffsets_df = self._find_yearly_offsets()

        # Correct wind directions
        self._correct_wind_directions()

    def get_corrected_wind_directions(self):
        """Return the offset-corrected wind direction series."""
        return self.winddir_shifted

    def get_yearly_offsets(self) -> DataFrame:
        """Return yearly wind direction offsets that yielded maximum absolute
        correlation with reference"""
        return self.yearlyoffsets_df

    def _correct_wind_directions(self):
        """Correct wind directions by yearly offsets"""
        for year in self.uniq_years:
            offset = int(self.yearlyoffsets_df.loc[self.yearlyoffsets_df['YEAR'] == year, 'OFFSET'].iloc[0])
            self.winddir_shifted.loc[self.winddir_shifted.index.year == year] += offset
        self.winddir_shifted = self._correct_degrees(self.winddir_shifted)

    def _find_yearly_offsets(self):
        yearlyoffsets_df = pd.DataFrame(columns=['YEAR', 'OFFSET'])
        for key, val in self.shiftdict.items():
            val_sorted = val.sort_values(by='CORR_ABS', ascending=False).copy()
            shift_maxcorr = val_sorted.iloc[0]['SHIFT']
            yearlyoffsets_df.loc[len(yearlyoffsets_df)] = [key, shift_maxcorr]
        return yearlyoffsets_df

    def _calc_histogram_correlations(self):
        """ """
        shiftdict = {}
        for year in self.uniq_years:
            detail(f"Working on year {year} ...", verbose=self.verbose)
            s_year = self.winddir.loc[self.winddir.index.year == year].copy()
            shiftdf_year = pd.DataFrame(columns=['SHIFT', 'CORR_ABS'])
            for shift in np.arange(self.offset_start, self.offset_end, 1):
                s_year_shifted = s_year.copy()
                s_year_shifted += shift
                s_year_shifted = self._correct_degrees(s=s_year_shifted)
                # hist_n_bins, not a hardcoded 360: with the two histograms built
                # at different bin counts their COUNTS series had different
                # lengths, and .corr() silently aligned them on the RangeIndex and
                # correlated the overlap only.
                counts_shifted = self._wind_histogram(s_year_shifted)
                corr_abs = abs(pd.Series(counts_shifted).corr(pd.Series(self.ref_results)))
                shiftdf_year.loc[len(shiftdf_year)] = [shift, corr_abs]
            shiftdict[year] = shiftdf_year
        return shiftdict

    @staticmethod
    def _correct_degrees(s: Series) -> Series:
        """Wrap degree values back onto the compass circle [0, 360).

        Wind direction is circular: 360 deg is the same direction as 0 deg. The
        modulo handles any shift magnitude (the previous single add/subtract only
        covered one wrap, so an offset range wider than +/-360 came out wrong) and
        maps an exact 360 to 0, keeping the domain half-open.
        """
        return s % 360

    def _wind_histogram(self, s: Series) -> np.ndarray:
        """Direction counts per bin over the full circle.

        The bin edges are fixed at ``linspace(0, 360, hist_n_bins + 1)`` rather
        than derived from the data. Both matter:

        - The offsets are chosen by correlating this histogram against the
          reference *bin by bin*, which is only meaningful if bin i covers the
          same directions in both. Data-derived edges (numpy's default for an
          integer ``bins``) differ between years and after every shift.
        - The domain is the half-open [0, 360) the compass actually has, so the
          wrap at north is not a discontinuity in the binning.
        """
        edges = np.linspace(0, 360, self.hist_n_bins + 1)
        counts, _ = np.histogram(s.dropna().to_numpy(), bins=edges)
        return counts

    def _reference_histogram(self):
        """Calculate reference histogram"""
        select_years = self.winddir.index.year.isin(self.hist_ref_years)
        ref_s = self._correct_degrees(self.winddir[select_years])
        return self._wind_histogram(ref_s)
