"""
OFFSET CORRECTION: MEASUREMENT OFFSET AND GAIN DETECTION
=========================================================

Detect and remove systematic offsets and gain changes in measurement pairs via histogram comparison.

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
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.analysis.histogram import Histogram
from diive.variables import DaytimeNighttimeFlag


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
                 offset_stepsize: float = 0.1):
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
        """
        self.measurement = measurement
        self.replicate = replicate
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.offset_stepsize = offset_stepsize

        d = decimal.Decimal(str(offset_stepsize))
        self.n_digits_after_comma = abs(d.as_tuple().exponent)

        # Find the best-fitting constant offset.
        self.offset = self._find_best_offset()

        # Apply it to the measurement.
        self.measurement_corrected = self._correct_measurement()

    def get_corrected_measurement(self):
        return self.measurement_corrected

    def get_offset(self):
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
                   f"found absolute difference: {abs_diff}")
            # fig = plt.figure()
            # r.plot(x_compat=True, label="true measurement")
            # m_shifted.plot(x_compat=True, label=f"corrected by offset")
            # plt.title(f"OFFSET: {offset}  /  SUM_ABS_DIFF: {abs_diff}")
            # plt.legend(loc='upper right')
            # path = rf"C:\Users\nopan\Desktop\temp\{counter}.png"
            # fig.tight_layout()
            # # fig.show()
            # fig.savefig(path)

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


@ConsoleOutputDecorator()
def remove_relativehumidity_offset(series: Series,
                                   showplot: bool = False) -> Series:
    """Correct relative humidity offset > 100%

    Works for relative humidity data where maximum values should not exceed 100%.

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
    series.name = "input_data"

    # Detect series data that exceeds 100% relative humidity
    _series_exceeds = series.loc[series > 100]
    _series_exceeds = _series_exceeds.rename("input_data_exceeds_100")
    exceeds_ix = _series_exceeds.index

    # Calculate daily mean of values > 100
    _daily_mean_above_100 = frames.aggregated_as_hires(aggregate_series=_series_exceeds,
                                                       to_freq='D',
                                                       to_agg='mean',
                                                       hires_timestamp=series.index,
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
    _series_corr = series.sub(_offset)

    # Set maximum to 100
    series_corr_max100 = _series_corr.copy()
    still_above_100_locs = series_corr_max100 > 100
    series_corr_max100.loc[still_above_100_locs] = 100
    series_corr_max100 = series_corr_max100.rename("corrected_by_offset_and_max100")

    # Plot
    if showplot:
        quickplot([series, _series_exceeds,
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
            (days without nighttime data filled with the median daily offset).
        corrected_by_offset: ``input - offset``, before nighttime is forced to
            zero and remaining negatives are clamped.
        corrected: Final corrected series (nighttime forced to zero, remaining
            negative values clamped to zero).
        nighttime_flag: 1 where nighttime, 0 where daytime.
        n_below_zero_before: Records below zero in ``input``.
        n_below_zero_before_night: Nighttime records below zero in ``input``.
        n_below_zero_after: Records below zero in ``corrected`` (expected 0).
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


def _nighttime_zero_offset(series: Series,
                           lat: float,
                           lon: float,
                           utc_offset: int,
                           clamp_negatives: bool = True) -> NighttimeZeroOffsetResult:
    """Compute the nighttime zero-offset correction and all intermediate series.

    Shared by :func:`remove_nighttime_zero_offset` (returns only the final series)
    and :func:`nighttime_zero_offset_diagnostics` (returns the full result). Does
    not mutate *series*. When ``clamp_negatives`` is False, daytime negatives that
    remain after the offset is removed are kept instead of being clamped to zero
    (nighttime is forced to zero either way).
    """
    outname = series.name
    work = series.rename("input_data")

    # Detect nighttime
    dnf = DaytimeNighttimeFlag(
        timestamp_index=work.index,
        nighttime_threshold=0.001,
        lat=lat,
        lon=lon,
        utc_offset=utc_offset)
    nighttimeflag = dnf.get_nighttime_flag()
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


@ConsoleOutputDecorator()
def remove_nighttime_zero_offset(series: Series,
                                 lat: float,
                                 lon: float,
                                 utc_offset: int,
                                 showplot: bool = False,
                                 clamp_negatives: bool = True) -> Series:
    """Remove a nighttime zero-offset from a variable that should read zero at night

    General-purpose: works for any variable expected to be zero during nighttime
    (e.g. shortwave radiation, PPFD). A nighttime flag is calculated and used to
    detect nighttime data. Then, for each day, the mean of that day's nighttime
    values is the offset by which all of the day's records are corrected (days
    without nighttime data use the median of all daily offsets). Finally, after
    the offset is applied, nighttime values are set to zero and (when
    ``clamp_negatives``) any remaining negative values are clamped to zero —
    enforcing the variable's physical floor of zero.

    Args:
        series: Data for the variable that is corrected
        lat: Latitude of the location where data were recorded
        lon: Longitude of the location where data were recorded
        utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
            The datetime index of the resulting Series will be in this timezone.
        showplot: Show plot
        clamp_negatives: If True (default), set remaining negative values (daytime
            included) to zero after the offset is removed, enforcing a physical
            floor of zero. If False, keep those negatives (nighttime is still
            forced to zero).

    Returns:
        Corrected series

    Example:
        See `examples/preprocessing/corrections/correction_radiation_offset.py`
    """
    result = _nighttime_zero_offset(series, lat=lat, lon=lon, utc_offset=utc_offset,
                                    clamp_negatives=clamp_negatives)

    if showplot:
        quickplot([result.input, result.corrected_by_offset,
                   result.corrected, result.offset],
                  subplots=True,
                  showplot=showplot,
                  title=f"Removing nighttime zero-offset from {result.input.name}")

    return result.corrected


def nighttime_zero_offset_diagnostics(series: Series,
                                      lat: float,
                                      lon: float,
                                      utc_offset: int,
                                      clamp_negatives: bool = True) -> NighttimeZeroOffsetResult:
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

    Returns:
        A :class:`NighttimeZeroOffsetResult`.
    """
    return _nighttime_zero_offset(series, lat=lat, lon=lon, utc_offset=utc_offset,
                                  clamp_negatives=clamp_negatives)


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
                 hist_n_bins: int = 360):
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
            hist_n_bins: Number of bins for building histograms
        """
        self.winddir = winddir
        self.hist_ref_years = hist_ref_years
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.hist_n_bins = hist_n_bins

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
            detail(f"Working on year {year} ...")
            s_year = self.winddir.loc[self.winddir.index.year == year].copy()
            shiftdf_year = pd.DataFrame(columns=['SHIFT', 'CORR_ABS'])
            for shift in np.arange(self.offset_start, self.offset_end, 1):
                s_year_shifted = s_year.copy()
                s_year_shifted += shift
                s_year_shifted = self._correct_degrees(s=s_year_shifted)
                histo_shifted = Histogram(series=s_year_shifted, method='n_bins', n_bins=360)
                results_shifted = histo_shifted.results
                corr_abs = abs(results_shifted['COUNTS'].corr(self.ref_results['COUNTS']))
                shiftdf_year.loc[len(shiftdf_year)] = [shift, corr_abs]
            shiftdict[year] = shiftdf_year
        return shiftdict

    def _correct_degrees(self, s: Series):
        """Correct degree values that go below zero or above 360"""
        _locs_above360 = s >= 360
        s[_locs_above360] -= 360
        _locs_belowzero = s < 0
        s[_locs_belowzero] += 360
        return s

    def _reference_histogram(self):
        """Calculate reference histogram"""
        select_years = self.winddir.index.year.isin(self.hist_ref_years)
        ref_s = self.winddir[select_years]
        ref_histo = Histogram(series=ref_s, method='n_bins', n_bins=self.hist_n_bins)
        ref_results = ref_histo.results
        return ref_results
