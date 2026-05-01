import decimal

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import diive.core.dfun.frames as frames
from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.analyses.histogram import Histogram
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


class MeasurementOffsetFromReplicate:
    """
    Compare yearly measurement histograms to reference, detect
    offset and gain in comparison to reference and correct measurement
    for offset and gain per year

    Example:
        See `examples/corrections/offsetcorrection.py` for complete examples.
    """

    def __init__(self,
                 measurement: Series,
                 replicate: Series,
                 offset_start: int = -100,
                 offset_end: int = 100,
                 offset_stepsize: float = 0.1):
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
            measurement: Time series of wind directions in degrees
            hist_ref_years: List of years for building reference histogram, e.g. "[2021, 2022]"
            offset_start: Minimum offset in degrees to shift *s*
            offset_end: Maximum offset in degrees to shift *s*
            hist_n_bins: Number of bins for building histograms
        """
        self.measurement = measurement
        self.replicate = replicate
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.offset_stepsize = offset_stepsize

        d = decimal.Decimal(str(offset_stepsize))
        self.n_digits_after_comma = abs(d.as_tuple().exponent)

        # Wind directions shifted by offset that yielded maximum absolute
        # correlation with reference
        self.measurement_shifted = self.measurement.copy()

        # Find offset per year and store results in dict
        self.offset = self._calc_shift_correlations()

        # Correct wind directions
        self.replicate_corrected = self._correct_measurement()

    def get_corrected_measurement(self):
        return self.replicate_corrected

    def get_offset(self):
        return self.offset

    def _correct_measurement(self):
        """Correct wind directions by yearly offsets"""
        measurement_corrected = self.measurement.add(self.offset)
        return measurement_corrected

    def _calc_shift_correlations(self):
        """ """
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

            print(f"#{counter}   trying with offset: {offset:.{self.n_digits_after_comma}f}   "
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
        See `examples/corrections/offsetcorrection.py` for complete examples.
    """

    # print(f"Removing RH offset from {series.name} ...")
    outname = series.name
    series.name = "input_data"

    # Detect series data that exceeds 100% relative humidity
    _series_exceeds = series.loc[series > 100]
    _series_exceeds.rename("input_data_exceeds_100", inplace=True)
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
    _offset.rename("offset", inplace=True)

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
    series_corr_max100.rename("corrected_by_offset_and_max100", inplace=True)

    # Plot
    if showplot:
        quickplot([series, _series_exceeds,
                   _daily_mean_above_100, _offset, _series_corr, series_corr_max100],
                  subplots=True,
                  showplot=showplot, hline=100,
                  title=f"Remove RH offset from {outname}")

    # Rename for output
    series_corr_max100.rename(outname, inplace=True)

    return series_corr_max100


@ConsoleOutputDecorator()
def remove_radiation_zero_offset(series: Series,
                                 lat: float,
                                 lon: float,
                                 utc_offset: int,
                                 showplot: bool = False) -> Series:
    """Correct nighttime offset from all radiation data and set nighttime to zero

    Works for radiation data where radiation should be zero during nighttime.

    Nighttime flag is calculated and used to detect nighttime data. Then,
    for each radiation variable, the nighttime radiation mean is calculated for
    each day in the dataset. This mean is the offset by which daytime and nighttime
    radiation data are corrected. Finally, after the application of the offset,
    nighttime data are set to zero.

    Args:
        series: Data for radiation variable that is corrected
        lat: Latitude of the location where radiation data were recorded
        lon: Longitude of the location where radiation data were recorded
        utc_offset: UTC offset of *timestamp_index*, e.g. 1 for UTC+01:00
            The datetime index of the resulting Series will be in this timezone.
        showplot: Show plot

    Returns:
        Corrected series

    Example:
        See `examples/corrections/offsetcorrection.py` for complete examples.
    """

    outname = series.name
    series.name = "input_data"

    # Detect nighttime
    dnf = DaytimeNighttimeFlag(
        timestamp_index=series.index,
        nighttime_threshold=0.001,
        lat=lat,
        lon=lon,
        utc_offset=utc_offset)
    nighttimeflag = dnf.get_nighttime_flag()
    # daytime = dnf.get_daytime_flag()

    # # Debug
    # import diive as dv
    # hm = dv.heatmap_datetime(series=series)
    # hm.show()

    nighttime_ix = nighttimeflag == 1

    # Get series nighttime data
    series_nighttime = series.loc[nighttime_ix]
    nighttime_datetimes = series_nighttime.index

    # Calculate offset as the daily nighttime mean
    _offset = frames.aggregated_as_hires(aggregate_series=series_nighttime,
                                         to_freq='D',
                                         to_agg='mean',
                                         hires_timestamp=series.index,
                                         interpolate_missing_vals=True)

    # from diive.core.plotting.timeseries import TimeSeries
    # TimeSeries(series=_offset).plot()

    # Gap-fill offset values
    _offset = _offset.fillna(_offset.median())
    # offset = offset.interpolate().ffill().bfill()
    _offset.rename("offset", inplace=True)

    # Subtract offset from radiation column (rad_col - offset)
    # Offset examples assuming measured radiation is 120:
    #   In case of negative offset -8:
    #       (measured radiation should be zero but -8 was measured)
    #       120 - (-8) = 128
    #   In case of positive offset +8:
    #       (measured radiation should be zero but +8 was measured)
    #       120 - (+8) = 112
    _series_corr = series.sub(_offset)
    _series_corr.rename("corrected_by_offset", inplace=True)

    # Set nighttime radiation to zero (based on potential radiation)
    series_corr_settozero = _series_corr.copy()
    series_corr_settozero.loc[nighttime_ix] = 0
    series_corr_settozero.rename("corrected_by_offset_and_night_settozero", inplace=True)

    # Set still remaining negative values to zero
    series_corr_settozero.loc[series_corr_settozero < 0] = 0

    # Plot
    if showplot:
        quickplot([series, _series_corr,
                   series_corr_settozero, _offset],
                  subplots=True,
                  showplot=showplot,
                  title=f"Removing radiation zero-offset from {outname}")

    # Rename for output
    series_corr_settozero.rename(outname, inplace=True)

    return series_corr_settozero


class WindDirOffset:
    """
    Compare yearly wind direction histograms to reference, detect
    offset in comparison to reference and correct wind directions
    for offset per year

    Example:
        See `examples/corrections/offsetcorrection.py` for complete examples.
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
            offset = int(self.yearlyoffsets_df.loc[self.yearlyoffsets_df['YEAR'] == year]['OFFSET'].values)
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
            print(f"Working on year {year} ...")
            s_year = self.winddir.loc[self.winddir.index.year == year].copy()
            shiftdf_year = pd.DataFrame(columns=['SHIFT', 'CORR_ABS'])
            for shift in np.arange(self.offset_start, self.offset_end, 1):
                s_year_shifted = s_year.copy()
                s_year_shifted += shift
                s_year_shifted = self._correct_degrees(s=s_year_shifted)
                histo_shifted = Histogram(s=s_year_shifted, method='n_bins', n_bins=360)
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
        ref_histo = Histogram(s=ref_s, method='n_bins', n_bins=self.hist_n_bins)
        ref_results = ref_histo.results
        return ref_results
