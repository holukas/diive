import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import decimal
import diive.core.dfun.frames as frames
from diive.core.plotting.plotfuncs import quickplot
from diive.core.utils.prints import ConsoleOutputDecorator
from diive.pkgs.createvar.daynightflag import DaytimeNighttimeFlag


class MeasurementOffsetFromReplicate:
    """
    Compare yearly measurement histograms to reference, detect
    offset and gain in comparison to reference and correct measurement
    for offset and gain per year

    - Example notebook available in:
        notebooks/Corrections/WindDirectionOffset.ipynb
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


def example():
    # from dbc_influxdb import dbcInflux
    # SITE = 'ch-cha'
    # MEASUREMENTS = ['TS']
    # FIELDS = ['TS_GF1_0.05_1', 'TS_LOWRES_GF1_0.05_3']
    # DIRCONF = r'L:\Sync\luhk_work\20 - CODING\22 - POET\configs'
    # BUCKET_RAW = f'{SITE}_processed'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
    # dbc = dbcInflux(dirconf=DIRCONF)
    # data_simple, data_detailed, assigned_measurements = dbc.download(
    #     bucket=BUCKET_RAW,
    #     measurements=MEASUREMENTS,
    #     fields=FIELDS,
    #     start='2022-01-01 00:00:01',
    #     stop='2025-01-01 00:00:01',
    #     timezone_offset_to_utc_hours=1,
    #     data_version='meteoscreening_diive'
    # )
    # print(data_simple)
    # print(data_detailed)
    # print(assigned_measurements)
    # # Export data to parquet for fast testing
    # from diive.core.io.files import save_parquet
    # filepath = save_parquet(filename="meteodata_simple", data=data_simple, outpath=r"F:\TMP")

    from diive.core.io.files import load_parquet
    df = load_parquet(filepath=r"F:\TMP\meteodata_simple.parquet")
    df = df.dropna()
    import matplotlib.pyplot as plt
    # df.plot(x_compat=True)
    # plt.show()
    df.plot(x_compat=True)
    plt.show()

    off = MeasurementOffsetFromReplicate(measurement=df['TS_GF1_0.05_1'],
                                         replicate=df['TS_LOWRES_GF1_0.05_3'],
                                         offset_start=-10, offset_end=0, offset_stepsize=.1)
    corrected = off.get_corrected_measurement()
    offset = off.get_offset()

    print(f"The offset with minimum absolute difference between data points is {offset}")

    # corrected.plot(x_compat=True, label=f"TS_GF1_0.05_1 corrected by offset {offset:.2f}")
    # df['TS_LOWRES_GF1_0.05_3'].plot(x_compat=True, label="TS_LOWRES_GF1_0.05_3 (replicate)")
    # plt.legend(loc='upper right')
    # plt.show()

    # col = 'WD'
    # wd = df[col].copy()
    #
    # # # Prepare input data
    # # wd = wd.loc[wd.index.year <= 2009]
    # # wd = wd.dropna()
    #
    # wds = WindDirOffset(winddir=wd, offset_start=-50, offset_end=50, hist_ref_years=[2006, 2009], hist_n_bins=360)
    # yearlyoffsets_df = wds.get_yearly_offsets()
    # s_corrected = wds.get_corrected_wind_directions()
    # print(yearlyoffsets_df)
    # print(s_corrected)
    # print(wd)
    #
    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=s_corrected).show()
    # HeatmapDateTime(series=wd).show()


if __name__ == '__main__':
    example()
