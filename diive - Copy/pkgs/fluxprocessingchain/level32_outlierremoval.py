from pandas import DataFrame

from diive.pkgs.outlierdetection.lof import LocalOutlierFactorDaytimeNighttime
from diive.pkgs.outlierdetection.seasonaltrend import OutlierSTLRIQRZ
from diive.pkgs.outlierdetection.zscore import zScoreDaytimeNighttime


class OutlierRemovalLevel32():

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 site_lat: float = None,
                 site_lon: float = None,
                 **kwargs):
        self._fulldf = df.copy()
        self.fluxcol = fluxcol
        self.site_lat = site_lat
        self.site_lon = site_lon
        self.kwargs = kwargs

        self.series = self._fulldf[self.fluxcol].copy()
        self.series_cleaned = self._fulldf[self.fluxcol].copy()
        self.swinpot = self._fulldf['SW_IN_POT'].copy()  # Is available in fluxnet files

        self._last_result = None

        # todo use metadata_df?
        # todo use SWINPOT

    @property
    def fulldf(self) -> DataFrame:
        """Return all data with new flag columns added"""
        if not isinstance(self._fulldf, DataFrame):
            raise Exception('No fit results available.')
        return self._fulldf

    def stl_iqrz(self, zfactor: float = 4.5, decompose_downsampling_freq:str='1H',
                 repeat: bool = False, showplot: bool = False):
        """Seasonsal trend decomposition with z-score on residuals"""
        _stl = OutlierSTLRIQRZ(series=self.series_cleaned, lat=self.site_lat, lon=self.site_lon, levelid='3.2')
        _stl.calc(zfactor=zfactor, decompose_downsampling_freq=decompose_downsampling_freq,
                  repeat=repeat, showplot=showplot)
        self._last_result = _stl

    def zscore_dtnt(self, threshold: float = 4, showplot: bool = False, verbose: bool = False):
        """z-score, calculated separately for daytime and nighttime"""
        # self.preview_series = self.series_cleaned.copy()
        _zscoredtnt = zScoreDaytimeNighttime(series=self.series_cleaned, site_lat=self.site_lat, site_lon=self.site_lon)
        _zscoredtnt.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        self._last_result = _zscoredtnt

    def addflag(self):
        """Add flag of most recent outlier test to data and update filtered series
        that will be used to continue with the next test"""
        flag = self._last_result.flag
        self.series_cleaned = self._last_result.filteredseries
        if flag.name in self._fulldf.columns:
            self._fulldf.drop([flag.name], axis=1, inplace=True)
        self._fulldf[flag.name] = flag
        print(f"++Added column {flag.name} to data")

    def lof_dtnt(self, n_neighbors: int = None, contamination: float = 'auto',
                 showplot: bool = False, verbose: bool = False):
        """Local outlier factor, separately for daytime and nighttime data"""
        # Number of neighbors is automatically calculated if not provided
        n_neighbors = int(len(self.series_cleaned.dropna()) / 100) if not n_neighbors else n_neighbors

        # Contamination is set automatically unless float is given
        contamination = contamination if isinstance(contamination, float) else 'auto'

        _lof = LocalOutlierFactorDaytimeNighttime(series=self.series_cleaned, site_lat=self.site_lat,
                                                  site_lon=self.site_lon)
        _lof.calc(n_neighbors=n_neighbors, contamination=contamination, showplot=showplot, verbose=verbose)
        self._last_result = _lof

    def run(self):
        pass


def example():
    pass


if __name__ == '__main__':
    example()
