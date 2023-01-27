from pandas import Series, DataFrame

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

        # todo use metadata_df?
        # todo use SWINPOT

    @property
    def fulldf(self) -> DataFrame:
        """Return all data with new flag columns added"""
        if not isinstance(self._fulldf, DataFrame):
            raise Exception('No fit results available.')
        return self._fulldf

    def stl_iqrz(self, zfactor: float = 4.5, repeat: bool = False, showplot: bool = False):
        """Seasonsal trend decomposition with z-score on residuals"""
        _stl = OutlierSTLRIQRZ(series=self.series_cleaned, lat=self.site_lat, lon=self.site_lon, levelid='3.2')
        _stl.calc(zfactor=zfactor, repeat=repeat, showplot=showplot)
        flag = _stl.flag
        self.series_cleaned = _stl.filteredseries
        self._addflag(flag=flag)

    def zscore_dtnt(self, threshold: float = 4, showplot: bool = False, verbose: bool = False):
        """z-score, calculated separately for daytime and nighttime"""
        _zscoredtnt = zScoreDaytimeNighttime(series=self.series_cleaned, site_lat=self.site_lat, site_lon=self.site_lon)
        _zscoredtnt.calc(threshold=threshold, showplot=showplot, verbose=verbose)
        flag = _zscoredtnt.flag
        self.series_cleaned = _zscoredtnt.filteredseries
        self._addflag(flag=flag)

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
        flag = _lof.flag
        self.series_cleaned = _lof.filteredseries
        self._addflag(flag=flag)

    def _addflag(self, flag: Series):
        if flag.name in self._fulldf.columns:
            self._fulldf.drop([flag.name], axis=1, inplace=True)
        self._fulldf[flag.name] = flag
        print(f"++Added column {flag.name} to data")

        # _test = _zscoredtnt.filteredseries
        # _test.plot(ls='none', markersize=3, marker='o', alpha=.5)
        # plt.show()

    def run(self):
        pass


def example():
    pass


if __name__ == '__main__':
    example()
