"""
OUTLIER DETECTION: HAMPEL TEST
==============================

This module is part of the diive library:
https://github.com/holukas/diive

"""

from pandas import DatetimeIndex, Series
from sktime.transformations.series.outlier_detection import HampelFilter

from diive.core.base.flagbase import FlagBase
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class Hampel(FlagBase):
    flagid = 'OUTLIER_HAMPEL'

    def __init__(self,
                 series: Series,
                 window_length: int = 10,
                 n_sigma: float = 5,
                 k: float = 1.4826,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Identify outliers in a sliding window based on the Hampel filter.

        The Hampel filter employs a moving window and utilizes the Median Absolute Deviation (MAD)
        as a measure of data variability. MAD is calculated by taking the median of the absolute
        differences between each data point and the median of the moving window.

        kudos:
        - https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.outlier_detection.HampelFilter.html
        - https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
        - https://medium.com/@miguel.otero.pedrido.1993/hampel-filter-with-python-17db1d265375

        Reference:
            Hampel F. R., “The influence curve and its role in robust estimation”,
            Journal of the American Statistical Association, 69, 382-393, 1974

        Args:
            series: Time series in which outliers are identified.
            idstr: Identifier, added as suffix to output variable names.
            window_length: Size of sliding window.
            winsize_min_periods: Minimum number of records in the time window.
            n_sigma: Number of standard deviations. Records with sd outside this value
                are flagged as outliers.
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.

        Returns:
            Flag series that combines flags from all iterations in one single flag.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        self.showplot = False
        self.verbose = False
        self.n_sigma = n_sigma
        self.window_length = window_length
        self.showplot = showplot
        self.verbose = verbose
        self.k = k  # Scale factor for Gaussian distribution

        # if self.showplot:
        #     self.fig, self.ax, self.ax2 = self._plot_init()

    def calc(self, repeat: bool = True):
        """Calculate overall flag, based on individual flags from multiple iterations.

        Args:
            repeat: If *True*, the outlier detection is repeated until all
                outliers are removed.

        """

        self._overall_flag, n_iterations = self.repeat(self.run_flagtests, repeat=repeat)
        if self.showplot:
            # Default plot for outlier tests, showing rejected values
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Perform tests required for this flag"""

        # Working data
        s = self.filteredseries.copy()
        s = s.dropna()

        transformer = HampelFilter(window_length=self.window_length,
                                   n_sigma=self.n_sigma,
                                   k=self.k,
                                   return_bool=True)

        is_outlier = transformer.fit_transform(s)

        ok = is_outlier == False
        ok = ok[ok].index
        rejected = is_outlier == True
        rejected = rejected[rejected].index

        n_outliers = len(rejected)

        if self.verbose:
            if self.verbose:
                print(f"ITERATION#{iteration}: Total found outliers: {len(rejected)} values")

        return ok, rejected, n_outliers


def example():
    import importlib.metadata
    import diive.configs.exampledata as ed
    from diive.pkgs.createvar.noise import add_impulse_noise
    from diive.core.plotting.timeseries import TimeSeries
    import warnings
    warnings.filterwarnings('ignore')
    version_diive = importlib.metadata.version("diive")
    print(f"diive version: v{version_diive}")
    df = ed.load_exampledata_parquet()

    # # Only nighttime data
    # keep = df['Rg_f'] < 50
    # df = df[keep].copy()

    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()

    s_noise = add_impulse_noise(series=s,
                                factor_low=-10,
                                factor_high=4,
                                contamination=0.04,
                                seed=42)  # Add impulse noise (spikes)
    s_noise.name = f"{s.name}+noise"
    TimeSeries(s_noise).plot()

    lsd = Hampel(
        series=s_noise,
        n_sigma=4,
        window_length=48 * 9,
        showplot=True,
        verbose=True
    )

    lsd.calc(repeat=True)


if __name__ == '__main__':
    example()
    # example_dtnt()
