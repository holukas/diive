"""
NEGATIVE NIGHTTIME: NIGHT VALUES CLEARLY BELOW ZERO
===================================================

Flag nighttime records that lie further below zero than their random uncertainty explains.

At night there is no photosynthesis, so net ecosystem exchange (NEE) is respiration and its true
value is zero or positive. A measured value can still be negative: random noise of size sigma
around a small respiration pushes some records below zero. Such a value is plausible as long as it
lies within a few sigma of zero. A value below ``-z * sigma`` is not: noise alone reaches that far
with a probability of at most Phi(-z) (2.3 % for z = 2, 0.13 % for z = 3), and much less when the
respiration is above zero. The test removes those records and nothing else, so it is one-sided on
purpose. Removing every negative value instead would cut off one side of the noise and push
nighttime NEE up. Daytime records are never tested: by day both signs are real.

The test also estimates how many of the removed records noise alone would have produced, from the
local nighttime level (a rolling median) and each record's uncertainty. If that number is a small
share of the removed records, the test removes errors rather than noise, and the bias it adds to a
sum is at most the expected noise removals.

Quality flags (overall_flag):
  - flag=0: daytime record, or nighttime record not clearly below zero (valid)
  - flag=2: nighttime record below ``-z * uncertainty`` (removed)
  - NaN: record missing in the input, so no test could be performed

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series
from scipy.stats import norm

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags


class NegativeNighttime(FlagBase):
    """Flag nighttime values below ``-z`` times their random uncertainty. See :meth:`__init__`."""

    flagid = 'OUTLIER_NEGNIGHT'

    def __init__(self,
                 series: Series,
                 uncertainty: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 z: float = 2.0,
                 level_window: int | str = '13D',
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Set up the test.

        Args:
            series: Flux in which nighttime values clearly below zero are flagged, e.g. NEE.
            uncertainty: Random uncertainty of each record, in the units of ``series`` and on its
                index, e.g. EddyPro's ``FC_RANDUNC_HF``, the PAS20 random uncertainty, or the
                larger of the two. A nighttime record without an uncertainty is not tested.
            lat: Latitude of the site, to tell day from night.
            lon: Longitude of the site.
            utc_offset: UTC offset of the timestamps in hours.
            z: Number of uncertainties below zero beyond which a nighttime value is removed.
            level_window: Window of the rolling median of nighttime values that serves as the local
                respiration level for the expected noise removals, as a time span or a record count.
            idstr: Identifier suffix for output variable names.
            showplot: If True, draw the diagnostic figure.
            verbose: If True, print the counts.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        if z <= 0:
            raise ValueError(f"z must be positive, got {z}.")
        self.uncertainty = uncertainty.reindex(self.series.index).astype(float)
        self.z = z
        self.level_window = level_window
        self.showplot = showplot
        self.verbose = verbose
        # One limit per record rather than one band: the GUI's band overlay does not apply.
        self.last_lower_bound = None
        self.last_upper_bound = None
        self.flag_daytime, _, self.is_daytime, self.is_nighttime = create_daytime_nighttime_flags(
            timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset)
        self.expected_noise = {}

    def calc(self, repeat: bool = False, progress_callback=None):
        """Flag the nighttime records below ``-z`` uncertainties. A single pass: the limit of a
        record depends on that record alone, so a second pass finds nothing new."""
        self._overall_flag, n_iterations = self.repeat(
            func=self.run_flagtests, repeat=False, progress_callback=progress_callback)
        self.expected_noise = self._expected_noise()
        if self.verbose:
            n_removed = int((self.overall_flag == 2).sum())
            detail(f"Nighttime records below -{self.z:g} uncertainties: {n_removed}; expected from "
                   f"noise alone: {self.expected_noise['records']:.1f}", verbose=self.verbose)
        if self.showplot:
            self._plot()

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        s = self.filteredseries.dropna()
        night = self.is_nighttime.reindex(s.index, fill_value=False)
        limit = -self.z * self.uncertainty.reindex(s.index)
        rejected_mask = night & (s < limit)
        ok = s.index[~rejected_mask]
        rejected = s.index[rejected_mask]
        return ok, rejected, int(rejected_mask.sum())

    def _night_frame(self) -> pd.DataFrame:
        """Nighttime records with a value and an uncertainty, with the local level beside them."""
        night = self.series[self.is_nighttime]
        level = night.rolling(self.level_window, center=True, min_periods=1).median()
        df = pd.DataFrame({'value': night, 'sigma': self.uncertainty[self.is_nighttime],
                           'level': level})
        return df.dropna()

    def _expected_noise(self) -> dict:
        """Removals noise alone would cause, if each value were its local level plus noise.

        For a record with level R and uncertainty sigma, noise pushes it below ``-z * sigma``
        with probability ``Phi(-z - R / sigma)``. The mean of such a value is that of a normal
        distribution truncated at the limit, ``R - sigma * phi(a) / Phi(a)`` with
        ``a = -z - R / sigma``. Summed over the records this gives the expected number of noise
        removals and their expected sum, which is the most the test can bias a total by.
        """
        df = self._night_frame()
        if df.empty:
            return {'records': 0.0, 'sum': 0.0}
        level = df['level'].clip(lower=0)  # the true nighttime value is not negative
        a = -self.z - level / df['sigma']
        p = norm.cdf(a)
        truncated_mean = level - df['sigma'] * norm.pdf(a) / np.where(p > 0, p, np.nan)
        return {'records': float(np.nansum(p)),
                'sum': float(np.nansum(p * np.nan_to_num(truncated_mean)))}

    def _plot(self):
        import matplotlib.pyplot as plt

        import diive.core.plotting.styles.chainplots as cp
        from diive.core.plotting.plotfuncs import nice_date_ticks

        df = self._night_frame()
        removed = (self.overall_flag == 2).reindex(df.index, fill_value=False)
        positive = df['value'] >= 0
        within = ~positive & ~removed
        c0, c1, c2 = cp.QCF_COLORS
        name = self.series.name

        fig = plt.figure(facecolor='white', figsize=(15, 5.8), layout='constrained')
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.2])
        ax_sc, ax_z, ax_m = (fig.add_subplot(gs[0, i]) for i in range(3))

        for mask, color, label, zo in ((positive, cp.KEPT, 'positive', 2),
                                       (within, c1, f'negative, within {self.z:g} sigma', 3),
                                       (removed, c2, f'removed, below -{self.z:g} sigma', 4)):
            ax_sc.plot(df.loc[mask, 'sigma'], df.loc[mask, 'value'], color=color, zorder=zo,
                       label=f"{label} ({int(mask.sum()):,})", **{**cp.POINTS, 'markersize': 3.5})
        xmax = df['sigma'].quantile(0.995)
        xs = np.linspace(0, xmax, 50)
        ax_sc.plot(xs, -self.z * xs, color=c2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=5,
                   label=f"limit, -{self.z:g} x uncertainty")
        ax_sc.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_sc.set_xlim(0, xmax)
        low = df['value'].quantile(0.002)
        ax_sc.set_ylim(min(low * 1.1, -1), df['value'].quantile(0.995))
        ax_sc.set_xlabel('random uncertainty')
        ax_sc.set_ylabel(f'nighttime {name}')
        cp.style_axes(ax_sc, grid_axis='both')
        cp.panel_label(ax_sc, "Value against its uncertainty")
        ax_sc.legend(loc='upper left', frameon=False, fontsize=8.5, markerscale=3)

        zs = df['value'] / df['sigma']
        lo = min(np.floor(zs.quantile(0.001)), -self.z - 1)
        hi = max(np.ceil(zs.quantile(0.999)), 1)
        zs = zs.clip(lo, hi)
        edges = np.linspace(lo, hi, 61)
        ax_z.hist([zs[positive], zs[within], zs[removed]], bins=edges, stacked=True,
                  color=[cp.KEPT, c1, c2], edgecolor='white', linewidth=0.3)
        ax_z.axvline(-self.z, color=c2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax_z.axvline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_z.set_yscale('log')
        ax_z.set_xlabel('value / uncertainty')
        ax_z.set_ylabel('records')
        cp.style_axes(ax_z)
        cp.panel_label(ax_z, "Value in units of its uncertainty")

        months = df.index.to_period('M')
        share_neg = (~positive).groupby(months).mean() * 100
        share_removed = removed.groupby(months).mean() * 100
        x = share_neg.index.to_timestamp() + pd.Timedelta(days=14)
        ax_m.bar(x, share_neg.to_numpy(), width=24, color=c1, label='negative')
        ax_m.bar(x, share_removed.to_numpy(), width=24, color=c2, label='removed')
        ax_m.set_ylabel('% of nighttime records')
        cp.style_axes(ax_m)
        cp.panel_label(ax_m, "Per month")
        ax_m.legend(loc='upper right', frameon=False, fontsize=8.5)
        nice_date_ticks(ax=ax_m)

        n, n_removed = len(df), int(removed.sum())
        expected = self.expected_noise
        cp.figure_title(
            fig, f"Negative nighttime values: {name}",
            f"{n_removed:,} of {n:,} nighttime records removed ({100 * n_removed / n:.1f} %), "
            f"below -{self.z:g} times their uncertainty. Noise alone would explain "
            f"{expected['records']:.1f} of them. Median uncertainty {df['sigma'].median():.2f}.")
        fig.show()
        self._fig = fig
