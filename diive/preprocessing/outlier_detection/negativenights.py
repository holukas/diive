"""
NEGATIVE NIGHTS: NIGHTS WHOSE MEDIAN LIES CLEARLY BELOW ZERO
============================================================

Flag every record of a night whose median lies further below zero than its uncertainty explains.

At night there is no photosynthesis, so net ecosystem exchange (NEE) is respiration and its true
value is zero or positive. ``NegativeNighttime`` tests each record against its own uncertainty and
catches single values far below zero. It does not catch an instrument error that shifts a whole
night by a few units: each record of such a night lies within a few sigma of zero, but the night
as a whole cannot. The median of ``n`` records with noise ``sigma`` around a true value of zero or
more lies below ``-z * sqrt(pi / 2) * sigma / sqrt(n)`` with a probability of at most Phi(-z) (2.3 %
for z = 2), so a night whose median lies there is shifted, not noisy. This test removes all records
of such nights.

Such nights occur, for example, with an open-path gas analyser in fog or rime, when the windows
are wet or iced and the flux reads as uptake for hours. Gap-filling learns from the remaining
records, so a few of these nights teach it nighttime uptake for similar conditions.

A night is the run of nighttime records between one noon and the next, so it is labelled with the
date of its evening. Daytime records are never tested. A night with fewer than ``min_records``
values, or without an uncertainty, is not tested.

Quality flags (overall_flag):
  - flag=0: daytime record, or record of a night whose median is not clearly below zero (valid)
  - flag=2: record of a night whose median lies below ``-z`` standard errors (removed)
  - NaN: record missing in the input, so no test could be performed

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import info
from diive.preprocessing.outlier_detection.common import create_daytime_nighttime_flags

#: Standard error of a median of n Gaussian values, in units of sigma / sqrt(n).
MEDIAN_SE_FACTOR = float(np.sqrt(np.pi / 2))


def night_of(index: DatetimeIndex) -> DatetimeIndex:
    """The night each timestamp belongs to, as the date of its evening (noon to noon)."""
    return pd.DatetimeIndex((index - pd.Timedelta('12h')).normalize(), name='NIGHT_OF')


class NegativeNights(FlagBase):
    """Flag all records of nights whose median lies clearly below zero. See :meth:`__init__`."""

    flagid = 'OUTLIER_NEGNIGHTS'

    def __init__(self,
                 series: Series,
                 uncertainty: Series,
                 lat: float,
                 lon: float,
                 utc_offset: int,
                 nighttime_threshold: float = 20,
                 z: float = 2.0,
                 min_records: int = 6,
                 idstr: str = None,
                 showplot: bool = False,
                 verbose: bool = False):
        """Set up the test.

        Args:
            series: Flux in which nights with a median clearly below zero are flagged, e.g. NEE.
            uncertainty: Random uncertainty of each record, in the units of ``series`` and on its
                index, e.g. the pooled spread of nighttime values (``POOLED`` of
                ``nighttime_uncertainty``). The uncertainty of a night is the median of the
                uncertainties of its records.
            lat: Latitude of the site, to tell day from night.
            lon: Longitude of the site.
            utc_offset: UTC offset of the timestamps in hours.
            nighttime_threshold: Potential radiation (W m-2) below which a record is nighttime.
            z: Number of standard errors of the night median below zero beyond which the whole
                night is removed.
            min_records: Fewest nighttime values a night needs to be tested.
            idstr: Identifier suffix for output variable names.
            showplot: If True, draw the diagnostic figure.
            verbose: If True, print the counts.
        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr)
        if z <= 0:
            raise ValueError(f"z must be positive, got {z}.")
        if min_records < 2:
            raise ValueError(f"min_records must be at least 2, got {min_records}.")
        self.uncertainty = uncertainty.reindex(self.series.index).astype(float)
        self.uncertainty_name = uncertainty.name
        self.z = z
        self.min_records = int(min_records)
        self.showplot = showplot
        self.verbose = verbose
        # One limit per night rather than one band: the GUI's band overlay does not apply.
        self.last_lower_bound = None
        self.last_upper_bound = None
        self.flag_daytime, _, self.is_daytime, self.is_nighttime = create_daytime_nighttime_flags(
            timestamp_index=self.series.index, lat=lat, lon=lon, utc_offset=utc_offset,
            nighttime_threshold=nighttime_threshold)
        self._nights = None

    @property
    def nights(self) -> DataFrame:
        """One row per night: records, median, uncertainty, standard error, limit and whether it
        was removed. Available after :meth:`calc`."""
        if self._nights is None:
            raise Exception("No night table available, run calc() first.")
        return self._nights

    def calc(self, repeat: bool = False, progress_callback=None):
        """Flag the nights whose median lies below the limit. A single pass: removing a night does
        not change the median of another."""
        self._overall_flag, n_iterations = self.repeat(
            func=self.run_flagtests, repeat=False, progress_callback=progress_callback)
        if self.verbose:
            removed = self.nights[self.nights['removed']]
            info(f"Nights with a median below -{self.z:g} standard errors: {len(removed):,} of "
                 f"{int(self.nights['tested'].sum()):,} tested nights, {int(removed['records'].sum()):,} "
                 f"records.", verbose=self.verbose)
        if self.showplot:
            self._plot()

    def _night_table(self, night_values: Series) -> DataFrame:
        """Per night: records, median, uncertainty, standard error, limit, tested, removed."""
        frame = DataFrame({'value': night_values,
                           'sigma': self.uncertainty.reindex(night_values.index)})
        groups = frame.groupby(night_of(frame.index))
        table = DataFrame({'records': groups['value'].count(),
                           'median': groups['value'].median(),
                           'sigma': groups['sigma'].median()})
        table['se'] = MEDIAN_SE_FACTOR * table['sigma'] / np.sqrt(table['records'])
        table['limit'] = -self.z * table['se']
        table['tested'] = (table['records'] >= self.min_records) & table['sigma'].notna()
        table['removed'] = table['tested'] & (table['median'] < table['limit'])
        return table

    def _flagtests(self, iteration) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        s = self.filteredseries.dropna()
        night = self.is_nighttime.reindex(s.index, fill_value=False)
        night_values = s[night.to_numpy()]
        self._nights = self._night_table(night_values)
        removed_nights = self._nights.index[self._nights['removed']]
        rejected = night_values.index[night_of(night_values.index).isin(removed_nights)]
        ok = s.index.difference(rejected)
        return ok, rejected, len(rejected)

    def _plot(self):
        import matplotlib.pyplot as plt

        import diive.core.plotting.styles.chainplots as cp
        from diive.core.plotting.plotfuncs import nice_date_ticks

        name = self.series.name
        t = self.nights[self.nights['tested']]
        fig = plt.figure(facecolor='white', figsize=(15, 5.2), layout='constrained')
        if t.empty:
            ax = fig.add_subplot()
            ax.set_axis_off()
            cp.note(ax, f"No night holds {self.min_records} values with an uncertainty, "
                        f"so no night was tested.")
            cp.figure_title(fig, f"Negative nights: {name}", "No night tested.")
            fig.show()
            self._fig = fig
            return
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
        ax_sc, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        c0, c1, c2 = cp.QCF_COLORS
        kept, removed = t[~t['removed']], t[t['removed']]
        ax_sc.plot(kept['se'], kept['median'], color=cp.KEPT, label=f"kept ({len(kept):,})",
                   **{**cp.POINTS, 'markersize': 4})
        ax_sc.plot(removed['se'], removed['median'], color=c2, label=f"removed ({len(removed):,})",
                   **{**cp.POINTS, 'markersize': 5})
        xs = np.linspace(0, t['se'].quantile(0.995), 50)
        ax_sc.plot(xs, -self.z * xs, color=c2, linewidth=1.2, linestyle=(0, (4, 3)),
                   label=f"limit, -{self.z:g} x standard error")
        ax_sc.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_sc.set_xlim(0, xs[-1])
        ax_sc.set_xlabel('standard error of the night median')
        ax_sc.set_ylabel(f'night median of {name}')
        cp.style_axes(ax_sc, grid_axis='both')
        cp.panel_label(ax_sc, "Night median against its standard error")
        ax_sc.legend(loc='lower left', frameon=False, fontsize=8.5, markerscale=2)

        ax_t.bar(kept.index, kept['median'], width=1.0, color=cp.KEPT)
        ax_t.bar(removed.index, removed['median'], width=3.0, color=c2)
        ax_t.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_t.set_ylabel(f'night median of {name}')
        cp.style_axes(ax_t)
        cp.panel_label(ax_t, "Night medians over time, removed nights in red")
        nice_date_ticks(ax=ax_t)

        n_rec = int(removed['records'].sum())
        cp.figure_title(
            fig, f"Negative nights: {name}",
            f"{len(removed):,} of {len(t):,} tested nights removed ({n_rec:,} records): their median "
            f"lies below -{self.z:g} standard errors (sqrt(pi/2) x uncertainty / sqrt(n)). "
            f"Nights with fewer than {self.min_records} values are not tested.")
        fig.show()
        self._fig = fig
