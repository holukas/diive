"""
MANUAL REMOVAL: EXPLICIT DATA FLAGGING
=======================================

Flag specific records or date ranges as outliers by listing them explicitly,
rather than detecting them statistically. The user names exactly which
timestamps to remove; matched records are flagged 2 (rejected) and set to
missing in the filtered series, everything else is flagged 0 (ok), and gaps
stay unflagged.

This is the manual counterpart to the automatic detectors (Hampel, LocalSD,
z-score, absolute limits, ...): instead of a rule, it removes a known list of
bad records. Selection is purely time-based — a bare date covers the whole day,
a [start, end] pair covers the closed interval — so it does not depend on the
values themselves.

Use cases:
    * Removing periods of known instrument malfunction, calibration, or
      maintenance recorded in a field/site logbook.
    * Excising power outages, sensor swaps, or physical disturbances (e.g.
      grazing, mowing, snow on a sensor) whose timing is known but whose values
      look plausible, so a statistical detector would miss them.
    * Discarding data flagged as bad during visual inspection of plots.
    * Forcing the removal of records that survived the automatic detectors but
      are known to be wrong, typically as a final manual step in a screening
      chain (see StepwiseOutlierDetection / StepwiseMeteoScreeningDb).
    * Reproducible, auditable cleaning: the exact removed dates live in the
      script/config, so the same input always yields the same result.

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd
from pandas import Series, DatetimeIndex

from diive.core.base.flagbase import FlagBase
from diive.core.utils.console import detail
from diive.core.utils.prints import ConsoleOutputDecorator


@ConsoleOutputDecorator()
class ManualRemoval(FlagBase):
    """Generate flag for data points that should be removed."""

    flagid = 'OUTLIER_MANUAL'

    def __init__(self,
                 series: Series,
                 remove_dates: list,
                 showplot: bool = False,
                 verbose: bool = False,
                 idstr: str = None):
        """

        Args:
            series: Time series in which outliers are identified.
            remove_dates: List of records to flag for removal. Each entry is either:
                * a single date(time) string, which flags that one record, e.g.
                  '2022-06-30 23:58:30'. A bare date such as '2006-05-01' flags the
                  whole day (partial-string match, both bounds inclusive).
                * a ``[start, end]`` list of two date(time) strings, which flags all
                  records in the closed interval [start, end], e.g.
                  ['2022-06-05 00:00:30', '2022-06-07 14:30:00']. Bare dates again
                  span whole days, so ['2006-05-01', '2006-07-18'] removes everything
                  from the start of 2006-05-01 to the end of 2006-07-18.
                A range MUST be given as a nested two-element list. A flat list of two
                strings is interpreted as two separate single-record removals, not a range.
                Example mixing both forms:
                    remove_dates=['2022-06-30 23:58:30',
                                  ['2022-06-05 00:00:30', '2022-06-07 14:30:00']]
            showplot: Show plot with removed data points.
            verbose: More text output to console if *True*.
            idstr: Identifier, added as suffix to output variable names.

        Returns:
            None. After calling .calc() (or .run()), results are available via
            .filteredseries, .flag and .overall_flag.

        """
        super().__init__(series=series, flagid=self.flagid, idstr=idstr, verbose=verbose)
        self.remove_dates = remove_dates
        self.showplot = showplot

    def calc(self, repeat: bool = False, progress_callback=None):
        """Calculate overall flag for manually removed data points.

        Args:
            repeat: Accepted for interface compatibility with the other outlier
                detectors but ignored — manual removal flags fixed timestamps, so a
                second pass would re-flag the same records (selection is index-based,
                not value-based) and never converge. Detection always runs once.
            progress_callback: Optional ``callable(iteration, n_outliers,
                filteredseries)`` invoked after the (single) iteration.
        """

        self._overall_flag, n_iterations = self.repeat(
            func=self.run_flagtests, repeat=False, progress_callback=progress_callback)
        if self.showplot:
            self.defaultplot(n_iterations=n_iterations)

    def _flagtests(self, iteration: int) -> tuple[DatetimeIndex, DatetimeIndex, int]:
        """Mark manually specified dates as outliers.

        Args:
            iteration: Current iteration number

        Returns:
            (ok_indices, rejected_indices, n_outliers) where:
                - ok_indices: DatetimeIndex of records to keep
                - rejected_indices: DatetimeIndex of manually removed records
                - n_outliers: Total number of removed records
        """

        flag = pd.Series(index=self.filteredseries.index, data=np.nan)

        for date_spec in self.remove_dates:
            if isinstance(date_spec, str):
                start = end = date_spec
            elif isinstance(date_spec, (list, tuple)):
                if len(date_spec) != 2:
                    raise ValueError(
                        f"A date range in remove_dates must have exactly two elements "
                        f"[start, end], got {len(date_spec)}: {date_spec!r}")
                start, end = date_spec
            else:
                raise TypeError(
                    f"Each entry in remove_dates must be a date(time) string or a "
                    f"[start, end] list, got {type(date_spec).__name__}: {date_spec!r}")
            # Partial-string slicing makes a bare date span the whole day, inclusive
            # of both bounds; a boolean >=/<= comparison would only match midnight.
            # Index is monotonic (FlagBase enforces a frequency), so label slicing
            # works even for bounds not present in the index.
            rejected_idx = self.filteredseries.loc[start:end].index
            flag.loc[rejected_idx] = 2

        rejected = flag[flag == 2].index
        n_outliers = len(rejected)
        # Records that still hold data and were not removed are flagged ok; missing
        # records stay unflagged (consistent with the other outlier detectors).
        ok = self.filteredseries.dropna().index.difference(rejected)

        if self.verbose:
            detail(f"ITERATION#{iteration}: Manually removed {n_outliers} values")

        return ok, rejected, n_outliers
