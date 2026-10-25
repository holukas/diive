"""
STORAGE TERM CORRECTION
=======================

Add gas storage term to flux measurements (single-point profile approximation).

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

from diive.core.dfun.stats import sstats  # Time series stats
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.utils.console import detail, info, rule, warn


class FluxStorageCorrectionSinglePointEddyPro:
    """Add a storage correction term to eddy covariance flux measurements.

    Applies storage flux correction at Level-3.1 in Swiss FluxNet processing.
    The storage term—the change in gas or heat concentration over time—is added
    to the measured eddy covariance flux.

    Automatically detects the storage variable from flux type, optionally removes
    spikes from the storage term, gap-fills missing and removed values using a
    rolling median, and tracks which values were filled.

    **Output naming:** FC becomes NEE (storage-corrected CO2 flux); other fluxes
    (LE, H, FH2O, etc.) keep their name with the suffix {idstr} added
    (e.g., LE_L3.1, H_L3.1).
    """

    def __init__(self,
                 df: DataFrame,
                 fluxcol: str,
                 basevar: str,
                 gapfill_storage_term: bool = True,
                 idstr: str = 'L3.1',
                 set_storage_to_zero: bool = False,
                 strgcol: str | None = None,
                 screen_storage_term: bool = False,
                 storage_screening: dict | None = None):
        """Initialize the storage corrector.

        Args:
            df: DataFrame with flux and storage term columns (e.g., 'FC' and 'SC_SINGLE').
            fluxcol: Name of the flux column to correct ('FC', 'LE', 'H', etc.).
                Used to auto-detect the storage variable.
            basevar: Name of the measured variable ('CO2', 'H2O', 'N2O', 'CH4').
                For logging; not used in calculations.
            gapfill_storage_term: If True, gap-fills missing storage values using a
                rolling median before adding the term to the flux. Default True.
                Set to False only if the storage column is already complete or you
                want to preserve NaN positions.
            idstr: Suffix for output columns (default 'L3.1'). Output is {fluxcol}{idstr}
                or NEE{idstr} for FC.
            set_storage_to_zero: If True, sets the storage term to zero instead of
                using measured or gap-filled storage data.  Use this for fluxes where
                no storage profile is available (e.g. H, LE at low-canopy sites) or
                when the single-point approximation is considered unreliable.
                No storage column is needed in ``df`` in this case.
                Default False.
            strgcol: Explicit name of the storage-term column in ``df``. Overrides
                the built-in auto-detection (FC -> SC_SINGLE, LE -> SLE_SINGLE,
                ...). Use this when your storage column does not follow the
                FLUXNET / Swiss FluxNet naming convention (e.g. ``'co2_strg'``
                from EddyPro full-output files, or a custom site-specific
                name). Default None (auto-detect).
            screen_storage_term: If True, remove spikes from the storage term before it
                is added to the flux, and fill them like missing values. A spike in the
                storage term would otherwise turn a good flux record into an outlier that
                the outlier tests of Level 3.2 remove, so the flux record would be lost.
                The screening is a Hampel filter on the double differences of the storage
                term (13 days, 5.5 sigma, day and night together, one pass). It removes
                isolated spikes only: a run of more than ``max_run`` (3) removed values in a
                row is kept as measured, because it is a noisy period, not a spike. Each run,
                together with one record on either side, is replaced by the mean of its
                measured values. The block keeps its sum, the real concentration change across
                it, so the storage term stays balanced; a single spike is spread over three
                records. Needs
                ``gapfill_storage_term=True``, so that every measured flux keeps a storage
                term. Default False.
            storage_screening: Settings that replace the defaults above, e.g.
                ``{'n_sigma': 7}``, ``{'repeat': True}`` or ``{'max_run': 5}``. The others go
                to the Hampel filter, see
                :class:`~diive.preprocessing.outlier_detection.hampel.Hampel`.

        Attributes:
            results: DataFrame with corrected flux and gap-filling info.
            flux_corrected_col: Name of the corrected flux column.
            gapfilled_strgcol: Name of the gap-filled storage column (if enabled).
            flag_isgapfilled: Flag column (0=original, 1=gap-filled because missing,
                2=replaced by the screening with the mean of its block).
            storage_outliers_: Timestamps of the spikes the screening found.
            storage_replaced_: Timestamps of the values it replaced: the spikes and one
                record on either side.
        """
        self.df = df.copy()
        self.fluxcol = fluxcol
        self.basevar = basevar
        self.gapfill_storage_term = gapfill_storage_term
        self.idstr = validate_id_string(idstr=idstr)
        # Storage-column resolution: when the user supplies an explicit
        # ``strgcol`` we trust it (and only validate it exists in df). This
        # path skips the auto-detect hardcoded-name lookup entirely so
        # fluxcols that aren't in the auto-detect table (e.g. EddyPro
        # full-output ``co2_flux``) still work.
        if strgcol is not None:
            if strgcol not in df.columns and not set_storage_to_zero:
                raise KeyError(
                    f"strgcol={strgcol!r} not found in df. "
                    f"Available columns: {list(df.columns)}"
                )
            self.strgcol = strgcol
            self.flux_corrected_col = (
                f'NEE{self.idstr}' if self.fluxcol == 'FC'
                else f"{self.fluxcol}{self.idstr}"
            )
        else:
            self.flux_corrected_col, self.strgcol = self._detect_storage_var()
        # The storage column is read only when the storage term is actually
        # used. With set_storage_to_zero=True it never is, so a missing column
        # must not be an error: that flag exists exactly for fluxes without a
        # storage profile (H, LE). Drop the name so the rest of the class knows
        # there is no storage column to read or show.
        if set_storage_to_zero and self.strgcol not in df.columns:
            self.strgcol = None
        # Note: this class deliberately does NOT emit a 'storage missing'
        # quality-test flag (i.e. no FLAG_..._TEST column tied to NaN in the
        # corrected flux). Whether a storage term was measured or gap-filled
        # is provenance, not a quality criterion — captured by the
        # FLAG_..._ISFILLED column when gap-fill is enabled. Quality
        # aggregation at L3.1 inherits the L2 flags via FlagQCF.
        self.set_storage_to_zero = set_storage_to_zero
        if screen_storage_term and not gapfill_storage_term and not set_storage_to_zero:
            raise ValueError("screen_storage_term=True needs gapfill_storage_term=True: the removed "
                             "storage values must be filled, or their flux records are lost.")
        self.screen_storage_term = screen_storage_term and not set_storage_to_zero
        screening = {'window_length': '13D', 'n_sigma': 5.5, 'use_differencing': True,
                     'separate_day_night': False, 'repeat': False, 'max_run': 3,
                     **(storage_screening or {})}
        self._screening_repeat = screening.pop('repeat')
        self._screening_max_run = screening.pop('max_run')
        self.storage_screening = screening
        self.storage_outliers_ = pd.DatetimeIndex([])
        self.storage_replaced_ = pd.DatetimeIndex([])

        # Name of gapfilled storage column and its flag
        self.gapfilled_strgcol = None
        # self.gapfilled_strgcol = f"{self.strgcol}_gfRF{self.idstr}"
        self.flag_isgapfilled = None
        # self.flag_isgapfilled = f"FLAG_{self.gapfilled_strgcol}_ISFILLED"

        self._results = None

    @property
    def results(self) -> DataFrame:
        """Return results as dataframe"""
        if not isinstance(self._results, DataFrame):
            raise Exception('Results for storage are empty')
        return self._results

    def storage_correction(self):
        """Calculate the corrected flux with storage term added.

        Extracts the measured flux and storage term, optionally gap-fills any
        missing storage values, then adds the storage term to the flux.

        Creates three output columns:
            - {flux_corrected_col}: the corrected flux
            - {gapfilled_strgcol}: gap-filled storage (if gap-filling enabled)
            - {flag_isgapfilled}: flag indicating which values were filled (0=original, 1=filled)

        If storage is missing and gap-filling is disabled, the output will be NaN.
        With gap-filling, it uses an expanding rolling median to fill gaps.

        Results are stored in `self.results`.
        """
        if self.set_storage_to_zero:
            info(f"Calculating storage-corrected flux {self.flux_corrected_col} "
                 f"from flux {self.fluxcol} + storage term set to zero")
        else:
            info(f"Calculating storage-corrected flux {self.flux_corrected_col} "
                 f"from flux {self.fluxcol} + storage term {self.strgcol}")

        # Collect flux and storage term data. There is no storage column when
        # the storage term is set to zero for a flux without a storage profile.
        cols = [self.fluxcol] if self.strgcol is None else [self.fluxcol, self.strgcol]
        self._results = self.df[cols].copy()

        if not self.set_storage_to_zero:
            if self.screen_storage_term:
                self._screen_storage_term()
            # Gap-fill storage term
            if self.gapfill_storage_term:
                gapfilled_df = self._gapfill_storage_term()
                self._results = pd.concat([self._results, gapfilled_df], axis=1)

                # Add gapfilled storage term to flux data
                self._results[self.flux_corrected_col] = self._results[self.fluxcol].add(
                    self._results[self.gapfilled_strgcol])
            else:
                # Add original (non-gapfilled) storage term to flux, this can result in NaNs
                self._results[self.flux_corrected_col] = self._results[self.fluxcol].add(self._results[self.strgcol])
        else:
            # Add custom, constant storage value to all fluxes
            self._results[self.flux_corrected_col] = self._results[self.fluxcol].add(0)

    def report(self):
        """Print a summary of the storage correction.

        Shows data availability (how many flux and storage records exist), the impact
        of missing storage values, and gap-filling results if enabled. For gap-filled
        data, compares statistics (median, percentiles) between original and filled
        values and shows what fraction of the output came from each source.
        """
        rule(f"Storage Correction: {self.flux_corrected_col}")

        n_flux = len(self.results[self.fluxcol].dropna())

        if self.set_storage_to_zero:
            # No storage term was used, so storage availability and gap-filling
            # statistics do not apply.
            info(f"Measured flux: {n_flux:,}  |  Storage term set to zero for all records")
            return

        # Check missing storage for measured fluxes
        _subset = pd.concat([self.results[self.fluxcol], self.results[self.strgcol]], axis=1)
        _subset = _subset.dropna(subset=[self.fluxcol])
        n_orig_missing_strg = _subset[self.strgcol].isnull().sum()
        n_storageterm = len(_subset) - n_orig_missing_strg  # Storage available for flux records

        if not self.gapfilled_strgcol:
            # No gap-filling case
            perc_lost = (n_orig_missing_strg / n_flux * 100) if n_flux > 0 else 0
            info(f"Measured flux: {n_flux:,}  |  Storage available: {n_storageterm:,}  |  Missing: {n_orig_missing_strg:,}")
            info(f"[yellow]Without gap-filling: {perc_lost:.1f}% of storage values missing ({n_orig_missing_strg} records)[/yellow]")
        else:
            # Gap-filling enabled
            locs_fluxmissing = self.results[self.fluxcol].isnull()
            fluxavailable = self.results[~locs_fluxmissing].copy()
            locs_isfilled = fluxavailable[self.flag_isgapfilled] >= 1  # 1 missing, 2 screened
            locs_isorig = fluxavailable[self.flag_isgapfilled] == 0

            n_isfilled = len(fluxavailable[locs_isfilled])
            n_isorig = len(fluxavailable[locs_isorig])
            n_flux_corrected = len(fluxavailable[self.flux_corrected_col].dropna())
            n_filled_missing = int((fluxavailable[self.flag_isgapfilled] == 1).sum())
            perc_recovered = (n_filled_missing / n_orig_missing_strg * 100) if n_orig_missing_strg > 0 else 0

            info(f"Measured flux: {n_flux:,}  |  Storage available: {n_storageterm:,}  |  Missing: {n_orig_missing_strg:,}")
            info(f"Gap-fill: recovered {n_filled_missing:,}  |  Recovery rate: {perc_recovered:.1f}%")
            n_screened = int((fluxavailable[self.flag_isgapfilled] == 2).sum())
            if self.screen_storage_term:
                info(f"Screening: {n_screened:,} storage spike(s) at measured fluxes removed and filled")
            info(f"Output corrected flux: {n_flux_corrected:,}  ({n_isorig} original + {n_isfilled} gap-filled)")

            # Statistics if gap-filled values exist
            if n_isfilled > 0:
                try:
                    filledstats = sstats(fluxavailable[locs_isfilled][self.gapfilled_strgcol])
                    measuredstats = sstats(fluxavailable[locs_isorig][self.gapfilled_strgcol])

                    fluxstats = sstats(fluxavailable[self.fluxcol])
                    correctedstats = sstats(fluxavailable[self.flux_corrected_col])

                    cols_to_show = ['MEDIAN', 'P01', 'P99']
                    orig_strg = measuredstats.T[cols_to_show].iloc[0]
                    filled_strg = filledstats.T[cols_to_show].iloc[0]
                    flux_orig = fluxstats.T[cols_to_show].iloc[0]
                    flux_corr = correctedstats.T[cols_to_show].iloc[0]

                    detail(f"Flux            median={flux_orig['MEDIAN']:>7.2f}  P01={flux_orig['P01']:>7.2f}  P99={flux_orig['P99']:>7.2f}")
                    detail(f"Corrected flux  median={flux_corr['MEDIAN']:>7.2f}  P01={flux_corr['P01']:>7.2f}  P99={flux_corr['P99']:>7.2f}")
                    detail(f"Orig storage    median={orig_strg['MEDIAN']:>7.2f}  P01={orig_strg['P01']:>7.2f}  P99={orig_strg['P99']:>7.2f}")
                    detail(f"Filled storage  median={filled_strg['MEDIAN']:>7.2f}  P01={filled_strg['P01']:>7.2f}  P99={filled_strg['P99']:>7.2f}")
                except (KeyError, IndexError, AttributeError):
                    pass

    def _screen_storage_term(self) -> None:
        """Find isolated spikes in the storage term with a Hampel filter.

        ``_gapfill_storage_term`` replaces each run of them, with one record on either side,
        by its mean (flag 2). The original storage column in the results keeps the measured
        values.
        """
        from diive.preprocessing.outlier_detection.hampel import Hampel
        strg = self._results[self.strgcol]
        if strg.notna().sum() < 3:
            return
        ham = Hampel(series=strg.copy(), **self.storage_screening)
        ham.calc(repeat=self._screening_repeat)
        flagged = (ham.overall_flag == 2).reindex(strg.index, fill_value=False)
        # Only isolated spikes: runs of more than max_run flagged records stay as measured.
        run_id = (flagged != flagged.shift()).cumsum()
        run_len = flagged.groupby(run_id).transform('size')
        keep = flagged & (run_len <= self._screening_max_run)
        self.storage_outliers_ = strg.index[keep.to_numpy()]
        info(f"Storage term {self.strgcol}: {len(self.storage_outliers_):,} spike(s) found "
             f"({100 * len(self.storage_outliers_) / max(strg.notna().sum(), 1):.2f} % of the "
             f"measured storage values)")

    def screening_summary(self, gain: float = 1.0) -> dict:
        """Numbers that show whether the screened and filled storage term still makes sense.

        On the records with a measured flux, compares the storage term without screening
        (measured values, gaps filled) with the one used (spikes replaced as well):

        - ``spikes``: number of spikes the screening found at measured fluxes.
        - ``replaced``: number of values replaced there (the spikes and their neighbours).
        - ``spikes_negative``: share of the spikes below zero. Real storage spikes have both
          signs; a large negative share in the morning hints at the real morning release being
          cut.
        - ``sum_unscreened``, ``sum_used``: the sums of the two versions, times ``gain``
          (``12.011e-6 * 1800`` gives gC m-2 for a 30-min CO2 storage term in umol m-2 s-1).
          Over a year the storage term should sum to about zero. Their difference is the
          direct effect of the screening on the sum of the flux.
        - ``daily_abs_median_unscreened``, ``daily_abs_median_used``: median of the absolute
          daily sums, times ``gain``. Lower means the daily balance improved.

        Needs ``screen_storage_term=True`` and a finished ``storage_correction()``.
        """
        d = self._screening_frame()
        spikes = d['spike']
        days = d.index.normalize()
        daily_unscreened = (d['unscreened'] * gain).groupby(days).sum(min_count=1)
        daily_used = (d['used'] * gain).groupby(days).sum(min_count=1)
        return {
            'spikes': int(spikes.sum()),
            'replaced': int(d['replaced'].sum()),
            'spikes_negative': float((d.loc[spikes, 'unscreened'] < 0).mean()) if spikes.any() else float('nan'),
            'sum_unscreened': float(d['unscreened'].sum() * gain),
            'sum_used': float(d['used'].sum() * gain),
            'daily_abs_median_unscreened': float(daily_unscreened.abs().median()),
            'daily_abs_median_used': float(daily_used.abs().median()),
        }

    def _screening_frame(self) -> DataFrame:
        """Storage term at the records with a measured flux: unscreened, used, replaced, spikes."""
        if not self.screen_storage_term or self.gapfilled_strgcol is None:
            raise RuntimeError("Needs screen_storage_term=True and storage_correction() first.")
        r = self.results
        r = r[r[self.fluxcol].notna()]
        replaced = r[self.flag_isgapfilled] == 2
        # Without screening the replaced records would carry their measured value.
        unscreened = r[self.gapfilled_strgcol].where(~replaced, r[self.strgcol])
        return DataFrame({'unscreened': unscreened, 'used': r[self.gapfilled_strgcol],
                          'replaced': replaced, 'spike': r.index.isin(self.storage_outliers_)})

    def plot_screening(self, gain: float = 1.0, units: str = '', showplot: bool = True):
        """Check the screened and filled storage term in three panels.

        1. Mean daily cycle of the storage term with and without screening. The real cycle is
           positive at night, when CO2 builds up, and negative in the morning, when it is released.
           If the screened line flattens the morning dip, the screening cut real signal.
        2. Spikes by hour of the day, positive above and negative below zero. Real spikes spread
           over the day with both signs.
        3. Cumulative sum of both versions over the record (times ``gain``). The storage term
           should sum to about zero over a year; the gap between the lines is the direct effect of
           the screening on the sum of the flux.

        Args:
            gain: Factor for the cumulative sums, e.g. ``12.011e-6 * 1800`` for gC m-2.
            units: Unit of the cumulative sums.
            showplot: Call ``plt.show()`` after drawing.
        """
        import numpy as np
        import diive.core.plotting.styles.chainplots as cp
        from diive.core.plotting.plotfuncs import nice_date_ticks

        d = self._screening_frame()
        summary = self.screening_summary(gain=gain)
        hour = d.index.hour + d.index.minute / 60
        fig = plt.figure(facecolor='white', figsize=(16, 5.6), layout='constrained')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4])
        ax_diel, ax_hour, ax_cum = (fig.add_subplot(gs[0, i]) for i in range(3))

        for col, color, label, width in (('unscreened', cp.KEPT, 'without screening', 2.6),
                                         ('used', cp.CLEANED, 'used (screened, filled)', 1.8)):
            diel = d[col].groupby(hour).mean()
            ax_diel.plot(diel.index, diel.to_numpy(), color=color, linewidth=width, label=label)
        ax_diel.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_diel.set_xlim(0, 24)
        ax_diel.set_xticks(range(0, 25, 3))
        ax_diel.set_xlabel('hour of the day')
        ax_diel.set_ylabel(f'mean {self.strgcol}')
        cp.style_axes(ax_diel)
        cp.panel_label(ax_diel, "Mean daily cycle")
        ax_diel.legend(loc='lower right', frameon=False, fontsize=8.5)

        removed = d.loc[d['spike'], 'unscreened']
        rh = removed.index.hour + removed.index.minute / 60
        bins = np.arange(0, 24.5, 1)
        pos = np.histogram(rh[removed.to_numpy() >= 0], bins=bins)[0]
        neg = np.histogram(rh[removed.to_numpy() < 0], bins=bins)[0]
        ax_hour.bar(bins[:-1] + 0.5, pos, width=0.85, color=cp.TEST_COLORS[0], label='positive')
        ax_hour.bar(bins[:-1] + 0.5, -neg, width=0.85, color=cp.TEST_COLORS[3], label='negative')
        ax_hour.axhline(0, color=cp.INK, linewidth=0.7)
        ax_hour.set_xlim(0, 24)
        ax_hour.set_xticks(range(0, 25, 3))
        ax_hour.set_xlabel('hour of the day')
        ax_hour.set_ylabel('spikes (negative below zero)')
        cp.style_axes(ax_hour)
        cp.panel_label(ax_hour, "Spikes by hour")
        ax_hour.legend(loc='upper right', frameon=False, fontsize=8.5)

        for col, color, label, width in (('unscreened', cp.KEPT, 'without screening', 2.6),
                                         ('used', cp.CLEANED, 'used (screened, filled)', 1.8)):
            cum = (d[col] * gain).cumsum()
            ax_cum.plot(cum.index, cum.to_numpy(), color=color, linewidth=width, label=label)
        ax_cum.axhline(0, color=cp.INK, linewidth=0.7, alpha=0.5)
        ax_cum.set_ylabel(f"cumulative {self.strgcol}{f' ({units})' if units else ''}")
        cp.style_axes(ax_cum)
        cp.panel_label(ax_cum, "Cumulative sum")
        ax_cum.legend(loc='best', frameon=False, fontsize=8.5)
        nice_date_ticks(ax=ax_cum)

        u = f" {units}" if units else ""
        cp.figure_title(
            fig, f"Storage term screening: {self.strgcol}",
            f"{summary['spikes']:,} spikes at measured fluxes, {100 * summary['spikes_negative']:.0f} % "
            f"negative; {summary['replaced']:,} values replaced by the mean of their block. "
            f"Sum without screening "
            f"{summary['sum_unscreened']:,.1f}{u}, used {summary['sum_used']:,.1f}{u}. Median absolute "
            f"daily sum {summary['daily_abs_median_unscreened']:.2f} -> {summary['daily_abs_median_used']:.2f}{u}.")
        self._fig = fig
        if showplot:
            plt.show()
        return fig

    def _screened_block_means(self) -> pd.Series:
        """The replacement values: for each spike block, the mean of its measured values.

        A block is a run of spikes plus one record on either side; blocks that touch merge. A
        neighbour without a measured value is a gap and stays out (the rolling median fills it).
        Replacing a block by its mean keeps its sum and removes the spike shape. A single spike
        is spread over three records. A concentration spike puts +X and -X into the storage
        term, next to each other or, with a central difference, one record apart; both lie in
        one block, which then becomes values near zero.
        """
        strg = self.results[self.strgcol]
        flagged = pd.Series(strg.index.isin(self.storage_outliers_), index=strg.index)
        block = (flagged | flagged.shift(1, fill_value=False) | flagged.shift(-1, fill_value=False))
        block &= strg.notna()
        block_id = (block != block.shift()).cumsum()
        means = strg.where(block).groupby(block_id).transform('mean')
        return means[block.to_numpy()]

    def _gapfill_storage_term(self) -> DataFrame:
        """Fill missing storage values using a rolling median.

        Fills gaps by applying a rolling median with an expanding window: starts at
        window size 3, then 5, 7, 9, etc., until all missing values are covered.

        Returns a DataFrame with two columns:
            - {gapfilled_strgcol}: storage term with gaps filled
            - {flag_isgapfilled}: flag marking original (0) vs filled (1) values
        """

        # New columns
        self.gapfilled_strgcol = f"{self.strgcol}_gfRMED{self.idstr}"
        self.flag_isgapfilled = f"FLAG_{self.gapfilled_strgcol}_ISFILLED"

        # Generate temporary subset where all flux values are available and check for missing
        # storage. Each spike block gets the mean of its measured values (flag 2): the storage
        # term is the change in concentration, so the sum over the block is the real
        # concentration change across it, and keeping it keeps the balance.
        strg_series = self.results[self.strgcol].copy()
        replacement = self._screened_block_means()
        self.storage_replaced_ = replacement.index
        strg_series[self.storage_replaced_] = replacement
        if len(self.storage_replaced_):
            info(f"Storage term {self.strgcol}: {len(self.storage_replaced_):,} value(s) replaced by "
                 f"the mean of their block (spikes and one record on either side)")
        gapfilled_df = pd.concat([self.results[self.fluxcol], self.results[self.strgcol]], axis=1)
        gapfilled_df[self.gapfilled_strgcol] = strg_series
        gapfilled_df[self.flag_isgapfilled] = 0
        gapfilled_df.loc[gapfilled_df.index.isin(self.storage_replaced_), self.flag_isgapfilled] = 2
        gapfilled_df = gapfilled_df.dropna(subset=[self.fluxcol])
        n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
        prev_n_still_missing_strg = n_still_missing_strg
        detail(f"Missing values for storage term {self.gapfilled_strgcol}: {n_still_missing_strg}")

        # Fill gaps with rolling median in expanding time window.
        # Two termination guards keep this from looping forever:
        #  - The rolling median requires min_periods=3 valid values within the
        #    window. If the storage column holds fewer than 3 valid values in
        #    total (all-NaN or very sparse), no window size can ever fill a gap,
        #    so skip the loop entirely and let the fallback below handle it.
        #  - Otherwise the window is capped at 2*len+1: a centered window that
        #    size already spans the entire series from any position, so once
        #    there are >=3 valid values the loop is guaranteed to fill every gap
        #    at or before the cap (it normally exits much earlier).
        n_total = len(strg_series)
        n_valid = int(strg_series.notna().sum())
        max_window = 2 * n_total + 1
        window_size = 0
        while n_valid >= 3 and n_still_missing_strg > 0 and window_size < max_window:
            window_size = 3 if window_size == 0 else window_size + 2
            rmedian = strg_series.rolling(window=window_size, center=True, min_periods=3).median()
            locs = gapfilled_df[self.gapfilled_strgcol].isnull()
            gapfilled_df.loc[locs, self.gapfilled_strgcol] = rmedian
            gapfilled_df.loc[locs, self.flag_isgapfilled] = 1
            n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
            if n_still_missing_strg < prev_n_still_missing_strg:
                detail(f"Gap-filling {self.strgcol} with rolling median "
                       f"(window={window_size})  |  still missing: {n_still_missing_strg}")
                prev_n_still_missing_strg = n_still_missing_strg

        # Every flux value must carry a storage term in the end. Whatever the
        # rolling median could not fill (sparse/all-NaN storage) falls back to 0,
        # i.e. no storage contribution, flagged as filled. Warn so it is not silent.
        if n_still_missing_strg > 0:
            locs = gapfilled_df[self.gapfilled_strgcol].isnull()
            n_fallback = int(locs.sum())
            gapfilled_df.loc[locs, self.gapfilled_strgcol] = 0.0
            gapfilled_df.loc[locs, self.flag_isgapfilled] = 1
            warn(f"Storage term '{self.strgcol}': {n_fallback} flux record(s) could not be "
                 f"gap-filled by rolling median (fewer than 3 valid storage values within "
                 f"reach); set to 0 so every flux value still carries a storage term.")

        gapfilled_df = gapfilled_df[[self.gapfilled_strgcol, self.flag_isgapfilled]].copy()

        return gapfilled_df

    def _detect_storage_var(self) -> tuple[str, str]:
        """Auto-detect the storage variable name and set the output column name.

        Maps each flux type to its storage term (FC→SC_SINGLE, LE→SLE_SINGLE, etc.).
        CO2 flux (FC) gets special naming—it outputs as NEE—while other fluxes output
        with the suffix {idstr} appended (e.g., LE_L3.1, H_L3.1).

        Returns a tuple of (corrected_flux_column_name, storage_variable_name).
        Example: ('NEE_L3.1', 'SC_SINGLE') for FC.
        """

        flux_corrected_col = None

        options = {
            'FC': 'SC_SINGLE',
            'FH2O': 'SH2O_SINGLE',
            'LE': 'SLE_SINGLE',
            'ET': 'SET_SINGLE',
            'FN2O': 'SN2O_SINGLE',
            'FCH4': 'SCH4_SINGLE',
            'H': 'SH_SINGLE'
        }
        if self.fluxcol == 'FC':
            flux_corrected_col = f'NEE{self.idstr}'

        # elif self.filetype == 'EDDYPRO-FULL-OUTPUT-CSV-30MIN':
        #     options = {
        #         'co2_flux': 'co2_strg',
        #         'h2o_flux': 'h2o_strg',
        #         'LE': 'LE_strg',
        #         'n2o_flux': 'n2o_strg',
        #         'ch4_flux': 'ch4_strg',
        #         'H': 'H_strg'
        #     }
        #     if self.fluxcol == 'co2_flux':
        #         flux_corrected_col = f'NEE{self.idstr}'

        strgcol = options[self.fluxcol]
        if not flux_corrected_col:
            flux_corrected_col = f"{self.fluxcol}{self.idstr}"
        detail(f"Detected storage variable {strgcol} for {self.fluxcol}.")
        return flux_corrected_col, strgcol

    def showplot(self, maxflux: float = None):
        """Display a 5-panel heatmap of the storage correction.

        Shows the original flux, storage-corrected flux, original storage term,
        gap-filled storage term (if enabled), and the fill flag (0=original, 1=gap-filled).

        Layout is day-of-year (y-axis) vs hour-of-day (x-axis), making temporal
        patterns visible.

        Args:
            maxflux: Maximum absolute value for colorbar scaling. If None, scales to
                the 95th percentile. Pass the same value across multiple plots to
                keep colors consistent.
        """

        fig = plt.figure(facecolor='white', figsize=(20, 9))
        gs = gridspec.GridSpec(1, 5)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.06, right=0.94, top=0.9, bottom=0.1)
        ax_flux = fig.add_subplot(gs[0, 0])
        ax_flux_storage_corrected = fig.add_subplot(gs[0, 1], sharey=ax_flux)
        ax_storage_term = fig.add_subplot(gs[0, 2], sharey=ax_flux)
        ax_storage_term_gf = fig.add_subplot(gs[0, 3], sharey=ax_flux)
        ax_storage_term_flag = fig.add_subplot(gs[0, 4], sharey=ax_flux)

        if not maxflux:
            maxflux = self.results[self.fluxcol].abs().quantile(.95)
            maxstrg = (maxflux if self.strgcol is None
                       else self.results[self.strgcol].abs().quantile(.95))
        else:
            maxstrg = maxflux

        HeatmapDateTime(series=self.results[self.fluxcol]).plot(ax=ax_flux, vmin=-maxflux, vmax=maxflux,
                        cb_digits_after_comma=0)
        HeatmapDateTime(series=self.results[self.flux_corrected_col]).plot(ax=ax_flux_storage_corrected, vmin=-maxflux,
                        vmax=maxflux, cb_digits_after_comma=0)
        if not self.set_storage_to_zero:
            HeatmapDateTime(series=self.results[self.strgcol]).plot(ax=ax_storage_term, vmin=-maxstrg, vmax=maxstrg,
                            cb_digits_after_comma=1)
        else:
            _allzeros = pd.Series(0, index=self.results.index, name='zero_storage')
            HeatmapDateTime(series=_allzeros).plot(ax=ax_storage_term,
                            cb_digits_after_comma=0)
        if self.gapfilled_strgcol:
            HeatmapDateTime(series=self.results[self.gapfilled_strgcol]).plot(ax=ax_storage_term_gf, vmin=-maxstrg,
                            vmax=maxstrg,
                            cb_digits_after_comma=1)
            HeatmapDateTime(series=self.results[self.flag_isgapfilled]).plot(ax=ax_storage_term_flag,
                            cb_digits_after_comma=0)

        plt.setp(ax_flux_storage_corrected.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term_gf.get_yticklabels(), visible=False)
        plt.setp(ax_storage_term_flag.get_yticklabels(), visible=False)

        ax_flux_storage_corrected.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term_gf.axes.get_yaxis().get_label().set_visible(False)
        ax_storage_term_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()
