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
from diive.core.utils.console import console as _console, detail, info, rule


class FluxStorageCorrectionSinglePointEddyPro:
    """Add a storage correction term to eddy covariance flux measurements.

    Applies storage flux correction at Level-3.1 in Swiss FluxNet processing.
    The storage term—the change in gas or heat concentration over time—is added
    to the measured eddy covariance flux.

    Automatically detects the storage variable from flux type, optionally gap-fills
    missing values using a rolling median, and tracks which values were filled.

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
                 set_storage_to_zero: bool = False):
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
                Default False.

        Attributes:
            results: DataFrame with corrected flux and gap-filling info.
            flux_corrected_col: Name of the corrected flux column.
            gapfilled_strgcol: Name of the gap-filled storage column (if enabled).
            flag_isgapfilled: Flag column (0=original, 1=gap-filled).

        Example:
            >>> corrector = FluxStorageCorrectionSinglePointEddyPro(
            ...     df=data, fluxcol='FC', basevar='CO2',
            ...     gapfill_storage_term=True, idstr='_L3.1'
            ... )
            >>> corrector.storage_correction()
            >>> corrected_data = corrector.results
        """
        self.df = df.copy()
        self.fluxcol = fluxcol
        self.basevar = basevar
        self.gapfill_storage_term = gapfill_storage_term
        self.idstr = validate_id_string(idstr=idstr)
        self.flux_corrected_col, self.strgcol = self._detect_storage_var()
        # Note: this class deliberately does NOT emit a 'storage missing'
        # quality-test flag (i.e. no FLAG_..._TEST column tied to NaN in the
        # corrected flux). Whether a storage term was measured or gap-filled
        # is provenance, not a quality criterion — captured by the
        # FLAG_..._ISFILLED column when gap-fill is enabled. Quality
        # aggregation at L3.1 inherits the L2 flags via FlagQCF.
        self.set_storage_to_zero = set_storage_to_zero

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
        info(f"Calculating storage-corrected flux {self.flux_corrected_col} "
             f"from flux {self.fluxcol} + storage term {self.strgcol}")

        # Collect flux and storage term data
        self._results = self.df[[self.fluxcol, self.strgcol]].copy()

        if not self.set_storage_to_zero:
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
            locs_isfilled = fluxavailable[self.flag_isgapfilled] == 1
            locs_isorig = fluxavailable[self.flag_isgapfilled] == 0

            n_isfilled = len(fluxavailable[locs_isfilled])
            n_isorig = len(fluxavailable[locs_isorig])
            n_flux_corrected = len(fluxavailable[self.flux_corrected_col].dropna())
            perc_recovered = (n_isfilled / n_orig_missing_strg * 100) if n_orig_missing_strg > 0 else 0

            info(f"Measured flux: {n_flux:,}  |  Storage available: {n_storageterm:,}  |  Missing: {n_orig_missing_strg:,}")
            info(f"Gap-fill: recovered {n_isfilled:,}  |  Recovery rate: {perc_recovered:.1f}%")
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

        # Generate temporary subset where all flux values are available and check for missing storage
        gapfilled_df = pd.concat([self.results[self.fluxcol], self.results[self.strgcol]], axis=1)
        gapfilled_df[self.gapfilled_strgcol] = self.results[self.strgcol].copy()
        gapfilled_df[self.flag_isgapfilled] = 0
        gapfilled_df = gapfilled_df.dropna(subset=[self.fluxcol])
        n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
        prev_n_still_missing_strg = n_still_missing_strg
        detail(f"Missing values for storage term {self.gapfilled_strgcol}: {n_still_missing_strg}")

        # Fill gaps with rolling mean in expanding time window
        window_size = 0
        while n_still_missing_strg > 0:
            window_size = 3 if window_size == 0 else window_size + 2
            rmedian = self.results[self.strgcol].rolling(window=window_size, center=True, min_periods=3).median()
            locs = gapfilled_df[self.gapfilled_strgcol].isnull()
            gapfilled_df.loc[locs, self.gapfilled_strgcol] = rmedian
            gapfilled_df.loc[locs, self.flag_isgapfilled] = 1
            n_still_missing_strg = gapfilled_df[self.gapfilled_strgcol].isnull().sum()
            if n_still_missing_strg < prev_n_still_missing_strg:
                detail(f"Gap-filling {self.strgcol} with rolling median "
                       f"(window={window_size})  |  still missing: {n_still_missing_strg}")
                prev_n_still_missing_strg = n_still_missing_strg

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

        Example:
            >>> corrector.showplot(maxflux=5)
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
            maxstrg = self.results[self.strgcol].abs().quantile(.95)
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
            _allzeros = pd.Series(0, index=self.results[self.strgcol].index, name='zero_storage')
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
