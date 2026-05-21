"""
LAG_PWB_BATCH: PARALLEL BATCH PWB TIME-LAG DETECTION
=====================================================

Runs ``PreWhiteningBootstrap`` (from ``lag_pwb.py``) across many
averaging-period files in parallel using ``ProcessPoolExecutor``.

Why this module exists
----------------------
A single call to ``PreWhiteningBootstrap.run()`` takes several seconds per
30-minute file at 20 Hz with 99 bootstrap replicates.  Processing a season
or year of data sequentially would take hours.  This module distributes the
work across CPU cores so that N workers process N files simultaneously,
reducing wall time by up to N times.

Architecture
------------
The worker function ``_pwb_file_worker`` is defined at module scope (not
inside a class or nested function) because ``ProcessPoolExecutor`` on Windows
uses the *spawn* start method: each worker is a fresh Python process that
re-imports the module, and only module-level objects are picklable.  The
worker loads one file, runs ``PreWhiteningBootstrap`` for each requested
scalar, and returns a single result dict.  All file I/O and plotting happen
inside the worker; the main process only collects dicts and writes the
checkpoint CSV.

Checkpoint CSV
--------------
After every completed file the main process writes a checkpoint CSV
(``tlag_results_checkpoint.csv``) so that a crash can be diagnosed and the
run can be restarted from the last partial state.  If the file is locked by
another application (e.g. Excel) the checkpoint write is silently skipped;
the final ``tlag_results.csv`` is still written at the end.

PWBOPT post-processing
----------------------
After all files complete, ``apply_pwbopt()`` assigns an S1/S2/S3 flag to
each period following Vitale et al. (2024) Section 2.3:

- **S1** — HDI range < ``hdi_thresh`` (default 0.5 s): detection is
  reliable; the lag is accepted directly.
- **S2** — HDI is wide but the lag is within ``dev_thresh`` (default 0.5 s)
  of the most recent S1/S2 lag: accepted for temporal continuity.
- **S3** — detection is unreliable; the preceding optimal lag is carried
  forward.

``apply_hdi_prefilter()`` optionally discards lags whose HDI exceeds a
threshold before PWBOPT runs, preventing S2 from accepting wide-uncertainty
detections that happen to be near the preceding optimal.

Summary figures
---------------
``plot_summary()`` generates per-scalar 5-panel figures and a
``PwboptLagPlot`` scatter/KDE comparison across all scalars:

1. **Detected lags** — scatter coloured by S1/S2/S3 flag + mode reference line.
2. **Final lags** — S1/S2 anchor points (filled, coloured) overlaid with the
   pre-filtered gap-filled lag as open black circles + mode reference line.
   This is the series written to ``{prefix}_tlag_final_pf_s`` and used
   directly for flux covariance calculation.
3. **HDI range** — bootstrap uncertainty bars with S1 and pre-filter thresholds.
4. **Flag bars** — standard vs. pre-filtered PWBOPT flags side by side.
5. **Lag histogram** — distribution of all detected lags with exact mode.

The mode is computed from rounded value counts — not histogram bin centers —
to match the discrete 1/hz lag resolution (e.g. 0.05 s at 20 Hz).

``fill_tlag_gaps()`` is called automatically in ``_cli_main()`` after
PWBOPT, adding ``{prefix}_tlag_final_s`` and ``{prefix}_tlag_final_pf_s``
to the results CSV so that every averaging period has a usable lag for
flux calculation (leading NaN periods are backward-filled from the first
reliable detection).

Input data requirement
----------------------
Input files must contain **wind-rotation-corrected** high-frequency data
(double rotation or planar-fit correction applied, e.g. by EddyPro's
"Advanced" processing with output to rotated files).  Wind rotation removes
the mean vertical-wind component so that W contains only turbulent
fluctuations.  Without rotation the mean W offset corrupts the
cross-correlation, and the detected lag is unreliable.

CLI entry point
---------------
The module is directly executable::

    python -m diive.pkgs.flux.hires.lag_pwb_batch --help

All detection parameters (``--n-bootstrap``, ``--lag-max``, ``--hz``, …),
PWBOPT thresholds (``--hdi-thresh``, ``--dev-thresh``, ``--hdi-prefilter``),
and output options (``--save-plots``) are exposed as CLI flags.  A Rich
progress bar with per-file log lines is shown during the run; summary
figures are written to ``--output-dir`` automatically.

Windows note
------------
``ProcessPoolExecutor`` uses the *spawn* start method on Windows.  Any
script that instantiates ``PwbBatchDetection`` must guard its entry point::

    if __name__ == '__main__':
        det = PwbBatchDetection(...)
        results = det.run()

References:
    Vitale D, Fratini G, Helfter C, Hortnagl L, et al. (2024) A pre-whitening
    with block-bootstrap cross-correlation procedure for temporal alignment of
    data sampled by eddy covariance systems. Environmental and Ecological
    Statistics 31:219-244. doi:10.1007/s10651-024-00615-9

Part of the diive library: https://github.com/holukas/diive
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

# Suppress the runpy double-import warning that fires in every worker process
# when diive.__init__ has already imported this module before -m re-executes it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from diive.pkgs.flux.hires.lag_pwb import PreWhiteningBootstrap, PwboptLagPlot

# Default NA strings matching EddyPro rotated-file conventions
_DEFAULT_NA_VALUES = ['-9999', '-9999.0', '-9999.0000000000000']


# ---------------------------------------------------------------------------
# Module-level worker (must be defined at module scope to be picklable)
# ---------------------------------------------------------------------------

def _pwb_file_worker(args: tuple) -> dict:
    """Process one averaging-period file and return one result-row dict."""
    (filepath, scalars, col_w, col_tsonic,
     hz, lag_max_s, n_bootstrap, block_length_s,
     usecols, col_names, skiprows, na_values,
     min_valid_frac, plot_dir, save_plots) = args

    period_name = Path(filepath).name
    row: dict = {'period': period_name}

    # Load file
    try:
        df = pd.read_csv(
            filepath,
            skiprows=skiprows + 1,
            header=None,
            sep=r'\s+',
            na_values=na_values,
            low_memory=False,
        )
        df = df.iloc[:, usecols].copy()
        df.columns = col_names[:len(df.columns)]
    except Exception:
        return row

    if len(df) < 25:
        return row

    # Wind validity guard
    w_arr = np.asarray(df[col_w], dtype=float)
    if (np.mean(~np.isnan(w_arr)) < min_valid_frac
            or np.nanstd(w_arr) < np.finfo(float).eps):
        return row

    for scalar_label, scalar_col in scalars.items():
        prefix = scalar_label.lower()
        _nan_keys = (
            'tlag_s', 'hdi_lo_s', 'hdi_hi_s',
            'hdi_range_s', 'tlag_pw_s', 'corr_est', 'ar_order',
        )
        nan_row = {f'{prefix}_{k}': np.nan for k in _nan_keys}

        if scalar_col not in df.columns:
            row.update(nan_row)
            continue

        s_arr = np.asarray(df[scalar_col], dtype=float)
        if (np.mean(~np.isnan(s_arr)) < min_valid_frac
                or np.nanstd(s_arr) < np.finfo(float).eps):
            row.update(nan_row)
            continue

        has_ts = col_tsonic is not None and col_tsonic in df.columns
        col_map = {col_w: 'W', scalar_col: scalar_label}
        keep_cols = [col_w, scalar_col]
        if has_ts:
            col_map[col_tsonic] = 'T_SONIC'
            keep_cols.append(col_tsonic)

        try:
            pwb = PreWhiteningBootstrap(
                df=df[keep_cols].rename(columns=col_map),
                var_w='W',
                var_scalar=scalar_label,
                var_tsonic='T_SONIC' if has_ts else None,
                hz=hz,
                lag_max_s=lag_max_s,
                n_bootstrap=n_bootstrap,
                block_length_s=block_length_s,
                segment_name=period_name,
            )
            pwb.run()
            res = pwb.results

            row[f'{prefix}_tlag_s'] = res['tlag_s']
            row[f'{prefix}_hdi_lo_s'] = res['hdi_lo_s']
            row[f'{prefix}_hdi_hi_s'] = res['hdi_hi_s']
            row[f'{prefix}_hdi_range_s'] = res['hdi_range_s']
            row[f'{prefix}_tlag_pw_s'] = res['tlag_pw_s']
            row[f'{prefix}_corr_est'] = res['corr_est']
            row[f'{prefix}_ar_order'] = res['ar_order']

            if save_plots and plot_dir:
                fig = pwb.plot(
                    title=f'{period_name} | {scalar_label}',
                    showplot=False,
                )
                fig.savefig(
                    Path(plot_dir) / f'{Path(period_name).stem}_{prefix}.png',
                    dpi=100, bbox_inches='tight',
                )
                plt.close(fig)

        except Exception:
            row.update(nan_row)

    return row


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PwbBatchDetection:
    """
    Parallel batch PWB time-lag detection across many averaging-period files.

    .. important::
        Input files must contain **wind-rotation-corrected** data (double
        rotation or planar-fit applied, e.g. EddyPro "Advanced" rotated
        output).  Wind rotation removes the mean vertical-wind offset so that
        W contains only turbulent fluctuations.  Without rotation the
        cross-correlation is biased and the detected lag is unreliable.

    Distributes ``PreWhiteningBootstrap`` across CPU cores using
    ``ProcessPoolExecutor``.  Results accumulate into a DataFrame; an optional
    checkpoint CSV is written after every completed file so that a crash can be
    resumed from the last checkpoint.

    PWBOPT post-processing (S1/S2/S3 selection and optional HDI pre-filter)
    from Vitale et al. (2024) Section 2.3 is available via the static methods
    ``apply_pwbopt()`` and ``apply_hdi_prefilter()``.

    Windows note: the calling script must guard its entry point with
    ``if __name__ == '__main__':`` because ``ProcessPoolExecutor`` uses the
    *spawn* start method on Windows.

    Parameters
    ----------
    files:
        Paths to averaging-period files (one file per 30-min period).
    scalars:
        ``{gas_label: column_name}`` mapping, e.g.
        ``{'CH4': 'ch4', 'N2O': 'n2o'}``.
    col_w:
        Column name for vertical wind W after loading and renaming.
    col_tsonic:
        Column name for sonic temperature T_SONIC, or ``None`` for
        2-combination mode (cw and wc only).
    hz:
        Sampling frequency in Hz. Defaults to 20.
    lag_max_s:
        CCF search half-width in seconds. Defaults to 10.0.
    n_bootstrap:
        Number of block-bootstrap replicates. Defaults to 99.
    block_length_s:
        Bootstrap block length in seconds. Defaults to 20.0.
    usecols:
        0-based column indices to select from each file.
    col_names:
        Column names to assign after selecting *usecols*.
    skiprows:
        Metadata rows before the column-name row (EddyPro default: 9).
    na_values:
        Strings to treat as NaN. Defaults to EddyPro conventions.
    min_valid_frac:
        Minimum fraction of non-NaN values for a series to be processed.
        Defaults to 0.3.
    output_dir:
        Optional directory for checkpoint CSV and diagnostic plots.
    save_plots:
        Save one diagnostic PNG per period per scalar. Requires *output_dir*.
    n_workers:
        Number of parallel worker processes. Defaults to ``os.cpu_count()``.

    Example
    -------
    See ``examples/flux/hires/flux_lag_pwb_batch.py`` for a complete example.

    See Also
    --------
    PreWhiteningBootstrap : Single-period PWB detection.
    """

    def __init__(
            self,
            files: list,
            scalars: dict,
            col_w: str,
            col_tsonic: str | None = None,
            hz: int = 20,
            lag_max_s: float = 10.0,
            n_bootstrap: int = 99,
            block_length_s: float = 20.0,
            usecols: list | None = None,
            col_names: list | None = None,
            skiprows: int = 9,
            na_values: list | None = None,
            min_valid_frac: float = 0.3,
            output_dir: Path | None = None,
            save_plots: bool = False,
            n_workers: int | None = None,
    ):
        if usecols is None or col_names is None:
            raise ValueError("usecols and col_names must both be provided.")
        if len(usecols) != len(col_names):
            raise ValueError("usecols and col_names must have the same length.")

        self.files = [Path(f) for f in files]
        self.scalars = scalars
        self.col_w = col_w
        self.col_tsonic = col_tsonic
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.usecols = usecols
        self.col_names = col_names
        self.skiprows = skiprows
        self.na_values = na_values if na_values is not None else _DEFAULT_NA_VALUES
        self.min_valid_frac = min_valid_frac
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_plots = save_plots
        self.n_workers = n_workers or os.cpu_count()

        self._results: DataFrame | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def results(self) -> DataFrame:
        """Accumulated results DataFrame (available after ``run()``)."""
        if self._results is None:
            raise RuntimeError("Call run() first.")
        return self._results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, on_progress: Callable | None = None) -> DataFrame:
        """
        Execute PWB detection across all files in parallel.

        Files are submitted to ``ProcessPoolExecutor`` in one batch.  Results
        are collected in completion order and then sorted back to the original
        file order before being returned.

        Args:
            on_progress: Optional callback ``f(completed: int, total: int, row: dict)``
                called each time a file completes.  *row* is the result dict for
                that file, allowing callers to display preliminary results live.

        Returns:
            DataFrame with one row per file.  Columns: ``period``,
            ``{prefix}_tlag_s``, ``{prefix}_hdi_range_s``, etc.
        """
        plot_dir = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_plots:
                plot_dir = self.output_dir / 'plots'
                plot_dir.mkdir(exist_ok=True)

        worker_args = [
            (
                str(f),
                self.scalars, self.col_w, self.col_tsonic,
                self.hz, self.lag_max_s, self.n_bootstrap, self.block_length_s,
                self.usecols, self.col_names, self.skiprows, self.na_values,
                self.min_valid_frac,
                str(plot_dir) if plot_dir else None,
                self.save_plots,
            )
            for f in self.files
        ]

        rows: list[dict] = []
        total = len(worker_args)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(_pwb_file_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_file):
                row = future.result()
                rows.append(row)
                completed += 1

                if self.output_dir:
                    try:
                        pd.DataFrame(rows).to_csv(
                            self.output_dir / 'tlag_results_checkpoint.csv',
                            index=False,
                        )
                    except PermissionError:
                        pass  # file open in another program; skip this checkpoint

                if on_progress is not None:
                    on_progress(completed, total, row)

        # Restore original file order
        name_to_idx = {Path(f).name: i for i, f in enumerate(self.files)}
        rows.sort(key=lambda r: name_to_idx.get(r.get('period', ''), total))

        self._results = pd.DataFrame(rows)
        if self.output_dir:
            self._results.to_csv(
                self.output_dir / 'tlag_results.csv',
                index=False,
            )

        return self._results

    # ------------------------------------------------------------------
    # PWBOPT post-processing (static: usable standalone or via instance)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_pwbopt(
            tlag_s,
            hdi_range_s,
            hdi_thresh: float = 0.5,
            dev_thresh: float = 0.5,
    ) -> DataFrame:
        """
        Apply PWBOPT S1/S2/S3 selection to a sequence of PWB lag estimates.

        Port of ``apply_pwbopt()`` from ``01_tlag_detection_pwb.R`` (lines
        73-107).  Processes periods sequentially; ``last_optimal`` carries
        across periods so earlier reliable detections anchor later S2 selections.

        S1 -- HDI range < *hdi_thresh* -> reliable, accept directly.
        S2 -- uncertain but within *dev_thresh* of the preceding optimal lag
              -> accept for temporal continuity.
        S3 -- unreliable -> carry forward the last known optimal lag.

        Args:
            tlag_s: Detected PWB lags in seconds (NaN where detection failed).
            hdi_range_s: 95% HDI range in seconds per period.
            hdi_thresh: S1 threshold (default 0.5 s).
            dev_thresh: S2 max deviation from the preceding optimal (default 0.5 s).

        Returns:
            DataFrame with columns ``pwbopt_s`` (optimal lag, s) and ``flag``.
        """
        tlag_s = np.asarray(tlag_s, dtype=float)
        hdi_range_s = np.asarray(hdi_range_s, dtype=float)
        n = len(tlag_s)
        flags = ['S3_unreliable'] * n
        optimal = np.full(n, np.nan)
        last_optimal = np.nan

        for i in range(n):
            tl = tlag_s[i]
            hdi = hdi_range_s[i]

            if np.isnan(tl) or np.isnan(hdi):
                optimal[i] = last_optimal
                continue

            if hdi < hdi_thresh:
                flags[i] = 'S1_optimal'
                optimal[i] = tl
                last_optimal = tl
            elif not np.isnan(last_optimal) and abs(tl - last_optimal) <= dev_thresh:
                flags[i] = 'S2_optimal'
                optimal[i] = tl
                last_optimal = tl
            else:
                optimal[i] = last_optimal

        return pd.DataFrame({'pwbopt_s': optimal, 'flag': flags})

    @staticmethod
    def fill_tlag_gaps(
            pwbopt_s,
            tlag_s_raw=None,
            fallback: float | None = None,
    ) -> np.ndarray:
        """
        Fill any remaining NaN values in a PWBOPT lag series so that every
        averaging period has a usable time lag for flux covariance calculation.

        Why NaN values remain after ``apply_pwbopt``
        --------------------------------------------
        PWBOPT carries the last known optimal lag forward in time.  Periods
        *before* the first S1/S2 detection have nothing to carry forward, so
        ``pwbopt_s`` is NaN for those leading periods.  Periods after the last
        S1/S2 detection (e.g. end-of-season low-flux episodes) are already
        filled by the forward carry.

        Fill strategy (applied in order):
        1. **Backward fill** — propagates the first reliable lag backward to
           cover the leading NaN periods.
        2. **Median of raw lags** — if the entire series is NaN (no S1/S2
           detection at all), the median of all non-NaN values in *tlag_s_raw*
           is used as a constant fallback.
        3. **Explicit fallback** — if *fallback* is provided it overrides the
           median and is used when both bfill and median leave NaN (e.g. raw
           lags are also entirely NaN).

        Args:
            pwbopt_s: Optimal lag series from ``apply_pwbopt()`` (NaN where
                no carry-forward value was available).
            tlag_s_raw: Raw PWB detected lags before PWBOPT.  Used only to
                compute a median fallback when ``pwbopt_s`` is entirely NaN.
                Ignored when ``None``.
            fallback: Constant lag in seconds used as last resort when all
                other strategies leave NaN.  Typically the nominal tube-delay
                for the gas/site.

        Returns:
            Array of the same length as *pwbopt_s* with no NaN values
            (unless *fallback* is ``None`` and no finite value can be found).
        """
        result = pd.Series(np.asarray(pwbopt_s, dtype=float))

        # 1. backward fill: first reliable optimal propagates to leading NaNs
        result = result.bfill()

        # 2. if still NaN (entire series unreliable), use median of raw lags
        if result.isna().any() and tlag_s_raw is not None:
            raw = np.asarray(tlag_s_raw, dtype=float)
            median_raw = np.nanmedian(raw)
            if np.isfinite(median_raw):
                result = result.fillna(median_raw)

        # 3. last resort: user-supplied constant
        if result.isna().any() and fallback is not None:
            result = result.fillna(fallback)

        return result.to_numpy()

    @staticmethod
    def apply_hdi_prefilter(
            tlag_s,
            hdi_range_s,
            threshold: float = 1.0,
    ) -> np.ndarray:
        """
        Replace lags whose HDI exceeds *threshold* with NaN before PWBOPT.

        Port of ``apply_hdi_prefilter()`` from
        ``02_tlag_compare_pwbopt_strategies.R`` (lines 96-101).

        More conservative than standard PWBOPT: wide-HDI detections are
        discarded before S1/S2/S3 runs, so S2 cannot accept them even if they
        happen to lie close to the preceding optimal lag.

        Args:
            tlag_s: Detected lags in seconds.
            hdi_range_s: 95% HDI range in seconds.
            threshold: Lags with HDI range wider than this are set to NaN.

        Returns:
            Array of pre-filtered lags (NaN where HDI > threshold).
        """
        tlag_filtered = np.asarray(tlag_s, dtype=float).copy()
        hdi = np.asarray(hdi_range_s, dtype=float)
        tlag_filtered[(hdi > threshold) & ~np.isnan(hdi)] = np.nan
        return tlag_filtered

    @staticmethod
    def plot_summary(
            results: DataFrame,
            scalars: dict,
            hdi_thresh: float = 0.5,
            hdi_prefilter: float = 1.0,
            lag_max_s: float = 10.0,
            output_dir: Path | None = None,
            showplot: bool = False,
    ) -> None:
        """
        Generate batch-level summary figures after PWBOPT post-processing.

        Produces one 5-panel figure per scalar and one scatter + KDE comparison
        figure across all scalars (``PwboptLagPlot``).  Figures are saved to
        *output_dir* when provided.

        Panel layout:

        1. Detected lags coloured by S1/S2/S3 flag (scatter, no lines) + mode.
        2. Final gap-filled lags: S1/S2 anchor points (filled markers) +
           pre-filtered final lag as open black circles + mode.
        3. 95% HDI range bars with S1 and pre-filter threshold lines.
        4. Flag bars per period: standard vs. pre-filtered side by side.
        5. Histogram of all detected lags with mode marker.

        Expects the *results* DataFrame to already contain the PWBOPT columns
        added by ``apply_pwbopt()`` / ``apply_hdi_prefilter()``:
        ``{prefix}_flag_std``, ``{prefix}_pwbopt_s_std``, and optionally
        ``{prefix}_flag_pf`` / ``{prefix}_pwbopt_s_pf`` /
        ``{prefix}_tlag_final_pf_s``.

        Args:
            results: Per-period results DataFrame (one row per file).
            scalars: ``{gas_label: column_name}`` mapping used during detection,
                e.g. ``{'CH4': 'ch4', 'N2O': 'n2o'}``.
            hdi_thresh: S1 HDI threshold in seconds (used for plot label only).
            hdi_prefilter: HDI pre-filter threshold in seconds (0 = disabled).
            lag_max_s: CCF half-width in seconds — sets y-axis limits.
            output_dir: Directory for saved PNGs.  ``None`` skips saving.
            showplot: Call ``plt.show()`` after each figure.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        FLAG_COLORS = {
            'S1_optimal': '#2ca02c',
            'S2_optimal': '#ff7f0e',
            'S3_unreliable': '#d62728',
        }
        px = np.arange(len(results))
        out = Path(output_dir) if output_dir else None

        for scalar_label in scalars:
            prefix = scalar_label.lower()
            tlag_col = f'{prefix}_tlag_s'
            hdi_col = f'{prefix}_hdi_range_s'
            flag_std_col = f'{prefix}_flag_std'
            flag_pf_col = f'{prefix}_flag_pf'
            opt_std_col = f'{prefix}_pwbopt_s_std'
            opt_pf_col = f'{prefix}_pwbopt_s_pf'

            if tlag_col not in results.columns or flag_std_col not in results.columns:
                continue

            tlag = results[tlag_col].values.astype(float)
            hdi = results[hdi_col].values.astype(float)
            flag_std = results[flag_std_col].values
            opt_std = results[opt_std_col].values.astype(float)
            has_pf = flag_pf_col in results.columns
            valid_lags = tlag[~np.isnan(tlag)]

            final_std_col = f'{prefix}_tlag_final_s'
            final_pf_col = f'{prefix}_tlag_final_pf_s'
            has_final_std = final_std_col in results.columns
            has_final_pf = final_pf_col in results.columns

            # Most frequent lag: exact value count on lags rounded to 2 decimal
            # places (PWB lags are discrete at 1/hz steps, e.g. 0.05 s at 20 Hz,
            # so rounding to 0.01 s preserves the true resolution without binning
            # artefacts that shift the apparent mode).
            if len(valid_lags) > 0:
                mode_lag = (pd.Series(np.round(valid_lags, 2))
                            .value_counts().idxmax())
            else:
                mode_lag = np.nan

            fig, axes = plt.subplots(
                5, 1, figsize=(14, 17),
                gridspec_kw={'height_ratios': [3, 2.5, 2, 1.5, 2]},
            )
            fig.suptitle(
                f'{scalar_label} -- PWB lag pipeline (PWBOPT strategy comparison)',
                fontsize=12,
            )

            # Panel 1: raw detected lags coloured by S1/S2/S3, no connecting lines
            ax = axes[0]
            ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
            for flag, color in FLAG_COLORS.items():
                mask = flag_std == flag
                ax.scatter(px[mask], tlag[mask], color=color, s=50, zorder=3,
                           label=flag)
            if not np.isnan(mode_lag):
                ax.axhline(mode_lag, color='#9467bd', linewidth=1.2,
                           linestyle='-.', zorder=2,
                           label=f'mode = {mode_lag:.2f} s')
            ax.set_ylabel('Time lag (s)')
            ax.set_title('Detected lags per period (coloured by standard PWBOPT flag)')
            ax.legend(frameon=False, fontsize=8, ncol=4)
            ax.set_ylim(-lag_max_s - 0.5, lag_max_s + 0.5)

            # Panel 2: final (gap-filled) lags + S1/S2 markers showing the anchor points
            ax = axes[1]
            ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
            # S1 and S2 scatter: the reliable detected values that anchor the fill
            for flag in ('S1_optimal', 'S2_optimal'):
                mask = flag_std == flag
                ax.scatter(px[mask], tlag[mask], color=FLAG_COLORS[flag], s=50,
                           zorder=4, label=flag)
            # Pre-filtered final lag as open black circles
            if has_final_pf:
                final_pf = results[final_pf_col].values.astype(float)
                ax.scatter(px, final_pf, color='none', edgecolors='black',
                           linewidths=1.0, s=50, zorder=3,
                           label='Final lag — pre-filtered')
            if not np.isnan(mode_lag):
                ax.axhline(mode_lag, color='#9467bd', linewidth=1.2,
                           linestyle='-.', zorder=2,
                           label=f'mode = {mode_lag:.2f} s')
            ax.set_ylabel('Time lag (s)')
            ax.set_title('Final (gap-filled) lags used for flux calculation')
            ax.legend(frameon=False, fontsize=8, ncol=3)
            ax.set_ylim(-lag_max_s - 0.5, lag_max_s + 0.5)

            # Panel 3: HDI range with S1 and pre-filter threshold lines
            ax = axes[2]
            ax.bar(px, hdi, color='#aec7e8', edgecolor='none', label='HDI range')
            ax.axhline(hdi_thresh, color='#2ca02c', linewidth=1.5, linestyle='--',
                       label=f'S1 threshold ({hdi_thresh} s)')
            if hdi_prefilter > 0:
                ax.axhline(hdi_prefilter, color='steelblue', linewidth=1.5,
                           linestyle=':',
                           label=f'Pre-filter threshold ({hdi_prefilter} s)')
            ax.set_ylabel('95% HDI range (s)')
            ax.set_title('Bootstrap uncertainty (HDI range) per period')
            ax.legend(frameon=False, fontsize=8)

            # Panel 4: side-by-side flag bars (standard left, pre-filtered right)
            ax = axes[3]
            bar_w = 0.4
            flag_cols = [(flag_std_col, 0)]
            if has_pf:
                flag_cols.append((flag_pf_col, 1))
            for flag_col, offset in flag_cols:
                for p in px:
                    flag = results[flag_col].iloc[p]
                    ax.bar(p + (offset - 0.5) * bar_w, 1, bar_w,
                           color=FLAG_COLORS.get(flag, '#aaaaaa'), alpha=0.85)
            patches = [mpatches.Patch(color=c, label=f)
                       for f, c in FLAG_COLORS.items()]
            ax.legend(handles=patches, frameon=False, fontsize=8)
            ax.set_yticks([])
            panel4_title = (
                'Flag per period: standard (left bar) vs. pre-filtered (right bar)'
                if has_pf else 'Flag per period'
            )
            ax.set_title(panel4_title)

            # Panel 5: histogram of detected lags
            ax = axes[4]
            ax.set_xlabel('Time lag (s)')
            ax.set_title(f'Distribution of detected lags  (n={len(valid_lags)})')
            if len(valid_lags) > 0:
                n_bins = min(50, max(10, len(valid_lags) // 4))
                ax.hist(valid_lags, bins=n_bins, range=(-lag_max_s, lag_max_s),
                        color='#aec7e8', edgecolor='white', linewidth=0.4)
                ax.axvline(0, color='#888888', linewidth=0.8, linestyle='-')
                ax.axvline(mode_lag, color='#9467bd', linewidth=1.5,
                           linestyle='-.', label=f'mode = {mode_lag:.2f} s')
                ax.legend(frameon=False, fontsize=8)
            ax.set_ylabel('Count')

            plt.tight_layout()
            if out:
                fig.savefig(out / f'summary_{prefix}.png', dpi=100,
                            bbox_inches='tight')
            if showplot:
                plt.show()
            else:
                plt.close(fig)

        # PwboptLagPlot: scatter + KDE — only when both std and pf columns exist
        scalars_plot = {}
        for label in scalars:
            pfx = label.lower()
            col_a, col_b = f'{pfx}_pwbopt_s_std', f'{pfx}_pwbopt_s_pf'
            if col_a in results.columns and col_b in results.columns:
                scalars_plot[label] = {'col_a': col_a, 'col_b': col_b}

        if scalars_plot:
            lag_plot = PwboptLagPlot(
                results=results,
                scalars=scalars_plot,
                label_a='PWBOPT standard',
                label_b='PWBOPT pre-filtered',
                color_a='#0072B2',
                color_b='#E05C2A',
            )
            lag_plot.plot(
                title='PWB optimal lag: standard vs. pre-filtered PWBOPT',
                showplot=showplot,
                outpath=str(out) if out else None,
                outname='summary_lag_comparison.png',
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m diive.pkgs.flux.hires.lag_pwb_batch',
        description='Parallel PWB time-lag detection across EddyPro high-frequency files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument('--input-dir', required=True,
                   help='Directory containing EddyPro rotated .txt files.')
    p.add_argument('--file-pattern', default='*.txt',
                   help='Glob pattern for input files.')
    p.add_argument('--output-dir', required=True,
                   help='Directory for results CSV, checkpoint, and optional plots.')
    # Scalars
    p.add_argument('--scalar', dest='scalars', action='append',
                   metavar='LABEL:column', required=True,
                   help='Gas label and column name, e.g. CH4:ch4. Repeat for each gas.')
    # Column mapping
    p.add_argument('--col-w', default='w',
                   help='Column name for vertical wind W.')
    p.add_argument('--col-tsonic', default=None,
                   help='Column name for sonic temperature T_SONIC (enables 4-combination mode).')
    p.add_argument('--usecols', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5],
                   help='0-based column indices to read from each file.')
    p.add_argument('--col-names', nargs='+', default=['u', 'v', 'w', 'ts', 'ch4', 'n2o'],
                   help='Column names to assign after selecting --usecols.')
    p.add_argument('--skiprows', type=int, default=9,
                   help='Metadata rows before the column-name row.')
    p.add_argument('--na-values', nargs='+',
                   default=['-9999', '-9999.0', '-9999.0000000000000'],
                   help='Strings to treat as NaN.')
    # PWB parameters
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    p.add_argument('--lag-max', type=float, default=10.0,
                   help='CCF search half-width in seconds.')
    p.add_argument('--n-bootstrap', type=int, default=99,
                   help='Number of block-bootstrap replicates.')
    p.add_argument('--block-length', type=float, default=20.0,
                   help='Bootstrap block length in seconds.')
    p.add_argument('--min-valid-frac', type=float, default=0.3,
                   help='Minimum non-NaN fraction for a series to be processed.')
    # PWBOPT thresholds
    p.add_argument('--hdi-thresh', type=float, default=0.5,
                   help='S1 HDI threshold in seconds.')
    p.add_argument('--dev-thresh', type=float, default=0.5,
                   help='S2 deviation threshold in seconds.')
    p.add_argument('--hdi-prefilter', type=float, default=1.0,
                   help='HDI pre-filter threshold in seconds (0 = disabled).')
    # Execution
    p.add_argument('--n-workers', type=int, default=None,
                   help='Parallel worker processes (default: os.cpu_count()).')
    p.add_argument('--save-plots', action='store_true',
                   help='Save one diagnostic PNG per period per scalar.')
    return p


def _cli_main():
    import sys
    try:
        from rich.console import Console as _Console
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                   SpinnerColumn, TextColumn,
                                   TimeElapsedColumn, TimeRemainingColumn)
        _rich = True
        console = _Console(log_path=False)  # suppress "file:line" suffix in log output
    except ImportError:
        _rich = False
        console = None

    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: --input-dir not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    files = sorted(input_dir.glob(args.file_pattern))
    if not files:
        print(f'ERROR: no files matching {args.file_pattern!r} in {input_dir}',
              file=sys.stderr)
        sys.exit(1)

    # Parse --scalar LABEL:column pairs
    scalars = {}
    for token in args.scalars:
        if ':' not in token:
            print(f'ERROR: --scalar must be LABEL:column, got {token!r}',
                  file=sys.stderr)
            sys.exit(1)
        label, col = token.split(':', 1)
        scalars[label] = col

    det = PwbBatchDetection(
        files=files,
        scalars=scalars,
        col_w=args.col_w,
        col_tsonic=args.col_tsonic,
        hz=args.hz,
        lag_max_s=args.lag_max,
        n_bootstrap=args.n_bootstrap,
        block_length_s=args.block_length,
        usecols=args.usecols,
        col_names=args.col_names,
        skiprows=args.skiprows,
        na_values=args.na_values,
        min_valid_frac=args.min_valid_frac,
        output_dir=Path(args.output_dir),
        save_plots=args.save_plots,
        n_workers=args.n_workers,
    )

    msg = (f'PWB batch detection  {len(files)} files  '
           f'{det.n_workers} workers  -> {args.output_dir}')

    if _rich:
        console.print(f'\n[bold]{msg}[/bold]\n')

        def _fmt(row, gas):
            pfx = gas.lower()
            v = row.get(f'{pfx}_tlag_s')
            h = row.get(f'{pfx}_hdi_range_s')
            if v is None or v != v:
                return f'[dim]{gas}=--[/dim]'
            hdi_color = 'green' if h == h and h < 0.5 else ('yellow' if h == h and h < 1.0 else 'red')
            return f'{gas}=[bold]{v:.2f}s[/bold] HDI=[{hdi_color}]{h:.2f}[/{hdi_color}]'

        prog = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=8,
        )
        task_id = prog.add_task(
            f'[cyan]{det.n_workers} workers[/cyan]', total=len(files))

        with prog:
            def _cb(done, total, row):
                period = row.get('period', '')
                # Shorten e.g. "20210815-1200_raw_dataset_..._adv.txt" -> "20210815-1200"
                period_short = Path(period).stem.split('_')[0]
                parts = '  '.join(_fmt(row, g) for g in scalars)
                console.log(f'[dim]{period_short}[/dim]  {parts}')
                prog.update(task_id, completed=done,
                            description=f'[cyan]{det.n_workers} workers[/cyan]  [dim]{period_short}[/dim]')

            results = det.run(on_progress=_cb)

        console.print(f'\n[green]Done — {len(results)} periods.[/green]')
    else:
        print(msg)
        results = det.run(
            on_progress=lambda done, total, row:
            print(f'  [{done}/{total}] {row.get("period", "")}')
        )
        print(f'Done — {len(results)} periods.')

    # PWBOPT
    for label in scalars:
        pfx = label.lower()
        tc, hc = f'{pfx}_tlag_s', f'{pfx}_hdi_range_s'
        if tc not in results.columns:
            continue
        std = PwbBatchDetection.apply_pwbopt(
            results[tc], results[hc], args.hdi_thresh, args.dev_thresh)
        results[f'{pfx}_pwbopt_s_std'] = std['pwbopt_s']
        results[f'{pfx}_flag_std'] = std['flag']

        if args.hdi_prefilter > 0:
            tpf = PwbBatchDetection.apply_hdi_prefilter(
                results[tc], results[hc], args.hdi_prefilter)
            pf = PwbBatchDetection.apply_pwbopt(tpf, results[hc],
                                                args.hdi_thresh, args.dev_thresh)
            results[f'{pfx}_pwbopt_s_pf'] = pf['pwbopt_s']
            results[f'{pfx}_flag_pf'] = pf['flag']

        # Fill leading/trailing NaN lags so every period has a usable lag
        # for flux covariance calculation.  The raw lags supply the median
        # fallback when the entire PWBOPT series is NaN.
        raw_tlag = results[tc] if tc in results.columns else None
        results[f'{pfx}_tlag_final_s'] = PwbBatchDetection.fill_tlag_gaps(
            results[f'{pfx}_pwbopt_s_std'], tlag_s_raw=raw_tlag)
        if f'{pfx}_pwbopt_s_pf' in results.columns:
            results[f'{pfx}_tlag_final_pf_s'] = PwbBatchDetection.fill_tlag_gaps(
                results[f'{pfx}_pwbopt_s_pf'], tlag_s_raw=raw_tlag)

    PwbBatchDetection.plot_summary(
        results=results,
        scalars=scalars,
        hdi_thresh=args.hdi_thresh,
        hdi_prefilter=args.hdi_prefilter,
        lag_max_s=args.lag_max,
        output_dir=Path(args.output_dir),
        showplot=False,
    )

    out = Path(args.output_dir) / 'tlag_results.csv'
    results.to_csv(out, index=False)
    print(f'Results saved to: {out}')


if __name__ == '__main__':
    _cli_main()
