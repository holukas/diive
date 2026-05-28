"""
=====================================================================
Time Lag Detection Pipeline (PWBOPT and Pre-filter Comparison)
=====================================================================

Batch pre-whitening with block-bootstrap (PWB) time lag detection across
multiple averaging periods, followed by the PWBOPT optimal lag selection
strategy from Vitale et al. (2024).

The workflow is a Python conversion of two R scripts from the RFlux pipeline:

  01_tlag_detection_pwb.R             -- batch PWB detection and S1/S2/S3 selection
  02_tlag_compare_pwbopt_strategies.R -- compare standard vs. pre-filtered PWBOPT

PWBOPT selection tiers (paper Section 2.3):

  S1 : HDI range < 0.5 s
       -> reliable, accept the detected lag directly
  S2 : HDI range >= 0.5 s but detected lag is within 0.5 s of the preceding
       optimal lag -> accept for temporal continuity
  S3 : unreliable
       -> carry forward the last known optimal lag

The pre-filter strategy (script 02) removes lags with HDI above a wider
threshold before S1/S2/S3 runs, preventing S2 from accepting uncertain
detections that happen to lie close to the preceding optimal lag.

Set USE_SYNTHETIC = False and fill in INPUT_DIR / OUTPUT_DIR to run the
pipeline on real EddyPro-rotated high-frequency files.  Set it to True
(the default) to run a self-contained demonstration on synthetic data.

References:
    Vitale D et al. (2024) A pre-whitening with block-bootstrap cross-correlation
    procedure for temporal alignment of eddy covariance data.
    Environmental and Ecological Statistics 31:219-244.
    doi:10.1007/s10651-024-00615-9
"""

# %%
# Settings
# ^^^^^^^^^
#
# Two modes are controlled by USE_SYNTHETIC:
#
#   True  -- use synthetic data (no files needed, good for testing)
#   False -- load real EddyPro-rotated files from INPUT_DIR and write
#            results and plots to OUTPUT_DIR
#
# All thresholds follow Vitale et al. (2024) Section 2.3 defaults and
# match 01_tlag_detection_pwb.R (lines 51-63).

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diive as dv

# ------------------------------------------------------------------
# Mode switch
# ------------------------------------------------------------------
USE_SYNTHETIC = True  # False: load from INPUT_DIR; True: use synthetic data

# ------------------------------------------------------------------
# I/O (only used when USE_SYNTHETIC = False)
# ------------------------------------------------------------------
# Folder that contains EddyPro-rotated high-frequency files (one file
# per 30-min averaging period), matching R: folder_path in script 01.
INPUT_DIR = Path(
    r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_fluxnet_ch4_n2o_timelag\test_input_diiveversion")

# Root output folder; sub-directories 'plots/' and the CSV are created
# automatically, matching R: output_csv and plot_dir in script 01.
OUTPUT_DIR = Path(r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_fluxnet_ch4_n2o_timelag\test_output_diiveversion")

# File pattern used to glob INPUT_DIR (R: file_pattern = "\\.txt$")
FILE_PATTERN = '*.txt'

# ------------------------------------------------------------------
# File format (only used when USE_SYNTHETIC = False)
# ------------------------------------------------------------------
# 0-based column positions after whitespace-splitting each data row.
# R: select = c(1, 2, 3, 4, 7, 8)  (1-based) -> Python 0-based: [0, 1, 2, 3, 6, 7]
# File columns: u(0) v(1) w(2) ts(3) co2(4) h2o(5) ch4(6) n2o/4th-gas(7) air_t(8) air_p(9)
FILE_USECOLS = [0, 1, 2, 3, 6, 7]

# Column names to assign after selecting (must match len(FILE_USECOLS))
# R: setnames(dat, new = c("u", "v", "w", "ts", "ch4", "n2o"))
FILE_COL_NAMES = ['u', 'v', 'w', 'ts', 'ch4', 'n2o']

# Number of metadata rows before the column-name row (R: skip = 9)
# EddyPro rotated files: rows 0-8 = metadata, row 9 = column names, row 10+ = data.
# load_period skips FILE_SKIPROWS + 1 rows total (metadata + column-name row).
FILE_SKIPROWS = 9

# Values treated as missing on load (R: na.strings = c("-9999.0", ...))
FILE_NA_VALUES = ['-9999', '-9999.0', '-9999.0000000000000']

# ------------------------------------------------------------------
# Column mapping
# ------------------------------------------------------------------
# Vertical wind column name (after FILE_COL_NAMES renaming)
COL_W = 'w'

# Sonic temperature column (required for the 4-combination RFlux v3.2.0 logic).
# T_SONIC combinations are critical for trace gases (N2O, CH4) where the scalar
# x W cross-correlation is too weak to produce a reliable peak.
# Set to None to fall back to 2-combination mode (cw, wc only).
COL_TSONIC = 'ts'

# Scalars to process: {display_label: column_name_in_file}
# Add or remove entries to change which gases are detected.
SCALARS = {
    'CH4': 'ch4',
    'N2O': 'n2o',
}

# ------------------------------------------------------------------
# PWB settings
# ------------------------------------------------------------------
HZ = 20  # sampling frequency (Hz);  R: mfreq = 20
LAG_MAX_S = 10.0  # CCF search half-width (s);  R: LAG.MAX = mfreq*10
N_BOOTSTRAP = 99  # bootstrap replicates — production: 99;  R: n_boot = 99
BLOCK_LENGTH_S = 20.0  # bootstrap block length (s);  paper Section 2.2

# ------------------------------------------------------------------
# PWBOPT thresholds  (paper Section 2.3;  R: HDI_THRESH_S, DEV_THRESH_S)
# ------------------------------------------------------------------
HDI_THRESH_S = 0.5  # S1: HDI range below this is reliable
DEV_THRESH_S = 0.5  # S2: max deviation from the preceding optimal lag

# Pre-filter threshold (02_tlag_compare_pwbopt_strategies.R line 47)
HDI_PREFILTER_S = 1.0  # lags with HDI wider than this are discarded before PWBOPT

# ------------------------------------------------------------------
# Quality guard  (R: MIN_VALID_FRAC = 0.3, valid_enough())
# ------------------------------------------------------------------
MIN_VALID_FRAC = 0.3  # minimum fraction of non-NaN values in a series

# ------------------------------------------------------------------
# Output flags
# ------------------------------------------------------------------
# Save one diagnostic PNG per period per scalar into OUTPUT_DIR/plots/.
# In synthetic mode plots are never saved (only shown in the summary figure).
SAVE_PLOTS = True

# ------------------------------------------------------------------
# Results CSV  (optional short-cut to skip batch detection)
# ------------------------------------------------------------------
# Path to a CSV produced by a previous run of this script, e.g.:
#   RESULTS_CSV = OUTPUT_DIR / 'tlag_results.csv'
# When set, the batch PWB detection and all file I/O are skipped entirely
# and results are loaded directly.  PWBOPT selection is re-applied so any
# threshold change takes effect, and the visualization runs as normal.
# Set to None to run the full pipeline from scratch.
RESULTS_CSV = None
# RESULTS_CSV = OUTPUT_DIR / 'tlag_results.csv'

# ------------------------------------------------------------------
# Synthetic-mode parameters  (ignored when USE_SYNTHETIC = False)
# ------------------------------------------------------------------
N_PERIODS = 20
RECORDS = HZ * 60 * 3  # 3 min at 20 Hz = 3 600 records per period
LAG_TRUE_S = 1.5  # known time lag embedded in synthetic data
LAG_TRUE_RECORDS = int(LAG_TRUE_S * HZ)


# %%
# Helper functions
# ^^^^^^^^^^^^^^^^^
#
# Python ports of the utility functions defined in the R scripts.


def is_valid_series(x, min_valid_frac=MIN_VALID_FRAC):
    """
    Return True when x has enough valid, non-constant data for PWB.

    Port of valid_enough() from 01_tlag_detection_pwb.R (lines 128-132).
    Guards against the NA p-value crash in near-empty or constant series.
    """
    x = np.asarray(x, dtype=float)
    if np.mean(~np.isnan(x)) < min_valid_frac:
        return False  # too many missing values
    if np.nanstd(x) < np.finfo(float).eps:
        return False  # constant series, no variance
    return True


def load_period(filepath, usecols, col_names, skiprows, na_values):
    """
    Load one averaging-period file and return a clean DataFrame.

    EddyPro rotated files are whitespace-separated.  The column-name row
    contains 'u v w ts co2 h2o ch4 4th gas air_t air_p' where '4th gas'
    has a space in the name.  To avoid a header/data column-count mismatch
    when splitting on whitespace, the header row is skipped entirely
    (skiprows + 1) and columns are selected by 0-based position.

    R equivalent: fread(file, skip=9, header=TRUE, sep="auto",
                        select=c(1,2,3,4,7,8), na.strings=c(...))
    from 01_tlag_detection_pwb.R (lines 192-206).

    Parameters
    ----------
    filepath : Path or str
        Path to the high-frequency data file.
    usecols : list of int
        0-based column positions in the data (after whitespace-splitting).
        R's 1-based select=c(1,2,3,4,7,8) maps to Python [0,1,2,3,6,7].
    col_names : list of str
        Column names to assign after selecting (len must match usecols).
    skiprows : int
        Number of metadata rows before the column-name row (R: skip=9).
        The column-name row itself is also skipped (+1 inside this function).
    na_values : list of str
        Strings to interpret as NaN.

    Returns
    -------
    pd.DataFrame or None if the file cannot be read.
    """
    try:
        # sep=r'\s+' handles variable-width whitespace between values.
        # skiprows + 1 skips both the metadata block and the column-name row,
        # so header=None reads pure data rows with no ambiguous '4th gas' token.
        # Columns are then selected by integer position via .iloc.
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
        return df
    except Exception as exc:
        print(f'  -> Read error: {exc}')
        return None


def apply_pwbopt(tlag_s, hdi_range_s,
                 hdi_thresh=HDI_THRESH_S, dev_thresh=DEV_THRESH_S):
    """
    Apply PWBOPT S1/S2/S3 selection logic (paper Section 2.3).

    Direct port of apply_pwbopt() from 01_tlag_detection_pwb.R (lines 73-107).
    Processes periods sequentially; last_optimal carries across periods so that
    earlier reliable detections anchor later S2 selections.

    Parameters
    ----------
    tlag_s : array-like
        Detected PWB lag in seconds per period (NaN if detection failed).
    hdi_range_s : array-like
        95% HDI range in seconds per period.
    hdi_thresh : float
        S1 threshold: HDI < hdi_thresh -> reliable (default 0.5 s).
    dev_thresh : float
        S2 threshold: max allowed deviation from last optimal (default 0.5 s).

    Returns
    -------
    pd.DataFrame with columns 'pwbopt_s' (optimal lag, s) and 'flag'.
    """
    tlag_s = np.asarray(tlag_s, dtype=float)
    hdi_range_s = np.asarray(hdi_range_s, dtype=float)
    n = len(tlag_s)
    flags = ['S3_unreliable'] * n
    optimal = np.full(n, np.nan)
    last_optimal = np.nan  # running state, updated on every S1 or S2 period

    for i in range(n):
        tl = tlag_s[i]
        hdi = hdi_range_s[i]

        if np.isnan(tl) or np.isnan(hdi):
            # Detection failed: carry forward last known optimal, flag stays S3
            optimal[i] = last_optimal
            continue

        if hdi < hdi_thresh:
            # S1: HDI below threshold -> reliable, accept directly
            flags[i] = 'S1_optimal'
            optimal[i] = tl
            last_optimal = tl

        elif not np.isnan(last_optimal) and abs(tl - last_optimal) <= dev_thresh:
            # S2: uncertain but consistent with preceding optimal -> accept for continuity
            flags[i] = 'S2_optimal'
            optimal[i] = tl
            last_optimal = tl

        else:
            # S3: unreliable and inconsistent -> carry forward last known optimal
            optimal[i] = last_optimal

    return pd.DataFrame({'pwbopt_s': optimal, 'flag': flags})


def apply_hdi_prefilter(tlag_s, hdi_range_s, threshold=HDI_PREFILTER_S):
    """
    Pre-filter: replace lags with HDI above threshold with NaN before PWBOPT.

    Port of apply_hdi_prefilter() from
    02_tlag_compare_pwbopt_strategies.R (lines 96-101).

    More conservative than standard PWBOPT: wide-HDI detections are discarded
    before S1/S2/S3 runs, so S2 cannot accept them even if they happen to be
    close to the preceding optimal lag.

    Parameters
    ----------
    tlag_s : array-like
        Detected lags in seconds.
    hdi_range_s : array-like
        95% HDI range in seconds.
    threshold : float
        Lags with HDI range wider than this are replaced with NaN.

    Returns
    -------
    np.ndarray with pre-filtered lags (NaN where HDI > threshold).
    """
    tlag_filtered = np.asarray(tlag_s, dtype=float).copy()
    hdi = np.asarray(hdi_range_s, dtype=float)
    tlag_filtered[(hdi > threshold) & ~np.isnan(hdi)] = np.nan
    return tlag_filtered


# %%
# Data source
# ^^^^^^^^^^^^
#
# Three modes are supported, controlled by RESULTS_CSV and USE_SYNTHETIC:
#
#   RESULTS_CSV set    -- load a previous-run CSV; skip batch detection entirely
#   USE_SYNTHETIC True -- generate synthetic periods in-memory; no files needed
#   USE_SYNTHETIC False-- load real EddyPro-rotated files from INPUT_DIR
#
# Modes 2 and 3 build ``period_sources`` (a list of (name, source) tuples)
# consumed by the batch detection loop.  CSV mode sets ``period_sources = []``
# so the loop runs zero iterations and leaves the loaded ``results`` untouched.

_csv_loaded = False
if RESULTS_CSV is not None:
    _results_path = Path(RESULTS_CSV)
    if not _results_path.exists():
        raise FileNotFoundError(f'RESULTS_CSV not found: {_results_path}')
    results = pd.read_csv(_results_path, na_values=['-9999', '-9999.0'])
    _csv_loaded = True
    output_csv = _results_path  # used by the summary-stats print below
    plot_dir = OUTPUT_DIR / 'plots'  # used by the summary-stats print below
    period_sources = []  # makes the batch loop a no-op
    print(f'CSV mode: {len(results)} periods loaded from {_results_path}')
    print('Batch detection skipped; re-applying PWBOPT and running visualization.')

if not _csv_loaded and USE_SYNTHETIC:
    # ---------- synthetic mode ----------
    np.random.seed(42)

    # Signal strength decreases from strong (S1 expected) to near-zero (S3 expected)
    flux_strengths = np.linspace(5.0, 0.05, N_PERIODS)

    _synthetic_dfs = []
    for _p in range(N_PERIODS):
        # AR(1) turbulent wind with phi=0.8 (realistic autocorrelation structure)
        _w = np.zeros(RECORDS)
        for _t in range(1, RECORDS):
            _w[_t] = 0.8 * _w[_t - 1] + np.random.normal(0, 0.3)

        # T_SONIC: correlated with W (driven by the same turbulent structures)
        _ts = _w * 0.6 + np.random.normal(0, 0.15, RECORDS)

        # Both scalars: lagged wind * flux_strength + growing noise
        _noise_std = 0.5 + (1.0 - flux_strengths[_p] / 5.0) * 0.8
        _ch4 = (np.roll(_w, LAG_TRUE_RECORDS) * flux_strengths[_p]
                + np.random.normal(0, _noise_std, RECORDS))
        _ch4[:LAG_TRUE_RECORDS] = _ch4[LAG_TRUE_RECORDS]

        # N2O: much weaker signal (trace-gas scenario)
        _n2o = (np.roll(_w, LAG_TRUE_RECORDS) * flux_strengths[_p] * 0.1
                + np.random.normal(0, 1.0, RECORDS))
        _n2o[:LAG_TRUE_RECORDS] = _n2o[LAG_TRUE_RECORDS]

        _synthetic_dfs.append(pd.DataFrame({'w': _w, 'ts': _ts, 'ch4': _ch4, 'n2o': _n2o}))

    period_sources = [(f'period_{p:02d}', df)
                      for p, df in enumerate(_synthetic_dfs)]

    print(f'Synthetic mode: {N_PERIODS} periods, '
          f'flux_strength {flux_strengths[0]:.2f} -> {flux_strengths[-1]:.3f}')

elif not _csv_loaded:
    # ---------- real-data mode ----------
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f'INPUT_DIR not found: {INPUT_DIR}')

    file_list = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not file_list:
        raise FileNotFoundError(f'No files matching {FILE_PATTERN!r} in {INPUT_DIR}')

    # Sources are Paths; DataFrames are loaded one at a time inside the loop
    period_sources = [(f.name, f) for f in file_list]

    # Create output directories
    plot_dir = OUTPUT_DIR / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Real-data mode: {len(file_list)} files in {INPUT_DIR}')

# %%
# Batch PWB detection
# ^^^^^^^^^^^^^^^^^^^^
#
# Iterate over all periods/files.  For each period and each configured scalar,
# run PreWhiteningBootstrap.run() and record the results.
#
# Corresponds to the main file loop in 01_tlag_detection_pwb.R (section 6,
# lines 185-268).  The validity guard (is_valid_series) matches valid_enough()
# in the R script (lines 128-132).
#
# Diagnostic plots are saved per period per scalar when SAVE_PLOTS=True and
# the pipeline is running in real-data mode, matching the PDF output produced
# by detect_gas() in the R script (lines 149-161).

# Checkpoint: accumulate rows here; save to CSV after each period (real-data mode)
rows = []

for period_name, source in period_sources:
    print(f'[{period_sources.index((period_name, source)) + 1}'
          f'/{len(period_sources)}]  {period_name}')

    # Load DataFrame (real-data mode loads from file; synthetic mode uses it directly)
    if USE_SYNTHETIC:
        df = source
    else:
        df = load_period(source, FILE_USECOLS, FILE_COL_NAMES,
                         FILE_SKIPROWS, FILE_NA_VALUES)
        if df is None or len(df) < 25:
            print('  -> Skipped (load failed or insufficient rows).')
            rows.append({'period': period_name})  # empty sentinel row
            continue

    # Guard: wind must be valid before attempting any scalar
    if not is_valid_series(df[COL_W]):
        print(f'  -> Skipped (W invalid).')
        rows.append({'period': period_name})
        continue

    row = {'period': period_name}

    for scalar_label, scalar_col in SCALARS.items():
        prefix = scalar_label.lower()  # e.g. 'ch4', 'n2o'

        if not is_valid_series(df[scalar_col]):
            print(f'  -> {scalar_label} skipped (too many NaN or constant series).')
            for suffix in ('tlag_s', 'hdi_lo_s', 'hdi_hi_s',
                           'hdi_range_s', 'tlag_pw_s', 'corr_est', 'ar_order'):
                row[f'{prefix}_{suffix}'] = np.nan
            continue

        # Run PWB (equivalent to detect_gas() -> tlag_detection() in R script)
        _has_ts = COL_TSONIC is not None and COL_TSONIC in df.columns
        _col_map = {COL_W: 'W', scalar_col: scalar_label}
        _keep_cols = [COL_W, scalar_col]
        if _has_ts:
            _col_map[COL_TSONIC] = 'T_SONIC'
            _keep_cols.append(COL_TSONIC)
        pwb = dv.flux.PreWhiteningBootstrap(
            df=df[_keep_cols].rename(columns=_col_map),
            var_w='W',
            var_scalar=scalar_label,
            var_tsonic='T_SONIC' if _has_ts else None,
            hz=HZ,
            lag_max_s=LAG_MAX_S,
            n_bootstrap=N_BOOTSTRAP,
            block_length_s=BLOCK_LENGTH_S,
            segment_name=period_name,
        )
        pwb.run()
        res = pwb.results

        # Collect results (equivalent to R: extr(res, "pwb"), hdi_range(res), ...)
        row[f'{prefix}_tlag_s'] = res['tlag_s']  # R: res$pwb / mfreq
        row[f'{prefix}_hdi_lo_s'] = res['hdi_lo_s']  # R: res$pwb_lci / mfreq
        row[f'{prefix}_hdi_hi_s'] = res['hdi_hi_s']  # R: res$pwb_uci / mfreq
        row[f'{prefix}_hdi_range_s'] = res['hdi_range_s']  # R: (pwb_uci - pwb_lci) / mfreq
        row[f'{prefix}_tlag_pw_s'] = res['tlag_pw_s']  # R: res$opt_tlag / mfreq
        row[f'{prefix}_corr_est'] = res['corr_est']  # R: res$cor_pwb
        row[f'{prefix}_ar_order'] = res['ar_order']

        print(f'  -> {scalar_label}: tlag={res["tlag_s"]:.3f} s  '
              f'HDI_range={res["hdi_range_s"]:.3f} s  '
              f'reliable={res["is_reliable"]}')

        # Save diagnostic plot per period per scalar (R: pdf() in detect_gas())
        if SAVE_PLOTS and not USE_SYNTHETIC:
            fig = pwb.plot(
                title=f'{period_name} | {scalar_label}',
                showplot=False,
            )
            fig.savefig(plot_dir / f'{Path(period_name).stem}_{prefix}.png',
                        dpi=100, bbox_inches='tight')
            plt.close(fig)

    rows.append(row)

    # Checkpoint: save growing results after every period so a crash can be resumed
    # (R: saveRDS(results_list, checkpoint_file) after each file)
    if not USE_SYNTHETIC:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'tlag_results_checkpoint.csv',
                                  index=False)

if not _csv_loaded:
    results = pd.DataFrame(rows)
    print(f'\nBatch detection complete: {len(results)} periods processed.')

# %%
# Apply PWBOPT for each scalar (standard strategy)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 01_tlag_detection_pwb.R section 7 (lines 272-283):
#   ch4_opt <- apply_pwbopt(results$ch4_tlag_sec, results$ch4_hdi_range_sec)
#
# Each scalar is processed independently in temporal order so that
# last_optimal from one gas does not contaminate another.

for scalar_label in SCALARS:
    prefix = scalar_label.lower()
    tlag_col = f'{prefix}_tlag_s'
    hdi_col = f'{prefix}_hdi_range_s'

    if tlag_col not in results.columns:
        continue

    std = apply_pwbopt(results[tlag_col].fillna(np.nan),
                       results[hdi_col].fillna(np.nan))
    results[f'{prefix}_pwbopt_s_std'] = std['pwbopt_s']
    results[f'{prefix}_flag_std'] = std['flag']

# %%
# Pre-filtered PWBOPT and strategy comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 02_tlag_compare_pwbopt_strategies.R (lines 129-148):
# Lags with HDI > HDI_PREFILTER_S are set to NaN before S1/S2/S3 runs.
# Both standard and pre-filtered results are kept for comparison.

for scalar_label in SCALARS:
    prefix = scalar_label.lower()
    tlag_col = f'{prefix}_tlag_s'
    hdi_col = f'{prefix}_hdi_range_s'

    if tlag_col not in results.columns:
        continue

    tlag_pf = apply_hdi_prefilter(results[tlag_col].fillna(np.nan),
                                  results[hdi_col].fillna(np.nan),
                                  threshold=HDI_PREFILTER_S)
    pf = apply_pwbopt(tlag_pf, results[hdi_col].fillna(np.nan))
    results[f'{prefix}_pwbopt_s_pf'] = pf['pwbopt_s']
    results[f'{prefix}_flag_pf'] = pf['flag']

# %%
# Save results to CSV
# ^^^^^^^^^^^^^^^^^^^^
#
# Equivalent to write.csv(results, file=output_csv) in
# 01_tlag_detection_pwb.R (line 298) and fwrite() in script 02.
# In synthetic mode the CSV is written next to this script for inspection.

if not _csv_loaded:
    output_csv = (OUTPUT_DIR / 'tlag_results.csv') if not USE_SYNTHETIC else Path('tlag_results_synthetic.csv')
    results.to_csv(output_csv, index=False)
    print(f'Results saved to: {output_csv}')

# %%
# Visualize: lag timeseries and strategy comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One figure per scalar showing:
#   Top    -- detected lags coloured by PWBOPT flag, optimal lag lines overlaid
#   Middle -- 95% HDI range per period with threshold reference lines
#   Bottom -- side-by-side flag comparison between the two strategies
#
# Equivalent to the aggregated view of the per-period PDFs from script 01.

FLAG_COLORS = {
    'S1_optimal': '#2ca02c',  # green  -- reliable
    'S2_optimal': '#ff7f0e',  # orange -- consistent
    'S3_unreliable': '#d62728',  # red    -- unreliable
}

px = np.arange(len(results))  # x-axis: period index

for scalar_label in SCALARS:
    prefix = scalar_label.lower()
    tlag_col = f'{prefix}_tlag_s'
    hdi_col = f'{prefix}_hdi_range_s'
    flag_std_col = f'{prefix}_flag_std'
    flag_pf_col = f'{prefix}_flag_pf'
    opt_std_col = f'{prefix}_pwbopt_s_std'
    opt_pf_col = f'{prefix}_pwbopt_s_pf'

    if tlag_col not in results.columns:
        continue

    tlag = results[tlag_col].values.astype(float)
    hdi = results[hdi_col].values.astype(float)
    flag_std = results[flag_std_col].values
    opt_std = results[opt_std_col].values.astype(float)
    opt_pf = results[opt_pf_col].values.astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{scalar_label} -- PWB lag pipeline (PWBOPT strategy comparison)',
                 fontsize=12)

    # ----- Panel 1: detected lags coloured by standard PWBOPT flag -----
    ax = axes[0]
    for flag, color in FLAG_COLORS.items():
        mask = flag_std == flag
        ax.scatter(px[mask], tlag[mask], color=color, s=70, zorder=3, label=flag)

    ax.plot(px, opt_std, color='black', linewidth=1.5, linestyle='-',
            label='PWBOPT standard')
    ax.plot(px, opt_pf, color='steelblue', linewidth=1.5, linestyle='--',
            label='PWBOPT pre-filtered')

    if USE_SYNTHETIC:
        # Show true lag reference only when it is known
        ax.axhline(LAG_TRUE_S, color='grey', linewidth=1, linestyle=':',
                   label=f'True lag ({LAG_TRUE_S} s)')

    ax.set_ylabel('Time lag (s)')
    ax.set_title('Detected lags per period (coloured by standard PWBOPT flag)')
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_ylim(-LAG_MAX_S - 0.5, LAG_MAX_S + 0.5)

    # ----- Panel 2: HDI range with threshold reference lines -----
    ax = axes[1]
    ax.bar(px, hdi, color='#aec7e8', edgecolor='none', label='HDI range')
    ax.axhline(HDI_THRESH_S, color='#2ca02c', linewidth=1.5, linestyle='--',
               label=f'S1 threshold ({HDI_THRESH_S} s)')
    ax.axhline(HDI_PREFILTER_S, color='steelblue', linewidth=1.5, linestyle=':',
               label=f'Pre-filter threshold ({HDI_PREFILTER_S} s)')
    ax.set_ylabel('95% HDI range (s)')
    ax.set_title('Bootstrap uncertainty (HDI range) per period')
    ax.legend(frameon=False, fontsize=8)

    # ----- Panel 3: side-by-side flag bars per period -----
    ax = axes[2]
    bar_w = 0.4
    for offset, flag_col in enumerate([flag_std_col, flag_pf_col]):
        for p in px:
            flag = results[flag_col].iloc[p]
            ax.bar(p + (offset - 0.5) * bar_w, 1, bar_w,
                   color=FLAG_COLORS.get(flag, '#aaaaaa'), alpha=0.85)

    patches = [mpatches.Patch(color=c, label=f) for f, c in FLAG_COLORS.items()]
    ax.legend(handles=patches, frameon=False, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel('Period index')
    ax.set_title('Flag per period: standard (left bar) vs. pre-filtered (right bar)')

    plt.tight_layout()

    if SAVE_PLOTS and not USE_SYNTHETIC:
        fig.savefig(OUTPUT_DIR / f'summary_{prefix}.png', dpi=100, bbox_inches='tight')

    plt.show()

# %%
# Summary statistics
# ^^^^^^^^^^^^^^^^^^^
#
# Mirrors the summary printed by 01_tlag_detection_pwb.R (lines 288-299) and
# 02_tlag_compare_pwbopt_strategies.R (lines 170-200).

pct_reliable = lambda col: (
    100 * np.mean(results[col].isin(['S1_optimal', 'S2_optimal']))
    if col in results.columns else np.nan
)

print('\n' + '=' * 70)
print('PWBOPT strategy comparison summary')
print('=' * 70)
print(f'\n{"Gas":<6s}  {"Strategy":<24s}  {"S1":>5s}  {"S2":>5s}  {"S3":>5s}  {"Reliable":>9s}')
print('-' * 70)

for scalar_label in SCALARS:
    prefix = scalar_label.lower()
    for strategy, flag_col in [('Standard', f'{prefix}_flag_std'),
                               ('Pre-filtered', f'{prefix}_flag_pf')]:
        if flag_col not in results.columns:
            continue
        vc = results[flag_col].value_counts()
        s1 = vc.get('S1_optimal', 0)
        s2 = vc.get('S2_optimal', 0)
        s3 = vc.get('S3_unreliable', 0)
        rel = pct_reliable(flag_col)
        print(f'  {scalar_label:<4s}  {strategy:<24s}  {s1:>5d}  {s2:>5d}  {s3:>5d}  {rel:>8.1f}%')

    hdi_col = f'{prefix}_hdi_range_s'
    if hdi_col in results.columns:
        n_pf = int(np.sum(results[hdi_col].fillna(0) > HDI_PREFILTER_S))
        print(f'         Pre-filter removed: {n_pf} / {len(results)} periods '
              f'(HDI > {HDI_PREFILTER_S} s)')
    print()

if not USE_SYNTHETIC:
    print(f'Results CSV : {output_csv}')
    if SAVE_PLOTS:
        print(f'Plots       : {plot_dir}/')

print('[OK] PWBOPT pipeline complete.')

# %%
# Lag scatter + KDE comparison figure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Reproduces the figure style from the R visualization scripts
# (plot_comparison_strategies.py):
#
# For each gas, the optimal lags from the two PWBOPT strategies are placed
# side by side.  Each panel shows:
#
#   Left sub-panel  -- scatter of lags over period index (or timestamps),
#                      with small random jitter to separate overlapping points
#   Right sub-panel -- normalized Gaussian KDE of the lag distribution;
#                      the KDE-estimated mode is drawn as a dashed black line
#                      on both sub-panels for visual alignment
#
# This is equivalent to the comparison figure produced by the R script
# 02_tlag_compare_pwbopt_strategies.R (section "visualization").
#
# To add real timestamps as the x-axis supply timestamp_col='period' after
# parsing timestamps from the filename strings in results['period'].
# In synthetic mode the x-axis is the integer period index.
#
# To mark site events (tillage, fertilization, ...) on the scatter panels
# pass an events list when timestamp_col is set.  Example:
#
#   events = [
#       {'date': '2021-08-20', 'label': 'Tillage',
#        'ls': '-', 'color': 'black', 'side': 'right'},
#   ]

# Build the scalars dict from whichever gases were processed
scalars_plot = {}
for _scalar_label in SCALARS:
    _prefix = _scalar_label.lower()
    _col_a = f'{_prefix}_pwbopt_s_std'
    _col_b = f'{_prefix}_pwbopt_s_pf'
    if _col_a in results.columns and _col_b in results.columns:
        scalars_plot[_scalar_label] = {'col_a': _col_a, 'col_b': _col_b}

if scalars_plot:
    lag_plot = dv.flux.PwboptLagPlot(
        results=results,
        scalars=scalars_plot,  # gases to plot
        label_a='PWBOPT standard',  # left panels
        label_b='PWBOPT pre-filtered',  # right panels
        color_a='#0072B2',  # Wong blue
        color_b='#E05C2A',  # coral-orange
        # ylim: omitted -> auto-scaled tight to the actual lag data
    )
    lag_plot.plot(
        title='PWB optimal lag: standard vs pre-filtered PWBOPT',
        showplot=True,
        # outpath=str(OUTPUT_DIR), outname='lag_strategy_comparison.png',
    )
else:
    print('No PWBOPT columns found in results; skipping lag comparison figure.')
