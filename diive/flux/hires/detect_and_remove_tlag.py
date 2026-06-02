"""
DETECT_AND_REMOVE_TLAG: per-30-min-chunk end-to-end PWB pipeline
=================================================================

Input raw EC files commonly cover multi-hour periods (e.g. 6 h), but EC
flux processing and the PWB time-lag detection both work on **30-minute
averaging intervals**. The wind-rotation angles drift over hours, and the
tube delay can drift too, so a single rotation / lag estimate across 6 h
is the wrong granularity.

This module therefore reads each long raw file once and processes it in
**fixed-length chunks** (default 30 minutes = ``hz * 1800`` rows). For
each chunk:

1. Slice the chunk out of the full-file DataFrame (unrotated; all original
   columns preserved).
2. Apply double rotation to the wind vector (u, v, w) of the chunk
   **in memory** using ``diive.flux.WindDoubleRotation``. The rotated W
   is the input to PWB — the rotated data are never written to disk.
3. Run ``PreWhiteningBootstrap`` (Vitale et al. 2024) on the rotated W +
   each scalar + sonic temperature, producing one ``tlag_s`` per gas
   per chunk.
4. Shift each scalar column in the **unrotated** chunk DataFrame backward
   by ``round(tlag_s * hz)`` rows (``pd.Series.shift(periods=-n)``),
   aligning it with the wind.
5. Write the lag-corrected (unrotated) chunk to ``--output-dir`` as its
   own file, with the original metadata header rows and column order
   preserved.

So one 6-hour input file produces up to twelve 30-minute output files.
Chunks shorter than ``--min-chunk-seconds`` (default 300 s = 5 min) are
skipped: PWB needs enough records to fit the block-bootstrap.

The unit of parallel work is **one chunk**: with ``--n-workers N`` set,
chunks across all files are dispatched into a ``ProcessPoolExecutor`` and
N chunks process simultaneously. A single input file with 12 chunks and
4 workers fully saturates all 4 cores (each worker reads only its chunk
slice via ``pd.read_csv(skiprows, nrows)``). Set ``--n-workers 1`` for
sequential in-process execution. All functions live in this single
module.

A checkpoint snapshot
(``<output-dir>/detect_and_remove_tlag_checkpoint.csv``) is written
after every chunk completes — so an interrupted run leaves a partial
result on disk you can inspect or post-process. When the run finishes
cleanly the full results land in
``detect_and_remove_tlag_summary.csv``.

Every CLI run writes a plain-text ``log.txt`` to the output directory
containing every console line (run metadata, per-file progress, errors,
final summary). The log is saved even on exception / KeyboardInterrupt
so a crashed run is still diagnosable.

Downstream flux processing must run with **EC time-lag maximization
disabled** — the tube delay has already been corrected here.

CLI
---
::

    uv run diive-tlag-pwb-detect-remove --help
    uv run python -m diive.flux.hires.detect_and_remove_tlag --help

Example (raw CSV: header on line 1, 2 extra rows for units + source)::

    uv run diive-tlag-pwb-detect-remove ^
        --input-dir  PATH/TO/raw_6h_csv_files ^
        --output-dir PATH/TO/aligned_30min_chunks ^
        --col-u  "U_[R350-B]"  --col-v  "V_[R350-B]"  --col-w  "W_[R350-B]" ^
        --col-tsonic "T_SONIC_[R350-B]" ^
        --scalar "CH4:CH4_DRY_[LGR-A]"  --scalar "N2O:N2O_DRY_[LGR-A]" ^
        --skiprows 0 --extra-rows 2 --sep "," ^
        --hz 20 --lag-max 10.0 --n-bootstrap 99 --block-length 20.0 ^
        --chunk-seconds 1800 --min-chunk-seconds 300 ^
        --chunk-name-template "{stem}_chunk{index:02d}{suffix}" ^
        --file-pattern "*.csv" --random-state 42 --n-workers 4

Filename templating
-------------------
``--chunk-name-template`` supports these placeholders:

- ``{stem}``    — input file stem (filename without extension)
- ``{suffix}``  — input file extension (``'.csv'`` etc.)
- ``{index}``   — 0-based chunk index within the file (use ``{index:02d}``
  to zero-pad)
- ``{starttime}`` — chunk start time formatted with
  ``--start-time-format`` (requires ``--start-time-regex`` to extract a
  timestamp from the input filename)

Examples:

- Default ``"{stem}_chunk{index:02d}{suffix}"`` →
  ``CH-CHA_20210722_1100_to_1700_chunk00.csv``,
  ``..._chunk01.csv``, ...
- With ``--start-time-regex "(\\d{8})_(\\d{4})"`` extracting the file's
  starting timestamp ``YYYYMMDD_HHMM``, the template
  ``"CH-CHA_{starttime}{suffix}"`` produces
  ``CH-CHA_20210722-1100.csv``, ``CH-CHA_20210722-1130.csv``, ...

Outputs in ``--output-dir``:

- One lag-corrected 30-min file per chunk (``hz * 1800`` rows each by
  default).
- ``detect_and_remove_tlag_summary.csv`` with one row per chunk. Schema
  mirrors ``tlag_results.csv`` from ``diive-tlag-pwb-batch`` plus this
  pipeline's extras: parent filename, ``chunk_index``, chunk filename,
  rotation angles ``theta_deg`` / ``phi_deg``, and per gas
  ``{prefix}_tlag_s`` / ``hdi_lo_s`` / ``hdi_hi_s`` / ``hdi_range_s`` /
  ``is_reliable`` / ``tlag_pw_s`` / ``corr_pw`` / ``cov_pwb`` /
  ``ar_order`` / ``best_combination``, plus the PWBOPT post-processing
  columns ``pwbopt_s_std`` / ``flag_std`` / ``pwbopt_s_pf`` /
  ``flag_pf`` / ``tlag_final_s`` / ``tlag_final_pf_s``.
- ``plots/`` (when ``--save-plots`` is set) containing:

  - One ``<chunk_stem>_<gas>.png`` per chunk per gas — the 3-panel PWB
    diagnostic: pre-whitened CCF + raw cross-covariance + bootstrap lag
    histogram with the 95% HDI shaded.
  - One ``summary_<gas>.png`` per scalar — the batch-level 5-panel
    overview from ``PwbBatchDetection.plot_summary``: detected lags
    coloured by S1/S2/S3 flag, gap-filled lags, 95% HDI bars with
    threshold lines, per-period flag bars (standard vs. pre-filtered
    PWBOPT side by side), and a histogram of detected lags.
  - ``summary_lag_comparison.png`` — the cross-scalar
    ``PwboptLagPlot`` scatter + KDE comparing standard vs. pre-filtered
    PWBOPT for every gas.
- ``detect_and_remove_tlag_checkpoint.csv`` — periodically written
  snapshot of the rows accumulated so far; left intact after a clean
  run so it can be diffed against the final summary if useful.
- ``log.txt`` capturing every console line from the run.

Public API
----------
This module exposes:

- ``PerFilePipeline`` — class wrapping the loop; ``.run()`` processes all
  files and returns a per-file summary DataFrame.
- ``process_one_file`` — module-level function that runs the full Read →
  Rotate → Detect → Remove → Write sequence on a single file. Useful from
  Python without the CLI.

Part of the diive library: https://github.com/holukas/diive
"""

import os
import queue as _queue_mod
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from multiprocessing import Manager
from pathlib import Path
from typing import Callable

# Suppress the runpy double-import warning that fires when ``diive.__init__``
# has already imported this module before ``-m`` re-executes it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

# Force headless matplotlib backend BEFORE importing anything that pulls in
# pyplot — required so worker processes can save plots without an X display
# (Windows spawn re-runs this top-level code in every worker).
import matplotlib  # noqa: E402
matplotlib.use('Agg', force=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas import DataFrame, Series  # noqa: E402

from diive.flux.hires.lag_pwb import (  # noqa: E402
    _DEFAULT_NA_VALUES,
    PreWhiteningBootstrap,
    PwbBatchDetection,
)
from diive.flux.hires.windrotation import WindDoubleRotation  # noqa: E402

# Sentinel used in the CLI parser for "whitespace separator". pandas parses
# regex ``r'\s+'`` on read; for writing we substitute a single space.
_WHITESPACE_SEP = r'\s+'


# ---------------------------------------------------------------------------
# File I/O — reads and writes the raw text format verbatim above the data
# block, so the output is a drop-in replacement that downstream flux
# software (EddyPro, FluxRun) sees as the original file with two changes:
# shifted scalar columns and trailing-NaN rows in those columns only.
# ---------------------------------------------------------------------------

def _read_raw_file(
        input_path: Path,
        skiprows: int,
        extra_rows: int,
        sep: str,
        na_values: list,
) -> tuple[list, DataFrame]:
    """Read a raw EC file. Return (preserved_lines, data_df).

    Format model:
        Lines 0 .. skiprows-1                = metadata block (preserved verbatim)
        Line  skiprows                       = column-name header row (preserved,
                                               also used to label the DataFrame)
        Lines skiprows+1 .. skiprows+extra_rows = units / instrument-source rows
                                               (preserved verbatim)
        From line skiprows+1+extra_rows on   = data
    """
    n_preserved = skiprows + 1 + extra_rows
    header_idx = skiprows

    with open(input_path, 'r', encoding='utf-8', errors='replace') as fh:
        preserved_lines = [next(fh) for _ in range(n_preserved)]

    header_line = preserved_lines[header_idx].rstrip('\n').rstrip('\r')
    if sep == _WHITESPACE_SEP:
        header_cols = header_line.split()
    else:
        header_cols = [c.strip() for c in header_line.split(sep)]

    df = pd.read_csv(
        input_path,
        skiprows=n_preserved,
        header=None,
        sep=sep,
        na_values=na_values,
        low_memory=False,
        engine='python' if sep == _WHITESPACE_SEP else 'c',
    )
    if df.shape[1] != len(header_cols):
        raise ValueError(
            f"Header has {len(header_cols)} columns but data has "
            f"{df.shape[1]} for {input_path.name}. Check --skiprows "
            f"({skiprows}), --extra-rows ({extra_rows}), --sep ({sep!r})."
        )
    df.columns = header_cols
    return preserved_lines, df


def _write_raw_file(
        output_path: Path,
        preserved_lines: list,
        df: DataFrame,
        sep: str,
        lineterm: str,
        na_rep: str,
) -> None:
    """Write the lag-corrected file: header lines verbatim, then the data.

    pandas ``to_csv`` requires a literal separator; the whitespace sentinel
    is written as a single space.
    """
    out_sep = ' ' if sep == _WHITESPACE_SEP else sep
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as fh:
        for line in preserved_lines:
            fh.write(line)
        df.to_csv(fh, sep=out_sep, index=False, header=False,
                  na_rep=na_rep, lineterminator=lineterm)


# ---------------------------------------------------------------------------
# Chunk filename templating
# ---------------------------------------------------------------------------

def _chunk_filename(
        input_path: Path,
        chunk_index: int,
        chunk_seconds: float,
        name_template: str,
        start_time_regex: str | None,
        start_time_format: str,
) -> tuple[str, 'datetime | None']:
    """Compose the output filename for one chunk inside ``input_path``.

    Returns a tuple of (chunk_filename, chunk_start_datetime). The second
    element is ``None`` when ``start_time_regex`` was not provided or did
    not match — callers (e.g. the row builder) can store an empty
    timestamp in that case.

    Supported placeholders: ``{stem}``, ``{suffix}``, ``{index}``,
    ``{starttime}`` (only available when ``start_time_regex`` matches
    ``input_path.name``).
    """
    fields: dict = {
        'stem': input_path.stem,
        'suffix': input_path.suffix,
        'index': chunk_index,
    }
    t_chunk: datetime | None = None

    # Catch the most common user-config bug eagerly so the error message
    # points at the actual mistake (regex didn't match the filename) rather
    # than at a downstream KeyError on ``{starttime}``.
    template_uses_starttime = '{starttime}' in name_template
    if template_uses_starttime and not start_time_regex:
        raise ValueError(
            f"chunk-name-template {name_template!r} uses {{starttime}} but "
            f"--start-time-regex was not provided."
        )

    if start_time_regex:
        m = re.search(start_time_regex, input_path.name)
        if m is None:
            if template_uses_starttime:
                raise ValueError(
                    f"--start-time-regex {start_time_regex!r} did not match "
                    f"input filename {input_path.name!r}. "
                    f"Adjust the regex to match your filename, or remove "
                    f"{{starttime}} from --chunk-name-template."
                )
            # Regex set but template doesn't use {starttime} — quietly skip.
        else:
            t0_str = (''.join(g for g in m.groups() if g is not None)
                      if m.groups() else m.group(0))
            try:
                t0 = datetime.strptime(t0_str, start_time_format)
            except ValueError as e:
                raise ValueError(
                    f"--start-time-regex matched {t0_str!r} in "
                    f"{input_path.name!r}, but parsing with "
                    f"--start-time-format {start_time_format!r} failed: {e}. "
                    f"Check that the format spec matches the captured text."
                )
            t_chunk = t0 + timedelta(seconds=chunk_index * chunk_seconds)
            fields['starttime'] = t_chunk.strftime(start_time_format)

    try:
        return name_template.format(**fields), t_chunk
    except KeyError as e:
        raise ValueError(
            f"--chunk-name-template {name_template!r} uses placeholder {e}; "
            f"available: {sorted(fields.keys())}"
        )


# ---------------------------------------------------------------------------
# Core per-file pipeline — Read → (chunk × Rotate → Detect → Remove → Write)
# ---------------------------------------------------------------------------

def process_one_file(
        input_path: Path,
        output_dir: Path,
        col_u: str,
        col_v: str,
        col_w: str,
        col_tsonic: str,
        scalars: dict,
        hz: int = 20,
        lag_max_s: float = 10.0,
        n_bootstrap: int = 99,
        block_length_s: float = 20.0,
        chunk_seconds: float = 1800.0,
        min_chunk_seconds: float = 300.0,
        chunk_name_template: str = '{stem}_chunk{index:02d}{suffix}',
        start_time_regex: str | None = None,
        start_time_format: str = '%Y%m%d-%H%M',
        skiprows: int = 0,
        extra_rows: int = 2,
        sep: str = ',',
        lineterm: str = '\n',
        na_values: list | None = None,
        na_rep: str = '-9999',
        random_state: int | None = None,
        strict: bool = False,
        save_plots: bool = False,
        plots_dir: Path | None = None,
        progress_queue=None,
) -> list:
    """Run the per-chunk PWB pipeline on one (possibly multi-hour) input file.

    The input is read once and split into fixed-length chunks of
    ``round(chunk_seconds * hz)`` rows. Each chunk independently goes
    through Rotate → Detect → Remove → Write. The last chunk may be
    shorter; if its length is below ``round(min_chunk_seconds * hz)`` it
    is skipped (PWB needs enough records for a meaningful bootstrap).

    Output filenames are composed via ``chunk_name_template`` (see the
    module docstring for placeholder semantics). The same preserved-
    header lines from the input are written verbatim at the top of every
    chunk file.

    ``scalars`` maps gas labels (e.g. ``'CH4'``) to column names in the
    raw file (e.g. ``'CH4_DRY_[LGR-A]'``).

    Returns a list of per-chunk summary dicts. Each dict has keys:
    ``period`` (output chunk filename), ``parent`` (input filename),
    ``chunk_index``, ``chunk_records``, ``status`` (``ok`` / ``error`` /
    ``skipped:short``), ``error``, ``theta_deg``, ``phi_deg``
    (rotation angles for the chunk), and per-gas ``{prefix}_tlag_s``,
    ``{prefix}_hdi_range_s``, ``{prefix}_is_reliable``,
    ``{prefix}_applied_records``, ``{prefix}_status``.

    On error (when ``strict=False``), one error row is returned for the
    file as a whole (status=``error``, ``error`` field carries the
    exception class + message).

    When ``progress_queue`` is provided (a ``multiprocessing.Queue`` or
    ``Manager.Queue`` proxy), the function emits live progress events as
    it works through the chunks: ``{'event': 'start', ...}`` when each
    chunk begins, and ``{'event': 'done', ...}`` (or ``'error'`` /
    ``'skipped'``) when it finishes. The main process can drain this
    queue to drive a per-chunk progress display.
    """
    if na_values is None:
        na_values = list(_DEFAULT_NA_VALUES)

    parent_name = Path(input_path).name
    chunk_records = int(round(chunk_seconds * hz))
    min_chunk_records = int(round(min_chunk_seconds * hz))

    def _empty_chunk_row(chunk_index: int, chunk_period: str,
                         n_rows: int, status: str, err: str = '',
                         t_chunk: 'datetime | None' = None) -> dict:
        """Build a row with all per-gas fields populated as NaN/default.

        Field layout mirrors ``tlag_results.csv`` produced by
        ``diive-tlag-pwb-batch`` so the summary CSV from this pipeline is
        a drop-in equivalent (downstream tools that already read the PWB
        batch results will work unchanged).
        """
        r: dict = {
            'period': chunk_period,
            'parent': parent_name,
            'timestamp': t_chunk.isoformat() if t_chunk is not None else '',
            'chunk_index': chunk_index,
            'chunk_records': n_rows,
            'status': status,
            'error': err,
            'theta_deg': np.nan,
            'phi_deg': np.nan,
        }
        for label in scalars:
            pfx = label.lower()
            # Mirror the PWB batch result schema
            r[f'{pfx}_tlag_s'] = np.nan
            r[f'{pfx}_hdi_lo_s'] = np.nan
            r[f'{pfx}_hdi_hi_s'] = np.nan
            r[f'{pfx}_hdi_range_s'] = np.nan
            r[f'{pfx}_is_reliable'] = False
            r[f'{pfx}_tlag_pw_s'] = np.nan
            r[f'{pfx}_corr_pw'] = np.nan
            r[f'{pfx}_cov_pwb'] = np.nan
            r[f'{pfx}_ar_order'] = np.nan
            r[f'{pfx}_best_combination'] = ''
            # Applied-shift bookkeeping (specific to this pipeline)
            r[f'{pfx}_applied_records'] = np.nan
            r[f'{pfx}_status'] = 'pending'
        return r

    rows: list = []
    output_dir = Path(output_dir)
    worker_pid = os.getpid()

    def _emit(event: str, **extra):
        """Push a progress event to the queue if one was provided."""
        if progress_queue is None:
            return
        try:
            progress_queue.put({
                'event': event,
                'pid': worker_pid,
                'parent': parent_name,
                **extra,
            })
        except Exception:
            # Never let a flaky queue kill the worker.
            pass

    try:
        # ---- READ once (all chunks share the preserved header lines) -----
        preserved_lines, df_raw = _read_raw_file(
            input_path, skiprows=skiprows, extra_rows=extra_rows,
            sep=sep, na_values=na_values,
        )
        required = (col_u, col_v, col_w, col_tsonic, *scalars.values())
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            raise ValueError(f'columns missing from {parent_name}: {missing}')

        n_total = len(df_raw)
        n_chunks = (n_total + chunk_records - 1) // chunk_records

        for ci in range(n_chunks):
            i0 = ci * chunk_records
            i1 = min(i0 + chunk_records, n_total)
            chunk_n = i1 - i0
            chunk_period, _t_chunk = _chunk_filename(
                input_path=Path(input_path),
                chunk_index=ci,
                chunk_seconds=chunk_seconds,
                name_template=chunk_name_template,
                start_time_regex=start_time_regex,
                start_time_format=start_time_format,
            )

            # Skip too-short trailing chunks (PWB needs enough records).
            if chunk_n < min_chunk_records:
                row = _empty_chunk_row(
                    ci, chunk_period, chunk_n,
                    status='skipped:short',
                    err=f'chunk has {chunk_n} rows < min {min_chunk_records}',
                    t_chunk=_t_chunk,
                )
                rows.append(row)
                _emit('done', chunk_index=ci,
                      chunk_period=chunk_period, row=row)
                continue

            _emit('start', chunk_index=ci, chunk_period=chunk_period)
            chunk_row = _empty_chunk_row(ci, chunk_period, chunk_n, 'ok',
                                          t_chunk=_t_chunk)
            try:
                df_chunk = df_raw.iloc[i0:i1].copy()

                # ---- ROTATE on this chunk only --------------------------
                wr = WindDoubleRotation(
                    u=df_chunk[col_u].astype(float),
                    v=df_chunk[col_v].astype(float),
                    w=df_chunk[col_w].astype(float),
                )
                chunk_row['theta_deg'] = float(np.degrees(wr.theta))
                chunk_row['phi_deg'] = float(np.degrees(wr.phi))

                # ---- DETECT (PWB on rotated W per scalar) ---------------
                detected_lags: dict = {}
                for gi, (label, col_name) in enumerate(scalars.items()):
                    pfx = label.lower()
                    pwb_df = pd.DataFrame({
                        'W_rot': wr.w2.reset_index(drop=True),
                        label: df_chunk[col_name].astype(float).reset_index(drop=True),
                        'T_SONIC': df_chunk[col_tsonic].astype(float).reset_index(drop=True),
                    })
                    # Per-chunk + per-gas reproducible seed.
                    seed = (None if random_state is None
                            else int(random_state) + ci * 100 + gi)
                    pwb = PreWhiteningBootstrap(
                        df=pwb_df,
                        var_w='W_rot',
                        var_scalar=label,
                        var_tsonic='T_SONIC',
                        hz=hz,
                        lag_max_s=lag_max_s,
                        n_bootstrap=n_bootstrap,
                        block_length_s=block_length_s,
                        segment_name=chunk_period,
                        random_state=seed,
                    )
                    pwb.run()
                    res = pwb.results
                    detected_lags[label] = res['tlag_s']

                    # Full schema (matches diive-tlag-pwb-batch results CSV)
                    chunk_row[f'{pfx}_tlag_s'] = float(res.get('tlag_s', np.nan))
                    chunk_row[f'{pfx}_hdi_lo_s'] = float(res.get('hdi_lo_s', np.nan))
                    chunk_row[f'{pfx}_hdi_hi_s'] = float(res.get('hdi_hi_s', np.nan))
                    chunk_row[f'{pfx}_hdi_range_s'] = float(res.get('hdi_range_s', np.nan))
                    chunk_row[f'{pfx}_is_reliable'] = bool(res.get('is_reliable', False))
                    if 'tlag_pw_s' in res:
                        chunk_row[f'{pfx}_tlag_pw_s'] = float(res['tlag_pw_s'])
                    if 'corr_pw' in res:
                        chunk_row[f'{pfx}_corr_pw'] = float(res['corr_pw'])
                    if 'cov_pwb' in res:
                        chunk_row[f'{pfx}_cov_pwb'] = float(res['cov_pwb'])
                    if 'ar_order' in res and res['ar_order'] is not None:
                        chunk_row[f'{pfx}_ar_order'] = float(res['ar_order'])
                    if 'best_combination' in res and res['best_combination']:
                        chunk_row[f'{pfx}_best_combination'] = str(res['best_combination'])

                    # Save the 3-panel diagnostic plot (one per chunk per gas)
                    # to <output_dir>/plots/. Matplotlib uses the Agg backend
                    # set at module top, so this is safe inside workers.
                    if save_plots and plots_dir is not None:
                        try:
                            chunk_stem = Path(chunk_period).stem
                            plot_name = f'{chunk_stem}_{pfx}.png'
                            pwb.plot(
                                title=f'{chunk_stem} | {label}',
                                showplot=False,
                                outpath=str(plots_dir),
                                outname=plot_name,
                            )
                        except Exception:
                            # Plotting must never fail the detection itself.
                            pass

                # ---- REMOVE (shift scalars in the UNROTATED chunk) ------
                df_out = df_chunk.copy()
                for label, col_name in scalars.items():
                    pfx = label.lower()
                    lag_s = detected_lags[label]
                    if not np.isfinite(lag_s):
                        chunk_row[f'{pfx}_status'] = 'skipped:lag_nan'
                        continue
                    n_records = int(round(lag_s * hz))
                    df_out[col_name] = df_out[col_name].shift(periods=-n_records)
                    chunk_row[f'{pfx}_applied_records'] = n_records
                    chunk_row[f'{pfx}_status'] = 'ok'

                # ---- WRITE chunk as its own file ------------------------
                output_path = output_dir / chunk_period
                _write_raw_file(
                    output_path, preserved_lines, df_out,
                    sep=sep, lineterm=lineterm, na_rep=na_rep,
                )

            except Exception as ce:
                if strict:
                    raise
                chunk_row['status'] = 'error'
                chunk_row['error'] = f'{type(ce).__name__}: {ce}'

            rows.append(chunk_row)
            _emit('done', chunk_index=ci, chunk_period=chunk_period,
                  row=chunk_row)

    except Exception as e:
        if strict:
            raise
        # File-level failure (read failed, missing columns, etc.). Emit one
        # row labelled with the input filename so the user sees it in the
        # summary CSV.
        err_row = _empty_chunk_row(
            chunk_index=-1, chunk_period=parent_name, n_rows=0,
            status='error', err=f'{type(e).__name__}: {e}',
        )
        rows.append(err_row)
        _emit('done', chunk_index=-1, chunk_period=parent_name,
              row=err_row)

    return rows


# ---------------------------------------------------------------------------
# Per-chunk processor — the actual unit of work dispatched by the parallel
# pipeline. Reads only its slice from the input file (via pd.read_csv with
# skiprows + nrows), runs Rotate → Detect → Remove → Write for that one
# chunk, and returns its row dict. Pickle-safe (module-level function).
# ---------------------------------------------------------------------------

def process_one_chunk(
        input_path: Path,
        output_dir: Path,
        chunk_index: int,
        chunk_records: int,
        min_chunk_records: int,
        col_u: str,
        col_v: str,
        col_w: str,
        col_tsonic: str,
        scalars: dict,
        hz: int,
        lag_max_s: float,
        n_bootstrap: int,
        block_length_s: float,
        chunk_seconds: float,
        chunk_name_template: str,
        start_time_regex: str | None,
        start_time_format: str,
        skiprows: int,
        extra_rows: int,
        sep: str,
        lineterm: str,
        na_values: list,
        na_rep: str,
        random_state: int | None,
        strict: bool,
        save_plots: bool,
        plots_dir: Path | None,
        progress_queue=None,
) -> dict:
    """Run Rotate → Detect → Remove → Write for one chunk inside one file.

    Reads only this chunk's data slice from the file (header lines from
    the top, then ``pd.read_csv(skiprows=n_header + chunk_index *
    chunk_records, nrows=chunk_records)``). Pickle-safe so it can be
    dispatched directly to a ``ProcessPoolExecutor``.

    Returns one chunk-row dict (same schema as one element of
    ``process_one_file``'s return list).
    """
    parent_name = Path(input_path).name
    worker_pid = os.getpid()
    output_dir = Path(output_dir)

    chunk_period, t_chunk = _chunk_filename(
        input_path=Path(input_path),
        chunk_index=chunk_index,
        chunk_seconds=chunk_seconds,
        name_template=chunk_name_template,
        start_time_regex=start_time_regex,
        start_time_format=start_time_format,
    )
    timestamp_iso = t_chunk.isoformat() if t_chunk is not None else ''

    def _emit(event: str, **extra):
        if progress_queue is None:
            return
        try:
            progress_queue.put({
                'event': event,
                'pid': worker_pid,
                'parent': parent_name,
                **extra,
            })
        except Exception:
            pass

    def _empty_row(status: str, n_rows: int, err: str = '') -> dict:
        r = {
            'period': chunk_period,
            'parent': parent_name,
            'timestamp': timestamp_iso,  # ISO chunk-start; empty if no regex
            'chunk_index': chunk_index,
            'chunk_records': n_rows,
            'status': status,
            'error': err,
            'theta_deg': np.nan,
            'phi_deg': np.nan,
        }
        for label in scalars:
            pfx = label.lower()
            r[f'{pfx}_tlag_s'] = np.nan
            r[f'{pfx}_hdi_lo_s'] = np.nan
            r[f'{pfx}_hdi_hi_s'] = np.nan
            r[f'{pfx}_hdi_range_s'] = np.nan
            r[f'{pfx}_is_reliable'] = False
            r[f'{pfx}_tlag_pw_s'] = np.nan
            r[f'{pfx}_corr_pw'] = np.nan
            r[f'{pfx}_cov_pwb'] = np.nan
            r[f'{pfx}_ar_order'] = np.nan
            r[f'{pfx}_best_combination'] = ''
            r[f'{pfx}_applied_records'] = np.nan
            r[f'{pfx}_status'] = 'pending'
        return r

    n_preserved = skiprows + 1 + extra_rows
    skiprows_total = n_preserved + chunk_index * chunk_records

    try:
        # ---- Read preserved header (small) -------------------------------
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fh:
            preserved_lines = [next(fh) for _ in range(n_preserved)]

        header_line = preserved_lines[skiprows].rstrip('\n').rstrip('\r')
        if sep == _WHITESPACE_SEP:
            header_cols = header_line.split()
        else:
            header_cols = [c.strip() for c in header_line.split(sep)]

        # ---- Read only this chunk's data slice ---------------------------
        df_chunk = pd.read_csv(
            input_path,
            skiprows=skiprows_total,
            nrows=chunk_records,
            header=None,
            sep=sep,
            na_values=na_values,
            low_memory=False,
            engine='python' if sep == _WHITESPACE_SEP else 'c',
        )
        chunk_n = len(df_chunk)

        # Short trailing chunk (file shorter than expected): emit a single
        # 'done' event with skipped status so the bar still advances.
        if chunk_n < min_chunk_records:
            row = _empty_row(
                status='skipped:short',
                n_rows=chunk_n,
                err=f'chunk has {chunk_n} rows < min {min_chunk_records}',
            )
            _emit('done', chunk_index=chunk_index,
                  chunk_period=chunk_period, row=row)
            return row

        if df_chunk.shape[1] != len(header_cols):
            raise ValueError(
                f"Header has {len(header_cols)} columns but chunk data "
                f"has {df_chunk.shape[1]} for {parent_name} chunk "
                f"{chunk_index}. Check --skiprows / --extra-rows / --sep."
            )
        df_chunk.columns = header_cols

        required = (col_u, col_v, col_w, col_tsonic, *scalars.values())
        missing = [c for c in required if c not in df_chunk.columns]
        if missing:
            raise ValueError(
                f'columns missing from {parent_name} chunk {chunk_index}: '
                f'{missing}'
            )

        _emit('start', chunk_index=chunk_index, chunk_period=chunk_period)

        row = _empty_row(status='ok', n_rows=chunk_n)

        # ---- Rotate ------------------------------------------------------
        wr = WindDoubleRotation(
            u=df_chunk[col_u].astype(float),
            v=df_chunk[col_v].astype(float),
            w=df_chunk[col_w].astype(float),
        )
        row['theta_deg'] = float(np.degrees(wr.theta))
        row['phi_deg'] = float(np.degrees(wr.phi))

        # ---- Detect (PWB on rotated W per scalar) -----------------------
        detected_lags: dict = {}
        for gi, (label, col_name) in enumerate(scalars.items()):
            pfx = label.lower()
            pwb_df = pd.DataFrame({
                'W_rot': wr.w2.reset_index(drop=True),
                label: df_chunk[col_name].astype(float).reset_index(drop=True),
                'T_SONIC': df_chunk[col_tsonic].astype(float).reset_index(drop=True),
            })
            seed = (None if random_state is None
                    else int(random_state) + gi)
            pwb = PreWhiteningBootstrap(
                df=pwb_df,
                var_w='W_rot',
                var_scalar=label,
                var_tsonic='T_SONIC',
                hz=hz,
                lag_max_s=lag_max_s,
                n_bootstrap=n_bootstrap,
                block_length_s=block_length_s,
                segment_name=chunk_period,
                random_state=seed,
            )
            pwb.run()
            res = pwb.results
            detected_lags[label] = res['tlag_s']

            row[f'{pfx}_tlag_s'] = float(res.get('tlag_s', np.nan))
            row[f'{pfx}_hdi_lo_s'] = float(res.get('hdi_lo_s', np.nan))
            row[f'{pfx}_hdi_hi_s'] = float(res.get('hdi_hi_s', np.nan))
            row[f'{pfx}_hdi_range_s'] = float(res.get('hdi_range_s', np.nan))
            row[f'{pfx}_is_reliable'] = bool(res.get('is_reliable', False))
            if 'tlag_pw_s' in res:
                row[f'{pfx}_tlag_pw_s'] = float(res['tlag_pw_s'])
            if 'corr_pw' in res:
                row[f'{pfx}_corr_pw'] = float(res['corr_pw'])
            if 'cov_pwb' in res:
                row[f'{pfx}_cov_pwb'] = float(res['cov_pwb'])
            if 'ar_order' in res and res['ar_order'] is not None:
                row[f'{pfx}_ar_order'] = float(res['ar_order'])
            if 'best_combination' in res and res['best_combination']:
                row[f'{pfx}_best_combination'] = str(res['best_combination'])

            if save_plots and plots_dir is not None:
                try:
                    chunk_stem = Path(chunk_period).stem
                    pwb.plot(
                        title=f'{chunk_stem} | {label}',
                        showplot=False,
                        outpath=str(plots_dir),
                        outname=f'{chunk_stem}_{pfx}.png',
                    )
                except Exception:
                    pass

        # ---- Remove (shift scalars in the UNROTATED chunk) --------------
        df_out = df_chunk.copy()
        for label, col_name in scalars.items():
            pfx = label.lower()
            lag_s = detected_lags[label]
            if not np.isfinite(lag_s):
                row[f'{pfx}_status'] = 'skipped:lag_nan'
                continue
            n_records = int(round(lag_s * hz))
            df_out[col_name] = df_out[col_name].shift(periods=-n_records)
            row[f'{pfx}_applied_records'] = n_records
            row[f'{pfx}_status'] = 'ok'

        # ---- Write -------------------------------------------------------
        output_path = output_dir / chunk_period
        _write_raw_file(
            output_path, preserved_lines, df_out,
            sep=sep, lineterm=lineterm, na_rep=na_rep,
        )

    except Exception as e:
        if strict:
            raise
        row = _empty_row(status='error', n_rows=0,
                         err=f'{type(e).__name__}: {e}')
        _emit('done', chunk_index=chunk_index, chunk_period=chunk_period,
              row=row)
        return row

    _emit('done', chunk_index=chunk_index, chunk_period=chunk_period,
          row=row)
    return row


# ---------------------------------------------------------------------------
# Module-level worker wrappers — single-arg so ProcessPoolExecutor can pickle
# the call easily. Each worker process re-imports this module via Windows
# spawn; module-level functions are required.
# ---------------------------------------------------------------------------

def _chunk_worker(kwargs: dict, progress_queue=None) -> dict:
    """Run ``process_one_chunk`` in a child process."""
    return process_one_chunk(progress_queue=progress_queue, **kwargs)


def _per_file_worker(kwargs: dict, progress_queue=None) -> list:
    """Run ``process_one_file`` in a child process (sequential per-file path)."""
    return process_one_file(progress_queue=progress_queue, **kwargs)


def _count_data_rows(path: Path, header_lines: int) -> int:
    """Fast newline count of a text file, minus the header rows.

    Used to estimate the per-file chunk count up-front so the progress
    bar can show an accurate total. Reads the file in 1 MiB chunks via
    binary mode — for a 100 MiB CSV this takes ~200 ms on SSD.
    """
    n_lines = 0
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            n_lines += block.count(b'\n')
    # Most files end without a trailing newline; the last data row still
    # counts. Subtract only the preserved header.
    return max(0, n_lines - header_lines)


# ---------------------------------------------------------------------------
# PerFilePipeline — class wrapper for the loop
# ---------------------------------------------------------------------------

class PerFilePipeline:
    """Sequential per-file PWB detect + remove pipeline.

    For each file matched by ``file_pattern`` in ``input_dir``, performs
    Read → Rotate → Detect → Remove → Write end-to-end before moving on to
    the next file.

    See the module docstring for the full workflow rationale and the CLI
    flag reference. After ``.run()``, ``.summary`` returns a DataFrame with
    one row per file (rotation angles, detected lag per gas, applied
    records, reliability flags, error messages).
    """

    def __init__(
            self,
            input_dir: Path,
            output_dir: Path,
            col_u: str,
            col_v: str,
            col_w: str,
            col_tsonic: str,
            scalars: dict,
            hz: int = 20,
            lag_max_s: float = 10.0,
            n_bootstrap: int = 99,
            block_length_s: float = 20.0,
            chunk_seconds: float = 1800.0,
            min_chunk_seconds: float = 300.0,
            chunk_name_template: str = '{stem}_chunk{index:02d}{suffix}',
            start_time_regex: str | None = None,
            start_time_format: str = '%Y%m%d-%H%M',
            file_pattern: str = '*.csv',
            skiprows: int = 0,
            extra_rows: int = 2,
            sep: str = ',',
            lineterm: str = '\n',
            na_values: list | None = None,
            na_rep: str = '-9999',
            random_state: int | None = None,
            n_workers: int | None = None,
            strict: bool = False,
            save_plots: bool = False,
            plots_subdir: str = 'plots',
            hdi_thresh: float = 0.5,
            dev_thresh: float = 0.5,
            hdi_prefilter: float = 1.0,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.col_u = col_u
        self.col_v = col_v
        self.col_w = col_w
        self.col_tsonic = col_tsonic
        self.scalars = scalars
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.chunk_seconds = chunk_seconds
        self.min_chunk_seconds = min_chunk_seconds
        self.chunk_name_template = chunk_name_template
        self.start_time_regex = start_time_regex
        self.start_time_format = start_time_format
        self.file_pattern = file_pattern
        self.skiprows = skiprows
        self.extra_rows = extra_rows
        self.sep = sep
        self.lineterm = lineterm
        self.na_values = na_values if na_values is not None else list(_DEFAULT_NA_VALUES)
        self.na_rep = na_rep
        self.random_state = random_state
        # n_workers=None or n_workers<=0 -> use all cpu cores. n_workers=1 -> sequential
        # in-process (cleaner stack traces for debugging).
        self.n_workers = n_workers if n_workers and n_workers > 0 else (os.cpu_count() or 1)
        self.strict = strict
        self.save_plots = save_plots
        self.plots_subdir = plots_subdir
        # PWBOPT post-processing thresholds (paper Section 2.3 defaults)
        self.hdi_thresh = hdi_thresh
        self.dev_thresh = dev_thresh
        self.hdi_prefilter = hdi_prefilter

        self._summary: DataFrame | None = None

    @property
    def summary(self) -> DataFrame:
        """Per-file summary (available after ``run()``)."""
        if self._summary is None:
            raise RuntimeError('Call run() first.')
        return self._summary

    def estimate_chunks_per_file(self, sample_file: Path) -> int:
        """Quick newline count of one file to estimate per-file chunk count."""
        header_lines = self.skiprows + 1 + self.extra_rows
        n_rows = _count_data_rows(sample_file, header_lines)
        chunk_records = int(round(self.chunk_seconds * self.hz))
        return max(1, (n_rows + chunk_records - 1) // chunk_records)

    def run(self,
            on_progress: Callable | None = None,
            on_active: Callable | None = None) -> DataFrame:
        """Process every matching file. Return a per-chunk summary DataFrame.

        Each file is processed in one worker (its 30-min chunks run
        sequentially within that worker). Multiple files run in parallel
        via ``ProcessPoolExecutor`` when ``n_workers > 1``; when
        ``n_workers == 1`` the loop runs in-process for cleaner stack
        traces during debugging.

        Args:
            on_progress: Optional callback
                ``f(chunks_completed, total_chunks_estimate, chunk_row)``
                fired once per chunk as work progresses.
            on_active: Optional callback ``f(active_dict)`` where
                ``active_dict`` maps worker pid -> ``{'parent', 'chunk_index',
                'chunk_period'}``. Fired whenever the in-flight set
                changes (a chunk starts or finishes), so a CLI can render
                a live "currently processing" display.

        Returns:
            DataFrame with one row per (file, chunk), sorted by input-file
            order then chunk_index.
        """
        files = sorted(self.input_dir.glob(self.file_pattern))
        if not files:
            raise FileNotFoundError(
                f'No files matched {self.file_pattern!r} in {self.input_dir}'
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = (self.output_dir / self.plots_subdir) if self.save_plots else None
        if plots_dir is not None:
            plots_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(files)
        chunks_per_file = self.estimate_chunks_per_file(files[0])
        total_chunks_est = total_files * chunks_per_file

        chunk_records = int(round(self.chunk_seconds * self.hz))
        min_chunk_records = int(round(self.min_chunk_seconds * self.hz))

        # Build a flat list of CHUNK tasks (file × chunk_index). The actual
        # number of chunks per file is determined when each worker reads
        # its slice — short trailing chunks are returned with
        # ``status='skipped:short'`` so the bar still advances cleanly.
        chunk_kwargs_list: list[dict] = []
        for i, f in enumerate(files):
            file_seed_base = (None if self.random_state is None
                              else int(self.random_state) + i * 10_000)
            for ci in range(chunks_per_file):
                chunk_seed = (None if file_seed_base is None
                              else file_seed_base + ci * 100)
                chunk_kwargs_list.append(dict(
                    input_path=f,
                    output_dir=self.output_dir,
                    chunk_index=ci,
                    chunk_records=chunk_records,
                    min_chunk_records=min_chunk_records,
                    col_u=self.col_u,
                    col_v=self.col_v,
                    col_w=self.col_w,
                    col_tsonic=self.col_tsonic,
                    scalars=self.scalars,
                    hz=self.hz,
                    lag_max_s=self.lag_max_s,
                    n_bootstrap=self.n_bootstrap,
                    block_length_s=self.block_length_s,
                    chunk_seconds=self.chunk_seconds,
                    chunk_name_template=self.chunk_name_template,
                    start_time_regex=self.start_time_regex,
                    start_time_format=self.start_time_format,
                    skiprows=self.skiprows,
                    extra_rows=self.extra_rows,
                    sep=self.sep,
                    lineterm=self.lineterm,
                    na_values=self.na_values,
                    na_rep=self.na_rep,
                    random_state=chunk_seed,
                    strict=self.strict,
                    save_plots=self.save_plots,
                    plots_dir=plots_dir,
                ))
        total_chunks_est = len(chunk_kwargs_list)

        # Path for the checkpoint snapshot. Written after every chunk
        # completes so an interrupted run can be inspected (and, in
        # principle, manually re-driven by feeding the surviving rows
        # back through `_apply_pwbopt_postprocessing`).
        checkpoint_path = self.output_dir / 'detect_and_remove_tlag_checkpoint.csv'

        rows: list[dict] = []
        active: dict = {}  # pid -> {'parent', 'chunk_index', 'chunk_period'}
        chunks_done = 0
        # Map input-file name -> dispatch order, for stable sorting
        parent_to_idx = {f.name: i for i, f in enumerate(files)}

        def _checkpoint_save():
            """Persist current rows so an interrupted run can be inspected.

            Sort by (parent, chunk_index) for reader-friendliness. Wrapped
            in try/except so a locked CSV (e.g. open in Excel) never kills
            the run — matches ``PwbBatchDetection``'s pattern.
            """
            if not rows:
                return
            sorted_rows = sorted(rows, key=lambda r: (
                parent_to_idx.get(r.get('parent', ''), total_files),
                r.get('chunk_index', -1),
            ))
            try:
                pd.DataFrame(sorted_rows).to_csv(checkpoint_path, index=False)
            except PermissionError:
                pass  # file locked, just skip this snapshot

        def _handle_event(ev: dict):
            """Apply one queue event: update active set, append row, fan out."""
            nonlocal chunks_done
            pid = ev.get('pid')
            kind = ev['event']
            if kind == 'start':
                active[pid] = {
                    'parent': ev['parent'],
                    'chunk_index': ev['chunk_index'],
                    'chunk_period': ev['chunk_period'],
                }
                if on_active is not None:
                    on_active(active)
            elif kind == 'done':
                active.pop(pid, None)
                row = ev['row']
                rows.append(row)
                chunks_done += 1
                # Snapshot after every chunk so a crash leaves a usable
                # CSV. ~thousands of small writes on local SSD is cheap;
                # if it ever becomes a bottleneck we can throttle.
                _checkpoint_save()
                if on_progress is not None:
                    on_progress(chunks_done, total_chunks_est, row)
                if on_active is not None:
                    on_active(active)

        if self.n_workers == 1:
            # In-process loop. Drain start/done events synchronously via
            # a small queue-like callback (no real IPC needed). Iterates
            # the same per-chunk task list the parallel path dispatches,
            # so behaviour matches exactly.
            class _SyncQueue:
                def put(self, ev):
                    _handle_event(ev)

            sync_q = _SyncQueue()
            for kwargs in chunk_kwargs_list:
                process_one_chunk(progress_queue=sync_q, **kwargs)
        else:
            # Parallel: spin up a Manager queue, dispatch ALL chunks as
            # independent tasks, drain the queue in the main process while
            # workers run. With N workers and M chunks, all N workers stay
            # busy until min(M, N) chunks remain — including the case of
            # one input file with 12 chunks and 4 workers (3 chunks each).
            with Manager() as manager:
                progress_queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(_chunk_worker, kwargs, progress_queue)
                        for kwargs in chunk_kwargs_list
                    ]

                    pending = set(futures)
                    while True:
                        try:
                            ev = progress_queue.get(timeout=0.1)
                        except _queue_mod.Empty:
                            pending = {f for f in pending if not f.done()}
                            if pending:
                                continue
                            # All workers finished — drain any straggler
                            # events queued between timeout and exit.
                            drained_any = False
                            while True:
                                try:
                                    ev = progress_queue.get_nowait()
                                except _queue_mod.Empty:
                                    break
                                _handle_event(ev)
                                drained_any = True
                            if not drained_any:
                                break
                            continue
                        _handle_event(ev)

                    for fut in futures:
                        try:
                            fut.result()
                        except Exception:
                            pass

        # Final sort (some chunks may have arrived out of order)
        rows.sort(key=lambda r: (
            parent_to_idx.get(r.get('parent', ''), total_files),
            r.get('chunk_index', -1),
        ))

        summary = pd.DataFrame(rows)
        summary = self._apply_pwbopt_postprocessing(summary)
        self._summary = summary
        return self._summary

    def _apply_pwbopt_postprocessing(self, summary: DataFrame) -> DataFrame:
        """Add PWBOPT std + pre-filtered + filled-final columns to the summary.

        Matches the schema produced by ``diive-tlag-pwb-batch``'s
        ``_cli_main``: for each scalar ``X`` (lowercased label), the columns
        ``x_pwbopt_s_std`` / ``x_flag_std`` / ``x_pwbopt_s_pf`` /
        ``x_flag_pf`` / ``x_tlag_final_s`` / ``x_tlag_final_pf_s`` are
        appended. PWBOPT is the per-scalar carry-forward decision rule
        from paper Section 2.3 (S1 / S2 / S3); the pre-filtered variant
        drops wide-HDI detections before PWBOPT runs.

        Operates on the summary as it stands after ``run()`` — chunks
        sorted by (parent, chunk_index) so the PWBOPT carry-forward
        respects temporal order across the full dataset.
        """
        if summary.empty:
            return summary
        for label in self.scalars:
            pfx = label.lower()
            tlag_col = f'{pfx}_tlag_s'
            hdi_col = f'{pfx}_hdi_range_s'
            if tlag_col not in summary.columns or hdi_col not in summary.columns:
                continue

            tlag = summary[tlag_col].to_numpy(dtype=float)
            hdi = summary[hdi_col].to_numpy(dtype=float)

            # Standard PWBOPT (S1/S2/S3 directly on the raw mode lag)
            std = PwbBatchDetection.apply_pwbopt(
                tlag, hdi, self.hdi_thresh, self.dev_thresh)
            summary[f'{pfx}_pwbopt_s_std'] = std['pwbopt_s'].to_numpy()
            summary[f'{pfx}_flag_std'] = std['flag'].to_numpy()

            # Pre-filtered PWBOPT: drop lags with HDI > prefilter, then S1/S2/S3
            if self.hdi_prefilter > 0:
                tlag_pf = PwbBatchDetection.apply_hdi_prefilter(
                    tlag, hdi, self.hdi_prefilter)
                pf = PwbBatchDetection.apply_pwbopt(
                    tlag_pf, hdi, self.hdi_thresh, self.dev_thresh)
                summary[f'{pfx}_pwbopt_s_pf'] = pf['pwbopt_s'].to_numpy()
                summary[f'{pfx}_flag_pf'] = pf['flag'].to_numpy()
            else:
                summary[f'{pfx}_pwbopt_s_pf'] = std['pwbopt_s'].to_numpy()
                summary[f'{pfx}_flag_pf'] = std['flag'].to_numpy()

            # Fill leading/trailing NaN lags so every chunk has a usable
            # final value for downstream alignment.
            summary[f'{pfx}_tlag_final_s'] = PwbBatchDetection.fill_tlag_gaps(
                summary[f'{pfx}_pwbopt_s_std'].to_numpy(),
                tlag_s_raw=tlag,
            )
            summary[f'{pfx}_tlag_final_pf_s'] = PwbBatchDetection.fill_tlag_gaps(
                summary[f'{pfx}_pwbopt_s_pf'].to_numpy(),
                tlag_s_raw=tlag,
            )
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m diive.flux.hires.detect_and_remove_tlag',
        description=(
            'Per-file PWB pipeline: read raw → rotate wind (in memory) → '
            'detect time lag → remove lag from the unrotated raw data → '
            'write. Alias: uv run diive-tlag-pwb-detect-remove'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- I/O ---
    p.add_argument('--input-dir', required=True,
                   help='Directory containing raw (unrotated) EC files.')
    p.add_argument('--output-dir', required=True,
                   help='Output directory for lag-corrected (still unrotated) files.')
    p.add_argument('--file-pattern', default='*.csv',
                   help='Glob pattern for input files.')
    # --- Wind columns ---
    p.add_argument('--col-u', required=True,
                   help='Column name for U (horizontal x).')
    p.add_argument('--col-v', required=True,
                   help='Column name for V (horizontal y).')
    p.add_argument('--col-w', required=True,
                   help='Column name for W (vertical).')
    p.add_argument('--col-tsonic', required=True,
                   help='Column name for sonic temperature.')
    # --- Scalars ---
    p.add_argument('--scalar', dest='scalars', action='append',
                   metavar='LABEL:column', required=True,
                   help='Gas label and column name in the raw file, e.g. '
                        '"CH4:CH4_DRY_[LGR-A]". Repeat for each gas.')
    # --- PWB parameters ---
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    p.add_argument('--lag-max', type=float, default=10.0,
                   help='CCF search half-width [s].')
    p.add_argument('--n-bootstrap', type=int, default=99,
                   help='Number of block-bootstrap replicates (paper: 99).')
    p.add_argument('--block-length', type=float, default=20.0,
                   help='Bootstrap block length [s] (paper: L = 20 s).')
    p.add_argument('--random-state', type=int, default=None,
                   help='Base seed for reproducible bootstrap. Each file, '
                        'chunk and gas gets a derived seed.')
    # --- Chunking (each input is split into fixed-length chunks) ---
    p.add_argument('--chunk-seconds', type=float, default=1800.0,
                   help='Chunk length [s] (default: 1800 = 30 min). One '
                        'output file per chunk per input file.')
    p.add_argument('--min-chunk-seconds', type=float, default=300.0,
                   help='Chunks shorter than this [s] are skipped '
                        '(default 300 s = 5 min). PWB needs enough records '
                        'to fit the block-bootstrap.')
    p.add_argument('--chunk-name-template',
                   default='{stem}_chunk{index:02d}{suffix}',
                   help='Template for chunk output filenames. Placeholders: '
                        '{stem}, {suffix}, {index}, {starttime} (last one '
                        'requires --start-time-regex).')
    p.add_argument('--start-time-regex', default=None,
                   help='Regex extracting the start timestamp of the input '
                        'file from its name. Concatenated capture groups (or '
                        'the whole match) parsed via --start-time-format.')
    p.add_argument('--start-time-format', default='%Y%m%d-%H%M',
                   help='strftime/strptime format for --start-time-regex '
                        'and for the {starttime} placeholder in the '
                        'chunk-name template.')
    # --- File format ---
    p.add_argument('--skiprows', type=int, default=0,
                   help='Lines BEFORE the column-name row '
                        '(raw CSV with header on line 1: 0; '
                        'EddyPro rotated: 9).')
    p.add_argument('--extra-rows', type=int, default=2,
                   help='Extra rows AFTER the header but BEFORE data '
                        '(e.g. units + instrument-source rows: 2 for the '
                        'typical raw EC CSV).')
    p.add_argument('--sep', default=',',
                   help=r"Field separator. Default ',' (CSV). Use '\s+' for "
                        r"whitespace or '\t' for TSV.")
    p.add_argument('--lineterm', default='\n',
                   help=r"Line terminator written between data rows. "
                        r"Default '\n'.")
    p.add_argument('--na-values', nargs='+',
                   default=list(_DEFAULT_NA_VALUES),
                   help='Strings to treat as NaN on read.')
    p.add_argument('--na-rep', default='-9999',
                   help='Value written for NaN on output.')
    # --- PWBOPT post-processing (paper Section 2.3 defaults) ---
    p.add_argument('--hdi-thresh', type=float, default=0.5,
                   help='S1 HDI threshold [s]: chunks with HDI range below '
                        'this are flagged S1_optimal (reliable).')
    p.add_argument('--dev-thresh', type=float, default=0.5,
                   help='S2 deviation threshold [s]: uncertain chunks are '
                        'accepted if within this distance of the preceding '
                        'optimal lag.')
    p.add_argument('--hdi-prefilter', type=float, default=1.0,
                   help='Pre-filter [s]: lags with HDI range above this are '
                        'set to NaN before PWBOPT (pre-filtered variant). '
                        'Set to 0 to disable.')
    # --- Plots ---
    p.add_argument('--save-plots', action='store_true',
                   help='Save the 3-panel PWB diagnostic figure per chunk '
                        'per scalar into <output-dir>/plots/.')
    # --- Execution ---
    p.add_argument('--n-workers', type=int, default=None,
                   help='Parallel worker processes. Default: os.cpu_count(). '
                        'Set to 1 for sequential in-process execution '
                        '(useful for debugging).')
    p.add_argument('--strict', action='store_true',
                   help='Re-raise exceptions on the first failure instead '
                        'of capturing them per file.')
    return p


def _cli_main():
    import sys
    from datetime import datetime, timezone
    from rich.console import Console as _Console
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    # record=True keeps every printed message in memory; we dump it to
    # <output-dir>/log.txt at the end (also on exception, via try/finally).
    console = _Console(log_path=False, record=True)

    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: --input-dir not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    scalars = {}
    for token in args.scalars:
        if ':' not in token:
            print(f'ERROR: --scalar must be LABEL:column, got {token!r}',
                  file=sys.stderr)
            sys.exit(1)
        label, col = token.split(':', 1)
        scalars[label] = col

    # Argparse delivers backslash-escapes literally; translate so the user
    # can type ``--sep "\t"`` or ``--lineterm "\r\n"`` and get the
    # intended characters.
    def _unescape(s: str) -> str:
        return (s.replace('\\t', '\t')
                 .replace('\\r', '\r')
                 .replace('\\n', '\n')
                 .replace('\\s+', _WHITESPACE_SEP))

    sep = _unescape(args.sep)
    lineterm = _unescape(args.lineterm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / 'log.txt'

    pipeline = PerFilePipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        col_u=args.col_u,
        col_v=args.col_v,
        col_w=args.col_w,
        col_tsonic=args.col_tsonic,
        scalars=scalars,
        hz=args.hz,
        lag_max_s=args.lag_max,
        n_bootstrap=args.n_bootstrap,
        block_length_s=args.block_length,
        chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        chunk_name_template=args.chunk_name_template,
        start_time_regex=args.start_time_regex,
        start_time_format=args.start_time_format,
        file_pattern=args.file_pattern,
        skiprows=args.skiprows,
        extra_rows=args.extra_rows,
        sep=sep,
        lineterm=lineterm,
        na_values=args.na_values,
        na_rep=args.na_rep,
        random_state=args.random_state,
        n_workers=args.n_workers,
        strict=args.strict,
        save_plots=args.save_plots,
        hdi_thresh=args.hdi_thresh,
        dev_thresh=args.dev_thresh,
        hdi_prefilter=args.hdi_prefilter,
    )

    files = sorted(input_dir.glob(args.file_pattern))
    total = len(files)
    if total == 0:
        print(f'ERROR: no files matched {args.file_pattern!r} in {input_dir}',
              file=sys.stderr)
        sys.exit(1)

    # Quick pre-scan of the first file so the progress bar's total is
    # measured in CHUNKS (the unit of actual work) rather than files.
    try:
        chunks_per_file = pipeline.estimate_chunks_per_file(files[0])
    except Exception as e:
        chunks_per_file = int(round(21600 / args.chunk_seconds))  # 6h fallback
        print(f'WARN: pre-scan failed ({e}); assuming {chunks_per_file} '
              f'chunks per file', file=sys.stderr)
    total_chunks_est = total * chunks_per_file

    started_at = datetime.now(timezone.utc).astimezone().isoformat(
        timespec='seconds')
    mode = 'sequential' if pipeline.n_workers == 1 else f'{pipeline.n_workers} workers'

    # Header: run metadata is recorded in the log along with the per-file
    # progress lines below, so the log.txt file is self-describing.
    console.print()
    console.print(f'[dim]started:[/dim] {started_at}')
    console.print(f'[dim]input :[/dim]  {input_dir}')
    console.print(f'[dim]output:[/dim]  {output_dir}')
    console.print(f'[dim]files :[/dim]  {total}  ([dim]pattern[/dim] {args.file_pattern!r})')
    console.print(f'[dim]mode  :[/dim]  {mode}')
    console.print(f'[dim]hz    :[/dim]  {args.hz}    '
                  f'[dim]lag-max[/dim] {args.lag_max} s    '
                  f'[dim]n-boot[/dim] {args.n_bootstrap}    '
                  f'[dim]block[/dim] {args.block_length} s')
    console.print(f'[dim]chunk :[/dim]  {args.chunk_seconds:g} s  '
                  f'[dim](min-chunk[/dim] {args.min_chunk_seconds:g} s[dim])[/dim]  '
                  f'[dim]≈{chunks_per_file}/file → {total_chunks_est} total[/dim]  '
                  f'[dim]template[/dim] {args.chunk_name_template!r}')
    if args.start_time_regex:
        console.print(f'[dim]time  :[/dim]  regex {args.start_time_regex!r}  '
                      f'[dim]format[/dim] {args.start_time_format!r}')
    console.print(f'[dim]wind  :[/dim]  '
                  f'U={args.col_u!r}  V={args.col_v!r}  W={args.col_w!r}  '
                  f'T_SONIC={args.col_tsonic!r}')
    for label, col in scalars.items():
        console.print(f'[dim]scalar:[/dim]  [bold]{label}[/bold]  <- {col!r}')
    console.print(f'[dim]PWBOPT:[/dim]  hdi-thresh {args.hdi_thresh} s    '
                  f'dev-thresh {args.dev_thresh} s    '
                  f'hdi-prefilter {args.hdi_prefilter} s')
    console.print(f'[dim]plots :[/dim]  '
                  f'{"yes (output_dir/plots/)" if args.save_plots else "no"}')
    console.print()

    msg = (f'PWB per-chunk pipeline  {total} files  '
           f'{mode}  -> {output_dir}')
    console.print(f'[bold]{msg}[/bold]\n')

    def _fmt(row, gas):
        pfx = gas.lower()
        v = row.get(f'{pfx}_tlag_s')
        h = row.get(f'{pfx}_hdi_range_s')
        if v is None or v != v:
            return f'[dim]{gas}=--[/dim]'
        hdi_color = ('green' if h == h and h < 0.5
                     else ('yellow' if h == h and h < 1.0 else 'red'))
        return (f'{gas}=[bold]{v:.2f}s[/bold] '
                f'HDI=[{hdi_color}]{h:.2f}[/{hdi_color}]')

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
    # Progress bar counts CHUNKS (the actual unit of work). Total is the
    # pre-scan estimate; the bar may finish slightly short if some files
    # have fewer chunks than the first (short trailing chunks are skipped
    # explicitly so still counted).
    task_id = prog.add_task(f'[cyan]{mode}[/cyan]', total=total_chunks_est)

    def _short_period(period: str) -> str:
        """Trim the chunk-period filename for compact display."""
        return Path(period).stem

    summary = None
    try:
        with prog:
            def _on_progress(done, total_chunks, row):
                # Fires once per chunk completion (or skip).
                period = row.get('period', '')
                period_short = _short_period(period)
                if row.get('status') == 'error':
                    parts = f'[red]ERR[/red] {row.get("error", "")[:60]}'
                elif row.get('status') == 'skipped:short':
                    parts = (f'[yellow]SKIP[/yellow] '
                             f'short chunk ({row.get("chunk_records", 0)} rows)')
                else:
                    parts = '  '.join(_fmt(row, g) for g in scalars)
                    theta = row.get('theta_deg', float('nan'))
                    phi = row.get('phi_deg', float('nan'))
                    if theta == theta and phi == phi:
                        parts += (f'  [dim]θ={theta:+.1f}° '
                                  f'φ={phi:+.1f}°[/dim]')
                console.log(f'[dim]{period_short}[/dim]  {parts}')
                prog.update(task_id, completed=done)

            def _on_active(active):
                # Fires every time the in-flight set changes. Show each
                # worker's current (file, chunk) compactly so the user can
                # see which chunks of which files are being processed in
                # parallel right now. ``active`` is a dict keyed by worker
                # pid; values have ``parent``, ``chunk_index``,
                # ``chunk_period``.
                if not active:
                    desc = f'[cyan]{mode}[/cyan]'
                else:
                    # Sort by pid for stable visual order.
                    parts = []
                    for pid in sorted(active.keys()):
                        info = active[pid]
                        parent_stem = Path(info['parent']).stem
                        # Keep it short — large filenames would overflow.
                        if len(parent_stem) > 24:
                            parent_stem = parent_stem[:11] + '…' + parent_stem[-12:]
                        parts.append(
                            f'[bold cyan]{parent_stem}[/bold cyan]'
                            f'[dim]·c{info["chunk_index"]:02d}[/dim]'
                        )
                    desc = (f'[cyan]{mode}[/cyan]  '
                            + '  '.join(parts))
                prog.update(task_id, description=desc)

            summary = pipeline.run(on_progress=_on_progress,
                                   on_active=_on_active)

        # Per-chunk outcome counts
        n_chunks = len(summary)
        n_ok = int((summary['status'] == 'ok').sum())
        n_err = int((summary['status'] == 'error').sum())
        n_skip = int((summary['status'] == 'skipped:short').sum())
        color = 'green' if n_err == 0 else 'yellow'
        msg_chunks = (f'[{color}]Done — {n_ok}/{n_chunks} chunks written'
                      f'{f", {n_skip} short" if n_skip else ""}'
                      f'{f", {n_err} errors" if n_err else ""}'
                      f' (across {total} files, estimate was '
                      f'{total_chunks_est} chunks).[/{color}]')
        console.print(f'\n{msg_chunks}')

        summary_csv = output_dir / 'detect_and_remove_tlag_summary.csv'
        summary.to_csv(summary_csv, index=False)
        console.print(f'[dim]Summary saved to:[/dim] [cyan]{summary_csv}[/cyan]')

        # Batch-level overview plots — one 5-panel figure per scalar plus
        # the cross-scalar PwboptLagPlot. Reuses the same routine that
        # ``diive-tlag-pwb-batch`` calls in its own _cli_main, so the
        # figures look identical (same flag colors, same axes, same KDE).
        if args.save_plots:
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            try:
                PwbBatchDetection.plot_summary(
                    results=summary,
                    scalars=scalars,
                    hdi_thresh=args.hdi_thresh,
                    hdi_prefilter=args.hdi_prefilter,
                    lag_max_s=args.lag_max,
                    output_dir=plots_dir,
                    showplot=False,
                )
                console.print(
                    f'[dim]Overview plots saved to:[/dim] '
                    f'[cyan]{plots_dir}[/cyan]  '
                    f'[dim](summary_<scalar>.png + summary_lag_comparison.png)[/dim]'
                )
            except Exception as e:
                # Never fail the run because of a plotting issue.
                console.print(
                    f'[yellow]WARN[/yellow] overview-plot generation failed: '
                    f'{type(e).__name__}: {e}'
                )

    except BaseException as e:
        # Capture into the log before re-raising. BaseException catches
        # KeyboardInterrupt + SystemExit too — important for crash diagnosis.
        console.print(f'\n[bold red]ABORT[/bold red] '
                      f'{type(e).__name__}: {e}')
        # Re-raise after the finally block has dumped the log.
        raise
    finally:
        finished_at = datetime.now(timezone.utc).astimezone().isoformat(
            timespec='seconds')
        console.print(f'[dim]finished:[/dim] {finished_at}')
        # save_text strips Rich markup, producing a clean plain-text log.
        try:
            console.save_text(str(log_path), clear=False)
        except Exception:
            # Never let log-saving fail the whole run.
            pass
        # Echo log location to stdout last so it survives even when Rich
        # output is redirected.
        print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    _cli_main()
