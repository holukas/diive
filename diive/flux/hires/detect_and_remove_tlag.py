"""
DETECT_AND_REMOVE_TLAG: per-30-min-chunk end-to-end PWB pipeline
=================================================================

Input raw EC files commonly cover multi-hour periods (e.g. 6 h), but EC
flux processing and the PWB time-lag detection both work on **30-minute
averaging intervals**. The wind-rotation angles drift over hours, and the
tube delay can drift too, so a single rotation / lag estimate across 6 h
is the wrong granularity.

This module therefore reads each long raw file once and processes it in
**fixed-length chunks** (default 30 minutes = ``hz * 1800`` rows), in two
phases.

**Wall-clock grid alignment.** 30-min flux files must sit on the wall-clock
grid (:00 / :30) so downstream software bins them by clock time. When a file
start time is available (via ``--start-time-regex`` / ``--start-time-format``),
chunk boundaries are snapped to that grid: a file starting off-grid (e.g.
10:10) yields a shorter **leading partial** chunk up to the first boundary
(10:10 -> 10:30, named by the file's real start), then full ``chunk_seconds``
chunks on the grid (10:30, 11:00, ...). A leading partial shorter than
``min_chunk_seconds`` is skipped. A file already on the grid, or one with no
parseable start time, is chunked the legacy way (fixed steps from its start).

**Phase 1 — detect (every chunk):**

1. Read the chunk's rows (unrotated; all original columns preserved).
2. Apply double rotation to the wind vector (u, v, w) of the chunk
   **in memory** using ``diive.flux.WindDoubleRotation``. The rotated W
   is the input to PWB — the rotated data are never written to disk.
3. Run ``PreWhiteningBootstrap`` (Vitale et al. 2024) on the rotated W +
   each scalar + sonic temperature, producing one ``tlag_s`` per gas
   per chunk. Nothing is written yet.

**Per-gas search windows.** Each gas can be given its own time-lag search
window via ``per_gas_lag`` (``{label: {lag_max_s, block_length_s, lws, uws}}``)
or, on the CLI, ``--scalar "H2O:h2o@lag=30;uws=25"`` plus global ``--lws`` /
``--uws``. A positive-only window (e.g. ``[0, 5]``) keeps only physical
tube-delay lags (closed-path delay > 0); a long-inlet gas such as H2O can use a
wider window than the dry gases in the same run -- needed because EddyPro
applies one lag setting to all gases downstream. ``window_to_lag_params`` maps a
window to per-gas params: ``lag_max_s = max(|lws|, |uws|)`` and
``block_length_s = max(20 s, 2*half)`` (the paper's 20 s block floor, growing
for a wide window). A gas without a window uses the global ``lag_max_s``
symmetrically.

**PWBOPT — choose the best lag:** with every chunk's raw detection in hand,
apply the PWBOPT decision rule (S1/S2/S3 carry-forward + gap-fill, paper
Section 2.3) across the full chunk sequence in temporal order. This is what
makes the split necessary — PWBOPT needs the *whole* sequence, so the lag
cannot be removed during phase 1.

**Phase 2 — remove (every successfully-detected chunk):**

4. Shift each scalar column in the **unrotated** chunk backward by
   ``round(tlag_best * hz)`` rows (``pd.Series.shift(periods=-n)``), where
   ``tlag_best`` is the PWBOPT-optimised lag (the column named by
   ``--lag-column-template``, default ``{prefix}_tlag_final_pf_s`` — the
   pre-filtered, gap-filled "best" lag, the same column
   ``diive-tlag-apply-batch`` removes). A wide-HDI chunk's *raw* mode lag
   can be spurious; PWBOPT replaces it with the neighbouring optimal lag.
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

Checkpoint snapshots are written after every chunk completes, so an
interrupted run leaves a partial result on disk — both inside the
step-1 detect folder (``--detect-subdir``, default ``1_lag_detection/``):
``detect_and_remove_tlag_checkpoint.csv`` for phase 1 (detect — the
expensive part) and ``detect_and_remove_tlag_remove_checkpoint.csv`` for
phase 2 (remove). When the run finishes cleanly the full results land in
``<--detect-subdir>/detect_and_remove_tlag_summary.csv``.

Every CLI run writes a plain-text ``log.txt`` to the output-directory root
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

Outputs in ``--output-dir`` (subfolders numbered by pipeline phase; the root
itself holds only those two folders plus ``log.txt``):

- ``<--data-subdir>/`` (default ``2_lag_removed/``) — step 2 (remove): the
  lag-corrected 30-min chunk files (``hz * 1800`` rows each by default), and
  **only** those. Kept separate from the CSV / log / plots so this folder can
  be handed straight to the next flux-processing step as its input directory.
- ``<--detect-subdir>/detect_and_remove_tlag_summary.csv`` (default
  ``1_lag_detection/``) with one row per chunk. Schema
  mirrors ``tlag_results.csv`` from ``diive-tlag-pwb-batch`` plus this
  pipeline's extras: parent filename, ``chunk_index``, chunk filename,
  rotation angles ``theta_deg`` / ``phi_deg``, and per gas
  ``{prefix}_tlag_s`` / ``hdi_lo_s`` / ``hdi_hi_s`` / ``hdi_range_s`` /
  ``is_reliable`` / ``tlag_pw_s`` / ``corr_pw`` / ``cov_pwb`` /
  ``ar_order`` / ``best_combination``, plus the PWBOPT post-processing
  columns ``pwbopt_s_std`` / ``flag_std`` / ``pwbopt_s_pf`` /
  ``flag_pf`` / ``tlag_final_s`` / ``tlag_final_pf_s``, and the applied-shift
  bookkeeping ``{prefix}_applied_records`` (records actually shifted = the
  PWBOPT lag from ``--lag-column-template``, **not** the raw ``tlag_s``) /
  ``{prefix}_status``.
- ``<--detect-subdir>/plots/`` (default ``1_lag_detection/plots/``, when
  ``--save-plots`` is set) — step 1 (detect) per-chunk diagnostics: one
  ``<chunk_stem>_<gas>.png`` per chunk per gas — the 3-panel PWB diagnostic
  (pre-whitened CCF + raw cross-covariance + bootstrap lag histogram with
  the 95% HDI shaded).
- ``<--detect-subdir>/plots_summary/`` (when ``--save-plots`` is set) — the
  batch-level overviews, kept separate from the per-chunk diagnostics:

  - One ``summary_<gas>.png`` per scalar — the 5-panel overview from
    ``PwbBatchDetection.plot_summary``: detected lags coloured by S1/S2/S3
    flag, gap-filled lags, 95% HDI bars with threshold lines, per-period
    flag bars (standard vs. pre-filtered PWBOPT side by side), and a
    histogram of detected lags.
  - ``summary_lag_comparison.png`` — the cross-scalar ``PwboptLagPlot``
    scatter + KDE comparing standard vs. pre-filtered PWBOPT for every gas.
- ``<--detect-subdir>/detect_and_remove_tlag_checkpoint.csv`` /
  ``<--detect-subdir>/detect_and_remove_tlag_remove_checkpoint.csv`` —
  per-phase snapshots of the rows accumulated so far; left intact after a
  clean run so they can be diffed against the final summary if useful.
- ``README.txt`` (at the output root) — auto-generated each run; describes the
  folder layout and points at the ``<--data-subdir>/`` folder as the next
  step's input.
- ``log.txt`` (at the output root) — plain-text record of the run: the static
  header, one line per completed chunk, and the final summary. The animated
  live display (spinners, bars, per-worker rows) is deliberately NOT written
  here, so the log stays clean and free of terminal control characters.

Public API
----------
This module exposes:

- ``PerFilePipeline`` — class wrapping the two-phase loop; ``.run()``
  processes all files and returns a per-(file, chunk) summary DataFrame.
- ``process_one_file`` — module-level function that runs the full
  read-once → detect-all-chunks → PWBOPT → remove-best-lag → write
  sequence on a single file. Useful from
  Python without the CLI.

Part of the diive library: https://github.com/holukas/diive
"""

import os
import queue as _queue_mod
import re
import warnings
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from multiprocessing import Manager
from pathlib import Path
from typing import Callable

# Suppress the runpy double-import warning that fires when ``diive.__init__``
# has already imported this module before ``-m`` re-executes it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas import DataFrame, Series  # noqa: E402

from diive.flux.hires.lag_pwb import (  # noqa: E402
    _DEFAULT_NA_VALUES,
    PreWhiteningBootstrap,
    PwbBatchDetection,
)
from diive.flux.hires.windrotation import WindDoubleRotation  # noqa: E402
from diive.core.utils.console import warn  # noqa: E402

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


def _detect_lineterm(path: Path, default: str = '\n') -> str:
    r"""Return the line terminator a text file uses (``'\r\n'`` or ``'\n'``).

    Raw EC files written by Windows loggers are almost always CRLF; Unix
    tooling writes LF. To keep the lag-corrected file a true drop-in
    replacement, the writer reproduces whichever the input used. Peeks the
    first 64 KiB in binary and reports CRLF when the first newline is
    preceded by a carriage return.
    """
    try:
        with open(path, 'rb') as fh:
            chunk = fh.read(65536)
    except Exception:
        return default
    i = chunk.find(b'\n')
    if i == -1:
        return default
    return '\r\n' if i > 0 and chunk[i - 1:i] == b'\r' else '\n'


def _resolve_lineterm(lineterm: str, input_path: Path) -> str:
    """Resolve the ``'auto'`` sentinel to the input file's line terminator."""
    return _detect_lineterm(Path(input_path)) if lineterm == 'auto' else lineterm


def _write_raw_file(
        output_path: Path,
        preserved_lines: list,
        df: DataFrame,
        sep: str,
        lineterm: str,
        na_rep: str,
) -> None:
    """Write the lag-corrected file: header lines verbatim, then the data.

    ``lineterm`` must already be resolved to a literal terminator (use
    ``_resolve_lineterm`` to expand the ``'auto'`` sentinel first). The
    preserved header lines — read in text mode, so always ``'\\n'``-ended —
    are re-terminated with ``lineterm`` so the whole file uses one
    consistent convention (no mixed CRLF/LF). pandas ``to_csv`` requires a
    literal separator; the whitespace sentinel is written as a single space.
    """
    out_sep = ' ' if sep == _WHITESPACE_SEP else sep
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as fh:
        for line in preserved_lines:
            fh.write(line.rstrip('\r\n') + lineterm)
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
            t_chunk = _chunk_start_time(t0, chunk_index, chunk_seconds)
            fields['starttime'] = t_chunk.strftime(start_time_format)

    try:
        return name_template.format(**fields), t_chunk
    except KeyError as e:
        raise ValueError(
            f"--chunk-name-template {name_template!r} uses placeholder {e}; "
            f"available: {sorted(fields.keys())}"
        )


def _parse_file_start_time(
        name: str,
        start_time_regex: str | None,
        start_time_format: str,
) -> 'datetime | None':
    """Extract a file's start timestamp from its name, or None.

    Same extraction rule as ``_chunk_filename``: search ``start_time_regex``
    in ``name``, concatenate capture groups (or take the whole match), and
    parse with ``start_time_format``. Returns None when the regex is absent,
    does not match, or the captured text does not parse — callers fall back to
    filename order in that case.
    """
    if not start_time_regex:
        return None
    m = re.search(start_time_regex, name)
    if m is None:
        return None
    s = (''.join(g for g in m.groups() if g is not None)
         if m.groups() else m.group(0))
    try:
        return datetime.strptime(s, start_time_format)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Wall-clock grid alignment
#
# 30-min flux files must sit on the wall-clock grid (:00 / :30) so downstream
# software bins them by clock time. A raw file that starts off-grid (e.g.
# 10:10) is therefore split so its FIRST chunk is a shorter partial running
# from the file start to the next grid boundary (10:10 -> 10:30), and every
# chunk after that is a full ``chunk_seconds`` window on the grid (10:30,
# 11:00, ...). Grid boundaries are integer multiples of ``chunk_seconds``
# since midnight. A file that already starts on the grid is chunked the legacy
# way (fixed steps from its start). Without a parseable file start time the
# grid is unknown, so the legacy fixed-offset chunking is used as a fallback.
# ---------------------------------------------------------------------------

def _grid_lead_seconds(file_start: 'datetime | None',
                       chunk_seconds: float) -> float:
    """Seconds from ``file_start`` to the next wall-clock grid boundary.

    Grid boundaries are integer multiples of ``chunk_seconds`` since midnight
    (:00 / :30 for the 1800 s default). Returns 0.0 when the file already
    starts on the grid, or when ``file_start`` is None.
    """
    if file_start is None:
        return 0.0
    secs = (file_start.hour * 3600 + file_start.minute * 60
            + file_start.second + file_start.microsecond / 1e6)
    return (chunk_seconds - (secs % chunk_seconds)) % chunk_seconds


def _chunk_start_time(file_start: 'datetime | None', chunk_index: int,
                      chunk_seconds: float) -> 'datetime | None':
    """Grid-aligned start time of chunk ``chunk_index``.

    Chunk 0 of an off-grid file keeps the file's real start time (the partial
    up to the first grid boundary); chunks 1..n start on the grid. A file that
    already starts on the grid behaves like fixed ``chunk_seconds`` steps.
    Returns None when ``file_start`` is None.
    """
    if file_start is None:
        return None
    lead = _grid_lead_seconds(file_start, chunk_seconds)
    if lead == 0:
        return file_start + timedelta(seconds=chunk_index * chunk_seconds)
    if chunk_index == 0:
        return file_start
    return file_start + timedelta(
        seconds=lead + (chunk_index - 1) * chunk_seconds)


def _chunk_row_slice(file_start: 'datetime | None', chunk_index: int,
                     chunk_seconds: float, hz: int) -> 'tuple[int, int]':
    """Data-row offset and requested length for a chunk (grid-aligned).

    Mirrors ``_chunk_start_time`` in row space: chunk 0 of an off-grid file is
    a short partial (``lead`` rows, up to the first grid boundary); later
    chunks are full ``chunk_seconds * hz`` windows. Returns
    ``(row_offset, n_rows)`` relative to the first data row; the reader caps
    the trailing chunk at EOF, so fewer rows may actually be read. With
    ``file_start = None`` this is the legacy fixed-offset chunking.
    """
    c = int(round(chunk_seconds * hz))
    if file_start is None:
        return chunk_index * c, c
    lead = int(round(_grid_lead_seconds(file_start, chunk_seconds) * hz))
    if lead == 0:
        return chunk_index * c, c
    if chunk_index == 0:
        return 0, lead
    return lead + (chunk_index - 1) * c, c


# ---------------------------------------------------------------------------
# PWBOPT lag selection — shared by the per-file path and PerFilePipeline
# ---------------------------------------------------------------------------

def _pwbopt_final_lags(
        rows: list,
        scalars: dict,
        hdi_thresh: float,
        dev_thresh: float,
        hdi_prefilter: float,
        lag_column_template: str,
) -> dict:
    """Pick the per-chunk PWBOPT lag to remove, as ``{(chunk_index, label): lag_s}``.

    Runs the same S1/S2/S3 carry-forward + gap-fill as
    ``PerFilePipeline._apply_pwbopt_postprocessing`` over the chunk
    detections (in temporal order) and returns the column selected by
    ``lag_column_template`` — by default ``{prefix}_tlag_final_pf_s`` (the
    pre-filtered, gap-filled "best" lag), the same column ``TlagApplier``
    removes. Detections for skipped/error chunks are NaN and treated as
    gaps, so PWBOPT carries forward the neighbouring optimal lag.
    """
    if not rows:
        return {}
    ordered = sorted(rows, key=lambda r: r.get('chunk_index', -1))
    cidx = [r.get('chunk_index') for r in ordered]
    out: dict = {}
    for label in scalars:
        pfx = label.lower()
        tlag = np.array([float(r.get(f'{pfx}_tlag_s', np.nan)) for r in ordered],
                        dtype=float)
        hdi = np.array([float(r.get(f'{pfx}_hdi_range_s', np.nan)) for r in ordered],
                       dtype=float)
        std = PwbBatchDetection.apply_pwbopt(tlag, hdi, hdi_thresh, dev_thresh)
        if hdi_prefilter and hdi_prefilter > 0:
            tlag_pf = PwbBatchDetection.apply_hdi_prefilter(tlag, hdi, hdi_prefilter)
            pf_pwbopt = PwbBatchDetection.apply_pwbopt(
                tlag_pf, hdi, hdi_thresh, dev_thresh)['pwbopt_s'].to_numpy()
        else:
            pf_pwbopt = std['pwbopt_s'].to_numpy()
        final_std = PwbBatchDetection.fill_tlag_gaps(
            std['pwbopt_s'].to_numpy(), tlag_s_raw=tlag)
        final_pf = PwbBatchDetection.fill_tlag_gaps(pf_pwbopt, tlag_s_raw=tlag)
        # Honour the requested column; default (and anything ending _pf_s) uses
        # the pre-filtered series, otherwise the standard PWBOPT series.
        col = lag_column_template.format(prefix=pfx)
        chosen = final_std if col.endswith('_tlag_final_s') else final_pf
        for k, ci in enumerate(cidx):
            out[(ci, label)] = float(chosen[k])
    return out


# ---------------------------------------------------------------------------
# Core per-file pipeline — Read once → detect every chunk → PWBOPT → remove
# the best lag from each chunk → write. Two-phase so the lag actually removed
# is the PWBOPT-optimised lag (needs the whole chunk sequence), not the raw
# per-chunk detection.
# ---------------------------------------------------------------------------

def _ensure_headless_backend() -> None:
    """Switch matplotlib to the non-interactive ``Agg`` backend in this process.

    PWB plots are only ever saved to disk, never shown, and worker processes
    (and headless servers) have no display. Forcing ``Agg`` here — at the entry
    of each plotting path rather than at module import — keeps ``import diive``
    from hijacking an interactive user's backend, while still letting every
    plotting path render without an X display. Idempotent: only switches when
    the current backend is not already ``Agg`` (so it never closes figures on a
    repeat call, and only ever switches once per process).
    """
    import matplotlib
    if matplotlib.get_backend().lower() != 'agg':
        matplotlib.use('Agg', force=True)


# ---------------------------------------------------------------------------
# Per-gas time-lag window resolution.
#
# By default every gas shares the global lag_max_s / block_length_s and the
# optional global lws/uws window. A long-inlet gas (e.g. H2O) often needs a
# wider or shifted search window than the dry gases (CH4/N2O) -- and because
# EddyPro applies one uniform lag setting to all gases, that per-gas alignment
# must be baked in here. Per-gas overrides are a dict {label -> {lag_max_s,
# block_length_s, lws, uws}}; unset keys fall back to the global value.
# ---------------------------------------------------------------------------

# Compact aliases accepted after '@' in a LABEL:column@... scalar token.
_GAS_SPEC_ALIASES = {
    'lag': 'lag_max_s', 'lag_max': 'lag_max_s', 'lagmax': 'lag_max_s',
    'block': 'block_length_s', 'block_length': 'block_length_s',
    'lws': 'lws', 'uws': 'uws',
}


def parse_scalar_spec(token: str) -> tuple[str, str, dict]:
    """Parse a ``LABEL:column`` scalar token with an optional per-gas window.

    Syntax::

        CH4:ch4                              # global lag settings
        H2O:h2o@lag=30                       # lag_max_s=30 for this gas
        H2O:h2o@lag=30;uws=25                # + asymmetric window upper bound
        H2O:h2o@lag=30;lws=0;uws=25;block=60 # all four, seconds

    The part after ``@`` is ``;``-separated ``key=value`` pairs. Keys: ``lag``
    / ``lag_max`` (lag_max_s), ``block`` (block_length_s), ``lws``, ``uws`` --
    all in seconds. Returns ``(label, column, overrides)`` with *overrides*
    keyed by the canonical ``PreWhiteningBootstrap`` argument names (empty when
    there is no ``@`` part). Raises ``ValueError`` on a malformed token.
    """
    main, _, spec = token.partition('@')
    if ':' not in main:
        raise ValueError(f"scalar must be LABEL:column, got {token!r}")
    label, col = main.split(':', 1)
    label, col = label.strip(), col.strip()
    if not label or not col:
        raise ValueError(f"scalar must be LABEL:column, got {token!r}")
    overrides: dict = {}
    for piece in spec.split(';'):
        piece = piece.strip()
        if not piece:
            continue
        if '=' not in piece:
            raise ValueError(
                f"per-gas option must be key=value, got {piece!r} in {token!r}")
        key, val = piece.split('=', 1)
        canon = _GAS_SPEC_ALIASES.get(key.strip().lower())
        if canon is None:
            raise ValueError(
                f"unknown per-gas option {key.strip()!r} in {token!r}; "
                f"allowed: lag, block, lws, uws")
        try:
            overrides[canon] = float(val.strip())
        except ValueError:
            raise ValueError(
                f"per-gas option {key.strip()!r} needs a number, got "
                f"{val.strip()!r} in {token!r}")
    return label, col, overrides


def _resolve_gas_lag(label, lag_max_s, block_length_s, lws, uws,
                     gas_lag_overrides):
    """Resolve ``(lag_max_s, block_length_s, lws, uws)`` for one gas.

    A gas listed in *gas_lag_overrides* uses its overrides, falling back to the
    global values for unset keys. Overriding ``lag_max_s`` but not
    ``block_length_s`` re-couples the block to ``2*lag_max_s`` (R's
    ``l = LAG.MAX*2``) rather than silently keeping the global block length.
    """
    o = (gas_lag_overrides or {}).get(label) or {}
    g_lag = o.get('lag_max_s', lag_max_s)
    if 'block_length_s' in o:
        g_block = o['block_length_s']
    elif 'lag_max_s' in o:
        g_block = 2.0 * g_lag
    else:
        g_block = block_length_s
    return g_lag, g_block, o.get('lws', lws), o.get('uws', uws)


def window_to_lag_params(lws: float, uws: float, min_block_s: float = 20.0) -> dict:
    """Convert an asymmetric search window ``[lws, uws]`` (seconds) to PWB params.

    The CCF computation half-width (``lag_max_s``) is the window's larger
    absolute bound, so the requested window always fits inside the computed
    range. The bootstrap block length follows R's ``l = 2*LAG.MAX`` coupling but
    is **floored at** ``min_block_s`` (default 20 s, the paper's value): a narrow
    window does not shrink the block below what is needed to preserve the
    autocorrelation structure, yet the block still grows past the floor for a
    wide window so it keeps containing a long lag (e.g. ``[0, 25]`` -> 50 s).
    Returns a ``per_gas_lag`` entry for one gas. A symmetric window ``[-x, x]``
    yields ``lag_max_s=x`` with ``lws=-x, uws=x``, which reduces to the plain
    symmetric search (the windowed argmax spans the whole CCF) -- i.e. setting a
    gas's window to ``[-lag_max, +lag_max]`` is identical to no override.
    """
    if uws <= lws:
        raise ValueError(
            f"search window upper bound ({uws}) must exceed lower bound ({lws}).")
    half = max(abs(lws), abs(uws))
    return {'lag_max_s': half, 'block_length_s': max(min_block_s, 2.0 * half),
            'lws': lws, 'uws': uws}


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
        lineterm: str = 'auto',
        na_values: list | None = None,
        na_rep: str = '-9999',
        random_state: int | None = None,
        strict: bool = False,
        save_plots: bool = False,
        plots_dir: Path | None = None,
        hdi_thresh: float = 0.5,
        dev_thresh: float = 0.5,
        hdi_prefilter: float = 1.0,
        lag_column_template: str = '{prefix}_tlag_final_pf_s',
        lws: float | None = None,
        uws: float | None = None,
        gas_lag_overrides: dict | None = None,
        progress_queue=None,
) -> list:
    """Run the two-phase PWB pipeline on one (possibly multi-hour) input file.

    The input is read once and split into fixed-length chunks of
    ``round(chunk_seconds * hz)`` rows. **Phase 1** detects the time lag for
    every chunk (Rotate in memory → PWB per scalar). **Phase 2** runs PWBOPT
    over the file's chunk detections (S1/S2/S3 carry-forward + gap-fill,
    paper Section 2.3) and shifts each scalar by the resulting *optimised*
    lag — the column named by ``lag_column_template`` (default
    ``{prefix}_tlag_final_pf_s``) — before writing one file per chunk. This
    is the same "best" lag ``TlagApplier`` removes; the raw per-chunk
    detection is reported in the summary but never applied.

    The last chunk may be shorter; if its length is below
    ``round(min_chunk_seconds * hz)`` it is skipped (PWB needs enough records
    for a meaningful bootstrap). Output filenames are composed via
    ``chunk_name_template`` (see the module docstring for placeholder
    semantics). The preserved header lines from the input are written
    verbatim at the top of every chunk file.

    ``scalars`` maps gas labels (e.g. ``'CH4'``) to column names in the
    raw file (e.g. ``'CH4_DRY_[LGR-A]'``).

    Returns a list of per-chunk summary dicts: ``period`` (output chunk
    filename), ``parent`` (input filename), ``chunk_index``,
    ``chunk_records``, ``status`` (``ok`` / ``error`` / ``skipped:short``),
    ``error``, ``theta_deg``, ``phi_deg``, and per-gas ``{prefix}_tlag_s``,
    ``{prefix}_hdi_range_s``, ``{prefix}_is_reliable``,
    ``{prefix}_applied_records`` (records shifted = the PWBOPT lag), and
    ``{prefix}_status``.

    On error (when ``strict=False``), one error row is returned for the
    file as a whole (status=``error``, ``error`` field carries the
    exception class + message).

    When ``progress_queue`` is provided, the function emits one
    ``{'event': 'start'}`` / ``{'event': 'done'}`` pair per chunk during
    phase 1 (detection, the expensive part); phase 2 runs silently.
    """
    _ensure_headless_backend()

    if na_values is None:
        na_values = list(_DEFAULT_NA_VALUES)

    parent_name = Path(input_path).name
    chunk_records = int(round(chunk_seconds * hz))
    min_chunk_records = int(round(min_chunk_seconds * hz))

    def _empty_chunk_row(chunk_index: int, chunk_period: str,
                         n_rows: int, status: str, err: str = '',
                         t_chunk: 'datetime | None' = None) -> dict:
        """Build a row with all per-gas fields populated as NaN/default."""
        return _empty_detect_row(
            scalars, chunk_period, parent_name,
            t_chunk.isoformat() if t_chunk is not None else '',
            chunk_index, n_rows, status, err,
        )

    rows: list = []
    # ok chunks carry forward to phase 2: (chunk_row, i0, i1, chunk_period)
    detect_ok: list = []
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
        # Grid-align chunks to the wall-clock :00/:30 boundaries: an off-grid
        # file's first chunk is a partial up to the first boundary, then full
        # chunks follow; a file already on the grid is chunked the legacy way.
        file_start = _parse_file_start_time(
            parent_name, start_time_regex, start_time_format)
        lead_records = int(round(
            _grid_lead_seconds(file_start, chunk_seconds) * hz))
        if lead_records > 0:
            n_chunks = 1 + max(
                0, (n_total - lead_records + chunk_records - 1) // chunk_records)
        else:
            n_chunks = (n_total + chunk_records - 1) // chunk_records
        n_chunks = max(1, n_chunks)

        # ================= PHASE 1: detect every chunk ===================
        for ci in range(n_chunks):
            row_offset, n_rows = _chunk_row_slice(
                file_start, ci, chunk_seconds, hz)
            i0 = row_offset
            i1 = min(i0 + n_rows, n_total)
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
                    g_lag, g_block, g_lws, g_uws = _resolve_gas_lag(
                        label, lag_max_s, block_length_s, lws, uws,
                        gas_lag_overrides)
                    pwb = PreWhiteningBootstrap(
                        df=pwb_df,
                        var_w='W_rot',
                        var_scalar=label,
                        var_tsonic='T_SONIC',
                        hz=hz,
                        lag_max_s=g_lag,
                        n_bootstrap=n_bootstrap,
                        block_length_s=g_block,
                        segment_name=chunk_period,
                        random_state=seed,
                        lws=g_lws,
                        uws=g_uws,
                    )
                    pwb.run()
                    res = pwb.results

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

                detect_ok.append((chunk_row, i0, i1, chunk_period))

            except Exception as ce:
                if strict:
                    raise
                chunk_row['status'] = 'error'
                chunk_row['error'] = f'{type(ce).__name__}: {ce}'

            rows.append(chunk_row)
            _emit('done', chunk_index=ci, chunk_period=chunk_period,
                  row=chunk_row)

        # ============ PWBOPT: best lag per (chunk, gas) ==================
        final_lags = _pwbopt_final_lags(
            rows, scalars, hdi_thresh, dev_thresh, hdi_prefilter,
            lag_column_template,
        )

        # ============ PHASE 2: remove the best lag + write ===============
        for chunk_row, i0, i1, chunk_period in detect_ok:
            ci = chunk_row['chunk_index']
            try:
                df_out = df_raw.iloc[i0:i1].copy()
                for label, col_name in scalars.items():
                    pfx = label.lower()
                    lag_s = final_lags.get((ci, label), np.nan)
                    if not np.isfinite(lag_s):
                        chunk_row[f'{pfx}_status'] = 'skipped:lag_nan'
                        continue
                    n_records = int(round(lag_s * hz))
                    df_out[col_name] = df_out[col_name].shift(periods=-n_records)
                    chunk_row[f'{pfx}_applied_records'] = n_records
                    chunk_row[f'{pfx}_status'] = 'ok'

                _write_raw_file(
                    output_dir / chunk_period, preserved_lines, df_out,
                    sep=sep, lineterm=_resolve_lineterm(lineterm, input_path),
                    na_rep=na_rep,
                )
            except Exception as ce:
                if strict:
                    raise
                chunk_row['status'] = 'error'
                chunk_row['error'] = f'{type(ce).__name__}: {ce}'

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
# Per-chunk processors for the TWO-PHASE parallel pipeline.
#
# Phase 1 (``detect_one_chunk``): read the chunk slice, rotate the wind in
# memory, run PWB per scalar, return the detection row. No data file is
# written — the lag cannot be removed yet, because the lag that *should* be
# removed is the PWBOPT-optimised lag, and PWBOPT needs the full temporal
# sequence of per-chunk detections (S1/S2/S3 carry-forward) which only exists
# once every chunk has been detected.
#
# Phase 2 (``remove_one_chunk``): re-read the chunk slice, shift each scalar
# by the PWBOPT lag chosen across all chunks, write the lag-corrected file.
#
# Both read only their own slice via ``pd.read_csv(skiprows, nrows)`` and are
# pickle-safe (module-level) for ``ProcessPoolExecutor``.
# ---------------------------------------------------------------------------

def _empty_detect_row(scalars: dict, period: str, parent: str,
                      timestamp_iso: str, chunk_index: int, n_rows: int,
                      status: str, err: str = '') -> dict:
    """Detection-row skeleton; per-gas fields default to NaN/pending.

    Schema mirrors ``tlag_results.csv`` from ``diive-tlag-pwb-batch`` so the
    summary CSV stays a drop-in equivalent. ``{pfx}_applied_records`` /
    ``{pfx}_status`` are placeholders filled in during phase 2 (remove).
    """
    r: dict = {
        'period': period,
        'parent': parent,
        'timestamp': timestamp_iso,
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


def detect_one_chunk(
        input_path: Path,
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
        na_values: list,
        random_state: int | None,
        strict: bool,
        save_plots: bool,
        plots_dir: Path | None,
        lws: float | None = None,
        uws: float | None = None,
        gas_lag_overrides: dict | None = None,
        progress_queue=None,
) -> dict:
    """Phase 1: detect the per-gas time lag for one chunk (writes no data).

    Reads only this chunk's data slice, applies double rotation in memory,
    and runs ``PreWhiteningBootstrap`` on the rotated W vs each scalar. The
    lag is *not* removed here — removal happens in ``remove_one_chunk`` using
    the PWBOPT-optimised lag chosen across all chunks. Pickle-safe.

    Returns one detection-row dict (one element of the per-file summary).
    """
    _ensure_headless_backend()

    parent_name = Path(input_path).name
    worker_pid = os.getpid()
    # Fallback label used only if filename templating fails below, so the
    # failure becomes a visible 'error' row instead of a vanished chunk
    # (a swallowed worker exception in the parallel path).
    chunk_period = f'{Path(input_path).stem}#chunk{chunk_index}'
    timestamp_iso = ''

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

    n_preserved = skiprows + 1 + extra_rows
    # Grid-aligned slice: chunk 0 of an off-grid file is a shorter partial up
    # to the first :00/:30 boundary, then full chunks follow.
    _file_start = _parse_file_start_time(
        Path(input_path).name, start_time_regex, start_time_format)
    _row_offset, _chunk_nrows = _chunk_row_slice(
        _file_start, chunk_index, chunk_seconds, hz)
    skiprows_total = n_preserved + _row_offset

    try:
        # Resolve the output filename first; a templating/regex error here is
        # a config mistake — surface it as an error row, not a lost chunk.
        chunk_period, t_chunk = _chunk_filename(
            input_path=Path(input_path),
            chunk_index=chunk_index,
            chunk_seconds=chunk_seconds,
            name_template=chunk_name_template,
            start_time_regex=start_time_regex,
            start_time_format=start_time_format,
        )
        timestamp_iso = t_chunk.isoformat() if t_chunk is not None else ''

        # ---- Read preserved header (small) -------------------------------
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fh:
            preserved_lines = [next(fh) for _ in range(n_preserved)]

        header_line = preserved_lines[skiprows].rstrip('\n').rstrip('\r')
        if sep == _WHITESPACE_SEP:
            header_cols = header_line.split()
        else:
            header_cols = [c.strip() for c in header_line.split(sep)]

        # ---- Read only this chunk's data slice ---------------------------
        try:
            df_chunk = pd.read_csv(
                input_path,
                skiprows=skiprows_total,
                nrows=_chunk_nrows,
                header=None,
                sep=sep,
                na_values=na_values,
                low_memory=False,
                engine='python' if sep == _WHITESPACE_SEP else 'c',
            )
        except pd.errors.EmptyDataError:
            # Chunk starts at/after EOF: a phantom chunk from the padded
            # row-count estimate. Fall through to the empty:eof path below.
            df_chunk = None
        chunk_n = 0 if df_chunk is None else len(df_chunk)

        # Phantom past-EOF chunk (nothing read at all): the padded estimate
        # dispatched one chunk too many for this file. Return a sentinel the
        # collector discards — never a user-visible row, never an output file.
        if chunk_n == 0:
            row = _empty_detect_row(
                scalars, chunk_period, parent_name, timestamp_iso,
                chunk_index, 0, 'empty:eof', '',
            )
            _emit('done', chunk_index=chunk_index,
                  chunk_period=chunk_period, row=row)
            return row

        # Short trailing chunk (real data, but fewer rows than the bootstrap
        # needs): emit a single 'done' with skipped status so the bar advances.
        if chunk_n < min_chunk_records:
            row = _empty_detect_row(
                scalars, chunk_period, parent_name, timestamp_iso,
                chunk_index, chunk_n, 'skipped:short',
                f'chunk has {chunk_n} rows < min {min_chunk_records}',
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

        row = _empty_detect_row(scalars, chunk_period, parent_name,
                                timestamp_iso, chunk_index, chunk_n, 'ok')

        # ---- Rotate ------------------------------------------------------
        wr = WindDoubleRotation(
            u=df_chunk[col_u].astype(float),
            v=df_chunk[col_v].astype(float),
            w=df_chunk[col_w].astype(float),
        )
        row['theta_deg'] = float(np.degrees(wr.theta))
        row['phi_deg'] = float(np.degrees(wr.phi))

        # ---- Detect (PWB on rotated W per scalar) -----------------------
        for gi, (label, col_name) in enumerate(scalars.items()):
            pfx = label.lower()
            pwb_df = pd.DataFrame({
                'W_rot': wr.w2.reset_index(drop=True),
                label: df_chunk[col_name].astype(float).reset_index(drop=True),
                'T_SONIC': df_chunk[col_tsonic].astype(float).reset_index(drop=True),
            })
            seed = (None if random_state is None
                    else int(random_state) + gi)
            g_lag, g_block, g_lws, g_uws = _resolve_gas_lag(
                label, lag_max_s, block_length_s, lws, uws, gas_lag_overrides)
            pwb = PreWhiteningBootstrap(
                df=pwb_df,
                var_w='W_rot',
                var_scalar=label,
                var_tsonic='T_SONIC',
                hz=hz,
                lag_max_s=g_lag,
                n_bootstrap=n_bootstrap,
                block_length_s=g_block,
                segment_name=chunk_period,
                random_state=seed,
                lws=g_lws,
                uws=g_uws,
            )
            pwb.run()
            res = pwb.results

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

            # Save the 3-panel PWB diagnostic plot (one per chunk per gas).
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

    except Exception as e:
        if strict:
            raise
        row = _empty_detect_row(scalars, chunk_period, parent_name,
                                timestamp_iso, chunk_index, 0, 'error',
                                f'{type(e).__name__}: {e}')
        _emit('done', chunk_index=chunk_index, chunk_period=chunk_period,
              row=row)
        return row

    _emit('done', chunk_index=chunk_index, chunk_period=chunk_period, row=row)
    return row


def remove_one_chunk(
        input_path: Path,
        output_dir: Path,
        chunk_index: int,
        chunk_records: int,
        chunk_period: str,
        scalars: dict,
        lags: dict,
        hz: int,
        skiprows: int,
        extra_rows: int,
        sep: str,
        lineterm: str,
        na_values: list,
        na_rep: str,
        strict: bool,
        chunk_seconds: float,
        start_time_regex: str | None,
        start_time_format: str,
        progress_queue=None,
) -> dict:
    """Phase 2: remove the PWBOPT lag from one chunk and write the file.

    Re-reads this chunk's slice, shifts each scalar column backward by
    ``round(lags[label] * hz)`` rows, and writes the lag-corrected
    (unrotated) chunk with the original header preserved. ``lags`` carries
    the per-gas PWBOPT-optimised lag chosen across all chunks in phase 1.
    Pickle-safe.

    Returns a dict carrying ``parent`` / ``chunk_index`` / ``period`` plus
    per-gas ``{pfx}_applied_records`` / ``{pfx}_status`` and the write
    outcome (``write_status`` / ``write_error``) for merging back into the
    detection summary.
    """
    parent_name = Path(input_path).name
    worker_pid = os.getpid()
    output_dir = Path(output_dir)
    n_preserved = skiprows + 1 + extra_rows
    # Grid-aligned slice, identical to the one phase 1 detected on.
    _file_start = _parse_file_start_time(
        parent_name, start_time_regex, start_time_format)
    _row_offset, _chunk_nrows = _chunk_row_slice(
        _file_start, chunk_index, chunk_seconds, hz)
    skiprows_total = n_preserved + _row_offset

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

    out: dict = {
        'parent': parent_name,
        'chunk_index': chunk_index,
        'period': chunk_period,
        'write_status': 'ok',
        'write_error': '',
    }
    for label in scalars:
        pfx = label.lower()
        out[f'{pfx}_applied_records'] = np.nan
        out[f'{pfx}_status'] = 'pending'

    _emit('start', chunk_index=chunk_index, chunk_period=chunk_period)
    try:
        # ---- Read preserved header + this chunk's slice ------------------
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fh:
            preserved_lines = [next(fh) for _ in range(n_preserved)]
        header_line = preserved_lines[skiprows].rstrip('\n').rstrip('\r')
        if sep == _WHITESPACE_SEP:
            header_cols = header_line.split()
        else:
            header_cols = [c.strip() for c in header_line.split(sep)]

        df_chunk = pd.read_csv(
            input_path,
            skiprows=skiprows_total,
            nrows=_chunk_nrows,
            header=None,
            sep=sep,
            na_values=na_values,
            low_memory=False,
            engine='python' if sep == _WHITESPACE_SEP else 'c',
        )
        if df_chunk.shape[1] != len(header_cols):
            raise ValueError(
                f"Header has {len(header_cols)} columns but chunk data "
                f"has {df_chunk.shape[1]} for {parent_name} chunk "
                f"{chunk_index}. Check --skiprows / --extra-rows / --sep."
            )
        df_chunk.columns = header_cols

        # ---- Shift scalars by the PWBOPT lag -----------------------------
        for label, col_name in scalars.items():
            pfx = label.lower()
            lag_s = lags.get(label, np.nan)
            if not np.isfinite(lag_s):
                out[f'{pfx}_status'] = 'skipped:lag_nan'
                continue
            n_records = int(round(lag_s * hz))
            df_chunk[col_name] = df_chunk[col_name].shift(periods=-n_records)
            out[f'{pfx}_applied_records'] = n_records
            out[f'{pfx}_status'] = 'ok'

        # ---- Write -------------------------------------------------------
        _write_raw_file(
            output_dir / chunk_period, preserved_lines, df_chunk,
            sep=sep, lineterm=_resolve_lineterm(lineterm, input_path),
            na_rep=na_rep,
        )

    except Exception as e:
        if strict:
            raise
        out['write_status'] = 'error'
        out['write_error'] = f'{type(e).__name__}: {e}'

    _emit('done', chunk_index=chunk_index, chunk_period=chunk_period, row=out)
    return out


# ---------------------------------------------------------------------------
# Module-level worker wrappers — single-arg so ProcessPoolExecutor can pickle
# the call easily. Each worker process re-imports this module via Windows
# spawn; module-level functions are required.
# ---------------------------------------------------------------------------

def _detect_worker(kwargs: dict, progress_queue=None) -> dict:
    """Run ``detect_one_chunk`` (phase 1) in a child process."""
    return detect_one_chunk(progress_queue=progress_queue, **kwargs)


def _remove_worker(kwargs: dict, progress_queue=None) -> dict:
    """Run ``remove_one_chunk`` (phase 2) in a child process."""
    return remove_one_chunk(progress_queue=progress_queue, **kwargs)


def _count_data_rows(path: Path, header_lines: int) -> int:
    """Exact newline count of a text file, minus the header rows.

    Reads the whole file in 1 MiB binary blocks — O(filesize). Used by the
    preflight Check and as the fallback for the fast sampling estimator
    ``_estimate_data_rows`` (tiny files / pathologically long lines). The
    per-file chunk planning in ``PerFilePipeline`` uses the sampling estimator
    instead, so a folder of large files is not read end-to-end just to count
    rows.
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


def _estimate_data_rows(path: Path, header_lines: int,
                        sample_bytes: int = 1 << 18) -> int:
    """Estimate the data-row count from file size — without reading it all.

    Reads only the first ``sample_bytes`` (256 KiB) to derive an average
    bytes-per-line, then scales by the file size from ``stat`` — so the cost
    is O(sample) per file instead of O(filesize). For EC raw data the row
    width is near-uniform (a few thousand rows sampled give a sub-percent
    estimate of the mean), so this is accurate to well under one 30-min chunk.

    The estimate is only ever used to decide how many per-chunk tasks to
    dispatch; callers pad it (``PerFilePipeline._padded_chunk_count``) so a
    small under-estimate can never drop a file's trailing chunk, and phantom
    chunks that land past EOF are discarded by the detect worker.

    Falls back to the exact ``_count_data_rows`` when the file fits inside the
    sample (then the count is exact and just as cheap) or when the sample does
    not even reach past the header rows.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return _count_data_rows(path, header_lines)
    with open(path, 'rb') as fh:
        sample = fh.read(sample_bytes)
    if not sample:
        return 0
    # Whole file fit in the sample -> the newline count is exact (and cheap).
    if len(sample) < sample_bytes:
        n_lines = sample.count(b'\n')
        if sample[-1:] not in (b'\n', b''):
            n_lines += 1  # last line without a trailing newline still counts
        return max(0, n_lines - header_lines)
    s_lines = sample.count(b'\n')
    if s_lines <= header_lines:
        # Header alone exceeds the sample (very long lines) -> count exactly.
        return _count_data_rows(path, header_lines)
    bytes_per_line = len(sample) / s_lines
    est_total_lines = int(size / bytes_per_line)
    return max(0, est_total_lines - header_lines)


# ---------------------------------------------------------------------------
# Summary data dictionary — describes every column of the summary CSV
# ---------------------------------------------------------------------------

# General (non per-gas) columns, in CSV order. {N}/{M} are filled per run.
_SUMMARY_GENERAL_COLS = [
    ('period',
     "Output filename of this chunk's lag-corrected file (written into the "
     "data subfolder). One row = one output file; this is the key that joins "
     "the summary to the files in the data folder."),
    ('parent',
     "Name of the raw input file this chunk was read from."),
    ('timestamp',
     "ISO-8601 start time of the chunk, derived from the parent filename via "
     "--start-time-regex / --start-time-format plus the chunk offset. Empty "
     "when no start-time regex was given."),
    ('chunk_index',
     "0-based index of the chunk within its parent file."),
    ('chunk_records',
     "Number of data rows in the chunk. A full chunk has chunk_seconds * hz = "
     "{N} rows; the trailing chunk of a file may be shorter."),
    ('status',
     "Detect-phase outcome of the chunk: 'ok' (lag detected and file written), "
     "'skipped:short' (fewer than min_chunk_seconds * hz = {M} rows, too few "
     "for the bootstrap, not written), 'skipped:duplicate' (its output name "
     "collided with a lower-index chunk and was skipped to avoid overwriting "
     "it), 'error' (processing failed - see the error column)."),
    ('error',
     "Exception type and message when status = 'error', or the reason text "
     "when status = 'skipped:duplicate'. Empty otherwise."),
    ('theta_deg',
     "First double-rotation angle (yaw): horizontal wind rotated into the mean "
     "wind direction, in degrees."),
    ('phi_deg',
     "Second double-rotation angle (pitch): vertical tilt correction so mean w "
     "= 0, in degrees."),
]

# Per-gas columns, in CSV order. Each applies to every scalar; {gas} is the
# lowercased scalar label (e.g. CH4 -> ch4_tlag_s). {hz} is filled per run.
_SUMMARY_PERGAS_COLS = [
    ('{gas}_tlag_s',
     "RAW detected time lag (s): the bootstrap mode lag between vertical wind W "
     "and this scalar for this chunk. The unprocessed PWB estimate - not "
     "necessarily the lag that was applied (see {gas}_tlag_final_pf_s)."),
    ('{gas}_hdi_lo_s',
     "Lower bound of the 95% highest-density interval (HDI) of the bootstrap "
     "lag distribution (s)."),
    ('{gas}_hdi_hi_s',
     "Upper bound of the 95% HDI (s)."),
    ('{gas}_hdi_range_s',
     "Width of the 95% HDI (hi - lo, s). The reliability metric: a narrow HDI "
     "means a well-determined lag, a wide HDI means a noisy / undetermined one."),
    ('{gas}_is_reliable',
     "True if the chunk passed the S1 reliability test (HDI range below "
     "--hdi-thresh)."),
    ('{gas}_tlag_pw_s',
     "Lag (s) of the peak of the pre-whitened cross-correlation function - the "
     "single, non-bootstrap pre-whitened estimate."),
    ('{gas}_corr_pw',
     "Correlation at that pre-whitened CCF peak (-1..1)."),
    ('{gas}_cov_pwb',
     "Raw (un-whitened) W-scalar cross-covariance evaluated at the selected "
     "lag."),
    ('{gas}_ar_order',
     "Order of the autoregressive model used to pre-whiten this chunk."),
    ('{gas}_best_combination',
     "Which of the four pre-whitening combinations won the selection: 'cw' "
     "(scalar x W, scalar AR), 'wc' (scalar x W, W AR), 'ct' (scalar x "
     "T_SONIC, scalar AR), 'tc' (scalar x T_SONIC, T_SONIC AR). Strong fluxes "
     "usually win on cw/wc; weak trace gases may fall back to the T_SONIC "
     "pair."),
    ('{gas}_applied_records',
     "Number of records the scalar column was shifted by in phase 2 = "
     "round(applied_lag_s * hz). The applied lag in seconds = this / hz "
     "(hz = {hz}). NaN if the chunk was not written."),
    ('{gas}_status',
     "Alignment (phase-2) outcome for this gas: 'ok' (shifted and written), "
     "'skipped:lag_nan' (no finite PWBOPT lag to apply), 'pending' (the chunk "
     "never reached phase 2, e.g. a detect error or short chunk)."),
    ('{gas}_pwbopt_s_std',
     "PWBOPT-selected lag (s), STANDARD rule: the S1/S2/S3 carry-forward "
     "applied directly to the raw mode lag across the full time-ordered "
     "sequence of chunks."),
    ('{gas}_flag_std',
     "PWBOPT decision flag for the standard series: 'S1_optimal' (reliable "
     "detection), 'S2_optimal' (uncertain but within --dev-thresh of the "
     "preceding optimal lag, so carried forward), 'S3_unreliable' (neither; "
     "filled from neighbours in the final column)."),
    ('{gas}_pwbopt_s_pf',
     "PWBOPT-selected lag (s), PRE-FILTERED rule: detections with an HDI range "
     "wider than --hdi-prefilter are dropped before the S1/S2/S3 logic runs."),
    ('{gas}_flag_pf',
     "PWBOPT decision flag for the pre-filtered series (same S1/S2/S3 meaning "
     "as flag_std)."),
    ('{gas}_tlag_final_s',
     "Final lag (s) from the STANDARD PWBOPT series after gap-filling any "
     "remaining leading/trailing NaNs."),
    ('{gas}_tlag_final_pf_s',
     "Final lag (s) from the PRE-FILTERED PWBOPT series after gap-filling. This "
     "is the pre-filtered, gap-filled best lag - removed by default."),
]


def _summary_columns_doc(scalars: dict, lag_column_template: str,
                         hz: int, chunk_seconds: float,
                         min_chunk_seconds: float) -> str:
    """Build a Markdown data dictionary for the summary CSV columns.

    Describes every column of ``detect_and_remove_tlag_summary.csv``: the
    general per-chunk columns plus the per-gas block (one set per scalar). The
    column actually removed in phase 2 (``lag_column_template``) is flagged so
    a reader knows which lag was applied. Returned as a Markdown string; the
    pipeline writes it next to the CSV as
    ``detect_and_remove_tlag_summary_columns.md``.
    """
    n_full = int(round(chunk_seconds * hz))
    n_min = int(round(min_chunk_seconds * hz))
    applied_suffix = lag_column_template.format(prefix='').lstrip('_')

    def _row(col: str, desc: str) -> str:
        return f"| `{col}` | {desc} |"

    lines: list[str] = []
    lines.append("# `detect_and_remove_tlag_summary.csv` - column reference")
    lines.append("")
    lines.append(
        "One row per processed chunk from the PWB time-lag detect + remove "
        "pipeline (`diive-tlag-pwb-detect-remove`). Each raw input file is "
        "split into fixed-length chunks; the scalar-vs-wind tube-delay lag is "
        "detected per chunk, optimised across all chunks (PWBOPT), then removed "
        "from the written output.")
    lines.append("")
    lines.append(
        f"Run settings reflected here: hz = {hz}, chunk = {chunk_seconds:g} s "
        f"({n_full} rows/full chunk), min chunk = {min_chunk_seconds:g} s "
        f"({n_min} rows), applied lag column = "
        f"`{lag_column_template}`.")
    lines.append("")

    lines.append("## General columns")
    lines.append("")
    lines.append("| Column | Description |")
    lines.append("|---|---|")
    for col, desc in _SUMMARY_GENERAL_COLS:
        lines.append(_row(col, desc.format(N=n_full, M=n_min)))
    lines.append("")

    lines.append("## Per-gas columns")
    lines.append("")
    prefixes = ', '.join(f"{label} -> `{label.lower()}_*`" for label in scalars)
    lines.append(
        f"The block below repeats once per scalar, with `{{gas}}` replaced by "
        f"the lowercased label. In this run: {prefixes}.")
    lines.append("")
    lines.append("| Column | Description |")
    lines.append("|---|---|")
    for col, desc in _SUMMARY_PERGAS_COLS:
        desc = desc.format(gas='{gas}', hz=hz)
        if col == f'{{gas}}_{applied_suffix}':
            desc += (" **This is the lag column removed in this run "
                     "(see applied lag column above).**")
        lines.append(_row(col, desc))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `tlag_s` is the raw per-chunk detection; the lag actually removed "
        "is the PWBOPT-optimised, gap-filled column named above. A wide-HDI "
        "chunk's raw lag can be spurious, so PWBOPT replaces it with the "
        "neighbouring reliable lag.")
    lines.append(
        "- Only rows with `status = ok` produce an output file. "
        "`skipped:short`, `skipped:duplicate` and `error` rows are reported "
        "for traceability but write nothing.")
    lines.append(
        "- The applied lag in seconds for a gas equals "
        "`{gas}_applied_records / hz`.")
    lines.append("")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# PerFilePipeline — class wrapper for the loop
# ---------------------------------------------------------------------------

class PerFilePipeline:
    """Two-phase per-chunk PWB detect + remove pipeline.

    Splits every file matched by ``file_pattern`` in ``input_dir`` into
    fixed-length chunks, then runs two parallel phases:

    1. **Detect** — rotate each chunk's wind in memory and run PWB per
       scalar. No data is written; this only collects per-chunk lags.
    2. **Remove** — after PWBOPT (S1/S2/S3 carry-forward + gap-fill) has
       chosen the *best* lag per chunk across the whole sequence, shift each
       scalar by that lag and write the lag-corrected chunk file.

    Removing the PWBOPT lag (not the raw per-chunk detection) is the whole
    point of the split: a wide-HDI chunk's raw mode lag can be spurious, so
    the lag actually applied is the same pre-filtered, gap-filled column
    ``TlagApplier`` uses (``lag_column_template``, default
    ``{prefix}_tlag_final_pf_s``).

    The unit of parallel work is one chunk: chunks across all files are
    dispatched into a ``ProcessPoolExecutor`` (``n_workers > 1``) so a single
    multi-chunk file still saturates every core.

    See the module docstring for the full workflow rationale and the CLI
    flag reference. After ``.run()``, ``.summary`` returns a DataFrame with
    one row per (file, chunk): rotation angles, detected lag per gas, the
    PWBOPT columns, applied records (= the removed PWBOPT lag), reliability
    flags, and error messages.
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
            lineterm: str = 'auto',
            na_values: list | None = None,
            na_rep: str = '-9999',
            random_state: int | None = None,
            n_workers: int | None = None,
            strict: bool = False,
            save_plots: bool = False,
            detect_subdir: str = '1_lag_detection',
            data_subdir: str = '2_lag_removed',
            plots_subdir: str = 'plots',
            hdi_thresh: float = 0.5,
            dev_thresh: float = 0.5,
            hdi_prefilter: float = 1.0,
            lag_column_template: str = '{prefix}_tlag_final_pf_s',
            lws: float | None = None,
            uws: float | None = None,
            per_gas_lag: dict | None = None,
    ):
        """Set up the per-file detect-and-remove pipeline. See the class docstring."""
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
        # Output folders numbered by pipeline phase so the layout reads in
        # order. Step 1 (detect): diagnostics under detect_subdir. Step 2
        # (remove): lag-corrected chunk files under data_subdir — kept on
        # their own so that folder is a clean input directory for the next
        # flux-processing step (conceptually step 3), uncluttered by the
        # summary CSV / log / checkpoints / plots.
        self.detect_subdir = detect_subdir
        self.data_subdir = data_subdir
        self.plots_subdir = plots_subdir
        # PWBOPT post-processing thresholds (paper Section 2.3 defaults)
        self.hdi_thresh = hdi_thresh
        self.dev_thresh = dev_thresh
        self.hdi_prefilter = hdi_prefilter
        # Which PWBOPT lag column to actually remove in phase 2. Default is
        # the pre-filtered, gap-filled "best" lag — the same column
        # TlagApplier removes.
        self.lag_column_template = lag_column_template
        # Global optional asymmetric search window (seconds), applied to every
        # gas unless a per-gas entry overrides it.
        self.lws = lws
        self.uws = uws
        # Per-gas time-lag window overrides: {label -> {lag_max_s,
        # block_length_s, lws, uws}}. A long-inlet gas (e.g. H2O) can take a
        # wider/shifted window than the dry gases in the same run -- which the
        # uniform EddyPro lag setting downstream cannot do.
        per_gas_lag = per_gas_lag or {}
        unknown_gas = set(per_gas_lag) - set(scalars)
        if unknown_gas:
            raise ValueError(
                f"per_gas_lag references unknown gas label(s) {sorted(unknown_gas)}; "
                f"known scalars are {sorted(scalars)}.")
        allowed = {'lag_max_s', 'block_length_s', 'lws', 'uws'}
        for lbl, ov in per_gas_lag.items():
            bad = set(ov) - allowed
            if bad:
                raise ValueError(
                    f"per_gas_lag[{lbl!r}] has unknown key(s) {sorted(bad)}; "
                    f"allowed: {sorted(allowed)}.")
        self.per_gas_lag = per_gas_lag

        self._summary: DataFrame | None = None
        # Result-file paths, populated by run() -> _write_summary_and_plots.
        self._summary_csv_path: Path | None = None
        self._summary_plots_dir: Path | None = None
        self._cancelled: bool = False  # set by run() when cancel_event fired

    @property
    def summary(self) -> DataFrame:
        """Per-file summary (available after ``run()``)."""
        if self._summary is None:
            raise RuntimeError('Call run() first.')
        return self._summary

    @property
    def cancelled(self) -> bool:
        """True if the last ``run()`` was stopped via its cancel_event."""
        return self._cancelled

    @property
    def summary_csv_path(self) -> Path | None:
        """Path to the summary CSV written by ``run()`` (or None)."""
        return self._summary_csv_path

    @property
    def summary_plots_dir(self) -> Path | None:
        """Path to the overview-plots folder written by ``run()`` (or None)."""
        return self._summary_plots_dir

    def estimate_chunks_per_file(self, sample_file: Path) -> int:
        """Estimate one file's chunk count (fast: reads only a 256 KiB sample).

        Uses ``_estimate_data_rows`` (file size + sampled bytes-per-line)
        rather than a full newline count, so scanning a folder of large files
        is near-instant. This returns the *honest* estimate; the detect
        dispatch pads it via ``_padded_chunk_count`` so an under-estimate can
        never silently drop a file's trailing chunk.

        Accounts for wall-clock grid alignment: an off-grid file is split into
        a leading partial plus full chunks, which is one chunk more than a
        naive ``ceil(rows / chunk)`` for the same row count.
        """
        header_lines = self.skiprows + 1 + self.extra_rows
        n_rows = _estimate_data_rows(sample_file, header_lines)
        chunk_records = int(round(self.chunk_seconds * self.hz))
        file_start = _parse_file_start_time(
            sample_file.name, self.start_time_regex, self.start_time_format)
        lead = int(round(
            _grid_lead_seconds(file_start, self.chunk_seconds) * self.hz))
        if lead > 0:
            # Leading partial + full chunks for the remaining rows.
            return max(1, 1 + max(
                0, (n_rows - lead + chunk_records - 1) // chunk_records))
        return max(1, (n_rows + chunk_records - 1) // chunk_records)

    @staticmethod
    def _padded_chunk_count(est_chunks: int) -> int:
        """Chunk count to dispatch for a file given its honest estimate.

        Dispatches a few extra chunks beyond the estimate so sampling error in
        ``_estimate_data_rows`` cannot drop trailing data: +1 covers the common
        case, +1 per 50 chunks covers proportionally larger error on very long
        files. Any dispatched chunk that lands past EOF is read as empty and
        discarded by the detect worker (status ``'empty:eof'``), so this
        over-count is invisible — no phantom rows, no extra output files.
        """
        return est_chunks + 1 + est_chunks // 50

    def run(self,
            on_progress: Callable | None = None,
            on_active: Callable | None = None,
            cancel_event=None,
            on_status: Callable | None = None) -> DataFrame:
        """Detect (phase 1) then remove the PWBOPT lag (phase 2) for every chunk.

        Phase 1 runs ``detect_one_chunk`` on every chunk of every file (no
        data written). PWBOPT then chooses the best lag per chunk across the
        full temporal sequence. Phase 2 runs ``remove_one_chunk`` on each
        successfully-detected chunk, shifting scalars by that PWBOPT lag and
        writing the output file. Each phase is an independent
        ``ProcessPoolExecutor`` fan-out (``n_workers > 1``) so all workers
        stay busy even for a single multi-chunk file; ``n_workers == 1`` runs
        in-process for cleaner stack traces.

        Args:
            on_progress: Optional callback
                ``f(chunks_completed, total, chunk_row, phase)`` fired once
                per chunk completion. ``phase`` is ``'detect'`` or
                ``'remove'``.
            on_active: Optional callback ``f(active_dict, phase)`` where
                ``active_dict`` maps worker pid -> ``{'parent', 'chunk_index',
                'chunk_period'}``. Fired whenever the in-flight set changes.
            cancel_event: Optional ``threading.Event``. When set, the current
                phase stops dispatching new chunks (running ones finish), the
                remove phase is skipped if cancelled during detect, and the
                partial summary collected so far is returned. ``run()`` adds a
                ``cancelled`` attribute (bool) to the returned DataFrame.
            on_status: Optional callback ``f(message: str)`` for coarse-grained
                progress that has no per-chunk granularity — chiefly the
                up-front file scan (counting each file's rows to plan chunks),
                which reads every input file and would otherwise leave a
                TUI/CLI silent before the first chunk callback fires.

        Returns:
            DataFrame with one row per (file, chunk), sorted by input-file
            order then chunk_index.
        """
        self._cancelled = False

        def _status(msg: str) -> None:
            """Emit a coarse status line (file scan / planning) if requested."""
            if on_status is None:
                return
            try:
                on_status(msg)
            except Exception:
                # A flaky status sink must never abort the run.
                pass

        files = sorted(self.input_dir.glob(self.file_pattern))
        if not files:
            raise FileNotFoundError(
                f'No files matched {self.file_pattern!r} in {self.input_dir}'
            )

        # PWBOPT carry-forward (S2/S3) is temporal: an unreliable chunk inherits
        # the *preceding* optimal lag, so the file sequence must be in time
        # order. When --start-time-regex is given, sort files by their parsed
        # start time; otherwise the order is the filename sort above, which is
        # only chronological if the names happen to sort that way -- warn so a
        # wrong order cannot silently feed the wrong neighbour's lag into the
        # carry-forward. (Single file: cross-file order is irrelevant.)
        if len(files) > 1:
            if self.start_time_regex:
                parsed = [
                    (_parse_file_start_time(
                        f.name, self.start_time_regex, self.start_time_format), f)
                    for f in files
                ]
                n_unparsed = sum(1 for t, _ in parsed if t is None)
                if n_unparsed:
                    warn(f"{n_unparsed}/{len(files)} input files did not yield a "
                         f"parseable start time with --start-time-regex "
                         f"{self.start_time_regex!r} / --start-time-format "
                         f"{self.start_time_format!r}; those files keep filename "
                         f"order for the PWBOPT carry-forward.")
                files = [f for _, f in sorted(
                    parsed,
                    key=lambda tf: (tf[0] is None, tf[0] or datetime.max, tf[1].name))]
            else:
                warn("PWBOPT carry-forward order is the filename sort of "
                     f"{self.file_pattern!r} in {self.input_dir} (no "
                     "--start-time-regex given). If these filenames do not sort "
                     "chronologically, pass --start-time-regex / "
                     "--start-time-format so the S2/S3 carry-forward uses true "
                     "time order.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Step 2 (remove) output: lag-corrected chunk files in their own
        # subfolder (clean input for the next flux-processing step).
        data_dir = self.output_dir / self.data_subdir
        data_dir.mkdir(parents=True, exist_ok=True)
        # Step 1 (detect) folder holds the checkpoints, the summary CSV
        # (written by the CLI), and the diagnostic plots — so the output root
        # stays just the two numbered folders + log.txt.
        detect_dir = self.output_dir / self.detect_subdir
        detect_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = (detect_dir / self.plots_subdir) if self.save_plots else None
        if plots_dir is not None:
            plots_dir.mkdir(parents=True, exist_ok=True)

        # Write the run settings and the folder README up front — the moment
        # the folders exist, before any processing — so a crashed, cancelled
        # or still-running job already documents what it was asked to do and
        # what each folder holds.
        self._write_run_settings()
        self._write_readme()

        total_files = len(files)
        parent_to_idx = {f.name: i for i, f in enumerate(files)}
        file_by_name = {f.name: f for f in files}
        chunk_records = int(round(self.chunk_seconds * self.hz))
        min_chunk_records = int(round(self.min_chunk_seconds * self.hz))

        # Per-file chunk count (NOT files[0] applied to all): a file longer
        # than the first no longer loses its trailing chunks, and shorter
        # files no longer spawn phantom over-read tasks. Each estimate reads
        # the whole file to count rows, so for many/large inputs this loop is
        # a slow, callback-free stretch before phase 1 — report scan progress
        # via on_status so a TUI/CLI is not left silent. Throttle to ~20
        # updates regardless of file count.
        _status(f'scanning {total_files} file(s) to plan chunks…')
        file_chunk_counts: dict = {}
        scan_step = max(1, total_files // 20)
        for fi, f in enumerate(files, start=1):
            file_chunk_counts[f] = self.estimate_chunks_per_file(f)
            if fi == total_files or fi % scan_step == 0:
                _status(f'scanning input files… {fi}/{total_files}')
        total_chunks = sum(file_chunk_counts.values())
        _status(f'planned {total_chunks} chunk(s) across {total_files} '
                f'file(s); starting detection…')

        # Surface a template / start-time-regex / format error up front (cheap),
        # instead of as one error row per chunk from every worker.
        _chunk_filename(
            input_path=files[0], chunk_index=0, chunk_seconds=self.chunk_seconds,
            name_template=self.chunk_name_template,
            start_time_regex=self.start_time_regex,
            start_time_format=self.start_time_format,
        )

        # Validate the chunk-name template can produce distinct names —
        # *structurally*, not by enumerating estimated chunk names. Enumerating
        # would (a) make a one-chunk over-estimate raise a false collision and
        # (b) reject legitimate start-time overlaps between contiguous files
        # (file N's trailing chunk shares a wall-clock slot with file N+1's
        # leading chunk). Those real, harmless overlaps are instead resolved at
        # write time (phase 2) by keeping one chunk and skipping the other.
        tmpl = self.chunk_name_template
        uses_index = '{index' in tmpl
        uses_start = '{starttime' in tmpl
        uses_stem = '{stem' in tmpl
        if total_files > 1 and not uses_stem and not uses_start:
            raise ValueError(
                f"--chunk-name-template {tmpl!r} has neither {{stem}} nor "
                f"{{starttime}}, so chunks from different input files collapse "
                f"to the same output name. Add {{stem}} or {{starttime}}."
            )
        if max(file_chunk_counts.values()) > 1 and not uses_index \
                and not uses_start:
            raise ValueError(
                f"--chunk-name-template {tmpl!r} has neither {{index}} nor "
                f"{{starttime}}, so the multiple chunks within a file collapse "
                f"to the same output name. Add {{index}} (e.g. {{index:02d}}) "
                f"or {{starttime}}."
            )

        # ============== PHASE 1: detect (parallel, no writes) ============
        detect_kwargs_list: list[dict] = []
        for i, f in enumerate(files):
            file_seed_base = (None if self.random_state is None
                              else int(self.random_state) + i * 10_000)
            # Dispatch a few extra chunks beyond the (sampled) estimate so a
            # small under-count never drops a file's trailing chunk; the extras
            # that land past EOF are recognised and dropped by the worker.
            for ci in range(self._padded_chunk_count(file_chunk_counts[f])):
                chunk_seed = (None if file_seed_base is None
                              else file_seed_base + ci * 100)
                detect_kwargs_list.append(dict(
                    input_path=f,
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
                    lws=self.lws,
                    uws=self.uws,
                    gas_lag_overrides=self.per_gas_lag,
                    chunk_seconds=self.chunk_seconds,
                    chunk_name_template=self.chunk_name_template,
                    start_time_regex=self.start_time_regex,
                    start_time_format=self.start_time_format,
                    skiprows=self.skiprows,
                    extra_rows=self.extra_rows,
                    sep=self.sep,
                    na_values=self.na_values,
                    random_state=chunk_seed,
                    strict=self.strict,
                    save_plots=self.save_plots,
                    plots_dir=plots_dir,
                ))

        detect_rows = self._run_pool(
            detect_kwargs_list, _detect_worker,
            total=total_chunks, phase='detect',
            checkpoint_path=detect_dir / 'detect_and_remove_tlag_checkpoint.csv',
            parent_to_idx=parent_to_idx, total_files=total_files,
            on_progress=on_progress, on_active=on_active,
            cancel_event=cancel_event,
        )

        # Drop phantom past-EOF chunks dispatched by the padded estimate: they
        # carry no data and must not appear in the summary, the counts, or the
        # phase-2 write list.
        detect_rows = [r for r in detect_rows
                       if r.get('status') != 'empty:eof']
        detect_rows.sort(key=lambda r: (
            parent_to_idx.get(r.get('parent', ''), total_files),
            r.get('chunk_index', -1),
        ))
        summary = pd.DataFrame(detect_rows)
        # PWBOPT across ALL chunks in temporal order -> best lag per chunk.
        summary = self._apply_pwbopt_postprocessing(summary)

        # If the user cancelled during detect, skip phase 2 entirely and
        # return the partial detection summary (no files aligned).
        if cancel_event is not None and cancel_event.is_set():
            self._cancelled = True
            self._summary = summary
            self._write_summary_and_plots(summary)
            return summary

        # ---- Resolve output-name collisions before writing --------------
        # Two real 'ok' chunks can map to the same output filename — typically
        # a start-time overlap where a file's trailing chunk lands on the same
        # 30-min slot as the next (contiguous) file's leading chunk. Rather
        # than abort the whole batch, keep one chunk and skip the other so it
        # is never overwritten. Prefer the LOWER chunk_index: a chunk 0 is the
        # next file's full leading period, which should win over the previous
        # file's overflow trailing chunk.
        if 'period' in summary.columns:
            kept_for_name: dict = {}  # output name -> summary index of winner
            for idx in summary.index:
                if summary.at[idx, 'status'] != 'ok':
                    continue
                name = summary.at[idx, 'period']
                ci = int(summary.at[idx, 'chunk_index'])
                win = kept_for_name.get(name)
                if win is None:
                    kept_for_name[name] = idx
                    continue
                if ci < int(summary.at[win, 'chunk_index']):
                    loser_idx, keep_idx = win, idx
                    kept_for_name[name] = idx
                else:
                    loser_idx, keep_idx = idx, win
                msg = (
                    f"duplicate output {name!r}: kept "
                    f"{summary.at[keep_idx, 'parent']} chunk "
                    f"{int(summary.at[keep_idx, 'chunk_index'])}, skipped "
                    f"{summary.at[loser_idx, 'parent']} chunk "
                    f"{int(summary.at[loser_idx, 'chunk_index'])} to avoid "
                    f"overwrite (output-name overlap between files, usually "
                    f"contiguous start times)")
                summary.at[loser_idx, 'status'] = 'skipped:duplicate'
                summary.at[loser_idx, 'error'] = msg
                warn(msg)
                _status(msg)

        # ============== PHASE 2: remove the PWBOPT lag + write ===========
        remove_kwargs_list: list[dict] = []
        for _, r in summary.iterrows():
            if r.get('status') != 'ok':
                continue  # error / skipped / duplicate chunks produce no file
            f = file_by_name.get(r['parent'])
            if f is None:
                continue
            lags = {}
            for label in self.scalars:
                col = self.lag_column_template.format(prefix=label.lower())
                lags[label] = (float(r[col]) if col in summary.columns
                               else np.nan)
            remove_kwargs_list.append(dict(
                input_path=f,
                output_dir=data_dir,
                chunk_index=int(r['chunk_index']),
                chunk_records=chunk_records,
                chunk_period=r['period'],
                scalars=self.scalars,
                lags=lags,
                hz=self.hz,
                skiprows=self.skiprows,
                extra_rows=self.extra_rows,
                sep=self.sep,
                lineterm=self.lineterm,
                na_values=self.na_values,
                na_rep=self.na_rep,
                strict=self.strict,
                chunk_seconds=self.chunk_seconds,
                start_time_regex=self.start_time_regex,
                start_time_format=self.start_time_format,
            ))

        remove_rows = self._run_pool(
            remove_kwargs_list, _remove_worker,
            total=len(remove_kwargs_list), phase='remove',
            checkpoint_path=detect_dir / 'detect_and_remove_tlag_remove_checkpoint.csv',
            parent_to_idx=parent_to_idx, total_files=total_files,
            on_progress=on_progress, on_active=on_active,
            cancel_event=cancel_event,
        )

        # ---- Merge phase-2 outcomes back into the detection summary -----
        remove_by_key = {(rr['parent'], int(rr['chunk_index'])): rr
                         for rr in remove_rows}
        for idx in summary.index:
            key = (summary.at[idx, 'parent'], int(summary.at[idx, 'chunk_index']))
            rr = remove_by_key.get(key)
            if rr is None:
                continue
            for label in self.scalars:
                pfx = label.lower()
                summary.at[idx, f'{pfx}_applied_records'] = rr.get(
                    f'{pfx}_applied_records', np.nan)
                summary.at[idx, f'{pfx}_status'] = rr.get(f'{pfx}_status', 'pending')
            if rr.get('write_status') == 'error':
                summary.at[idx, 'status'] = 'error'
                summary.at[idx, 'error'] = rr.get('write_error', '')

        # Cancelled during the remove phase: some chunks aligned, some not.
        self._cancelled = bool(cancel_event is not None and cancel_event.is_set())
        self._summary = summary
        self._write_summary_and_plots(summary)
        return self._summary

    def _write_summary_and_plots(self, summary: DataFrame) -> None:
        """Write the summary CSV and (when enabled) the batch overview plots.

        Called automatically at the end of ``run()`` so every caller — the
        CLI, the TUI, or a bare ``PerFilePipeline(...).run()`` from Python —
        produces the same set of result files. The resulting paths are
        stored on ``self.summary_csv_path`` / ``self.summary_plots_dir``
        for callers that want to log them.

        - ``<detect_subdir>/detect_and_remove_tlag_summary.csv``
        - ``<detect_subdir>/plots_summary/`` (only when ``save_plots``):
          the per-scalar 5-panel overviews + the cross-scalar comparison,
          via ``PwbBatchDetection.plot_summary``.
        """
        self._summary_csv_path = None
        self._summary_plots_dir = None
        if summary is None or summary.empty:
            return

        detect_dir = self.output_dir / self.detect_subdir
        detect_dir.mkdir(parents=True, exist_ok=True)

        summary_csv = detect_dir / 'detect_and_remove_tlag_summary.csv'
        try:
            summary.to_csv(summary_csv, index=False)
            self._summary_csv_path = summary_csv
        except PermissionError:
            # File locked (e.g. open in Excel) — leave the checkpoint as the
            # best available snapshot rather than aborting the run.
            warn(f'could not write summary CSV (locked?): {summary_csv}')

        # Companion data dictionary: a Markdown file describing every column of
        # the summary CSV (general + per-gas), so the CSV is self-documenting.
        doc_md = detect_dir / 'detect_and_remove_tlag_summary_columns.md'
        try:
            doc_md.write_text(
                _summary_columns_doc(
                    self.scalars, self.lag_column_template, self.hz,
                    self.chunk_seconds, self.min_chunk_seconds),
                encoding='utf-8')
        except Exception:
            # A failed data dictionary must never abort the run.
            pass

        if self.save_plots:
            _ensure_headless_backend()
            summary_plots_dir = detect_dir / 'plots_summary'
            summary_plots_dir.mkdir(parents=True, exist_ok=True)
            try:
                PwbBatchDetection.plot_summary(
                    results=summary,
                    scalars=self.scalars,
                    hdi_thresh=self.hdi_thresh,
                    hdi_prefilter=self.hdi_prefilter,
                    lag_max_s=self.lag_max_s,
                    output_dir=summary_plots_dir,
                    showplot=False,
                )
                self._summary_plots_dir = summary_plots_dir
            except Exception as e:
                # Plotting must never fail the run.
                warn(f'overview-plot generation failed: '
                     f'{type(e).__name__}: {e}')

    def _write_run_settings(self) -> None:
        """Write the run's settings to ``run_settings.txt`` in the output root.

        Called at the start of ``run()`` (the moment the output folders exist,
        before any processing) so an interrupted or crashed run still records
        exactly what it was asked to do. Plain text: each setting is its value
        followed by a one-line explanation, so the file is self-documenting.
        """

        def _block(title: str, rows: list) -> list:
            """Render a section: 'key : value' + an indented description each."""
            out = [title, '-' * len(title)]
            for key, val, desc in rows:
                out.append(f'{key:<20} : {val}')
                out.append(f'                       {desc}')
            out.append('')
            return out

        scalars_str = (', '.join(f'{k} <- {v}' for k, v in self.scalars.items())
                       if self.scalars else '(none)')
        lines = [
            'diive PWB time-lag detect + remove -- run settings',
            '==================================================',
            '',
            'Settings used for this run (written at run start, before any',
            'processing, so a crashed/cancelled run is still documented). Each',
            'entry shows the value used and a short explanation. The output is',
            f'produced by diive-tlag-pwb-detect-remove into: {self.output_dir}',
            '',
        ]
        lines += _block('paths', [
            ('input_dir', self.input_dir,
             'Folder scanned for raw EC files (read-only; never modified).'),
            ('output_dir', self.output_dir,
             'Folder all results are written to (this folder).'),
            ('file_pattern', self.file_pattern,
             'Glob selecting which files in input_dir are processed.'),
        ])
        lines += _block('columns', [
            ('wind U / V / W', f'{self.col_u} / {self.col_v} / {self.col_w}',
             'Sonic wind component columns (used for the double rotation).'),
            ('sonic temperature', self.col_tsonic,
             'Sonic temperature column (T_SONIC fallback signal for PWB).'),
            ('scalars', scalars_str,
             'Gas columns to time-lag-correct, as LABEL <- column.'),
        ])
        lines += _block('PWB detection', [
            ('hz', self.hz,
             'Sampling frequency in Hz (records per second).'),
            ('lag_max_s', self.lag_max_s,
             'Half-width of the lag search window (s); the CCF is computed '
             'over +/-lag_max_s and the peak searched within it.'),
            ('n_bootstrap', self.n_bootstrap,
             'Block-bootstrap replicates per chunk (PWB; paper uses 99).'),
            ('block_length_s', self.block_length_s,
             'Bootstrap block length (s); long enough to contain the lag '
             '(paper floor 20 s).'),
            ('lws / uws (window)', f'{self.lws} / {self.uws}',
             'Optional asymmetric search window [lower, upper] (s); '
             'None = the full symmetric +/-lag_max_s.'),
            ('per_gas_lag', self.per_gas_lag or "(none)",
             'Per-gas window overrides {label: {lag_max_s, block_length_s, '
             'lws, uws}}; a gas without an entry uses the global values.'),
            ('random_state', self.random_state,
             'Seed for a reproducible bootstrap; None = non-deterministic.'),
        ])
        lines += _block('chunking', [
            ('chunk_seconds', self.chunk_seconds,
             'Length of each averaging period a file is split into (s).'),
            ('min_chunk_seconds', self.min_chunk_seconds,
             'Chunks shorter than this (lead/trailing remnants) are skipped.'),
            ('chunk_name_template', self.chunk_name_template,
             'Pattern for naming each output chunk file.'),
            ('start_time_regex', self.start_time_regex,
             'Regex capturing the start timestamp from the input filename '
             '(needed for the {starttime} field).'),
            ('start_time_format', self.start_time_format,
             'strptime format parsing the captured timestamp text.'),
        ])
        lines += _block('PWBOPT (best-lag selection)', [
            ('hdi_thresh', self.hdi_thresh,
             'S1 threshold (s): a chunk whose 95% HDI range is below this is '
             'trusted directly as reliable.'),
            ('dev_thresh', self.dev_thresh,
             'S2 threshold (s): an uncertain chunk is still accepted if its '
             'lag is within this of the previous reliable lag.'),
            ('hdi_prefilter', self.hdi_prefilter,
             'Drop detections with an HDI range above this (s) before '
             'PWBOPT runs; 0 = off.'),
            ('lag_column_template', self.lag_column_template,
             'Which PWBOPT lag column is removed in phase 2 (default the '
             'pre-filtered, gap-filled best lag).'),
        ])
        lines += _block('file format', [
            ('skiprows', self.skiprows,
             'Metadata lines before the header row.'),
            ('extra_rows', self.extra_rows,
             'Units/source rows after the header row.'),
            ('sep', repr(self.sep),
             'Column separator.'),
            ('lineterm', repr(self.lineterm),
             "Output line terminator; 'auto' reproduces the input file's."),
            ('na_values', self.na_values,
             'Strings read as missing values.'),
            ('na_rep', self.na_rep,
             'String written for missing values in the output.'),
        ])
        lines += _block('execution / output layout', [
            ('n_workers', self.n_workers,
             'Parallel worker processes; None = all CPU cores.'),
            ('strict', self.strict,
             'True = abort the whole run on the first chunk error; '
             'False = skip the chunk and log it.'),
            ('save_plots', self.save_plots,
             'Whether per-chunk + summary diagnostic figures are written.'),
            ('detect_subdir', self.detect_subdir,
             'Subfolder for step-1 detection diagnostics + results.'),
            ('data_subdir', self.data_subdir,
             'Subfolder for the step-2 lag-corrected chunks (deliverable).'),
        ])
        try:
            (self.output_dir / 'run_settings.txt').write_text(
                '\n'.join(lines), encoding='utf-8')
        except Exception:
            # A settings dump must never abort the run.
            pass

    def _write_readme(self) -> None:
        """Write a README.txt to ``output_dir`` describing the folder layout.

        Written up front in ``run()`` (the moment the folders exist) so the
        output folder is self-documenting even for an interrupted run — which
        subfolder holds what, and which one to feed to the next flux step.
        """
        scalars_str = ', '.join(f'{k} <- {v}' for k, v in self.scalars.items())
        plots_note = ('present' if self.save_plots
                      else 'not generated (pass --save-plots to enable)')
        # The TUI-loadable settings YAML is written by the CLI/TUI wrapper
        # before the run, not by PerFilePipeline itself — so only mention it
        # when it is actually present (a direct Python-API run has none).
        yaml_note = ''
        if (self.output_dir / 'detect_remove_tui_settings.yaml').exists():
            yaml_note = (
                "detect_remove_tui_settings.yaml\n"
                "                    Machine-readable copy of the settings,\n"
                "                    reloadable in the detect+remove TUI (Load\n"
                "                    button or drag-drop) to inspect or rerun.\n")
        text = (
            "diive PWB time-lag detection + removal -- output folder\n"
            "=======================================================\n"
            "\n"
            "Produced by diive-tlag-pwb-detect-remove (PerFilePipeline): a\n"
            "two-phase, per-chunk pre-whitening-bootstrap (PWB) time-lag\n"
            "pipeline. Each raw high-resolution EC file is split into\n"
            "fixed-length chunks; the scalar-vs-wind tube-delay lag is detected\n"
            "per chunk (phase 1), optimised across all chunks (PWBOPT), then\n"
            "removed (phase 2).\n"
            "\n"
            "Folder layout\n"
            "-------------\n"
            f"{self.detect_subdir}/   STEP 1 -- DETECT (diagnostics + results)\n"
            f"    {self.plots_subdir}/             per-chunk PWB diagnostic figures "
            f"({plots_note})\n"
            f"    plots_summary/      batch-level overview figures ({plots_note})\n"
            "    detect_and_remove_tlag_summary.csv           one row per chunk:\n"
            "                        detected lag, HDI, reliability, PWBOPT\n"
            "                        columns, applied records (written by the CLI)\n"
            "    detect_and_remove_tlag_summary_columns.md    data dictionary:\n"
            "                        describes every column of the summary CSV\n"
            "    detect_and_remove_tlag_checkpoint.csv        phase-1 snapshot\n"
            "    detect_and_remove_tlag_remove_checkpoint.csv phase-2 snapshot\n"
            "\n"
            f"{self.data_subdir}/   STEP 2 -- REMOVE (the deliverable)\n"
            "    Lag-corrected chunk files, and ONLY those. Each scalar column\n"
            "    has been shifted by the PWBOPT-optimised lag.\n"
            f"    >>> Use THIS folder ({self.data_subdir}/) as the input directory\n"
            "    for the next flux-processing step. <<<\n"
            "\n"
            "run_settings.txt    Every setting used for this run, each with a\n"
            "                    one-line explanation (written at run start).\n"
            f"{yaml_note}"
            "log.txt             Plain-text console log of the run.\n"
            "README.txt          This file.\n"
            "\n"
            "Run configuration\n"
            "-----------------\n"
            f"scalars : {scalars_str}\n"
            f"hz      : {self.hz}\n"
            f"chunk   : {self.chunk_seconds:g} s (min {self.min_chunk_seconds:g} s)\n"
            f"removed : column {self.lag_column_template} "
            "(pre-filtered, gap-filled PWBOPT lag)\n"
            "\n"
            "IMPORTANT\n"
            "---------\n"
            "Downstream flux software MUST run with EC time-lag maximization DISABLED:\n"
            f"the tube delay has already been removed in {self.data_subdir}/.\n"
        )
        try:
            (self.output_dir / 'README.txt').write_text(text, encoding='utf-8')
        except Exception:
            # A README failure must never abort the run.
            pass

    def _run_pool(self, kwargs_list, worker_fn, total, phase,
                  checkpoint_path, parent_to_idx, total_files,
                  on_progress, on_active, cancel_event=None) -> list:
        """Dispatch one phase's chunk tasks; collect and checkpoint result rows.

        ``worker_fn`` is the picklable module-level worker (``_detect_worker``
        or ``_remove_worker``). Drains the progress queue in the main process,
        snapshots a checkpoint CSV after every completion, and forwards the
        ``on_progress`` / ``on_active`` callbacks tagged with ``phase``.

        ``cancel_event`` (a ``threading.Event``) lets a caller abort mid-phase:
        it is checked while draining; when set, pending futures are cancelled
        and already-running chunks are allowed to finish, then the partial
        rows collected so far are returned.
        """
        cancelled = False
        rows: list[dict] = []
        active: dict = {}  # pid -> {'parent', 'chunk_index', 'chunk_period'}
        done = 0

        def _checkpoint_save():
            """Persist current rows so an interrupted phase can be inspected.

            Wrapped in try/except so a locked CSV (e.g. open in Excel) never
            kills the run — matches ``PwbBatchDetection``'s pattern.
            """
            if not rows or checkpoint_path is None:
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
            nonlocal done
            pid = ev.get('pid')
            kind = ev['event']
            if kind == 'start':
                active[pid] = {
                    'parent': ev['parent'],
                    'chunk_index': ev['chunk_index'],
                    'chunk_period': ev['chunk_period'],
                }
                if on_active is not None:
                    on_active(active, phase)
            elif kind == 'done':
                active.pop(pid, None)
                row = ev['row']
                rows.append(row)
                done += 1
                _checkpoint_save()
                # Phantom past-EOF chunks (from the padded estimate) are kept
                # for pool accounting but hidden from the user-facing log/bar;
                # run() drops them from the summary afterwards.
                if on_progress is not None and row.get('status') != 'empty:eof':
                    on_progress(done, total, row, phase)
                if on_active is not None:
                    on_active(active, phase)

        if not kwargs_list:
            return rows

        if self.n_workers == 1:
            # In-process loop: drain start/done events synchronously through a
            # tiny queue-like shim (no real IPC), so behaviour matches the
            # parallel path exactly.
            class _SyncQueue:
                def put(self, ev):
                    _handle_event(ev)

            sync_q = _SyncQueue()
            for kwargs in kwargs_list:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                worker_fn(kwargs, sync_q)
        else:
            # Parallel: dispatch ALL tasks as independent chunks and drain the
            # Manager queue in the main process while workers run. With N
            # workers and M chunks all N workers stay busy until min(M, N)
            # chunks remain — including one file with 12 chunks and 4 workers.
            with Manager() as manager:
                progress_queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=self.n_workers,
                                         initializer=_ensure_headless_backend) as executor:
                    futures = [
                        executor.submit(worker_fn, kwargs, progress_queue)
                        for kwargs in kwargs_list
                    ]

                    pending = set(futures)
                    while True:
                        if (cancel_event is not None and cancel_event.is_set()
                                and not cancelled):
                            # Abort: cancel queued chunks, let running ones
                            # finish (bounded by n_workers), stop draining.
                            cancelled = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
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

                    # A worker that raised (strict mode re-raise) or died hard
                    # (OOM, segfault, BrokenProcessPool) emits no 'done' event,
                    # so its chunk silently vanishes from ``rows``. Collect the
                    # future exceptions instead of swallowing them: re-raise in
                    # strict mode, otherwise warn so the loss is visible.
                    worker_errors = []
                    for fut in futures:
                        try:
                            fut.result()
                        except CancelledError:
                            pass  # pending chunk cancelled by Stop — expected
                        except Exception as fe:
                            worker_errors.append(fe)
                    if worker_errors:
                        if self.strict:
                            raise worker_errors[0]
                        warn(f"{phase}: {len(worker_errors)} worker process(es) "
                             f"failed without reporting a result row "
                             f"(first error: {type(worker_errors[0]).__name__}: "
                             f"{worker_errors[0]}). The affected chunks were "
                             f"NOT written.")

        # Every dispatched task should produce exactly one row. A shortfall
        # means chunks were lost (e.g. a killed worker) — surface it rather
        # than returning a silently-incomplete summary. (Skip when the user
        # cancelled: a shortfall is then expected.)
        if not cancelled and len(rows) < len(kwargs_list):
            warn(f"{phase}: only {len(rows)} of {len(kwargs_list)} dispatched "
                 f"chunks reported a result; {len(kwargs_list) - len(rows)} "
                 f"are missing from the summary.")
        return rows

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
                   metavar='LABEL:column[@lag=..;uws=..]', required=True,
                   help='Gas label and column name in the raw file, e.g. '
                        '"CH4:CH4_DRY_[LGR-A]". Repeat for each gas. Append '
                        'an optional per-gas time-lag window after "@" as '
                        '";"-separated key=value pairs (seconds): lag (lag_max), '
                        'block, lws, uws -- e.g. "H2O:h2o@lag=30;uws=25" gives '
                        'a long-inlet gas a wider window than the dry gases.')
    # --- PWB parameters ---
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    p.add_argument('--lag-max', type=float, default=10.0,
                   help='CCF search half-width [s].')
    p.add_argument('--n-bootstrap', type=int, default=99,
                   help='Number of block-bootstrap replicates (paper: 99).')
    p.add_argument('--block-length', type=float, default=20.0,
                   help='Bootstrap block length [s] (paper: L = 20 s).')
    p.add_argument('--lws', type=float, default=None,
                   help='Optional lower limit [s] of an asymmetric lag search '
                        'window applied to all gases (per-gas "@lws=" overrides).')
    p.add_argument('--uws', type=float, default=None,
                   help='Optional upper limit [s] of the asymmetric lag search '
                        'window applied to all gases (per-gas "@uws=" overrides).')
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
    p.add_argument('--lineterm', default='auto',
                   help=r"Line terminator for the output file. Default "
                        r"'auto' reproduces the input file's convention "
                        r"(CRLF for typical Windows EC logger files, LF for "
                        r"Unix). Override with '\r\n' or '\n' to force one.")
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
    p.add_argument('--lag-column-template', default='{prefix}_tlag_final_pf_s',
                   help='Which PWBOPT lag column to actually remove in phase '
                        '2. Use {prefix} for the lowercased scalar label. '
                        'Default {prefix}_tlag_final_pf_s (pre-filtered, '
                        'gap-filled "best" lag; matches diive-tlag-apply-batch). '
                        'Use {prefix}_tlag_final_s for the non-pre-filtered '
                        'PWBOPT lag.')
    # --- Output layout (numbered by pipeline phase) ---
    p.add_argument('--detect-subdir', default='1_lag_detection',
                   help='Subfolder of --output-dir for step-1 (detect) '
                        'diagnostics: plots/ and plots_summary/ '
                        '(default: 1_lag_detection).')
    p.add_argument('--data-subdir', default='2_lag_removed',
                   help='Subfolder of --output-dir for step-2 (remove) output: '
                        'the lag-corrected chunk files (default: 2_lag_removed). '
                        'Kept separate from the summary CSV / log / plots so it '
                        'can be used directly as the input directory for the '
                        'next flux step.')
    # --- Plots ---
    p.add_argument('--save-plots', action='store_true',
                   help='Save the 3-panel PWB diagnostic figure per chunk per '
                        'scalar into <output-dir>/<detect-subdir>/plots/, and '
                        'the batch-overview figures into '
                        '<output-dir>/<detect-subdir>/plots_summary/.')
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
    _ensure_headless_backend()
    import sys
    from datetime import datetime, timezone
    from rich.console import Console as _Console, Group
    from rich.live import Live
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    from rich.text import Text as _RichText
    # On-screen console only. The plain-text log is built explicitly in
    # ``log_lines`` (NOT via console recording), so the animated live display
    # — spinners, pulsing bars, the stacked worker rows — never pollutes
    # log.txt; only the static header, per-chunk completions, and the summary
    # are written there.
    console = _Console(log_path=False)
    log_lines: list = []

    def _logline(markup: str = '') -> None:
        """Append a plain-text (markup-stripped) line to the log buffer."""
        try:
            log_lines.append(_RichText.from_markup(markup).plain)
        except Exception:
            log_lines.append(str(markup))

    def out(markup: str = '') -> None:
        """Print a static line on screen AND record it for log.txt."""
        console.print(markup)
        _logline(markup)

    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: --input-dir not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    scalars = {}
    per_gas_lag: dict = {}
    for token in args.scalars:
        try:
            label, col, overrides = parse_scalar_spec(token)
        except ValueError as e:
            print(f'ERROR: {e}', file=sys.stderr)
            sys.exit(1)
        scalars[label] = col
        if overrides:
            per_gas_lag[label] = overrides

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

    # Drop a TUI-loadable settings YAML so this CLI run can be reopened/edited
    # in the detect+remove TUI. Optional: needs the textual ('gui') extra, and
    # a missing TUI module or any write error must never stop the run.
    settings_yaml = None
    try:
        from diive.flux.hires.detect_and_remove_tlag_tui import (
            write_run_settings_yaml)
        settings_yaml = write_run_settings_yaml(
            output_dir, args, scalars, per_gas_lag)
    except Exception:
        settings_yaml = None

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
        detect_subdir=args.detect_subdir,
        data_subdir=args.data_subdir,
        hdi_thresh=args.hdi_thresh,
        dev_thresh=args.dev_thresh,
        hdi_prefilter=args.hdi_prefilter,
        lag_column_template=args.lag_column_template,
        lws=args.lws,
        uws=args.uws,
        per_gas_lag=per_gas_lag,
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

    # Header: static run metadata, mirrored into log.txt via out().
    out()
    out(f'[dim]started:[/dim] {started_at}')
    out(f'[dim]input :[/dim]  {input_dir}')
    out(f'[dim]output:[/dim]  {output_dir}')
    out(f'[dim]files :[/dim]  {total}  ([dim]pattern[/dim] {args.file_pattern!r})')
    out(f'[dim]mode  :[/dim]  {mode}')
    out(f'[dim]hz    :[/dim]  {args.hz}    '
        f'[dim]lag-max[/dim] {args.lag_max} s    '
        f'[dim]n-boot[/dim] {args.n_bootstrap}    '
        f'[dim]block[/dim] {args.block_length} s')
    out(f'[dim]chunk :[/dim]  {args.chunk_seconds:g} s  '
        f'[dim](min-chunk[/dim] {args.min_chunk_seconds:g} s[dim])[/dim]  '
        f'[dim]≈{chunks_per_file}/file → {total_chunks_est} total[/dim]  '
        f'[dim]template[/dim] {args.chunk_name_template!r}')
    if args.start_time_regex:
        out(f'[dim]time  :[/dim]  regex {args.start_time_regex!r}  '
            f'[dim]format[/dim] {args.start_time_format!r}')
    out(f'[dim]wind  :[/dim]  '
        f'U={args.col_u!r}  V={args.col_v!r}  W={args.col_w!r}  '
        f'T_SONIC={args.col_tsonic!r}')
    for label, col in scalars.items():
        out(f'[dim]scalar:[/dim]  [bold]{label}[/bold]  <- {col!r}')
    out(f'[dim]PWBOPT:[/dim]  hdi-thresh {args.hdi_thresh} s    '
        f'dev-thresh {args.dev_thresh} s    '
        f'hdi-prefilter {args.hdi_prefilter} s')
    out(f'[dim]align :[/dim]  lag column '
        f'[bold]{args.lag_column_template}[/bold] '
        f'[dim](phase 2 shifts scalars by this PWBOPT lag to remove it)[/dim]')
    if settings_yaml:
        out(f'[dim]config:[/dim]  {settings_yaml}  '
            f'[dim](reload in the TUI to inspect/reproduce)[/dim]')
    out(f'[dim]data  :[/dim]  lag-corrected chunks -> '
        f'[bold]{output_dir / args.data_subdir}[/bold]  '
        f'[dim](step 2; next-step input)[/dim]')
    out(f'[dim]plots :[/dim]  '
        f'{f"yes (step 1: {args.detect_subdir}/plots + plots_summary)" if args.save_plots else "no"}')
    out()

    msg = (f'PWB per-chunk pipeline  {total} files  '
           f'{mode}  -> {output_dir}')
    out(f'[bold]{msg}[/bold]')
    out()

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

    # Two-tier live display:
    #   - one stacked row per worker (spinner + pulsing bar + current chunk),
    #     so many workers stack vertically instead of overflowing the width;
    #   - one overall bar with M/N count, elapsed, and ETA.
    overall = Progress(
        TextColumn('[progress.description]{task.description}'),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    workers = Progress(
        SpinnerColumn(),
        BarColumn(bar_width=18),
        TextColumn('{task.description}'),
        console=console,
    )
    overall_id = overall.add_task(f'[cyan]{mode}[/cyan]', total=total_chunks_est)

    # One reusable row per worker slot (capped so a huge core count can't
    # flood the screen). Rows are assigned to worker PIDs as they appear and
    # released when that worker finishes its chunk. total=None -> the bar
    # pulses while the chunk is in flight (no sub-chunk progress to report).
    n_rows = max(1, min(pipeline.n_workers, 24))
    IDLE_DESC = '[dim]idle[/dim]'
    worker_ids = [workers.add_task(IDLE_DESC, total=None, start=False)
                  for _ in range(n_rows)]
    free_rows = list(worker_ids)
    pid_to_row: dict = {}

    live_group = Group(overall, workers)

    def _short_parent(name: str) -> str:
        stem = Path(name).stem
        return stem if len(stem) <= 24 else stem[:11] + '…' + stem[-12:]

    def _short_period(period: str) -> str:
        """Trim the chunk-period filename for compact display."""
        return Path(period).stem

    summary = None
    # Tracks the active phase so the overall bar resets (new total) when
    # phase 1 (detect) hands off to phase 2 (remove).
    phase_state = {'phase': None}

    def _phase_tag(phase: str) -> str:
        # 'remove' = remove the time LAG (align scalar to wind), not any file;
        # shown as 'align' (the paper's 'temporal alignment').
        return ('[magenta]detect[/magenta]' if phase == 'detect'
                else '[blue]align[/blue]')

    try:
        with Live(live_group, console=console, refresh_per_second=8,
                  transient=True):
            def _on_progress(done, total_chunks, row, phase):
                # Fires once per chunk completion. ``phase`` is 'detect' or
                # 'remove'; the two phases have different row schemas.
                if phase != phase_state['phase']:
                    phase_state['phase'] = phase
                    overall.reset(overall_id, total=total_chunks)
                    overall.update(
                        overall_id,
                        description=f'[cyan]{mode}[/cyan] {_phase_tag(phase)}')
                chunk_short = _short_period(row.get('period', ''))
                parent_short = _short_period(row.get('parent', ''))
                # Show source (6 h) file then chunk, so each line is
                # traceable to its input file.
                period_short = (f'{parent_short} › {chunk_short}'
                                if parent_short else chunk_short)
                if phase == 'remove':
                    if row.get('write_status') == 'error':
                        parts = f'[red]ERR[/red] {row.get("write_error", "")[:60]}'
                    else:
                        # Applied lag in seconds = shifted records / hz (the
                        # shift is by whole records, so this is the exact lag).
                        cells = []
                        for g in scalars:
                            rec = row.get(f'{g.lower()}_applied_records')
                            if rec is None or rec != rec:
                                cells.append(f'{g}=--')
                            else:
                                rec = int(rec)
                                cells.append(f'{g}={rec / args.hz:.2f}s ({rec}rec)')
                        parts = f'[green]aligned[/green] ' + '  '.join(cells)
                elif row.get('status') == 'error':
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
                _logline(f'{period_short}  {parts}')
                overall.update(overall_id, completed=done)

            def _on_active(active, phase):
                # Fires whenever the in-flight set changes. Each active worker
                # gets its own stacked row (PID -> row), so the display height
                # is bounded by worker count and never overflows the width.
                # ``active`` maps worker pid -> {'parent', 'chunk_index',
                # 'chunk_period'}.
                for pid in [p for p in pid_to_row if p not in active]:
                    tid = pid_to_row.pop(pid)
                    workers.stop_task(tid)
                    workers.update(tid, description=IDLE_DESC)
                    free_rows.append(tid)
                for pid in sorted(active):
                    info = active[pid]
                    if pid not in pid_to_row:
                        if not free_rows:
                            continue  # more workers than rows; skip extras
                        tid = free_rows.pop(0)
                        pid_to_row[pid] = tid
                        workers.start_task(tid)
                    tid = pid_to_row[pid]
                    workers.update(
                        tid,
                        description=(
                            f'{_phase_tag(phase)} '
                            f'[bold cyan]{_short_parent(info["parent"])}[/bold cyan]'
                            f'[dim]·c{info["chunk_index"]:02d}[/dim]'),
                    )

            def _on_status(msg):
                # Coarse pre-phase-1 progress (the up-front file scan reads
                # every input file before any chunk callback fires).
                console.log(f'[dim]{msg}[/dim]')
                _logline(msg)

            summary = pipeline.run(on_progress=_on_progress,
                                   on_active=_on_active,
                                   on_status=_on_status)

        # Per-chunk outcome counts
        n_chunks = len(summary)
        if n_chunks == 0 or 'status' not in summary.columns:
            out('[yellow]WARN[/yellow] No chunk rows were produced — '
                'nothing written. Check --col-* names, '
                '--skiprows / --extra-rows / --sep, and '
                '--chunk-name-template.')
            raise SystemExit(1)
        n_ok = int((summary['status'] == 'ok').sum())
        n_err = int((summary['status'] == 'error').sum())
        n_skip = int((summary['status'] == 'skipped:short').sum())
        color = 'green' if n_err == 0 else 'yellow'
        msg_chunks = (f'[{color}]Done — {n_ok}/{n_chunks} chunks written'
                      f'{f", {n_skip} short" if n_skip else ""}'
                      f'{f", {n_err} errors" if n_err else ""}'
                      f' (across {total} files, estimate was '
                      f'{total_chunks_est} chunks).[/{color}]')
        out()
        out(msg_chunks)

        # run() already wrote the summary CSV and (when --save-plots) the
        # batch overview plots via _write_summary_and_plots — log the paths
        # it stored. (Centralising the writing there means the TUI and bare
        # Python callers produce the same files as the CLI.)
        if pipeline.summary_csv_path:
            out(f'[dim]Summary saved to:[/dim] '
                f'[cyan]{pipeline.summary_csv_path}[/cyan]')
        if pipeline.summary_plots_dir:
            out(f'[dim]Overview plots saved to:[/dim] '
                f'[cyan]{pipeline.summary_plots_dir}[/cyan]  '
                f'[dim](summary_<scalar>.png + summary_lag_comparison.png)[/dim]')

    except BaseException as e:
        # Capture into the log before re-raising. BaseException catches
        # KeyboardInterrupt + SystemExit too — important for crash diagnosis.
        out(f'[bold red]ABORT[/bold red] {type(e).__name__}: {e}')
        # Re-raise after the finally block has dumped the log.
        raise
    finally:
        finished_at = datetime.now(timezone.utc).astimezone().isoformat(
            timespec='seconds')
        out(f'[dim]finished:[/dim] {finished_at}')
        # Write the explicit plain-text buffer — only static lines + per-chunk
        # completions, never the animated live frames.
        try:
            log_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
        except Exception:
            # Never let log-saving fail the whole run.
            pass
        # Echo log location to stdout last so it survives even when Rich
        # output is redirected.
        print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    _cli_main()
