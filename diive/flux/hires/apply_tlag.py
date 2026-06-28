"""
APPLY_TLAG: APPLY PRE-WHITENING-BOOTSTRAP TIME LAGS TO RAW EC FILES
====================================================================

After ``PwbBatchDetection`` (``lag_pwb.py``) writes a ``tlag_results.csv`` with
one detected lag per averaging period and gas, this module reads that file
and applies the lags to the original high-frequency EC files by shifting each
scalar column backward along the time axis. The result is a parallel directory
of time-lag-corrected files ready for downstream flux calculation (e.g. via
``diive.flux.run_chain`` or EddyPro).

Recommended workflow
--------------------
PWB time-lag detection is typically run on **rotated** files (wind-rotation-
corrected EddyPro "Advanced" output), because a stable W component sharpens
the cross-correlation peak. The lag is then **removed from the original raw
(unrotated) files**, which are subsequently fed into the flux-processing
pipeline. During that flux processing, the EC time-lag maximization step
must be **disabled** (the lag has already been corrected here).

The detection format and the apply format therefore often differ. This
module handles arbitrary text formats via the ``--sep`` / ``--header-row`` /
``--extra-rows`` / ``--lineterm`` flags (default settings target EddyPro
rotated files; see the CLI section below for a raw CSV example). When the
filenames in the rotated detection dir and the raw apply dir differ, the
``--period-key-regex`` / ``--file-key-regex`` flags let you match files
across the two layouts via a common key (typically a YYYYMMDDHHMM
timestamp).

Sign convention
---------------
Positive lag = scalar arrives later than wind (tube delay). To align the
scalar with the wind, the scalar series is shifted backward in row index by
``n = round(tlag_s * hz)`` records using ``pd.Series.shift(periods=-n)``. The
trailing ``n`` rows become NaN and are written as ``-9999`` in EddyPro
convention.

Negative lags would shift the scalar forward; in practice the default lag
column (``{prefix}_tlag_final_pf_s``) is the pre-filtered, gap-filled PWBOPT
result, where unreliable detections have already been replaced with the
carry-forward / median-fallback value, so this should not happen for the
defaults.

Other columns (wind components, sonic temperature, gases not listed in
``--scalar``) pass through unchanged. The output preserves the original
metadata header rows, the column-name row, and the column order, so the
aligned files are drop-in replacements for downstream EddyPro / flux tools.

CLI
---
Run from the command line::

    uv run diive-tlag-apply-batch --help

or via the module form::

    uv run python -m diive.flux.hires.apply_tlag --help

**Required flags**

``--input-dir PATH``
    Directory containing the original EddyPro rotated ``.txt`` files (the same
    directory that was passed to ``diive-tlag-pwb-batch``).
``--output-dir PATH``
    Output directory for the lag-corrected files.
``--results-csv PATH``
    The ``tlag_results.csv`` produced by ``diive-tlag-pwb-batch``.
``--scalar LABEL:column``
    Gas label and column name to shift, e.g. ``CH4:ch4``. Repeat for each
    gas. The label is used to look up the lag column
    ``{label_lowercased}_tlag_final_pf_s`` (override the template with
    ``--lag-column-template``).

**Optional flags**

``--lag-column-template TEMPLATE``
    Template for the lag column in the results CSV. Use ``{prefix}`` as the
    placeholder for the lowercased scalar label. Default:
    ``{prefix}_tlag_final_pf_s`` (pre-filtered PWBOPT after gap-fill).
``--hz N``
    Sampling frequency in Hz. Default: 20.

**File-format flags** (defaults target EddyPro rotated files)

``--skiprows N``
    Number of lines BEFORE the column-name (header) row. EddyPro default: 9.
    Set to 0 if the header is on the first line.
``--extra-rows N``
    Number of extra rows AFTER the header but BEFORE data (e.g. a units row
    and an instrument-source row). Default: 0. For typical raw EC CSV files
    with header + units + source rows, use ``--extra-rows 2``.
``--sep STR``
    Field separator used both for parsing the input and writing the output.
    Default: ``whitespace`` (any run of whitespace, regex ``\\s+``). Common
    overrides: ``,`` for CSV, ``\\t`` for TSV.
``--lineterm STR``
    Line terminator written between data rows. Default: ``\\n``. Use
    ``\\r\\n`` to match a Windows-CRLF input file.
``--na-values V [V ...]``
    Strings treated as NaN on read. Default: ``-9999``, ``-9999.0``,
    ``-9999.0000000000000``.
``--na-rep STR``
    Value written for NaN on output. Default: ``-9999``.

**Filename mapping flags** (use when raw and rotated dirs have different filenames)

``--period-key-regex REGEX``
    Regex applied to each ``period`` value in the results CSV; the
    concatenation of its capture groups (or the whole match if no groups)
    becomes the "key". When set, the apply step looks up the matching raw
    file in ``--input-dir`` by key instead of by literal filename.
``--file-key-regex REGEX``
    Regex applied to each filename in ``--input-dir`` to extract its key,
    same semantics as ``--period-key-regex``. Set both flags together to
    bridge a name-mismatch between detection and application dirs.

**Misc**

``--period-col NAME``
    Name of the filename column in the results CSV. Default: ``period``.
``--n-workers N``
    Number of parallel worker processes. Default: all available CPU cores.
``--strict``
    Re-raise worker exceptions instead of capturing them into the summary
    DataFrame. Useful for debugging.

**Example: detection on rotated, application on raw CSV**

Detection (rotated files, EddyPro-style whitespace, 9-line metadata,
filenames like ``20210722-1100_raw_dataset_..._adv.txt``)::

    uv run diive-tlag-pwb-batch ... (writes tlag_results.csv) ...

Application (raw files, comma-separated, 3-line header [names/units/source],
filenames like ``CH-CHA_202107221100.csv``)::

    uv run diive-tlag-apply-batch ^
        --input-dir   PATH/TO/raw_csv_files ^
        --output-dir  PATH/TO/aligned_raw ^
        --results-csv PATH/TO/tlag_results.csv ^
        --scalar "CH4:CH4_DRY_[LGR-A]" --scalar "N2O:N2O_DRY_[LGR-A]" ^
        --skiprows 0 --extra-rows 2 --sep "," ^
        --na-values NAN -9999 ^
        --period-key-regex "(\\d{8})-(\\d{4})" ^
        --file-key-regex   "(\\d{8})(\\d{4})" ^
        --hz 20 --n-workers 4

The ``--period-key-regex`` extracts ``YYYYMMDD`` and ``HHMM`` from the
period filename (rotated style with a hyphen) and concatenates them.
``--file-key-regex`` does the same on each raw filename (no hyphen). Same
12-character key on both sides ⇒ match.

Windows note:
    ``ProcessPoolExecutor`` uses the *spawn* start method on Windows. Any
    script that instantiates ``TlagApplier`` must guard its entry point::

        if __name__ == '__main__':
            applier = TlagApplier(...)
            summary = applier.run()

Example
-------
See ``examples/flux/hires/flux_apply_tlag_cli.py`` for a complete example.

Part of the diive library: https://github.com/holukas/diive
"""

import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

# Suppress the runpy double-import warning that fires in every worker process
# when diive.__init__ has already imported this module before -m re-executes it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

import numpy as np
import pandas as pd
from pandas import DataFrame

# Same NA strings as the PWB worker, so the round-trip preserves EddyPro
# semantics. Importing keeps the two modules in lockstep.
from diive.flux.hires.lag_pwb import _DEFAULT_NA_VALUES

# Sentinel used in the CLI parser and class default for "whitespace separator".
# pandas understands the regex ``r'\s+'`` for reading; for writing we fall back
# to a single space character (pandas ``to_csv`` requires a literal sep).
_WHITESPACE_SEP = r'\s+'


def _extract_key(pattern: str | None, name: str) -> str | None:
    """Extract a filename key via regex.

    - If ``pattern`` is None, the whole name is used as the key (exact match).
    - If the pattern has one or more capture groups, the concatenation of all
      non-None groups is the key (lets users combine date + time parts).
    - If the pattern has no groups, the whole match is the key.
    - Returns None if the pattern does not match (caller treats that as
      "no raw file for this period").
    """
    if not pattern:
        return name
    m = re.search(pattern, name)
    if m is None:
        return None
    if m.groups():
        return ''.join(g for g in m.groups() if g is not None)
    return m.group(0)


def _build_filename_map(input_dir: Path, file_pattern: str | None) -> dict:
    """Scan ``input_dir`` and return ``{key: Path}`` for regular files.

    Files are visited in sorted order so the mapping is deterministic. A key
    collision (two raw files extracting the same key, e.g. an over-broad
    ``--file-key-regex`` that only captures the date) is a configuration
    error: it would silently apply one file's lag to the other and drop a
    file. Raise with both offending names rather than picking one silently.
    """
    name_map: dict = {}
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        key = _extract_key(file_pattern, f.name)
        if key is None:
            continue
        if key in name_map:
            raise ValueError(
                f"--file-key-regex {file_pattern!r} maps two input files to "
                f"the same key {key!r}: {name_map[key].name!r} and "
                f"{f.name!r}. Tighten the regex so every raw file gets a "
                f"unique key, otherwise one file's lag would be applied to "
                f"the wrong file."
            )
        name_map[key] = f
    return name_map


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope to be picklable by
# ProcessPoolExecutor (Windows spawn: each worker re-imports this module).
# ---------------------------------------------------------------------------

def _apply_tlag_file_worker(args: tuple) -> dict:
    """Apply per-gas lags to one EC raw file and write the aligned output.

    Format model:
    - Lines 0 .. skiprows-1                = metadata block (preserved verbatim)
    - Line skiprows                        = column-name header row (preserved)
    - Lines skiprows+1 .. skiprows+extra_rows = extra header rows
                                             (e.g. units, instrument tags)
    - From line skiprows+1+extra_rows on   = data
    """
    (input_path, output_path, scalars, lags_per_gas, hz,
     skiprows, extra_rows, sep, lineterm, na_values, na_rep, strict) = args

    period_name = Path(input_path).name
    row: dict = {'period': period_name, 'status': 'ok', 'error': ''}
    for label in scalars:
        row[f'{label.lower()}_applied_lag_s'] = np.nan
        row[f'{label.lower()}_applied_records'] = np.nan
        row[f'{label.lower()}_status'] = 'pending'

    try:
        n_preserved = skiprows + 1 + extra_rows
        header_idx = skiprows

        # Preserve metadata + header + extra-header rows verbatim
        with open(input_path, 'r', encoding='utf-8', errors='replace') as fh:
            preserved_lines = [next(fh) for _ in range(n_preserved)]

        header_line = preserved_lines[header_idx].rstrip('\n').rstrip('\r')
        # Tokenize header by the same separator used for the data. For the
        # regex whitespace sentinel, fall back to ``str.split()`` (which
        # collapses any whitespace run).
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
                f"{df.shape[1]}. Check --skiprows ({skiprows}), --extra-rows "
                f"({extra_rows}), --sep ({sep!r})."
            )
        df.columns = header_cols

        for label, col_name in scalars.items():
            prefix = label.lower()
            lag_s = lags_per_gas.get(label, np.nan)

            if col_name not in df.columns:
                row[f'{prefix}_status'] = 'skipped:column_missing'
                continue
            if not np.isfinite(lag_s):
                row[f'{prefix}_status'] = 'skipped:lag_nan'
                continue

            n_records = int(round(lag_s * hz))
            # Positive lag (scalar delayed) -> shift backward (negative periods)
            # so that, at each row index t, the scalar value now refers to the
            # same air parcel as the wind at t. The trailing n_records rows
            # become NaN and are written as `na_rep`.
            df[col_name] = df[col_name].shift(periods=-n_records)
            row[f'{prefix}_applied_lag_s'] = float(lag_s)
            row[f'{prefix}_applied_records'] = n_records
            row[f'{prefix}_status'] = 'ok'

        # pandas ``to_csv`` needs a literal separator; for the whitespace
        # sentinel we write a single space (round-trips cleanly through any
        # whitespace-tolerant reader).
        out_sep = ' ' if sep == _WHITESPACE_SEP else sep

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as fh:
            for line in preserved_lines:
                fh.write(line)
            df.to_csv(fh, sep=out_sep, index=False, header=False,
                      na_rep=na_rep, lineterminator=lineterm)

    except Exception as e:
        if strict:
            raise
        row['status'] = 'error'
        row['error'] = f'{type(e).__name__}: {e}'

    return row


# ---------------------------------------------------------------------------
# TlagApplier
# ---------------------------------------------------------------------------

class TlagApplier:
    """
    Apply PWB-detected time lags to raw, high-frequency EC files.

    Reads a ``tlag_results.csv`` produced by ``PwbBatchDetection``, looks up
    the per-period lag for each requested gas, and writes a parallel directory
    of lag-corrected files. Each scalar column is shifted backward by
    ``round(tlag_s * hz)`` rows; all other columns pass through unchanged.

    See the module docstring for the sign convention and CLI usage.

    Parameters
    ----------
    input_dir : Path
        Directory containing the original EddyPro rotated ``.txt`` files (the
        same directory passed to ``diive-tlag-pwb-batch``).
    output_dir : Path
        Output directory for the lag-corrected files. Created if it does not
        exist. Originals are not touched.
    results_csv : Path
        Path to the ``tlag_results.csv`` produced by ``PwbBatchDetection``.
    scalars : dict
        ``{label: column_name}`` mapping, e.g. ``{'CH4': 'ch4', 'N2O': 'n2o'}``.
        The label is lowercased to look up the lag column via
        ``lag_column_template``; the column name identifies the column in the
        EddyPro file to shift.
    lag_column_template : str
        Template for the lag column in the results CSV. ``{prefix}`` is
        replaced by the lowercased scalar label. Default:
        ``{prefix}_tlag_final_pf_s`` (pre-filtered PWBOPT after gap-fill —
        the production-ready, NaN-free lag column).
    hz : int
        Sampling frequency in Hz. Default: 20.
    skiprows : int
        Number of metadata rows BEFORE the column-name row in each input
        file. EddyPro default: 9. Set to 0 if the header is on the first
        line. The column-name row itself is preserved verbatim.
    extra_rows : int
        Number of extra rows AFTER the header but BEFORE data (e.g. a units
        row and an instrument-source row). Default: 0. Typical raw EC CSV
        files with header + units + source rows use ``extra_rows=2``.
    sep : str
        Field separator used for parsing input and writing output. Default:
        whitespace (regex ``\\s+``). Use ``,`` for CSV, ``\\t`` for TSV.
    lineterm : str
        Line terminator written between data rows. Default: ``\\n``.
    na_values : list of str, optional
        Strings treated as NaN on read. Default: EddyPro conventions
        (``-9999``, ``-9999.0``, ``-9999.0000000000000``).
    na_rep : str
        Value written for NaN on output. Default: ``-9999``.
    period_col : str
        Name of the filename column in the results CSV. Default: ``period``.
    period_key_regex : str, optional
        Regex with capture groups that extracts a "key" (typically a
        timestamp like ``YYYYMMDDHHMM``) from each ``period`` value in the
        results CSV. Combined with ``file_key_regex``, lets the apply step
        match raw files to results-CSV rows by key when the filename
        conventions differ.
    file_key_regex : str, optional
        Regex with capture groups that extracts a key from each filename in
        ``input_dir``. Must extract the same key format as
        ``period_key_regex`` to make a match.
    n_workers : int, optional
        Number of parallel worker processes. Default: ``os.cpu_count()``.
    strict : bool
        If True, re-raise worker exceptions instead of capturing them into the
        summary. Useful for debugging.

    Example
    -------
    See ``examples/flux/hires/flux_apply_tlag_cli.py`` for a complete example.

    See Also
    --------
    PwbBatchDetection : Detects the lags that this class applies.
    """

    def __init__(
            self,
            input_dir: Path,
            output_dir: Path,
            results_csv: Path,
            scalars: dict,
            lag_column_template: str = '{prefix}_tlag_final_pf_s',
            hz: int = 20,
            skiprows: int = 9,
            extra_rows: int = 0,
            sep: str = _WHITESPACE_SEP,
            lineterm: str = '\n',
            na_values: list | None = None,
            na_rep: str = '-9999',
            period_col: str = 'period',
            period_key_regex: str | None = None,
            file_key_regex: str | None = None,
            n_workers: int | None = None,
            strict: bool = False,
    ):
        """Set up time-lag application from a results file. See the class docstring."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results_csv = Path(results_csv)
        self.scalars = scalars
        self.lag_column_template = lag_column_template
        self.hz = hz
        self.skiprows = skiprows
        self.extra_rows = extra_rows
        self.sep = sep
        self.lineterm = lineterm
        self.na_values = na_values if na_values is not None else _DEFAULT_NA_VALUES
        self.na_rep = na_rep
        self.period_col = period_col
        self.period_key_regex = period_key_regex
        self.file_key_regex = file_key_regex
        self.n_workers = n_workers or os.cpu_count()
        self.strict = strict

        self._summary: DataFrame | None = None

    @property
    def summary(self) -> DataFrame:
        """Per-file summary DataFrame (available after ``run()``)."""
        if self._summary is None:
            raise RuntimeError("Call run() first.")
        return self._summary

    def run(self, on_progress: Callable | None = None) -> DataFrame:
        """
        Apply lags in parallel and return a summary DataFrame.

        Each row of the returned DataFrame corresponds to one input file and
        contains: ``period`` (filename), ``status`` (``ok`` / ``error``),
        ``error`` (exception text, empty on success), and per-gas
        ``{prefix}_applied_lag_s`` (lag in seconds actually applied),
        ``{prefix}_applied_records`` (rows shifted), and ``{prefix}_status``
        (``ok`` / ``skipped:lag_nan`` / ``skipped:column_missing`` /
        ``pending``).

        Args:
            on_progress: Optional callback ``f(completed, total, row)`` called
                each time a file finishes.

        Returns:
            Summary DataFrame, one row per input file, sorted to match the
            order of the results CSV.
        """
        # Read the results CSV; require the period column. Force the period
        # column to string: a numeric-looking filename (e.g. "202107221100")
        # would otherwise be inferred as int/float, breaking regex key
        # extraction and producing a "...1100.0" path that never matches.
        results = pd.read_csv(self.results_csv, dtype={self.period_col: str})
        if self.period_col not in results.columns:
            raise ValueError(
                f"Results CSV missing required column {self.period_col!r}. "
                f"Available: {list(results.columns)}"
            )

        # Resolve per-file lag columns up-front so the worker only needs the
        # lag values, not the whole results DataFrame.
        lag_cols = {
            label: self.lag_column_template.format(prefix=label.lower())
            for label in self.scalars
        }
        missing = [c for c in lag_cols.values() if c not in results.columns]
        if missing:
            raise ValueError(
                f"Lag column(s) missing from results CSV: {missing}. "
                f"Check --lag-column-template (was {self.lag_column_template!r}). "
                f"Available columns: {list(results.columns)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # If filename mapping is configured, pre-scan the input directory
        # once so the worker tuples carry resolved paths. When neither
        # regex is set, fall back to direct ``input_dir / period_name``
        # (the simple case where both directories use the same names).
        use_keymap = bool(self.period_key_regex or self.file_key_regex)
        file_map = (_build_filename_map(self.input_dir, self.file_key_regex)
                    if use_keymap else None)

        worker_args: list = []
        prerows: list[dict] = []  # rows we resolve here (e.g. lookup failures)
        # Guard against two results rows resolving to the same output file:
        # each would overwrite the other with a different lag, silently.
        seen_outputs: dict = {}
        for _, row in results.iterrows():
            period_name = row[self.period_col]
            if use_keymap:
                key = _extract_key(self.period_key_regex, period_name)
                input_path = file_map.get(key) if key is not None else None
                if input_path is None:
                    # Record a stub row directly so the user sees the failure;
                    # skip dispatching to the worker.
                    stub = {'period': period_name, 'status': 'error',
                            'error': f'no raw file matched key {key!r}'}
                    for label in self.scalars:
                        stub[f'{label.lower()}_applied_lag_s'] = np.nan
                        stub[f'{label.lower()}_applied_records'] = np.nan
                        stub[f'{label.lower()}_status'] = 'skipped:no_raw_file'
                    prerows.append(stub)
                    continue
                output_path = self.output_dir / input_path.name
            else:
                input_path = self.input_dir / period_name
                output_path = self.output_dir / period_name

            out_key = str(output_path)
            if out_key in seen_outputs:
                raise ValueError(
                    f"Results CSV maps two rows to the same output file "
                    f"{output_path.name!r} (periods {seen_outputs[out_key]!r} "
                    f"and {period_name!r}). One would silently overwrite the "
                    f"other with a different lag. Check the {self.period_col!r} "
                    f"column for duplicates or a too-broad --period-key-regex."
                )
            seen_outputs[out_key] = period_name

            lags_per_gas = {
                label: float(row[lag_cols[label]]) for label in self.scalars
            }
            worker_args.append((
                str(input_path), str(output_path),
                self.scalars, lags_per_gas, self.hz,
                self.skiprows, self.extra_rows, self.sep, self.lineterm,
                self.na_values, self.na_rep, self.strict,
            ))

        rows: list[dict] = list(prerows)
        total = len(worker_args) + len(prerows)
        completed = len(prerows)
        # Stream the pre-resolved failures through the progress callback so
        # users see them immediately, before parallel workers start.
        if on_progress is not None:
            for r in prerows:
                on_progress(completed, total, r)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_period = {
                executor.submit(_apply_tlag_file_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_period):
                result_row = future.result()
                rows.append(result_row)
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total, result_row)

        # Sort to match the order of the results CSV
        name_to_idx = {
            row[self.period_col]: i for i, row in results.iterrows()
        }
        rows.sort(key=lambda r: name_to_idx.get(r.get('period', ''), total))

        self._summary = pd.DataFrame(rows)
        return self._summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m diive.flux.hires.apply_tlag',
        description=(
            'Apply PWB-detected time lags to raw EC files.\n'
            'Alias: uv run diive-tlag-apply-batch'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input-dir', required=True,
                   help='Directory containing the original EddyPro rotated .txt files.')
    p.add_argument('--output-dir', required=True,
                   help='Output directory for lag-corrected files.')
    p.add_argument('--results-csv', required=True,
                   help='tlag_results.csv produced by diive-tlag-pwb-batch.')
    p.add_argument('--scalar', dest='scalars', action='append',
                   metavar='LABEL:column', required=True,
                   help='Gas label and column name, e.g. CH4:ch4. Repeat for each gas.')
    p.add_argument('--lag-column-template', default='{prefix}_tlag_final_pf_s',
                   help='Lag-column template (use {prefix} for lowercased scalar label).')
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    # --- File format ---
    p.add_argument('--skiprows', type=int, default=9,
                   help='Lines BEFORE the column-name row '
                        '(EddyPro rotated: 9; raw CSV with header on line 1: 0).')
    p.add_argument('--extra-rows', type=int, default=0,
                   help='Extra rows AFTER the header but BEFORE data '
                        '(e.g. units + instrument-source rows: 2). '
                        'Each preserved verbatim in the output.')
    p.add_argument('--sep', default=_WHITESPACE_SEP,
                   help=r"Field separator. Default: whitespace (regex '\s+'). "
                        r"Use ',' for CSV, '\t' for TSV.")
    p.add_argument('--lineterm', default='\n',
                   help=r"Line terminator written between data rows. "
                        r"Default '\n'; use '\r\n' to match Windows-CRLF input.")
    p.add_argument('--na-values', nargs='+',
                   default=list(_DEFAULT_NA_VALUES),
                   help='Strings to treat as NaN on read.')
    p.add_argument('--na-rep', default='-9999',
                   help='Value written for NaN on output.')
    # --- Filename mapping (use when raw and rotated dirs differ in naming) ---
    p.add_argument('--period-col', default='period',
                   help='Name of the filename column in the results CSV.')
    p.add_argument('--period-key-regex', default=None,
                   help='Regex applied to each period value; concatenated '
                        'capture groups form the key. Pair with '
                        '--file-key-regex to bridge name mismatches.')
    p.add_argument('--file-key-regex', default=None,
                   help='Regex applied to each filename in --input-dir; '
                        'concatenated capture groups form the key.')
    # --- Execution ---
    p.add_argument('--n-workers', type=int, default=None,
                   help='Parallel worker processes (default: os.cpu_count()).')
    p.add_argument('--strict', action='store_true',
                   help='Re-raise worker exceptions instead of capturing them.')
    return p


def _cli_main():
    import sys
    from rich.console import Console as _Console
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    console = _Console(log_path=False)

    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: --input-dir not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        print(f'ERROR: --results-csv not found: {results_csv}', file=sys.stderr)
        sys.exit(1)

    scalars = {}
    for token in args.scalars:
        if ':' not in token:
            print(f'ERROR: --scalar must be LABEL:column, got {token!r}',
                  file=sys.stderr)
            sys.exit(1)
        label, col = token.split(':', 1)
        scalars[label] = col

    # argparse delivers backslash-escape sequences as literal characters
    # (e.g. "\\t" rather than a real tab). Translate the common ones so the
    # user can type ``--sep "\t"`` or ``--lineterm "\r\n"`` on the command
    # line and have it interpreted as intended.
    def _unescape(s: str) -> str:
        return (s.replace('\\t', '\t')
                 .replace('\\r', '\r')
                 .replace('\\n', '\n'))

    sep = _unescape(args.sep)
    lineterm = _unescape(args.lineterm)

    applier = TlagApplier(
        input_dir=input_dir,
        output_dir=Path(args.output_dir),
        results_csv=results_csv,
        scalars=scalars,
        lag_column_template=args.lag_column_template,
        hz=args.hz,
        skiprows=args.skiprows,
        extra_rows=args.extra_rows,
        sep=sep,
        lineterm=lineterm,
        na_values=args.na_values,
        na_rep=args.na_rep,
        period_col=args.period_col,
        period_key_regex=args.period_key_regex,
        file_key_regex=args.file_key_regex,
        n_workers=args.n_workers,
        strict=args.strict,
    )

    # Need the total up-front for the progress bar; read once here.
    total = len(pd.read_csv(results_csv))

    msg = (f'PWB lag application  {total} files  '
           f'{applier.n_workers} workers  -> {args.output_dir}')
    console.print(f'\n[bold]{msg}[/bold]\n')

    def _fmt(row, gas):
        pfx = gas.lower()
        v = row.get(f'{pfx}_applied_lag_s')
        if v is None or v != v:
            return f'[dim]{gas}=--[/dim]'
        # Colour the lag value by plausibility, mirroring PWB's HDI colour
        # rule (green / yellow / red). 0–5 s is the typical tube-delay
        # range; > 5 s or negative lags are unusual but legal (PWBOPT
        # fallback). >= |10| s is the lag-max edge — suspect.
        lag_color = ('green' if 0.0 <= v <= 5.0
                     else ('yellow' if abs(v) < 10.0 else 'red'))
        return (f'{gas}=[bold]{v:.2f}s[/bold] '
                f'[{lag_color}]{v:+.2f}[/{lag_color}]')

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
        f'[cyan]{applier.n_workers} workers[/cyan]', total=total)

    with prog:
        def _cb(done, _total, row):
            period = row.get('period', '')
            period_short = Path(period).stem.split('_')[0]
            if row.get('status') == 'error':
                parts = f'[red]ERR[/red] {row.get("error", "")[:60]}'
            else:
                parts = '  '.join(_fmt(row, g) for g in scalars)
            console.log(f'[dim]{period_short}[/dim]  {parts}')
            prog.update(task_id, completed=done,
                        description=f'[cyan]{applier.n_workers} workers[/cyan]  '
                                    f'[dim]{period_short}[/dim]')

        summary = applier.run(on_progress=_cb)

    console.print(f'\n[green]Done — {len(summary)} files.[/green]')

    # Always write a summary CSV next to the output dir
    summary_csv = Path(args.output_dir) / 'apply_tlag_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print(f'Summary saved to: {summary_csv}')


if __name__ == '__main__':
    _cli_main()
