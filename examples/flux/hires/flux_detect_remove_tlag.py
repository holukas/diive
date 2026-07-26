"""
=============================================================
Per-Chunk Time Lag Detection and Removal (PerFilePipeline)
=============================================================

Raw eddy-covariance files often cover several hours, but flux processing and
PWB time-lag detection both work on **30-minute averaging intervals**. Wind
rotation angles drift over hours and so does the tube delay, so one rotation
and one lag estimate for a 6-hour file is the wrong granularity.

``PerFilePipeline`` (CLI: ``diive-tlag-pwb-detect-remove``) reads each long
raw file once and processes it in fixed-length chunks, in two phases:

**Phase 1 — detect (every chunk):** read the chunk's rows, apply double
rotation to (u, v, w) *in memory*, and run ``PreWhiteningBootstrap`` on the
rotated W against each scalar. Nothing is written; the rotated data never
touch the disk.

**Phase 2 — remove:** PWBOPT (S1/S2/S3) picks the best lag per chunk from the
whole chunk sequence, then each scalar is shifted by ``round(tlag_s * hz)``
records and one file per chunk is written.

The split matters: the lag actually removed is the PWBOPT-optimised lag, not
the raw per-chunk detection. A wide-HDI chunk's mode lag can be spurious, so
what gets applied is the pre-filtered, gap-filled column
(``{prefix}_tlag_final_pf_s``) — the same one ``TlagApplier`` uses.

This example generates two synthetic 1-hour files with a known tube delay per
gas, runs the pipeline over them, and shows what came back. Temporary
directories are removed at the end.

Wall-clock grid alignment
-------------------------
30-minute flux files must sit on the wall-clock grid (:00 / :30) so downstream
software bins them by clock time. Given ``start_time_regex`` /
``start_time_format``, chunk boundaries snap to that grid: a file starting at
10:10 yields a short leading chunk (10:10 -> 10:30), then full chunks on the
grid (10:30, 11:00, ...). A leading partial shorter than
``min_chunk_seconds`` is skipped.

Per-gas search windows
----------------------
Each gas can get its own time-lag search window. A positive-only window such
as ``[0, 5]`` keeps only physical tube-delay lags, while a long-inlet gas like
H2O can use a wider window than the dry gases in the same run — something the
single uniform lag setting in EddyPro cannot express. Keep the expected lag
mid-window: a detection pinned at a window boundary is treated as a failed
detection and is never applied.

Equivalent CLI call
-------------------
.. code-block:: text

    uv run diive-tlag-pwb-detect-remove \\
        --input-dir  /path/to/raw_files \\
        --output-dir /path/to/aligned \\
        --col-u u --col-v v --col-w w --col-tsonic ts \\
        --scalar CH4:ch4 \\
        --scalar "H2O:h2o@lag=30;lws=0;uws=25" \\
        --hz 20 \\
        --chunk-seconds 1800 \\
        --min-chunk-seconds 300 \\
        --lag-max 10.0 \\
        --n-bootstrap 99 \\
        --file-pattern "*.csv" \\
        --skiprows 0 --extra-rows 2 --sep "," \\
        --start-time-regex "(\\d{8})_(\\d{4})" \\
        --start-time-format "%Y%m%d%H%M" \\
        --chunk-name-template "CH-CHA_{starttime}{suffix}" \\
        --n-workers 4 --save-plots

Run ``uv run diive-tlag-pwb-detect-remove --help`` for every flag, or drive
the same pipeline from a terminal UI with
``uv run diive-tlag-pwb-detect-remove-tui`` (``--demo`` previews the interface
without any data).

Downstream
----------
The lag has already been corrected here, so flux processing over
``2_lag_removed/`` must run with **EC time-lag maximization disabled**.

See Also
--------
- ``flux_lag_pwb.py`` — the PWB algorithm on a single averaging period.
- ``flux_lag_pwb_batch.py`` — detection only, across many 30-min files.
- ``flux_apply_tlag_cli.py`` — applying lags from an existing results CSV.
"""

# %%
# All executable code sits inside the ``if __name__ == '__main__':`` guard
# because the pipeline uses ``ProcessPoolExecutor`` (Windows spawn).

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from diive.flux.hires.detect_and_remove_tlag import PerFilePipeline

if __name__ == '__main__':

    # %%
    # Settings
    # --------
    # Two 1-hour files at 20 Hz, split into 30-minute chunks -> 4 chunks total.
    # ``N_BOOTSTRAP`` is kept small so the example runs quickly; use 99 in
    # production.
    HZ = 20
    FILE_SECONDS = 3600
    CHUNK_SECONDS = 1800
    RECORDS = HZ * FILE_SECONDS
    N_BOOTSTRAP = 9
    CH4_LAG_S = 1.5   # short tube, dry gas
    H2O_LAG_S = 12.0  # long inlet, beyond the default +/-10 s window

    # %%
    # Generate synthetic raw files
    # ----------------------------
    # Each scalar is the rotated vertical wind delayed by its own tube delay
    # plus noise, which is what the cross-correlation has to recover. The
    # filenames carry their start time (``YYYYMMDD_HHMM``) so chunks can be
    # named by wall-clock time.
    rng = np.random.default_rng(42)
    input_dir = Path(tempfile.mkdtemp(prefix='tlag_input_'))
    output_dir = Path(tempfile.mkdtemp(prefix='tlag_output_'))

    for start in ('20240715_1000', '20240715_1100'):
        # Autocorrelated vertical wind, as in real turbulence.
        w = np.zeros(RECORDS)
        for t in range(1, RECORDS):
            w[t] = 0.8 * w[t - 1] + rng.normal(0, 0.3)

        def delayed(signal, lag_s, scale, noise):
            """Signal delayed by lag_s seconds (positive = arrives later)."""
            k = int(round(lag_s * HZ))
            out = np.r_[np.zeros(k), signal[:-k]] * scale
            return out + rng.normal(0, noise, RECORDS)

        df = pd.DataFrame({
            'u': rng.normal(2.0, 0.5, RECORDS),
            'v': rng.normal(0.1, 0.3, RECORDS),
            'w': w,
            'ts': w * 0.6 + rng.normal(0, 0.15, RECORDS),
            'ch4': delayed(w, CH4_LAG_S, 2.0, 0.4),
            'h2o': delayed(w, H2O_LAG_S, 3.0, 0.4),
        })

        # Raw CSV layout: header row + units row + source row, then data.
        path = input_dir / f'CH-CHA_{start}.csv'
        with open(path, 'w', newline='') as fh:
            fh.write(','.join(df.columns) + '\n')
            fh.write('m/s,m/s,m/s,degC,ppb,ppt\n')
            fh.write('sonic,sonic,sonic,sonic,lgr,lgr\n')
            df.to_csv(fh, index=False, header=False, lineterminator='\n')

    print(f'Input : {input_dir}  (2 files, {FILE_SECONDS // 60} min each)')
    print(f'Output: {output_dir}')

    # %%
    # Run the pipeline
    # ----------------
    # ``per_gas_lag`` gives H2O its own wider, positive-only window; CH4 keeps
    # the global +/-10 s. Without it, H2O's 12 s delay would fall outside the
    # search window and could not be detected.
    pipeline = PerFilePipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        col_u='u', col_v='v', col_w='w', col_tsonic='ts',
        scalars={'CH4': 'ch4', 'H2O': 'h2o'},
        hz=HZ,
        chunk_seconds=CHUNK_SECONDS,
        min_chunk_seconds=300,
        lag_max_s=10.0,
        n_bootstrap=N_BOOTSTRAP,
        per_gas_lag={'H2O': {'lag_max_s': 20.0, 'lws': 0.0, 'uws': 20.0}},
        file_pattern='*.csv',
        skiprows=0,
        extra_rows=2,
        sep=',',
        # Name each output chunk by its own wall-clock start time. The regex's
        # capture groups are *concatenated* before parsing, so the format spec
        # describes the joined text ('202407151000') and carries no separator.
        start_time_regex=r'(\d{8})_(\d{4})',
        start_time_format='%Y%m%d%H%M',
        chunk_name_template='CH-CHA_{starttime}{suffix}',
        n_workers=2,
        random_state=42,
    )
    summary = pipeline.run()

    # %%
    # Detected lags per chunk
    # -----------------------
    # ``*_tlag_s`` is the raw PWB detection, ``*_hdi_range_s`` its 95% HDI
    # width (the S1 reliability criterion, < 0.5 s = reliable), and
    # ``*_applied_records`` the records actually shifted — derived from the
    # PWBOPT lag, so it can differ from the raw detection.
    # 'period' is the output chunk's filename, mirroring the 'period' column of
    # tlag_results.csv from diive-tlag-pwb-batch.
    cols = ['parent', 'chunk_index', 'period',
            'ch4_tlag_s', 'ch4_hdi_range_s', 'ch4_applied_records',
            'h2o_tlag_s', 'h2o_hdi_range_s', 'h2o_applied_records']
    print('\nPer-chunk results:')
    print(summary[cols].to_string(index=False))

    print(f'\nTrue lags: CH4 = {CH4_LAG_S} s '
          f'({int(CH4_LAG_S * HZ)} records), '
          f'H2O = {H2O_LAG_S} s ({int(H2O_LAG_S * HZ)} records)')

    # %%
    # Output layout
    # -------------
    # ``2_lag_removed/`` holds only the lag-corrected chunk files, so it can be
    # handed straight to the next flux-processing step as its input directory.
    # Diagnostics and the summary CSV live under ``1_lag_detection/``.
    print('\nOutput layout:')
    for item in sorted(output_dir.rglob('*')):
        if item.is_file():
            print(f'  {item.relative_to(output_dir)}')

    written = sorted((output_dir / '2_lag_removed').glob('*.csv'))
    print(f'\n{len(written)} lag-corrected chunk files written, '
          f'named by wall-clock start time.')

    # %%
    # What changed in the data
    # ------------------------
    # The scalar columns moved; every other column passed through untouched.
    # The rows shifted off the end become the missing-value sentinel (-9999),
    # so the corrected file stays a drop-in replacement for the original.
    # Three preserved header lines (names + units + source), then the data.
    chunk = pd.read_csv(written[0], skiprows=3, header=None,
                        names=['u', 'v', 'w', 'ts', 'ch4', 'h2o'])
    n_missing = int(chunk['ch4'].eq(-9999).sum())
    print(f'\nFirst chunk: {len(chunk)} rows, '
          f'{n_missing} trailing CH4 records set to -9999 by the shift '
          f'(= the {int(CH4_LAG_S * HZ)}-record lag that was removed).')

    # %%
    # Cleanup
    # -------
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    print('\nTemporary directories removed.')
    print('[OK] Detect-and-remove example complete.')
