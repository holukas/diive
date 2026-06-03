"""
=================================================================
Apply Detected Time Lags to Raw EC Files (TlagApplier)
=================================================================

Demonstrates calling ``TlagApplier`` from the command line via::

    python -m diive.flux.hires.apply_tlag [options]

or the installed console-script alias::

    uv run diive-tlag-apply-batch [options]

This script:

1. Generates synthetic EddyPro-format raw files in a temporary input
   directory.
2. Runs ``diive-tlag-pwb-batch`` to detect time lags and write a
   ``tlag_results.csv``.
3. Runs ``diive-tlag-apply-batch`` to read that results CSV and shift the
   scalar columns of each raw file by the detected (per-period, per-gas) lag,
   writing aligned files to a separate output directory.
4. Prints a comparison of one raw file before and after alignment.

All temporary directories are removed at the end.

CLI usage reference
-------------------
.. code-block:: text

    python -m diive.flux.hires.apply_tlag \\
        --input-dir   /path/to/raw_files \\
        --output-dir  /path/to/aligned_files \\
        --results-csv /path/to/tlag_results.csv \\
        --scalar CH4:ch4 --scalar N2O:n2o \\
        --lag-column-template "{prefix}_tlag_final_pf_s" \\
        --hz 20 \\
        --skiprows 9 \\
        --na-rep -9999 \\
        --period-col period \\
        --n-workers 4

Example — bash (multi-line)
---------------------------
.. code-block:: text

    uv run diive-tlag-apply-batch \\
        --input-dir   "F:\\path\\to\\raw_files" \\
        --output-dir  "F:\\path\\to\\aligned_files" \\
        --results-csv "F:\\path\\to\\01_pwb_results\\tlag_results.csv" \\
        --scalar CH4:ch4 --scalar N2O:n2o \\
        --hz 20 --skiprows 9 --n-workers 16

Example — PowerShell / Windows (one-liner)
-------------------------------------------
.. code-block:: text

    uv run diive-tlag-apply-batch --input-dir "F:\\path\\to\\raw_files" --output-dir "F:\\path\\to\\aligned_files" --results-csv "F:\\path\\to\\01_pwb_results\\tlag_results.csv" --scalar CH4:ch4 --scalar N2O:n2o --hz 20 --skiprows 9 --n-workers 16

Real-world example (CH-Cha 2021 LGR-4 PWB results)
---------------------------------------------------
.. code-block:: text

    uv run diive-tlag-apply-batch `
        --input-dir   "F:\\CURRENT\\CHA_TIMELAG-COMPARISON_2021\\1-RAWDATA\\rotated" `
        --output-dir  "F:\\CURRENT\\CHA_TIMELAG-COMPARISON_2021\\2-FLUXRUN\\LGR-4_PWB\\02_aligned" `
        --results-csv "F:\\CURRENT\\CHA_TIMELAG-COMPARISON_2021\\2-FLUXRUN\\LGR-4_PWB\\01_pwb_results\\tlag_results.csv" `
        --scalar CH4:ch4 --scalar N2O:n2o `
        --hz 20 --skiprows 9 --n-workers 16

Sign convention
---------------
Positive ``tlag_s`` means the scalar arrives later than the wind (tube
delay). The alignment shifts the scalar column **backward** by
``round(tlag_s * hz)`` rows so that, at each row index ``t``, the scalar
value corresponds to the air parcel sampled by the sonic anemometer at
``t``. The trailing rows of the shifted column become NaN and are written
as ``-9999`` (EddyPro NA convention).

By default the alignment uses the ``{prefix}_tlag_final_pf_s`` column from
the results CSV: the pre-filtered, gap-filled PWBOPT lag. Override with
``--lag-column-template`` to use a different column (e.g.
``{prefix}_tlag_final_s`` for the standard PWBOPT).
"""

# %%
# All executable code is inside the ``if __name__ == '__main__':`` guard
# because the module CLI uses ``ProcessPoolExecutor`` (Windows spawn).

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    HZ = 20
    N_PERIODS = 4
    RECORDS = HZ * 60 * 30  # 30 min at 20 Hz
    LAG_TRUE_S = 1.5
    LAG_TRUE_RECORDS = int(LAG_TRUE_S * HZ)
    N_BOOTSTRAP = 9  # 99 in production
    N_WORKERS = 4

    np.random.seed(42)

    input_dir = tempfile.mkdtemp(prefix='apply_tlag_input_')
    pwb_dir = tempfile.mkdtemp(prefix='apply_tlag_pwb_')
    aligned_dir = tempfile.mkdtemp(prefix='apply_tlag_out_')

    # ------------------------------------------------------------------
    # 1. Generate synthetic EddyPro rotated files
    #    Column order: u(0) v(1) w(2) ts(3) co2(4) h2o(5) ch4(6) n2o(7) air_t(8) air_p(9)
    # ------------------------------------------------------------------
    for i in range(N_PERIODS):
        w = np.zeros(RECORDS)
        for t in range(1, RECORDS):
            w[t] = 0.8 * w[t - 1] + np.random.normal(0, 0.3)
        ts = w * 0.6 + np.random.normal(0, 0.15, RECORDS)
        ch4 = np.roll(w, LAG_TRUE_RECORDS) * 4.0 + np.random.normal(0, 0.3, RECORDS)
        ch4[:LAG_TRUE_RECORDS] = ch4[LAG_TRUE_RECORDS]
        n2o = np.roll(w, LAG_TRUE_RECORDS) * 0.4 + np.random.normal(0, 0.6, RECORDS)
        n2o[:LAG_TRUE_RECORDS] = n2o[LAG_TRUE_RECORDS]
        u = np.random.normal(2.0, 0.5, RECORDS)
        v = np.random.normal(0.1, 0.3, RECORDS)
        co2 = np.random.normal(400.0, 5.0, RECORDS)
        h2o = np.random.normal(10.0, 0.5, RECORDS)
        air_t = np.random.normal(15.0, 2.0, RECORDS)
        air_p = np.random.normal(1013.0, 1.0, RECORDS)

        path = Path(input_dir) / f'period_{i:04d}.txt'
        with open(path, 'w') as fh:
            for j in range(9):  # 9 metadata rows (EddyPro convention)
                fh.write(f'metadata_line_{j}\n')
            fh.write('u v w ts co2 h2o ch4 n2o air_t air_p\n')
            fh.write(pd.DataFrame({
                'u': u, 'v': v, 'w': w, 'ts': ts,
                'co2': co2, 'h2o': h2o, 'ch4': ch4, 'n2o': n2o,
                'air_t': air_t, 'air_p': air_p,
            }).to_csv(sep=' ', index=False, header=False))

    print(f'Input    : {input_dir}  ({N_PERIODS} files)')
    print(f'PWB out  : {pwb_dir}')
    print(f'Aligned  : {aligned_dir}')

    # Inherit PYTHONWARNINGS to suppress runpy double-import warnings in workers.
    _env = {**os.environ, 'PYTHONWARNINGS': 'ignore::RuntimeWarning:runpy'}

    # ------------------------------------------------------------------
    # 2. Run the PWB batch CLI to produce tlag_results.csv
    # ------------------------------------------------------------------
    pwb_cmd = [
        sys.executable, '-m', 'diive.flux.hires.lag_pwb',
        '--input-dir', input_dir,
        '--output-dir', pwb_dir,
        '--scalar', 'CH4:ch4',
        '--scalar', 'N2O:n2o',
        '--col-w', 'w',
        '--col-tsonic', 'ts',
        '--usecols', '0', '1', '2', '3', '6', '7',
        '--col-names', 'u', 'v', 'w', 'ts', 'ch4', 'n2o',
        '--skiprows', '9',
        '--hz', str(HZ),
        '--lag-max', '10.0',
        '--n-bootstrap', str(N_BOOTSTRAP),
        '--n-workers', str(N_WORKERS),
        '--hdi-thresh', '0.5',
        '--dev-thresh', '0.5',
        '--hdi-prefilter', '1.0',
    ]
    print('\n' + '=' * 60)
    print('STEP 1/2: PWB time-lag detection')
    print('=' * 60, flush=True)
    pwb_res = subprocess.run(pwb_cmd, env=_env, check=False)
    if pwb_res.returncode != 0:
        print(f'\nPWB CLI exited with code {pwb_res.returncode}. Aborting.')
        sys.exit(1)

    results_csv = Path(pwb_dir) / 'tlag_results.csv'
    if not results_csv.exists():
        print(f'\ntlag_results.csv not found at {results_csv}. Aborting.')
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Run the apply-tlag CLI to write aligned files
    # ------------------------------------------------------------------
    # NOTE: the apply step reads the *original* EddyPro file header (the
    # column-name row at line ``skiprows``) verbatim — it does NOT use the
    # PWB step's ``--col-names`` renaming. So ``--scalar LABEL:column`` must
    # name the column as it appears literally in the file. Here the
    # synthetic file already uses ``n2o`` and ``ch4`` directly, mirroring
    # how most real EddyPro rotated files name their gas slots.
    apply_cmd = [
        sys.executable, '-m', 'diive.flux.hires.apply_tlag',
        '--input-dir', input_dir,
        '--output-dir', aligned_dir,
        '--results-csv', str(results_csv),
        '--scalar', 'CH4:ch4',
        '--scalar', 'N2O:n2o',
        '--hz', str(HZ),
        '--skiprows', '9',
        '--n-workers', str(N_WORKERS),
    ]

    print('\n' + '=' * 60)
    print('STEP 2/2: Apply detected lags to raw files')
    print('=' * 60, flush=True)

    # Print a readable multi-line representation of the CLI we are about to
    # run: keep each ``--flag VALUE`` pair on a single line (so the output is
    # copy-pasteable), and put repeated ``--scalar`` pairs on their own line.
    def _format_cli(parts: list[str]) -> str:
        pieces: list[str] = []
        i = 0
        while i < len(parts):
            tok = parts[i]
            if tok.startswith('--') and i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                # Group consecutive non-flag tokens with the flag (handles
                # multi-value flags like ``--usecols 0 1 2 3``).
                grp = [tok]
                j = i + 1
                while j < len(parts) and not parts[j].startswith('--'):
                    grp.append(parts[j])
                    j += 1
                pieces.append(' '.join(grp))
                i = j
            else:
                pieces.append(tok)
                i += 1
        return ' \\\n    '.join(pieces)

    args_str = _format_cli(apply_cmd[2:])
    print(f'\nCLI command:\n  python -m {args_str}\n')

    apply_res = subprocess.run(apply_cmd, env=_env, check=False)

    # ------------------------------------------------------------------
    # 4. Inspect one aligned file to confirm the shift was applied
    # ------------------------------------------------------------------
    if apply_res.returncode == 0:
        sample_in = Path(input_dir) / 'period_0000.txt'
        sample_out = Path(aligned_dir) / 'period_0000.txt'
        if sample_out.exists():
            cols = ['u', 'v', 'w', 'ts', 'co2', 'h2o',
                    'ch4', 'n2o', 'air_t', 'air_p']
            raw = pd.read_csv(sample_in, skiprows=10, sep=r'\s+', header=None,
                              names=cols, na_values=['-9999', '-9999.0'])
            ali = pd.read_csv(sample_out, skiprows=10, sep=r'\s+', header=None,
                              names=cols, na_values=['-9999', '-9999.0'])
            print(f'\nSample file: {sample_out.name}')
            print(f'  Rows in : {len(raw)}, Rows out: {len(ali)}')
            print(f'  Trailing NaN in ch4 (aligned): {ali["ch4"].isna().sum()} rows')
            print(f'  Trailing NaN in n2o (aligned): {ali["n2o"].isna().sum()} rows')
            unshifted = [c for c in cols if c not in ('ch4', 'n2o')]
            print('  Non-shifted columns identical to input:',
                  all(
                      raw[c].fillna(-9999).round(6).equals(
                          ali[c].fillna(-9999).round(6))
                      for c in unshifted
                  ))
            corr_before_ch4 = raw[['w', 'ch4']].dropna().corr().iloc[0, 1]
            corr_after_ch4 = ali[['w', 'ch4']].dropna().corr().iloc[0, 1]
            corr_before_n2o = raw[['w', 'n2o']].dropna().corr().iloc[0, 1]
            corr_after_n2o = ali[['w', 'n2o']].dropna().corr().iloc[0, 1]
            print(f'  corr(w, ch4) before alignment: {corr_before_ch4:+.3f}')
            print(f'  corr(w, ch4) after  alignment: {corr_after_ch4:+.3f}'
                  ' (should be markedly higher)')
            print(f'  corr(w, n2o) before alignment: {corr_before_n2o:+.3f}')
            print(f'  corr(w, n2o) after  alignment: {corr_after_n2o:+.3f}'
                  ' (should be markedly higher)')

        summary_csv = Path(aligned_dir) / 'apply_tlag_summary.csv'
        if summary_csv.exists():
            summary = pd.read_csv(summary_csv)
            print(f'\nSummary  : {summary_csv}')
            print(f'  Shape  : {summary.shape[0]} rows x {summary.shape[1]} columns')
            print(summary[['period', 'status',
                           'ch4_applied_lag_s', 'n2o_applied_lag_s']]
                  .to_string(index=False))
    else:
        print(f'\nApply CLI exited with code {apply_res.returncode}.')

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    import shutil

    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(pwb_dir, ignore_errors=True)
    shutil.rmtree(aligned_dir, ignore_errors=True)
    print('\nTemporary directories removed.')
    print('[OK] CLI example complete.')
