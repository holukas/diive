"""
=======================================================
Batch Time Lag Detection via CLI (PwbBatchDetection)
=======================================================

Demonstrates calling ``PwbBatchDetection`` from the command line via::

    python -m diive.flux.hires.lag_pwb [options]

This script creates synthetic EddyPro-format files in a temporary input
directory, invokes the module CLI as a subprocess, and prints the generated
results CSV.  The temporary directories are removed at the end.

CLI usage reference
-------------------
.. code-block:: text

    python -m diive.flux.hires.lag_pwb \\
        --input-dir  /path/to/hires_files \\
        --output-dir /path/to/results \\
        --scalar CH4:ch4 --scalar N2O:n2o \\
        --col-w w \\
        --col-tsonic ts \\
        --usecols 0 1 2 3 6 7 \\
        --col-names u v w ts ch4 n2o \\
        --skiprows 9 \\
        --hz 20 \\
        --lag-max 10.0 \\
        --n-bootstrap 99 \\
        --block-length 20.0 \\
        --min-valid-frac 0.3 \\
        --hdi-thresh 0.5 \\
        --dev-thresh 0.5 \\
        --hdi-prefilter 1.0 \\
        --n-workers 4 \\
        --file-date-format %Y%m%d-%H%M \\
        --save-plots

``--file-date-format`` is optional: when given (here for filenames like
``20210820-0930_rotated.txt``) the results gain a ``timestamp`` column and the
summary plots use real dates instead of the period index.

Example with uv alias (EddyPro rotated files, all parameters explicit)
-----------------------------------------------------------------------
.. code-block:: text

    uv run diive-tlag-pwb-batch \\
        --input-dir  "F:\\Sync\\luhk_work\\dev-data\\datasets-data\\dataset_ch-cha_flux_product-data\\docs\\notebooks\\05_TIME_LAG_COMPARISON\\input\\test_input" \\
        --output-dir "F:\\Sync\\luhk_work\\dev-data\\datasets-data\\dataset_ch-cha_flux_product-data\\docs\\notebooks\\05_TIME_LAG_COMPARISON\\input\\test_output" \\
        --scalar CH4:ch4 --scalar N2O:n2o \\
        --col-w w --col-tsonic ts \\
        --usecols 0 1 2 3 6 7 \\
        --col-names u v w ts ch4 n2o \\
        --skiprows 9 --hz 20 --lag-max 10.0 \\
        --n-bootstrap 9 --block-length 20.0 --min-valid-frac 0.3 \\
        --hdi-thresh 0.5 --dev-thresh 0.5 --hdi-prefilter 1.0 \\
        --n-workers 16 --save-plots

One-liner for PowerShell / Windows (no line continuation needed)
-----------------------------------------------------------------
.. code-block:: text

    uv run diive-tlag-pwb-batch --input-dir "F:\Sync\luhk_work\dev-data\datasets-data\dataset_ch-cha_flux_product-data\docs\notebooks\05_TIME_LAG_COMPARISON\input\test_input" --output-dir "F:\Sync\luhk_work\dev-data\datasets-data\dataset_ch-cha_flux_product-data\docs\notebooks\05_TIME_LAG_COMPARISON\input\test_output" --scalar CH4:ch4 --scalar N2O:n2o --col-w w --col-tsonic ts --usecols 0 1 2 3 6 7 --col-names u v w ts ch4 n2o --skiprows 9 --hz 20 --lag-max 10.0 --n-bootstrap 99 --block-length 20.0 --min-valid-frac 0.3 --hdi-thresh 0.5 --dev-thresh 0.5 --hdi-prefilter 1.0 --n-workers 16 --save-plots

Column layout for EddyPro rotated files (10 columns, 0-indexed)
----------------------------------------------------------------
.. code-block:: text

    0:u  1:v  2:w  3:ts  4:co2  5:h2o  6:ch4  7:4th_gas  8:air_t  9:air_p

    --usecols 0 1 2 3 6 7 selects u, v, w, ts, ch4, 4th_gas
    --col-names renames them to: u v w ts ch4 n2o

Short alias (after ``uv sync`` or ``pip install diive``)
---------------------------------------------------------
.. code-block:: text

    uv run diive-tlag-pwb-batch --input-dir ... --output-dir ... [options]
    uv run diive-tlag-pwb-batch --help

``diive-tlag-pwb-batch`` is a console-script entry point defined in
``pyproject.toml`` that calls the same ``_cli_main()`` function as
``python -m diive.flux.hires.lag_pwb``.

Run ``uv run diive-tlag-pwb-batch --help`` for all options.
"""

# %%
# All executable code is inside the ``if __name__ == '__main__':`` guard
# because the module CLI itself uses ``ProcessPoolExecutor`` (Windows spawn).

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
    N_PERIODS = 8
    RECORDS = HZ * 60 * 30  # 30 min at 20 Hz
    LAG_TRUE_S = 1.5
    LAG_TRUE_RECORDS = int(LAG_TRUE_S * HZ)
    N_BOOTSTRAP = 9  # 99 in production
    N_WORKERS = 4

    # ------------------------------------------------------------------
    # Generate synthetic files in a temporary input directory
    # ------------------------------------------------------------------
    # Full column order mirrors real EddyPro rotated files:
    #   u(0) v(1) w(2) ts(3) co2(4) h2o(5) ch4(6) 4th_gas(7) air_t(8) air_p(9)
    np.random.seed(42)
    flux_strengths = np.linspace(5.0, 0.2, N_PERIODS)

    input_dir = tempfile.mkdtemp(prefix='pwb_input_')
    output_dir = tempfile.mkdtemp(prefix='pwb_output_')

    for i in range(N_PERIODS):
        w = np.zeros(RECORDS)
        for t in range(1, RECORDS):
            w[t] = 0.8 * w[t - 1] + np.random.normal(0, 0.3)
        ts = w * 0.6 + np.random.normal(0, 0.15, RECORDS)
        noise_std = 0.5 + (1.0 - flux_strengths[i] / 5.0) * 0.8
        ch4 = np.roll(w, LAG_TRUE_RECORDS) * flux_strengths[i] + np.random.normal(0, noise_std, RECORDS)
        ch4[:LAG_TRUE_RECORDS] = ch4[LAG_TRUE_RECORDS]
        n2o = np.roll(w, LAG_TRUE_RECORDS) * flux_strengths[i] * 0.1 + np.random.normal(0, 1.0, RECORDS)
        n2o[:LAG_TRUE_RECORDS] = n2o[LAG_TRUE_RECORDS]
        u = np.random.normal(2.0, 0.5, RECORDS)
        v = np.random.normal(0.1, 0.3, RECORDS)
        co2 = np.random.normal(400.0, 5.0, RECORDS)
        h2o = np.random.normal(10.0, 0.5, RECORDS)
        air_t = np.random.normal(15.0, 2.0, RECORDS)
        air_p = np.random.normal(1013.0, 1.0, RECORDS)

        path = Path(input_dir) / f'period_{i:04d}.txt'
        with open(path, 'w') as fh:
            for j in range(9):  # 9 metadata rows
                fh.write(f'metadata_line_{j}\n')
            fh.write('u v w ts co2 h2o ch4 4th_gas air_t air_p\n')  # header row
            fh.write(pd.DataFrame({
                'u': u, 'v': v, 'w': w, 'ts': ts,
                'co2': co2, 'h2o': h2o, 'ch4': ch4, '4th_gas': n2o,
                'air_t': air_t, 'air_p': air_p,
            }).to_csv(sep=' ', index=False, header=False))

    # ------------------------------------------------------------------
    # Build the CLI command
    # ------------------------------------------------------------------
    cmd = [
        sys.executable, '-m', 'diive.flux.hires.lag_pwb',
        '--input-dir', input_dir,
        '--output-dir', output_dir,
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

    print(f'Input  : {input_dir}  ({N_PERIODS} files)')
    print(f'Output : {output_dir}')
    print('\nCLI command:')
    # Format as a readable multi-line shell command (skip sys.executable + -m)
    args_str = ' \\\n    '.join(cmd[2:])
    print(f'  python -m {args_str}')

    # ------------------------------------------------------------------
    # Run the CLI
    # ------------------------------------------------------------------
    # Suppress the runpy double-import warning that appears when diive.__init__
    # pre-imports lag_pwb and then -m re-executes it as __main__.
    # PYTHONWARNINGS is inherited by all child processes (including workers).
    _env = {**os.environ, 'PYTHONWARNINGS': 'ignore::RuntimeWarning:runpy'}

    print('\n' + '=' * 60, flush=True)
    result = subprocess.run(cmd, env=_env, check=False)
    print('=' * 60)

    if result.returncode != 0:
        print(f'\nCLI exited with code {result.returncode}.')
    else:
        # ------------------------------------------------------------------
        # Read and display the results CSV written by the CLI
        # ------------------------------------------------------------------
        csv_path = Path(output_dir) / 'tlag_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f'\nResults CSV: {csv_path}')
            print(f'Shape      : {df.shape[0]} rows x {df.shape[1]} columns')
            print(f'\nColumns    : {list(df.columns)}')
            print(f'\nFirst rows :')
            print(df[['period', 'ch4_tlag_s', 'ch4_hdi_range_s',
                      'n2o_tlag_s', 'n2o_hdi_range_s']].to_string(index=False))
        else:
            print('Results CSV not found — check CLI output above.')

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    import shutil

    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    print('\nTemporary directories removed.')
    print('[OK] CLI example complete.')


