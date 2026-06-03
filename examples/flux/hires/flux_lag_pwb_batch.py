"""
=========================================================
Batch Time Lag Detection (PwbBatchDetection, batch mode)
=========================================================

Demonstrate ``PwbBatchDetection`` across multiple averaging-period files
using synthetic data.  Each period is written as an EddyPro-format text file
in a temporary directory; the directory is removed automatically at the end.
No results or plots are saved to disk.

``PwbBatchDetection`` wraps ``PreWhiteningBootstrap`` in a multiprocessing
worker and distributes files across CPU cores.  PWBOPT post-processing
(S1/S2/S3 flag assignment and optional HDI pre-filter) is available as static
methods on the class.

Windows requirement: ``ProcessPoolExecutor`` uses the *spawn* start method on
Windows, so the entry point of any script using ``PwbBatchDetection`` must be
protected with ``if __name__ == '__main__':``, as shown below.  All setup and
execution code lives inside the guard; this also prevents worker subprocesses
from re-running setup code when they are spawned.

Reference: Vitale D et al. (2024), Environmental and Ecological Statistics
31:219-244. doi:10.1007/s10651-024-00615-9
"""

# %%
# All code runs inside the ``if __name__ == '__main__':`` guard.
# Worker processes import this module but never enter the guard, so the
# synthetic-data generation and temp-file writing happen only once in the
# main process.

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

import diive as dv

console = Console()

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    HZ = 20  # sampling frequency (Hz)
    N_PERIODS = 12  # number of averaging periods (files)
    N_WORKERS = 4  # parallel worker processes (adjust to available cores)
    RECORDS = HZ * 60 * 30
    LAG_TRUE_S = 1.5  # known lag embedded in each synthetic period
    LAG_TRUE_RECORDS = int(LAG_TRUE_S * HZ)
    N_BOOTSTRAP = 9  # bootstrap replicates (99 in production)

    # File format (matches the EddyPro rotated-file layout written below)
    # Full column order: u(0) v(1) w(2) ts(3) co2(4) h2o(5) ch4(6) 4th_gas(7) air_t(8) air_p(9)
    # Select only the columns needed for detection (skip co2, h2o, air_t, air_p).
    FILE_USECOLS = [0, 1, 2, 3, 6, 7]
    FILE_COL_NAMES = ['u', 'v', 'w', 'ts', 'ch4', 'n2o']  # rename col 7 ("4th gas") to n2o
    FILE_SKIPROWS = 9  # 9 metadata rows + 1 header row = 10 rows skipped total

    # ------------------------------------------------------------------
    # Generate synthetic averaging periods
    # ------------------------------------------------------------------
    # Flux strength decreases from strong (CH4 and N2O both easy) to near-zero
    # (N2O unreliable, mimicking a trace-gas low-flux scenario).
    np.random.seed(42)
    flux_strengths = np.linspace(5.0, 0.05, N_PERIODS)

    synthetic_dfs = []
    for p in range(N_PERIODS):
        w = np.zeros(RECORDS)
        for t in range(1, RECORDS):
            w[t] = 0.8 * w[t - 1] + np.random.normal(0, 0.3)

        ts = w * 0.6 + np.random.normal(0, 0.15, RECORDS)

        noise_std = 0.5 + (1.0 - flux_strengths[p] / 5.0) * 0.8
        ch4 = (np.roll(w, LAG_TRUE_RECORDS) * flux_strengths[p]
               + np.random.normal(0, noise_std, RECORDS))
        ch4[:LAG_TRUE_RECORDS] = ch4[LAG_TRUE_RECORDS]

        # N2O: 10x weaker signal (trace-gas scenario; T_SONIC combinations help)
        n2o = (np.roll(w, LAG_TRUE_RECORDS) * flux_strengths[p] * 0.1
               + np.random.normal(0, 1.0, RECORDS))
        n2o[:LAG_TRUE_RECORDS] = n2o[LAG_TRUE_RECORDS]

        u = np.random.normal(2.0, 0.5, RECORDS)
        v = np.random.normal(0.1, 0.3, RECORDS)
        # Filler columns present in real EddyPro rotated files but not used here
        co2 = np.random.normal(400.0, 5.0, RECORDS)
        h2o = np.random.normal(10.0, 0.5, RECORDS)
        air_t = np.random.normal(15.0, 2.0, RECORDS)
        air_p = np.random.normal(1013.0, 1.0, RECORDS)

        synthetic_dfs.append(
            pd.DataFrame({
                'u': u, 'v': v, 'w': w, 'ts': ts,
                'co2': co2, 'h2o': h2o, 'ch4': ch4, '4th_gas': n2o,
                'air_t': air_t, 'air_p': air_p,
            })
        )

    print(f"Synthetic periods: {N_PERIODS}")
    print(f"  Records/period : {RECORDS} ({RECORDS / HZ / 60:.0f} min at {HZ} Hz)")
    print(f"  True lag       : {LAG_TRUE_S} s ({LAG_TRUE_RECORDS} records)")
    print(f"  Flux strengths : {flux_strengths[0]:.2f} -> {flux_strengths[-1]:.3f}")

    # ------------------------------------------------------------------
    # Write synthetic files to a temporary directory
    # ------------------------------------------------------------------
    # Each period is stored as a whitespace-separated text file with 9 dummy
    # metadata rows and one header row (10 rows total to skip), matching the
    # EddyPro rotated-file format used in real-data workflows.
    # TemporaryDirectory cleans up automatically when cleanup() is called.
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp_dir_obj.name)

    # Filenames carry a leading datetime token (yyyymmdd-HHMM) for consecutive
    # 30-minute periods; file_date_format lets PwbBatchDetection parse it into a
    # 'timestamp' column and use real dates as the summary-plot x-axis.
    base_time = pd.Timestamp('2021-08-20 09:30')
    file_paths = []
    for i, df_period in enumerate(synthetic_dfs):
        stamp = (base_time + i * pd.Timedelta(minutes=30)).strftime('%Y%m%d-%H%M')
        path = tmp_dir / f'{stamp}_rotated.txt'
        with open(path, 'w') as fh:
            for j in range(FILE_SKIPROWS):
                fh.write(f'metadata_line_{j}\n')
            fh.write('u v w ts co2 h2o ch4 4th_gas air_t air_p\n')  # header (skipped by skiprows+1)
            fh.write(df_period.to_csv(sep=' ', index=False, header=False))
        file_paths.append(path)

    print(f"\nTemporary files written : {len(file_paths)}")
    print(f"  Directory            : {tmp_dir}")

    # ------------------------------------------------------------------
    # Run batch detection
    # ------------------------------------------------------------------
    det = dv.flux.PwbBatchDetection(
        files=file_paths,
        scalars={'CH4': 'ch4', 'N2O': 'n2o'},
        col_w='w',
        col_tsonic='ts',  # enables 4-combination RFlux v3.2.0 logic
        hz=HZ,
        lag_max_s=10.0,  # CCF search half-width (s)
        n_bootstrap=N_BOOTSTRAP,  # 99 in production
        block_length_s=20.0,  # bootstrap block length (s)
        usecols=FILE_USECOLS,  # [0,1,2,3,6,7] -> u,v,w,ts,ch4,4th_gas
        col_names=FILE_COL_NAMES,  # rename 4th_gas -> n2o after loading
        skiprows=FILE_SKIPROWS,
        min_valid_frac=0.3,
        output_dir=None,  # no checkpoint CSV written
        save_plots=False,  # no PNG files written
        n_workers=N_WORKERS,
        random_state=42,  # reproducible regardless of worker completion order
        file_date_format='%Y%m%d-%H%M',  # parse 'timestamp' from the filename
    )

    console.print(f"\n[bold]Batch PWB detection[/bold]  "
                  f"[dim]{len(file_paths)} files · {det.n_workers} workers[/dim]")

    # Live display: growing results table + overall progress bar.
    # _live_rows accumulates one entry per completed file so the table
    # grows in real time as workers finish.
    _live_rows: list[dict] = []
    _n_active = [0]  # mutable counter: files currently being processed


    def _make_display(progress_bar: Progress) -> Table:
        """Compose the live display: results table on top, progress below."""
        grid = Table.grid(padding=(0, 0))
        grid.add_column()

        # Results table
        tbl = Table(
            'Period', 'CH4 lag (s)', 'CH4 HDI (s)', 'N2O lag (s)', 'N2O HDI (s)',
            title=f'[bold]Results[/bold]  [dim]active workers: {_n_active[0]}/{det.n_workers}[/dim]',
            title_justify='left',
            show_lines=False,
            header_style='bold dim',
            border_style='dim',
            min_width=72,
        )
        for r in _live_rows:
            def _fmt_lag(prefix):
                v = r.get(f'{prefix}_tlag_s')
                return f'{v:.3f}' if v is not None and not (isinstance(v, float) and v != v) else '[dim]--[/dim]'

            def _fmt_hdi(prefix):
                v = r.get(f'{prefix}_hdi_range_s')
                if v is None or (isinstance(v, float) and v != v):
                    return '[dim]--[/dim]'
                color = 'green' if v < 0.5 else ('yellow' if v < 1.0 else 'red')
                return f'[{color}]{v:.3f}[/{color}]'

            tbl.add_row(
                Text(r.get('period', ''), style='dim'),
                _fmt_lag('ch4'), _fmt_hdi('ch4'),
                _fmt_lag('n2o'), _fmt_hdi('n2o'),
            )

        grid.add_row(tbl)
        grid.add_row(progress_bar)
        return grid


    progress = Progress(
        SpinnerColumn(),
        TextColumn('  {task.description}'),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=8,
    )
    overall = progress.add_task('[cyan]periods completed[/cyan]', total=len(file_paths))

    with Live(_make_display(progress), console=console,
              refresh_per_second=8, vertical_overflow='visible') as live:

        # Track in-flight files: all submitted immediately, so active =
        # min(n_workers, remaining) until workers start finishing.
        _n_active[0] = min(det.n_workers, len(file_paths))


        def _progress(done, total, row):
            _live_rows.append(row)
            _n_active[0] = min(det.n_workers, total - done)
            progress.update(overall, completed=done)
            live.update(_make_display(progress))


        results = det.run(on_progress=_progress)

    console.print(f"[green]Done — {len(results)} periods processed.[/green]")

    # Timestamps parsed from the filenames (file_date_format='%Y%m%d-%H%M')
    ts = pd.to_datetime(results['timestamp'])
    print(f"\nParsed timestamps : {ts.min()} -> {ts.max()}  (n={ts.notna().sum()})")

    # ------------------------------------------------------------------
    # PWBOPT S1/S2/S3 selection -- standard strategy
    # ------------------------------------------------------------------
    for scalar_label in ('CH4', 'N2O'):
        prefix = scalar_label.lower()
        tlag_col = f'{prefix}_tlag_s'
        hdi_col = f'{prefix}_hdi_range_s'
        if tlag_col not in results.columns:
            continue
        std = dv.flux.PwbBatchDetection.apply_pwbopt(
            results[tlag_col],
            results[hdi_col],
            hdi_thresh=0.5,  # S1: HDI < 0.5 s -> reliable
            dev_thresh=0.5,  # S2: max deviation from preceding optimal
        )
        results[f'{prefix}_pwbopt_s_std'] = std['pwbopt_s']
        results[f'{prefix}_flag_std'] = std['flag']

    # ------------------------------------------------------------------
    # PWBOPT -- conservative pre-filtered strategy
    # ------------------------------------------------------------------
    # Periods with HDI > 1.0 s are discarded before S1/S2/S3 runs so that
    # S2 cannot accept wide-HDI detections that happen to be near the
    # preceding optimal lag.
    for scalar_label in ('CH4', 'N2O'):
        prefix = scalar_label.lower()
        tlag_col = f'{prefix}_tlag_s'
        hdi_col = f'{prefix}_hdi_range_s'
        if tlag_col not in results.columns:
            continue
        tlag_pf = dv.flux.PwbBatchDetection.apply_hdi_prefilter(
            results[tlag_col],
            results[hdi_col],
            threshold=1.0,  # discard lags with HDI > 1.0 s before PWBOPT
        )
        pf = dv.flux.PwbBatchDetection.apply_pwbopt(tlag_pf, results[hdi_col])
        results[f'{prefix}_pwbopt_s_pf'] = pf['pwbopt_s']
        results[f'{prefix}_flag_pf'] = pf['flag']

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\nSummary")
    print("=" * 60)
    print(f"\n  {'Gas':<6}  {'Strategy':<20}  {'S1':>4}  {'S2':>4}  {'S3':>4}  {'Reliable':>9}")
    print("  " + "-" * 52)

    for scalar_label in ('CH4', 'N2O'):
        prefix = scalar_label.lower()
        for strategy, flag_col in [
            ('Standard', f'{prefix}_flag_std'),
            ('Pre-filtered', f'{prefix}_flag_pf'),
        ]:
            if flag_col not in results.columns:
                continue
            vc = results[flag_col].value_counts()
            s1 = vc.get('S1_optimal', 0)
            s2 = vc.get('S2_optimal', 0)
            s3 = vc.get('S3_unreliable', 0)
            rel = 100 * (s1 + s2) / len(results)
            print(f"  {scalar_label:<6}  {strategy:<20}  {s1:>4}  {s2:>4}  {s3:>4}  {rel:>8.1f}%")

        hdi_col = f'{prefix}_hdi_range_s'
        if hdi_col in results.columns:
            n_pf = int(np.sum(results[hdi_col].fillna(0) > 1.0))
            print(f"         Pre-filter removed: {n_pf}/{len(results)} periods (HDI > 1.0 s)")
        print()

    print(f"True lag  : {LAG_TRUE_S} s")
    for scalar_label in ('CH4', 'N2O'):
        prefix = scalar_label.lower()
        col = f'{prefix}_tlag_s'
        if col in results.columns:
            valid = results[col].dropna()
            print(f"  {scalar_label}: detected mean = {valid.mean():.3f} s  "
                  f"median = {valid.median():.3f} s  (n={len(valid)})")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    tmp_dir_obj.cleanup()
    print("\nTemporary files removed.")
    print("[OK] PwbBatchDetection example complete.")
