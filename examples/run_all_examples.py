"""
Run all examples in parallel and report results with execution time.

Executes all example scripts in examples/ in parallel and reports
success/failure with detailed error messages and execution times.

Usage:
    python examples/run_all_examples.py
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Example files to run (organized by functional domain)
EXAMPLE_FILES = [
    # Visualization
    'visualization/plot_heatmap_datetime_basic.py',
    'visualization/plot_heatmap_advanced.py',
    'visualization/plot_heatmap_xyz_basic.py',
    'visualization/plot_hexbin_basic.py',
    'visualization/plot_hexbin_advanced.py',
    'visualization/plot_cumulative_basic.py',
    'visualization/plot_cumulative_year.py',
    'visualization/plot_other_plots.py',
    'visualization/plot_timeseries.py',
    'visualization/plot_timeseries_interactive.py',
    'visualization/plot_dielcycle.py',
    'visualization/plot_histogram_basic.py',
    'visualization/plot_histogram_yearly.py',
    'visualization/plot_ridgeline_basic.py',
    'visualization/plot_ridgeline_advanced.py',
    'visualization/plot_scatter_xy_basic.py',
    'visualization/plot_scatter_xy_colored.py',
    'visualization/plot_treering_temperature.py',
    # Times
    'times/times_timestamp_sanitizer.py',
    'times/times_frequency_detection.py',
    'times/times_time_features.py',
    'times/times_diel_cycles.py',
    'times/times_temporal_matrices.py',
    'times/times_statistics.py',
    # Analysis
    'analysis/analysis_daily_correlation.py',
    'analysis/analysis_granger.py',
    'analysis/analysis_decoupling.py',
    'analysis/analysis_gapfinder.py',
    'analysis/analysis_gridaggregator.py',
    'analysis/analysis_harmonic.py',
    'analysis/analysis_histogram_distribution.py',
    'analysis/analysis_optimumrange.py',
    'analysis/analysis_quantiles.py',
    'analysis/analysis_seasonaltrend.py',
    # I/O
    'io/io_load_save_parquet.py',
    'io/io_read_single_file_with_datafilereader.py',
    'io/io_read_multiple_files_with_multidatafilereader.py',
    'io/io_read_single_file_with_readfiletype.py',
    'io/io_extract.py',
    # Preprocessing - Corrections
    'preprocessing/corrections/correction_relativehumidity_offset.py',
    'preprocessing/corrections/correction_radiation_offset.py',
    'preprocessing/corrections/correction_measurement_offset_replicate.py',
    'preprocessing/corrections/correction_winddir_offset.py',
    'preprocessing/corrections/correction_set_exact_values_to_missing.py',
    'preprocessing/corrections/correction_setto_value.py',
    'preprocessing/corrections/correction_setto_threshold.py',
    # Preprocessing - Outlier Detection
    'preprocessing/outlier_detection/outlier_absolutelimits.py',
    'preprocessing/outlier_detection/outlier_hampel.py',
    'preprocessing/outlier_detection/outlier_incremental.py',
    'preprocessing/outlier_detection/outlier_localsd.py',
    'preprocessing/outlier_detection/outlier_lof.py',
    'preprocessing/outlier_detection/outlier_manualremoval.py',
    'preprocessing/outlier_detection/outlier_stepwise.py',
    'preprocessing/outlier_detection/outlier_trim.py',
    'preprocessing/outlier_detection/outlier_zscore.py',
    # Preprocessing - QA/QC
    'preprocessing/qaqc/qc_overall_flag.py',
    'preprocessing/qaqc/qc_eddypro_flags.py',
    # Features
    'features/feature_engineer.py',
    'features/feature_sonic_temp_conversion.py',
    'features/feature_latent_heat.py',
    'features/feature_evapotranspiration.py',
    'features/feature_air.py',
    'features/feature_daynightflag.py',
    'features/feature_laggedvariants.py',
    'features/feature_noise.py',
    'features/feature_potentialradiation.py',
    'features/feature_timesince.py',
    'features/feature_vpd.py',
    # Fits
    'fits/fit_binfittercp.py',
    'fits/fit_fitter.py',
    # Flux - Processing chain
    'flux/fluxprocessingchain/fluxprocessingchain.py',
    'flux/fluxprocessingchain/fluxprocessingchain_quick.py',
    # Flux - Low-resolution processing
    'flux/lowres/flux_timelag_analysis.py',
    'flux/lowres/flux_common.py',
    'flux/lowres/flux_hqflux.py',
    'flux/lowres/flux_selfheating.py',
    'flux/lowres/flux_selfheating_production.py',
    'flux/lowres/flux_uncertainty.py',
    'flux/lowres/flux_ustar_mp_detection.py',
    'flux/lowres/flux_ustar_vekuri_detection.py',
    'flux/lowres/flux_ustar_method_comparison.py',
    # Flux - High-resolution analysis
    'flux/hires/flux_fluxdetectionlimit.py',
    'flux/hires/flux_lag.py',
    'flux/hires/flux_lag_pwb.py',
    'flux/hires/flux_lag_pwbopt.py',
    'flux/hires/flux_windrotation.py',
    # Gap-filling
    'gapfilling/gapfill_interpolate_generous.py',
    'gapfilling/gapfill_interpolate_conservative.py',
    'gapfilling/gapfill_mds.py',
    'gapfilling/gapfill_mds_comparison.py',
    'gapfilling/gapfill_randomforest.py',
    'gapfilling/gapfill_randomforest_longterm.py',
    'gapfilling/gapfill_quickfill.py',
    'gapfilling/gapfill_optimize_randomforest.py',
    'gapfilling/gapfill_xgboost.py',
    'gapfilling/gapfill_optimize_xgboost.py',
    'gapfilling/gapfill_comparison.py',
]

MAX_WORKERS = 8  # Number of parallel workers


def run_example(example_file, examples_dir):
    """Run a single example and return result with timing."""
    start_time = time.time()
    example_path = examples_dir / example_file

    if not example_path.exists():
        return {
            'file': example_file,
            'status': 'skip',
            'error': 'File not found',
            'time': 0.0
        }

    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            return {
                'file': example_file,
                'status': 'pass',
                'error': None,
                'time': elapsed
            }
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return {
                'file': example_file,
                'status': 'fail',
                'error': error_msg,
                'time': elapsed
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'file': example_file,
            'status': 'timeout',
            'error': 'Timeout (exceeded 60 seconds)',
            'time': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'file': example_file,
            'status': 'error',
            'error': str(e),
            'time': elapsed
        }


def run_all_examples():
    """Run all example files in parallel and report results."""
    examples_dir = Path(__file__).parent
    total_start = time.time()

    print("=" * 80)
    print(f"Running {len(EXAMPLE_FILES)} example files (100+ functions total) in parallel (max {MAX_WORKERS} workers)...")
    print("=" * 80 + "\n")

    results = {'passed': [], 'failed': [], 'skipped': []}
    completed = 0

    # Run examples in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_example, example_file, examples_dir): example_file
            for example_file in EXAMPLE_FILES
        }

        for future in as_completed(futures):
            result = future.result()
            example_file = result['file']
            status = result['status']
            elapsed = result['time']
            completed += 1
            progress = (completed / len(EXAMPLE_FILES)) * 100

            if status == 'pass':
                print(f"[PASS] {example_file:<40} ({elapsed:6.2f}s) [{completed:2d}/{len(EXAMPLE_FILES)} {progress:5.1f}%]")
                results['passed'].append((example_file, elapsed))
            elif status == 'fail':
                print(f"[FAIL] {example_file:<40} ({elapsed:6.2f}s) [{completed:2d}/{len(EXAMPLE_FILES)} {progress:5.1f}%]")
                error_line = result['error'].split('\n')[0][:60]
                print(f"       Error: {error_line}")
                results['failed'].append((example_file, result['error'], elapsed))
            elif status == 'timeout':
                print(f"[TIMEOUT] {example_file:<38} ({elapsed:6.2f}s) [{completed:2d}/{len(EXAMPLE_FILES)} {progress:5.1f}%]")
                results['failed'].append((example_file, result['error'], elapsed))
            elif status == 'error':
                print(f"[ERROR] {example_file:<40} ({elapsed:6.2f}s) [{completed:2d}/{len(EXAMPLE_FILES)} {progress:5.1f}%]")
                error_msg = result['error'][:60]
                print(f"        {error_msg}")
                results['failed'].append((example_file, result['error'], elapsed))
            else:  # skip
                print(f"[SKIP] {example_file:<41} [{completed:2d}/{len(EXAMPLE_FILES)} {progress:5.1f}%]")
                results['skipped'].append(example_file)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[PASS] Passed: {len(results['passed'])}/{len(EXAMPLE_FILES)}")
    print(f"[FAIL] Failed: {len(results['failed'])}/{len(EXAMPLE_FILES)}")
    if results['skipped']:
        print(f"[SKIP] Skipped: {len(results['skipped'])}/{len(EXAMPLE_FILES)}")

    if results['passed']:
        print("\nPassed examples:")
        total_pass_time = sum(t for _, t in results['passed'])
        for example, elapsed in sorted(results['passed'], key=lambda x: x[1], reverse=True):
            print(f"   {example:<50} {elapsed:6.2f}s")
        print(f"   {'Total (passed)':<50} {total_pass_time:6.2f}s")

    if results['failed']:
        print("\nFailed examples:")
        for example_file, error, elapsed in results['failed']:
            print(f"   {example_file:<50} {elapsed:6.2f}s")
            if error and error != "File not found":
                error_line = error.split('\n')[0][:80]
                print(f"     {error_line}")
        print("\nRun individual examples for full error details:")
        for example_file, _, _ in results['failed']:
            print(f"   python examples/{example_file}")

    print(f"\nTotal execution time: {total_elapsed:.2f}s")
    print("=" * 80)

    # Exit with error if any failed
    return 0 if not results['failed'] else 1


if __name__ == '__main__':
    exit_code = run_all_examples()
    sys.exit(exit_code)
