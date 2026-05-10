"""
Run all examples in parallel and report results with execution time.

Executes all example scripts in examples/ (core and pkgs) in parallel and reports
success/failure with detailed error messages and execution times.

Usage:
    python examples/run_all_examples.py
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Example files to run (organized by core/ and pkgs/ structure)
EXAMPLE_FILES = [
    # CORE: Visualization and times
    'core/visualization/heatmap_datetime.py',
    'core/visualization/hexbin.py',
    'core/visualization/timeseries_and_cumulative.py',
    'core/visualization/other_plots.py',
    'core/visualization/timeseries.py',
    'core/visualization/dielcycle.py',
    'core/visualization/histogram.py',
    'core/visualization/ridgeline.py',
    'core/visualization/scatter_xy.py',
    'core/times/timestamp_sanitizer.py',
    # PKGS: Analysis
    'pkgs/analysis/correlation.py',
    'pkgs/analysis/decoupling.py',
    'pkgs/analysis/gapfinder.py',
    'pkgs/analysis/gridaggregator.py',
    'pkgs/analysis/histogram_distribution.py',
    'pkgs/analysis/optimumrange.py',
    'pkgs/analysis/quantiles.py',
    'pkgs/analysis/seasonaltrend.py',
    # PKGS: IO
    'pkgs/io/extract.py',
    # PKGS: Preprocessing - Corrections
    'pkgs/preprocessing/corrections/correction_relativehumidity_offset.py',
    'pkgs/preprocessing/corrections/correction_radiation_offset.py',
    'pkgs/preprocessing/corrections/correction_measurement_offset_replicate.py',
    'pkgs/preprocessing/corrections/correction_winddir_offset.py',
    'pkgs/preprocessing/corrections/correction_set_exact_values_to_missing.py',
    'pkgs/preprocessing/corrections/correction_setto_value.py',
    'pkgs/preprocessing/corrections/correction_setto_threshold.py',
    # PKGS: Preprocessing - Outlier Detection
    'pkgs/preprocessing/outlierdetection/outlier_absolutelimits.py',
    'pkgs/preprocessing/outlierdetection/outlier_hampel.py',
    'pkgs/preprocessing/outlierdetection/outlier_incremental.py',
    'pkgs/preprocessing/outlierdetection/outlier_localsd.py',
    'pkgs/preprocessing/outlierdetection/outlier_lof.py',
    'pkgs/preprocessing/outlierdetection/outlier_manualremoval.py',
    'pkgs/preprocessing/outlierdetection/outlier_stepwise.py',
    'pkgs/preprocessing/outlierdetection/outlier_trim.py',
    'pkgs/preprocessing/outlierdetection/outlier_zscore.py',
    # PKGS: Preprocessing - QA/QC
    'pkgs/preprocessing/qaqc/qc_overall_flag.py',
    'pkgs/preprocessing/qaqc/qc_eddypro_flags.py',
    # PKGS: Features (formerly createvar)
    'pkgs/features/air.py',
    'pkgs/features/conversions.py',
    'pkgs/features/daynightflag.py',
    'pkgs/features/laggedvariants.py',
    'pkgs/features/noise.py',
    'pkgs/features/potentialradiation.py',
    'pkgs/features/timesince.py',
    'pkgs/features/vpd.py',
    # PKGS: Fits
    'pkgs/fits/fitter.py',
    # PKGS: Flux - Processing chain
    'pkgs/flux/fluxprocessingchain/fluxprocessingchain.py',
    # PKGS: Flux - Common and methods
    'pkgs/flux/common.py',
    'pkgs/flux/hqflux/hqflux.py',
    'pkgs/flux/selfheating/selfheating.py',
    'pkgs/flux/uncertainty/uncertainty.py',
    'pkgs/flux/ustar_mp_detection/ustar_mp_detection.py',
    'pkgs/flux/ustarthreshold/ustarthreshold.py',
    # PKGS: Flux - High-resolution (echires)
    'pkgs/flux/hires/fluxdetectionlimit.py',
    'pkgs/flux/hires/lag.py',
    'pkgs/flux/hires/windrotation.py',
    # PKGS: Gap-filling
    'pkgs/gapfilling/interpolate.py',
    'pkgs/gapfilling/mds.py',
    'pkgs/gapfilling/mds_comparison.py',
    'pkgs/gapfilling/randomforest_ts.py',
    'pkgs/gapfilling/xgboost_ts.py',
    'pkgs/gapfilling/comparison.py',
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

            if status == 'pass':
                print(f"[PASS] {example_file:<45} ({elapsed:6.2f}s)")
                results['passed'].append((example_file, elapsed))
            elif status == 'fail':
                print(f"[FAIL] {example_file:<45} ({elapsed:6.2f}s)")
                error_line = result['error'].split('\n')[0][:80]
                print(f"       Error: {error_line}")
                results['failed'].append((example_file, result['error'], elapsed))
            elif status == 'timeout':
                print(f"[TIMEOUT] {example_file:<43} ({elapsed:6.2f}s)")
                results['failed'].append((example_file, result['error'], elapsed))
            elif status == 'error':
                print(f"[ERROR] {example_file:<45} ({elapsed:6.2f}s)")
                print(f"        {result['error'][:80]}")
                results['failed'].append((example_file, result['error'], elapsed))
            else:  # skip
                print(f"[SKIP] {example_file:<46} (not found)")
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
