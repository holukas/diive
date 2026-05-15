# Quality Assurance / Quality Control (QA/QC) Examples

Examples demonstrating quality control methods, flag generation, and data quality assessment.

## Contents

- **qc_overall_flag.py** — Overall Quality Control Flag (QCF) combining multiple test flags
- **qc_eddypro_flags.py** — EddyPro quality flag extraction (signal strength, VM97 tests, completeness)

## Related Documentation

See `diive.pkgs.preprocessing.qaqc` for:
- `FlagQCF` — Overall quality flag combining individual tests
- `flag_signal_strength_eddypro_test()` — Signal quality
- `flags_vm97_eddypro_fluxnetfile_tests()` — Vickers & Mahrt (1997) raw data tests
- `flag_ssitc_eddypro_test()` — Steady State and Integral Turbulence Characteristics
- `flag_spectral_correction_factor_eddypro_test()` — Spectral correction assessment

## Use Cases

**Generate overall quality flag (QCF):**
```python
from diive.pkgs.preprocessing.qaqc import FlagQCF

# Combine multiple individual quality tests into single QCF
qcf = FlagQCF(
    df=df,
    target_col='NEE',
    swinpot_col='SW_IN_POT',  # Optional: enables day/night separation
    idstr='_L41'
)
qcf.calculate(daytime_accept_qcf_below=2)  # Accept good+medium daytime

# QCF values: 0=good, 1=marginal, 2=poor
filtered = df[qcf.filteredseries.notna()]  # Keep only good quality
highest_quality = df[qcf.filteredseries_hq.notna()]  # Keep only best

# Get comprehensive report
qcf.report_qcf_series()  # Summary statistics
qcf.report_qcf_flags()  # Per-test breakdown
qcf.showplot_qcf_heatmaps()  # Visualization
```

**Extract EddyPro quality flags:**
```python
from diive.pkgs.preprocessing.qaqc import (
    flag_signal_strength_eddypro_test,
    flags_vm97_eddypro_fluxnetfile_tests,
    flag_ssitc_eddypro_test
)

# Signal quality (IRGA, anemometer)
sig_flag = flag_signal_strength_eddypro_test(
    df['AGC_MEAN'],
    method='discard above',  # Closed-path: high AGC = bad
    threshold=0.85
)

# VM97 raw data tests (8 individual tests)
vm97_flags = flags_vm97_eddypro_fluxnetfile_tests(
    df['QC_FLAGS_VM97']  # 8-digit integer
)
# Returns: spikes, amplitude, dropout, absolute_limits,
#          skew_hard, skew_soft, discontin_hard, discontin_soft

# Stationarity test
stl_flag = flag_ssitc_eddypro_test(df['SSITC_TEST'])
```

## Quality Flag Schema

**QCF (Overall Quality Control Flag):**
- **0** = Good quality (all tests pass)
- **1** = Marginal quality (1-3 soft warnings, no hard fails)
- **2** = Poor quality (>3 soft warnings OR ≥2 hard fails)

**EddyPro test results:**
- **0** = Pass
- **1** = Soft warning (marginal)
- **2** = Hard fail (reject)

## Running Examples

```bash
# Generate overall quality flags from multiple tests
uv run python examples/pkgs/preprocessing/qaqc/qc_overall_flag.py

# Extract and convert EddyPro-specific quality flags
uv run python examples/pkgs/preprocessing/qaqc/qc_eddypro_flags.py

# Run all QA/QC examples
uv run python examples/run_all_examples.py
```

## FLUXNET Standards

Quality control follows FLUXNET conventions:
- Quality tests applied independently
- Results combined into overall QCF score
- Day/night thresholds differ (nighttime stricter)
- USTAR filtering applied to flux only, not energy variables
- Multiple percentile scenarios for uncertainty quantification
