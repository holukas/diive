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

## Usage

```bash
uv run python examples/pkgs/preprocessing/qaqc/qc_overall_flag.py
uv run python examples/pkgs/preprocessing/qaqc/qc_eddypro_flags.py
```

Or run all QA/QC examples:

```bash
uv run python examples/run_all_examples.py
```
