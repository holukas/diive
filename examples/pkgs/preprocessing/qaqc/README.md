# Quality Assurance / Quality Control (QA/QC) Examples

Examples demonstrating quality control methods, flag generation, and data quality assessment.

## Contents

- **qcf.py** — Overall Quality Control Flag (QCF) generation combining multiple QC tests
- **eddyproflags.py** — EddyPro-specific quality flags (VM97, signal strength, SSITC, etc.)

## Related Documentation

See `diive.pkgs.preprocessing.qaqc` for:
- `FlagQCF` — Overall quality flag combining individual tests
- `flag_signal_strength_eddypro_test()` — Signal quality
- `flags_vm97_eddypro_fluxnetfile_tests()` — Vickers & Mahrt (1997) raw data tests
- `flag_ssitc_eddypro_test()` — Steady State and Integral Turbulence Characteristics
- `flag_spectral_correction_factor_eddypro_test()` — Spectral correction assessment

## Usage

```bash
uv run python examples/pkgs/preprocessing/qaqc/qcf.py
uv run python examples/pkgs/preprocessing/qaqc/eddyproflags.py
```

Or run all QA/QC examples:

```bash
uv run python examples/run_all_examples.py
```
