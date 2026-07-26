# Contributing to DIIVE

This guide covers development setup, running tests, and contributing code.

## Development Setup

### Prerequisites

- Python 3.12 or 3.13
- Git
- **uv** (modern, fast package manager) - [Install uv](https://docs.astral.sh/uv/getting-started/)

### Setup Steps with uv (Recommended)

1. **Clone the repository:**

```bash
git clone https://github.com/holukas/diive.git
cd diive
```

2. **Install dependencies and development tools:**

```bash
uv sync                              # Core dependencies + the 'dev' group (synced by default)
uv sync --all-extras --all-groups    # Everything: 'gui'/'gui3d' extras + 'db'/'dev'/'build' groups
```

The optional pieces are split across two uv mechanisms, so `--all-extras` alone is not "everything":

| Kind | Name | Pulls in | Install |
|---|---|---|---|
| extra | `gui` | PySide6 desktop GUI | `uv sync --extra gui` |
| extra | `gui3d` | PyVista/VTK 3D surface tabs | `uv sync --extra gui3d` |
| extra + group | `db` | `influxdb-client` (InfluxDB engine) | `uv sync --group db` |
| group | `dev` | test/lint/docs/notebook tooling | synced by default |
| group | `build` | PyInstaller (standalone GUI build) | `uv sync --group build` |

3. **Verify installation:**

```bash
uv run pytest tests/ -v
```

All tests should pass.

### Alternative Setup with pip and venv

`uv` is the supported path. If you use pip instead, note that the development
tooling lives in dependency groups (a PEP 735 feature), not in an extra, so
`pip install -e .[dev]` does not exist. Install the package, then the test
tooling by hand:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .          # Add extras if needed, e.g. pip install -e '.[gui]'
pip install pytest
pytest tests/ -v
```

## Running Tests

Using uv (recommended):

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_gapfilling.py -v

# Run specific test class
uv run pytest tests/test_gapfilling.py::TestGapFilling -v

# Run a single test
uv run pytest tests/test_gapfilling.py::TestGapFilling::test_gapfilling_randomforest -v
```

Or directly with pytest (if environment is activated):

```bash
pytest tests/ -v
```

The GUI tests (`tests/test_gui.py`) run offscreen and skip themselves unless the
`gui` extra is installed (`uv sync --extra gui`).

`pytest-cov` is in the `dev` group, so coverage works out of the box:

```bash
# Coverage for the whole package
uv run pytest tests/ --cov=diive --cov-report=term-missing

# Record which test covered which line (then filter by test in the HTML report)
uv run pytest tests/ --cov=diive --cov-context=test --cov-report=html
```

Note that `tests/test_gui.py` drives a lot of library code on its way through
the widgets, so it inflates the library figure. To see what the library tests
cover on their own, deselect it:

```bash
uv run pytest tests/ --ignore=tests/test_gui.py --cov=diive --cov-report=term-missing
```

Omitting `diive/gui` from the *report* is not the same thing — it hides those
lines but still counts the coverage `test_gui.py` contributes elsewhere.

[COVERAGE_GAPS.md](COVERAGE_GAPS.md) tracks what is still uncovered and why —
worth a look before writing new tests, so you pick something that matters.

The suite runs real models on real data (gap-filling, the flux processing chain,
the partitioning ports), so expect it to take minutes rather than seconds. Use
`-k` or a single test file while iterating.

## Coding Standards

### Input Validation

Validate input **only at system boundaries** (user input, external data). Don't validate internal contracts between functions.

```python
# Good: validate user input at API boundary
def process_data(df, target_col):
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be empty")
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found")

# Bad: validating internal contract
def _internal_helper(result):
    assert result is not None  # Don't do this internally
```

### Error Handling

Let exceptions propagate unless you can recover. Be specific about what you catch.

```python
# Good: specific and recoverable
try:
    result = operation()
except FileNotFoundError:
    logger.info("Using default fallback")
    return default_value

# Bad: too broad
try:
    result = operation()
except Exception:
    pass  # Never silence exceptions
```

### Comments

**Only comment the WHY, not the WHAT.**

Well-named code already explains what it does. Only comment when the reason is non-obvious.

```python
# Good: explains hidden constraint
# Exclude dot columns to avoid circular dependency with gap-filling
cols = [c for c in df.columns if not c.startswith('.')]

# Bad: explains what code does
# Add 1 to result
result = result + 1
```

### Code Style

- Use snake_case for functions and variables
- Use PascalCase for classes
- Use ALL_CAPS for constants
- Type hints are encouraged
- The configured tooling is `ruff` (linting, `[tool.ruff]` in `pyproject.toml`) and
  `autopep8` (formatting), both in the `dev` group: `uv run ruff check .`, `uv run autopep8 --diff <file>`

```python
from typing import Optional
import pandas as pd

class FeatureEngineer:
    """Extract and engineer features from time series data."""

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Engineer features and return enriched dataframe."""
        ...
```

### No File I/O in Examples

Examples should show the API, not file operations. Keep I/O in user code.

```python
# Good: shows how to use
df_engineered = engineer.fit_transform(df)
model.fillgaps()
result = model.get_gapfilled_target()

# Bad: includes I/O (remove this from examples)
result.to_csv('output.csv')
```

## Adding New Features

### Adding a Feature Engineering Stage

1. Add parameter to `FeatureEngineer.__init__()` (default None)
2. Implement `_stagename_features()` method
3. Call from `_create_features()` orchestrator
4. Use naming: `.{col}_TYPE{detail}` (e.g., `.Tair_f_POL2`)
5. Update docstring with new parameter
6. Add example in `examples/features/` if applicable

### Adding a Gap-Filling Method to the Flux Processing Chain

The chain uses composable per-level callables (`diive.flux.fluxprocessingchain`), not
a monolithic class. To add an L4.1 gap-filling method:

1. Add a `run_level41_newmethod(data, ...)` callable that builds the model and stores
   results in `data.levels.level41_newmethod` (keyed by ustar_scenario), returning a new
   `data` via `dataclasses.replace` — never mutate the input
2. Build a `FeatureEngineer`, train the model, gap-fill
3. Wire it into `run_chain` behind a `FluxConfig` flag and update `codegen.py`
4. Update tests (`tests/test_fluxprocessingchain.py`) and add an example

### Adding an Outlier Detection Method

Implementations live in `diive/preprocessing/outlier_detection/`; `diive.outliers`
is only the public re-export namespace. See `hampel.py` for a worked example.

1. Add a module in `diive/preprocessing/outlier_detection/` with a class that inherits
   from `FlagBase` (`diive/core/base/flagbase.py`)
2. Set a `flagid` class attribute (e.g. `flagid = 'OUTLIER_HAMPEL'`) and pass it to
   `super().__init__(series=..., flagid=self.flagid, idstr=idstr)`
3. Implement `_flagtests(iteration)`, returning `(ok, rejected, n_outliers)` as
   `(DatetimeIndex, DatetimeIndex, int)`, and `calc(repeat=True, progress_callback=None)`,
   which drives the iterations via `self.repeat(func=self.run_flagtests, ...)`. `run(**kwargs)`
   delegates to `calc()`. Results are exposed by the base class through the `overall_flag`,
   `filteredseries` and `flag` properties
4. Add comprehensive docstring with parameters
5. Re-export the class from `diive/preprocessing/outlier_detection/__init__.py` and
   `diive/outliers/__init__.py`
6. Create example in `examples/preprocessing/outlier_detection/`
7. Add unit test in `tests/test_outlierdetection.py`

## Writing Tests

Tests are in `tests/` with one module per feature:

```python
import unittest
import diive as dv

class TestGapFilling(unittest.TestCase):
    def setUp(self):
        """Load data once for all tests."""
        self.df = dv.load_exampledata_parquet()

    def test_randomforest_basic(self):
        """Random Forest gap-filling produces valid output."""
        engineer = dv.gapfilling.FeatureEngineer(
            target_col='NEE',
            features_lag=[-1, 1],
        )
        df_eng = engineer.fit_transform(self.df)

        model = dv.gapfilling.RandomForestTS(
            input_df=df_eng,
            target_col='NEE',
        )
        model.trainmodel()
        model.fillgaps()

        result = model.get_gapfilled_target()
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(result.notna().all())
```

**Guidelines:**
- Use flexible assertions for SHAP importance (±5-10% variability is normal)
- Test at boundaries — validate user input, check outputs
- Don't mock internals — test with real data
- Expect variability — ML models have inherent randomness

## Creating Examples

Examples are organized in `examples/`:

```bash
examples/
├── gapfilling/         # Gap-filling methods
├── preprocessing/      # Outlier detection, corrections, QA/QC
├── visualization/      # Plotting examples
├── features/           # Feature engineering / variable creation
├── analysis/           # Time series analysis
├── flux/               # Flux-specific analysis (incl. hires/ high-res EC)
├── events/             # Event markers
├── fits/               # Curve fitting
├── io/                 # File I/O
├── times/              # Timestamp handling
└── ...
```

**Guidelines:**
1. Keep it simple — 1-4 focused examples per file
2. Runnable end-to-end — No user interaction needed
3. Load test data — Use `dv.load_exampledata_parquet()`
4. Add docstrings — Explain what each example demonstrates
5. No file I/O — Show API, not CSV exports
6. Self-contained — Examples run independently

**Example structure:**

```python
"""
Title: What This Example Shows

Description of 2-3 sentences explaining the use case and key concepts.
See diive.classname for API details.
"""

import diive as dv
import matplotlib.pyplot as plt

# Load example data
df = dv.load_exampledata_parquet()

# Example 1: Basic usage
def example_basic_usage():
    """Description of this example."""
    model = dv.gapfilling.RandomForestTS(
        input_df=df,
        target_col='NEE',
    )
    model.trainmodel()
    return model

# Example 2: Advanced usage
def example_advanced_usage():
    """Description of this example."""
    ...

if __name__ == '__main__':
    model = example_basic_usage()
    print(f"R² score: {model.scores_traintest_['r2']:.3f}")
```

## Documentation

### Building Docs Locally

With uv:

```bash
cd docs
uv run sphinx-build -b html . _build/html
```

Or if environment is activated:

```bash
cd docs
sphinx-build -b html . _build/html
```

Open `docs/_build/html/index.html` in a browser to preview.

### Docstring Style

Use Google-style docstrings for clarity:

```python
class MyClass:
    """Short description of the class.

    Longer description with context and typical usage patterns.

    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2. Defaults to 10.

    Attributes:
        attr1 (float): Description of computed attribute.

    Example:
        Basic usage example here. See examples/category/file.py
        for complete examples.

    Raises:
        ValueError: When param1 is invalid.
    """

    def method(self, arg1: str) -> pd.DataFrame:
        """Short description of method.

        Args:
            arg1: Description

        Returns:
            Dataframe with processed results.
        """
```

## Git Workflow

1. **Create a branch** for your feature/fix:

```bash
git checkout -b feature/my-new-feature
```

2. **Make changes** and commit:

```bash
git add .
git commit -m "Add my new feature"
```

3. **Push to GitHub:**

```bash
git push origin feature/my-new-feature
```

4. **Open a Pull Request** with description of changes

5. **Address review feedback** and update the PR

### Before committing, ensure:
- All tests pass: `pytest tests/ -v`
- Code is clean and readable
- Docstrings are complete
- Example works (if applicable)

## Debugging Tips

**SHAP importance fluctuates:**
Normal variability (±5-10%). Use flexible assertions with `assertGreater/assertLess`.

**Feature reduction too strict:**
Reduce `shap_threshold_factor` in gap-filling config (default 0.5).

**Import errors in Sphinx autodoc:**
Check that imports work: `python -c "from diive.module import Class"`

**Examples fail during doc build:**
Set `'abort_on_example_error': False` in `docs/conf.py`. Check build logs.

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/holukas/diive/issues)
- **Discussions:** [GitHub Discussions](https://github.com/holukas/diive/discussions)
- **Documentation:** [DIIVE ReadTheDocs](https://diive.readthedocs.io/)

