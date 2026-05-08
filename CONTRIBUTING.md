# Contributing to DIIVE

We welcome contributions! This guide explains how to set up your development environment, run tests, and contribute code.

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
uv sync                    # Install all dependencies from pyproject.toml
```

3. **Verify installation:**

```bash
uv run pytest tests/ -v
```

All tests should pass.

### Alternative Setup with conda (Legacy)

If you prefer conda:

```bash
conda env create -f environment.yml
conda activate diive
pip install -e .[dev]
pytest tests/ -v
```

Or with pip and venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
pytest tests/ -v
```

## Running Tests

Using uv (recommended):

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_gapfilling.py -v

# Run specific test
uv run pytest tests/test_gapfilling.py::TestRandomForest -v

# Run with coverage
uv run pytest tests/ --cov=diive --cov-report=html
```

Or directly with pytest (if environment is activated):

```bash
pytest tests/ -v
```

**Expected times:**
- Gap-filling tests: ~3-5 sec
- Flux processing chain: ~20-25 sec
- Full test suite: ~30-40 sec

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
- Black formatting (optional, but recommended)

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
6. Add example in `examples/createvar/` if applicable

### Adding a Gap-Filling Method to FluxProcessingChain

1. Create `level41_newmethod()` with all 24 feature parameters
2. Create `FeatureEngineer`, apply to data
3. Create and train gap-filling model
4. Store results in `self._level41['new_method'][ustar_scenario]`
5. Update tests and add example

### Adding an Outlier Detection Method

1. Inherit from appropriate base class (see `diive.pkgs.preprocessing.outlierdetection`)
2. Implement required methods (`flag_outliers()`, `get_flagged_data()`)
3. Add comprehensive docstring with parameters
4. Create example in `examples/outlierdetection/`
5. Add unit test in `tests/test_outlierdetection.py`

## Writing Tests

Tests are in `tests/` with one module per feature:

```python
import unittest
import diive as dv

class TestGapFilling(unittest.TestCase):
    def setUp(self):
        """Load data once for all tests."""
        self.df = dv.load_exampledata_parquet(data_id='TLL')

    def test_randomforest_basic(self):
        """Random Forest gap-filling produces valid output."""
        engineer = dv.FeatureEngineer(
            target_col='NEE',
            features_lag=[-1, 1],
        )
        df_eng = engineer.fit_transform(self.df)

        model = dv.RandomForestTS(
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
├── gap_filling/        # Gap-filling methods
├── outlierdetection/    # Outlier detection
├── visualization/       # Plotting examples
├── createvar/          # Variable creation
├── analyses/           # Time series analysis
├── corrections/        # Data corrections
├── flux/               # Flux-specific analysis
├── echires/            # High-resolution EC data
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
df = dv.load_exampledata_parquet(data_id='TLL')

# Example 1: Basic usage
def example_basic_usage():
    """Description of this example."""
    model = dv.RandomForestTS(
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
    print(f"R² score: {model.r2_test_pred:.3f}")
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

**XGBoost scientific notation in base_score:**
Already monkey-patched in `MlRegressorGapFillingBase`. No action needed.

**Import errors in Sphinx autodoc:**
Check that imports work: `python -c "from diive.module import Class"`

**Examples fail during doc build:**
Set `'abort_on_example_error': False` in `docs/conf.py`. Check build logs.

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/holukas/diive/issues)
- **Discussions:** [GitHub Discussions](https://github.com/holukas/diive/discussions)
- **Documentation:** [DIIVE ReadTheDocs](https://diive.readthedocs.io/)

## Thank You!

We appreciate your contributions. Whether it's code, documentation, examples, or bug reports, you're helping make DIIVE better for everyone.
