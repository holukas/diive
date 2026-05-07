.. _contributing:

============
Contributing
============

We welcome contributions! This guide explains how to set up your development environment, run tests, and contribute code.

Development Setup
=================

Prerequisites
-------------

- Python 3.10 or 3.11 (3.11 recommended)
- Git
- Conda or pip (we recommend conda)

Setup Steps
-----------

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/holukas/diive.git
      cd diive

2. **Create a conda environment:**

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate diive

   Or manually with pip:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -e .[dev]

3. **Verify installation:**

   .. code-block:: bash

      pytest tests/ -v

   All tests should pass.

Running Tests
=============

**Run all tests:**

.. code-block:: bash

   pytest tests/ -v

**Run specific test file:**

.. code-block:: bash

   pytest tests/test_gapfilling.py -v

**Run specific test:**

.. code-block:: bash

   pytest tests/test_gapfilling.py::TestRandomForest -v

**Run with coverage:**

.. code-block:: bash

   pytest tests/ --cov=diive --cov-report=html

Expected times:
- Gap-filling tests: ~3-5 sec
- Flux processing chain: ~20-25 sec
- Full test suite: ~30-40 sec

Coding Standards
================

Input Validation
----------------

Validate input **only at system boundaries** (user input, external data). Don't validate internal contracts between functions.

.. code-block:: python

   # Good: validate user input at API boundary
   def process_data(df, target_col):
       if df is None or df.empty:
           raise ValueError("DataFrame cannot be empty")
       if target_col not in df.columns:
           raise KeyError(f"Column '{target_col}' not found")

   # Bad: validating internal contract
   def _internal_helper(result):
       assert result is not None  # Don't do this internally

Error Handling
--------------

Let exceptions propagate unless you can recover. Be specific about what you catch.

.. code-block:: python

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

Comments
--------

**Only comment the WHY, not the WHAT.**

Well-named code already explains what it does. Only comment when the reason is non-obvious.

.. code-block:: python

   # Good: explains hidden constraint
   # Exclude dot columns to avoid circular dependency with gap-filling
   cols = [c for c in df.columns if not c.startswith('.')]

   # Bad: explains what code does
   # Add 1 to result
   result = result + 1

No Comments for:
   - How to use: good naming handles this
   - Code references: belongs in commit message
   - Obvious logic: if you feel compelled to explain it, rename the variable

Code Style
----------

- Use snake_case for functions and variables
- Use PascalCase for classes
- Use ALL_CAPS for constants
- Type hints are encouraged (see example below)
- Black formatting (optional, but recommended)

.. code-block:: python

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

No File I/O in Examples
------------------------

Examples should show the API, not file operations. Keep I/O in user code.

.. code-block:: python

   # Good: shows how to use
   df_engineered = engineer.fit_transform(df)
   model.fillgaps()
   result = model.get_gapfilled_target()

   # Bad: includes I/O (remove this from examples)
   result.to_csv('output.csv')

Adding New Features
===================

Adding a Feature Engineering Stage
-----------------------------------

1. Add parameter to ``FeatureEngineer.__init__()`` (default None)
2. Implement ``_stagename_features()`` method
3. Call from ``_create_features()`` orchestrator
4. Use naming: ``.{col}_TYPE{detail}`` (e.g., ``.Tair_f_POL2``)
5. Update docstring with new parameter
6. Add example in ``examples/createvar/`` if applicable

Example:

.. code-block:: python

   class FeatureEngineer:
       def __init__(
           self,
           ...,
           features_mynewthing: Optional[List[int]] = None,
       ):
           self.features_mynewthing = features_mynewthing

       def _mynewthing_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """Implement the new feature type."""
           ...
           return df

       def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
           """Call all feature stages."""
           ...
           if self.features_mynewthing is not None:
               df = self._mynewthing_features(df)
           ...
           return df

Adding a Gap-Filling Method to FluxProcessingChain
----------------------------------------------------

1. Create ``level41_newmethod()`` with all 24 feature parameters
2. Create ``FeatureEngineer``, apply to data
3. Create and train gap-filling model
4. Store results in ``self._level41['new_method'][ustar_scenario]``
5. Update tests and add example

Adding an Outlier Detection Method
-----------------------------------

1. Inherit from appropriate base class (see ``diive.pkgs.outlierdetection``)
2. Implement required methods (``flag_outliers()``, ``get_flagged_data()``)
3. Add comprehensive docstring with parameters
4. Create example in ``examples/outlierdetection/``
5. Add unit test in ``tests/test_outlierdetection.py``

Writing Tests
=============

Test Structure
--------------

Tests are in ``tests/`` with one module per feature:

.. code-block:: bash

   tests/
   ├── test_gapfilling.py
   ├── test_outlierdetection.py
   ├── test_fluxprocessingchain.py
   ├── test_visualizations.py
   └── ...

Example Test
~~~~~~~~~~~~

.. code-block:: python

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

Testing Guidelines
-------------------

- **Use flexible assertions** for SHAP importance (±5-10% variability is normal)
- **Test at boundaries** — validate user input, check outputs
- **Don't mock internals** — test with real data, not mocks
- **Expect variability** — ML models have inherent randomness

.. code-block:: python

   # Good: flexible for natural variability
   self.assertGreater(model.r2_test_pred, 0.5)
   self.assertLess(model.r2_test_pred, 0.9)

   # Bad: too strict
   self.assertEqual(model.r2_test_pred, 0.7234567)

Creating Examples
=================

Example Guidelines
-------------------

1. **Keep it simple** — 1-4 focused examples per file
2. **Runnable end-to-end** — No user interaction needed
3. **Load test data** — Use ``dv.load_exampledata_parquet()``
4. **Add docstrings** — Explain what each example demonstrates
5. **No file I/O** — Show API, not CSV exports
6. **Self-contained** — Examples run independently

Example Structure
-----------------

.. code-block:: python

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

Example Files by Category
---------------------------

Examples are organized in ``examples/``:

.. code-block:: bash

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

To add an example:

1. Create file in appropriate ``examples/`` subfolder
2. Follow naming: ``feature_name.py``
3. Use function structure above
4. Test locally: ``python examples/feature_name.py``
5. Will be auto-generated into :ref:`Example Gallery <auto_examples/index>`

Documentation
==============

Building Docs Locally
---------------------

Build Sphinx documentation:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

Open ``docs/_build/html/index.html`` in a browser to preview.

Docstring Style
---------------

Use Google-style docstrings for clarity:

.. code-block:: python

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

Git Workflow
============

1. **Create a branch** for your feature/fix:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. **Make changes** and commit:

   .. code-block:: bash

      git add .
      git commit -m "Add my new feature"

3. **Push to GitHub:**

   .. code-block:: bash

      git push origin feature/my-new-feature

4. **Open a Pull Request** with description of changes

5. **Address review feedback** and update the PR

Before committing, ensure:
- All tests pass: ``pytest tests/ -v``
- Code is clean and readable
- Docstrings are complete
- Example works (if applicable)

Debugging Tips
==============

**SHAP importance fluctuates:**
   Normal variability (±5-10%). Use flexible assertions with ``assertGreater/assertLess``.

**Feature reduction too strict:**
   Reduce ``shap_threshold_factor`` in gap-filling config (default 0.5).

**XGBoost scientific notation in base_score:**
   Already monkey-patched in ``MlRegressorGapFillingBase``. No action needed.

**Import errors in Sphinx autodoc:**
   Check that imports work: ``python -c "from diive.module import Class"``

**Examples fail during doc build:**
   Set ``'abort_on_example_error': False`` in ``docs/conf.py``. Check build logs.

Getting Help
============

- **Issues:** `GitHub Issues <https://github.com/holukas/diive/issues>`_
- **Discussions:** `GitHub Discussions <https://github.com/holukas/diive/discussions>`_
- **Pull Requests:** Open a PR with your changes

Thank You!
==========

We appreciate your contributions. Whether it's code, documentation, examples, or bug reports, you're helping make DIIVE better for everyone.
