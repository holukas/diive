.. _installation:

============
Installation
============

System Requirements
===================

- **Python:** 3.12 (exact match recommended)
- **OS:** Windows, macOS, Linux
- **Disk space:** ~1 GB (for development environment)

Installation Methods
====================

Option 1: PyPI with pip (Recommended)
-------------------------------------

Install the latest released version using pip:

.. code-block:: bash

   pip install diive

Option 2: Using uv (Modern, Fast)
----------------------------------

`uv <https://docs.astral.sh/uv/>`_ is a modern Python package installer that's 5-10x faster than pip:

.. code-block:: bash

   # Install diive with uv
   uv pip install diive

   # Or create a new project with uv
   uv venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   uv pip install diive

Option 3: Development Setup with uv
------------------------------------

For development and contributing to DIIVE, clone the repository and use uv:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   uv sync                       # Install all dependencies
   uv run pytest tests/          # Run tests
   uv run python -m pip install -e .  # Install in editable mode

Option 4: Development Setup with conda (Legacy)
------------------------------------------------

If you prefer conda, use the provided environment file:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   conda env create -f environment.yml
   conda activate diive
   pip install -e .

Verifying Installation
======================

Check that DIIVE is installed correctly:

.. code-block:: python

   import diive as dv
   print(dv.__version__)  # Should print 0.91.0 or later

Verify with a simple example:

.. code-block:: python

   import diive as dv

   # Load example data
   df = dv.load_exampledata_parquet(data_id='TLL')
   print(f"Loaded {len(df)} records")
   print(df.head())

Key Dependencies
================

DIIVE depends on:

- **Data processing:** pandas, numpy, polars
- **Machine learning:** scikit-learn, xgboost, prophet
- **Visualization:** matplotlib, seaborn, bokeh
- **Statistical analysis:** scipy, statsmodels, scikit-optimize
- **Interpretability:** shap, eli5, yellowbrick
- **Time series:** sktime, pymannkendall

All dependencies are automatically installed with DIIVE.

Optional Dependencies
=====================

For Jupyter notebooks and interactive visualization:

.. code-block:: bash

   pip install jupyterlab jupyter-bokeh ipywidgets

Troubleshooting
===============

**ImportError: No module named 'diive'**
   Make sure DIIVE is installed (see above) and your Python environment is activated.

**EnvironmentError: Conda environment not found**
   If using conda, activate the environment first:

   .. code-block:: bash

      conda activate diive

**Version mismatch in XGBoost**
   Some systems may have XGBoost version conflicts. Install a compatible version:

   .. code-block:: bash

      pip install --upgrade xgboost

**Matplotlib backend issues**
   If plotting doesn't work, ensure a backend is available:

   .. code-block:: python

      import matplotlib
      matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.

Updating DIIVE
==============

To update to the latest version:

.. code-block:: bash

   pip install --upgrade diive

For development installations, pull the latest changes:

.. code-block:: bash

   cd diive
   git pull
   pip install -e .

Next Steps
==========

After installation, check out the :ref:`Getting Started <getting_started>` guide or browse the :ref:`Example Gallery <auto_examples/index>`.
