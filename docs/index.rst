.. diive documentation master file

========
diive
========

**Data Integration and Interactive Visualization Engine**

diive is a Python library for time series processing and analysis with focus on ecosystem flux data. It provides:

- **Feature engineering** — 8-stage composable pipeline for ML preparation
- **Gap-filling** — Multiple ML methods (Random Forest, XGBoost) and meteorological matching
- **Quality control** — 10+ outlier detection algorithms and data validation
- **Flux processing** — Multi-level workflow (Levels 2-4.1) for eddy covariance data
- **Visualization** — 14+ specialized plot types for time series analysis
- **Analysis** — Correlation, decomposition, seasonal trends, and more

Quick Links
===========

- :doc:`installation` — Installation guide
- :doc:`getting_started` — Quick start tutorial
- :doc:`examples` — Example gallery (113 examples)
- :doc:`api_reference` — API reference

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   getting_started
   examples
   api_reference
   contributing
   faq

Features
========

🔧 **Feature Engineering**
   Automated pipeline: lags, rolling stats, differencing, EMA, polynomial terms, STL, timestamps, record number.

🤖 **Gap-Filling**
   - Random Forest (R² 0.60-0.80)
   - XGBoost (R² 0.65-0.85)
   - Meteorological matching
   - Linear interpolation

🎯 **Quality Control**
   - Absolute limits
   - Hampel filter (spike detection)
   - Local standard deviation
   - Z-score (global and rolling)
   - Local Outlier Factor
   - Manual removal and trim-low

📊 **Visualization**
   - Time series plots
   - Heatmaps (datetime and regular)
   - Hexbin density plots
   - Diel cycles and cumulative curves
   - Histograms and quantile plots
   - Custom scatter and ridge plots

📈 **Analysis**
   - Correlation and autocorrelation
   - Trend decomposition
   - Seasonal and diurnal analysis
   - Harmonic analysis
   - Quantile analysis

Installation
============

Install via pip:

.. code-block:: bash

   pip install diive

Or with **uv** (modern, fast package manager):

.. code-block:: bash

   uv pip install diive

For development, clone the repository and install with uv:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   uv sync                    # Install dependencies
   uv run pytest tests/       # Run tests

See :doc:`installation` for more options (conda, poetry, etc.).

Quick Example
=============

.. code-block:: python

   import diive as dv

   TARGET = 'NEE_CUT_REF_orig'

   # Load the bundled example data (CH-DAV, 30-minute records)
   df = dv.load_exampledata_parquet()
   df = df.loc['2020', [TARGET, 'Tair_f', 'VPD_f', 'Rg_f']].copy()

   # Create engineered features
   engineer = dv.gapfilling.FeatureEngineer(
       target_col=TARGET,
       features_lag=[-2, -1],
       features_rolling=[12, 24],
       vectorize_timestamps=True,
   )
   df_engineered = engineer.fit_transform(df)

   # Gap-fill with Random Forest
   model = dv.gapfilling.RandomForestTS(
       input_df=df_engineered,
       target_col=TARGET,
       n_estimators=100,
       random_state=42,
   )
   model.run(showplot_scores=False, showplot_importance=False)
   gapfilled = model.results.gapfilled
   print(f"R2 on the held-out test set: {model.results.scores_traintest['r2']:.3f}")

Next Steps
==========

- **New to diive?** Start with the :ref:`Getting Started <getting_started>` guide.
- **Looking for examples?** Browse the :ref:`Example Gallery <examples_gallery>`.
- **Want the full API?** See the :ref:`API Reference <api_reference>`.
- **Contributing?** Check out :ref:`Contributing <contributing>`.

Support
=======

- 📖 Full documentation: https://diive.readthedocs.io/
- 🐛 Bug reports: https://github.com/holukas/diive/issues
- 💬 Discussions: https://github.com/holukas/diive/discussions
- 📦 PyPI: https://pypi.org/project/diive/
