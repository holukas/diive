.. _getting_started:

===============
Getting Started
===============

Welcome to DIIVE! This guide introduces the core concepts and shows you how to get started with the library.

Core Concepts
=============

**Feature Engineering**
   DIIVE's feature engineering pipeline automatically creates 8 types of features from your data:
   lag features, rolling statistics, differencing, EMA, polynomial terms, STL decomposition, timestamps, and record numbers.
   This produces rich features for machine learning models.

**Gap-Filling**
   Gap-filling uses machine learning or statistical methods to estimate missing values in time series data.
   DIIVE supports Random Forest, XGBoost, and meteorological matching approaches.

**Quality Control & Outlier Detection**
   Before analyzing data, outliers and bad measurements must be removed or flagged.
   DIIVE provides 10+ methods including Hampel filters, z-scores, and Local Outlier Factor.

**Flux Processing Chain**
   For eddy covariance data, DIIVE implements a multi-level workflow following standardized protocols:
   Levels 2-4.1 handle quality flags, storage correction, USTAR filtering, and gap-filling.

Your First Analysis
===================

Here's a minimal example to get you started:

.. code-block:: python

   import diive as dv
   import matplotlib.pyplot as plt

   # 1. Load example data
   df = dv.load_exampledata_parquet(data_id='TLL')
   print(f"Loaded {len(df)} records")
   print(df.columns.tolist())

   # 2. Detect outliers
   detector = dv.AbsoluteLimits(
       dfin=df,
       col='NEE',
       lim_lower=-20,
       lim_upper=10,
   )
   detector.flag_outliers()
   clean_data = detector.get_flagged_data()

   # 3. Create engineered features for ML
   engineer = dv.FeatureEngineer(
       target_col='NEE',
       features_lag=[-1, 1],
       features_rolling=[12, 24],
       features_diff=[1],
       features_ema=[6, 24],
   )
   df_engineered = engineer.fit_transform(df)

   # 4. Train gap-filling model
   model = dv.RandomForestTS(
       input_df=df_engineered,
       target_col='NEE',
       n_estimators=100,
   )
   model.trainmodel()
   model.fillgaps()
   gapfilled = model.get_gapfilled_target()

   # 5. Visualize results
   plot = dv.TimeSeries(
       series=[df['NEE'], gapfilled],
       labels=['Original', 'Gap-filled'],
   )
   plot.plot()
   plt.show()

Working with Data
=================

**Loading Data**

Load example data for testing:

.. code-block:: python

   import diive as dv

   # Available datasets: 'TLL', 'CH-AWI', 'CH-CHA', etc.
   df = dv.load_exampledata_parquet(data_id='TLL')

Load your own data:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv('mydata.csv', index_col=0, parse_dates=True)
   # Ensure datetime index
   assert pd.api.types.is_datetime64_any_dtype(df.index)

**Data Structure**

DIIVE expects a pandas DataFrame with a datetime index:

.. code-block:: python

   # Good structure
   assert df.index.name == 'datetime'  # or similar
   assert pd.api.types.is_datetime64_any_dtype(df.index)

   # View data
   print(df.head())
   #                        NEE   TA   RH  SW_IN
   # datetime
   # 2020-01-01 00:00:00 -0.50  5.2  0.85  0.0
   # 2020-01-01 01:00:00 -0.45  4.8  0.87  0.0
   # ...

Common Workflows
================

**Workflow 1: Quality Control**

Remove outliers before analysis:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet(data_id='TLL')

   # Step 1: Absolute limits
   detector = dv.AbsoluteLimits(dfin=df, col='NEE', lim_lower=-20, lim_upper=10)
   detector.flag_outliers()
   df = detector.get_flagged_data()

   # Step 2: Hampel filter for spikes
   detector = dv.Hampel(dfin=df, col='NEE', site_lat=47.286, site_lon=7.734)
   detector.flag_outliers_hampel_test(n_sigma=5.5)
   df = detector.get_flagged_data()

   # Continue with clean data
   print(f"Removed {detector.n_flagged_total} outliers")

**Workflow 2: Gap-Filling**

Fill missing values with machine learning:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet(data_id='TLL')

   # Create features once, reuse across models
   engineer = dv.FeatureEngineer(
       target_col='NEE',
       features_lag=[-1, 1],
       features_rolling=[12, 24],
       features_diff=[1],
       features_ema=[6, 24],
       vectorize_timestamps=True,
   )
   df_engineered = engineer.fit_transform(df)

   # Try Random Forest
   rf_model = dv.RandomForestTS(
       input_df=df_engineered,
       target_col='NEE',
       n_estimators=100,
   )
   rf_model.trainmodel()
   rf_model.fillgaps()
   rf_gapfilled = rf_model.get_gapfilled_target()

   # Try XGBoost
   xgb_model = dv.XGBoostTS(
       input_df=df_engineered,
       target_col='NEE',
       n_estimators=100,
   )
   xgb_model.trainmodel()
   xgb_model.fillgaps()
   xgb_gapfilled = xgb_model.get_gapfilled_target()

   # Compare results
   print(f"RF R²: {rf_model.r2_test_pred:.3f}")
   print(f"XGB R²: {xgb_model.r2_test_pred:.3f}")

**Workflow 3: Visualization**

Create publication-ready plots:

.. code-block:: python

   import diive as dv
   import matplotlib.pyplot as plt

   df = dv.load_exampledata_parquet(data_id='TLL')

   # Time series plot
   plot = dv.TimeSeries(series=[df['NEE']], labels=['NEE'], figsize=(12, 4))
   plot.plot()
   plt.show()

   # Heatmap (daily pattern)
   plot = dv.HeatmapDateTime(
       series=df['TA'],
       label='Temperature',
       figsize=(14, 6),
   )
   plot.plot()
   plt.show()

   # Diel cycle (average daily pattern)
   plot = dv.DielCycle(
       series=df['LE'],
       label='Latent Heat Flux',
   )
   plot.plot()
   plt.show()

**Workflow 4: Time Series Analysis**

Decompose trends and seasonality:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet(data_id='TLL')

   # STL decomposition (Trend, Seasonal, Residual)
   decomposer = dv.SeasonalTrendDecomposition(
       series=df['TA'],
       periods=48,  # 2 days for 30-min data
   )
   trend, seasonal, residual = decomposer.decompose()

   # Plot decomposition
   decomposer.plot()

Example Gallery
===============

For more examples, browse the :ref:`Example Gallery <auto_examples/index>`:

- **Visualization:** Time series, heatmaps, histograms, diel cycles, and more
- **Gap-Filling:** Random Forest, XGBoost, MDS, linear interpolation
- **Outlier Detection:** Hampel, z-score, Local Outlier Factor, step-wise orchestration
- **Analysis:** Correlation, decomposition, trend analysis, harmonic analysis
- **Flux Processing:** Multi-level quality control and gap-filling

API Reference
=============

For detailed API documentation, see the :ref:`API Reference <api_reference>`.

Key Classes:
   - :py:class:`diive.core.ml.FeatureEngineer` — Feature creation pipeline
   - :py:class:`diive.pkgs.gapfilling.RandomForestTS` — Random Forest gap-filling
   - :py:class:`diive.pkgs.gapfilling.XGBoostTS` — XGBoost gap-filling
   - :py:class:`diive.pkgs.preprocessing.outlierdetection.AbsoluteLimits` — Simple threshold outlier detection
   - :py:class:`diive.pkgs.preprocessing.outlierdetection.Hampel` — Robust spike detection
   - :py:class:`diive.core.plotting.TimeSeries` — Time series visualization

Helpful Resources
=================

- :ref:`FAQ <faq>` — Common questions and troubleshooting
- :ref:`Contributing <contributing>` — How to contribute and development setup
- `GitHub Issues <https://github.com/holukas/diive/issues>`_ — Report bugs and request features
- `DIIVE GitHub <https://github.com/holukas/diive>`_ — Source code and repository

Tips
====

1. **Use FeatureEngineer once** — Create features once and reuse them with multiple gap-filling models.

2. **Start simple** — Begin with absolute limits for outlier detection, then refine.

3. **Validate on examples** — Test your workflows on the provided example data first.

4. **Check docstrings** — Most classes have detailed docstrings with parameters explained.

5. **Explore examples** — The :ref:`Example Gallery <auto_examples/index>` has 94+ working examples covering all features.

Next Steps
==========

- Start with a simple quality control workflow
- Explore the :ref:`Example Gallery <auto_examples/index>` for your use case
- Read the :ref:`API Reference <api_reference>` for detailed documentation
- Check the :ref:`FAQ <faq>` for common issues
