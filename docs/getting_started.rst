.. _getting_started:

===============
Getting Started
===============

This guide introduces the core concepts of diive and shows the workflows most users
start with: quality control, gap-filling, plotting and time series analysis.

``import diive as dv`` gives access to ten domain namespaces: ``dv.outliers``,
``dv.gapfilling``, ``dv.flux``, ``dv.analysis``, ``dv.plotting``, ``dv.times``,
``dv.variables``, ``dv.corrections``, ``dv.qaqc`` and ``dv.events``. A few utilities
such as the example-data loaders sit directly on ``dv``. Every code sample below runs
against the bundled example data.

Core Concepts
=============

**Feature Engineering**
   The feature engineering pipeline builds 8 types of features from your data:
   lag features, rolling statistics, differencing, EMA, polynomial terms, STL
   decomposition, timestamp components and a record number. The result feeds the
   machine learning gap-fillers.

**Gap-Filling**
   Gap-filling estimates missing values in a time series. diive provides Random
   Forest, XGBoost, the MDS meteorological-similarity method and linear interpolation.

**Quality Control & Outlier Detection**
   Outliers and bad measurements are flagged before analysis. diive has more than ten
   detection methods, including Hampel filters, z-scores, local standard deviation and
   Local Outlier Factor. Every detector returns both a flag series and a filtered
   series, so nothing is deleted silently.

**Flux Processing Chain**
   For eddy covariance data, diive implements the Swiss FluxNet post-processing
   workflow: Level 2 (quality flags), Level 3.1 (storage correction), Level 3.2
   (outlier removal), Level 3.3 (u* filtering), Level 4.1 (gap-filling) and
   Level 4.2 (NEE partitioning into GPP and RECO).

Your First Analysis
===================

A minimal end-to-end run: load data, remove outliers, engineer features, gap-fill,
plot.

.. code-block:: python

   import matplotlib.pyplot as plt

   import diive as dv

   # 1. Load the bundled example data (CH-DAV, 2013-2022, 30-minute records)
   df = dv.load_exampledata_parquet()
   df = df.loc['2020'].copy()
   print(f"{len(df)} records, {len(df.columns)} variables")

   # 2. Flag implausible values in the CO2 flux and remove them
   detector = dv.outliers.AbsoluteLimits(series=df['NEE_CUT_REF_orig'], minval=-20, maxval=10)
   detector.calc()
   print(f"{(detector.flag == 2).sum()} records flagged as outliers")

   # 3. Build features for the gap-filling model
   model_df = df[['Tair_f', 'VPD_f', 'Rg_f']].copy()
   model_df['NEE'] = detector.filteredseries
   engineer = dv.gapfilling.FeatureEngineer(
       target_col='NEE',
       features_lag=[-1, -1],
       features_rolling=[12],
       vectorize_timestamps=True,
   )
   df_engineered = engineer.fit_transform(model_df)

   # 4. Train a Random Forest and fill the gaps
   model = dv.gapfilling.RandomForestTS(
       input_df=df_engineered,
       target_col='NEE',
       verbose=0,
       n_estimators=20,
       random_state=42,
       n_jobs=-1,
       shap_max_rows=500,
   )
   model.trainmodel(showplot_scores=False, showplot_importance=False)
   model.fillgaps(showplot_scores=False, showplot_importance=False)
   gapfilled = model.results.gapfilled
   print(f"R2 on the held-out test set: {model.scores_traintest_['r2']:.3f}")

   # 5. Plot the result
   fig, ax = plt.subplots(figsize=(12, 4))
   dv.plotting.TimeSeries(series=gapfilled).plot(ax=ax)
   plt.show()

``n_estimators=20`` keeps the sample fast. Use a few hundred trees for real work.

Working with Data
=================

**Loading Data**

The example datasets ship with the package and take no arguments:

.. code-block:: python

   import diive as dv

   # CH-DAV eddy covariance fluxes and meteo, 2013-2022, 30-minute records
   df = dv.load_exampledata_parquet()
   print(df.shape)
   print(df.index.name, df.index.min(), df.index.max())

   # CH-LAE flux processing chain data, 2016-2017
   df_lae = dv.load_exampledata_parquet_lae()

More loaders for specific file formats (EddyPro, FLUXNET, TOA5, ICOS, generic CSV)
live in ``diive.configs.exampledata``.

For your own data, ``dv.load_parquet`` reads a parquet file and ``dv.ReadFileType``
reads the supported text formats. Anything pandas can read works too, as long as the
result ends up with a datetime index.

**Data Structure**

diive expects a pandas DataFrame with a datetime index named following the diive
convention (``TIMESTAMP_MIDDLE``, ``TIMESTAMP_START`` or ``TIMESTAMP_END``).
``TimestampSanitizer`` checks the naming, detects the time resolution, removes
duplicates and fills date gaps with NaN rows:

.. code-block:: python

   import numpy as np
   import pandas as pd

   import diive as dv

   index = pd.date_range('2020-01-01 00:15:00', periods=48 * 30, freq='30min',
                         name='TIMESTAMP_MIDDLE')
   df = pd.DataFrame({'TA': np.random.normal(5, 3, len(index))}, index=index)

   sanitizer = dv.times.TimestampSanitizer(data=df, nominal_freq='30min', verbose=False)
   df = sanitizer.get()
   print(sanitizer.get_status())

Common Workflows
================

**Workflow 1: Quality Control**

Outlier detectors take a Series, not a DataFrame. Call ``.calc()``, then read the
results from ``.flag`` (0 = fine, 2 = outlier, NaN = no test possible) and
``.filteredseries`` (the input with flagged records set to NaN). Chain tests by
feeding the filtered series of one detector into the next:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet()
   series = df.loc['2020', 'NEE_CUT_REF_orig']

   # Step 1: absolute limits
   abslim = dv.outliers.AbsoluteLimits(series=series, minval=-20, maxval=10)
   abslim.calc()

   # Step 2: Hampel filter for spikes, applied to what the first test left behind
   hampel = dv.outliers.Hampel(
       series=abslim.filteredseries,
       lat=46.815,
       lon=9.855,
       utc_offset=1,
       n_sigma=5.5,
       window_length=48 * 13,
   )
   hampel.calc(repeat=True)

   cleaned = hampel.filteredseries
   n_removed = series.notna().sum() - cleaned.notna().sum()
   print(f"{n_removed} of {series.notna().sum()} measured records removed")

``Hampel`` separates daytime and nighttime by default, which is why it needs ``lat``,
``lon`` and ``utc_offset``. For one threshold over all records, pass
``separate_day_night=False``.

For longer chains with a combined quality flag, use ``StepwiseOutlierDetection``
from ``diive.preprocessing.outlier_detection`` together with ``dv.qaqc.FlagQCF``.

**Workflow 2: Gap-Filling**

Engineer the features once and reuse them across models:

.. code-block:: python

   import diive as dv

   TARGET = 'NEE_CUT_REF_orig'

   df = dv.load_exampledata_parquet()
   df = df.loc['2020', [TARGET, 'Tair_f', 'VPD_f', 'Rg_f']].copy()

   engineer = dv.gapfilling.FeatureEngineer(
       target_col=TARGET,
       features_lag=[-2, -1],
       features_rolling=[12, 24],
       features_diff=[1],
       features_ema=[6, 24],
       vectorize_timestamps=True,
   )
   df_engineered = engineer.fit_transform(df)

   rf = dv.gapfilling.RandomForestTS(
       input_df=df_engineered, target_col=TARGET, verbose=0,
       n_estimators=20, random_state=42, n_jobs=-1, shap_max_rows=500)
   rf.run(showplot_scores=False, showplot_importance=False)

   xgb = dv.gapfilling.XGBoostTS(
       input_df=df_engineered, target_col=TARGET, verbose=0,
       n_estimators=20, random_state=42, n_jobs=-1, shap_max_rows=500)
   xgb.run(showplot_scores=False, showplot_importance=False)

   print(f"Random Forest R2 (held-out): {rf.results.scores_traintest['r2']:.3f}")
   print(f"XGBoost R2 (held-out):       {xgb.results.scores_traintest['r2']:.3f}")

   gapfilled = rf.results.gapfilled
   flag = rf.results.flag  # 0 = observed, 1 = gap-filled, 2 = fallback
   print(f"{(flag > 0).sum()} of {len(flag)} records filled")

``.run()`` is shorthand for ``.trainmodel()`` followed by ``.fillgaps()``. Results are
collected in ``.results``: the gap-filled series, the flag, the in-sample scores
(``.scores``), the held-out scores (``.scores_traintest``), the SHAP feature
importances and the trained model.

``.scores_traintest`` comes from a random split of the complete rows, not a temporal
block. That is the right test for gap-filling, where gaps sit scattered among observed
records.

**Workflow 3: Visualization**

Plot classes are two-phase. The constructor takes data and computation parameters, and
``plot()`` takes the axes and all styling. Chrome such as title, labels, units, font
sizes and grid goes through ``FormatStyle``:

.. code-block:: python

   import matplotlib.pyplot as plt

   import diive as dv

   df = dv.load_exampledata_parquet()
   df = df.loc['2020'].copy()

   # Time series
   fig, ax = plt.subplots(figsize=(12, 4))
   dv.plotting.TimeSeries(series=df['Tair_f']).plot(
       ax=ax, format_style=dv.plotting.FormatStyle(title='Air temperature', yunits='(degC)'))
   plt.show()

   # Heatmap: one row per date, one column per time of day
   fig, ax = plt.subplots(figsize=(6, 9))
   dv.plotting.HeatmapDateTime(series=df['Tair_f']).plot(ax=ax)
   plt.show()

   # Diel cycle: median with interquartile band, one curve per month
   fig, ax = plt.subplots(figsize=(9, 5))
   dv.plotting.DielCycle(series=df['LE_f']).plot(
       ax=ax, agg='median', band='iqr', each_month=True)
   plt.show()

Because ``plot()`` accepts an ``ax``, the same object can be drawn several times with
different styling, and diive plots can be placed into your own matplotlib figures.

**Workflow 4: Time Series Analysis**

Split a series into trend, seasonal and residual components:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet()
   series = df.loc['2020', 'Tair_f']

   # A seasonal period of 48 records is one day at 30-minute resolution
   stl = dv.analysis.SeasonalTrendDecomposition(series=series, method='stl',
                                                seasonal_period=48)

   trend = stl.trend
   seasonal = stl.seasonal
   residual = stl.residual

   print(stl.summary())
   print(f"Seasonality strength: {stl.seasonality_strength:.3f}")

The components are computed on first access and cached. ``detrend()`` and
``deseasonalize()`` return the series with one component removed.

Examples
========

The repository ships runnable example scripts under
`examples/ <https://github.com/holukas/diive/tree/main/examples>`__, grouped by topic:

- **visualization**: time series, heatmaps, histograms, diel cycles, ridgelines, wind roses
- **gapfilling**: Random Forest, XGBoost, MDS, linear interpolation, long-term filling
- **preprocessing**: outlier detection, corrections, stepwise screening
- **flux**: processing chain, u* threshold detection, NEE partitioning, uncertainty
- **analysis**: correlation, decomposition, gap statistics, harmonic analysis
- **times**, **features**, **fits**, **io**, **events**: supporting tools

`examples/CATALOG.md <https://github.com/holukas/diive/blob/main/examples/CATALOG.md>`_
lists all of them. The same scripts are rendered as a gallery in the built
documentation.

API Reference
=============

See the :ref:`API Reference <api_reference>` for the full list of exported symbols,
with one page per namespace:

- :ref:`dv.outliers <api_outliers>`: detection methods, including ``AbsoluteLimits`` and ``Hampel``
- :ref:`dv.gapfilling <api_gapfilling>`: ``FeatureEngineer``, ``RandomForestTS``, ``XGBoostTS``, ``FluxMDS``
- :ref:`dv.plotting <api_plotting>`: ``TimeSeries``, ``HeatmapDateTime``, ``DielCycle``, ``FormatStyle``
- :ref:`dv.analysis <api_analysis>`: ``SeasonalTrendDecomposition``, ``GapStats``, ``Histogram``
- :ref:`dv.flux <api_flux>`: processing chain, u* filtering, partitioning, uncertainty
- :ref:`dv.times <api_times>`: ``TimestampSanitizer``, ``DetectFrequency``, resampling
- :ref:`dv.variables <api_variables>`, :ref:`dv.corrections <api_corrections>`,
  :ref:`dv.qaqc <api_qaqc>`, :ref:`dv.events <api_events>`

Helpful Resources
=================

- :ref:`FAQ <faq>`: common questions and troubleshooting
- :ref:`Contributing <contributing>`: how to contribute and set up a development environment
- `GitHub Issues <https://github.com/holukas/diive/issues>`_: bug reports and feature requests
- `diive on GitHub <https://github.com/holukas/diive>`_: source code

Tips
====

1. **Sanitize the timestamp first.** Almost every diive workflow assumes a regular,
   correctly named datetime index.

2. **Engineer features once.** The same engineered DataFrame can feed several
   gap-filling models, which makes model comparison cheap and fair.

3. **Start simple.** Absolute limits catch the obvious problems. Add the
   statistics-based detectors afterwards and check what each one removes.

4. **Read the docstrings.** Parameters, expected units and the day/night arguments are
   documented on each class. Units are not validated at runtime.

5. **Copy from the examples.** The scripts under ``examples/`` are kept runnable
   against the current API.

Next Steps
==========

- Run the quality control workflow above on one of your own variables
- Browse `examples/ <https://github.com/holukas/diive/tree/main/examples>`__ for your use case
- Read the :ref:`API Reference <api_reference>` for the full symbol list
- Check the :ref:`FAQ <faq>` if something does not behave as expected
