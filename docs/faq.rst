.. _faq:

==========================
Frequently Asked Questions
==========================

Installation & Setup
====================

**Q: What Python versions does DIIVE support?**

A: DIIVE requires Python 3.12 or 3.13. We recommend 3.13 for the best experience.
   Check your version:

   .. code-block:: bash

      python --version

**Q: I get "ModuleNotFoundError: No module named 'diive'"**

A: Make sure DIIVE is installed:

   .. code-block:: bash

      pip install diive

   Or with uv:

   .. code-block:: bash

      uv pip install diive

   For development, clone and install:

   .. code-block:: bash

      git clone https://github.com/holukas/diive.git
      cd diive
      uv sync

   If using conda, activate the environment:

   .. code-block:: bash

      conda activate diive

**Q: Which version of DIIVE am I using?**

A: Check the version in Python:

   .. code-block:: python

      import diive as dv
      print(dv.__version__)

**Q: Can I install DIIVE on Windows/Mac/Linux?**

A: Yes, DIIVE works on all major platforms. Use the same installation steps for all.

Data & Loading
==============

**Q: How do I load my own data?**

A: Load any pandas DataFrame:

   .. code-block:: python

      import pandas as pd
      import diive as dv

      # Read from CSV
      df = pd.read_csv('mydata.csv', index_col=0, parse_dates=True)

      # Read from Parquet
      df = pd.read_parquet('mydata.parquet')

      # Read from Excel
      df = pd.read_excel('mydata.xlsx', index_col=0, parse_dates=True)

**Q: What data format does DIIVE expect?**

A: DIIVE expects a pandas DataFrame with:
   - Datetime index (hourly, 30-min, or any regular frequency)
   - Column names as strings
   - Numeric values (floats or ints)

   Example:

   .. code-block:: python

      import pandas as pd
      df = pd.DataFrame(
          {'NEE': [-0.5, -0.45, ...], 'TA': [5.2, 4.8, ...]},
          index=pd.date_range('2020-01-01', periods=1000, freq='h')
      )
      df.index.name = 'datetime'

**Q: How do I load DIIVE's example data?**

A: Use the built-in loader:

   .. code-block:: python

      import diive as dv

      # Available: 'TLL', 'CH-AWI', 'CH-CHA', 'CH-CRM', 'CH-DAV', 'CH-LAE'
      df = dv.load_exampledata_parquet(data_id='TLL')

   See what's available:

   .. code-block:: python

      import diive as dv
      print(dv.configs.exampledata.EXAMPLE_DATA_INFO)

**Q: My data has gaps (NaN values). Should I fill them before outlier detection?**

A: Generally, work with your data as-is. Most outlier detection methods handle NaNs.
   For gap-filling, you need clean data first:

   1. Apply outlier detection → remove bad values
   2. Create engineered features
   3. Train gap-filling model
   4. Fill gaps in missing values

Gap-Filling & Feature Engineering
==================================

**Q: What's the difference between different gap-filling methods?**

A: Each method has trade-offs:

   ============= ============= ========== ======== ============
   Method        Training      Speed      Accuracy Notes
   ============= ============= ========== ======== ============
   Random Forest Yes           3-8 min    R² 0.60+ Interpretable
   XGBoost       Yes           2-5 min    R² 0.65+ Better accuracy
   MDS           No            Very fast  R² 0.40+ No training needed
   Linear        No            Instant    Simple   Small gaps only
   ============= ============= ========== ======== ============

   Start with Random Forest, try XGBoost if you want higher accuracy and have time.

**Q: How do I choose feature engineering parameters?**

A: Start with defaults and adjust based on results:

   .. code-block:: python

      engineer = dv.FeatureEngineer(
          target_col='NEE',
          features_lag=[-1, 1],              # Look 1 step back/forward
          features_rolling=[12, 24],         # 12 and 24 step rolling stats
          features_diff=[1],                 # First-order differences
          features_ema=[6, 24],              # Exponential moving averages
          features_poly_degree=2,            # Squared terms
          features_stl=False,                # Trend/seasonal/residual (expensive)
          vectorize_timestamps=True,         # Add year, month, hour, etc.
      )

   See the :ref:`Getting Started <getting_started>` guide for more details.

**Q: Why do different models give different R² scores on the same data?**

A: Normal variability due to:
   - Random forest stochasticity
   - Different train/test splits
   - Different hyperparameters
   - Floating-point arithmetic

   Use consistent random seeds for reproducibility:

   .. code-block:: python

      import numpy as np
      np.random.seed(42)

**Q: Can I use pre-engineered features with multiple models?**

A: Yes! That's the whole point of ``FeatureEngineer``:

   .. code-block:: python

      # Engineer once
      engineer = dv.FeatureEngineer(...)
      df_engineered = engineer.fit_transform(df)

      # Use with multiple models
      rf = dv.RandomForestTS(input_df=df_engineered, target_col='NEE')
      rf.trainmodel()

      xgb = dv.XGBoostTS(input_df=df_engineered, target_col='NEE')
      xgb.trainmodel()

      # Compare results
      print(f"RF: {rf.r2_test_pred:.3f}, XGB: {xgb.r2_test_pred:.3f}")

Outlier Detection
=================

**Q: Which outlier detection method should I use?**

A: Depends on your goal:

   - **Absolute limits**: Simple thresholds (temperature -50 to 50°C)
   - **Hampel**: Robust spike detection (sudden jumps)
   - **LocalSD**: Adaptive thresholding for diel patterns
   - **Z-score**: Standard statistical approach
   - **Local Outlier Factor**: Density-based (for multivariate outliers)
   - **Manual**: Explicit removal of known bad periods

   Start with Hampel or Absolute Limits, refine with Z-score.

**Q: How do I chain multiple outlier detection methods?**

A: Use ``StepwiseOutlierDetection`` to apply methods sequentially:

   .. code-block:: python

      import diive as dv

      detector = dv.StepwiseOutlierDetection(
          dfin=df,
          col='NEE',
          site_lat=47.286,
          site_lon=7.734,
          utc_offset=1,
      )

      # Step 1: Aggressive detection
      detector.flag_outliers_hampel_dtnt_test(n_sigma=5.5)
      detector.addflag()

      # Step 2: Refine
      detector.flag_outliers_zscore_test(thres_zscore=3)
      detector.addflag()

      # Results
      cleaned = detector.series_hires_cleaned

   See ``examples/outlierdetection/stepwise.py`` for complete example.

**Q: Why isn't Hampel detecting my outliers?**

A: The ``n_sigma`` parameter controls sensitivity:

   .. code-block:: python

      detector = dv.Hampel(dfin=df, col='NEE', ...)

      # Larger n_sigma = less sensitive
      detector.flag_outliers_hampel_test(n_sigma=5.5)  # Aggressive

      # Smaller n_sigma = more sensitive
      detector.flag_outliers_hampel_test(n_sigma=2)    # Lenient

   Start with 3-4 and adjust.

Visualization
=============

**Q: How do I create a time series plot?**

A: Use the ``TimeSeries`` class:

   .. code-block:: python

      import diive as dv
      import matplotlib.pyplot as plt

      df = dv.load_exampledata_parquet(data_id='TLL')

      plot = dv.TimeSeries(
          series=[df['NEE'], df['TA']],
          labels=['NEE', 'TA'],
          figsize=(14, 6),
      )
      plot.plot()
      plt.show()

   See ``examples/visualization/`` for more plot types.

**Q: How do I save plots?**

A: Use matplotlib's ``savefig``:

   .. code-block:: python

      plot = dv.TimeSeries(series=[df['NEE']])
      plot.plot()
      plt.savefig('myplot.png', dpi=300, bbox_inches='tight')

**Q: Can I customize plot colors and styles?**

A: Yes, use matplotlib directly after creating the plot:

   .. code-block:: python

      plot = dv.TimeSeries(series=[df['NEE']])
      ax = plot.plot()
      ax.set_ylabel('NEE (µmol/m²/s)', fontsize=14)
      ax.set_title('Net Ecosystem Exchange', fontsize=16)
      plt.show()

Flux Processing
===============

**Q: What's the FluxProcessingChain?**

A: A complete multi-level workflow for eddy covariance data:

   - **Level 2**: Quality flags
   - **Level 3.1**: Storage correction
   - **Level 3.2-3.3**: USTAR filtering
   - **Level 4.1**: Gap-filling

   .. code-block:: python

      fpc = dv.FluxProcessingChain(df, site_lat=47.286, site_lon=7.734)

      fpc.level2_qualityflags(cols=['FC', 'LE', 'H'])
      fpc.level31_storagecorrection(...)
      fpc.level33_ustarfiltering(...)
      fpc.level41_longterm_random_forest(...)

      result = fpc.level41['long_term_random_forest']['CUT_50']

**Q: What data do I need for USTAR filtering?**

A: ``u*`` (friction velocity) and air temperature data:

   .. code-block:: python

      fpc.level33_ustarfiltering(
          col_ustar='USTAR',
          col_temp='TA',
      )

**Q: How do I access results from FluxProcessingChain?**

A: Results are stored in hierarchical dictionaries:

   .. code-block:: python

      # Get gap-filled flux for a specific USTAR percentile
      filled_nee = fpc.level41['long_term_random_forest']['CUT_50']

      # Get storage-corrected flux
      corrected_flux = fpc.level31['storage_corrected_fc']

   See FluxProcessingChain docstring for full structure.

Debugging & Performance
=======================

**Q: SHAP importance values keep changing between runs**

A: Normal variability (±5-10%). Use flexible assertions in tests:

   .. code-block:: python

      # Good: allows natural variability
      self.assertGreater(importance, 0.5)
      self.assertLess(importance, 0.9)

      # Bad: too strict
      self.assertEqual(importance, 0.7234567)

**Q: Feature reduction is removing too many features**

A: Reduce the ``shap_threshold_factor`` in your gap-filling model:

   .. code-block:: python

      # Default is 0.5 (strict)
      model = dv.RandomForestTS(
          input_df=df_engineered,
          target_col='NEE',
          shap_threshold_factor=0.3,  # More lenient
      )

**Q: Training is taking too long**

A: Reduce ``n_estimators`` or sample size:

   .. code-block:: python

      model = dv.RandomForestTS(
          input_df=df_engineered,
          target_col='NEE',
          n_estimators=50,  # Default 100, reduce for faster training
          train_fraction=0.7,
      )

**Q: I'm getting memory errors**

A: Use a smaller dataset or reduce number of features:

   .. code-block:: python

      # Subsample data
      df_small = df[::4]  # Every 4th row

      engineer = dv.FeatureEngineer(
          target_col='NEE',
          features_lag=[-1, 1],  # Fewer lags
          features_rolling=[12],  # Fewer rolling windows
      )

**Q: Plot isn't showing in Jupyter notebooks**

A: Make sure matplotlib backend is set:

   .. code-block:: python

      %matplotlib inline
      import matplotlib.pyplot as plt

      # Then create plots normally

Examples & Documentation
========================

**Q: Where are the examples?**

A: Browse the :ref:`Example Gallery <auto_examples/index>` with 103 runnable examples across 52 files.

Or clone and run locally:

.. code-block:: bash

   cd diive
   uv run python examples/gap_filling/randomforest_ts.py

**Q: Can I run all examples at once?**

A: Yes, use the example runner (parallel):

   .. code-block:: bash

      uv run python examples/run_all_examples.py

**Q: The example doesn't work for my use case**

A: Check the :ref:`Getting Started <getting_started>` guide or :ref:`Contributing <contributing>` guide.
   Common workflows are documented with code.

**Q: Can I modify and run examples locally?**

A: Yes! Examples are standalone Python files. Edit and run:

   .. code-block:: bash

      python examples/visualization/timeseries.py

Getting Help
============

- **Documentation:** https://diive.readthedocs.io/
- **GitHub Issues:** https://github.com/holukas/diive/issues (bug reports)
- **Discussions:** https://github.com/holukas/diive/discussions (questions)
- **PyPI:** https://pypi.org/project/diive/ (package info)

Still have questions? Open an issue or discussion on GitHub!
