.. _faq:

==========================
Frequently Asked Questions
==========================

Installation and setup
======================

**Q: What Python versions does DIIVE support?**

A: Python 3.12 and 3.13. Check your version:

   .. code-block:: bash

      python --version

**Q: I get "ModuleNotFoundError: No module named 'diive'"**

A: Install the package:

   .. code-block:: bash

      pip install diive

   Or with uv:

   .. code-block:: bash

      uv pip install diive

   For development, clone the repository and sync:

   .. code-block:: bash

      git clone https://github.com/holukas/diive.git
      cd diive
      uv sync

   If you installed into a conda environment, activate it first:

   .. code-block:: bash

      conda activate diive

**Q: Which version of DIIVE am I using?**

A: Print the version:

   .. code-block:: python

      import diive as dv

      print(dv.__version__)

**Q: Can I install DIIVE on Windows, macOS and Linux?**

A: Yes. The installation steps are the same on all three.

**Q: Is there a graphical user interface?**

A: Yes, a PySide6 desktop app. It ships as the optional ``gui`` extra:

   .. code-block:: bash

      uv sync --extra gui
      uv run diive-gui

   The 3D surface tabs need a second extra, ``gui3d``. The GUI only calls into
   the library, so anything you can do there can also be scripted.

Data and loading
================

**Q: How do I load my own data?**

A: Any pandas DataFrame with a datetime index works. Read it however you like:

   .. code-block:: python

      import pandas as pd

      df = pd.read_csv('mydata.csv', index_col=0, parse_dates=True)

   ``pd.read_excel`` works too, but it needs the ``openpyxl`` package, which
   DIIVE does not install.

   DIIVE reads parquet directly and has parsers for common flux and logger file
   formats:

   .. code-block:: python

      import diive as dv
      from diive.configs.filetypes import get_filetypes

      df = dv.load_parquet(filepath='mydata.parquet')

      print(sorted(get_filetypes().keys()))  # supported filetype IDs

      data_df, metadata_df = dv.ReadFileType(
          filetype='EDDYPRO-FLUXNET-CSV-30MIN',
          filepath='eddypro_fluxnet_output.csv',
      ).get_filedata()

**Q: What data format does DIIVE expect?**

A: A pandas DataFrame with:

   - a ``DatetimeIndex`` named ``TIMESTAMP_START``, ``TIMESTAMP_MIDDLE`` or
     ``TIMESTAMP_END``,
   - a regular frequency (30 min, hourly, daily, whatever your data is),
   - string column names and numeric values.

   ``TimestampSanitizer`` gets your index into that shape. It sorts, removes
   duplicates and NaT, fills date gaps so the index is continuous, and by
   default converts the timestamp to the middle of the averaging period:

   .. code-block:: python

      import pandas as pd
      import diive as dv

      idx = pd.date_range('2024-06-01 00:30', periods=96, freq='30min',
                          name='TIMESTAMP_END')
      df = pd.DataFrame({'TA': range(96)}, index=idx)

      df = dv.times.TimestampSanitizer(data=df, nominal_freq='30min').get()
      print(df.index.name)  # TIMESTAMP_MIDDLE

**Q: How do I load DIIVE's example data?**

A: The loaders live in ``diive.configs.exampledata``. Two of them are
   re-exported at the top level. None of them take arguments, each one loads one
   specific bundled file:

   .. code-block:: python

      import diive as dv

      df = dv.load_exampledata_parquet()  # CH-DAV, 2013-2022, 30-min

   To see all of them:

   .. code-block:: python

      from diive.configs import exampledata

      print([n for n in dir(exampledata) if n.startswith('load_exampledata')])

   The two used most often are ``load_exampledata_parquet`` (CH-DAV half-hourly
   fluxes and meteo) and ``load_exampledata_parquet_lae_level1_30MIN`` (CH-LAE
   EddyPro output, the input the flux processing chain expects).

**Q: My data has gaps (NaN values). Should I fill them before outlier detection?**

A: No. Work with the data as it is, the outlier detection methods handle NaN.
   The usual order is:

   1. detect and remove outliers,
   2. engineer features,
   3. train a gap-filling model,
   4. fill the gaps.

Gap-filling and feature engineering
===================================

**Q: What is the difference between the gap-filling methods?**

A: They differ in whether they train a model and in what they need as input:

   ======================== ======== ======================================= ==============================
   Method                   Training Features                                Use case
   ======================== ======== ======================================= ==============================
   ``RandomForestTS``       yes      engineered features                     robust, interpretable
   ``XGBoostTS``            yes      engineered features                     non-linear, fast
   ``SWINGapFillerXGBoost`` yes      SW_IN_POT, timestamps, opt. drivers     incoming shortwave radiation
   ``FluxMDS``              no       meteorological similarity               fluxes, ONEFlux-compatible
   ``linear_interpolation`` no       none                                    short gaps only
   ======================== ======== ======================================= ==============================

   All of them live in ``dv.gapfilling``. Start with Random Forest. MDS is the
   right choice when you need output comparable to FLUXNET or ONEFlux
   processing, since it is a faithful port of that algorithm.

**Q: How do I choose feature engineering parameters?**

A: Start with the defaults and adjust from there. Every stage is off unless you
   switch it on:

   .. code-block:: python

      import diive as dv

      df = dv.load_exampledata_parquet()
      df = df.loc['2022-06':'2022-07',
                  ['NEE_CUT_REF_orig', 'Tair_f', 'Rg_f', 'VPD_f']]

      engineer = dv.gapfilling.FeatureEngineer(
          target_col='NEE_CUT_REF_orig',
          features_lag=[-1, 1],                     # one step back and forward
          features_rolling=[12, 24],                # rolling window sizes
          features_rolling_stats=['mean', 'std'],
          features_diff=[1],                        # first-order differences
          features_ema=[6, 24],                     # exponential moving averages
          features_poly_degree=2,                   # squared terms
          features_stl=False,                       # STL is expensive
          vectorize_timestamps=True,                # year, month, hour, ...
      )
      df_engineered = engineer.fit_transform(df)

   ``target_col`` is required, but it only serves to keep the target out of the
   feature set. Engineered columns are named ``.{col}_TYPE{detail}``, for
   example ``.Tair_f_POL2``.

   See :ref:`Getting Started <getting_started>` for a full walk-through.

**Q: Can I use pre-engineered features with more than one model?**

A: Yes, that is what ``FeatureEngineer`` is for. Engineer once, then hand the
   same frame to as many models as you want:

   .. code-block:: python

      import diive as dv

      df = dv.load_exampledata_parquet()
      df = df.loc['2022-06':'2022-07',
                  ['NEE_CUT_REF_orig', 'Tair_f', 'Rg_f', 'VPD_f']]

      engineer = dv.gapfilling.FeatureEngineer(
          target_col='NEE_CUT_REF_orig',
          features_lag=[-1, 1],
          features_rolling=[12, 24],
          features_rolling_stats=['mean', 'std'],
          vectorize_timestamps=True,
      )
      df_engineered = engineer.fit_transform(df)

      rf = dv.gapfilling.RandomForestTS(input_df=df_engineered,
                                        target_col='NEE_CUT_REF_orig',
                                        n_estimators=20, random_state=42, verbose=0)
      rf.run()

      xgb = dv.gapfilling.XGBoostTS(input_df=df_engineered,
                                    target_col='NEE_CUT_REF_orig',
                                    n_estimators=20, random_state=42, verbose=0)
      xgb.run()

      print(f"RF  r2: {rf.results.scores_traintest['r2']:.3f}")
      print(f"XGB r2: {xgb.results.scores_traintest['r2']:.3f}")

      gapfilled = rf.results.gapfilled

**Q: Which score tells me how good the gap-filling is?**

A: ``results.scores_traintest``. That is the held-out score, measured on the
   test split the model never saw. ``results.scores`` is the in-sample score of
   the final model predicting on all complete rows including its own training
   rows, so it is optimistically biased.

   The test split is a random subset of the complete rows, not a temporal block.
   That is on purpose: gap-filling predicts each gap from the driver values at
   that timestamp, and gaps sit scattered among observed records, so a random
   hold-out reproduces the actual task. A block split answers a different
   question, namely how well the model transfers to an unseen period.

**Q: Why do two runs give different R2 scores on the same data?**

A: Three sources of variation, in order of size:

   - the train/test split, redrawn on every run,
   - the model itself (Random Forest bootstrapping, XGBoost subsampling),
   - SHAP importances, which are estimated on a subsample when
     ``shap_max_rows`` is set.

   Pin ``random_state`` on the model to make a run reproducible:

   .. code-block:: python

      import diive as dv

      model = dv.gapfilling.RandomForestTS(
          input_df=df_engineered,
          target_col='NEE_CUT_REF_orig',
          random_state=42,
      )

   ``np.random.seed()`` does not do this, scikit-learn and XGBoost use their own
   random state.

Outlier detection
=================

**Q: Which outlier detection method should I use?**

A: Depends on what you are looking for:

   - ``AbsoluteLimits``: fixed physical thresholds, for example air temperature
     between -50 and 50 degrees C.
   - ``Hampel``: robust spike detection from a rolling median and MAD.
   - ``LocalSD``: local standard deviation around a running median.
   - ``zScore`` / ``zScoreRolling`` / ``zScoreIncrements``: standard statistics,
     globally, in a rolling window, or on the increments between records.
   - ``LocalOutlierFactor``: density based.
   - ``ManualRemoval`` / ``TrimLow``: known bad periods, and trimming a low tail.

   All of them are in ``dv.outliers``. Start with ``AbsoluteLimits`` to get the
   physically impossible values out, then ``Hampel`` for spikes.

**Q: How do I chain several outlier detection methods?**

A: Use ``StepwiseOutlierDetection``. It is not exported on the ``dv`` namespace,
   import it from its module. Each test runs on what survived the previous one,
   and ``addflag()`` commits a test:

   .. code-block:: python

      import diive as dv
      from diive.preprocessing.outlier_detection import StepwiseOutlierDetection

      df = dv.load_exampledata_parquet().loc['2022-06':'2022-07',
                                             ['NEE_CUT_REF_orig']]

      sod = StepwiseOutlierDetection(
          dfin=df,
          col='NEE_CUT_REF_orig',
          site_lat=46.815,
          site_lon=9.855,
          utc_offset=1,
      )

      sod.flag_outliers_hampel_test(n_sigma=5.5, separate_day_night=True,
                                    showplot=False)
      sod.addflag()

      sod.flag_outliers_zscore_test(thres_zscore=3, showplot=False)
      sod.addflag()

      cleaned = sod.series_hires_cleaned
      flags = sod.flags

   See ``examples/preprocessing/outlier_detection/`` for complete scripts.

**Q: Why is Hampel not flagging my outliers?**

A: ``n_sigma`` is the sensitivity, and the default of 5.5 is deliberately
   conservative. Lower it and more records are flagged:

   .. code-block:: python

      import diive as dv

      series = dv.load_exampledata_parquet().loc['2022-06':'2022-07',
                                                 'NEE_CUT_REF_orig']

      for n_sigma in (5.5, 3, 2):
          hampel = dv.outliers.Hampel(
              series=series,
              lat=46.815, lon=9.855, utc_offset=1,
              n_sigma=n_sigma,
              separate_day_night=True,
          )
          hampel.run(repeat=True)
          n_flagged = int((hampel.overall_flag == 2).sum())
          print(f"n_sigma={n_sigma}: {n_flagged} records flagged")

   Two things to know about the arguments. The detector takes a *Series*, not a
   dataframe plus a column name, and ``lat`` / ``lon`` / ``utc_offset`` are only
   needed when ``separate_day_night=True``. The older ``dfin=`` / ``col=`` /
   ``site_lat=`` spelling now raises an error naming its replacement.

   With ``separate_day_night=True`` you can set ``n_sigma_daytime`` and
   ``n_sigma_nighttime`` separately. Both fall back to ``n_sigma``.

Visualization
=============

**Q: How do I create a time series plot?**

A: ``dv.plotting.TimeSeries`` takes one Series. Plotting classes work in two
   phases: the constructor takes data only, ``plot()`` takes the styling and the
   axes:

   .. code-block:: python

      import matplotlib.pyplot as plt
      import diive as dv

      df = dv.load_exampledata_parquet().loc['2022-06':'2022-07']

      fig, ax = plt.subplots(figsize=(14, 5))
      dv.plotting.TimeSeries(series=df['NEE_CUT_REF_orig']).plot(ax=ax)
      plt.show()

   With ``plot(ax=None)`` the class creates its own figure instead. See
   ``examples/visualization/`` for the other plot types.

**Q: How do I save a plot?**

A: Pass your own axes, then save its figure:

   .. code-block:: python

      import matplotlib.pyplot as plt
      import diive as dv

      df = dv.load_exampledata_parquet().loc['2022-06':'2022-07']

      fig, ax = plt.subplots(figsize=(14, 5))
      dv.plotting.TimeSeries(series=df['NEE_CUT_REF_orig']).plot(ax=ax)
      fig.savefig('myplot.png', dpi=300, bbox_inches='tight')

**Q: Can I customize plot colors and styles?**

A: Titles, labels, units, fonts, colors, grid and legend all go through
   ``FormatStyle``. Data rendering arguments such as ``color``, ``cmap`` and
   ``marker`` stay on ``plot()``:

   .. code-block:: python

      import matplotlib.pyplot as plt
      import diive as dv

      df = dv.load_exampledata_parquet().loc['2022-06':'2022-07']

      style = dv.plotting.FormatStyle(
          title='Net ecosystem exchange',
          ylabel='NEE',
          yunits='(umol m-2 s-1)',
          show_grid=False,
      )

      fig, ax = plt.subplots(figsize=(14, 5))
      ts = dv.plotting.TimeSeries(series=df['NEE_CUT_REF_orig'])
      ts.plot(ax=ax, format_style=style, color='#2196F3')

      # vary a single field
      ts.plot(ax=ax, format_style=style.merged(title='Zoom'))

   A bare ``FormatStyle()`` is the house style, every unset field falls back to
   the standard theme. You can still reach for matplotlib on the axes afterwards.

Flux processing
===============

**Q: What is the flux processing chain?**

A: The Swiss FluxNet post-processing workflow for eddy covariance data, in six
   levels:

   - **L2**: quality flag expansion from the EddyPro output
   - **L3.1**: storage correction (``FC`` plus storage term becomes ``NEE``)
   - **L3.2**: outlier removal
   - **L3.3**: USTAR filtering
   - **L4.1**: gap-filling (MDS, Random Forest, XGBoost)
   - **L4.2**: NEE partitioning into GPP and RECO (optional)

   Each level is a function that takes a ``FluxLevelData`` container and returns
   a new one. Nothing is mutated in place, so you can branch off any level:

   .. code-block:: python

      from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
      from diive.flux.fluxprocessingchain import (
          init_flux_data, run_level2, run_level31,
          make_level32_detector, run_level32,
          run_level33_constant_ustar, run_level41_mds,
      )

      df = load_exampledata_parquet_lae_level1_30MIN().loc['2024-07':'2024-07']
      # init_flux_data calculates these itself and refuses to overwrite them
      df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                            if c in df.columns])

      data = init_flux_data(df=df, fluxcol='FC', site_lat=47.478,
                            site_lon=8.364, utc_offset=1)

      data = run_level2(data, ssitc={'apply': True, 'setflag_timeperiod': None})
      data = run_level31(data, gapfill_storage_term=True)

      data, sod = make_level32_detector(data)
      sod.flag_outliers_hampel_test(n_sigma=5.5, separate_day_night=True,
                                    showplot=False)
      sod.addflag()
      data = run_level32(data, outlier_detector=sod)

      data = run_level33_constant_ustar(data, thresholds=[0.30],
                                        threshold_labels=['CUT_50'],
                                        showplot=False)

      data = run_level41_mds(data, swin='SW_IN_T1_47_1_gfXG',
                             ta='TA_T1_47_1_gfXG', vpd='VPD_T1_47_1_gfXG')

      final_df = data.fpc_df

   The full pipeline including partitioning is in
   ``examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py``.

   An older ``FluxProcessingChain`` class with methods such as
   ``level2_qualityflags()`` has been removed. The functions above replace it.

**Q: Is there a single call for the standard pipeline?**

A: Yes. Put the per-flux decisions into a ``FluxConfig`` and hand it to
   ``run_chain``:

   .. code-block:: python

      from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
      from diive.flux.fluxprocessingchain import FluxConfig, init_flux_data, run_chain

      df = load_exampledata_parquet_lae_level1_30MIN().loc['2024-07':'2024-07']
      df = df.drop(columns=[c for c in ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
                            if c in df.columns])

      cfg = FluxConfig(
          fluxcol='FC',
          ustar_thresholds=[0.30],
          ustar_labels=['CUT_50'],
          outlier_sigma_daytime=5.5,
          outlier_sigma_nighttime=5.5,
          level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
          gapfill_mds=True,
          gapfill_rf=False,
          gapfill_xgb=False,
          mds_swin='SW_IN_T1_47_1_gfXG',
          mds_ta='TA_T1_47_1_gfXG',
          mds_vpd='VPD_T1_47_1_gfXG',
      )

      data = init_flux_data(df=df, fluxcol='FC', site_lat=47.478,
                            site_lon=8.364, utc_offset=1)
      data = run_chain(data, cfg)

   ``run_chain`` picks fixed defaults for everything ``FluxConfig`` does not
   expose. Custom outlier logic, MDS tolerances and model hyperparameters are
   reachable only through the per-level functions.

**Q: What data do I need for USTAR filtering?**

A: A friction velocity column. ``init_flux_data`` reads it under the name given
   by ``ustarcol``, which defaults to ``'USTAR'``, and the column has to exist at
   that point. Levels 3.1 and 3.2 have to have run first, ``run_level33_*``
   raises otherwise: filtering outlier-contaminated data biases what the
   threshold does.

   .. code-block:: python

      # Continues from a chain that has already run levels 2, 3.1 and 3.2.
      # The USTAR column is named once, at init_flux_data(..., ustarcol='USTAR').

      data = run_level33_constant_ustar(
          data,
          thresholds=[0.10, 0.18, 0.25],
          threshold_labels=['CUT_16', 'CUT_50', 'CUT_84'],
      )

   USTAR filtering applies to CO2, CH4 and N2O only. For H and LE pass
   ``thresholds=[0], threshold_labels=['CUT_NONE']``, which flags nothing and
   keeps the level ordering intact.

   To derive the thresholds from the data instead of supplying them, use
   ``run_level33_ustar_detection``, which also needs an air temperature and an
   incoming shortwave radiation column.

**Q: How do I get results out of the chain?**

A: ``data.fpc_df`` is the working dataframe, everything the chain produced is a
   column in it. The lookup helpers give you the column names:

   .. code-block:: python

      cols = data.gapfilled_cols()
      # {'mds': {'CUT_50': 'NEE_L3.1_L3.3_CUT_50_QCF_gfMDS'}}

      gapfilled = data.fpc_df[cols['mds']['CUT_50']]

      # the filtered flux per USTAR scenario, before gap-filling
      after_l33 = data.levels.filteredseries_level33_qcf['CUT_50']

      print(data.summary())  # data availability per level, day and night

   ``data.partitioned_cols()`` does the same for the L4.2 GPP and RECO columns.
   The fitted objects sit in ``data.levels``, for example
   ``data.levels.level41_mds['CUT_50']``. After Level 3.3 ``data.filteredseries``
   is ``None``, because with several USTAR scenarios there is no single filtered
   series. Access those per scenario.

Debugging and performance
=========================

**Q: SHAP importance values keep changing between runs**

A: Expected, they move by roughly 5 to 10 percent. Pin ``random_state`` if you
   need a reproducible run, and use ranges rather than equality in tests:

   .. code-block:: python

      # good, allows the natural variability
      self.assertGreater(importance, 0.5)
      self.assertLess(importance, 0.9)

      # bad, too strict
      self.assertEqual(importance, 0.7234567)

**Q: Feature reduction is removing too many features**

A: ``reduce_features()`` adds a random benchmark column, computes SHAP
   importances, and rejects every feature whose importance is at or below
   ``random_importance + k * random_sd``, where ``k`` is
   ``shap_threshold_factor``. Lower it to keep more features:

   .. code-block:: python

      import diive as dv

      model = dv.gapfilling.XGBoostTS(input_df=df_engineered,
                                      target_col='NEE_CUT_REF_orig',
                                      n_estimators=20, random_state=42, verbose=0)
      model.reduce_features(shap_threshold_factor=0.3)  # default is 0.5
      model.run()

      print(model.accepted_features_)
      print(model.rejected_features_)

   ``model.results.feature_importances_reduction`` holds the SHAP table as it
   stood before anything was dropped, including the ``.RANDOM`` benchmark. It is
   the only view that carries the benchmark, so it is what to look at when a
   feature was dropped and you want to know why.

**Q: Training is taking too long**

A: Reduce the number of trees, and cap the rows SHAP has to explain. TreeSHAP
   cost is linear in the number of rows, and it is often the larger half of the
   runtime:

   .. code-block:: python

      import diive as dv

      model = dv.gapfilling.RandomForestTS(
          input_df=df_engineered,
          target_col='NEE_CUT_REF_orig',
          n_estimators=50,        # fewer trees
          n_jobs=-1,              # all cores
          shap_max_rows=10_000,   # seeded subsample for the SHAP pass
      )

   ``shap_max_rows`` leaves predictions and scores untouched. It only changes
   which rows the importances are estimated from, and the ranking converges long
   before the full record is used.

**Q: I am getting memory errors**

A: Cut the feature count or the record length. Every feature engineering stage
   multiplies the number of columns:

   .. code-block:: python

      import diive as dv

      df_small = df[::4]  # every 4th record

      engineer = dv.gapfilling.FeatureEngineer(
          target_col='NEE_CUT_REF_orig',
          features_lag=[-1, 1],    # fewer lags
          features_rolling=[12],   # fewer windows
          features_stl=False,      # STL is the expensive stage
      )

**Q: There is too much (or too little) console output**

A: Output goes through DIIVE's Rich console, controlled globally:

   .. code-block:: python

      import diive as dv

      dv.set_verbosity(0)  # silent
      dv.set_verbosity(1)  # errors and warnings only
      dv.set_verbosity(2)  # progress, the default
      dv.set_verbosity(3)  # debug

   Most classes also take their own ``verbose`` argument, which overrides the
   global level for that object.

**Q: Plots are not showing in Jupyter notebooks**

A: Set the matplotlib backend in the first cell:

   .. code-block:: python

      %matplotlib inline
      import matplotlib.pyplot as plt

Examples and documentation
==========================

**Q: Where are the examples?**

A: In the ``examples/`` folder of the repository, 113 scripts sorted by domain:
   ``analysis``, ``events``, ``features``, ``fits``, ``flux``, ``gapfilling``,
   ``io``, ``preprocessing``, ``times``, ``visualization``. They are plain
   standalone Python files, so clone the repository, edit them, and run them:

   .. code-block:: bash

      cd diive
      uv run python examples/gapfilling/gapfill_randomforest.py

   ``examples/CATALOG.md`` lists all of them with a one-line description. The
   rendered versions are in the example gallery of this documentation.

**Q: Can I run all examples at once?**

A: There is a runner:

   .. code-block:: bash

      uv run python examples/run_all_examples.py

   It runs all 113 of them and several take minutes, so expect a long wall time.

**Q: The example does not cover my use case**

A: Check :ref:`Getting Started <getting_started>` for the common workflows, and
   ``examples/COOKBOOK.md`` for task-oriented recipes. If that is not enough,
   open a discussion on GitHub.

Getting help
============

- **Documentation:** https://diive.readthedocs.io/
- **GitHub issues:** https://github.com/holukas/diive/issues (bug reports)
- **Discussions:** https://github.com/holukas/diive/discussions (questions)
- **PyPI:** https://pypi.org/project/diive/ (package info)
- **Contributing:** see :ref:`Contributing <contributing>`
