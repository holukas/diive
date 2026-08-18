Eddy Covariance Flux Processing Examples
========================================

Examples demonstrating flux processing, quality control, and gap-filling for averaged (e.g. 30-minute) eddy
covariance data. Raw high-frequency (10/20 Hz) tooling lives in `dyco <https://github.com/holukas/dyco>`_.

Terminology
-----------

**Directory abbreviations** used throughout flux processing examples:

- ``lowres/`` — Low-resolution (e.g., 30-minute) flux data processing. Typically averaged or aggregated time series.

Contents
--------

Processing Chain
~~~~~~~~~~~~~~~~

- **fluxprocessingchain/fluxprocessingchain_level2.py** — Level 2 in isolation: load a real EddyPro FLUXNET output file, ``init_flux_data``, then ``run_level2`` to expand the EddyPro quality diagnostics into per-test flags and one overall QCF. Shows ``level2_test_inputs`` (which column each test reads), the QCF-filtered vs. high-quality (QCF=0) series, and the effect of the accept threshold. The smallest standalone entry point into the chain.
- **fluxprocessingchain/fluxprocessingchain_runchain.py** — Single-call ``run_chain(data, FluxConfig)`` example. Minimal config drives the full L2→L4.2 pipeline with sensible defaults. The easy path; use this when you want the chain to "just work".
- **fluxprocessingchain/fluxprocessingchain_composable.py** — Full L2→L4.2 pipeline using composable callables; RF, XGBoost, and MDS gap-filling from the same L3.3 state; on-demand ``gap_stats()`` after L3.3; ``plot_gapfilled_heatmaps()`` (side-by-side heatmap comparison) and ``plot_cumulative_comparison()`` (all methods on one axes) after L4.1; then all four ``run_level42_*`` partitioning callables. The full-control path — every detector class, model hyperparameter, MDS tolerance, and diagnostic flag is reachable here.
- **fluxprocessingchain/fluxprocessingchain_partitioning.py** — Level 4.2 on the ``run_chain`` path: all four partitioning ports enabled through the ``partition_*`` fields of ``FluxConfig``. Shows which driver columns each port reads, ``partition_gapfill_method`` (which L4.1 gap-filled NEE feeds the nighttime variants), and how the per-USTAR-scenario output columns are named (``RECO_NT_OF_CUT_50``).

NEE Partitioning (Level 4.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **partitioning/partitioning_nighttime_oneflux.py** — Nighttime method (Reichstein et al. 2005), ONEFlux port → ``*_NT_OF``. Fits a Lloyd & Taylor temperature response to nighttime NEE, extrapolates to daytime for RECO, then ``GPP = RECO - NEE``.
- **partitioning/partitioning_nighttime_reddyproc.py** — Nighttime method (Reichstein et al. 2005), REddyProc ``sMRFluxPartition`` port → ``*_NT_RP``. A second, independent port of the same paper: differs in the day/night split (potential radiation, needs longitude + UTC offset), the E0 fitting, and it partitions the whole record with a single E0.
- **partitioning/partitioning_daytime_reddyproc.py** — Daytime method (Lasslop et al. 2010), REddyProc ``partitionNEEGL`` port → ``*_DT_RP``. Fits a rectangular-hyperbola light-response curve to daytime NEE in short windows, with E0 held fixed from a nighttime estimate.
- **partitioning/partitioning_daytime_oneflux.py** — Daytime method (Lasslop et al. 2010), ONEFlux ``flux_part_gl2010`` / FLUXNET2015 port → ``*_DT_OF``. Uses both measured and gap-filled drivers; the day/night split is measured ``Rg > 4``, with no solar geometry or latitude.
- **partitioning/partitioning_comparison.py** — All four ports on the same input, compared against each other and against the bundled REddyProc reference columns. Best for choosing a method, or seeing how much the choice matters.

Low-Resolution Flux Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **lowres/flux_timelag_analysis.py** — Time lag detection and visualization for gas concentrations
- **lowres/flux_common.py** — Flux variable base detection and nomenclature
- **lowres/flux_hqflux.py** — Highest-quality flux filtering with Hampel outlier detection
- **lowres/flux_selfheating.py** — SCOP self-heating correction (quick demo)
- **lowres/flux_selfheating_production.py** — Complete production workflow: scaling factors from parallel measurements, applied to long-term data
- **lowres/flux_uncertainty.py** — Random uncertainty estimation (PAS20 method)
- **lowres/flux_ustar_mp_detection.py** — Moving Point (MP) USTAR detection (Papale et al. 2006) with multi-year bootstrap
- **lowres/flux_ustar_vekuri_detection.py** — Quantile-based USTAR detection (Vekuri method) with multi-year bootstrap
- **lowres/flux_ustar_method_comparison.py** — Side-by-side comparison of ONEFlux and Vekuri USTAR approaches

Related Documentation
---------------------

Available classes and functions in ``dv.flux``:

- **TimeLagAnalysis** — Time lag detection and visualization for gas concentrations
- **RandomUncertaintyPAS20** — Measurement uncertainty quantification
- **FlagMultipleConstantUstarThresholds** — USTAR filtering with multiple constant thresholds
- **FlagMultipleVariableUstarThresholds** — USTAR filtering with time-varying (per-record, e.g. per-year VUT) thresholds. It backs ``run_level33_variable_ustar``, which is the usual way to reach it; the instance ends up in ``data.levels.level33``.
- **UstarMovingPointDetection** — Moving-point USTAR detection (Papale et al. 2006)
- **UstarVekuriThresholdDetection** — Quantile-based USTAR detection (Vekuri method)
- **UstarBootstrapThresholds** — Multi-year bootstrap wrapper for any USTAR detector; 3-year sliding window. Returns **VUT** (variable, per-year p16/p50/p84 via ``get_vut_thresholds()`` / ``run()``) and **CUT** (constant, pooled across years via ``get_cut_threshold()``). diive's VUT is smoothed over the 3-year window (differs from strict single-year ONEFlux VUT)
- **ScopApplicator** — SCOP self-heating correction for open-path IRGA. Not re-exported on ``dv.flux``; import it from ``diive.flux.lowres`` (``from diive.flux.lowres import ScopApplicator``).
- **run_chain / FluxConfig** — Single-call driver for the full L2→L4.2 flux processing pipeline; one ``FluxConfig`` per flux variable. L4.2 partitioning is opt-in via the ``partition_*`` fields (``partition_nighttime_oneflux``, ``partition_daytime_oneflux``, …)
- **Composable level callables** — ``init_flux_data``, ``run_level2``, ``run_level31``, ``make_level32_detector`` + ``run_level32``, ``run_level33_constant_ustar`` / ``run_level33_variable_ustar`` / ``run_level33_ustar_detection`` (mode ``'cut'``/``'vut'``), ``run_level41_mds`` / ``_rf`` / ``_xgb``, ``run_level42_nighttime_oneflux`` / ``_nighttime_reddyproc`` / ``_daytime_reddyproc`` / ``_daytime_oneflux``; pure functions on a typed ``FluxLevelData`` container. Import them from ``diive.flux.fluxprocessingchain`` — only ``init_flux_data``, ``add_driver``, ``run_chain`` and ``FluxConfig`` are also re-exported on ``dv.flux``
- **Partitioning ports** — ``NighttimePartitioningOneFlux`` (``*_NT_OF``), ``NighttimePartitioningReddyProc`` (``*_NT_RP``), ``DaytimePartitioningReddyProc`` (``*_DT_RP``), ``DaytimePartitioningOneFlux`` (``*_DT_OF``), plus the ``partition_nee_*`` function wrappers. The standalone classes behind Level 4.2; usable on their own dataframe outside the chain
- **add_driver(data, series)** — Add a computed driver column to ``data.full_df``, where L4.1 gap-filling and the L4.2 ``partition_*`` drivers read from (not ``fpc_df``)
- Flux variable detection and nomenclature

Use Cases
---------

**Process eddy covariance flux data — single-call driver:**

.. code-block:: python

   from diive.flux.fluxprocessingchain import (
       FluxConfig, init_flux_data, run_chain,
   )

   cfg = FluxConfig(
       fluxcol='FC',
       ustar_thresholds=[0.18], ustar_labels=['CUT_50'],
       outlier_sigma_daytime=5.5, outlier_sigma_nighttime=5.5,
       gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_kPa_1_1_1'],
       level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
       mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa_1_1_1',
   )
   data = init_flux_data(df, fluxcol='FC', site_lat=47.48, site_lon=8.36, utc_offset=1)
   data = run_chain(data, cfg)

   results = data.fpc_df          # all per-level and gap-filled columns
   cols = data.gapfilled_cols()   # {'rf': {'CUT_50': '...'}, 'xgb': ..., 'mds': ...}

**Composable per-level API** — for custom L3.2 outlier pipelines, custom feature engineering, or per-level inspection, call each ``run_level*`` directly. See ``examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py``.

**Analyze time lag and measurement quality:**

.. code-block:: python

   import diive as dv

   # Detect optimal time lags for gas concentrations
   analysis = dv.flux.TimeLagAnalysis(
       df=df,
       ignore_fringe_bins=[5, 10],
       lag_window_min=0.10,
       lag_window_max=1.00
   )
   co2_results = analysis.analyze_gas('CO2')
   fig = analysis.plot_gas('CO2', outdir='output/')

   # Quantify measurement uncertainty (VPD in kPa; pass vpd_in_kpa=False for hPa)
   randunc = dv.flux.RandomUncertaintyPAS20(
       df=df,
       fluxcol='NEE_CUT_REF_orig',        # measured flux
       fluxgapfilledcol='NEE_CUT_REF_f',  # gap-filled flux
       tacol='Tair_f',
       vpdcol='VPD_f',
       swincol='Rg_f',
   )
   randunc.run()
   uncertainty = randunc.randunc_series

Running Examples
----------------

.. code-block:: bash

   # Complete multi-level processing workflow (recommended starting point)
   uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py

   # NEE partitioning (Level 4.2)
   uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_partitioning.py
   uv run python examples/flux/partitioning/partitioning_nighttime_oneflux.py
   uv run python examples/flux/partitioning/partitioning_nighttime_reddyproc.py
   uv run python examples/flux/partitioning/partitioning_daytime_reddyproc.py
   uv run python examples/flux/partitioning/partitioning_daytime_oneflux.py
   uv run python examples/flux/partitioning/partitioning_comparison.py

   # Low-resolution (30-min) processing
   uv run python examples/flux/lowres/flux_timelag_analysis.py
   uv run python examples/flux/lowres/flux_selfheating.py
   uv run python examples/flux/lowres/flux_uncertainty.py
   uv run python examples/flux/lowres/flux_ustar_mp_detection.py
   uv run python examples/flux/lowres/flux_ustar_vekuri_detection.py
   uv run python examples/flux/lowres/flux_ustar_method_comparison.py

   # Run all flux examples
   uv run python examples/run_all_examples.py

Standards & Best Practices
--------------------------

- **FLUXNET conventions** — Data flows through 6 levels (L2→L3.1→L3.2→L3.3→L4.1→L4.2); L4.2 (NEE→GPP+RECO partitioning) is optional
- **Swiss FluxNet methodology** — Quality flags, storage correction, USTAR filtering
- **Unit consistency** — Always use SI units (W/m², K, hPa)
- **QC/QF flags** — Combine multiple quality tests into single QCF flag
- **Uncertainty propagation** — Random + systematic uncertainty estimation
