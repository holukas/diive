Quality Assurance / Quality Control (QA/QC) Examples
====================================================

Examples demonstrating quality control methods, flag generation, and data quality assessment.

Contents
--------

- **qc_overall_flag.py** — Overall Quality Control Flag (QCF) combining multiple test flags
- **qc_eddypro_flags.py** — EddyPro quality flag extraction (signal strength, VM97 tests, completeness)
- **qaqc_detect_timestamp_shifts.py** — Detect clock/timestamp errors by comparing measured vs potential shortwave radiation (FFT phase shift, cross-correlation, noon-shift peak detection)

Related Documentation
---------------------

See ``dv.qaqc`` for:

- ``FlagQCF`` — Overall quality flag combining individual tests

The EddyPro flag functions are not re-exported by ``dv.qaqc``. Import them from
``diive.preprocessing.qaqc``:

- ``flag_signal_strength_eddypro_test()`` — Signal quality
- ``flags_vm97_eddypro_fluxnetfile_tests()`` — Vickers & Mahrt (1997) raw data tests
- ``flag_ssitc_eddypro_test()`` — Steady State and Integral Turbulence Characteristics
- ``flag_spectral_correction_factor_eddypro_test()`` — Spectral correction assessment

``DetectTimestampShifts`` — timestamp shift detection via radiation phase analysis — is
imported from its module:

.. code-block:: python

   from diive.preprocessing.qaqc.detect_timestamp_shifts import DetectTimestampShifts

Use Cases
---------

**Generate overall quality flag (QCF):**

.. code-block:: python

   from diive.preprocessing.qaqc import FlagQCF

   # Combine multiple individual quality tests into single QCF
   qcf = FlagQCF(
       df=df,
       target_col='NEE',
       swinpot_col='SW_IN_POT',  # Optional: enables day/night separation
       idstr='_L41'
   )
   qcf.calculate(daytime_accept_qcf_below=2)  # Accept good+medium daytime

   # QCF values: 0=good, 1=marginal, 2=poor
   filtered = df[qcf.filteredseries.notna()]  # Keep only good quality
   highest_quality = df[qcf.filteredseries_hq.notna()]  # Keep only best

   # Reports and plots
   qcf.report_qcf_series()  # Summary statistics
   qcf.report_qcf_flags()  # Per-test breakdown
   qcf.showplot_qcf_heatmaps()  # Visualization

**Extract EddyPro quality flags:**

.. code-block:: python

   from diive.preprocessing.qaqc import (
       flag_signal_strength_eddypro_test,
       flags_vm97_eddypro_fluxnetfile_tests,
       flag_ssitc_eddypro_test
   )

   # Signal quality (IRGA, anemometer)
   sig_flag = flag_signal_strength_eddypro_test(
       df=df,
       signal_strength_col='CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
       var_col='FC',  # only used to name the flag
       method='discard below',  # flag 2 where signal strength < threshold
       threshold=99,
       idstr='_L41'
   )

   # VM97 raw data tests, read from the 8-digit integer in column CO2_VM97_TEST
   vm97_flags = flags_vm97_eddypro_fluxnetfile_tests(
       df=df,
       flux='FC',
       fluxbasevar='CO2',
       idstr='_L41',
       spikes=True,
       amplitude=True,
       dropout=True,
       abslim=True,
       skewkurt_hf=True,
       skewkurt_sf=True,
       discont_hf=True,
       discont_sf=True
   )
   # One flag column per test switched on above: spikes, amplitude, dropout,
   # abslim, skewkurt_hf, skewkurt_sf, discont_hf, discont_sf

   # Stationarity test, read from column FC_SSITC_TEST
   ssitc_flag = flag_ssitc_eddypro_test(df=df, flux='FC', idstr='_L41')

Quality Flag Schema
-------------------

**QCF (Overall Quality Control Flag):**

- **0** = Good quality (all tests pass)
- **1** = Marginal quality (1-3 soft warnings, no hard fails)
- **2** = Poor quality (>3 soft warnings OR ≥1 hard fail)

**EddyPro test results:**

- **0** = Pass
- **1** = Soft warning (marginal)
- **2** = Hard fail (reject)

Running Examples
----------------

.. code-block:: bash

   # Generate overall quality flags from multiple tests
   uv run python examples/preprocessing/qaqc/qc_overall_flag.py

   # Extract and convert EddyPro-specific quality flags
   uv run python examples/preprocessing/qaqc/qc_eddypro_flags.py

   # Detect clock/timestamp errors from measured vs potential radiation
   uv run python examples/preprocessing/qaqc/qaqc_detect_timestamp_shifts.py

   # Run all QA/QC examples
   uv run python examples/run_all_examples.py

FLUXNET Standards
-----------------

Quality control follows FLUXNET conventions:

- Quality tests applied independently
- Results combined into overall QCF score
- Day/night thresholds differ (nighttime stricter)
- USTAR filtering applied to flux only, not energy variables
- Multiple percentile scenarios for uncertainty quantification
