Low-Resolution Flux Processing Examples
=======================================

Low-resolution (30-minute) eddy-covariance flux processing, mostly with data from the Swiss FluxNet
site CH-LAE (some examples use CH-DAV or CH-CHA).
Covers time lag detection, quality filtering, self-heating correction, uncertainty estimation, and turbulence thresholds.

Examples
--------

Time Lag Detection
~~~~~~~~~~~~~~~~~~

- **flux_timelag_analysis.py** — Time lag detection and visualization for gas concentrations using covariance analysis

Base Flux Detection & Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **flux_common.py** — Detect flux variable nomenclature and base gas measurements
- **flux_hqflux.py** — Extract highest-quality flux records using Hampel outlier filtering

Self-Heating Correction (SCOP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **flux_selfheating.py** — Quick demonstration of SCOP workflow (5 USTAR classes, 5 bootstrap runs)
- **flux_selfheating_production.py** — Complete production workflow: calibrate a scaling-factor table on one period with parallel measurements (20 classes, 20 bootstrap runs for the example; use 100 in production) and apply it to a held-back period with no reference

Measurement Uncertainty & Turbulence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **flux_uncertainty.py** — Random uncertainty estimation using the PAS20 method (Pastorello et al. 2020)
- **flux_ustar_detection.py** — u* threshold detection as in ONEFlux: moving point (Papale et al. 2006) and change point (Barr et al. 2013) thresholds per year, pooled
- **flux_ustar_method_comparison.py** — Compare the four u* detection modes and the share of nighttime NEE records each threshold removes
- **flux_ustar_data_availability.py** — Share of NEE records kept after u* filtering over a range of constant thresholds, by day and by night (``UstarThresholdConstantScenarios``), and the gaps the CUT percentiles leave

The first two u* examples use ``UstarThresholdDetection``. ``mode`` selects the method: ``'oneflux'`` (default), ``'oneflux_mp'``, ``'oneflux_cp'`` or ``'reddyproc'``. Each calendar year is bootstrapped on its own data. ``run()`` returns the **VUT** (one threshold per year, p16/p50/p84 for ONEFlux) and ``get_cut_threshold()`` the **CUT** (one threshold for all years). As in ONEFlux, the VUT of a year pools the thresholds of the year and its two neighbours.

Running Examples
----------------

.. code-block:: bash

   # Run one example
   uv run python examples/flux/lowres/flux_timelag_analysis.py

   # Run more low-res flux examples
   uv run python examples/flux/lowres/flux_selfheating.py
   uv run python examples/flux/lowres/flux_selfheating_production.py
   uv run python examples/flux/lowres/flux_uncertainty.py

Key Concepts
------------

**Self-heating correction (SCOP method):**

- Sun-induced heating of open-path IRGA sensors creates spurious negative CO2 flux
- Correction uses physics-based unscaled term (FCT_UNSC) scaled by factors from parallel measurements
- Scaling factors binned by USTAR and time-of-day, reusable across seasons

**Quality filtering:**

- Hampel filter for identifying robust flux estimates
- USTAR-based selection for stable atmospheric conditions

**Uncertainty:** Random measurement uncertainty following the PAS20 method (Pastorello et al. 2020)

Related Documentation
---------------------

- `Flux Post-Processing Chain <../postprocessingchain/index.html>`_ — Multi-level L2-L4.2 workflow
- Source: ``diive/flux/lowres/``
