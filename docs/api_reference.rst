.. _api_reference:

=============
API Reference
=============

This page contains the API documentation for DIIVE, automatically generated from the source code.

Core Modules
============

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   diive.core.ml.feature_engineer
   diive.core.ml.common
   diive.core.ml.optimization
   diive.core.plotting
   diive.core.times.times
   diive.core.dfun.frames

Domain-Specific Packages
========================

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   diive.pkgs.gapfilling.randomforest_ts
   diive.pkgs.gapfilling.xgboost_ts
   diive.pkgs.gapfilling.mds
   diive.pkgs.preprocessing.outlierdetection.absolutelimits
   diive.pkgs.preprocessing.outlierdetection.hampel
   diive.pkgs.preprocessing.outlierdetection.localsd
   diive.pkgs.preprocessing.outlierdetection.zscore
   diive.pkgs.preprocessing.outlierdetection.lof
   diive.pkgs.preprocessing.outlierdetection.stepwiseoutlierdetection
   diive.pkgs.analysis.correlation
   diive.pkgs.analysis.decoupling
   diive.pkgs.createvar.laggedvariants
   diive.configs.exampledata

Flux Processing
---------------

*Flux processing chain documentation coming soon (import issue in source code)*

NEE Partitioning
^^^^^^^^^^^^^^^^

Split net ecosystem exchange (NEE) into gross primary production (GPP) and
ecosystem respiration (RECO). Four faithful ports of the standard reference
routines — two nighttime (Reichstein et al. 2005) and two daytime
(Lasslop et al. 2010), one each from ONEFlux (``_OF``) and REddyProc (``_RP``):

* ``NighttimePartitioningOneFlux`` / ``partition_nee_nighttime_oneflux`` — ``*_NT_OF``
* ``NighttimePartitioningReddyProc`` / ``partition_nee_nighttime_reddyproc`` — ``*_NT_RP``
* ``DaytimePartitioningReddyProc`` / ``partition_nee_daytime_reddyproc`` — ``*_DT_RP``
* ``DaytimePartitioningOneFlux`` / ``partition_nee_daytime_oneflux`` — ``*_DT_OF``

Inputs are in physical units (air temperature in °C, VPD in kPa).

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   diive.flux.partitioning.nighttime_oneflux
   diive.flux.partitioning.nighttime_reddyproc
   diive.flux.partitioning.daytime_reddyproc
   diive.flux.partitioning.daytime_oneflux
