"""
FLUX PARTITIONING: NEE -> GPP + RECO
=====================================

Partition net ecosystem exchange (NEE) into its gross components, gross primary
production (GPP) and ecosystem respiration (RECO).

Four faithful, vectorized ports are available: two nighttime methods (Reichstein
et al. 2005, fitting the temperature response of nighttime NEE) and two daytime
methods (Lasslop et al. 2010, fitting a light-response curve to daytime NEE).
Each method is a port of a different reference implementation, so they do not
produce identical numbers. Output columns carry a token after the ``_NT``
(nighttime) / ``_DT`` (daytime) suffix - ``_OF`` (ONEFlux) and ``_RP``
(REddyProc) - so all four can coexist in one dataframe:

- Nighttime method ONEFlux (Reichstein et al. 2005),
  ``NighttimePartitioningOneFlux`` / ``partition_nee_nighttime_oneflux`` - a
  faithful port of the ONEFlux ``oneflux.partition.nighttime`` reference
  implementation. Per-calendar-year processing; emits ``*_NT_OF`` columns
  (including outlier-robust ``*_NT_OF_ROB`` variants).

- Nighttime method REddyProc (Reichstein et al. 2005),
  ``NighttimePartitioningReddyProc`` / ``partition_nee_nighttime_reddyproc`` - a
  faithful port of REddyProc's ``sMRFluxPartition``. Whole-record processing
  with a single E0; emits ``*_NT_RP`` columns (no robust variant, as in
  REddyProc).

- Daytime method REddyProc (Lasslop et al. 2010),
  ``DaytimePartitioningReddyProc`` / ``partition_nee_daytime_reddyproc`` - a
  faithful port of REddyProc's ``partitionNEEGL``; per-window light-response-curve
  fit with E0 fixed from a GP-smoothed nighttime estimate, then distance-weighted
  interpolation. Emits ``*_DT_RP`` columns.

- Daytime method ONEFlux (Lasslop et al. 2010),
  ``DaytimePartitioningOneFlux`` / ``partition_nee_daytime_oneflux`` - a faithful
  port of ONEFlux's ``flux_part_gl2010`` (FLUXNET2015); per-window LRC fit with a
  measured-radiation day/night split (no latitude) and an internal NEE-uncertainty
  look-up. Emits ``*_DT_OF`` columns (including the GPP standard error
  ``SE_GPP_DT_OF``).

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.partitioning.nighttime_oneflux import NighttimePartitioningOneFlux
from diive.flux.partitioning.nighttime_oneflux import partition_nee_nighttime_oneflux
from diive.flux.partitioning.nighttime_oneflux import lloyd_taylor
from diive.flux.partitioning.nighttime_oneflux import sunrise_sunset
from diive.flux.partitioning.nighttime_reddyproc import NighttimePartitioningReddyProc
from diive.flux.partitioning.nighttime_reddyproc import partition_nee_nighttime_reddyproc
from diive.flux.partitioning.nighttime_reddyproc import lloyd_taylor_kelvin
from diive.flux.partitioning.nighttime_reddyproc import potential_radiation
from diive.flux.partitioning.daytime_reddyproc import DaytimePartitioningReddyProc
from diive.flux.partitioning.daytime_reddyproc import partition_nee_daytime_reddyproc
from diive.flux.partitioning.daytime_oneflux import DaytimePartitioningOneFlux
from diive.flux.partitioning.daytime_oneflux import partition_nee_daytime_oneflux

__all__ = [
    "NighttimePartitioningOneFlux",
    "partition_nee_nighttime_oneflux",
    "lloyd_taylor",
    "sunrise_sunset",
    "NighttimePartitioningReddyProc",
    "partition_nee_nighttime_reddyproc",
    "lloyd_taylor_kelvin",
    "potential_radiation",
    "DaytimePartitioningReddyProc",
    "partition_nee_daytime_reddyproc",
    "DaytimePartitioningOneFlux",
    "partition_nee_daytime_oneflux",
]
