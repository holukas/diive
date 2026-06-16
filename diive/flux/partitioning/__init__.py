"""
FLUX PARTITIONING: NEE -> GPP + RECO
=====================================

Partition net ecosystem exchange (NEE) into its gross components, gross primary
production (GPP) and ecosystem respiration (RECO).

Two faithful, vectorized ports of the same Reichstein et al. (2005) nighttime
method are available; they differ in window geometry, day/night split, the E0
fit, and units, so they do not produce identical numbers. Output columns carry
a variant token after the ``_NT`` suffix - ``_OF`` (ONEFlux) and ``_RP``
(REddyProc) - so both can coexist in one dataframe:

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
]
