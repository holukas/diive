"""
GUI.TABS.PARTITIONING_NIGHTTIME_REDDYPROC: NIGHTTIME PARTITIONING (REDDYPROC)
============================================================================

Tab for the nighttime NEE -> GPP + RECO partitioning, REddyProc port
(``sMRFluxPartition``, Reichstein et al. 2005), emitting ``*_NT_RP`` columns.
Wraps the library's :class:`diive.flux.NighttimePartitioningReddyProc`; the tab
collects the input columns + site coordinates (longitude + UTC offset are needed
for REddyProc's potential-radiation day/night split) and previews the result.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.flux import NighttimePartitioningReddyProc
from diive.gui.tabs._partitioning_base import BasePartitioningTab


class NighttimePartitioningReddyProcTab(BasePartitioningTab):
    """Nighttime partitioning (REddyProc, Reichstein et al. 2005) -> *_NT_RP."""

    title = "Nighttime partitioning (REddyProc)"
    intro = ("Partition NEE into GPP and RECO with the nighttime method "
             "(REddyProc, Reichstein et al. 2005). Whole-record single E0. "
             "Emits *_NT_RP columns.")
    inputs = [
        {"key": "nee", "label": "NEE (measured)", "needle": "NEE",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured net ecosystem exchange (µmol m⁻² s⁻¹)."},
        {"key": "ta", "label": "TA (measured)", "needle": "TA",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured air temperature (°C)."},
        {"key": "sw_in", "label": "SW_IN (measured)", "needle": ["SW_IN", "RG"],
         "prefer": "ORIG", "avoid": "POT", "tip": "Incoming shortwave radiation (W m⁻²) for the day/night split."},
        {"key": "nee_f", "label": "NEE (gap-filled)", "needle": "NEE",
         "prefer": "_F", "tip": "Gap-filled NEE (µmol m⁻² s⁻¹) — used for the GPP residual."},
        {"key": "ta_f", "label": "TA (gap-filled)", "needle": "TA",
         "prefer": "_F", "tip": "Gap-filled air temperature (°C) — used to compute RECO everywhere."},
    ]
    needs_lat = True
    needs_lon = True
    needs_utc = True
    reco_col = "RECO_NT_RP"
    gpp_col = "GPP_NT_RP"
    method_suffix = "NT_RP"

    def _build_partitioner(self, series_map, coords, vpd_in_kpa):
        return NighttimePartitioningReddyProc(
            nee=series_map["nee"], ta=series_map["ta"], sw_in=series_map["sw_in"],
            nee_f=series_map["nee_f"], ta_f=series_map["ta_f"],
            lat=coords["lat"], lon=coords["lon"], utc_offset=coords["utc_offset"],
            verbose=2)
