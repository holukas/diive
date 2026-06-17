"""
GUI.TABS.PARTITIONING_DAYTIME_REDDYPROC: DAYTIME PARTITIONING (REDDYPROC)
========================================================================

Tab for the daytime NEE -> GPP + RECO partitioning, REddyProc port
(``partitionNEEGL`` / Lasslop et al. 2010 light-response curve), emitting
``*_DT_RP`` columns. Wraps the library's
:class:`diive.flux.DaytimePartitioningReddyProc`; the daytime method uses
gap-filled meteo drivers (TA / VPD / SW_IN) and quality-filters only NEE.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.flux import DaytimePartitioningReddyProc
from diive.gui.tabs._partitioning_base import BasePartitioningTab


class DaytimePartitioningReddyProcTab(BasePartitioningTab):
    """Daytime partitioning (REddyProc, Lasslop et al. 2010) -> *_DT_RP."""

    title = "Daytime partitioning (REddyProc)"
    intro = ("Partition NEE into GPP and RECO with the daytime light-response "
             "method (REddyProc, Lasslop et al. 2010). Measured NEE + gap-filled "
             "TA / VPD / SW_IN drivers. Emits *_DT_RP columns.")
    inputs = [
        {"key": "nee", "label": "NEE (measured)", "needle": "NEE",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured net ecosystem exchange (µmol m⁻² s⁻¹); the LRC is fitted on measured daytime values."},
        {"key": "ta", "label": "TA (gap-filled)", "needle": "TA",
         "prefer": "_F", "tip": "Gap-filled air temperature (°C)."},
        {"key": "vpd", "label": "VPD (gap-filled)", "needle": "VPD",
         "prefer": "_F", "tip": "Gap-filled vapour pressure deficit (kPa by default)."},
        {"key": "sw_in", "label": "SW_IN (gap-filled)", "needle": ["SW_IN", "RG"],
         "prefer": "_F", "avoid": "POT", "tip": "Gap-filled incoming shortwave radiation (W m⁻²)."},
        {"key": "nee_sd", "label": "NEE SD (optional)", "needle": "NEE",
         "prefer": "_SD", "optional": True, "tip": "Per-record NEE uncertainty to weight the fit. Leave as (none) to reproduce the REddyProc default."},
    ]
    needs_lat = True
    needs_lon = True
    needs_utc = True
    has_vpd_unit = True
    reco_col = "RECO_DT_RP"
    gpp_col = "GPP_DT_RP"
    method_suffix = "DT_RP"

    def _build_partitioner(self, series_map, coords, vpd_in_kpa):
        return DaytimePartitioningReddyProc(
            nee=series_map["nee"], ta=series_map["ta"], vpd=series_map["vpd"],
            sw_in=series_map["sw_in"], lat=coords["lat"], lon=coords["lon"],
            utc_offset=coords["utc_offset"], nee_sd=series_map.get("nee_sd"),
            vpd_in_kpa=vpd_in_kpa, verbose=2)
