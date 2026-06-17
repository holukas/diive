"""
GUI.TABS.PARTITIONING_NIGHTTIME_ONEFLUX: NIGHTTIME PARTITIONING (ONEFLUX)
========================================================================

Tab for the nighttime NEE -> GPP + RECO partitioning, ONEFlux port (Reichstein
et al. 2005), emitting ``*_NT_OF`` columns. Wraps the library's
:class:`diive.flux.NighttimePartitioningOneFlux`; the tab only collects the
input columns + site latitude and previews the result.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.flux import NighttimePartitioningOneFlux
from diive.gui.tabs._partitioning_base import BasePartitioningTab


class NighttimePartitioningOneFluxTab(BasePartitioningTab):
    """Nighttime partitioning (ONEFlux, Reichstein et al. 2005) -> *_NT_OF."""

    title = "Nighttime partitioning (ONEFlux)"
    intro = ("Partition NEE into GPP and RECO with the nighttime method "
             "(ONEFlux, Reichstein et al. 2005). Per calendar year. Emits "
             "*_NT_OF columns.")
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
    reco_col = "RECO_NT_OF"
    gpp_col = "GPP_NT_OF"
    method_suffix = "NT_OF"

    def _build_partitioner(self, series_map, coords, vpd_in_kpa):
        return NighttimePartitioningOneFlux(
            nee=series_map["nee"], ta=series_map["ta"], sw_in=series_map["sw_in"],
            nee_f=series_map["nee_f"], ta_f=series_map["ta_f"],
            lat=coords["lat"], verbose=2)
