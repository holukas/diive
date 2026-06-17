"""
GUI.TABS.PARTITIONING_DAYTIME_ONEFLUX: DAYTIME PARTITIONING (ONEFLUX)
====================================================================

Tab for the daytime NEE -> GPP + RECO partitioning, ONEFlux port
(``flux_part_gl2010`` / Lasslop et al. 2010), emitting ``*_DT_OF`` columns.
Wraps the library's :class:`diive.flux.DaytimePartitioningOneFlux`. Day/night
split is the measured-Rg ≤4/>4 threshold (no solar geometry). Uses measured
NEE / TA / SW_IN plus gap-filled TA / SW_IN / VPD drivers.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.flux import DaytimePartitioningOneFlux
from diive.gui.tabs._partitioning_base import BasePartitioningTab


class DaytimePartitioningOneFluxTab(BasePartitioningTab):
    """Daytime partitioning (ONEFlux, Lasslop et al. 2010) -> *_DT_OF."""

    title = "Daytime partitioning (ONEFlux)"
    intro = ("Partition NEE into GPP and RECO with the daytime light-response "
             "method (ONEFlux, Lasslop et al. 2010). Measured NEE / TA / SW_IN + "
             "gap-filled TA / SW_IN / VPD. Per calendar year; emits *_DT_OF columns.")
    inputs = [
        {"key": "nee", "label": "NEE (measured)", "needle": "NEE",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured net ecosystem exchange (µmol m⁻² s⁻¹)."},
        {"key": "ta", "label": "TA (measured)", "needle": "TA",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured air temperature (°C) for day/night classification + uncertainty."},
        {"key": "sw_in", "label": "SW_IN (measured)", "needle": ["SW_IN", "RG"],
         "prefer": "ORIG", "avoid": "POT", "tip": "Measured incoming shortwave radiation (W m⁻²); Rg-threshold day/night split."},
        {"key": "ta_f", "label": "TA (gap-filled)", "needle": "TA",
         "prefer": "_F", "tip": "Gap-filled air temperature (°C) — used in fits and RECO everywhere."},
        {"key": "sw_in_f", "label": "SW_IN (gap-filled)", "needle": ["SW_IN", "RG"],
         "prefer": "_F", "avoid": "POT", "tip": "Gap-filled incoming shortwave radiation (W m⁻²) — the LRC light driver."},
        {"key": "vpd", "label": "VPD (gap-filled)", "needle": "VPD",
         "prefer": "_F", "tip": "Gap-filled vapour pressure deficit (kPa by default)."},
    ]
    has_vpd_unit = True
    reco_col = "RECO_DT_OF"
    gpp_col = "GPP_DT_OF"
    method_suffix = "DT_OF"

    def _build_partitioner(self, series_map, coords, vpd_in_kpa):
        return DaytimePartitioningOneFlux(
            nee=series_map["nee"], ta=series_map["ta"], sw_in=series_map["sw_in"],
            ta_f=series_map["ta_f"], sw_in_f=series_map["sw_in_f"],
            vpd=series_map["vpd"], vpd_in_kpa=vpd_in_kpa, verbose=2)
