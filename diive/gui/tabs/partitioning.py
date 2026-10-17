"""
GUI.TABS.PARTITIONING: NEE PARTITIONING (METHOD + MODE)
=======================================================

One tab for the NEE -> GPP + RECO partitioning with two switches, the way the
library's :class:`diive.flux.PartitioningNEE` works: **Method** picks the
algorithm (nighttime, Reichstein et al. 2005, or daytime, Lasslop et al. 2010)
and **Mode** the reference implementation (ONEFlux, the ICOS and FLUXNET2015
processing and the default, or REddyProc). The four combinations write
``*_NT_OF``, ``*_NT_RP``, ``*_DT_OF`` and ``*_DT_RP`` columns.

The tab shows only the input pickers and site coordinates the chosen variant
needs (``partitioning_inputs``). It replaces the four per-variant tabs.
Projects saved with one of those reopen here, preset to that variant (see
``_MERGED_MENU_TABS`` in ``diive/gui/app.py``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout

from diive.flux.partitioning import (
    PartitioningNEE,
    partitioning_inputs,
    partitioning_optional_inputs,
)
from diive.flux.partitioning.codegen import partition_nee_to_code
from diive.gui.tabs._partitioning_base import BasePartitioningTab
from diive.gui.widgets.partitioning_mode import (
    combo_value,
    make_method_combo,
    make_mode_combo,
    set_combo_value,
)

#: Column suffix per (method, mode), as the library writes it.
_SUFFIX = {("nighttime", "oneflux"): "NT_OF", ("nighttime", "reddyproc"): "NT_RP",
           ("daytime", "oneflux"): "DT_OF", ("daytime", "reddyproc"): "DT_RP"}

_COORDS = ("lat", "lon", "utc_offset")

_INTRO = {
    ("nighttime", "oneflux"): (
        "Nighttime method (Reichstein et al. 2005), ONEFlux as used by ICOS. "
        "Per calendar year. Emits *_NT_OF columns."),
    ("nighttime", "reddyproc"): (
        "Nighttime method (Reichstein et al. 2005), REddyProc sMRFluxPartition. "
        "One temperature sensitivity for the whole record. Emits *_NT_RP columns."),
    ("daytime", "oneflux"): (
        "Daytime light-response method (Lasslop et al. 2010), ONEFlux as used by "
        "ICOS. Day and night are split on measured SW_IN, so no site location is "
        "needed. Emits *_DT_OF columns."),
    ("daytime", "reddyproc"): (
        "Daytime light-response method (Lasslop et al. 2010), REddyProc "
        "partitionNEEGL. Fits on measured NEE with gap-filled drivers. Emits "
        "*_DT_RP columns."),
}


class PartitioningNEETab(BasePartitioningTab):
    """NEE partitioning with Method (nighttime / daytime) and Mode (ONEFlux /
    REddyProc) switches, via :class:`diive.flux.PartitioningNEE`."""

    title = "NEE partitioning"
    # Keys are the PartitioningNEE input names, so picks map 1:1 to kwargs.
    inputs = [
        {"key": "nee", "label": "NEE (measured)", "needle": "NEE",
         "prefer": "ORIG", "avoid": "_F",
         "tip": "Measured net ecosystem exchange (µmol m⁻² s⁻¹), gaps as missing values."},
        {"key": "nee_f", "label": "NEE (gap-filled)", "needle": "NEE",
         "prefer": "_F", "optional_label": "NEE (gap-filled, optional)",
         "tip": "Gap-filled NEE (µmol m⁻² s⁻¹). Nighttime: the GPP residual. "
                "Daytime ONEFlux (optional): ONEFlux reads it for windows without "
                "measured nighttime NEE, so pick it to reproduce ONEFlux there."},
        {"key": "ta", "label": "TA (measured)", "needle": "TA",
         "prefer": "ORIG", "avoid": "_F", "tip": "Measured air temperature (°C)."},
        {"key": "ta_f", "label": "TA (gap-filled)", "needle": "TA",
         "prefer": "_F", "tip": "Gap-filled air temperature (°C), used to compute RECO everywhere."},
        {"key": "sw_in", "label": "SW_IN (measured)", "needle": ["SW_IN", "RG"],
         "prefer": "ORIG", "avoid": "POT",
         "tip": "Measured incoming shortwave radiation (W m⁻²), for the day/night split."},
        {"key": "sw_in_f", "label": "SW_IN (gap-filled)", "needle": ["SW_IN", "RG"],
         "prefer": "_F", "avoid": "POT",
         "tip": "Gap-filled incoming shortwave radiation (W m⁻²), the light driver of the fit."},
        {"key": "vpd", "label": "VPD (gap-filled)", "needle": "VPD",
         "prefer": "_F", "tip": "Gap-filled vapour pressure deficit (kPa by default)."},
        {"key": "nee_sd", "label": "NEE SD (optional)", "needle": "NEE",
         "prefer": "_SD", "optional": True,
         "tip": "Per-record NEE uncertainty to weight the fit. Leave as (none) to "
                "reproduce the REddyProc default."},
    ]

    # --- variant -------------------------------------------------------
    @property
    def method(self) -> str:
        return combo_value(self.method_combo)

    @property
    def mode(self) -> str:
        return combo_value(self.mode_combo)

    def _needed(self) -> tuple[str, ...]:
        return partitioning_inputs(self.method, self.mode)

    def _optional(self) -> tuple[str, ...]:
        return partitioning_optional_inputs(self.method, self.mode)

    def _active_inputs(self) -> list[dict]:
        # An input this variant takes only when given (nee_f for daytime
        # ONEFlux) is marked optional, so "(none)" is accepted for it.
        optional = set(self._optional())
        keys = {"nee", *self._needed(), *optional}
        return [dict(s, optional=True) if s["key"] in optional else s
                for s in self.inputs if s["key"] in keys]

    def _apply_variant(self) -> None:
        self.picker.set_optional_keys(self._optional())
        super()._apply_variant()

    @property
    def needs_lat(self) -> bool:
        return "lat" in self._needed()

    @property
    def needs_lon(self) -> bool:
        return "lon" in self._needed()

    @property
    def needs_utc(self) -> bool:
        return "utc_offset" in self._needed()

    @property
    def has_vpd_unit(self) -> bool:
        return "vpd" in self._needed()

    @property
    def method_suffix(self) -> str:
        return _SUFFIX[(self.method, self.mode)]

    @property
    def reco_col(self) -> str:
        return f"RECO_{self.method_suffix}"

    @property
    def gpp_col(self) -> str:
        return f"GPP_{self.method_suffix}"

    @property
    def intro(self) -> str:
        return _INTRO[(self.method, self.mode)]

    def _variant(self) -> dict:
        return {"method": self.method, "mode": self.mode,
                "suffix": self.method_suffix,
                "operation": f"{self.title} ({self.method_suffix})"}

    # --- build ---------------------------------------------------------
    def build(self):
        # The combos exist before the base builds anything that reads the
        # variant properties above.
        self.method_combo = make_method_combo()
        self.mode_combo = make_mode_combo()
        root = super().build()
        self.method_combo.currentIndexChanged.connect(self._on_variant_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_variant_changed)
        return root

    def _add_variant_controls(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Partitioning")
        form = QFormLayout(box)
        form.addRow("Method", self.method_combo)
        form.addRow("Mode", self.mode_combo)
        layout.addWidget(box)

    def _on_variant_changed(self, *_) -> None:
        self._apply_variant()
        if self._results is None:
            self.status.setText(self.intro)

    # --- state ---------------------------------------------------------
    def save_state(self) -> dict:
        state = super().save_state()
        state["method"] = self.method
        state["mode"] = self.mode
        return state

    def restore_state(self, state: dict) -> None:
        # Variant first, so the right rows show when the picks come back.
        set_combo_value(self.method_combo, state.get("method"))
        set_combo_value(self.mode_combo, state.get("mode"))
        super().restore_state(state)

    # --- library calls -------------------------------------------------
    def _build_partitioner(self, series_map, coords, vpd_in_kpa, variant):
        series = dict(series_map)
        nee = series.pop("nee")
        return PartitioningNEE(nee, method=variant["method"], mode=variant["mode"],
                               vpd_in_kpa=vpd_in_kpa, verbose=2, **series, **coords)

    def _python_code_for(self, picks, coords, vpd_in_kpa):
        return partition_nee_to_code(self.method, self.mode, picks,
                                     vpd_in_kpa=vpd_in_kpa, **coords)
