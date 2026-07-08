"""
GUI.TABS.DERIVED_VPD: VAPOR PRESSURE DEFICIT FROM TA + RH
========================================================

Compute vapor pressure deficit (VPD, kPa) from air temperature (degC) and
relative humidity (%) via :func:`diive.variables.calc_vpd_from_ta_rh`. A thin
:class:`~diive.gui.tabs._derived_variable_base.BaseDerivedVariableTab` subclass:
it only declares the two input roles and wires them to the library function and
its codegen — all maths lives in the library.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd

import diive as dv
from diive.gui.tabs._derived_variable_base import BaseDerivedVariableTab


class VpdFromTaRhTab(BaseDerivedVariableTab):
    """Vapor pressure deficit from air temperature + relative humidity."""

    title = "VPD from TA + RH"
    intro = ("Calculate vapor pressure deficit (VPD, kPa) from air temperature "
             "(degC) and relative humidity (%).")
    inputs = [
        {"key": "ta", "label": "Air temperature (degC)", "short": "Air temperature",
         "needle": ["TA", "AIRTEMP", "T_AIR"], "avoid": "SONIC",
         "tip": "Air temperature column, in degrees Celsius."},
        {"key": "rh", "label": "Relative humidity (%)", "short": "Relative humidity",
         "needle": ["RH", "RELHUM"],
         "tip": "Relative humidity column, in percent."},
    ]
    default_name = "VPD_kPa"
    out_unit = "kPa"
    method_tags = ["thermodynamic", "vpd"]

    def _compute(self, df: pd.DataFrame, picks: dict[str, str]) -> pd.Series:
        return dv.variables.calc_vpd_from_ta_rh(
            df, rh_col=picks["rh"], ta_col=picks["ta"])

    def _code(self, picks: dict[str, str], name: str | None) -> str:
        return dv.variables.calc_vpd_from_ta_rh_to_code(
            ta_col=picks["ta"], rh_col=picks["rh"], name=name)
