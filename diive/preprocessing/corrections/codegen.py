"""
CORRECTIONS.CODEGEN: RENDER A CORRECTION CHAIN AS A RUNNABLE SCRIPT
===================================================================

Turn an ordered list of corrections (the same ``{"key", "kwargs"}`` dicts that
:func:`diive.preprocessing.corrections.apply.apply_corrections` consumes) into a
runnable diive snippet, so a point-and-click correction run stays reproducible.

This mirrors the key -> function mapping in ``apply_corrections`` and must stay
in lockstep with it. It belongs in the library (not the GUI) for the same reason
the outlier-detection codegen does.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def corrections_to_code(corrections: list[dict],
                        *,
                        site_lat: float,
                        site_lon: float,
                        utc_offset: int,
                        in_var: str = "cleaned",
                        out_var: str = "corrected") -> str:
    """Render a chain of corrections as a runnable snippet (no imports).

    The snippet starts ``<out_var> = <in_var>.copy()`` and then applies each
    correction in turn via ``dv.corrections.*``. ``dv`` is assumed already
    imported (the snippet is appended to a block that imports it).

    Args:
        corrections: Ordered list of ``{"key": str, "kwargs": dict}`` dicts.
        site_lat, site_lon, utc_offset: site coordinates the radiation
            zero-offset correction needs.
        in_var: name of the series variable the corrections start from.
        out_var: name of the corrected-series variable produced.

    Returns:
        A snippet (string) with no leading/trailing imports, ending in a newline.
        Returns an empty string if ``corrections`` is empty.
    """
    from diive.preprocessing.qaqc.measurements import (
        CORR_RADIATION_ZERO_OFFSET,
        CORR_RELATIVEHUMIDITY_OFFSET,
        CORR_SETTO_MAX,
        CORR_SETTO_MIN,
        CORR_SETTO_VALUE,
        CORR_SET_EXACT_TO_MISSING,
    )

    if not corrections:
        return ""

    lines = [f"{out_var} = {in_var}.copy()"]
    for corr in corrections:
        key = corr["key"]
        kw = corr.get("kwargs", {})
        if key == CORR_RADIATION_ZERO_OFFSET:
            lines.append(
                f"{out_var} = dv.corrections.remove_radiation_zero_offset("
                f"series={out_var}, lat={site_lat!r}, lon={site_lon!r}, "
                f"utc_offset={utc_offset!r})")
        elif key == CORR_RELATIVEHUMIDITY_OFFSET:
            lines.append(
                f"{out_var} = dv.corrections.remove_relativehumidity_offset("
                f"series={out_var})")
        elif key == CORR_SETTO_MAX:
            lines.append(
                f"{out_var} = dv.corrections.setto_threshold("
                f"series={out_var}, threshold={kw['threshold']!r}, type='max')")
        elif key == CORR_SETTO_MIN:
            lines.append(
                f"{out_var} = dv.corrections.setto_threshold("
                f"series={out_var}, threshold={kw['threshold']!r}, type='min')")
        elif key == CORR_SETTO_VALUE:
            lines.append(
                f"{out_var} = dv.corrections.setto_value("
                f"series={out_var}, dates={kw['dates']!r}, value={kw.get('value', 0)!r})")
        elif key == CORR_SET_EXACT_TO_MISSING:
            lines.append(
                f"{out_var} = dv.corrections.set_exact_values_to_missing("
                f"series={out_var}, values={kw['values']!r})")
        else:
            raise ValueError(f"Unknown correction key: {key!r}")
    return "\n".join(lines) + "\n"
