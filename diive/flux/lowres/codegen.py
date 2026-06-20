"""
FLUX.LOWRES.CODEGEN: RENDER LOW-RES FLUX CHOICES AS A RUNNABLE SCRIPT
====================================================================

Turn the choices a caller makes (e.g. in the GUI's random-uncertainty tab) into
a self-contained, runnable diive snippet — so a point-and-click run stays
reproducible. Belongs in the library (not the GUI): it encodes the exact API
call shape and must stay correct as that API evolves; the GUI only calls it.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def randunc_to_code(fluxcol: str, fluxgapfilledcol: str, tacol: str, vpdcol: str,
                    swincol: str, *, vpd_in_kpa: bool = True,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render a ``RandomUncertaintyPAS20(...).run()`` snippet as a string.

    Mirrors what the GUI's random-uncertainty tab runs: the hierarchical 4-method
    PAS20 random-uncertainty estimate for a measured flux from its three
    similarity drivers (TA / VPD / SW_IN), emitting the ``{fluxcol}_RANDUNC``
    column.

    Args:
        fluxcol: measured flux to estimate the uncertainty for.
        fluxgapfilledcol: gap-filled flux (for the cumulative propagation).
        tacol: air-temperature similarity driver (deg C).
        vpdcol: vapour-pressure-deficit similarity driver.
        swincol: short-wave incoming radiation similarity driver (W m-2).
        vpd_in_kpa: whether ``vpdcol`` is in kPa (diive convention).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += [f"flux = {fluxcol!r}",
              "",
              "randunc = dv.flux.RandomUncertaintyPAS20(",
              f"    df={df_var},",
              "    fluxcol=flux,",
              f"    fluxgapfilledcol={fluxgapfilledcol!r},",
              f"    tacol={tacol!r},",
              f"    vpdcol={vpdcol!r},",
              f"    swincol={swincol!r},",
              f"    vpd_in_kpa={vpd_in_kpa!r},",
              ")",
              "randunc.run()",
              "",
              f"randunc_series = randunc.randunc_series  # {fluxcol}_RANDUNC",
              "randunc.report_method_summary()"]
    return "\n".join(lines) + "\n"
