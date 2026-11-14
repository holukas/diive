"""
FLUX.LOWRES.CODEGEN: RENDER LOW-RES FLUX CHOICES AS A RUNNABLE SCRIPT
=====================================================================

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


def jointunc_to_code(randunccol: str, scenario_lower_col: str, scenario_upper_col: str,
                     *, fluxgapfilledcol: str | None = None, divisor: float = 2.0,
                     df_var: str = "df", load_hint: str | None = None) -> str:
    """Render a ``JointUncertaintyPAS20(...).run()`` snippet as a string.

    Mirrors what the GUI's joint-uncertainty tab runs: combine a per-record
    random uncertainty with the scenario-ensemble (e.g. USTAR-threshold)
    percentile spread in quadrature, emitting the ``{base}_JOINTUNC`` column.

    Args:
        randunccol: per-record random uncertainty column (e.g. ``NEE_CUT_REF_RANDUNC``).
        scenario_lower_col: lower-percentile scenario flux (NEE 16th / LE-H 25th).
        scenario_upper_col: upper-percentile scenario flux (NEE 84th / LE-H 75th).
        fluxgapfilledcol: optional gap-filled flux for the cumulative propagation.
        divisor: percentile-range -> 1-sigma factor (2.0 for NEE 16/84,
            1.349 for the LE/H 25/75 IQR).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += ["jointunc = dv.flux.JointUncertaintyPAS20(",
              f"    df={df_var},",
              f"    randunccol={randunccol!r},",
              f"    scenario_lower_col={scenario_lower_col!r},",
              f"    scenario_upper_col={scenario_upper_col!r},",
              f"    fluxgapfilledcol={fluxgapfilledcol!r},",
              f"    divisor={divisor!r},",
              ")",
              "jointunc.run()",
              "",
              "jointunc_series = jointunc.jointunc_series  # {base}_JOINTUNC",
              "jointunc.report_summary()"]
    return "\n".join(lines) + "\n"


def ffp_footprint_to_code(zm: float, h: float, ol: float, sigmav: float, ustar: float,
                          *, z0: float | None = None, umean: float | None = None,
                          wind_dir: float | None = None, rs="default", rslayer: bool = False,
                          nx: int = 1000, crop: bool = False,
                          source_area: float = 0.8) -> str:
    """Render a ``ffp_footprint(...)`` snippet with its ``FootprintPlot`` as a string.

    Mirrors what the GUI's flux footprint tab runs: the footprint of one record
    (Kljun et al. 2015) and its map with the crosswind-integrated profile.
    Arguments that equal the defaults of ``ffp_footprint`` are left out.

    Args:
        zm: Measurement height above displacement height (m).
        h: Boundary-layer height (m).
        ol: Obukhov length (m).
        sigmav: Standard deviation of the lateral wind velocity (m s-1).
        ustar: Friction velocity (m s-1).
        z0: Roughness length (m), or None to use ``umean``.
        umean: Mean wind speed at ``zm`` (m s-1), used without ``z0``.
        wind_dir: Wind direction (degrees), or None for the along-wind frame.
        rs: Source-area fractions of the contour lines, None for no contour lines.
            ``"default"`` or the default fractions leave the argument out.
        rslayer: Allow ``zm`` in the roughness sublayer.
        nx: Number of grid elements.
        crop: Crop the grid to the largest contour line.
        source_area: Source area reported and highlighted in the plot (fraction).

    Returns:
        A runnable Python snippet as a string.
    """
    args = [f"zm={zm!r}", f"h={h!r}", f"ol={ol!r}", f"sigmav={sigmav!r}", f"ustar={ustar!r}"]
    if z0 is not None:
        args.append(f"z0={z0!r}")
    else:
        args.append(f"umean={umean!r}")
    if wind_dir is not None:
        args.append(f"wind_dir={wind_dir!r}")
    from diive.flux.lowres.footprint import DEFAULT_RS
    if rs is None:
        args.append("rs=None")
    elif not isinstance(rs, str):
        rs = [float(r) for r in rs]
        is_default = len(rs) == len(DEFAULT_RS) and all(
            abs(a - b) < 1e-9 for a, b in zip(rs, DEFAULT_RS))
        if not is_default:
            args.append(f"rs={rs!r}")
    if rslayer:
        args.append("rslayer=True")
    if nx != 1000:
        args.append(f"nx={nx!r}")
    if crop:
        args.append("crop=True")
    lines = ["import matplotlib.pyplot as plt",
             "",
             "import diive as dv",
             "",
             "# Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple",
             "# two-dimensional parameterisation for Flux Footprint Prediction (FFP).",
             "# Geoscientific Model Development, 8(11), 3695-3713.",
             "# https://doi.org/10.5194/gmd-8-3695-2015",
             "fp = dv.flux.ffp_footprint("]
    lines += [f"    {a}," for a in args]
    lines += [")",
              f"print(f\"Peak at {{fp.x_ci_max:.1f}} m\")",
              "",
              f"dv.plotting.FootprintPlot(fp, source_area={source_area!r}).plot()",
              "plt.show()"]
    return "\n".join(lines) + "\n"
