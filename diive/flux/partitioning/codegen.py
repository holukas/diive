"""
FLUX.PARTITIONING.CODEGEN: RENDER NEE-PARTITIONING CHOICES AS A RUNNABLE SCRIPT
===============================================================================

Turn the choices a caller makes (e.g. in the GUI's NEE partitioning tab) into a
self-contained, runnable diive snippet with the picked input columns, site
coordinates and VPD unit, so a point-and-click partitioning stays reproducible.
``partition_nee_to_code`` renders ``partition_nee(..., method=..., mode=...)``,
``partitioning_to_code`` the matching per-port ``partition_nee_*`` function.
Belongs in the library (not the GUI): it encodes the exact API call shape per
method and must stay correct as that API evolves. The GUI only calls it.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

# Per-method call shape: the functional wrapper plus the ordered (kwarg, column-key)
# pairs that feed it. Column keys match the GUI tabs' input "key" entries.
_METHODS = {
    "NT_OF": {
        "func": "partition_nee_nighttime_oneflux",
        "series": [("nee", "nee"), ("ta", "ta"), ("sw_in", "sw_in"),
                   ("nee_f", "nee_f"), ("ta_f", "ta_f")],
        "needs_lat": True, "needs_lon": False, "needs_utc": False,
        "has_vpd_unit": False,
    },
    "NT_RP": {
        "func": "partition_nee_nighttime_reddyproc",
        "series": [("nee", "nee"), ("ta", "ta"), ("sw_in", "sw_in"),
                   ("nee_f", "nee_f"), ("ta_f", "ta_f")],
        "needs_lat": True, "needs_lon": True, "needs_utc": True,
        "has_vpd_unit": False,
    },
    "DT_RP": {
        "func": "partition_nee_daytime_reddyproc",
        "series": [("nee", "nee"), ("ta", "ta"), ("vpd", "vpd"),
                   ("sw_in", "sw_in"), ("nee_sd", "nee_sd")],
        "needs_lat": True, "needs_lon": True, "needs_utc": True,
        "has_vpd_unit": True,
    },
    "DT_OF": {
        "func": "partition_nee_daytime_oneflux",
        "series": [("nee", "nee"), ("ta", "ta"), ("sw_in", "sw_in"),
                   ("ta_f", "ta_f"), ("sw_in_f", "sw_in_f"), ("vpd", "vpd"),
                   ("nee_f", "nee_f")],
        "needs_lat": False, "needs_lon": False, "needs_utc": False,
        "has_vpd_unit": True,
    },
}


def partitioning_to_code(method_suffix: str, picks: dict[str, str], *,
                         lat: float | None = None, lon: float | None = None,
                         utc_offset: int | None = None, vpd_in_kpa: bool = True,
                         df_var: str = "df", load_hint: str | None = None) -> str:
    """Render a ``dv.flux.partition_nee_*(...)`` snippet as a string.

    Mirrors what the GUI's NEE-partitioning tabs run: partition measured NEE into
    GPP and RECO with one of the four faithful ports, building each input Series
    from a column of ``df_var`` and emitting the method's ``RECO_*`` / ``GPP_*``
    columns.

    Args:
        method_suffix: which port to call - one of ``"NT_OF"`` (nighttime ONEFlux),
            ``"NT_RP"`` (nighttime REddyProc), ``"DT_RP"`` (daytime REddyProc) or
            ``"DT_OF"`` (daytime ONEFlux).
        picks: maps each input key (``nee`` / ``ta`` / ``sw_in`` / ``nee_f`` /
            ``ta_f`` / ``sw_in_f`` / ``vpd`` / ``nee_sd``) to the chosen column
            name. Optional inputs (``nee_sd``, ``nee_f`` for ``"DT_OF"``) may be
            missing or empty.
        lat: site latitude (decimal degrees); required by the methods that need it.
        lon: site longitude (decimal degrees); required by the methods that need it.
        utc_offset: UTC offset (hours) of the timestamps; required where needed.
        vpd_in_kpa: whether the VPD column is in kPa (daytime methods only).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    spec = _METHODS[method_suffix]

    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]

    lines.append(f"results = dv.flux.{spec['func']}(")
    for kwarg, key in spec["series"]:
        col = picks.get(key)
        if not col:
            continue  # optional input left unset
        lines.append(f"    {kwarg}={df_var}[{col!r}],")
    if spec["needs_lat"]:
        lines.append(f"    lat={lat!r},")
    if spec["needs_lon"]:
        lines.append(f"    lon={lon!r},")
    if spec["needs_utc"]:
        lines.append(f"    utc_offset={utc_offset!r},")
    if spec["has_vpd_unit"]:
        lines.append(f"    vpd_in_kpa={vpd_in_kpa!r},")
    lines.append(")")
    lines += ["",
              "# Results DataFrame holds the partitioned RECO_{0} / GPP_{0} columns.".format(method_suffix),
              "print(results.columns.tolist())"]
    return "\n".join(lines) + "\n"


#: Column suffix each (method, mode) of ``partition_nee`` writes.
_NEE_SUFFIX = {("nighttime", "oneflux"): "NT_OF", ("nighttime", "reddyproc"): "NT_RP",
               ("daytime", "oneflux"): "DT_OF", ("daytime", "reddyproc"): "DT_RP"}

_COORDS = ("lat", "lon", "utc_offset")


def partition_nee_to_code(method: str, mode: str, picks: dict[str, str], *,
                          lat: float | None = None, lon: float | None = None,
                          utc_offset: float | None = None, vpd_in_kpa: bool = True,
                          df_var: str = "df", load_hint: str | None = None) -> str:
    """Render a ``dv.flux.partition_nee(..., method=..., mode=...)`` snippet.

    Mirrors what the GUI's NEE partitioning tab runs. Only the inputs the chosen
    method and mode need are written (see
    :func:`diive.flux.partitioning.partitioning_inputs`), each built from a
    column of ``df_var``.

    Args:
        method: ``'nighttime'`` or ``'daytime'``.
        mode: ``'oneflux'`` or ``'reddyproc'``.
        picks: maps each input name (``nee`` / ``ta`` / ``sw_in`` / ``ta_f`` /
            ``sw_in_f`` / ``nee_f`` / ``vpd``, plus the optional inputs of
            :func:`diive.flux.partitioning.partitioning_optional_inputs`:
            ``nee_f`` for daytime / oneflux, ``nee_sd`` for daytime /
            reddyproc) to the chosen column name. Inputs the variant does not
            use are ignored. An optional input may be missing or empty.
        lat: site latitude (decimal degrees), written where the variant needs it.
        lon: site longitude (decimal degrees), written where the variant needs it.
        utc_offset: UTC offset (hours) of the timestamps, written where needed.
        vpd_in_kpa: whether the VPD column is in kPa (daytime methods only).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    # Imported here so the module stays light for callers that only need the
    # per-port renderer above.
    from diive.flux.partitioning.nee import partitioning_inputs, partitioning_optional_inputs

    needed = partitioning_inputs(method, mode)
    coords = {"lat": lat, "lon": lon, "utc_offset": utc_offset}

    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]

    lines += ["results = dv.flux.partition_nee(",
              f"    nee={df_var}[{picks['nee']!r}],",
              f"    method={method!r},",
              f"    mode={mode!r},"]
    series = [n for n in needed if n not in _COORDS]
    series += list(partitioning_optional_inputs(method, mode))
    for name in series:
        col = picks.get(name)
        if not col:
            continue  # optional input left unset
        lines.append(f"    {name}={df_var}[{col!r}],")
    for name in _COORDS:
        if name in needed:
            lines.append(f"    {name}={coords[name]!r},")
    if "vpd" in needed:
        lines.append(f"    vpd_in_kpa={vpd_in_kpa!r},")
    lines.append(")")
    suffix = _NEE_SUFFIX[(method, mode)]
    lines += ["",
              f"# Results DataFrame holds the partitioned RECO_{suffix} / GPP_{suffix} columns.",
              "print(results.columns.tolist())"]
    return "\n".join(lines) + "\n"
