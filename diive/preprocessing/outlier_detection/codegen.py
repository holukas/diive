"""
OUTLIER_DETECTION.CODEGEN: RENDER OUTLIER-DETECTION CHOICES AS A RUNNABLE SCRIPT
===============================================================================

Turn the parameter choices a caller makes (e.g. in the GUI's Hampel tab) into a
self-contained, runnable diive snippet — so a point-and-click run stays
reproducible.

Default-valued kwargs are omitted from the rendered call (introspected from the
``Hampel.__init__`` signature), so the snippet shows only the decisions that
actually differ from the defaults.

This belongs in the library (not the GUI): it encodes the exact API call shape
and must stay correct as that API evolves; the GUI only calls it.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import inspect


def _hampel_defaults() -> dict:
    from diive.preprocessing.outlier_detection.hampel import Hampel
    return {p.name: p.default
            for p in inspect.signature(Hampel.__init__).parameters.values()
            if p.default is not inspect.Parameter.empty}


def hampel_to_code(kwargs: dict, repeat: bool = True, *,
                   series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.Hampel(...).run(...)`` snippet as a string.

    Args:
        kwargs: keyword arguments for ``Hampel`` (without ``series``).
        repeat: the ``run(repeat=...)`` value (default ``True``, then omitted).
        series_var: variable name used for the input Series.
        var_name: if given, prepend ``series = df['var_name']`` so the snippet
            is runnable as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    defaults = _hampel_defaults()
    lines = ["import diive as dv", ""]
    if var_name is not None:
        lines += [f"{series_var} = df[{var_name!r}]", ""]
    lines.append("h = dv.outliers.Hampel(")
    lines.append(f"    series={series_var},")
    for key, value in kwargs.items():
        if key in defaults and value == defaults[key]:
            continue
        lines.append(f"    {key}={value!r},")
    lines.append(").run()" if repeat else ").run(repeat=False)")
    lines += ["",
              "cleaned = h.filteredseries  # outliers set to NaN",
              "flag = h.overall_flag       # 0 = ok, 2 = outlier"]
    return "\n".join(lines) + "\n"
