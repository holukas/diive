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


def _signature_defaults(cls) -> dict:
    return {p.name: p.default
            for p in inspect.signature(cls.__init__).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _hampel_defaults() -> dict:
    from diive.preprocessing.outlier_detection.hampel import Hampel
    return _signature_defaults(Hampel)


def _render_call(class_path: str, kwargs: dict, defaults: dict, repeat: bool,
                 series_var: str, var_name: str | None) -> str:
    """Shared renderer: ``h = <class_path>(series=…, …).run(...)`` + result lines,
    dropping any kwarg equal to its constructor default."""
    lines = ["import diive as dv", ""]
    if var_name is not None:
        lines += [f"{series_var} = df[{var_name!r}]", ""]
    lines.append(f"h = {class_path}(")
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


def localsd_to_code(kwargs: dict, repeat: bool = True, *,
                    series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.LocalSD(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.localsd import LocalSD
    return _render_call("dv.outliers.LocalSD", kwargs, _signature_defaults(LocalSD),
                        repeat, series_var, var_name)


def zscore_to_code(kwargs: dict, repeat: bool = True, *,
                   series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.zScore(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.zscore import zScore
    return _render_call("dv.outliers.zScore", kwargs, _signature_defaults(zScore),
                        repeat, series_var, var_name)


def zscorerolling_to_code(kwargs: dict, repeat: bool = True, *,
                          series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.zScoreRolling(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.zscore import zScoreRolling
    return _render_call("dv.outliers.zScoreRolling", kwargs, _signature_defaults(zScoreRolling),
                        repeat, series_var, var_name)


def zscoreincrements_to_code(kwargs: dict, repeat: bool = True, *,
                             series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.zScoreIncrements(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.incremental import zScoreIncrements
    return _render_call("dv.outliers.zScoreIncrements", kwargs,
                        _signature_defaults(zScoreIncrements), repeat, series_var, var_name)


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
    return _render_call("dv.outliers.Hampel", kwargs, _hampel_defaults(),
                        repeat, series_var, var_name)
