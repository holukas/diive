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


def _step_defaults(method: str) -> dict:
    """Default kwargs of a ``StepwiseOutlierDetection.flag_*`` method, so the
    rendered call omits values left at their default."""
    from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
    fn = getattr(StepwiseOutlierDetection, method)
    return {p.name: p.default
            for p in inspect.signature(fn).parameters.values()
            if p.default is not inspect.Parameter.empty}


def stepwise_to_code(steps: list[dict], *, var_name: str,
                     site_lat: float, site_lon: float, utc_offset: int,
                     df_var: str = "df", load_hint: str | None = None) -> str:
    """Render a ``StepwiseOutlierDetection`` chain + overall QCF as a runnable snippet.

    Each step (``{"method": str, "kwargs": dict}``) becomes one ``flag_*`` call
    followed by ``addflag()``; the accumulated test flags are aggregated with
    ``FlagQCF`` into the overall quality flag and the QCF-filtered series — the
    same path the GUI's Stepwise screening tab runs. Default-valued kwargs are
    dropped so the snippet shows only the decisions that differ from the defaults.

    Args:
        steps: ordered chain of ``{"method", "kwargs"}`` outlier tests.
        var_name: the column screened (used for ``col=`` and the result frame).
        site_lat, site_lon, utc_offset: site coordinates the detector needs for
            the day/night split.
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import pandas as pd",
             "import diive as dv",
             "from diive.preprocessing.outlier_detection import StepwiseOutlierDetection",
             "from diive.qaqc import FlagQCF",
             ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]
    # output_middle_timestamp=False keeps the input index so the flags align back
    # to the source frame on merge (what the GUI relies on).
    lines += [
        "sod = StepwiseOutlierDetection(",
        f"    dfin={df_var}[[{var_name!r}]],",
        f"    col={var_name!r},",
        f"    site_lat={site_lat!r},",
        f"    site_lon={site_lon!r},",
        f"    utc_offset={utc_offset!r},",
        "    output_middle_timestamp=False,",
        ")",
    ]
    for step in steps:
        method = step["method"]
        defaults = _step_defaults(method)
        kwarg_lines = [f"    {k}={v!r}," for k, v in step.get("kwargs", {}).items()
                       if not (k in defaults and v == defaults[k])]
        if kwarg_lines:
            lines += [f"sod.{method}(", *kwarg_lines, ")"]
        else:
            lines.append(f"sod.{method}()")
        lines.append("sod.addflag()")
    lines += [
        "",
        f"qcf_input = pd.concat([sod.series_hires_orig.to_frame({var_name!r}), "
        "sod.flags], axis=1)",
        f"qcf = FlagQCF(df=qcf_input, target_col={var_name!r}, idstr='STEPWISE')",
        "qcf.calculate()",
        "",
        "cleaned = qcf.filteredseries  # QCF-filtered series (outliers -> NaN)",
        "flag = qcf.flagqcf            # overall flag: 0 ok / 1 marginal / 2 rejected",
    ]
    return "\n".join(lines) + "\n"


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


def lof_to_code(kwargs: dict, repeat: bool = True, *,
                series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.LocalOutlierFactor(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.lof import LocalOutlierFactor
    return _render_call("dv.outliers.LocalOutlierFactor", kwargs,
                        _signature_defaults(LocalOutlierFactor), repeat, series_var, var_name)


def absolutelimits_to_code(kwargs: dict, repeat: bool = True, *,
                           series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.AbsoluteLimits(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.absolutelimits import AbsoluteLimits
    return _render_call("dv.outliers.AbsoluteLimits", kwargs,
                        _signature_defaults(AbsoluteLimits), repeat, series_var, var_name)


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


def trimlow_to_code(kwargs: dict, repeat: bool = True, *,
                    series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.TrimLow(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`.
    """
    from diive.preprocessing.outlier_detection.trim import TrimLow
    return _render_call("dv.outliers.TrimLow", kwargs, _signature_defaults(TrimLow),
                        repeat, series_var, var_name)


def manualremoval_to_code(kwargs: dict, repeat: bool = True, *,
                          series_var: str = "series", var_name: str | None = None) -> str:
    """Render a ``dv.outliers.ManualRemoval(...).run(...)`` snippet as a string.

    Args mirror :func:`hampel_to_code`. ``repeat`` is accepted for a uniform
    signature but rendered as a plain ``.run()`` — manual removal flags fixed
    timestamps and ignores ``repeat``.
    """
    from diive.preprocessing.outlier_detection.manualremoval import ManualRemoval
    return _render_call("dv.outliers.ManualRemoval", kwargs,
                        _signature_defaults(ManualRemoval), repeat=True,
                        series_var=series_var, var_name=var_name)


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
