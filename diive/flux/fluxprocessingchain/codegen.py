"""
FLUXPROCESSINGCHAIN.CODEGEN: RENDER CHAIN CHOICES AS A RUNNABLE SCRIPT
=====================================================================

Turn the parameter choices a caller makes (e.g. in the GUI's flux-chain tab)
into a self-contained, runnable diive script — so a point-and-click run stays
reproducible. Two renderers:

- :func:`chain_to_code` — the single-call ``FluxConfig`` + ``run_chain`` form.
- :func:`level2_to_code` — the composable ``init_flux_data`` + ``run_level2``
  form (the incremental per-level path).

Default-valued kwargs are omitted from the rendered call (introspected from the
``FluxConfig`` dataclass and the ``init_flux_data`` signature), so the script
shows only the decisions that actually differ from the defaults.

This belongs in the library (not the GUI): it encodes the exact API call shape
and must stay correct as that API evolves; the GUI only calls it.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import dataclasses
import inspect

#: Columns init_flux_data computes itself and rejects if already present.
_RESERVED = ("SW_IN_POT", "DAYTIME", "NIGHTTIME")


def _init_defaults() -> dict:
    from diive.flux.fluxprocessingchain.levels import init_flux_data
    return {p.name: p.default
            for p in inspect.signature(init_flux_data).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _config_defaults() -> dict:
    from diive.flux.fluxprocessingchain.container import FluxConfig
    return {f.name: f.default for f in dataclasses.fields(FluxConfig)
            if f.default is not dataclasses.MISSING}


def _level31_defaults() -> dict:
    from diive.flux.fluxprocessingchain.levels import run_level31
    return {p.name: p.default
            for p in inspect.signature(run_level31).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _kwargs_lines(kwargs: dict, defaults: dict) -> list[str]:
    """One ``key=repr(value),`` line per kwarg, dropping any that equal their default."""
    lines = []
    for key, value in kwargs.items():
        if key in defaults and value == defaults[key]:
            continue
        lines.append(f"    {key}={value!r},")
    return lines


def _drop_reserved_line(df_var: str) -> str:
    return (f"{df_var} = {df_var}.drop(columns=[c for c in {_RESERVED!r} "
            f"if c in {df_var}.columns])")


def chain_to_code(init_kwargs: dict, config_kwargs: dict,
                  df_var: str = "df", load_hint: str | None = None) -> str:
    """Render ``init_flux_data`` + ``FluxConfig`` + ``run_chain`` as a script.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        config_kwargs: kwargs for ``FluxConfig``.
        df_var: variable name used for the input DataFrame.
        load_hint: optional expression to assign to ``df_var`` at the top
            (e.g. ``"dv.load_parquet('data.parquet')"``).

    Returns:
        A runnable Python script as a string.
    """
    lines = ["import diive as dv",
             "from diive.flux.fluxprocessingchain import "
             "FluxConfig, init_flux_data, run_chain", ""]
    if load_hint:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += [f"# init_flux_data computes {', '.join(_RESERVED)} -- drop any pre-existing",
              _drop_reserved_line(df_var), ""]
    lines += ["data = init_flux_data(", f"    df={df_var},",
              *_kwargs_lines(init_kwargs, _init_defaults()), ")", ""]
    lines += ["cfg = FluxConfig(", *_kwargs_lines(config_kwargs, _config_defaults()), ")", ""]
    lines += ["data = run_chain(data, cfg)", "final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def _init_level2_lines(imports: str, init_kwargs: dict, level2_settings: dict,
                       df_var: str, load_hint: str | None) -> list[str]:
    """Shared header through the ``run_level2`` call (no trailing ``final_df``)."""
    lines = ["import diive as dv", imports, ""]
    if load_hint:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += [f"# init_flux_data computes {', '.join(_RESERVED)} -- drop any pre-existing",
              _drop_reserved_line(df_var), ""]
    lines += ["data = init_flux_data(", f"    df={df_var},",
              *_kwargs_lines(init_kwargs, _init_defaults()), ")", ""]
    lines.append("data = run_level2(")
    lines.append("    data,")
    for test, settings in level2_settings.items():
        lines.append(f"    {test}={settings!r},")
    lines.append(")")
    return lines


def level2_to_code(init_kwargs: dict, level2_settings: dict,
                   df_var: str = "df", load_hint: str | None = None) -> str:
    """Render ``init_flux_data`` + ``run_level2`` (composable form) as a script.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    lines = _init_level2_lines(
        "from diive.flux.fluxprocessingchain import init_flux_data, run_level2",
        init_kwargs, level2_settings, df_var, load_hint)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def _level31_block(level31_kwargs: dict) -> list[str]:
    """The ``run_level31`` call block (blank-line separated, no trailing ``final_df``)."""
    return ["", "data = run_level31(", "    data,",
            *_kwargs_lines(level31_kwargs, _level31_defaults()), ")"]


def level31_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render ``init_flux_data`` + ``run_level2`` + ``run_level31`` (composable form).

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    lines = _init_level2_lines(
        "from diive.flux.fluxprocessingchain import "
        "init_flux_data, run_level2, run_level31",
        init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def _step_defaults(method_name: str) -> dict:
    """Default values of a ``StepwiseOutlierDetection.flag_*`` method's parameters."""
    from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
    method = getattr(StepwiseOutlierDetection, method_name)
    return {p.name: p.default
            for p in inspect.signature(method).parameters.values()
            if p.default is not inspect.Parameter.empty}


def level32_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    level32_steps: list[dict],
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render the composable chain through Level 3.2 (outlier detection).

    Level 3.2 is a stateful chain: ``make_level32_detector`` builds a
    ``StepwiseOutlierDetection``, then each step calls one ``flag_outliers_*`` /
    ``flag_*`` method followed by ``addflag()``, and ``run_level32`` aggregates
    the flags into the level QCF.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]`` — one
            committed outlier test per entry, in chain order.
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    lines = _init_level2_lines(
        "from diive.flux.fluxprocessingchain import (init_flux_data, run_level2, "
        "run_level31,\n    make_level32_detector, run_level32)",
        init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += ["", "data, sod = make_level32_detector(data)"]
    for step in level32_steps:
        method = step["method"]
        kwarg_lines = _kwargs_lines(step.get("kwargs", {}), _step_defaults(method))
        if kwarg_lines:
            lines += [f"sod.{method}(", *kwarg_lines, ")"]
        else:
            lines.append(f"sod.{method}()")
        lines.append("sod.addflag()")
    lines += ["data = run_level32(data, outlier_detector=sod)", "final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"
