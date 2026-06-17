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
    lines += _level32_block(level32_steps)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def _level32_block(level32_steps: list[dict]) -> list[str]:
    """The L3.2 stateful chain block (no trailing ``final_df``)."""
    lines = ["", "data, sod = make_level32_detector(data)"]
    for step in level32_steps:
        method = step["method"]
        kwarg_lines = _kwargs_lines(step.get("kwargs", {}), _step_defaults(method))
        if kwarg_lines:
            lines += [f"sod.{method}(", *kwarg_lines, ")"]
        else:
            lines.append(f"sod.{method}()")
        lines.append("sod.addflag()")
    lines.append("data = run_level32(data, outlier_detector=sod)")
    return lines


def _level33_defaults() -> dict:
    from diive.flux.fluxprocessingchain.levels import run_level33_constant_ustar
    return {p.name: p.default
            for p in inspect.signature(run_level33_constant_ustar).parameters.values()
            if p.default is not inspect.Parameter.empty}


def level33_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    level32_steps: list[dict], level33_kwargs: dict,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render the composable chain through Level 3.3 (constant-USTAR filtering).

    L3.3 requires L3.2 to have run, so the L3.2 outlier chain is always rendered
    before the ``run_level33_constant_ustar`` call.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]``.
        level33_kwargs: kwargs for ``run_level33_constant_ustar`` (``thresholds``,
            optional ``threshold_labels``).
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    lines = _init_level2_lines(
        "from diive.flux.fluxprocessingchain import (init_flux_data, run_level2, "
        "run_level31,\n    make_level32_detector, run_level32, "
        f"{_level33_import_name(level33_kwargs)})",
        init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += _level32_block(level32_steps)
    lines += _level33_block(level33_kwargs)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def _level33_is_detection(level33_kwargs: dict) -> bool:
    """True when the L3.3 spec is auto-detection (vs constant thresholds)."""
    return bool(level33_kwargs) and bool(level33_kwargs.get("_detection"))


def _level33_import_name(level33_kwargs: dict) -> str:
    """The L3.3 function symbol to import for this spec."""
    return ("run_level33_ustar_detection" if _level33_is_detection(level33_kwargs)
            else "run_level33_constant_ustar")


def _level33_detection_defaults() -> dict:
    from diive.flux.fluxprocessingchain.levels import run_level33_ustar_detection
    return {p.name: p.default
            for p in inspect.signature(run_level33_ustar_detection).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _level33_block(level33_kwargs: dict) -> list[str]:
    """The L3.3 call block (constant thresholds or auto-detection; no ``final_df``)."""
    if _level33_is_detection(level33_kwargs):
        # Strip the internal marker; render the detection call.
        kw = {k: v for k, v in level33_kwargs.items() if k != "_detection"}
        return ["", "data = run_level33_ustar_detection(", "    data,",
                *_kwargs_lines(kw, _level33_detection_defaults()), ")"]
    return ["", "data = run_level33_constant_ustar(", "    data,",
            *_kwargs_lines(level33_kwargs, _level33_defaults()), ")"]


def _level41_block(level41_cfg: dict) -> list[str]:
    """The L4.1 gap-filling block: one ``run_level41_*`` per selected method.

    ``level41_cfg`` shape (the GUI flux-chain tab's ``_level41_cfg``)::

        {"methods": ["rf", "xgb", "mds"],   # subset, in this canonical order
         "features": ["TA_...", "SW_IN_...", "VPD_..."],   # rf / xgb predictors
         "mds": {"swin": "SW_IN_...", "ta": "TA_...", "vpd": "VPD_..."}}

    RF and XGBoost share one ``make_level41_engineer`` instance (feature
    engineering runs once and is reused across methods and USTAR scenarios).
    """
    methods = level41_cfg.get("methods", [])
    features = level41_cfg.get("features") or []
    reduce = level41_cfg.get("reduce_features")
    lines: list[str] = []
    if any(m in methods for m in ("rf", "xgb")):
        lines += ["", "engineer = make_level41_engineer(", "    data,",
                  f"    features={features!r},", ")"]

    def _ml_call(fn: str, model_kwargs: dict) -> list[str]:
        body = ["", f"data = {fn}(", "    data,",
                f"    features={features!r},", "    engineer=engineer,"]
        if reduce:
            body.append("    reduce_features=True,")
        for k, val in (model_kwargs or {}).items():
            body.append(f"    {k}={val!r},")
        body.append(")")
        return body

    if "rf" in methods:
        lines += _ml_call("run_level41_rf", level41_cfg.get("rf_kwargs", {}))
    if "xgb" in methods:
        lines += _ml_call("run_level41_xgb", level41_cfg.get("xgb_kwargs", {}))
    if "mds" in methods:
        mds = level41_cfg.get("mds") or {}
        body = ["", "data = run_level41_mds(", "    data,",
                f"    swin={mds.get('swin')!r},", f"    ta={mds.get('ta')!r},",
                f"    vpd={mds.get('vpd')!r},"]
        for k in ("ta_tol", "vpd_tol"):
            if k in mds:
                body.append(f"    {k}={mds[k]!r},")
        body.append(")")
        lines += body
    return lines


#: L4.2 partitioning variant token -> (function suffix, kwarg keys it accepts).
#: Kwarg keys are rendered from level42_cfg in this order when present.
_LEVEL42_VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nt_of": ("nighttime_oneflux", ("ta", "sw_in", "ta_f", "gapfill_method")),
    "nt_rp": ("nighttime_reddyproc", ("ta", "sw_in", "ta_f", "gapfill_method")),
    "dt_rp": ("daytime_reddyproc", ("ta_f", "vpd_f", "sw_in_f", "nee_sd", "vpd_in_kpa")),
    "dt_of": ("daytime_oneflux", ("ta", "sw_in", "ta_f", "sw_in_f", "vpd_f", "vpd_in_kpa")),
}

#: Kwargs whose default is omitted from the rendered L4.2 call.
_LEVEL42_DEFAULTS: dict = {"gapfill_method": "mds", "vpd_in_kpa": True, "nee_sd": None}


def _level42_fn_names(variants: list[str]) -> list[str]:
    """The ``run_level42_*`` symbols to import for the selected variants."""
    return [f"run_level42_{_LEVEL42_VARIANTS[v][0]}"
            for v in ("nt_of", "nt_rp", "dt_rp", "dt_of") if v in variants]


def _level42_block(level42_cfg: dict) -> list[str]:
    """The L4.2 partitioning block: one ``run_level42_*`` per selected variant.

    ``level42_cfg`` shape::

        {"variants": ["nt_of", "nt_rp", "dt_rp", "dt_of"],  # subset, canonical order
         "gapfill_method": "mds",          # nighttime variants (default 'mds')
         "ta": "...", "sw_in": "...",      # measured meteo
         "ta_f": "...", "sw_in_f": "...", "vpd_f": "...",   # gap-filled meteo
         "nee_sd": "...",                  # optional, daytime REddyProc
         "vpd_in_kpa": True}               # optional, daytime variants

    Partitioning is additive across variants — each selected variant emits its
    own ``run_level42_*`` call, just like the L4.1 gap-filling methods.
    """
    variants = level42_cfg.get("variants", [])
    lines: list[str] = []
    for variant in ("nt_of", "nt_rp", "dt_rp", "dt_of"):
        if variant not in variants:
            continue
        suffix, keys = _LEVEL42_VARIANTS[variant]
        body = ["", f"data = run_level42_{suffix}(", "    data,"]
        for key in keys:
            if key not in level42_cfg or level42_cfg[key] is None:
                continue
            # Drop kwargs that equal their library default to keep the script minimal.
            if key in _LEVEL42_DEFAULTS and level42_cfg[key] == _LEVEL42_DEFAULTS[key]:
                continue
            body.append(f"    {key}={level42_cfg[key]!r},")
        body.append(")")
        lines += body
    return lines


def level42_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    level32_steps: list[dict], level33_kwargs: dict, level41_cfg: dict,
                    level42_cfg: dict,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render the composable chain through Level 4.2 (NEE partitioning).

    L4.2 sits after L4.1, so the full L2 -> L4.1 chain is rendered first.
    Partitioning is additive across variants: each selected variant
    (``nt_of`` / ``nt_rp`` / ``dt_rp`` / ``dt_of``) emits its own
    ``run_level42_*`` call. The nighttime variants additionally consume an L4.1
    gap-filled NEE (``level42_cfg['gapfill_method']``).

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]``.
        level33_kwargs: kwargs for ``run_level33_constant_ustar``.
        level41_cfg: gap-filling selection — see :func:`_level41_block`.
        level42_cfg: partitioning selection — see :func:`_level42_block`.
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    methods = level41_cfg.get("methods", [])
    variants = level42_cfg.get("variants", [])
    names = ["init_flux_data", "run_level2", "run_level31",
             "make_level32_detector", "run_level32",
             _level33_import_name(level33_kwargs)]
    if any(m in methods for m in ("rf", "xgb")):
        names.append("make_level41_engineer")
    for method in ("mds", "rf", "xgb"):
        if method in methods:
            names.append(f"run_level41_{method}")
    names += _level42_fn_names(variants)
    imports = ("from diive.flux.fluxprocessingchain import (\n    "
               + ", ".join(names) + ")")
    lines = _init_level2_lines(imports, init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += _level32_block(level32_steps)
    lines += _level33_block(level33_kwargs)
    lines += _level41_block(level41_cfg)
    lines += _level42_block(level42_cfg)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"


def level41_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    level32_steps: list[dict], level33_kwargs: dict, level41_cfg: dict,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render the composable chain through Level 4.1 (gap-filling).

    L4.1 requires the full L2 -> L3.3 chain to have run, so that chain is always
    rendered before the ``run_level41_*`` calls.  Gap-filling is additive across
    methods: each selected method (``rf`` / ``xgb`` / ``mds``) emits its own
    ``run_level41_*`` call; RF and XGBoost share one ``FeatureEngineer``.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]``.
        level33_kwargs: kwargs for ``run_level33_constant_ustar``.
        level41_cfg: gap-filling selection — see :func:`_level41_block`.
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    methods = level41_cfg.get("methods", [])
    names = ["init_flux_data", "run_level2", "run_level31",
             "make_level32_detector", "run_level32",
             _level33_import_name(level33_kwargs)]
    if any(m in methods for m in ("rf", "xgb")):
        names.append("make_level41_engineer")
    for method in ("mds", "rf", "xgb"):
        if method in methods:
            names.append(f"run_level41_{method}")
    imports = ("from diive.flux.fluxprocessingchain import (\n    "
               + ", ".join(names) + ")")
    lines = _init_level2_lines(imports, init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += _level32_block(level32_steps)
    lines += _level33_block(level33_kwargs)
    lines += _level41_block(level41_cfg)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"
