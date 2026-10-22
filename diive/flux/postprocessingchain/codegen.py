"""
POSTPROCESSINGCHAIN.CODEGEN: RENDER CHAIN CHOICES AS A RUNNABLE SCRIPT
======================================================================

Turn the parameter choices a caller makes (e.g. in the GUI's flux-chain tab)
into a self-contained, runnable diive script — so a point-and-click run stays
reproducible. Two forms:

- :func:`chain_to_code`: the single-call ``FluxConfig`` + ``run_chain`` form.
- :func:`level2_to_code`, :func:`level31_to_code`, :func:`level32_to_code`,
  :func:`level33_to_code`, :func:`level41_to_code` and
  :func:`level42_to_code`: the composable per-level form, from
  ``init_flux_data`` up to the named level.

Default-valued kwargs are omitted from the rendered call (introspected from the
``FluxConfig`` dataclass and the function signatures), so the script shows
only the decisions that differ from the defaults.

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
    from diive.flux.postprocessingchain.levels import init_flux_data
    return {p.name: p.default
            for p in inspect.signature(init_flux_data).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _config_defaults() -> dict:
    from diive.flux.postprocessingchain.container import FluxConfig
    return {f.name: f.default for f in dataclasses.fields(FluxConfig)
            if f.default is not dataclasses.MISSING}


def _level31_defaults() -> dict:
    from diive.flux.postprocessingchain.levels import run_level31
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
             "from diive.flux.postprocessingchain import "
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
        "from diive.flux.postprocessingchain import init_flux_data, run_level2",
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
        "from diive.flux.postprocessingchain import "
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
        "from diive.flux.postprocessingchain import (init_flux_data, run_level2, "
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
    from diive.flux.postprocessingchain.levels import run_level33_constant_ustar
    return {p.name: p.default
            for p in inspect.signature(run_level33_constant_ustar).parameters.values()
            if p.default is not inspect.Parameter.empty}


def level33_to_code(init_kwargs: dict, level2_settings: dict, level31_kwargs: dict,
                    level32_steps: list[dict], level33_kwargs: dict,
                    df_var: str = "df", load_hint: str | None = None) -> str:
    """Render the composable chain through Level 3.3 (USTAR filtering).

    L3.3 requires L3.2 to have run, so the L3.2 outlier chain is always rendered
    before the L3.3 call.

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]``.
        level33_kwargs: kwargs for ``run_level33_constant_ustar`` (``thresholds``,
            optional ``threshold_labels``, ``nighttime_only``, ``swin_col``), or
            for ``run_level33_ustar_detection`` when it holds
            ``"_detection": True`` (``detection_mode``, ``detection_kwargs``,
            ``thresholds``, ``reference``, ``nighttime_only``, ...). Constant
            thresholds labelled by percentile as ``run_level33_ustar_detection``
            labels them (``CUT_16``, ``CUT_50``, ...) are written as
            ``run_level33_ustar_detection(thresholds={16: ..., 50: ...})``,
            which gives the same flags. The deprecated ``detector_class`` and
            ``detector_kwargs`` are written in their replacement form
            (``detection_mode='oneflux_mp'``, ``detection_kwargs={'mp_kwargs': ...}``).
        df_var, load_hint: as in :func:`chain_to_code`.
    """
    lines = _init_level2_lines(
        "from diive.flux.postprocessingchain import (init_flux_data, run_level2, "
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


def _constant_by_percentile(level33_kwargs: dict) -> dict | None:
    """``{percentile: threshold}`` when a constant spec is labelled exactly as
    ``run_level33_ustar_detection`` labels percentiles (``CUT_16``, ``CUT_50``,
    ...), else None (free labels such as ``CUT_NONE`` stay constant)."""
    import re

    from diive.flux.postprocessingchain.levels.level33 import ustar_scenario_labels
    thresholds = level33_kwargs.get("thresholds")
    labels = level33_kwargs.get("threshold_labels")
    if not thresholds or not labels or len(thresholds) != len(labels):
        return None
    pcts = []
    for label in labels:
        m = re.fullmatch(r"CUT_(\d+(?:\.\d+)?)", str(label))
        if m is None:
            return None
        pcts.append(float(m.group(1)))
    if ustar_scenario_labels("CUT", pcts) != list(labels):
        return None
    return {(int(p) if p.is_integer() else p): v for p, v in zip(pcts, thresholds, strict=True)}


def _level33_import_name(level33_kwargs: dict) -> str:
    """The L3.3 function symbol to import for this spec."""
    if _level33_is_detection(level33_kwargs) or _constant_by_percentile(level33_kwargs or {}):
        return "run_level33_ustar_detection"
    return "run_level33_constant_ustar"


def _level33_detection_defaults() -> dict:
    from diive.flux.postprocessingchain.levels import run_level33_ustar_detection
    return {p.name: p.default
            for p in inspect.signature(run_level33_ustar_detection).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _level33_detection_kwargs(kw: dict) -> dict:
    """Detection kwargs with the deprecated ``detector_class`` / ``detector_kwargs``
    replaced by ``detection_mode`` / ``detection_kwargs``, so a script never
    uses the deprecated arguments.

    Only the moving point detector was supported there, so it maps to
    ``detection_mode='oneflux_mp'`` with the old arguments as ``mp_kwargs``.
    """
    if "detector_class" not in kw and "detector_kwargs" not in kw:
        return kw
    kw = dict(kw)
    kw.pop("detector_class", None)
    mp_kwargs = {k: v for k, v in (kw.pop("detector_kwargs", None) or {}).items()
                 if k not in ("nee_col", "ta_col", "ustar_col", "swin_col", "verbose")}
    kw["detection_mode"] = "oneflux_mp"
    if mp_kwargs:
        det = dict(kw.get("detection_kwargs") or {})
        det.setdefault("mp_kwargs", mp_kwargs)
        kw["detection_kwargs"] = det
    return kw


def _level33_block(level33_kwargs: dict) -> list[str]:
    """The L3.3 call block (constant thresholds or auto-detection; no ``final_df``)."""
    if _level33_is_detection(level33_kwargs):
        # Strip the internal marker; render the detection call.
        kw = _level33_detection_kwargs(
            {k: v for k, v in level33_kwargs.items() if k != "_detection"})
        return ["", "data = run_level33_ustar_detection(", "    data,",
                *_kwargs_lines(kw, _level33_detection_defaults()), ")"]
    by_pct = _constant_by_percentile(level33_kwargs)
    if by_pct:
        kw = {"thresholds": by_pct,
              **{k: v for k, v in level33_kwargs.items()
                 if k not in ("thresholds", "threshold_labels")}}
        return ["", "data = run_level33_ustar_detection(", "    data,",
                *_kwargs_lines(kw, _level33_detection_defaults()), ")"]
    return ["", "data = run_level33_constant_ustar(", "    data,",
            *_kwargs_lines(level33_kwargs, _level33_defaults()), ")"]


def _level41_block(level41_cfg: dict) -> list[str]:
    """The L4.1 gap-filling block: one ``run_level41_*`` per selected method.

    ``level41_cfg`` shape (the GUI flux-chain tab's ``_level41_cfg``)::

        {"methods": ["rf", "xgb", "mds"],   # subset, in this canonical order
         "features": ["TA_...", "SW_IN_...", "VPD_..."],   # rf / xgb predictors
         "reduce_features": True,           # optional, rf / xgb
         "rf_kwargs": {...}, "xgb_kwargs": {...},   # optional model kwargs
         "mds": {"swin": "SW_IN_...", "ta": "TA_...", "vpd": "VPD_...",
                 # optional: "ta_tol", "vpd_tol", "mode", and the driver QC flag
                 # columns "swin_qc", "ta_qc", "vpd_qc" (omitted when None)
                 },
         "ustar_reference": True}           # optional: REF after MDS, then rf / xgb

    RF and XGBoost share one ``make_level41_engineer`` instance (feature
    engineering runs once and is reused across methods and USTAR scenarios).
    """
    methods = level41_cfg.get("methods", [])
    features = level41_cfg.get("features") or []
    reduce = level41_cfg.get("reduce_features")
    lines: list[str] = []
    ml: list[str] = []
    if any(m in methods for m in ("rf", "xgb")):
        ml += ["", "engineer = make_level41_engineer(", "    data,",
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
        ml += _ml_call("run_level41_rf", level41_cfg.get("rf_kwargs", {}))
    if "xgb" in methods:
        ml += _ml_call("run_level41_xgb", level41_cfg.get("xgb_kwargs", {}))
    with_ref = "mds" in methods and level41_cfg.get("ustar_reference")
    if not with_ref:
        lines += ml
    if "mds" in methods:
        mds = level41_cfg.get("mds") or {}
        body = ["", "data = run_level41_mds(", "    data,",
                f"    swin={mds.get('swin')!r},", f"    ta={mds.get('ta')!r},",
                f"    vpd={mds.get('vpd')!r},"]
        for k in ("ta_tol", "vpd_tol", "mode"):
            if k in mds:
                body.append(f"    {k}={mds[k]!r},")
        for k in ("swin_qc", "ta_qc", "vpd_qc"):
            if mds.get(k) is not None:
                body.append(f"    {k}={mds[k]!r},")
        body.append(")")
        lines += body
        if with_ref:
            # REF before RF / XGBoost: they leave out the 40 REF versions but
            # include the scenario the REF maps to, once it is selected.
            lines += ["", "data = run_level41_ustar_reference(data)"]
            lines += ml
    return lines


#: L4.2 partitioning variant token -> (function suffix, kwarg keys it accepts).
#: Kwarg keys are rendered from level42_cfg in this order when present.
_LEVEL42_VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "nt_of": ("nighttime_oneflux", ("ta", "sw_in", "ta_f", "gapfill_method")),
    "nt_rp": ("nighttime_reddyproc", ("ta", "sw_in", "ta_f", "gapfill_method")),
    "dt_rp": ("daytime_reddyproc", ("ta_f", "vpd_f", "sw_in_f", "nee_sd", "vpd_in_kpa")),
    "dt_of": ("daytime_oneflux", ("ta", "sw_in", "ta_f", "sw_in_f", "vpd_f", "vpd_in_kpa",
                                  "gapfill_method")),
}


def _level42_selected(level42_cfg: dict) -> str | None:
    """The variant token picked by ``method`` / ``mode`` in ``level42_cfg``, or None."""
    method = level42_cfg.get("method")
    if method is None:
        return None
    from diive.flux.postprocessingchain.levels.level42 import PARTITION_VARIANTS
    mode = level42_cfg.get("mode", "oneflux")
    if (method, mode) not in PARTITION_VARIANTS:
        raise ValueError(f"level42_cfg: unknown method/mode {method!r}/{mode!r}.")
    return PARTITION_VARIANTS[(method, mode)]


def _level42_fn_names(level42_cfg: dict) -> list[str]:
    """The ``run_level42_*`` symbols to import for the selected variants."""
    selected = _level42_selected(level42_cfg)
    variants = level42_cfg.get("variants", [])
    names = ["run_level42_partitioning"] if selected else []
    names += [f"run_level42_{_LEVEL42_VARIANTS[v][0]}"
              for v in ("nt_of", "nt_rp", "dt_rp", "dt_of")
              if v in variants and v != selected]
    if level42_cfg.get("ustar_reference") and names:
        names.append("run_level42_ustar_reference")
    return names


def _level42_defaults(fn_name: str) -> dict:
    """Default values of the parameters of the ``run_level42_*`` function ``fn_name``."""
    from diive.flux.postprocessingchain.levels import level42
    return {p.name: p.default
            for p in inspect.signature(getattr(level42, fn_name)).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _level42_kwarg_lines(level42_cfg: dict, keys: tuple[str, ...], fn_name: str) -> list[str]:
    """One line per kwarg in ``keys`` that differs from the default of ``fn_name``.

    None is written when the default is not None, e.g. ``gapfill_method=None``
    (daytime ONEFlux without gap-filled NEE). A parameter without a default is
    left out when it is None.
    """
    defaults = _level42_defaults(fn_name)
    lines = []
    for key in keys:
        if key not in level42_cfg:
            continue
        value = level42_cfg[key]
        if key in defaults:
            if value == defaults[key]:
                continue
        elif value is None:
            continue
        lines.append(f"    {key}={value!r},")
    return lines


def _level42_block(level42_cfg: dict) -> list[str]:
    """The L4.2 partitioning block: one ``run_level42_*`` per selected variant.

    ``level42_cfg`` shape::

        {"method": "nighttime",            # or "daytime", renders run_level42_partitioning
         "mode": "oneflux",                # or "reddyproc" (default 'oneflux')
         "variants": ["nt_of", "nt_rp", "dt_rp", "dt_of"],  # subset, canonical order
         "gapfill_method": "mds",          # nighttime variants and daytime ONEFlux (default 'mds')
         "ta": "...", "sw_in": "...",      # measured meteo
         "ta_f": "...", "sw_in_f": "...", "vpd_f": "...",   # gap-filled meteo
         "nee_sd": "...",                  # optional, daytime REddyProc
         "vpd_in_kpa": True,               # optional, daytime variants
         "ustar_reference": True}          # optional: GPP / RECO REF per variant

    With ``method`` set, the block starts with one ``run_level42_partitioning``
    call for that method and mode, with only the kwargs that variant uses.
    ``variants`` is then optional. Partitioning is additive across variants —
    each further selected variant emits its own ``run_level42_*`` call, just
    like the L4.1 gap-filling methods. A variant selected both ways is
    rendered once. With ``ustar_reference`` each rendered variant is followed
    by a ``run_level42_ustar_reference`` call (the GPP and RECO REF, after
    ``run_level41_ustar_reference`` at L4.1).
    """
    selected = _level42_selected(level42_cfg)
    variants = level42_cfg.get("variants", [])
    lines: list[str] = []
    if selected:
        body = ["", "data = run_level42_partitioning(", "    data,",
                f"    method={level42_cfg['method']!r},",
                f"    mode={level42_cfg.get('mode', 'oneflux')!r},"]
        body += _level42_kwarg_lines(level42_cfg, _LEVEL42_VARIANTS[selected][1],
                                     "run_level42_partitioning")
        body.append(")")
        lines += body
    for variant in ("nt_of", "nt_rp", "dt_rp", "dt_of"):
        if variant not in variants or variant == selected:
            continue
        suffix, keys = _LEVEL42_VARIANTS[variant]
        body = ["", f"data = run_level42_{suffix}(", "    data,"]
        body += _level42_kwarg_lines(level42_cfg, keys, f"run_level42_{suffix}")
        body.append(")")
        lines += body
    if level42_cfg.get("ustar_reference"):
        from diive.flux.postprocessingchain.levels.level42 import PARTITION_VARIANTS
        method_mode = {v: mm for mm, v in PARTITION_VARIANTS.items()}
        rendered = ([selected] if selected else []) + [
            v for v in ("nt_of", "nt_rp", "dt_rp", "dt_of") if v in variants and v != selected]
        for variant in rendered:
            method, mode = method_mode[variant]
            body = ["", "data = run_level42_ustar_reference(", "    data,",
                    f"    method={method!r},", f"    mode={mode!r},"]
            body += _level42_kwarg_lines(level42_cfg, _LEVEL42_VARIANTS[variant][1],
                                         "run_level42_ustar_reference")
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
    ``run_level42_*`` call. With ``level42_cfg['method']`` (and ``'mode'``)
    set, that variant is rendered as one ``run_level42_partitioning`` call.
    The nighttime variants additionally consume an L4.1 gap-filled NEE
    (``level42_cfg['gapfill_method']``).

    Args:
        init_kwargs: kwargs for ``init_flux_data`` (without ``df``).
        level2_settings: ``{test_name: settings_dict}`` for ``run_level2``.
        level31_kwargs: kwargs for ``run_level31`` (storage correction).
        level32_steps: ordered ``[{"method": str, "kwargs": dict}, ...]``.
        level33_kwargs: L3.3 spec, as in :func:`level33_to_code`.
        level41_cfg: gap-filling selection — see :func:`_level41_block`.
        level42_cfg: partitioning selection — see :func:`_level42_block`.
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
    if "mds" in methods and level41_cfg.get("ustar_reference"):
        names.append("run_level41_ustar_reference")
    names += _level42_fn_names(level42_cfg)
    imports = ("from diive.flux.postprocessingchain import (\n    "
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
        level33_kwargs: L3.3 spec, as in :func:`level33_to_code`.
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
    if "mds" in methods and level41_cfg.get("ustar_reference"):
        names.append("run_level41_ustar_reference")
    imports = ("from diive.flux.postprocessingchain import (\n    "
               + ", ".join(names) + ")")
    lines = _init_level2_lines(imports, init_kwargs, level2_settings, df_var, load_hint)
    lines += _level31_block(level31_kwargs)
    lines += _level32_block(level32_steps)
    lines += _level33_block(level33_kwargs)
    lines += _level41_block(level41_cfg)
    lines += ["final_df = data.fpc_df"]
    return "\n".join(lines) + "\n"
