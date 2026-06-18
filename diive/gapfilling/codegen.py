"""
GAPFILLING.CODEGEN: RENDER GAP-FILLING CHOICES AS A RUNNABLE SCRIPT
==================================================================

Turn the choices a caller makes (e.g. in the GUI's XGBoost gap-filling tab) into
a self-contained, runnable diive snippet — so a point-and-click run stays
reproducible.

This belongs in the library (not the GUI): it encodes the exact API call shape
and must stay correct as that API evolves; the GUI only calls it.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def ml_gapfill_to_code(class_name: str, gapfilled_suffix: str,
                       target: str, features: list[str], kwargs: dict, *,
                       reduce: bool = False,
                       shap_threshold_factor: float = 0.5,
                       df_var: str = "df",
                       load_hint: str | None = None) -> str:
    """Render an ``<class_name>(...).run()`` gap-filling snippet as a string.

    Mirrors what the GUI tabs run for the ML gap-fillers (``XGBoostTS`` /
    ``RandomForestTS`` / ...): the model trained on the selected feature columns
    directly (no feature engineering), optionally reducing features by SHAP
    importance first. The only per-method differences are the class name and the
    gap-filled column suffix, so all ML methods share this one renderer.

    Args:
        class_name: the gap-filling class (e.g. ``"XGBoostTS"``), under ``dv.gapfilling``.
        gapfilled_suffix: the gap-filled column suffix, for the comment (e.g. ``"_gfXG"``).
        target: the column to gap-fill.
        features: predictor columns fed to the model.
        kwargs: keyword args for the model (without ``input_df``/``target_col``).
        reduce: emit a ``reduce_features(...)`` call before the run.
        shap_threshold_factor: the reduction threshold (only when ``reduce``).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += [f"target = {target!r}",
              f"features = {list(features)!r}",
              "",
              f"model = dv.gapfilling.{class_name}(",
              f"    input_df={df_var}[[target] + features],",
              "    target_col=target,"]
    for key, value in kwargs.items():
        lines.append(f"    {key}={value!r},")
    lines.append(")")
    if reduce:
        lines.append(f"model.reduce_features(shap_threshold_factor={shap_threshold_factor!r})")
    lines += ["model.run()",
              "",
              f"gapfilled = model.get_gapfilled_target()  # observed + filled (*{gapfilled_suffix})",
              "flag = model.get_flag()  # 0 = observed, 1 = gap-filled, 2 = fallback"]
    return "\n".join(lines) + "\n"


def xgboost_gapfill_to_code(target: str, features: list[str], kwargs: dict, *,
                            reduce: bool = False,
                            shap_threshold_factor: float = 0.5,
                            df_var: str = "df",
                            load_hint: str | None = None) -> str:
    """Render an ``XGBoostTS(...).run()`` snippet (thin wrapper over
    :func:`ml_gapfill_to_code`)."""
    return ml_gapfill_to_code(
        "XGBoostTS", "_gfXG", target, features, kwargs,
        reduce=reduce, shap_threshold_factor=shap_threshold_factor,
        df_var=df_var, load_hint=load_hint)


def randomforest_gapfill_to_code(target: str, features: list[str], kwargs: dict, *,
                                 reduce: bool = False,
                                 shap_threshold_factor: float = 0.5,
                                 df_var: str = "df",
                                 load_hint: str | None = None) -> str:
    """Render a ``RandomForestTS(...).run()`` snippet (thin wrapper over
    :func:`ml_gapfill_to_code`)."""
    return ml_gapfill_to_code(
        "RandomForestTS", "_gfRF", target, features, kwargs,
        reduce=reduce, shap_threshold_factor=shap_threshold_factor,
        df_var=df_var, load_hint=load_hint)


def longterm_ml_gapfill_to_code(class_name: str, gapfilled_suffix: str,
                                target: str, features: list[str], kwargs: dict, *,
                                reduce: bool = False,
                                shap_threshold_factor: float = 0.5,
                                df_var: str = "df",
                                load_hint: str | None = None) -> str:
    """Render a ``<class_name>(...).run()`` long-term gap-filling snippet.

    The long-term gap-fillers (``LongTermGapFillingXGBoostTS`` /
    ``LongTermGapFillingRandomForestTS``) build a separate model per calendar
    year from that year plus its two closest neighbouring years, then stitch the
    per-year gap-filled series back together. The constructor takes the same
    target/features/hyperparameters as the single-model classes; the optional
    per-year SHAP feature reduction is requested via ``run(reduce_features=...)``.

    Args:
        class_name: the long-term class (e.g. ``"LongTermGapFillingXGBoostTS"``).
        gapfilled_suffix: the gap-filled column suffix, for the comment.
        target: the column to gap-fill.
        features: predictor columns fed to the per-year models.
        kwargs: keyword args for the model (without ``input_df``/``target_col``).
        reduce: request per-year SHAP feature reduction in ``run()``.
        shap_threshold_factor: the reduction threshold (only when ``reduce``).
        df_var: variable name used for the input DataFrame.
        load_hint: if given, prepend ``df = <load_hint>`` so the snippet runs as-is.

    Returns:
        A runnable Python snippet as a string.
    """
    lines = ["import diive as dv", ""]
    if load_hint is not None:
        lines += [f"{df_var} = {load_hint}", ""]
    lines += [f"target = {target!r}",
              f"features = {list(features)!r}",
              "",
              f"model = dv.gapfilling.{class_name}(",
              f"    input_df={df_var}[[target] + features],",
              "    target_col=target,"]
    for key, value in kwargs.items():
        lines.append(f"    {key}={value!r},")
    lines.append(")")
    if reduce:
        lines.append(f"model.run(reduce_features=True, "
                     f"shap_threshold_factor={shap_threshold_factor!r})")
    else:
        lines.append("model.run()")
    lines += ["",
              f"gapfilled = model.get_gapfilled_target()  # observed + filled (*{gapfilled_suffix})",
              "flag = model.get_flag()  # 0 = observed, 1 = gap-filled, 2 = fallback",
              "scores_per_year = model.scores_traintest_  # {year: held-out test scores}"]
    return "\n".join(lines) + "\n"


def longterm_xgboost_gapfill_to_code(target: str, features: list[str], kwargs: dict, *,
                                     reduce: bool = False,
                                     shap_threshold_factor: float = 0.5,
                                     df_var: str = "df",
                                     load_hint: str | None = None) -> str:
    """Render a ``LongTermGapFillingXGBoostTS(...).run()`` snippet."""
    return longterm_ml_gapfill_to_code(
        "LongTermGapFillingXGBoostTS", "_gfXG", target, features, kwargs,
        reduce=reduce, shap_threshold_factor=shap_threshold_factor,
        df_var=df_var, load_hint=load_hint)


def longterm_randomforest_gapfill_to_code(target: str, features: list[str], kwargs: dict, *,
                                          reduce: bool = False,
                                          shap_threshold_factor: float = 0.5,
                                          df_var: str = "df",
                                          load_hint: str | None = None) -> str:
    """Render a ``LongTermGapFillingRandomForestTS(...).run()`` snippet."""
    return longterm_ml_gapfill_to_code(
        "LongTermGapFillingRandomForestTS", "_gfRF", target, features, kwargs,
        reduce=reduce, shap_threshold_factor=shap_threshold_factor,
        df_var=df_var, load_hint=load_hint)
