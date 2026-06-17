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


def xgboost_gapfill_to_code(target: str, features: list[str], kwargs: dict, *,
                            reduce: bool = False,
                            shap_threshold_factor: float = 0.5,
                            df_var: str = "df",
                            load_hint: str | None = None) -> str:
    """Render an ``XGBoostTS(...).run()`` gap-filling snippet as a string.

    Mirrors what the GUI tab runs: an ``XGBoostTS`` trained on the selected
    feature columns directly (no feature engineering), optionally reducing
    features by SHAP importance first.

    Args:
        target: the column to gap-fill.
        features: predictor columns fed to the model.
        kwargs: keyword args for ``XGBoostTS`` (without ``input_df``/``target_col``).
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
              "model = dv.gapfilling.XGBoostTS(",
              f"    input_df={df_var}[[target] + features],",
              "    target_col=target,"]
    for key, value in kwargs.items():
        lines.append(f"    {key}={value!r},")
    lines.append(")")
    if reduce:
        lines.append(f"model.reduce_features(shap_threshold_factor={shap_threshold_factor!r})")
    lines += ["model.run()",
              "",
              "gapfilled = model.get_gapfilled_target()  # observed + filled (*_gfXG)",
              "flag = model.get_flag()  # 0 = observed, 1 = gap-filled, 2 = fallback"]
    return "\n".join(lines) + "\n"
