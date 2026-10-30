"""
SHARED LEVEL HELPERS: COMMON LOGIC FOR THE LATE-STAGE LEVEL CALLABLES
=====================================================================

Small helpers shared by the *additive* late levels — Level-4.1 (gap-filling)
and Level-4.2 (NEE partitioning). Both levels run one independent model per
USTAR scenario, merge external result frames back into ``fpc_df``, accumulate
their per-scenario instances in scenario-keyed dicts on ``LevelResults``, and
warn on the same suspicious re-run patterns. Factoring the shared pieces here
keeps the two levels implemented the same way rather than drifting apart.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import warnings

import pandas as pd

from diive.flux.postprocessingchain.container import FluxLevelData


def assert_aligned_index(left: pd.DataFrame, *others, context: str) -> None:
    """Raise if any of ``others`` has an index that does not match ``left``.

    ``pd.concat([..., axis=1])`` aligns by label and silently NaN-pads
    divergent indexes, so an underlying class that reindexes internally
    (e.g. dropping rows with all-NaN drivers) would silently poison
    ``fpc_df`` with phantom rows. Mirror the index-equality guard from
    :func:`finalize_level` at every concat site that brings external
    results back into the chain's working dataframe.
    """
    for i, other in enumerate(others, start=1):
        if other is None:
            continue
        idx = other.index if hasattr(other, 'index') else None
        if idx is None:
            continue
        if not idx.equals(left.index):
            only_in_other = idx.difference(left.index)
            only_in_left = left.index.difference(idx)
            raise RuntimeError(
                f"{context}: operand #{i} index does not match the working "
                f"dataframe index. {len(only_in_other)} timestamp(s) only in "
                f"the operand, {len(only_in_left)} only in fpc_df. "
                f"The two must align exactly before concat."
            )


def require_level33(data: FluxLevelData) -> dict[str, pd.Series]:
    """Return the per-USTAR-scenario QCF-filtered flux dict, or raise.

    Both L4.1 and L4.2 consume the L3.3 output (one measured-flux series per
    USTAR scenario), so the same ordering guard serves both.
    """
    if not data.levels.filteredseries_level33_qcf:
        raise RuntimeError(
            "run_level33_constant_ustar() must be called before any "
            "run_level41_* or run_level42_* function."
        )
    return data.levels.filteredseries_level33_qcf


def select_scenarios(data: FluxLevelData, scenarios=None) -> dict[str, pd.Series]:
    """The L3.3 scenarios a Level 4.1 machine learning or Level 4.2 step runs on.

    ``scenarios=None`` takes all of them, except the 40 versions that
    ``reference=True`` adds only to select the ONEFlux REF. Those are gap-filled
    by MDS for the selection and are left out here, so the slow steps run on
    the requested percentiles (e.g. CUT_16, CUT_50, CUT_84) plus the scenario
    the REF maps to, if ``run_level41_ustar_reference`` ran before. A list of
    labels picks the scenarios explicitly, in that order.

    Raises:
        RuntimeError: If Level 3.3 has not run.
        KeyError: If a requested label is not an L3.3 scenario.
    """
    filtered = require_level33(data)
    if scenarios is not None:
        if isinstance(scenarios, str):
            scenarios = [scenarios]
        unknown = [s for s in scenarios if s not in filtered]
        if unknown:
            raise KeyError(f"Unknown USTAR scenario(s) {unknown}. "
                           f"Level 3.3 has: {list(filtered)}.")
        return {s: filtered[s] for s in dict.fromkeys(scenarios)}
    helpers = reference_helpers(data)
    return {s: v for s, v in filtered.items() if s not in helpers}


def reference_helpers(data: FluxLevelData) -> set[str]:
    """Labels of the REF versions that serve only the REF selection.

    The 40 versions ``reference=True`` adds, without the requested percentiles
    and without the scenario the REF maps to (once ``run_level41_ustar_reference``
    ran). Empty without ``reference=True``.
    """
    rec = getattr(data.levels, 'ustar_scenarios', None)
    if rec is None or not rec.reference_labels:
        return set()
    helpers = set(rec.reference_labels) - set(rec.main_labels)
    ref = getattr(data.levels, 'ustar_reference', None)
    if ref is not None:
        helpers.discard(ref.scenario)
    return helpers


def scenario_name(data: FluxLevelData, scenario: str) -> str:
    """Name of a USTAR scenario in reports and plots: the scenario the REF maps
    to is named as REF, e.g. ``CUT_REF (CUT_4875)``."""
    ref = data.levels.ustar_reference
    if ref is not None and scenario == ref.scenario:
        return f"{ref.label} ({scenario})"
    return scenario


def middle_scenario(data: FluxLevelData, labels) -> str:
    """The scenario of ``labels`` nearest the 50th percentile, e.g. ``CUT_50``.

    Taken by percentile, not by list position: of ``CUT_16``, ``CUT_50``,
    ``CUT_84`` and a REF scenario the middle position is ``CUT_84``. Without
    known percentiles (free labels of ``run_level33_constant_ustar``) the middle
    position is used.
    """
    labels = list(labels)
    rec = getattr(data.levels, 'ustar_scenarios', None)
    percentiles = rec.percentiles if rec is not None else {}
    known = [s for s in labels if s in percentiles]
    if known:
        return min(known, key=lambda s: abs(percentiles[s] - 50))
    return labels[len(labels) // 2]


def append_level_id(level_ids: list[str], idstr: str) -> list[str]:
    """Append ``idstr`` to ``level_ids`` once (idempotent)."""
    new = list(level_ids)
    if idstr not in new:
        new.append(idstr)
    return new


def warn_scenario_overwrite(existing: dict, new: dict, *,
                            fn_label: str, results_attr: str) -> None:
    """Warn on suspicious additive-level re-runs (overlap or unexpected-additive).

    Each ``run_level41_*`` / ``run_level42_*`` call stores its per-scenario
    instance in ``data.levels.<results_attr>[scen]``. Two patterns are worth
    warning about:

    - **Overlap** (``existing & new`` non-empty): re-running with overlapping
      scenario labels silently drops the prior instance via the standard
      ``{**existing, **new}`` dict-merge pattern; earlier runs become
      unrecoverable. The most common cause is a parameter sweep with the same
      scenario labels.
    - **Unexpected-additive** (``existing`` non-empty, no overlap): the user is
      *growing* the results dict with completely new scenario labels on top of
      existing ones. This is almost always a sign of confusion — a re-run of
      L3.3 with different USTAR labels should *cascade-clear* the additive
      levels first, leaving ``existing`` empty. If you see this warning, the
      level holds scenarios from two different L3.3 states simultaneously and
      the chain bookkeeping is inconsistent.

    Both cases skip the warning when ``existing`` is empty (a first run is not
    a re-run).

    Args:
        existing: The scenario dict already stored on ``LevelResults``.
        new: The scenario dict produced by the current call.
        fn_label: The public function name for the message (e.g.
            ``'run_level41_mds'`` / ``'run_level42_nighttime_oneflux'``).
        results_attr: The ``LevelResults`` attribute name (e.g.
            ``'level41_mds'`` / ``'level42_nt_of'``) for the recovery hint.
    """
    if not existing:
        return  # first run; nothing to warn about
    overlap = sorted(set(existing).intersection(new))
    only_existing = sorted(set(existing) - set(new))
    if overlap:
        warnings.warn(
            f"{fn_label}() is replacing previously-stored "
            f"scenario result(s) for: {overlap}. The earlier instance(s) for "
            f"these scenarios will be lost. To keep both runs, give the "
            f"USTAR scenarios distinct labels (e.g. add a suffix at L3.3) or "
            f"copy data.levels.{results_attr} before re-running.",
            UserWarning,
            stacklevel=3,
        )
    elif only_existing:
        # No overlap, but existing is non-empty — user is adding new
        # scenarios on top of old ones. Normally an L3.3 re-run cascades the
        # additive levels clean before this can happen; reaching here means
        # the chain bookkeeping is inconsistent.
        warnings.warn(
            f"{fn_label}() is adding new scenario(s) "
            f"({sorted(set(new) - set(existing))}) on top of existing "
            f"scenarios ({only_existing}) — the results now mix "
            f"outputs from two different L3.3 states. Normally a re-run of "
            f"L3.3 with different labels cascade-clears the additive levels; "
            f"if you see this warning the cascade did not fire. Inspect "
            f"data.levels.{results_attr} and consider rebuilding "
            f"the chain from the relevant earlier level.",
            UserWarning,
            stacklevel=3,
        )
