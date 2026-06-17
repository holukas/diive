"""
RERUN CASCADE: SHARED STATE-RESET LOGIC FOR LEVEL RE-RUNS
==========================================================

When a level is re-run on a ``FluxLevelData`` that has already passed through
that level, two things have to happen:

1. Columns the previous run added to ``fpc_df`` must be dropped — otherwise
   ``pd.concat([axis=1])`` produces duplicate column labels, downstream lookups
   become ambiguous, and ``FlagQCF`` consumes stale flags.
2. Any state on ``LevelResults`` belonging to the re-run level (or to a
   downstream level whose inputs just became stale) must be cleared.

L2 / L3.1 / L3.2 / L3.3 use a **cascade**: re-running level N invalidates N
and every level after N in the chain order, because those later levels' state
was computed against the now-stale upstream inputs. L4.1 (gap-filling) and
L4.2 (NEE partitioning) are **additive** by design (each method holds an
independent dict keyed by USTAR scenario), so each ``run_level41_*`` /
``run_level42_*`` cleans up only its own method's columns and does not cascade.
A cascade from any earlier (cascade-aware) level still clears both additive
levels, because their state was computed against the now-stale upstream.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import dataclasses

from diive.flux.fluxprocessingchain.container import FluxLevelData

# Ordering of cascade-aware levels. The additive levels (L4.1 gap-filling,
# L4.2 partitioning) are NOT in this list because their re-run semantics are
# per-method and additive across methods.
_LEVEL_ORDER = ['L2', 'L3.1', 'L3.2', 'L3.3']

# Additive late-level idstrs (not cascade-ordered). A cascade from any
# cascade-aware level clears every one of these.
_ADDITIVE_LEVEL_IDS: tuple[str, ...] = ('L4.1', 'L4.2')

# Fields on LevelResults that each cascade-aware level owns. Re-running the
# level clears these along with any downstream level's fields.
#
# ``filteredseries_hq`` is deliberately listed under both L2 and L3.1: L2
# initialises it (the QCF=0 mask on the raw flux) and L3.1 overwrites it
# (same mask re-applied to the storage-corrected flux). Listing it under
# both keys means a cascade from either level correctly nulls it before
# the new run repopulates it. Resetting a single ``None`` field twice is
# idempotent; the duplication is intentional, not a bug.
_LEVEL_FIELDS: dict[str, tuple[str, ...]] = {
    'L2': (
        'level2', 'level2_qcf', 'filteredseries_level2_qcf',
        'filteredseries_hq',
    ),
    'L3.1': (
        'level31', 'level31_qcf', 'flux_corrected_col',
        'filteredseries_level31_qcf', 'filteredseries_hq',
    ),
    'L3.2': (
        'level32', 'level32_qcf', 'filteredseries_level32_qcf',
    ),
    'L3.3': (
        'level33', 'level33_qcf', 'filteredseries_level33_qcf',
        'filteredseries_level33_hq', 'ustar_detection',
    ),
}

# Additive-level (L4.1 + L4.2) LevelResults fields — always reset by a cascade
# from any earlier level.
_ADDITIVE_FIELDS: tuple[str, ...] = (
    'level41_mds', 'level41_rf', 'level41_xgb',
    'level42_nt_of', 'level42_nt_rp', 'level42_dt_rp', 'level42_dt_of',
)

# Additive-level tracking sub-keys in added_columns.
_ADDITIVE_TRACKED_KEYS: tuple[str, ...] = (
    'L4.1_mds', 'L4.1_rf', 'L4.1_xgb',
    'L4.2_nt_of', 'L4.2_nt_rp', 'L4.2_dt_rp', 'L4.2_dt_of',
)


def _field_default(value):
    """Return the appropriate default for a LevelResults field given its current value."""
    if isinstance(value, dict):
        return {}
    return None


def cascade_reset(data: FluxLevelData, from_idstr: str) -> FluxLevelData:
    """Drop state owned by ``from_idstr`` and every downstream level.

    Called by each cascade-aware ``run_level*`` at entry, when its idstr is
    already present in ``data.level_ids``. The returned container has:

    - ``fpc_df`` with tracked columns from the affected levels dropped.
    - ``levels`` with the affected LevelResults fields reset to their defaults
      (None for scalar fields, {} for dict fields).
    - ``level_ids`` with the affected level idstrs removed.
    - ``added_columns`` with the affected entries removed.
    - ``filteredseries`` set to ``None`` (it always belongs to "the most
      recently completed level", which has just been invalidated).

    Args:
        data: Current FluxLevelData.
        from_idstr: One of ``'L2'``, ``'L3.1'``, ``'L3.2'``, ``'L3.3'``.

    Returns:
        New ``FluxLevelData`` with the cascaded state cleared.
    """
    if from_idstr not in _LEVEL_ORDER:
        raise ValueError(
            f"cascade_reset() called with unknown level {from_idstr!r}. "
            f"Expected one of {_LEVEL_ORDER}."
        )

    from_idx = _LEVEL_ORDER.index(from_idstr)
    cleared_levels = list(_LEVEL_ORDER[from_idx:])  # this level + downstream
    # A cascade from any cascade-aware level also invalidates the additive
    # levels (L4.1 gap-filling, L4.2 partitioning).
    affected_tracking_keys = set(cleared_levels) | set(_ADDITIVE_TRACKED_KEYS)

    # 1. Drop tracked columns from fpc_df. Keep the order stable by filtering
    # the current columns list.
    cols_to_drop: set[str] = set()
    new_added: dict[str, list[str]] = {}
    for key, cols in data.added_columns.items():
        if key in affected_tracking_keys:
            cols_to_drop.update(cols)
        else:
            new_added[key] = list(cols)
    # Always copy: the new FluxLevelData must not share fpc_df with the
    # input. Downstream levels concat into the working frame, and an
    # alias would let one container's writes leak into another.
    keep = [c for c in data.fpc_df.columns if c not in cols_to_drop]
    fpc_df = data.fpc_df[keep].copy() if cols_to_drop else data.fpc_df.copy()

    # 2. Reset LevelResults fields owned by the cleared levels (plus the
    # additive levels L4.1 / L4.2).
    resets: dict = {}
    for lvl in cleared_levels:
        for fname in _LEVEL_FIELDS.get(lvl, ()):
            resets[fname] = _field_default(getattr(data.levels, fname))
    for fname in _ADDITIVE_FIELDS:
        resets[fname] = _field_default(getattr(data.levels, fname))
    new_levels = dataclasses.replace(data.levels, **resets)

    # 3. Trim level_ids: drop the cleared cascade-aware levels and the
    # additive levels (L4.1 / L4.2).
    new_level_ids = [
        lid for lid in data.level_ids
        if lid not in cleared_levels and lid not in _ADDITIVE_LEVEL_IDS
    ]

    # 4. Restore filteredseries to whatever the most-recently-surviving level
    # had set, so the re-running level sees a consistent input. (The
    # filteredseries from the now-cleared levels is gone with the LevelResults
    # fields we just reset.)
    new_filteredseries = _surviving_filteredseries(new_levels, new_level_ids)

    return dataclasses.replace(
        data,
        fpc_df=fpc_df,
        filteredseries=new_filteredseries,
        levels=new_levels,
        level_ids=new_level_ids,
        added_columns=new_added,
    )


def _surviving_filteredseries(levels, level_ids):
    """Pick the right ``filteredseries`` for the most-recently-completed level."""
    if 'L3.3' in level_ids:
        # L3.3 deliberately sets filteredseries=None (multi-scenario).
        return None
    if 'L3.2' in level_ids:
        return levels.filteredseries_level32_qcf.copy() if levels.filteredseries_level32_qcf is not None else None
    if 'L3.1' in level_ids:
        return levels.filteredseries_level31_qcf.copy() if levels.filteredseries_level31_qcf is not None else None
    if 'L2' in level_ids:
        return levels.filteredseries_level2_qcf.copy() if levels.filteredseries_level2_qcf is not None else None
    return None


def record_added_columns(
        data: FluxLevelData,
        idstr: str,
        pre_columns: list[str],
) -> dict[str, list[str]]:
    """Compute the new ``added_columns`` mapping after a level has run.

    ``pre_columns`` is the column list snapshot taken at level entry (after
    any cascade reset). Any column now in ``data.fpc_df.columns`` that wasn't
    in ``pre_columns`` was added by this level invocation.

    Used by every level callable — both cascade-aware levels and the L4.1
    per-method runs. The returned dict replaces ``data.added_columns`` in the
    final ``replace(data, ...)`` call.
    """
    pre_set = set(pre_columns)
    new_cols = [c for c in data.fpc_df.columns if c not in pre_set]
    out = {k: list(v) for k, v in data.added_columns.items()}
    out[idstr] = new_cols
    return out


def drop_columns_for_key(
        data: FluxLevelData,
        key: str,
) -> FluxLevelData:
    """Drop columns previously recorded under ``added_columns[key]``.

    Used by ``run_level41_*`` to clean up the previous run of *its own method*
    (e.g. ``'L4.1_mds'``) without disturbing the other gap-filling methods'
    output. The matching ``added_columns`` entry is also removed.
    """
    cols = data.added_columns.get(key, [])
    if not cols:
        return data
    keep = [c for c in data.fpc_df.columns if c not in set(cols)]
    fpc_df = data.fpc_df[keep].copy()
    new_added = {k: list(v) for k, v in data.added_columns.items() if k != key}
    return dataclasses.replace(data, fpc_df=fpc_df, added_columns=new_added)
