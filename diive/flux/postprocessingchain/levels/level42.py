"""
LEVEL 4.2: NEE PARTITIONING (NEE -> GPP + RECO)
================================================

Composable callables for the four faithful NEE partitioning ports: nighttime
ONEFlux / nighttime REddyProc (Reichstein et al. 2005) and daytime REddyProc /
daytime ONEFlux (Lasslop et al. 2010). Each runs one partitioning instance per
USTAR scenario produced at Level-3.3 and merges its result columns into
``fpc_df`` with the scenario label appended (so all scenarios coexist).
Daytime ONEFlux runs its scenarios together, because they share the days
removed after a failed fit, as in ONEFlux. ``run_level42_partitioning``
selects one of the four with ``method`` and ``mode`` (ONEFlux by default) and
calls the matching per-variant function.

Mirrors Level-4.1 gap-filling structurally — same per-scenario loop, same
additive-across-variants semantics, same re-run cleanup and scenario-overwrite
warnings (shared via ``levels/_shared.py``). The difference is what each level
*controls*: L4.1 picks a gap-filling model + feature engineering, L4.2 picks a
partitioning algorithm + its measured/gap-filled meteo driver columns.

Inputs and column-name conventions
----------------------------------

The measured NEE for each USTAR scenario is taken from L3.3
(``filteredseries_level33_qcf[scen]``). Meteorological drivers are read by name
from ``data.full_df`` (as for MDS at L4.1). The site ``lat`` / ``lon`` /
``utc_offset`` come from ``data.meta``.

The **nighttime** variants additionally need a gap-filled NEE for the GPP
residual; ``gapfill_method`` selects which L4.1 output to use (default
``'mds'``). The **daytime** variants fit measured NEE only and do not require
L4.1 to have run. Daytime ONEFlux uses the L4.1 gap-filled NEE when it exists:
ONEFlux reads one gap-filled value per year (the first record), for windows
without measured nighttime NEE.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pandas as pd

from diive.core.utils.console import rule
from diive.flux.postprocessingchain.container import FluxLevelData
from diive.flux.postprocessingchain.levels._rerun import (
    drop_columns_for_key,
    record_added_columns,
)
from diive.flux.postprocessingchain.levels._shared import (
    append_level_id,
    assert_aligned_index,
    require_level33,
    warn_scenario_overwrite,
)

_LEVEL42_IDSTR = 'L4.2'
_GAPFILL_METHODS = ('mds', 'rf', 'xgb')


def _require_full_df_cols(data: FluxLevelData, cols: dict[str, str]) -> None:
    """Raise a single ``KeyError`` listing every requested column absent from full_df.

    ``cols`` maps a parameter name to the column it points at; partitioning
    drivers — like MDS drivers at L4.1 — must live in ``data.full_df`` (the
    original input), not only in ``data.fpc_df``.
    """
    missing = {param: col for param, col in cols.items()
               if col not in data.full_df.columns}
    if missing:
        pretty = ", ".join(f"{p}={c!r}" for p, c in missing.items())
        raise KeyError(
            f"Partitioning driver column(s) not found in data.full_df: {pretty}. "
            f"Driver columns must exist in the original input DataFrame "
            f"(data.full_df), not only in data.fpc_df. Use add_driver() to "
            f"register a computed driver. "
            f"Available columns: {list(data.full_df.columns)}"
        )


def _resolve_nee_f(data: FluxLevelData, gapfill_method: str,
                   scenarios: list[str]) -> dict[str, pd.Series]:
    """Return the L4.1 gap-filled NEE Series per USTAR scenario, or raise.

    The nighttime partitioning variants need a gap-filled NEE for the GPP
    residual; ``gapfill_method`` selects which L4.1 method's output to use.
    """
    if gapfill_method not in _GAPFILL_METHODS:
        raise ValueError(
            f"gapfill_method must be one of {_GAPFILL_METHODS}; "
            f"got {gapfill_method!r}."
        )
    gf = data.gapfilled_cols()
    if gapfill_method not in gf:
        raise RuntimeError(
            f"Nighttime partitioning needs a gap-filled NEE, but L4.1 method "
            f"{gapfill_method!r} has not been run. Call "
            f"run_level41_{gapfill_method}() first, or pass a "
            f"gapfill_method= that has run (available: {sorted(gf)})."
        )
    out: dict[str, pd.Series] = {}
    for scen in scenarios:
        if scen not in gf[gapfill_method]:
            raise RuntimeError(
                f"USTAR scenario {scen!r} was not gap-filled by L4.1 method "
                f"{gapfill_method!r} (available scenarios: "
                f"{sorted(gf[gapfill_method])})."
            )
        out[scen] = data.fpc_df[gf[gapfill_method][scen]]
    return out


def _available_nee_f(data: FluxLevelData, gapfill_method: str | None,
                     scenarios: list[str]) -> dict[str, pd.Series]:
    """Return the L4.1 gap-filled NEE per USTAR scenario where it exists.

    For daytime ONEFlux, which works without it: a method that has not run,
    or a scenario it did not fill, is simply left out. ``None`` uses none.
    """
    if gapfill_method is None:
        return {}
    if gapfill_method not in _GAPFILL_METHODS:
        raise ValueError(
            f"gapfill_method must be one of {_GAPFILL_METHODS} or None; "
            f"got {gapfill_method!r}."
        )
    filled = data.gapfilled_cols().get(gapfill_method, {})
    return {scen: data.fpc_df[filled[scen]] for scen in scenarios if scen in filled}


def _run_level42(
        data: FluxLevelData,
        *,
        results_attr: str,
        tracking_key: str,
        fn_label: str,
        make_instance: Callable[[str, pd.Series, pd.Series | None], object] | None = None,
        run_instances: Callable[[dict[str, pd.Series]], dict[str, object]] | None = None,
        needs_nee_f: bool,
        gapfill_method: str | None,
) -> FluxLevelData:
    """Internal: shared per-scenario partitioning loop for the four L4.2 variants.

    ``make_instance(scen, nee_meas, nee_f)`` builds the partitioning instance
    for one USTAR scenario; ``nee_f`` is the gap-filled NEE for the nighttime
    variants and ``None`` for the daytime ones. Each instance is run on its own.

    ``run_instances(nee_by_scen)`` replaces that for a variant whose scenarios
    depend on each other (daytime ONEFlux shares removed days): it gets the
    measured NEE of all scenarios and returns ``{scen: instance}``, already
    run. Pass exactly one of the two.
    """
    # Require L3.3 *before* mutating state — same ordering guard as L4.1: if we
    # dropped columns first and then raised, a caller recovering from the
    # exception would be left with a half-cleaned container.
    filteredseries_l33 = require_level33(data)
    scenarios = list(filteredseries_l33)

    # Resolve the gap-filled NEE source for the nighttime variants up front, so
    # a missing L4.1 run fails before any column is dropped.
    nee_f_by_scen: dict[str, pd.Series] = (
        _resolve_nee_f(data, gapfill_method, scenarios) if needs_nee_f else {}
    )

    # Re-run cleanup: drop this variant's previous columns from fpc_df. L4.2 is
    # additive *across variants* (each holds an independent scenario-keyed dict),
    # so this cleanup is per-variant; the other variants' columns are untouched.
    data = drop_columns_for_key(data, tracking_key)
    pre_columns = list(data.fpc_df.columns)
    fpc_df = data.fpc_df.copy()
    results: dict = {}

    if run_instances is not None:
        instances = run_instances(dict(filteredseries_l33))
    else:
        instances = {}
        for scen, nee_meas in filteredseries_l33.items():
            instance = make_instance(scen, nee_meas, nee_f_by_scen.get(scen))
            instance.run()
            instances[scen] = instance

    for scen, instance in instances.items():
        res = instance.results
        assert_aligned_index(fpc_df, res,
                             context=f"{fn_label}[{scen!r}] merge into fpc_df")
        # Each variant emits fixed result column names (RECO_NT_OF, ...); append
        # the USTAR scenario label so multiple scenarios coexist in fpc_df.
        res = res.add_suffix(f"_{scen}")
        collisions = [c for c in res.columns if c in fpc_df.columns]
        if collisions:
            raise RuntimeError(
                f"{fn_label}[{scen!r}]: partitioning output column(s) "
                f"{collisions} already exist in fpc_df. This should have been "
                f"cleaned by the L4.2 re-run logic — please open an issue."
            )
        fpc_df = pd.concat([fpc_df, res], axis=1)
        results[scen] = instance

    existing = getattr(data.levels, results_attr)
    warn_scenario_overwrite(existing, results,
                            fn_label=fn_label, results_attr=results_attr)
    new_levels = replace(data.levels, **{results_attr: {**existing, **results}})
    final = replace(
        data,
        fpc_df=fpc_df,
        levels=new_levels,
        level_ids=append_level_id(data.level_ids, _LEVEL42_IDSTR),
    )
    return replace(final, added_columns=record_added_columns(final, tracking_key, pre_columns))


def run_level42_nighttime_oneflux(
        data: FluxLevelData,
        *,
        ta: str,
        sw_in: str,
        ta_f: str,
        gapfill_method: str = 'mds',
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO with the nighttime method (ONEFlux).

    Faithful port of ONEFlux ``oneflux.partition.nighttime`` (Reichstein et al.
    2005). Runs one partitioning per USTAR scenario found at L3.3; site latitude
    is taken from ``data.meta``.

    Args:
        data: FluxLevelData after ``run_level33_*`` and an L4.1 gap-filling run
            (the nighttime method needs a gap-filled NEE for the GPP residual).
        ta: Measured air temperature column (**deg C**) in ``data.full_df``.
        sw_in: Measured shortwave incoming radiation column (**W m-2**) in
            ``data.full_df`` (used for the day/night split).
        ta_f: Gap-filled air temperature column (**deg C**) in ``data.full_df``
            (used to compute RECO at every record).
        gapfill_method: Which L4.1 gap-filled NEE to use for the GPP residual —
            ``'mds'`` (default), ``'rf'``, or ``'xgb'``. Must have run at L4.1.
        verbose: Console verbosity passed to the partitioning instance
            (0 silent, 1 warnings, 2 progress + report, 3 debug).

    Returns:
        Updated FluxLevelData; instances accessible via
        ``data.levels.level42_nt_of[ustar_scenario]`` and output column names
        via ``data.partitioned_cols()['nt_of']``.
    """
    from diive.flux.partitioning import NighttimePartitioningOneFlux

    rule("Level 4.2: NEE Partitioning (Nighttime ONEFlux, Reichstein 2005)")
    _require_full_df_cols(data, {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f})
    full_df = data.full_df
    lat = data.meta.site_lat

    def make_instance(scen, nee_meas, nee_f):
        return NighttimePartitioningOneFlux(
            nee=nee_meas, ta=full_df[ta], sw_in=full_df[sw_in],
            nee_f=nee_f, ta_f=full_df[ta_f], lat=lat, verbose=verbose,
        )

    return _run_level42(
        data,
        results_attr='level42_nt_of',
        tracking_key='L4.2_nt_of',
        fn_label='run_level42_nighttime_oneflux',
        make_instance=make_instance,
        needs_nee_f=True,
        gapfill_method=gapfill_method,
    )


def run_level42_nighttime_reddyproc(
        data: FluxLevelData,
        *,
        ta: str,
        sw_in: str,
        ta_f: str,
        gapfill_method: str = 'mds',
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO with the nighttime method (REddyProc).

    Faithful port of REddyProc's ``sMRFluxPartition`` (Reichstein et al. 2005),
    whole-record with a single E0. Runs one partitioning per USTAR scenario;
    site ``lat`` / ``lon`` / ``utc_offset`` are taken from ``data.meta`` (the
    REddyProc day/night split needs longitude and the UTC offset).

    Args:
        data: FluxLevelData after ``run_level33_*`` and an L4.1 gap-filling run.
        ta: Measured air temperature column (**deg C**) in ``data.full_df``.
        sw_in: Measured shortwave incoming radiation column (**W m-2**) in
            ``data.full_df``.
        ta_f: Gap-filled air temperature column (**deg C**) in ``data.full_df``.
        gapfill_method: Which L4.1 gap-filled NEE to use (``'mds'`` default /
            ``'rf'`` / ``'xgb'``). Must have run at L4.1.
        verbose: Console verbosity passed to the partitioning instance.

    Returns:
        Updated FluxLevelData; instances via
        ``data.levels.level42_nt_rp[ustar_scenario]``, columns via
        ``data.partitioned_cols()['nt_rp']``.
    """
    from diive.flux.partitioning import NighttimePartitioningReddyProc

    rule("Level 4.2: NEE Partitioning (Nighttime REddyProc, Reichstein 2005)")
    _require_full_df_cols(data, {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f})
    full_df = data.full_df
    meta = data.meta

    def make_instance(scen, nee_meas, nee_f):
        return NighttimePartitioningReddyProc(
            nee=nee_meas, ta=full_df[ta], sw_in=full_df[sw_in],
            nee_f=nee_f, ta_f=full_df[ta_f],
            lat=meta.site_lat, lon=meta.site_lon, utc_offset=meta.utc_offset,
            verbose=verbose,
        )

    return _run_level42(
        data,
        results_attr='level42_nt_rp',
        tracking_key='L4.2_nt_rp',
        fn_label='run_level42_nighttime_reddyproc',
        make_instance=make_instance,
        needs_nee_f=True,
        gapfill_method=gapfill_method,
    )


def run_level42_daytime_reddyproc(
        data: FluxLevelData,
        *,
        ta_f: str,
        vpd_f: str,
        sw_in_f: str,
        nee_sd: str | None = None,
        vpd_in_kpa: bool = True,
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO with the daytime method (REddyProc).

    Faithful port of REddyProc's ``partitionNEEGL`` (Lasslop et al. 2010):
    per-window light-response-curve fit on the measured daytime NEE, with
    gap-filled meteo as drivers. Runs one partitioning per USTAR scenario; site
    ``lat`` / ``lon`` / ``utc_offset`` are taken from ``data.meta``. The daytime
    method uses **measured NEE only** (no L4.1 gap-filled NEE required).

    Args:
        data: FluxLevelData after ``run_level33_*``.
        ta_f: Gap-filled air temperature column (**deg C**) in ``data.full_df``.
        vpd_f: Gap-filled VPD column in ``data.full_df`` — **kPa** by default
            (``vpd_in_kpa=True``; converted to hPa internally).
        sw_in_f: Gap-filled shortwave incoming radiation column (**W m-2**) in
            ``data.full_df`` (day/night split and LRC light driver).
        nee_sd: Optional per-record NEE uncertainty column (umol m-2 s-1) in
            ``data.full_df`` used to weight the LRC fit. ``None`` reproduces
            REddyProc's default ``max(0.7, 0.2*|NEE|)``.
        vpd_in_kpa: Whether ``vpd_f`` is in kPa (default). Set ``False`` if hPa.
        verbose: Console verbosity passed to the partitioning instance.

    Returns:
        Updated FluxLevelData; instances via
        ``data.levels.level42_dt_rp[ustar_scenario]``, columns via
        ``data.partitioned_cols()['dt_rp']``.
    """
    from diive.flux.partitioning import DaytimePartitioningReddyProc

    rule("Level 4.2: NEE Partitioning (Daytime REddyProc, Lasslop 2010)")
    required = {'ta_f': ta_f, 'vpd_f': vpd_f, 'sw_in_f': sw_in_f}
    if nee_sd is not None:
        required['nee_sd'] = nee_sd
    _require_full_df_cols(data, required)
    full_df = data.full_df
    meta = data.meta
    nee_sd_series = full_df[nee_sd] if nee_sd is not None else None

    def make_instance(scen, nee_meas, nee_f):
        return DaytimePartitioningReddyProc(
            nee=nee_meas, ta=full_df[ta_f], vpd=full_df[vpd_f], sw_in=full_df[sw_in_f],
            lat=meta.site_lat, lon=meta.site_lon, utc_offset=meta.utc_offset,
            nee_sd=nee_sd_series, vpd_in_kpa=vpd_in_kpa, verbose=verbose,
        )

    return _run_level42(
        data,
        results_attr='level42_dt_rp',
        tracking_key='L4.2_dt_rp',
        fn_label='run_level42_daytime_reddyproc',
        make_instance=make_instance,
        needs_nee_f=False,
        gapfill_method=None,
    )


def run_level42_daytime_oneflux(
        data: FluxLevelData,
        *,
        ta: str,
        sw_in: str,
        ta_f: str,
        sw_in_f: str,
        vpd_f: str,
        vpd_in_kpa: bool = True,
        gapfill_method: str | None = 'mds',
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO with the daytime method (ONEFlux).

    Faithful port of ONEFlux's ``flux_part_gl2010`` (Lasslop et al. 2010,
    FLUXNET2015). Per-window LRC fit with a measured-radiation day/night split
    (no latitude). Runs one partitioning per USTAR scenario. Fits **measured
    NEE only**, so L4.1 is not required. If L4.1 ran ``gapfill_method``, its
    gap-filled NEE of each scenario is passed on as ``nee_f``: ONEFlux reads
    it for windows without measured nighttime NEE (see
    :class:`~diive.flux.partitioning.DaytimePartitioningOneFlux`). Without it
    such windows are not fitted and the partitioning warns.

    The scenarios are partitioned together with
    :func:`~diive.flux.partitioning.partition_nee_daytime_oneflux_scenarios`:
    when a fit fails, the measured NEE in that window's night range is removed
    from every scenario of that year, and the scenarios are partitioned again
    until no fit fails, as ONEFlux shares its error file between u*
    percentiles. The removed ranges are in ``removed_days`` of each instance.
    This equals ONEFlux only when the scenarios are ONEFlux's percentiles in
    its order.

    Args:
        data: FluxLevelData after ``run_level33_*``.
        ta: Measured air temperature column (**deg C**) in ``data.full_df``.
        sw_in: Measured shortwave incoming radiation column (**W m-2**) in
            ``data.full_df`` (Rg-threshold day/night split + uncertainty lookup).
        ta_f: Gap-filled air temperature column (**deg C**) in ``data.full_df``.
        sw_in_f: Gap-filled shortwave incoming radiation column (**W m-2**) in
            ``data.full_df`` (LRC light driver).
        vpd_f: Gap-filled VPD column in ``data.full_df`` — **kPa** by default.
        vpd_in_kpa: Whether ``vpd_f`` is in kPa (default). Set ``False`` if hPa.
        gapfill_method: Which L4.1 gap-filled NEE to pass as ``nee_f``
            (``'mds'`` default / ``'rf'`` / ``'xgb'``), used where it has run.
            ``None`` passes none.
        verbose: Console verbosity passed to the partitioning instance.

    Returns:
        Updated FluxLevelData; instances via
        ``data.levels.level42_dt_of[ustar_scenario]``, columns via
        ``data.partitioned_cols()['dt_of']``.
    """
    from diive.flux.partitioning import partition_nee_daytime_oneflux_scenarios

    rule("Level 4.2: NEE Partitioning (Daytime ONEFlux, Lasslop 2010)")
    _require_full_df_cols(data, {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f,
                                 'sw_in_f': sw_in_f, 'vpd_f': vpd_f})
    full_df = data.full_df
    nee_f_by_scen = _available_nee_f(data, gapfill_method,
                                     list(require_level33(data)))

    def run_instances(nee_by_scen):
        # ONEFlux shares removed days only within one product (CUT or VUT).
        # All scenarios here come from one L3.3 run, which applies either
        # constant or variable thresholds and replaces any earlier scenarios,
        # so they form one product and share every range.
        return partition_nee_daytime_oneflux_scenarios(
            nee=nee_by_scen, ta=full_df[ta], sw_in=full_df[sw_in],
            ta_f=full_df[ta_f], sw_in_f=full_df[sw_in_f], vpd=full_df[vpd_f],
            vpd_in_kpa=vpd_in_kpa, nee_f=nee_f_by_scen, verbose=verbose,
        )

    return _run_level42(
        data,
        results_attr='level42_dt_of',
        tracking_key='L4.2_dt_of',
        fn_label='run_level42_daytime_oneflux',
        run_instances=run_instances,
        needs_nee_f=False,
        gapfill_method=None,
    )


#: (method, mode) -> L4.2 variant token, as used by ``data.partitioned_cols()``
#: and ``data.levels.level42_<token>``.
PARTITION_VARIANTS: dict[tuple[str, str], str] = {
    ('nighttime', 'oneflux'): 'nt_of',
    ('nighttime', 'reddyproc'): 'nt_rp',
    ('daytime', 'oneflux'): 'dt_of',
    ('daytime', 'reddyproc'): 'dt_rp',
}

#: Driver-column arguments each variant needs in run_level42_partitioning.
_VARIANT_DRIVERS: dict[str, tuple[str, ...]] = {
    'nt_of': ('ta', 'sw_in', 'ta_f'),
    'nt_rp': ('ta', 'sw_in', 'ta_f'),
    'dt_rp': ('ta_f', 'vpd_f', 'sw_in_f'),
    'dt_of': ('ta', 'sw_in', 'ta_f', 'sw_in_f', 'vpd_f'),
}


def partition_variant(method: str, mode: str = 'oneflux') -> str:
    """Return the L4.2 variant token (``'nt_of'``, ...) for a method and mode."""
    from diive.flux.partitioning.nee import _check_method_mode
    _check_method_mode(method, mode)
    return PARTITION_VARIANTS[(method, mode)]


def run_level42_partitioning(
        data: FluxLevelData,
        *,
        method: str,
        mode: str = 'oneflux',
        ta: str | None = None,
        sw_in: str | None = None,
        ta_f: str | None = None,
        sw_in_f: str | None = None,
        vpd_f: str | None = None,
        nee_sd: str | None = None,
        gapfill_method: str = 'mds',
        vpd_in_kpa: bool = True,
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO, with the method and mode as switches.

    One entry point for the four L4.2 variants, the chain counterpart of
    :class:`~diive.flux.partitioning.PartitioningNEE`. It calls the matching
    per-variant function, so the result is the same as calling that function:

    - ``method='nighttime', mode='oneflux'``: :func:`run_level42_nighttime_oneflux`
      (needs ``ta``, ``sw_in``, ``ta_f`` and an L4.1 run of ``gapfill_method``)
    - ``method='nighttime', mode='reddyproc'``: :func:`run_level42_nighttime_reddyproc`
      (same inputs)
    - ``method='daytime', mode='oneflux'``: :func:`run_level42_daytime_oneflux`
      (needs ``ta``, ``sw_in``, ``ta_f``, ``sw_in_f``, ``vpd_f``; uses the
      L4.1 run of ``gapfill_method`` if there is one)
    - ``method='daytime', mode='reddyproc'``: :func:`run_level42_daytime_reddyproc`
      (needs ``ta_f``, ``sw_in_f``, ``vpd_f``, optional ``nee_sd``)

    ONEFlux, the ICOS and FLUXNET2015 approach, is the default mode. Driver
    columns the chosen variant does not use are ignored. Each call adds the
    columns of one variant, so calls with different modes or methods add up,
    as with the per-variant functions.

    Args:
        data: FluxLevelData after ``run_level33_*`` (and an L4.1 run for the
            nighttime method).
        method: ``'nighttime'`` (Reichstein et al. 2005) or ``'daytime'``
            (Lasslop et al. 2010).
        mode: ``'oneflux'`` (default) or ``'reddyproc'``.
        ta: Measured air temperature column (**deg C**) in ``data.full_df``.
        sw_in: Measured shortwave incoming radiation column (**W m-2**).
        ta_f: Gap-filled air temperature column (**deg C**).
        sw_in_f: Gap-filled shortwave incoming radiation column (**W m-2**).
        vpd_f: Gap-filled VPD column, **kPa** by default (see ``vpd_in_kpa``).
        nee_sd: Optional NEE uncertainty column, daytime REddyProc only.
        gapfill_method: Which L4.1 gap-filled NEE the nighttime method uses
            (``'mds'`` default / ``'rf'`` / ``'xgb'``). Daytime ONEFlux passes
            it on as ``nee_f`` where it has run.
        vpd_in_kpa: Whether ``vpd_f`` is in kPa (default). Daytime method only.
        verbose: Console verbosity passed to the partitioning instances.

    Returns:
        Updated FluxLevelData. The variant token for ``data.partitioned_cols()``
        and ``data.levels.level42_<token>`` is ``partition_variant(method, mode)``.
    """
    variant = partition_variant(method, mode)
    given = {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f, 'sw_in_f': sw_in_f,
             'vpd_f': vpd_f}
    needed = _VARIANT_DRIVERS[variant]
    missing = [name for name in needed if given[name] is None]
    if missing:
        raise ValueError(
            f"run_level42_partitioning(method={method!r}, mode={mode!r}) needs "
            f"the driver column(s) {', '.join(needed)}. "
            f"Missing: {', '.join(missing)}.")
    drivers = {name: given[name] for name in needed}

    if variant == 'nt_of':
        return run_level42_nighttime_oneflux(
            data, **drivers, gapfill_method=gapfill_method, verbose=verbose)
    if variant == 'nt_rp':
        return run_level42_nighttime_reddyproc(
            data, **drivers, gapfill_method=gapfill_method, verbose=verbose)
    if variant == 'dt_rp':
        return run_level42_daytime_reddyproc(
            data, **drivers, nee_sd=nee_sd, vpd_in_kpa=vpd_in_kpa, verbose=verbose)
    return run_level42_daytime_oneflux(
        data, **drivers, vpd_in_kpa=vpd_in_kpa, gapfill_method=gapfill_method,
        verbose=verbose)
