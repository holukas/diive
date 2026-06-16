"""
LEVEL 4.2: NEE PARTITIONING (NEE -> GPP + RECO)
================================================

Composable callables for the four faithful NEE partitioning ports: nighttime
ONEFlux / nighttime REddyProc (Reichstein et al. 2005) and daytime REddyProc /
daytime ONEFlux (Lasslop et al. 2010). Each runs one partitioning instance per
USTAR scenario produced at Level-3.3 and merges its result columns into
``fpc_df`` with the scenario label appended (so all scenarios coexist).

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
``'mds'``). The **daytime** variants use measured NEE only and therefore do not
require L4.1 to have run.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pandas as pd

from diive.core.utils.console import rule
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._rerun import (
    drop_columns_for_key,
    record_added_columns,
)
from diive.flux.fluxprocessingchain.levels._shared import (
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


def _run_level42(
        data: FluxLevelData,
        *,
        results_attr: str,
        tracking_key: str,
        fn_label: str,
        make_instance: Callable[[str, pd.Series, pd.Series | None], object],
        needs_nee_f: bool,
        gapfill_method: str | None,
) -> FluxLevelData:
    """Internal: shared per-scenario partitioning loop for the four L4.2 variants.

    ``make_instance(scen, nee_meas, nee_f)`` builds the partitioning instance
    for one USTAR scenario; ``nee_f`` is the gap-filled NEE for the nighttime
    variants and ``None`` for the daytime ones.
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

    for scen, nee_meas in filteredseries_l33.items():
        instance = make_instance(scen, nee_meas, nee_f_by_scen.get(scen))
        instance.run()
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
        verbose: int = 1,
) -> FluxLevelData:
    """
    Level-4.2: Partition NEE into GPP + RECO with the daytime method (ONEFlux).

    Faithful port of ONEFlux's ``flux_part_gl2010`` (Lasslop et al. 2010,
    FLUXNET2015). Per-window LRC fit with a measured-radiation day/night split
    (no latitude). Runs one partitioning per USTAR scenario. Uses **measured
    NEE only** (no L4.1 gap-filled NEE required).

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
        verbose: Console verbosity passed to the partitioning instance.

    Returns:
        Updated FluxLevelData; instances via
        ``data.levels.level42_dt_of[ustar_scenario]``, columns via
        ``data.partitioned_cols()['dt_of']``.
    """
    from diive.flux.partitioning import DaytimePartitioningOneFlux

    rule("Level 4.2: NEE Partitioning (Daytime ONEFlux, Lasslop 2010)")
    _require_full_df_cols(data, {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f,
                                 'sw_in_f': sw_in_f, 'vpd_f': vpd_f})
    full_df = data.full_df

    def make_instance(scen, nee_meas, nee_f):
        return DaytimePartitioningOneFlux(
            nee=nee_meas, ta=full_df[ta], sw_in=full_df[sw_in],
            ta_f=full_df[ta_f], sw_in_f=full_df[sw_in_f], vpd=full_df[vpd_f],
            vpd_in_kpa=vpd_in_kpa, verbose=verbose,
        )

    return _run_level42(
        data,
        results_attr='level42_dt_of',
        tracking_key='L4.2_dt_of',
        fn_label='run_level42_daytime_oneflux',
        make_instance=make_instance,
        needs_nee_f=False,
        gapfill_method=None,
    )
