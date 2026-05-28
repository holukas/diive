"""
LEVEL INIT: BUILD INITIAL FLUXLEVELDATA CONTAINER
==================================================

Constructs a ``FluxLevelData`` from raw EddyPro data: adds potential
radiation, day/night flags, and assembles the frozen ``FluxMeta`` record.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from pandas import DataFrame

from diive.core.utils.console import detail, info, rule
from diive.variables import daytime_nighttime_flag_from_swinpot
from diive.variables.radiation import potrad
from diive.flux.fluxprocessingchain.container import FluxLevelData, FluxMeta
from diive.flux.lowres.common import detect_fluxbasevar


def init_flux_data(
        df: DataFrame,
        fluxcol: str,
        site_lat: float,
        site_lon: float,
        utc_offset: int,
        nighttime_threshold: float = 20,
        daytime_accept_qcf_below: int = 1,
        nighttime_accept_qcf_below: int = 1,
        ustarcol: str = 'USTAR',
) -> FluxLevelData:
    """
    Build the initial FluxLevelData container from raw EddyPro data.

    Adds potential radiation and day/night flags to the working dataframe,
    assembles the frozen FluxMeta record, and returns a FluxLevelData ready
    for the first level callable.

    Args:
        df: Input DataFrame containing flux and meteorological data.
        fluxcol: Name of the raw flux column (e.g. 'FC', 'LE', 'H').
        site_lat: Site latitude (decimal degrees).
        site_lon: Site longitude (decimal degrees).
        utc_offset: UTC offset in hours.
        nighttime_threshold: Potential radiation threshold (W m-2) below
            which records are treated as nighttime. Defaults to 20.
        daytime_accept_qcf_below: QCF threshold for daytime data retention.
            Records with QCF **below** this value are kept; those at or above
            are set to NaN.  Allowed values:

            - ``1`` (default) — keep only QCF=0 (all tests pass).  Strictest.
            - ``2`` — keep QCF=0 and QCF=1 (soft warnings tolerated).
              This matches the conventional FLUXNET / Swiss FluxNet choice
              (Papale et al. 2006, Reichstein et al. 2005) and maximises
              data availability while still rejecting hard failures.

            **QCF value meanings:**

            - ``QCF=0`` — all quality tests passed; highest confidence.
            - ``QCF=1`` — one or more soft warnings (minor issues); marginal quality.
            - ``QCF=2`` — at least one hard failure or many soft warnings; rejected.

        nighttime_accept_qcf_below: Same as ``daytime_accept_qcf_below`` but
            applied to nighttime records.  Defaults to 1.  Set to 2 for the
            conventional FLUXNET treatment.
        ustarcol: Name of the friction velocity column. Defaults to 'USTAR'.

    Returns:
        FluxLevelData ready to be passed to run_level2().

    Note:
        **FC → NEE rename:** when ``fluxcol='FC'``, the output column produced
        by Level-3.1 (storage-corrected flux) is automatically named ``NEE``
        following the FLUXNET convention.  All subsequent level outputs and
        gap-filling results use ``NEE`` as the base name.  For any other flux
        (e.g. ``'LE'``, ``'H'``, ``'FCH4'``), the original column name is kept.
    """
    missing = [c for c in (fluxcol, ustarcol) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) not found in df: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Reserved column names this function computes and writes into full_df.
    # full_df is the read-only driver source for downstream levels; silently
    # overwriting a user-supplied column with the same name would mean the
    # user's data is replaced by ours without them noticing.
    reserved = ('SW_IN_POT', 'DAYTIME', 'NIGHTTIME')
    conflicts = [c for c in reserved if c in df.columns]
    if conflicts:
        raise ValueError(
            f"Input df already contains reserved column(s) {conflicts}. "
            f"init_flux_data() computes these from potential radiation and "
            f"writes them into full_df, which is treated as a read-only "
            f"driver source by later levels. Rename or drop these columns "
            f"before calling init_flux_data()."
        )

    rule(f"Initializing flux data: {fluxcol}")

    full_df = df.copy()
    fluxbasevar = detect_fluxbasevar(fluxcol=fluxcol)

    # Working dataframe: flux + ustar only at this stage
    fpc_df = full_df[[fluxcol, ustarcol]].copy()

    # Potential radiation → daytime / nighttime flags
    swinpot = potrad(timestamp_index=fpc_df.index,
                     lat=site_lat, lon=site_lon, utc_offset=utc_offset)
    swinpot_col = str(swinpot.name)
    info(f"Potential radiation calculated ({swinpot_col})")

    daytime_flag, nighttime_flag = daytime_nighttime_flag_from_swinpot(
        swinpot=swinpot,
        nighttime_threshold=nighttime_threshold,
        daytime_col='DAYTIME',
        nighttime_col='NIGHTTIME')
    daytime_flag_col = str(daytime_flag.name)
    nighttime_flag_col = str(nighttime_flag.name)
    info(f"Day/night flags: {daytime_flag_col}, {nighttime_flag_col}")

    fpc_df[swinpot_col] = swinpot
    fpc_df[daytime_flag_col] = daytime_flag.copy()
    fpc_df[nighttime_flag_col] = nighttime_flag.copy()

    # Mirror new columns back into full_df so they are available as ML
    # features in later levels. Reserved-name check above guarantees these
    # do not collide with user-supplied columns.
    for col in (swinpot_col, daytime_flag_col, nighttime_flag_col):
        full_df[col] = fpc_df[col].copy()
        detail(f"Added column {col}")

    # CO2 flux (FC) is renamed to NEE during Level-3.1
    outname = 'NEE' if fluxcol == 'FC' else fluxcol

    meta = FluxMeta(
        fluxcol=fluxcol,
        fluxbasevar=fluxbasevar,
        ustarcol=ustarcol,
        swinpot_col=swinpot_col,
        site_lat=site_lat,
        site_lon=site_lon,
        utc_offset=utc_offset,
        nighttime_threshold=nighttime_threshold,
        daytime_accept_qcf_below=daytime_accept_qcf_below,
        nighttime_accept_qcf_below=nighttime_accept_qcf_below,
        outname=outname,
    )

    return FluxLevelData(
        fpc_df=fpc_df,
        full_df=full_df,
        filteredseries=None,
        meta=meta,
    )
