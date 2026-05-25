"""
LEVEL INIT: BUILD INITIAL FLUXLEVELDATA CONTAINER
==================================================

Constructs a ``FluxLevelData`` from raw EddyPro data: adds potential
radiation, day/night flags, and assembles the frozen ``FluxMeta`` record.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from pandas import DataFrame

from diive.pkgs.features.variables import daytime_nighttime_flag_from_swinpot
from diive.pkgs.features.variables.potentialradiation import potrad
from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData, FluxMeta
from diive.pkgs.flux.lowres.common import detect_fluxbasevar


def init_flux_data(
        df: DataFrame,
        fluxcol: str,
        site_lat: float,
        site_lon: float,
        utc_offset: int,
        nighttime_threshold: float = 20,
        daytime_accept_qcf_below: int = 1,
        nighttimetime_accept_qcf_below: int = 1,
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
        daytime_accept_qcf_below: QCF value below which daytime data are
            retained (0, 1, or 2). Defaults to 1.
        nighttimetime_accept_qcf_below: QCF value below which nighttime
            data are retained. Defaults to 1.
        ustarcol: Name of the friction velocity column. Defaults to 'USTAR'.

    Returns:
        FluxLevelData ready to be passed to run_level2().
    """
    full_df = df.copy()
    fluxbasevar = detect_fluxbasevar(fluxcol=fluxcol)

    # Working dataframe: flux + ustar only at this stage
    fpc_df = full_df[[fluxcol, ustarcol]].copy()

    # Potential radiation → daytime / nighttime flags
    swinpot = potrad(timestamp_index=fpc_df.index,
                     lat=site_lat, lon=site_lon, utc_offset=utc_offset)
    swinpot_col = str(swinpot.name)
    print(f"Calculated potential radiation from latitude and longitude ({swinpot_col}) ...")

    daytime_flag, nighttime_flag = daytime_nighttime_flag_from_swinpot(
        swinpot=swinpot,
        nighttime_threshold=nighttime_threshold,
        daytime_col='DAYTIME',
        nighttime_col='NIGHTTIME')
    daytime_flag_col = str(daytime_flag.name)
    nighttime_flag_col = str(nighttime_flag.name)
    print(f"Calculated daytime flag {daytime_flag_col} and "
          f"nighttime flag {nighttime_flag_col} from {swinpot_col} ...")

    fpc_df[swinpot_col] = swinpot
    fpc_df[daytime_flag_col] = daytime_flag.copy()
    fpc_df[nighttime_flag_col] = nighttime_flag.copy()

    # Mirror new columns back into full_df so they are available as ML
    # features in later levels.
    for col in (swinpot_col, daytime_flag_col, nighttime_flag_col):
        overwritten = col in full_df.columns
        full_df[col] = fpc_df[col].copy()
        tag = " (!) Existing column overwritten." if overwritten else ""
        print(f"++ Added new column {col} to input data.{tag}")

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
        nighttimetime_accept_qcf_below=nighttimetime_accept_qcf_below,
        outname=outname,
    )

    return FluxLevelData(
        fpc_df=fpc_df,
        full_df=full_df,
        filteredseries=None,
        meta=meta,
    )
