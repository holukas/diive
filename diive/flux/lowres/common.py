"""
COMMON: FLUX VARIABLE NOMENCLATURE AND HELPERS
===============================================

Mapping of flux variable names and base variable detection helpers.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.core.utils.console import info

# Names of flux variables and their base variable in EddyPro,
# names are different depending on output file (_full_output_ or _fluxnet_)
fluxbasevars_fluxnetfile = {
    'FC': 'CO2',
    'FH2O': 'H2O',
    'LE': 'H2O',
    'ET': 'H2O',
    'H': 'T_SONIC',
    'FN2O': 'N2O',
    'FCH4': 'CH4',
}

fluxbasevars_fulloutputfile = {
    'co2_flux': 'co2',
    'h2o_flux': 'h2o',
    'LE': 'h2o',
    'ET': 'h2o',
    'H': 'sonic_temperature',
    'n2o_flux': 'n2o',
    'ch4_flux': 'ch4',
}


def detect_fluxbasevar(fluxcol: str) -> str:
    """
    Detect name of base variable that was used to calculate the respective flux.

    Maps flux variable names (e.g., 'FC', 'LE', 'H') to their base gas/variable
    names (e.g., 'CO2', 'H2O', 'T_SONIC').

    Args:
        fluxcol (str): Name of the flux column (e.g., 'FC', 'FH2O', 'LE', 'H', 'FN2O', 'FCH4').

    Returns:
        str: The base variable name (e.g., 'CO2', 'H2O', 'T_SONIC').

    Raises:
        KeyError: If the flux variable name is not found in the mapping.

    See Also:
        analyze_highest_quality_flux : Filter highest-quality flux data using Hampel filter.

    Example:
        See `examples/pkgs/flux/lowres/flux_common.py` for examples of flux variable
        detection and nomenclature mapping.
    """
    fluxbasevar = fluxbasevars_fluxnetfile[fluxcol]
    if not fluxbasevar:
        raise KeyError(f'No base variable for {fluxcol} could be detected.')
    info(f"Detected base variable {fluxbasevar} for {fluxcol}. "
         f"({fluxbasevar} was used to calculate {fluxcol}.)")
    return fluxbasevar
