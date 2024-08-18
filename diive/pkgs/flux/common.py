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
    """Detect name of base variable that was used to calculate
    the respective flux."""
    fluxbasevar = fluxbasevars_fluxnetfile[fluxcol]
    if not fluxbasevar:
        raise KeyError(f'No base variable for {fluxcol} could be detected.')
    print(f"Detected base variable {fluxbasevar} for {fluxcol}. "
          f"({fluxbasevar} was used to calculate {fluxcol}.)")
    return fluxbasevar
