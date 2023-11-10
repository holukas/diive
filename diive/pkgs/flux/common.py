from typing import Literal

# Names of flux variables and their base variable in EddyPro,
# names are different depending on output file (_full_output_ or _fluxnet_)
basevars_fluxnetfile = {
    'FC': 'CO2',
    'FH2O': 'H2O',
    'LE': 'H2O',
    'ET': 'H2O',
    'H': 'T_SONIC',
    'FN2O': 'N2O',
    'FCH4': 'CH4',
}

basevars_fulloutputfile = {
    'co2_flux': 'co2',
    'h2o_flux': 'h2o',
    'LE': 'h2o',
    'ET': 'h2o',
    'H': 'sonic_temperature',
    'n2o_flux': 'n2o',
    'ch4_flux': 'ch4',
}


def detect_basevar(fluxcol: str,
                   filetype: Literal['EDDYPRO_FLUXNET_30MIN', 'EDDYPRO_FULL_OUTPUT_30MIN']) -> str:
    """Detect name of base variable that was used to calculate
    the respective flux."""
    if filetype == 'EDDYPRO_FLUXNET_30MIN':
        basevar = basevars_fluxnetfile[fluxcol]
    elif filetype == 'EDDYPRO_FULL_OUTPUT_30MIN':
        basevar = basevars_fulloutputfile[fluxcol]
    else:
        raise Exception(f"(!) Filetype {filetype} is not defined. No basevar could be detected for {fluxcol}.")
    print(f"Detected base variable {basevar} for {fluxcol}.")
    return basevar
