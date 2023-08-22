def detect_flux_basevar(fluxcol: str) -> str:
    """Detect name of base variable that was used to calculate
    the respective flux"""
    basevar = None
    if fluxcol == 'FC':
        basevar = 'CO2'
    elif (fluxcol == 'FH2O') \
            | (fluxcol == 'LE') \
            | (fluxcol == 'ET'):
        basevar = 'H2O'
    elif fluxcol == 'H':
        basevar = 'T_SONIC'
    elif fluxcol == 'FN2O':
        basevar = 'N2O'
    elif fluxcol == 'FCH4':
        basevar = 'CH4'
    print(f"Detected base variable {basevar} for {fluxcol}.")
    return basevar
