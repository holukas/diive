# TODO what if flux is sensible heat H???
def detect_fluxgas(fluxcol: str) -> str:
    """Detect name of gas column that was used to calculate the flux"""
    gascol = None
    if fluxcol == 'FC':
        gascol = 'CO2'
    elif (fluxcol == 'FH2O') \
            | (fluxcol == 'LE') \
            | (fluxcol == 'ET'):
        gascol = 'H2O'
    elif fluxcol == 'FN2O':
        gascol = 'N2O'
    elif fluxcol == 'FCH4':
        gascol = 'CH4'
    print(f"Detected gas {gascol} for {fluxcol}.")
    return gascol
