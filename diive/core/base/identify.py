from pandas import DataFrame


def identify_relevants(seriescol: str) -> list:
    """Find relevant series column.

    Needed because variables can change their naming over the
    course of the QC checks, e.g. for NEE, checks done on the
    variable FC are relevant.
    """
    if seriescol.startswith('NEE_') or seriescol == 'FC' or seriescol == 'co2_flux':
        relevant = ['_FC_', '_NEE_', '_co2_flux_']
    elif seriescol.startswith('co2_flux_'):
        relevant = ['CHECK', '_NEE_']  # todo
    elif seriescol.startswith('H_') or seriescol == 'H':
        relevant = ['_H_']
    elif seriescol.startswith('LE_') or seriescol == 'LE':
        relevant = ['_LE_']
    elif seriescol.startswith('ET_') or seriescol == 'ET':
        relevant = ['_ET_']
    elif seriescol.startswith('FH2O_') or seriescol == 'FH2O':
        relevant = ['_FH2O_']
    elif seriescol.startswith('h2o_flux_') or seriescol == 'h2o_flux':
        relevant = ['_h2o_flux_']
    elif seriescol.startswith('TAU_') or seriescol == 'TAU':
        relevant = ['_TAU_']
    elif seriescol.startswith('FN2O_') or seriescol == 'FN2O':
        relevant = ['_FN2O_']
    elif seriescol.startswith('FCH4_') or seriescol == 'FCH4':
        relevant = ['_FCH4_']
    else:
        relevant = [seriescol]
    return relevant


def identify_flagcols(df: DataFrame, seriescol: str) -> list:
    """Identify flag columns."""
    flagcols = [c for c in df.columns
                if str(c).startswith('FLAG_')
                and (str(c).endswith(('_TEST', '_QCF')))]

    # Collect columns relevant for this flux
    relevant = identify_relevants(seriescol=seriescol)
    flagcols = [f for f in flagcols if any(n in f for n in relevant)]

    return flagcols
