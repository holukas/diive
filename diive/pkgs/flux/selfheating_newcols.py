class NewCols:
    """
    New columns that will be created

    FCT_unsc ... unscaled flux correction term
    FCT ... scaled flux correction term
    SF ... scaling factor

    """
    class_var_group_col = ('GROUP_CLASSVAR', '[group#]')
    daytime_col = ('DAYTIME', '[1=daytime]')
    k_air_col = ('k_air', '[W m-1 K-1]')  # thermal conductivity of air (W m-1 K-1)
    ra_col = ('ra', '[s m-1]')  # Aerodynamic resistance (s m-1)
    rho_d_col = ('rho_d', '[kg m-3]')  # Dry air density (kg m-3)
    ts_col = ('TS', '[°C]')  # Instrument surface temperature (°C) from Järvi et al. (2009), merged column from daytime and nighttime temperatures
    op_co2_flux_corr_col = ('co2_flux_CORR', '[µmol m-2 s-1]')  # Corrected CO2 flux; calculated as: WPL-corrected-only NEE + scaled correction flux, Ts from JAR09
    fct_unsc_col = ('FCT_unsc', '[µmol m-2 s-1]')
    fct_unsc_gf_col = ('FCT_unsc_gf', '[µmol m-2 s-1]')
    fct_unsc_lutvals_col = ('FCT_unsc_lutvals', '[µmol m-2 s-1]')
    fct_col = ('FCT', '[µmol m-2 s-1]')
    sf_col = ('SF', '[#]')
    sf_gf_col = ('SF_gf', '[#]')
    sf_lutvals_col = ('SF_lutvals', '[#]')
