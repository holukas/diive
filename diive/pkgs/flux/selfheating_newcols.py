class NewCols:
    """
    New columns that will be created

    FCT_unsc ... unscaled flux correction term
    FCT ... scaled flux correction term
    SF ... scaling factor
    LUT ... lookup table

    """
    class_var_group_col = '.GROUP_CLASSVAR'
    daytime_col = '.DAYTIME'
    k_air_col = '.AIR_THERMAL_CONDUCTIVITY'  # Thermal conductivity of air (W m-1 K-1)
    ra_col = '.AERODYNAMIC_RESISTANCE'  # Aerodynamic resistance (s m-1)
    rho_d_col = '.DRY_AIR_DENSITy'  # Dry air density (kg m-3)
    ts_col = '.T_INSTRUMENT_SURFACE'  # Instrument surface temperature (°C) from Järvi et al. (2009), merged column from daytime and nighttime temperatures
    op_co2_flux_corr_col = '.NEE_OP_CORR'  # Corrected CO2 flux; calculated as: WPL-corrected-only NEE + scaled correction flux, Ts from JAR09
    fct_unsc_col = '.FCT_UNSC'  # Unscaled flux correction term (µmol m-2 s-1)
    fct_unsc_gf_col = '.FCT_UNSC_GF'  # Unscaled flux correction term, gap-filled (µmol m-2 s-1)
    fct_unsc_lutvals_col = '.FCT_UNSC_LUTVALS'  # Unscaled flux correction term, LUT values (µmol m-2 s-1)
    fct_col = '.FCT'  # Scaled flux correction term (µmol m-2 s-1)
    sf_col = '.SF'  # Scaling factor
    sf_gf_col = '.SF_GF'  # Scaling factor, gap-filled
    sf_lutvals_col = '.SF_LUTVALS'  # Scaling factor, LUT values
