class NewCols:
    """
    New columns that will be created

    FCT_unsc ... unscaled flux correction term
    FCT ... scaled flux correction term
    SF ... scaling factor
    LUT ... lookup table

    """
    class_var_group = '.GROUP_CLASSVAR'
    daytime = '.DAYTIME'
    air_thermal_conductivity = '.AIR_THERMAL_CONDUCTIVITY'  # Thermal conductivity of air (W m-1 K-1) [k_air]
    aerodynamic_resistance = '.AERODYNAMIC_RESISTANCE'  # Aerodynamic resistance (s m-1) [ra]
    dry_air_density = '.DRY_AIR_DENSITy'  # Dry air density (kg m-3) [rho_d]
    t_instrument_surface = '.T_INSTRUMENT_SURFACE'  # Instrument surface temperature (°C) from Järvi et al. (2009), merged column from daytime and nighttime temperatures
    nee_op_corr = '.NEE_OP_CORR'  # Corrected CO2 flux; calculated as: WPL-corrected-only NEE + scaled correction flux, Ts from JAR09
    fct_unsc = '.FCT_UNSC'  # Unscaled flux correction term (µmol m-2 s-1)
    fct_unsc_gf = '.FCT_UNSC_GF'  # Unscaled flux correction term, gap-filled (µmol m-2 s-1)
    fct_unsc_lutvals = '.FCT_UNSC_LUTVALS'  # Unscaled flux correction term, LUT values (µmol m-2 s-1)
    fct = '.FCT'  # Scaled flux correction term (µmol m-2 s-1)
    sf = '.SF'  # Scaling factor
    sf_gf = '.SF_GF'  # Scaling factor, gap-filled
    sf_lutvals_col = '.SF_LUTVALS'  # Scaling factor, LUT values
