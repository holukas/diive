AIR_TEMPERATURE_METEO = ['*TA_*', 'Ta_*']
AIR_TEMPERATURE_EDDYPRO = ['*air_temperature*']
AIR_TEMPERATURE = AIR_TEMPERATURE_METEO + AIR_TEMPERATURE_EDDYPRO

AIR_DENSITY = ['*air_density*']

MOLAR_DENSITY_CO2 = ['*co2_molar_density*']
MOLAR_DENSITY_H2O = ['*h2o_molar_density*']

H2O_VAPOR_DENSITY = ['*water_vapor_density*']

NEWLY_CREATED_VARS = ['.*']

FLUXES_EDDYPRO = ['co2_flux', 'h2o_flux', 'LE', 'H', 'ET', 'n2o_flux', 'ch4_flux']
FLUXES_FLUXNET = ['NEE*', 'GPP*', 'RECO*']
PRIORITY_VARS = FLUXES_EDDYPRO + FLUXES_FLUXNET + NEWLY_CREATED_VARS
FLUXES_VARS = FLUXES_EDDYPRO + FLUXES_FLUXNET
FLUXES_GENERAL_CO2 = ['*co2_flux*']
FLUXES_GENERAL_H2O = ['*h2o_flux*']
FLUXES_GENERAL_ET = ['*ET*']
FLUXES_GENERAL_LE = ['*LE*']
FLUXES_GENERAL_H = ['*H*']
FLUXES_GENERAL_CH4 = ['*ch4_flux*']
FLUXES_GENERAL_N2O = ['*n2o_flux*']
FLUX_SCALARS_CO2_HIRES = ['co2_ppb*', 'CO2_ppm*', 'CO2_mmol_m-3*']

GAPFILLED_GENERAL_SCALARS = ['*-f*']

MONIN_OBUKHOV_STABILITY = ['(z-d)/L']

NIGHTTIME_DETECTION = ['*SW_IN*', '*Rg_*', '*daytime*', '*PPFD_*']

QCFLAGS_DIIVE_USTAR = ['QCF_USTAR_MPT_THRES_SEASON', 'QCF_USTAR_MPT_THRES_YEAR']
QCFLAGS_EDDYPRO_RAWDATA_ABSLIM = ['*absolute_limits_hf*']
QCFLAGS_EDDYPRO_RAWDATA_AMPLRES = ['*amplitude_resolution_hf*']
QCFLAGS_EDDYPRO_RAWDATA_ATTACKANGLE = ['*attack_angle_hf*']
QCFLAGS_EDDYPRO_RAWDATA_DISCONT_HF = ['*discontinuities_hf*']
QCFLAGS_EDDYPRO_RAWDATA_DISCONT_SF = ['*discontinuities_sf*']
QCFLAGS_EDDYPRO_RAWDATA_DROPOUT = ['*drop_out_hf*']
QCFLAGS_EDDYPRO_RAWDATA_NONSTEADYWIND = ['*non_steady_wind_hf*']
QCFLAGS_EDDYPRO_RAWDATA_SKEWKURT_HF = ['*skewness_kurtosis_hf*']
QCFLAGS_EDDYPRO_RAWDATA_SKEWKURT_SF = ['*skewness_kurtosis_sf*']
QCFLAGS_EDDYPRO_RAWDATA_SPIKES = ['*spikes_hf*']
QCFLAGS_EDDYPRO_RAWDATA_COMPLETENESS = ['*file_records*']
QCFLAGS_EDDYPRO_SSITC = ['*qc_*']  # Default EddyPro flag
QCFLAGS_EDDYPRO_SCF = ['*_scf*']  # Spectral correction factor
# QCFLAGS = QCFLAGS_DIIVE + QCFLAGS_EDDYPRO_SSITC + QCFLAGS_EDDYPRO_SCF

SHORTWAVE_IN = ['*SW_IN_*', 'Rg_*']
SIGNAL_STRENGTH_GA = ['*signal_strength*', '*window_dirtiness*', '*status_byte*', '*agc*']
SPECTRAL_CORRECTION_FACTOR = ['*_scf*']
STORAGE = ['*_strg*']

TEMP_SONIC_VIRT = ['*sonic_temperature*']

USTAR_EDDYPRO = ['u[*]?', 'u*']  # Star symbol in brackets w/ question mark searches symbol, not wildcard

VPD = ['*VPD*']

WIND_DIR = ['wind_dir']
WIND_SPEED = ['*wind_speed*', '*u_rot*']
WIND_U_HIRES = ['u_ms-1']
WIND_V_HIRES = ['v_ms-1']
WIND_W_HIRES = ['w_ms-1']


# QCFLAGS_RAWDATA_EDDYPRO = ['spikes_hf', 'amplitude_resolution_hf', 'drop_out_hf',
#                            'absolute_limits_hf', 'skewness_kurtosis_hf', 'skewness_kurtosis_sf',
#                            'discontinuities_hf', 'discontinuities_sf', 'timelag_hf',
#                            'timelag_sf', 'attack_angle_hf', 'non_steady_wind_hf']
# QCFLAGS_AMP_AGG = ['QCF_AGG_*']