"""
cr ... correction
pm ... parallel measurements

"""

from configparser import ConfigParser
import ast
from pathlib import Path

class ReadSettingsFile():
    settingsfile = 'settings.yaml'

    def __init__(self):
        self.config = self._read_config()
        self._mapvars()

    def _searchfile(self):
        pass

    def _mapvars(self):
        self.site = self.config['GENERAL']['site']
        self.title = self.config['GENERAL']['title']
        self.outdir = Path(self.config['GENERAL']['outdir'])

        self.pm_file = self.config['PM']['pm_file']
        self.pm_op_flux_wpl_col = ast.literal_eval(self.config['PM']['pm_op_flux_wpl_col'])
        self.pm_qcf_op_flux_wpl_col = self.config['PM']['pm_qcf_op_flux_wpl_col']
        self.pm_cp_trueflux_col = self.config['PM']['pm_cp_trueflux_col']
        self.pm_qcf_cp_trueflux_col = self.config['PM']['pm_qcf_cp_trueflux_col']
        self.pm_airheatcap_jkkg_col = self.config['PM']['pm_airheatcap_jkkg_col']
        self.pm_qc_mmol_col = self.config['PM']['pm_qc_mmol_col']
        self.pm_u_col = self.config['PM']['pm_u_col']
        self.pm_ustar_col = self.config['PM']['pm_ustar_col']
        self.pm_rho_v_col = self.config['PM']['pm_rho_v_col']
        self.pm_rho_a_col = self.config['PM']['pm_rho_a_col']
        self.pm_ta_col = self.config['PM']['pm_ta_col']
        self.pm_swin_col = self.config['PM']['pm_swin_col']
        self.pm_num_classes = self.config['PM']['pm_num_classes']
        self.pm_num_bootstrap_runs = self.config['PM']['pm_num_bootstrap_runs']
        self.pm_class_var_col = self.config['PM']['pm_class_var_col']

        self.cr_file = Path(self.config['CR']['cr_file'])
        self.cr_sf_file = Path(self.config['CR']['cr_sf_file'])
        self.cr_ta_col = ast.literal_eval(self.config['CR']['cr_ta_col'])
        self.cr_swin_col = ast.literal_eval(self.config['CR']['cr_swin_col'])
        self.cr_ustar_col = ast.literal_eval(self.config['CR']['cr_ustar_col'])
        self.cr_u_col = ast.literal_eval(self.config['CR']['cr_u_col'])
        self.cr_rho_v_col = ast.literal_eval(self.config['CR']['cr_rho_v_col'])
        self.cr_rho_a_col = ast.literal_eval(self.config['CR']['cr_rho_a_col'])
        self.cr_qc_mmol_col = ast.literal_eval(self.config['CR']['cr_qc_mmol_col'])
        self.cr_op_flux_wpl_col = ast.literal_eval(self.config['CR']['cr_op_flux_wpl_col'])
        self.cr_class_var_col = ast.literal_eval(self.config['CR']['cr_class_var_col'])

    def get(self):
        return self.config

    def _read_config(self):
        config = ConfigParser()
        config.read(self.settingsfile, encoding='utf-8')
        # config.sections()
        # config.options('PM')
        return config


# class Settings():
#     # General
#     site = "CH-DAV"
#     title = "CH-DAV (2013-2016): same tower (2013-2016), same sonics (2013-2014), separate sonics (2014-2016)"
#     outdir = r"L:\Dropbox\luhk_work\programming\SCOP_Self-heating_Correction_Open-Path\OUT"
#
#     # 1-Calculation
#     # Needed for calculation of scaling factors from parallel measurements
#     pm_file = r"F:\00-CALCS-IRGA-INTERCOMP\DAV\0-1__IRGA75+IRGA72_2013-2016_PM2+PM3_+METEO_3-7\OUT_DIIVE-20210621-222325\Dataset_DIIVE-20210621-222325_Original-30T.diive.csv"
#     pm_op_co2_flux_nocorr_col = ('co2_flux_QC01_IRGA75', '[µmol+1s-1m-2]')  # Contains fluxes of quality 0 and 1
#     pm_qcf_op_co2_flux_nocorr_col = ('QCF_co2_flux_IRGA75', '[2=bad]')
#     pm_cp_co2_flux_col = ('co2_flux_QC01_IRGA72', '[µmol+1s-1m-2]')  # Contains fluxes of quality 0 and 1
#     pm_qcf_cp_co2_flux_nocorr_col = ('QCF_co2_flux_IRGA72', '[2=bad]')
#     # pm_op_le_flux_col = ('LE_QC01_IRGA75', '[W+1m-2]')  # Contains fluxes of quality 0 and 1
#     pm_airheatcap_JKkg_col = (
#         'air_heat_capacity_IRGA75', '[J+1kg-1K-1]')  # Specific heat at constant pressure of ambient air (J K-1 kg-1)
#     pm_qc_mmol_col = ('co2_molar_density_IRGA75', '[mmol+1m-3]')  # CO2 molar density column (mmol m-3)
#     pm_u_col = ('wind_speed_IRGA75', '[m+1s-1]')  # Horizontal wind speed (m s-1)
#     pm_ustar_col = ('u*_IRGA75', '[m+1s-1]')  # Ustar (m s-1)
#     pm_winddir_col = ('wind_dir_IRGA75', '[deg_from_north]')  # Ustar (m s-1)
#     pm_sensible_heat_col = ('H_QC01_IRGA75', '[W+1m-2]')  # Ustar (m s-1)
#     pm_rho_v_col = ('water_vapor_density_IRGA75', '[kg+1m-3]')  # Water vapor density (kg m-3)
#     pm_rho_a_col = ('air_density_IRGA75', '[kg+1m-3]')  # Air density (kg m-3)
#     pm_ta_col = ('TA', '--')  # Ambient air temperature (°C)
#     pm_swin_col = ('SW_IN', '--')  # Shortwave-incoming radiation (W m-2)
#     pm_vpd_col = ('VPD', '--')  # Vapor pressure deficit (hPa?)
#     # pm_class_var_col = 'custom'  # Scaling factors are calculated for each class of the class_var
#     pm_num_classes = 20  # Each class is bootstrapped, ignored if class_var_col = 'custom'
#     pm_num_bootstrap_runs = 99  # Number of bootstraps in each class, *0* uses measured data only w/o bootstrapping
#     pm_class_var_col = pm_ustar_col  # Scaling factors are calculated for each class of the class_var
#     # pm_class_var_col = NewCols.ra_col  # Scaling factors are calculated for each class of the class_var
#
#     # Needed for 2-Application of scaling factors to (uncorrected) fluxes
#     cr_file = r"F:\CH-DAV\[CALC]__EFDC_flux_update_WW2020_2005\2-3__IRGA75__Level-1_colSubsetForBurbaCorr\OUT_DIIVE-20210628-102759\subsetForBurbaCorr_Dataset_DIIVE-20210628-102759_Original-30T.diive.csv"
#     cr_sf_file = r"L:\Dropbox\luhk_work\programming\SCOP_Self-heating_Correction_Open-Path\OUT\1-Calculation\self-heating_scaling_factors.csv"
#     cr_ta_col = ('TA', '--')  # Ambient air temperature (°C)
#     cr_swin_col = ('SW_IN', '--')  # Shortwave-incoming radiation (W m-2)
#     cr_ustar_col = ('u*', '[m+1s-1]')  # Ustar (m s-1)
#     cr_u_col = ('wind_speed', '[m+1s-1]')  # Horizontal wind speed (m s-1)
#     cr_rho_v_col = ('water_vapor_density', '[kg+1m-3]')  # Water vapor density (kg m-3)
#     cr_rho_a_col = ('air_density', '[kg+1m-3]')  # Air density (kg m-3)
#     cr_qc_mmol_col = ('co2_molar_density', '[mmol+1m-3]')  # CO2 molar density column (mmol m-3)
#     cr_op_co2_flux_nocorr_col = ('co2_flux', '[µmol+1s-1m-2]')  # Contains fluxes
#     cr_class_var_col = cr_ustar_col  # Scaling factors are calculated for each class of the class_var
#
#     # Shared
#     # class_var_col = cr_op_co2_flux_nocorr_col  # Scaling factors are calculated for each class of the class_var
