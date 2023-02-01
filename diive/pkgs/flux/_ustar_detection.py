"""

USTAR THRESHOLD DETECTION
=========================

How thresholds are calculated
-----------------------------
- Data are divided into S seasons
- In each season, data are divided into X air temperature (TA) classes
- Each air temperature class is divided into Y ustar (USTAR) subclasses (data are binned per subclass)
- Thresholds are calculated in each TA class
- Season thresholds per bootstrapping run are calculated from all found TA class thresholds, e.g. the max of thresholds
- Overall season thresholds are calculated from all season thresholds from all bootstrapping runs, e.g. the median
- Yearly thresholds are calculated from overall season thresholds

"""

import fnmatch

import matplotlib.gridspec as gridspec
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', 7)
# pd.set_option('display.max_rows', 999)
from PyQt5 import QtWidgets as qw
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import gui.elements
import gui.plotfuncs
from gui import elements
from help import tooltips
from utils.vargroups import *
from modboxes.plots.styles.LightTheme import *


class MakeNewTab:
    """
    Create tab with instance data for ustar analysis.
    """

    dir_out_option = 'ustar_detection_auto'

    def __init__(self, TabWidget, data_df, col_list_pretty, col_dict_tuples, new_tab_id, ctx, dir_out):

        self.TabWidget = TabWidget
        self.col_list_pretty = col_list_pretty
        self.col_dict_tuples = col_dict_tuples
        self.data_df = data_df
        self.dir_out = dir_out / self.dir_out_option
        self.ctx = ctx

        linebreak = '\n'
        self.title = f"Ustar Threshold{linebreak}Detection {new_tab_id}"  ## used in tab name and header

        # Create tab menu (refinements)
        self.drp_flux, self.drp_ta, self.lne_TA_numClasses, self.drp_uStar, self.lne_uStar_numClasses, \
        self.btn_run, self.ref_frame, self.drp_define_nighttime_col, self.drp_seasons, self.lne_flux_plateau_thres_perc, \
        self.drp_define_bts_season_thres, self.drp_define_year_thres, \
        self.lne_ta_ustar_corr_thres, self.lne_set_to_ustar_data_percentile, self.lne_allowed_min_ustar_threshold, \
        self.lne_num_bootstrap_runs, self.lne_set_nighttime_thres, \
        self.btn_add_results_as_new_cols, self.btn_export_results_to_file, \
        self.drp_define_overall_season_thres, self.drp_flux_plateau_method = \
            self.create_ref_tab_menu()

        # self.drp_templateReference.currentTextChanged.connect(
        #     lambda: self.update_refinements_with_template(template=self.drp_templateReference.currentText()))

        # Create and add new tab
        self.canvas, self.figure, self.axes_dict \
            = self.create_tab()

        # Update the gui dropdown menus of the new tab
        self.update_refinements()

    def create_tab(self):
        tabContainerVertLayout, tab_ix = \
            self.TabWidget.add_new_tab(title=self.title)
        self.TabWidget.setCurrentIndex(tab_ix)  ## select newly created tab

        # HEADER (top): Create and add
        elements.add_label_to_layout(txt=self.title, css_id='lbl_Header2', layout=tabContainerVertLayout)

        # MENU (left): Stack for refinement menu
        tabMenu_ref_stack = qw.QStackedWidget()
        tabMenu_ref_stack.setProperty('labelClass', '')
        tabMenu_ref_stack.setContentsMargins(0, 0, 0, 0)
        tabMenu_ref_stack.addWidget(self.ref_frame)

        # PLOT & NAVIGATION CONTROLS (right)
        # in horizontal layout

        # Figure (right)
        # Figure axes are created later in MakeUstarAnalysis#setup_plot_axes
        tabFigure = plt.Figure(facecolor='white')
        # tabFigure = plt.Figure(facecolor='#29282d')
        tabCanvas = FigureCanvas(tabFigure)

        # Make axes
        axes_dict = self.make_axes_dict(tabFigure=tabFigure)

        # Navigation (right)
        tabToolbar = NavigationToolbar(tabCanvas, parent=self.TabWidget)

        # Frame for figure & navigation (right)
        frm_fig_nav = qw.QFrame()
        lytVert_fig_nav = qw.QVBoxLayout()  # create layout for frame
        frm_fig_nav.setLayout(lytVert_fig_nav)  # assign layout to frame
        lytVert_fig_nav.setContentsMargins(0, 0, 0, 0)
        lytVert_fig_nav.addWidget(tabCanvas)  # add widget to layout
        lytVert_fig_nav.addWidget(tabToolbar)

        # ASSEMBLE MENU AND PLOT
        # Splitter for menu on the left and plot on the right
        tabSplitter = qw.QSplitter()
        tabSplitter.addWidget(tabMenu_ref_stack)  ## add ref menu first (left)
        tabSplitter.addWidget(frm_fig_nav)  ## add plot & navigation second (right)
        tabSplitter.setStretchFactor(0, 1)
        tabSplitter.setStretchFactor(1, 2)  ## stretch right more than left
        tabContainerVertLayout.addWidget(tabSplitter, stretch=1)  ## add Splitter to tab layout

        return tabCanvas, tabFigure, axes_dict

    def make_axes_dict(self, tabFigure):

        gs = gridspec.GridSpec(13, 6)  # rows, cols
        gs.update(wspace=0.4, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

        # Time series plots with class colors

        # Plot headers
        ax_header_timeseries_with_TA_classes = tabFigure.add_subplot(gs[0, 0:2])
        ax_header_seasonal_bins = tabFigure.add_subplot(gs[0, 2:4])
        ax_header_results = tabFigure.add_subplot(gs[0, 4:6])

        # Air temperature
        ax_TA_with_classes = tabFigure.add_subplot(
            gs[1:5, 0:2])
        ax_TA_vs_USTAR_subclass_avg_season = tabFigure.add_subplot(
            gs[1:5, 2:4])

        # Flux
        ax_FLUX_with_TA_classes = tabFigure.add_subplot(
            gs[5:9, 0:2], sharex=ax_TA_with_classes)
        ax_FLUX_vs_USTAR_subclass_avg_season = tabFigure.add_subplot(
            gs[5:9, 2:4])

        # Ustar
        ax_USTAR_with_TA_classes = tabFigure.add_subplot(
            gs[9:13, 0:2], sharex=ax_TA_with_classes)
        ax_USTAR_vs_USTAR_subclass_avg_season = tabFigure.add_subplot(
            gs[9:13, 2:4])

        # Overall
        ax_USTAR_thres_run_results = tabFigure.add_subplot(
            gs[1:5, 4:6])
        ax_USTAR_thres_seasons_boxenplot = tabFigure.add_subplot(
            gs[5:9, 4:6])
        ax_USTAR_thres_seasons_scatterplot = tabFigure.add_subplot(
            gs[9:13, 4:6])

        axes_dict = {'ax_header_timeseries_with_TA_classes': ax_header_timeseries_with_TA_classes,
                     'ax_header_seasonal_bins': ax_header_seasonal_bins,
                     'ax_header_results': ax_header_results,

                     'ax_TA_with_classes': ax_TA_with_classes,
                     'ax_FLUX_with_TA_classes': ax_FLUX_with_TA_classes,
                     'ax_USTAR_with_TA_classes': ax_USTAR_with_TA_classes,

                     'ax_TA_vs_USTAR_subclass_avg_season': ax_TA_vs_USTAR_subclass_avg_season,
                     'ax_FLUX_vs_USTAR_subclass_avg_season': ax_FLUX_vs_USTAR_subclass_avg_season,
                     'ax_USTAR_vs_USTAR_subclass_avg_season': ax_USTAR_vs_USTAR_subclass_avg_season,

                     'ax_USTAR_thres_run_results': ax_USTAR_thres_run_results,
                     'ax_USTAR_thres_seasons_boxenplot': ax_USTAR_thres_seasons_boxenplot,
                     'ax_USTAR_thres_seasons_scatterplot': ax_USTAR_thres_seasons_scatterplot}

        for key, ax in axes_dict.items():
            gui.plotfuncs.default_format(ax=ax, txt_xlabel=False)

        return axes_dict

    def create_ref_tab_menu(self):
        ref_frame, ref_layout = elements.add_frame_grid()
        # Reichstein et al (2005), Papale et al (2006)

        # BOOTSTRAP SETTINGS
        # ------------------
        gui.elements.add_header_in_grid_row(layout=ref_layout, txt='Bootstrap Settings', row=0)
        lne_num_bootstrap_runs = elements.add_label_linedit_pair_to_grid(txt='Number of Bootstrap Runs',
                                                                         css_ids=['', 'cyan'],
                                                                         layout=ref_layout,
                                                                         row=1, col=0,
                                                                         orientation='horiz')

        ref_layout.addWidget(qw.QLabel(), 2, 0, 1, 1)  ## spacer

        # VARIABLES
        # ---------
        gui.elements.add_header_in_grid_row(
            layout=ref_layout, txt='Variables',
            row=3)

        drp_flux = elements.grd_LabelDropdownPair(
            txt='Flux',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=4, col=0)

        drp_TA = elements.grd_LabelDropdownPair(
            txt='Air Temperature',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=5, col=0)

        drp_uStar = elements.grd_LabelDropdownPair(
            txt='u*',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=6, col=0)

        ref_layout.addWidget(qw.QLabel(), 7, 0, 1, 1)  # Spacer

        drp_define_nighttime_col = elements.grd_LabelDropdownPair(
            txt='Define Nighttime Based On Column',
            css_ids=['', ''], layout=ref_layout, orientation='horiz',
            row=8, col=0)

        lne_set_nighttime_thres = elements.add_label_linedit_pair_to_grid(
            txt='Set As Nighttime If Value Smaller Than',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=9, col=0)

        ref_layout.addWidget(qw.QLabel(), 10, 0, 1, 1)  # Spacer

        txt_info_hover = 'Season Type: one threshold per season type, e.g. summer\n' \
                         'Season: one threshold per season, e.g. summer 2018, summer 2019, ...'
        drp_seasons = elements.add_label_dropdown_info_triplet_to_grid(
            txt='Seasons',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz', txt_info_hover=txt_info_hover,
            row=11, col=0)

        ref_layout.addWidget(qw.QLabel(), 12, 0, 1, 1)  # Spacer

        # DETECTION SETTINGS
        # ------------------
        gui.elements.add_header_in_grid_row(layout=ref_layout, txt='Detection Settings', row=13)

        # Air temperature classes
        lne_TA_numClasses = elements.add_label_linedit_info_triplet_to_grid(
            txt='Number of Air Temperature Classes', css_ids=['', 'cyan'], layout=ref_layout,
            row=14, col=0, orientation='horiz', txt_info_hover=tooltips.lne_TA_numClasses_txt_info_hover)

        # Ustar subclasses
        lne_uStar_numClasses = elements.add_label_linedit_info_triplet_to_grid(
            txt='Number of u* Subclasses',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=15, col=0, txt_info_hover=tooltips.lne_uStar_numClasses_txt_info_hover)

        # Flux subclass percentage threshold
        lne_flux_plateau_thres_perc = elements.add_label_linedit_info_triplet_to_grid(
            txt='Flux Plateau Threshold (%)',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=16, col=0, txt_info_hover=tooltips.lne_flux_plateau_thres_perc_txt_info_hover)

        # Flux subclass percentage method
        drp_flux_plateau_method = elements.add_label_dropdown_info_triplet_to_grid(
            txt='Flux Plateau Detection Method',
            css_ids=['', ''], layout=ref_layout, orientation='horiz',
            txt_info_hover=tooltips.drp_flux_plateau_method_txt_info_hover,
            row=17, col=0)

        # Correlation threshold of TA vs ustar in class
        lne_ta_ustar_corr_thres = elements.add_label_linedit_info_triplet_to_grid(
            txt='TA vs USTAR Class Correlation Threshold',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=18, col=0, txt_info_hover=tooltips.lne_ta_ustar_corr_thres_txt_info_hover)

        # Set to percentile
        lne_set_to_ustar_data_percentile = elements.add_label_linedit_info_triplet_to_grid(
            txt='Set To Percentile of USTAR data (0-1)',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=19, col=0, txt_info_hover=tooltips.lne_set_to_ustar_data_percentile_txt_info_hover)

        # Minimum allowed ustar threshold
        lne_allowed_min_ustar_threshold = elements.add_label_linedit_info_triplet_to_grid(
            txt='Allowed Minimum USTAR Threshold (m s-1)',
            css_ids=['', 'cyan'], layout=ref_layout, orientation='horiz',
            row=20, col=0, txt_info_hover=tooltips.lne_allowed_min_ustar_threshold_txt_info_hover)

        ref_layout.addWidget(qw.QLabel(), 21, 0, 1, 1)  ## spacer

        # THRESHOLD DEFINITIONS
        # ---------------------
        gui.elements.add_header_in_grid_row(
            layout=ref_layout, txt='Threshold Definitions',
            row=22)

        drp_define_bts_season_thres = elements.grd_LabelDropdownPair(
            txt='Threshold Per Bootstrap Season',
            css_ids=['', ''], layout=ref_layout, orientation='horiz',
            row=23, col=0)

        drp_define_overall_season_thres = elements.grd_LabelDropdownPair(
            txt='Overall Season Threshold',
            css_ids=['', ''], layout=ref_layout, orientation='horiz',
            row=24, col=0)

        drp_define_year_thres = elements.grd_LabelDropdownPair(
            txt='Threshold Per Year',
            css_ids=['', ''], layout=ref_layout, orientation='horiz',
            row=25, col=0)

        ref_layout.addWidget(qw.QLabel(), 26, 0, 1, 1)  # Spacer

        # # TEMPLATES
        # # ---------
        # gui.gui_elements.add_header_in_grid_row(layout=ref_layout, header='Templates', row=28)
        #
        # drp_templateReference = gui_elements.grd_LabelDropdownPair(txt='Reference',
        #                                                            css_ids=['', ''],
        #                                                            layout=ref_layout,
        #                                                            row=29, col=0,
        #                                                            orientation='horiz')
        #
        # ref_layout.addWidget(qw.QLabel(), 30, 0, 1, 1)  ## spacer

        # BUTTONS
        # -------
        btn_run = elements.add_button_to_grid(grid_layout=ref_layout,
                                              txt='Run', css_id='btn_cat_ControlsRun',
                                              row=27, col=0, rowspan=1, colspan=3)

        ref_layout.addWidget(qw.QLabel(), 28, 0, 1, 1)  # Spacer

        btn_add_results_as_new_cols = elements.add_iconbutton_to_grid(grid_layout=ref_layout,
                                                                      txt='Add Results As New Columns',
                                                                      css_id='icon_btn_controls_add',
                                                                      row=29, col=0, rowspan=1, colspan=3,
                                                                      icon=self.ctx.icon_btn_controls_add)
        btn_add_results_as_new_cols.setDisabled(True)

        btn_export_results_to_file = elements.add_iconbutton_to_grid(grid_layout=ref_layout,
                                                                     txt='Export Results To File',
                                                                     css_id='icon_btn_controls_add',
                                                                     row=30, col=0, rowspan=1, colspan=3,
                                                                     icon=self.ctx.icon_btn_controls_add)
        btn_export_results_to_file.setDisabled(True)

        ref_layout.setRowStretch(31, 2)  ## empty row

        return drp_flux, drp_TA, lne_TA_numClasses, drp_uStar, lne_uStar_numClasses, btn_run, ref_frame, \
               drp_define_nighttime_col, drp_seasons, lne_flux_plateau_thres_perc, drp_define_bts_season_thres, \
               drp_define_year_thres, lne_ta_ustar_corr_thres, lne_set_to_ustar_data_percentile, \
               lne_allowed_min_ustar_threshold, lne_num_bootstrap_runs, lne_set_nighttime_thres, \
               btn_add_results_as_new_cols, btn_export_results_to_file, \
               drp_define_overall_season_thres, drp_flux_plateau_method

    def update_refinements(self):
        self.drp_flux.clear()
        self.drp_ta.clear()
        self.drp_uStar.clear()
        self.drp_seasons.clear()
        self.drp_define_nighttime_col.clear()

        # Default values for the text fields
        self.lne_num_bootstrap_runs.setText('1')
        self.lne_TA_numClasses.setText('4')  ## todo will be 6 by default
        self.lne_uStar_numClasses.setText('20')  ## todo will be 20 by default
        self.lne_ta_ustar_corr_thres.setText('0.4')
        self.lne_set_to_ustar_data_percentile.setText('0.9')
        self.lne_allowed_min_ustar_threshold.setText('0.01')
        self.drp_seasons.addItem('None')

        # self.drp_templateReference.addItem('Custom')
        # # self.drp_templateReference.addItem('Reichstein et al. (2005)')
        # self.drp_templateReference.addItem('Papale et al. (2006)')

        i = ['Maximum Of Season Thresholds', 'Minimum Of Season Thresholds',
             'Median Of Season Thresholds', 'Mean Of Season Thresholds']
        self.drp_define_year_thres.addItems(i)

        i = ['Maximum of Class Thresholds', 'Minimum of Class Thresholds',
             'Median of Class Thresholds', 'Mean of Class Thresholds']
        self.drp_define_bts_season_thres.addItems(i)
        self.drp_define_bts_season_thres.setCurrentIndex(2)

        i = ['Maximum Of Bootstrap Season Thresholds', 'Minimum Of Bootstrap Season Thresholds',
             'Median Of Bootstrap Season Thresholds', 'Mean Of Bootstrap Season Thresholds']
        self.drp_define_overall_season_thres.addItems(i)
        self.drp_define_overall_season_thres.setCurrentIndex(2)

        i = ['NEE > 10+10 Higher USTAR Subclasses', 'NEE > 10 Higher USTAR Subclasses']
        self.drp_flux_plateau_method.addItems(i)

        # i = ['Season Type', 'Season']
        # self.drp_season_data_pool.addItems(i)

        self.lne_flux_plateau_thres_perc.setText('99')
        self.lne_set_nighttime_thres.setText('1')

        default_flux_ix = default_ta_ix = default_ustar_ix = 0
        default_flux_ix_found = default_ta_ix_found = default_ustar_ix_found = False

        for ix, colname_tuple in enumerate(self.data_df.columns):
            # Add all variables to dropdown menus for full flexibility yay
            self.drp_flux.addItem(self.col_list_pretty[ix])
            self.drp_ta.addItem(self.col_list_pretty[ix])
            self.drp_uStar.addItem(self.col_list_pretty[ix])

            if any(fnmatch.fnmatch(colname_tuple[0], vid) for vid in FLUXES_GENERAL_CO2):
                default_flux_ix = ix if not default_flux_ix_found else default_flux_ix
                default_flux_ix_found = True

            elif any(fnmatch.fnmatchcase(colname_tuple[0], vid) for vid in AIR_TEMPERATURE_METEO):
                default_ta_ix = ix if not default_ta_ix_found else default_ta_ix
                default_ta_ix_found = True

            elif any(fnmatch.fnmatchcase(colname_tuple[0], vid) for vid in USTAR_EDDYPRO):
                default_ustar_ix = ix if not default_ustar_ix_found else default_ustar_ix
                default_ustar_ix_found = True

            elif any(fnmatch.fnmatchcase(colname_tuple[0], vid) for vid in NIGHTTIME_DETECTION):
                self.drp_define_nighttime_col.addItem(self.col_list_pretty[ix])

            # DETECT SEASONS
            # --------------
            # A bit different than for the other dropdowns, b/c here we only add GRP_SEASON_* variables.
            elif any(fnmatch.fnmatchcase(colname_tuple[0], vid) for vid in ['GRP_SEASON_*']):
                self.drp_seasons.addItem(self.col_list_pretty[ix])

        # Set dropdown selection to found ix
        self.drp_flux.setCurrentIndex(default_flux_ix)
        self.drp_uStar.setCurrentIndex(default_ustar_ix)
        self.drp_ta.setCurrentIndex(default_ta_ix)
        # self.drp_define_nighttime_col.setCurrentIndex(default_nighttime_ix)
        # self.drp_seasons.setCurrentIndex(default_season_ix)


# todo check CLASS SLOTS https://stackoverflow.com/questions/472000/usage-of-slots

class Run:
    bts_results_df = pd.DataFrame()  # Collects detailed results from bootstrapping runs
    results_seasons_df = pd.DataFrame()  # Collects essential results from bts runs
    daynight_ustar_fullres_df = pd.DataFrame()  # Contains full-resolution daytime data cols for return to main
    night_ustar_fullres_df = pd.DataFrame()  # Contains full-resolution nighttime data cols for calcs

    ix_slice = pd.IndexSlice  # Create an object to more easily perform multi-index slicing
    markers = gui.plotfuncs.wheel_markers_7()  # Plot marker types

    # Return columns
    out_thres_season_col = ('USTAR_MPT_THRES_SEASON', '[m s-1]')  # Season threshold calculated from bootstrap results
    out_qcflag_season_col = ('QCF_USTAR_MPT_THRES_SEASON', '[2=bad]')
    out_thres_year_col = ('USTAR_MPT_THRES_YEAR', '[m s-1]')  # Year threshold calculated from seasons
    out_qcflag_year_col = ('QCF_USTAR_MPT_THRES_YEAR', '[2=bad]')

    # Required columns
    thres_year_col = ('USTAR_MPT_THRES_YEAR', '[m s-1]')
    thres_season_col = ('USTAR_MPT_THRES_SEASON', '[m s-1]')
    thres_class_col = ('USTAR_MPT_THRES_CLASS', '[m s-1]')
    bts_run_col = ('BTS_RUN', '[#]')  # Bootstrapping run
    year_col = ('YEAR', '[yyyy]')

    results_max_bts_col = ('MAX_BTS', '[m s-1]')
    results_min_bts_col = ('MIN_BTS', '[m s-1]')
    results_median_bts_col = ('MEDIAN_BTS', '[m s-1]')
    results_mean_bts_col = ('MEAN_BTS', '[m s-1]')
    results_p25_bts_col = ('P25_BTS', '[m s-1]')
    results_p75_bts_col = ('P75_BTS', '[m s-1]')
    results_bts_runs_col = ('BTS_RUNS', '[#]')
    results_bts_results_col = ('BTS_RESULTS', '[#]')

    def __init__(self, tab_instance):

        # General
        self.axes_dict = tab_instance.axes_dict
        self.col_dict_tuples = tab_instance.col_dict_tuples
        self.col_list_pretty = tab_instance.col_list_pretty
        self.data_df = tab_instance.tab_data_df.copy()
        self.fig = tab_instance.fig
        self.dir_out = tab_instance.dir_out

        # Settings GUI elements
        self.lne_allowed_min_ustar_threshold = tab_instance.lne_allowed_min_ustar_threshold
        self.drp_define_nighttime_col = tab_instance.drp_define_nighttime_col
        self.lne_set_nighttime_thres = tab_instance.lne_set_nighttime_thres
        self.drp_define_year_thres = tab_instance.drp_define_year_thres
        self.drp_define_bts_season_thres = tab_instance.drp_define_bts_season_thres
        self.drp_define_overall_season_thres = tab_instance.drp_define_overall_season_thres
        self.drp_flux = tab_instance.drp_flux
        self.lne_flux_plateau_thres_perc = tab_instance.lne_flux_plateau_thres_perc
        self.drp_flux_plateau_method = tab_instance.drp_flux_plateau_method
        self.lne_num_bootstrap_runs = tab_instance.lne_num_bootstrap_runs
        self.drp_seasons = tab_instance.drp_seasons
        self.lne_set_to_ustar_data_percentile = tab_instance.lne_set_to_ustar_data_percentile
        self.drp_ta = tab_instance.drp_ta
        self.lne_TA_numClasses = tab_instance.lne_TA_numClasses
        self.lne_ta_ustar_corr_thres = tab_instance.lne_ta_ustar_corr_thres
        self.drp_uStar = tab_instance.drp_uStar
        self.lne_uStar_numClasses = tab_instance.lne_uStar_numClasses

        # Buttons
        self.btn_add_results_as_new_cols = tab_instance.btn_add_results_as_new_cols
        self.btn_export_results_to_file = tab_instance.btn_export_results_to_file
        self.btn_run = tab_instance.btn_run
        self.btn_run_orig_text = tab_instance.btn_run.text()

    def run(self):
        """Bootstrap(bts) and analyse data."""
        self.daynight_ustar_fullres_df, self.night_ustar_fullres_df, \
        overall_runs, timestamp_col = \
            self.prepare_bts_run()

        season_colors_dict = self.assign_colors_to_seasons()

        # Collect essential results from all bts runs
        self.results_seasons_df = self.init_results_seasons_df()  # Reset df
        self.results_years_df = self.init_results_years_df()  # Reset df

        # BOOTSTRAP LOOP
        # ==============
        # The bootstrap loop is also used when analysis is done on measured data only
        for bts_run in range(0, overall_runs):
            print('\n\n\nBOOTSTRAP RUN {}'.format(bts_run))

            # SAMPLE DATA
            # -----------
            night_ustar_fullres_df_sample = self.sample_data(df=self.night_ustar_fullres_df)
            night_ustar_fullres_df_sample.reset_index(inplace=True)  # Keeps timestamp as column in df

            # SEPARATE BY SEASON DATA POOL
            # ----------------------------

            # Loop through all available seasons
            season_counter = -1  # Count if first, second, ... season in this bts run
            season_grouped_df = night_ustar_fullres_df_sample.groupby(self.season_grouping_col)
            for season_key, season_df in season_grouped_df:
                season_counter += 1
                # ASSIGN CLASSES
                # --------------
                # Create df with assigned TA and USTAR classes for this season.
                season_df, ta_class_col, ustar_class_col = \
                    self.assign_classes(season_df=season_df,
                                        class_col=self.ta_data_col,
                                        subclass_col=self.ustar_data_col,
                                        num_classes=self.ta_num_classes,
                                        num_subclasses=self.ustar_num_classes)

                # CALCULATE SEASON THRESHOLDS
                # ---------------------------
                # Calculate average values in ustar subclasses
                season_subclass_avg_df = \
                    self.calculate_subclass_averages(season_df=season_df,
                                                     ta_class_id_col=ta_class_col,
                                                     ustar_subclass_id_col=ustar_class_col,
                                                     season_key=season_key)

                # Detect thresholds in each air temperature class
                season_subclass_avg_df = \
                    self.detect_class_thresholds(season_subclass_avg_df=season_subclass_avg_df,
                                                 ta_class_id_col=ta_class_col,
                                                 season_key=season_key)

                # Set season thresholds based on results in temperature classes
                season_subclass_avg_df, this_ustar_thres_season = \
                    self.set_season_thresholds(season_subclass_avg_df=season_subclass_avg_df,
                                               season_key=season_key)

                # COLLECT SEASON RESULTS
                # ----------------------
                this_bts_results_df = self.prepare_results_for_collection(df=season_subclass_avg_df,
                                                                          bts_run=bts_run)
                self.bts_results_df = self.bts_results_df.append(this_bts_results_df)

                self.results_seasons_df = self.collect_season_results(bts_results_df=self.bts_results_df,
                                                                      season_key=season_key,
                                                                      results_df=self.results_seasons_df)

                # Collect found season threshold in full-resolution dataframe
                self.daynight_ustar_fullres_df = self.insert_this_season_thres_into_fullres(
                    df=self.daynight_ustar_fullres_df,
                    season_key=season_key,
                    this_ustar_thres_season=this_ustar_thres_season)

                # PLOT SEASON RESULTS
                # -------------------
                self.make_plots(fullres_df_with_classes=season_df,
                                ta_class_col=ta_class_col,
                                timestamp_col=timestamp_col,
                                this_bts_results_df=this_bts_results_df,
                                season_subclass_avg_df=season_subclass_avg_df,
                                season_key=season_key,
                                bts_run=bts_run,
                                results_df=self.results_seasons_df,
                                season_colors_dict=season_colors_dict,
                                season_counter=season_counter)

                # Update Figure during calculations
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

                elements.btn_txt_live_update(btn=self.btn_run, txt=self.btn_run_orig_text, perc=-9999)

            # TODO hier weiter
            # CALCULATE YEARLY THRESHOLDS
            # ---------------------------
            # After last season of current bts_run is reached, calculate the yearly threshold(s)
            self.daynight_ustar_fullres_df = self.set_yearly_thresholds(df=self.daynight_ustar_fullres_df)

            yearly_thresholds_df = self.collect_yearly_thresholds(bts_run=bts_run)

            self.results_years_df = self.collect_year_results(yearly_thresholds_df=yearly_thresholds_df,
                                                              results_years_df=self.results_years_df)

            # # Add yearly threshold results to previously collected class and season thresholds
            # self.bts_results_df = self.insert_yearly_thresholds(
            #     daynight_ustar_fullres_df=self.daynight_ustar_fullres_df,
            #     bts_results_df=self.bts_results_df)

        # Now we have results, therefore activate buttons
        self.btn_add_results_as_new_cols.setEnabled(True)
        self.btn_export_results_to_file.setEnabled(True)

    def collect_yearly_thresholds(self, bts_run):
        years = self.daynight_ustar_fullres_df.index.year.unique()
        if bts_run == 0:  # TODO hier weiter
            yearly_thresholds_df = pd.DataFrame(columns=years)  # Reset df
        grouped_df = self.daynight_ustar_fullres_df.groupby(self.daynight_ustar_fullres_df.index.year)
        for year_key, year_df in grouped_df:
            threshold = year_df[self.thres_year_col].unique()
            threshold = threshold[~np.isnan(threshold)]  # Remove NaN
            yearly_thresholds_df.loc[bts_run, year_key] = float(threshold)
        return yearly_thresholds_df

    def insert_yearly_thresholds(self, daynight_ustar_fullres_df, bts_results_df):
        """Combine yearly thresholds from the full-resolution df with the df that collects
        class and season thresholds.
        """
        grouped_df = daynight_ustar_fullres_df.groupby(self.season_grouping_col)
        for season_group_key, season_group_df in grouped_df:
            threshold = season_group_df[self.thres_year_col].unique()
            threshold = threshold[~np.isnan(threshold)]  # Remove NaN
            bts_results_df.loc[self.ix_slice[season_group_key, :, :], self.thres_year_col] = float(threshold)
            # df.loc[self.ix_slice[season_key, class_key, :], self.thres_class_col] = cur_ustar
        return bts_results_df

    def get_settings_from_gui(self):
        """ Get current settings from GUI elements. """
        self.allowed_min_ustar_thres = float(self.lne_allowed_min_ustar_threshold.text())
        self.nighttime_col = self.drp_define_nighttime_col.currentText()
        self.nighttime_thres = float(self.lne_set_nighttime_thres.text())
        self.define_year_thres = self.drp_define_year_thres.currentText()
        self.define_bts_season_thres = self.drp_define_bts_season_thres.currentText()
        self.define_overall_season_thres = self.drp_define_overall_season_thres.currentText()
        self.flux_data_col = self.drp_flux.currentText()
        self.flux_plateau_thres_perc = float(self.lne_flux_plateau_thres_perc.text())
        self.flux_plateau_method = str(self.drp_flux_plateau_method.currentText())
        self.num_bootstrap_runs = int(self.lne_num_bootstrap_runs.text())
        self.seasons = self.drp_seasons.currentText()
        self.set_to_ustar_data_percentile = float(self.lne_set_to_ustar_data_percentile.text())
        self.ta_data_col = self.drp_ta.currentText()
        self.ta_num_classes = int(self.lne_TA_numClasses.text())
        self.ta_ustar_corr_thres = float(self.lne_ta_ustar_corr_thres.text())
        self.ustar_data_col = self.drp_uStar.currentText()
        self.ustar_num_classes = int(self.lne_uStar_numClasses.text())

    def prepare_bts_run(self):
        """Setup before bootstrapping."""
        self.bts_results_df = pd.DataFrame()  # Reset when run button was clicked

        # Settings
        self.get_settings_from_gui()
        self.season_grouping_col = self.get_colname_tuples(colname_pretty=self.drp_seasons.currentText())

        # Check if season info available
        self.data_df = self.check_season_info(data_df=self.data_df)

        # If data are not bootstrapped, still use the bootstrap loop but don't sample
        overall_runs = 1 if self.num_bootstrap_runs == 0 else self.num_bootstrap_runs

        # Data for analysis
        daynight_ustar_fullres_df, night_ustar_fullres_df, self.flux_data_col, self.ta_data_col, \
        self.ustar_data_col, self.nighttime_col = \
            self.make_ustar_df()

        timestamp_col = night_ustar_fullres_df.index.name
        return daynight_ustar_fullres_df, night_ustar_fullres_df, overall_runs, timestamp_col

    def assign_colors_to_seasons(self):
        """Assign a unique color to each season for plotting. Colors can then be
        accessed by the season key during plotting.
        """
        unique_seasons = self.night_ustar_fullres_df[self.season_grouping_col].unique()
        colors = colorwheel_36()
        season_colors_dict = {}
        for ix, s in enumerate(unique_seasons):
            season_colors_dict[s] = colors[ix]
        return season_colors_dict

    def set_yearly_thresholds(self, df):
        """Calculate yearly thresholds from season thresholds in full-resolution dataframe."""

        elements.btn_txt_live_update(btn=self.btn_run, txt='Calculating yearly thresholds ...', perc=-9999)

        _df = df.copy()

        # Group data by year
        grouped_df = _df.groupby(_df.index.year)
        for year_key, year_df in grouped_df:

            # Check which thresholds were found for this year
            found_season_thresholds = year_df.loc[:, self.thres_season_col].unique()
            found_season_thresholds = found_season_thresholds[~np.isnan(found_season_thresholds)]  # Remove NaN

            if len(found_season_thresholds) > 0:
                # Set threshold for this year based on season results
                if self.define_year_thres == 'Maximum Of Season Thresholds':
                    threshold = found_season_thresholds.max()
                elif self.define_year_thres == 'Minimum Of Season Thresholds':
                    threshold = found_season_thresholds.min()
                elif self.define_year_thres == 'Median Of Season Thresholds':
                    threshold = np.median(found_season_thresholds)
                elif self.define_year_thres == 'Mean Of Season Thresholds':
                    threshold = found_season_thresholds.mean()
                else:
                    threshold = '-unknown-option-'
            else:
                # If no season thresholds for this year were found, the threshold for the year
                # is set to the selected percentile of the ustar data (Papale et al., 2006).
                threshold = _df[self.ustar_data_col].quantile(self.set_to_ustar_data_percentile)

                # Check if threshold is larger than the minimum allowed threshold.
                threshold = self.allowed_min_ustar_thres if threshold < self.allowed_min_ustar_thres else threshold

            _df.loc[_df.index.year == year_key, self.out_thres_year_col] = threshold

        # Add flag integers
        # Flag, season thresholds
        _df.loc[_df[self.ustar_data_col] > _df[self.out_thres_year_col],
                self.out_qcflag_year_col] = 0  # Good data
        _df.loc[_df[self.ustar_data_col] < _df[self.out_thres_year_col],
                self.out_qcflag_year_col] = 2  # Hard flag

        return _df

    def get_ustar_results(self, main_df):

        # Init columns in main df
        main_df[self.out_thres_season_col] = np.nan
        main_df[self.out_qcflag_season_col] = np.nan
        main_df[self.out_thres_year_col] = np.nan
        main_df[self.out_qcflag_year_col] = np.nan

        main_df[self.out_thres_season_col] = self.daynight_ustar_fullres_df[self.out_thres_season_col]
        main_df[self.out_qcflag_season_col] = self.daynight_ustar_fullres_df[self.out_qcflag_season_col]
        main_df[self.out_thres_year_col] = self.daynight_ustar_fullres_df[self.out_thres_year_col]
        main_df[self.out_qcflag_year_col] = self.daynight_ustar_fullres_df[self.out_qcflag_year_col]
        return main_df

    def insert_this_season_thres_into_fullres(self, df, season_key, this_ustar_thres_season):
        """Insert current season result into full-resolution dataframe.

        Combines the results from bootstrapping with the full-resolution data and
        the data timestamp. This connection with the full-resolution timestamp is
        needed in a later step, when the yearly threshold is calculated.
        """
        _df = df.copy()
        season_filter = _df[self.season_grouping_col] == season_key
        _df.loc[season_filter, self.out_thres_season_col] = this_ustar_thres_season
        print(f"season: {season_key}    season ustar: {this_ustar_thres_season}")

        # Add flag integers
        # Flag, season thresholds
        _df.loc[_df[self.ustar_data_col] > _df[self.out_thres_season_col],
                self.out_qcflag_season_col] = 0  # Good data
        _df.loc[_df[self.ustar_data_col] < _df[self.out_thres_season_col],
                self.out_qcflag_season_col] = 2  # Hard flag
        return _df

    # def insert_season_results_in_fullres(self, df):
    #     """Insert season results into full-resolution dataframe. Combines
    #     the results from bootstrapping with the full-resolution data.
    #     """
    #
    #     _df = df.copy()
    #
    #     # Init columns
    #     _df[self.out_thres_season_col] = np.nan
    #     _df[self.out_qcflag_season_col] = np.nan
    #
    #     for ix, row in self.results_df.iterrows():  # Season is in ix
    #         thres = row[self.thres_season_col]
    #         season_filter = _df[self.season_grouping_col] == ix
    #         _df.loc[season_filter, self.out_thres_season_col] = thres
    #         print(f"season: {ix}    season ustar: {thres}")
    #
    #     # Add flag integers
    #     # Flag, season thresholds
    #     _df.loc[_df[self.ustar_data_col] > _df[self.out_thres_season_col],
    #             self.out_qcflag_season_col] = 0  # Good data
    #     _df.loc[_df[self.ustar_data_col] < _df[self.out_thres_season_col],
    #             self.out_qcflag_season_col] = 2  # Hard flag
    #
    #     return _df

    def init_results_years_df(self):
        """Create dataframe that contains various threshold results for each year."""

        unique_years = self.night_ustar_fullres_df.index.year.unique()

        results_cols = [self.thres_year_col, self.results_max_bts_col, self.results_min_bts_col,
                        self.results_median_bts_col, self.results_mean_bts_col, self.results_p25_bts_col,
                        self.results_p75_bts_col, self.results_bts_runs_col, self.results_bts_results_col]

        results_df = pd.DataFrame(index=unique_years, columns=results_cols)  # Reset df
        results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

        return results_df

    def init_results_seasons_df(self):
        """Create dataframe that contains various threshold results for each season."""

        unique_seasons = self.night_ustar_fullres_df[self.season_grouping_col].unique()

        results_cols = [self.thres_season_col, self.results_max_bts_col, self.results_min_bts_col,
                        self.results_median_bts_col, self.results_mean_bts_col, self.results_p25_bts_col,
                        self.results_p75_bts_col, self.results_bts_runs_col, self.results_bts_results_col]

        results_df = pd.DataFrame(index=unique_seasons, columns=results_cols)  # Reset df
        results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

        return results_df

    def collect_year_results(self, yearly_thresholds_df, results_years_df):
        """Calculate different variants of year thresholds and the overall
        year threshold, based on bts results. Calculations are done afer
        each bts run."""

        years = yearly_thresholds_df.columns
        for year in years:
            found_yearly_thresholds = yearly_thresholds_df[year]

            filter_year = results_years_df.index == year

            # Different variants
            results_years_df.loc[filter_year, self.results_max_bts_col] = found_yearly_thresholds.max()
            results_years_df.loc[filter_year, self.results_min_bts_col] = found_yearly_thresholds.min()
            results_years_df.loc[filter_year, self.results_median_bts_col] = np.median(found_yearly_thresholds)
            results_years_df.loc[filter_year, self.results_mean_bts_col] = found_yearly_thresholds.mean()
            results_years_df.loc[filter_year, self.results_p25_bts_col] = np.quantile(found_yearly_thresholds, 0.25)
            results_years_df.loc[filter_year, self.results_p75_bts_col] = np.quantile(found_yearly_thresholds, 0.75)
            results_years_df.loc[filter_year, self.results_bts_runs_col] = self.num_bootstrap_runs
            results_years_df.loc[filter_year, self.results_bts_results_col] = len(found_yearly_thresholds)

            # Insert the overall season threshold, depending on selected option in dropdown menu
            if self.define_year_thres == 'Maximum Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_max_bts_col]
            elif self.define_year_thres == 'Minimum Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_min_bts_col]
            elif self.define_year_thres == 'Median Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_median_bts_col]
            elif self.define_year_thres == 'Mean Of Season Thresholds':
                results_years_df.loc[filter_year, self.thres_year_col] = \
                    results_years_df.loc[filter_year, self.results_mean_bts_col]
            else:
                results_years_df.loc[filter_year, self.thres_year_col] = np.nan

        return results_years_df

    def collect_season_results(self, bts_results_df, season_key, results_df):
        """Calculate different variants of season thresholds and the overall
        season threshold, based on bts results. Calculations are done after
        each bts run."""

        elements.btn_txt_live_update(btn=self.btn_run, txt='Collecting season results ...', perc=-9999)

        # Calculate different variants of season thresholds
        found_season_thresholds = bts_results_df.loc[season_key, self.thres_season_col].unique()
        found_season_thresholds = found_season_thresholds[~np.isnan(found_season_thresholds)]  # Remove NaN
        if len(found_season_thresholds) == 0:
            return results_df

        filter_season = results_df.index == season_key  # Current season

        # Different variants
        results_df.loc[filter_season, self.results_max_bts_col] = found_season_thresholds.max()
        results_df.loc[filter_season, self.results_min_bts_col] = found_season_thresholds.min()
        results_df.loc[filter_season, self.results_median_bts_col] = np.median(found_season_thresholds)
        results_df.loc[filter_season, self.results_mean_bts_col] = found_season_thresholds.mean()
        results_df.loc[filter_season, self.results_p25_bts_col] = np.quantile(found_season_thresholds, 0.25)
        results_df.loc[filter_season, self.results_p75_bts_col] = np.quantile(found_season_thresholds, 0.75)
        results_df.loc[filter_season, self.results_bts_runs_col] = self.num_bootstrap_runs
        results_df.loc[filter_season, self.results_bts_results_col] = len(found_season_thresholds)

        # Insert the overall season threshold, depending on selected option in dropdown menu
        if self.define_overall_season_thres == 'Maximum Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_max_bts_col]
        elif self.define_overall_season_thres == 'Minimum Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_min_bts_col]
        elif self.define_overall_season_thres == 'Median Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_median_bts_col]
        elif self.define_overall_season_thres == 'Mean Of Bootstrap Season Thresholds':
            results_df.loc[filter_season, self.thres_season_col] = \
                results_df.loc[filter_season, self.results_mean_bts_col]
        else:
            results_df.loc[filter_season, self.thres_season_col] = np.nan

        return results_df

    def export_results_to_file(self):
        """ Export results to file. """
        pkgs.dfun.files.verify_dir(dir=self.dir_out)
        current_timestamp = pkgs.dfun.times.make_timestamp_suffix()
        filepath_out_csv = self.dir_out / f'ustar_threshold_detection_auto_bts_results_{current_timestamp}.csv'
        self.bts_results_df.to_csv(filepath_out_csv)
        filepath_out_csv = self.dir_out / f'ustar_threshold_detection_auto_overall_results_{current_timestamp}.csv'
        self.results_seasons_df.to_csv(filepath_out_csv)

    def make_plots(self, fullres_df_with_classes, ta_class_col, timestamp_col, this_bts_results_df,
                   season_subclass_avg_df, season_key, bts_run, results_df, season_colors_dict,
                   season_counter):

        elements.btn_txt_live_update(btn=self.btn_run, txt='Plotting ...', perc=-9999)

        class_color_dict = colorwheel_36()

        # Add plot headers
        self.add_plot_headers()

        # LEFT: CLASS COLORS
        # ------------------
        # Plot data with class colors
        self.make_class_color_plots(fullres_df_with_classes=fullres_df_with_classes,
                                    ta_class_col=ta_class_col,
                                    class_color_dict=class_color_dict,
                                    timestamp_col=timestamp_col,
                                    season_key=season_key)

        # MIDDLE
        # ------
        self.make_season_plots(season_subclass_avg_df=season_subclass_avg_df, ta_class_col=ta_class_col,
                               thres_class_col=self.thres_class_col, thres_season_col=self.thres_season_col,
                               class_color_dict=class_color_dict,
                               season_key=season_key)

        # RIGHT: BTS RUN RESULTS
        # ----------------------
        self.plot_bts_season_thresholds(this_bts_results_df=this_bts_results_df,
                                        bts_run=bts_run,
                                        season_key=season_key,
                                        season_colors_dict=season_colors_dict,
                                        season_counter=season_counter)

        self.plot_boxenplots(bts_results_df=self.bts_results_df)

        self.plot_expanding_thresholds(results_df=results_df,
                                       bts_run=bts_run,
                                       season_colors_dict=season_colors_dict,
                                       season_counter=season_counter,
                                       season_key=season_key)

        # todo overall threshold?
        # self.plot_bootstrap_ustar(bts_results_df=self.bts_results_df,
        #                           this_bts_results_df=this_bts_results_df,
        #                           first_season=first_season,
        #                           season_key=season_key,
        #                           bts_run=bts_run)

        # else:
        #     self.plot_bootstrap_ustar(ax=self.axes_dict['ax_USTAR_thres_run_results'],
        #                               bts_results_df=self.bts_results_df, this_bts_results_df=this_bts_results_df)

        # Update Figure during calculations
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return None

    def plot_bts_season_thresholds(self, this_bts_results_df, bts_run, season_key, season_colors_dict, season_counter):
        """Plot threshold results for each season in each bts run."""

        ax = self.axes_dict['ax_USTAR_thres_run_results']
        if bts_run == 0 and season_counter == 0:
            ax.clear()  # Clear when new bootstrapping session

        season_ustar = this_bts_results_df[self.thres_season_col].unique()  # Contains only one unique value
        ax.scatter(bts_run, season_ustar,
                   color=season_colors_dict[season_key],
                   label=f'{season_key}',
                   s=SIZE_MARKER_LARGE,
                   marker=self.markers[0],
                   edgecolors='black')

        gui.plotfuncs.default_format(ax=ax, txt_xlabel='Bootstrap run #',
                                     txt_ylabel=self.ustar_data_col[0],
                                     txt_ylabel_units=self.ustar_data_col[1],
                                     label_color='black', fontsize=FONTSIZE_LABELS_AXIS)
        if bts_run == 0:
            gui.plotfuncs.default_legend(ax=ax, loc='upper right', bbox_to_anchor=(1, 0.95), labelspacing=1)
            ax.text(0.05, 0.95, f"{self.define_bts_season_thres} Per Bootstrap Season",
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, size=FONTSIZE_HEADER_AXIS,
                    color='black', backgroundcolor='none')

    def plot_boxenplots(self, bts_results_df):
        """Plot results from all bts runs so far. Data are not added to the plot, but
        the plot is generated each time from all collected results so far.

        Reduce df to contain only one season result per season. This reduced df will
        be the basis for further outputs. bts_results_df repeats the found season
        threshold for each temperature class, which would lead to stat calculations
        that would be based on this repeated values, which is not desirable.

        This plot uses the seaborn arg 'hue' to plot different colors for data pools,
        which seems to be much faster than first grouping and the plotting.
        """
        ax = self.axes_dict['ax_USTAR_thres_seasons_boxenplot']
        ax.clear()

        # For all seasons, make a subset that contains one result per bts season
        subset_df = bts_results_df.loc[self.ix_slice[:, 0, 0], :]

        # Make boxenplots from subset
        boxenplot_df = subset_df.copy().reset_index()  # Boxenplot cannot handle MultiIndex
        boxenplot_df['x'] = boxenplot_df[self.season_grouping_col]
        boxenplot_df['y'] = boxenplot_df[self.thres_season_col]
        boxenplot_df['hue'] = boxenplot_df[self.season_grouping_col]
        boxenplot_df = boxenplot_df[['x', 'y', 'hue']]
        sns.boxenplot(x='x', y='y', hue='hue',
                      data=boxenplot_df,
                      palette='RdYlBu',
                      ax=ax, scale='area')
        gui.plotfuncs.default_format(ax=ax, txt_xlabel='Season', txt_ylabel=self.ustar_data_col[0],
                                     txt_ylabel_units=self.ustar_data_col[1],
                                     label_color='black', fontsize=FONTSIZE_LABELS_AXIS)
        gui.plotfuncs.default_legend(ax=ax, loc='upper right', bbox_to_anchor=(1, 1), labelspacing=0.4)
        ax.text(0.05, 0.95, "Results per data pool per season",
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, size=FONTSIZE_HEADER_AXIS, color='black', backgroundcolor='none')

    def plot_expanding_thresholds(self, results_df, bts_run, season_colors_dict, season_counter, season_key):
        """Plot expanding overall threshold (from all bts runs so far).
        :param season_counter:
        """
        ax = self.axes_dict['ax_USTAR_thres_seasons_scatterplot']

        if bts_run == 0 and season_counter == 0:
            ax.clear()  # Reset plot for new bts run
            ax.text(0.05, 0.95, f"Expanding {self.define_overall_season_thres}",
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, size=FONTSIZE_HEADER_AXIS,
                    color='black', backgroundcolor='none')

        # for season_key, row in results_df.iterrows():
        filter_season = results_df.index == season_key
        threshold = results_df.loc[filter_season, self.thres_season_col]
        p25 = results_df.loc[filter_season, self.results_p25_bts_col]
        p75 = results_df.loc[filter_season, self.results_p75_bts_col]

        ax.plot(bts_run, threshold, color=season_colors_dict[season_key],
                marker=self.markers[0], ls='',
                lw=0, label=f'{season_key}',
                markersize=SIZE_MARKER_DEFAULT * 2, markeredgecolor='black', markeredgewidth=1)

        # Assymetric "error" (it is not an error, but the interquartile range)
        # https://stackoverflow.com/questions/31812469/plotting-asymmetric-error-bars-for-a-single-point-using-errorbar
        ax.vlines(bts_run, p25, p75,
                  color=season_colors_dict[season_key], lw=2, alpha=0.1)

        if bts_run == 0:
            gui.plotfuncs.default_legend(ax=ax, loc='upper right', bbox_to_anchor=(1, 0.95), labelspacing=1)

    def prepare_results_for_collection(self, df, bts_run):
        """Narrow df to ustar results."""
        # Subset: only ustar results columns
        ustar_resultcolumns = [self.thres_class_col, self.thres_season_col, self.thres_year_col]
        results_ustar_df = df[ustar_resultcolumns].copy()
        # results_ustar_df = df.loc[self.ix_slice[:, :, :], ustar_resultcolumns]
        results_ustar_df[self.bts_run_col] = bts_run
        return results_ustar_df

    def add_plot_headers(self):
        header_axes = ['ax_header_timeseries_with_TA_classes',
                       'ax_header_seasonal_bins',
                       'ax_header_results']

        header_txts = ['Time Series with Air Temperature Classes',
                       'Seasonal Thresholds per Air Temperature Class, From Bins',
                       'Results']

        for ix, ha in enumerate(header_axes):
            ax = self.axes_dict[ha]
            ax.clear()
            ax.text(0.5, 0.5, header_txts[ix],
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, size=FONTSIZE_HEADER_AXIS_LARGE, color='black', backgroundcolor='none')
            gui.plotfuncs.make_patch_spines_invisible(ax=ax)
            gui.plotfuncs.hide_ticks_and_ticklabels(ax=ax)

    def make_class_color_plots(self, fullres_df_with_classes, ta_class_col, class_color_dict, timestamp_col,
                               season_key):

        plot_dict = {'TA': (self.ta_data_col, self.axes_dict['ax_TA_with_classes']),
                     'FLUX': (self.flux_data_col, self.axes_dict['ax_FLUX_with_TA_classes']),
                     'USTAR': (self.ustar_data_col, self.axes_dict['ax_USTAR_with_TA_classes'])}

        for key, values in plot_dict.items():
            data_col = values[0]
            ax = values[1]

            # Plot with class colors
            self.plot_season_data_with_class_colors(fullres_df_with_classes=fullres_df_with_classes,
                                                    data_col=data_col,
                                                    ta_class_col=ta_class_col,
                                                    ax=ax,
                                                    class_colors_dict=class_color_dict,
                                                    current_season_data_pool_key=season_key,
                                                    timestamp_col=timestamp_col)

    def make_season_plots(self, season_subclass_avg_df, ta_class_col, thres_class_col, thres_season_col,
                          class_color_dict, season_key):
        # Plot averages in USTAR subclasses, per season and per TA class

        plot_dict = {
            'TA': (self.ta_data_col, 'ax_TA_vs_USTAR_subclass_avg_season'),
            'FLUX': (self.flux_data_col, 'ax_FLUX_vs_USTAR_subclass_avg_season'),
            'USTAR': (self.ustar_data_col, 'ax_USTAR_vs_USTAR_subclass_avg_season'),
        }

        scalar_ix = -1
        for scalar_key, values in plot_dict.items():
            data_col = values[0]
            ax_dict_id = values[1]
            scalar_ix += 1

            self.plot_per_season(season_subclass_avg_df=season_subclass_avg_df, ax_dict_id=ax_dict_id,
                                 x_col=self.ustar_data_col, y_col=data_col, ta_class_col=ta_class_col,
                                 class_color_dict=class_color_dict, thres_class_col=thres_class_col,
                                 thres_season_col=thres_season_col, scalar_ix=scalar_ix,
                                 current_season_data_pool_key=season_key)

    def calculate_subclass_averages(self, season_df, ta_class_id_col, ustar_subclass_id_col,
                                    season_key):
        """ In each TA class, calculate the TA and FLUX averages in USTAR subclasses.

        Create df that contains all the subclass averages per class and season.

        :param season_df: DataFrame
        :param ta_class_id_col: column name in df_with_classes
        :param ustar_subclass_id_col: column name in df_with_classes
        :return: DataFrame with MultiIndex (3 levels)
        """

        elements.btn_txt_live_update(btn=self.btn_run, txt='Calculating averages '
                                                               'per USTAR subclass ...', perc=-9999)
        means_of_group_subclass_df = pd.DataFrame()

        # CLASSES
        # -------
        class_grouped = season_df.groupby(ta_class_id_col)
        for class_key, class_group_df in class_grouped:

            # SUBCLASSES
            # ----------
            subclass_grouped = class_group_df.groupby(ustar_subclass_id_col)
            for subclass_key, subclass_group_df in subclass_grouped:
                means_of_group_subclass_S = subclass_group_df.mean()  ## Returns Series
                counts_of_group_subclass_S = subclass_group_df.count()
                SD_of_group_subclass_S = subclass_group_df.std_col()

                counts_of_group_subclass_S = counts_of_group_subclass_S.add_suffix('_COUNTS')
                SD_of_group_subclass_S = SD_of_group_subclass_S.add_suffix('_SD')

                means_of_group_subclass_S = means_of_group_subclass_S.append(counts_of_group_subclass_S)
                means_of_group_subclass_S = means_of_group_subclass_S.append(SD_of_group_subclass_S)

                _temp_df = pd.DataFrame(data=means_of_group_subclass_S)  # Convert Series to df
                _temp_df = _temp_df.T  # Transpose: Series index will be column names
                _temp_df.loc[:, self.season_grouping_col] = season_key

                if (class_key == 0) & (subclass_key == 0):
                    means_of_group_subclass_df = _temp_df.copy()
                else:
                    means_of_group_subclass_df = means_of_group_subclass_df.append(_temp_df)

        # Insert season data pool, class and subclass as pandas MultiIndex and drop respective data cols from df
        multi_ix_cols = [self.season_grouping_col, ta_class_id_col, ustar_subclass_id_col]
        _multi_ix_df = means_of_group_subclass_df[multi_ix_cols]  # df only used for creating MultiIndex
        _multi_ix = pd.MultiIndex.from_frame(df=_multi_ix_df)
        means_of_group_subclass_df.set_index(_multi_ix, inplace=True)
        means_of_group_subclass_df.drop(multi_ix_cols, axis=1, inplace=True)  # Remove data cols that are in MultiIndex

        return means_of_group_subclass_df

    def add_thres_cols(self, df):
        df[self.thres_year_col] = np.nan
        # df[self.thres_season_col] = np.nan
        # df[self.thres_class_col] = np.nan
        return df

    def detect_class_thresholds(self, season_subclass_avg_df, ta_class_id_col, season_key):
        """Calculate ustar thresholds in air temperature classes

        Detection is done for each USTAR subclass in each TA class separately
        for each season.

        Analysis is done using the pandas .groupby method. In addition, a
        MultiIndex DataFrame that contains the subclass averages within each class
        needs to be accessed. Since the index of the DataFrame is a MultiIndex,
        it can be accessed via tuples, e.g. SEASON 3 with TA class 0 and with
        USTAR subclass 4 is accessed with the tuple (3,0,4). Yes, this is a bit tricky,
        but also very useful, isn't it.

        This analysis could also have been done in an earlier step, but it
        is useful to have all results in one DataFrame.

        """
        elements.btn_txt_live_update(btn=self.btn_run, txt='Calculating USTAR thresholds ...', perc=-9999)

        # Create df copy to work with
        df = season_subclass_avg_df.copy()
        df[self.thres_class_col] = np.nan  # Add column for class threhold
        # Reset_index is needed to avoid duplicates in index and columns during grouping
        # df.reset_index(drop=True, inplace=True)

        # Loop TA classes and calculate the ustar threshold in each class
        class_grouped = df.groupby(ta_class_id_col)
        for class_key, class_group_df in class_grouped:

            # SUBCLASSES
            # ----------
            num_subclasses = len(class_group_df)
            for cur_subclass_ix in list(range(0, num_subclasses)):
                # Uses the refined method of Pastorello et al. (2020)
                cur_flux = class_group_df.iloc[cur_subclass_ix][self.flux_data_col]  # Current subclass flux
                cur_ustar = class_group_df.iloc[cur_subclass_ix][self.ustar_data_col]
                nxt_subclass = cur_subclass_ix + 1
                nxt_flux = class_group_df.iloc[nxt_subclass][self.flux_data_col]  # Next flux
                nxt_nxt_subclass = cur_subclass_ix + 2

                # Calculate percentage of current subclass flux in comparison to following subclasses
                if nxt_nxt_subclass < num_subclasses:
                    # In comparison to mean of next 10 subclasses
                    cur_following_mean = \
                        class_group_df.iloc[nxt_subclass:nxt_subclass + 10][self.flux_data_col].mean()
                    cur_flux_perc = (cur_flux / cur_following_mean) * 100
                    # Check also next subclass in comparison to its next 10 subclasses
                    nxt_following_mean = \
                        class_group_df.iloc[nxt_nxt_subclass:nxt_nxt_subclass + 10][self.flux_data_col].mean()
                    nxt_flux_perc = (nxt_flux / nxt_following_mean) * 100
                else:
                    cur_following_mean = cur_flux_perc = '-no-more-subclasses-'
                    nxt_following_mean = nxt_flux_perc = '-no-more-next-subclasses-'

                self.ustar_subclass_info(cur_season=season_key, cur_class=class_key,
                                         cur_subclass=cur_subclass_ix, cur_ustar=cur_ustar,
                                         cur_flux=cur_flux, cur_following_mean=cur_following_mean,
                                         cur_flux_perc=cur_flux_perc, nxt_flux=nxt_flux,
                                         nxt_following_mean=nxt_following_mean, nxt_flux_perc=nxt_flux_perc)

                if nxt_nxt_subclass == num_subclasses:
                    # When the last subclass is reached, stop for loop
                    print(f"    *END* Last USTAR subclass {cur_subclass_ix} reached, moving to next class.")
                    break

                # Check current flux and mean of 10 following
                if cur_flux_perc > self.flux_plateau_thres_perc:
                    print(f"SUCCESS ... current flux is {cur_flux_perc:.1f}% of 10-following-mean")
                else:
                    print(f"(!)FAILED ... current flux is {cur_flux_perc:.1f}% of 10-following-mean")
                    continue

                # Check next flux and mean of its 10 following
                if self.flux_plateau_method == 'NEE > 10+10 Higher USTAR Subclasses':
                    if nxt_flux_perc > self.flux_plateau_thres_perc:
                        print(f"SUCCESS ... next flux is {nxt_flux_perc:.1f}% of next-10-following-mean")
                    else:
                        print(f"(!)FAILED ... next flux is {nxt_flux_perc:.1f}% of next-10-following-mean")
                        continue

                # Check if correlation b/w TA and RH below threshold
                print("Testing correlation between TA and RH ...")
                abs_corr = abs(class_group_df[self.ustar_data_col].corr(class_group_df[self.ta_data_col]))
                if abs_corr < self.ta_ustar_corr_thres:
                    print(f"SUCCESS ... abs correlation {abs_corr:.2f} < {self.ta_ustar_corr_thres:.2f}")
                else:
                    print(f"(!)FAILED ... abs correlation {abs_corr:.2f} >= {self.ta_ustar_corr_thres:.2f}")
                    continue

                # Stop for loop once a ustar threshold was found
                df.loc[self.ix_slice[season_key, class_key, :], self.thres_class_col] = \
                    cur_ustar
                print(f"USTAR threshold set to {cur_ustar:.3f} m s-1")
                print("\n\n")
                break

        return df

    def set_season_thresholds(self, season_subclass_avg_df, season_key):
        """Set threshold for current season, based on class thresholds."""

        elements.btn_txt_live_update(btn=self.btn_run, txt='Setting season threshold ...', perc=-9999)

        df = season_subclass_avg_df.copy()
        df[self.thres_season_col] = np.nan

        # Season threshold is calculated from the class thresholds.
        all_ustar_thres_classes = \
            df.loc[self.ix_slice[season_key, :, :],
                   self.thres_class_col].dropna().unique()

        # Check if array has contents (i.e. not emtpy)
        if all_ustar_thres_classes.size:
            if self.define_bts_season_thres == 'Median of Class Thresholds':
                ustar_thres_season = np.median(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Mean of Class Thresholds':
                ustar_thres_season = np.mean(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Maximum of Class Thresholds':
                ustar_thres_season = np.max(all_ustar_thres_classes)
            elif self.define_bts_season_thres == 'Minimum of Class Thresholds':
                ustar_thres_season = np.min(all_ustar_thres_classes)
            else:
                ustar_thres_season = np.median(all_ustar_thres_classes)
        else:  # if empty
            ustar_thres_season = np.nan

        df.loc[self.ix_slice[season_key, :, :],
               self.thres_season_col] = ustar_thres_season

        return df, ustar_thres_season

        # # OVERALL THRESHOLD
        # # -----------------
        # # The overall threshold is the max of the season thresholds todo per year?
        # if df[self.thres_season_col].dropna().empty:
        #     # If Series is empty, this means that no USTAR threshold was found in any
        #     # of the seasons. Following Papale et al. (2006), the threshold for the year
        #     # is then set to the 90th percentile of the USTAR data.
        #     ustar_thres_overall = season_df[self.ustar_data_col].quantile(self.set_to_ustar_data_percentile)
        #
        # else:
        #     ustar_thres_overall = self.get_ustar_overall_thres(method=self.define_overall_thres,
        #                                                        df=df,
        #                                                        ustar_thres_season_col=self.thres_season_col)
        #
        # # Finally, check if the overall USTAR threshold is below the allowed limit
        # if ustar_thres_overall > self.allowed_min_ustar_thres:
        #     pass
        # else:
        #     ustar_thres_overall = self.allowed_min_ustar_thres
        #
        # df.loc[self.ix_slice[:, :, :], self.thres_year_col] = ustar_thres_overall

        # return df, self.thres_year_col, self.thres_season_col, self.thres_class_col

    def get_ustar_overall_thres(self, method, df, ustar_thres_season_col):
        if method == 'Maximum Of Season Thresholds':
            ustar_thres_overall = df[ustar_thres_season_col].max()
        elif method == 'Minimum Of Season Thresholds':
            ustar_thres_overall = df[ustar_thres_season_col].min()
        elif method == 'Median Of Season Thresholds':
            ustar_thres_overall = df[ustar_thres_season_col].median_col()
        elif method == 'Mean Of Season Thresholds':
            ustar_thres_overall = df[ustar_thres_season_col].mean()
        else:  # default max
            ustar_thres_overall = df[ustar_thres_season_col].max()
        return ustar_thres_overall

    def ustar_subclass_info(self, cur_season, cur_class, cur_subclass, cur_flux, cur_ustar, cur_following_mean,
                            nxt_following_mean, nxt_flux, cur_flux_perc, nxt_flux_perc):
        if cur_following_mean == '-no-more-subclasses-':
            cur_following_mean = cur_flux_perc = nxt_following_mean = nxt_flux_perc = nxt_flux = -9999

        try:
            print(f"\n\n[SEASON] {cur_season}  "
                  f"[TA CLASS] {cur_class}  "
                  f"[USTAR SUBCLASS] {cur_subclass}  "
                  f"USTAR  {cur_ustar:.2f}  "
                  f"FLUX  {cur_flux:.2f}  "
                  f"FLUX following mean {cur_following_mean:.2f}  "
                  f"FLUX/MEAN {cur_flux_perc:.2f}%  "
                  f"(NEXT) FLUX {nxt_flux:.2f}  "
                  f"(NEXT) FLUX following mean {nxt_following_mean:.2f}  "
                  f"(NEXT) FLUX/MEAN {nxt_flux_perc:.2f}  "
                  )
        except ValueError:
            print("-ValueError-")

    def check_season_info(self, data_df):
        """Add season and season type info to data in case none is available,
        i.e. 'None' was selected in the 'Seasons' drop-down menu.

        This additional info is needed to perform data calculations always the
        same way, regardless if already available seasons are used of no seasons
        are used.
        """

        # ADD SEASON INFO
        # ---------------
        if not self.season_grouping_col:
            # Create season column in case none is available. This basically defines all data
            # as one season and the data pool for threshold calcs is the whole dataset.
            # Doing it this way makes data handling of seasons more harmonized.
            self.season_col = ('GRP_SEASON_0', '[#]')
            self.season_type_col = ('GRP_SEASON_TYPE_0', '[#]')
            data_df.loc[:, self.season_col] = 0
            data_df.loc[:, self.season_type_col] = 0
            self.season_grouping_col = self.season_col  # All data is basically one single season
        else:
            # The selected column will be used for season grouping.
            pass

        # else:
        #     # Use already available season column.
        #     data_df.loc[:, self.season_type_col] = self.data_df[self.season_type_col].copy()
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_type_col].copy()

        # if self.season_data_pool == 'Season Type':
        #     # Calculate one threshold per season type, e.g. one for all summers. In this case,
        #     # all data from all e.g. summers are first pooled, and then one single threshold is
        #     # calculated based on the pooled data. The threshold is then valid for all summers across
        #     # all years.
        #     data_df.loc[:, self.season_data_pool_col] = self.data_df[self.season_grouping_col].copy()
        #
        # elif self.season_data_pool == 'Season':
        #     # Calculate threshold for each season, e.g. summer 2018, summer 2019, etc. To differentiate
        #     # between multiple summers, the year is added as additional information so the seasons
        #     # can be grouped later during calcs.
        #     data_df['year_aux'] = data_df.index.year.astype(str)
        #     data_df.loc[:, self.season_data_pool_col] = \
        #         data_df['year_aux'] + '_' + data_df.loc[:, self.season_data_pool_col].astype(str)
        #     data_df.drop('year_aux', axis=1, inplace=True)

        return data_df

    def make_ustar_df(self):
        """Assemble df that contains all needed data cols for analyses. A separate
        df is created that contains only the nighttime data of data columns that
        are needed for threshold calculations.
        """

        flux_pretty_ix = self.col_list_pretty.index(self.flux_data_col)
        ta_pretty_ix = self.col_list_pretty.index(self.ta_data_col)
        ustar_pretty_ix = self.col_list_pretty.index(self.ustar_data_col)
        nighttime_pretty_ix = self.col_list_pretty.index(self.nighttime_col)

        flux_data_col = self.col_dict_tuples[flux_pretty_ix]
        ta_data_col = self.col_dict_tuples[ta_pretty_ix]
        ustar_data_col = self.col_dict_tuples[ustar_pretty_ix]
        nighttime_col = self.col_dict_tuples[nighttime_pretty_ix]

        ustar_df_cols = [flux_data_col, ta_data_col, ustar_data_col, nighttime_col,
                         self.season_grouping_col]

        # Daytime and nighttime data, df collects fullres results per bts run
        ustar_daynight_df = self.data_df[ustar_df_cols]
        ustar_daynight_df[self.out_thres_season_col] = np.nan
        ustar_daynight_df[self.out_qcflag_season_col] = np.nan
        ustar_daynight_df[self.out_thres_year_col] = np.nan
        ustar_daynight_df[self.out_qcflag_year_col] = np.nan

        # Keep only nighttime values for nighttime df, keep only respiration
        ustar_night_df = ustar_daynight_df[ustar_daynight_df[nighttime_col] < self.nighttime_thres]

        return ustar_daynight_df, ustar_night_df, flux_data_col, ta_data_col, ustar_data_col, nighttime_col

    def assign_classes(self, season_df, class_col, subclass_col, num_classes, num_subclasses):
        """
        Generate TA classes for this season and generate USTAR subclasses for each TA class.

        :param season_df: full resolution dataframe
        :param class_col: column name to create classes
        :param subclass_col: column name to create subclasses
        :param num_classes: number of classes
        :param num_subclasses: number of subclasses
        :return:
        """
        elements.btn_txt_live_update(btn=self.btn_run, txt='Assigning classes ...', perc=-9999)
        df = season_df.copy()

        # Insert new class and subclass columns in df
        ta_class_col = ('{}_CLASS'.format(class_col[0]), '[#]')
        ustar_subclass_col = ('{}_SUBCLASS'.format(subclass_col[0]), '[#]')
        df[ta_class_col] = np.nan
        df[ustar_subclass_col] = np.nan

        # TA CLASSES
        # ----------
        # Divide season TA data into q classes of TA.
        # Quantile-based discretization function, the fact that .qcut exists is beautiful.
        class_S = pd.qcut(df[class_col], q=num_classes, labels=False, duplicates='drop')  # Series
        df[ta_class_col] = df[ta_class_col].combine_first(class_S)  # Replace NaN w/ class number

        # USTAR SUBCLASSES
        # ----------------
        class_grouped_df = df.groupby(ta_class_col)  # Group by TA class
        for class_key, class_df in class_grouped_df:  # Loop through data of each TA class
            class_S = pd.qcut(class_df[subclass_col], q=num_subclasses, labels=False, duplicates='drop')  # Series
            df[ustar_subclass_col] = df[ustar_subclass_col].combine_first(class_S)  # replaces NaN w/ class number

        return df, ta_class_col, ustar_subclass_col

    def get_colname_tuples(self, colname_pretty):
        if colname_pretty == 'None':
            colname_tuple = False
        else:
            col_pretty_ix = self.col_list_pretty.index(colname_pretty)  # get ix in pretty list
            colname_tuple = self.col_dict_tuples[col_pretty_ix]  # use ix to get col name as tuple
        return colname_tuple

    def plot_season_data_with_class_colors(self, fullres_df_with_classes, data_col, ta_class_col,
                                           ax, class_colors_dict, current_season_data_pool_key, timestamp_col):

        ax.clear()
        # Datetime index needed for time series plots
        fullres_df_with_classes = fullres_df_with_classes.set_index(timestamp_col)

        # AIR TEMPERATURE GROUPS
        # ----------------------
        # Classes of TA
        class_grouped_df = fullres_df_with_classes.groupby(ta_class_col)
        for class_key, class_group in class_grouped_df:
            ax.plot_date(x=class_group.index,
                         y=class_group[data_col],
                         c=class_colors_dict[class_key],
                         alpha=0.5,
                         markeredgecolor='none',
                         marker=self.markers[0],
                         ms=SIZE_MARKER_DEFAULT)

        gui.plotfuncs.default_format(ax=ax, txt_xlabel='-', txt_ylabel=data_col[0], txt_ylabel_units=data_col[1],
                                     label_color='black', fontsize=FONTSIZE_LABELS_AXIS)

        ax.text(0.05, 0.95, f"Season {current_season_data_pool_key} / {data_col[0]} per {ta_class_col[0]}",
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,
                size=FONTSIZE_HEADER_AXIS, color='black', backgroundcolor='none')

        # Automatic nice ticks
        gui.plotfuncs.nice_date_ticks(ax=ax, minticks=5, maxticks=15, which='x')

    def plot_per_season(self, season_subclass_avg_df, ax_dict_id, x_col, y_col, ta_class_col, class_color_dict,
                        thres_class_col, thres_season_col, scalar_ix, current_season_data_pool_key):
        """Plot subclass averages per season."""

        # season_subclass_avg_df = season_subclass_avg_df.reset_index(inplace=False, drop=True)  # Remove index for grouping
        ax = self.axes_dict[ax_dict_id]
        ax.clear()

        # AIR TEMPERATURE CLASSES
        # -----------------------
        # Classes of TA
        class_grouped = season_subclass_avg_df.groupby(ta_class_col)
        color_class = -1
        for class_key, class_group_df in class_grouped:
            color_class += 1
            x = class_group_df[x_col]
            y = class_group_df[y_col]

            counts_col = (y_col[0] + '_COUNTS', y_col[1] + '_COUNTS')
            min_counts = int(class_group_df[counts_col].min())
            max_counts = int(class_group_df[counts_col].max())

            ax.scatter(x=x, y=y,
                       c=class_color_dict[color_class],
                       alpha=0.9,
                       edgecolors='none',
                       marker=self.markers[0],
                       s=SIZE_MARKER_LARGE,
                       label=f'season: {current_season_data_pool_key}, class: {class_key:.0f},'
                             f' {min_counts}-{max_counts} vals per bin')

            SD_col = (y_col[0] + '_SD', y_col[1] + '_SD')
            ax.errorbar(x=x, y=y,
                        yerr=class_group_df[SD_col], alpha=0.2,
                        color=class_color_dict[color_class], ls='none',
                        elinewidth=2)

            # Show subclass threshold
            # Mark data points where a ustar threshold was found
            if scalar_ix == 1:
                ustar_thres_class = class_group_df[thres_class_col].unique()[0]
                if np.isnan(ustar_thres_class):
                    pass
                else:
                    yy = class_group_df.loc[
                        class_group_df[self.ustar_data_col] == ustar_thres_class, self.flux_data_col]
                    ax.scatter(x=ustar_thres_class, y=yy, edgecolors='black', c='none',
                               alpha=0.9,
                               marker=self.markers[0],
                               s=SIZE_MARKER_LARGE)

            # Show legend only for first scalar, also valid for other two scalars
            if scalar_ix == 0:
                min_counts = int(season_subclass_avg_df[counts_col].min())
                max_counts = int(season_subclass_avg_df[counts_col].max())
                ax.text(0.95, 0.95, "{}-{} values / avg".format(min_counts, max_counts),
                        horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes, size=FONTSIZE_INFO_DEFAULT,
                        color=FONTCOLOR_LABELS_AXIS, backgroundcolor='none')

                gui.plotfuncs.default_legend(ax=ax, loc='lower right', bbox_to_anchor=(1, 0))

            # ax.axvline(x=class_ustar_thres, color=class_color_dict[class_key], ls='--', zorder=99)

        # Show additional info in ustar plot (scalar index 1)
        if scalar_ix == 1:
            # Show season threshold in ustar plot (scalar index 1)
            ustar_thres_season = season_subclass_avg_df[thres_season_col].unique()[0]
            ax.text(ustar_thres_season, 0.97, "{:.3f}".format(ustar_thres_season),
                    horizontalalignment='center', verticalalignment='top',
                    transform=ax.get_xaxis_transform(), size=FONTSIZE_INFO_DEFAULT * 0.7,
                    color='black', backgroundcolor=yellow(500), zorder=100)
            ax.axvline(x=ustar_thres_season, color=yellow(500), ls='-',
                       lw=WIDTH_LINE_SPINES * 10, alpha=0.4, zorder=0)  # red 900

            # Add info about how threshold was found
            ax.text(0.95, 0.95, f"Threshold per season: {self.define_bts_season_thres}",
                    horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes, size=FONTSIZE_INFO_DEFAULT,
                    color=FONTCOLOR_LABELS_AXIS, backgroundcolor='none')

    def sample_data(self, df):
        """Draw samples from data (bootstrapping)."""
        if self.num_bootstrap_runs == 0:
            # w/o bootstrapping, the data are simply not sampled
            ustar_fullres_df_sample = df.copy()
        else:
            num_rows = df.shape[0]
            ustar_fullres_df_sample = df.sample(n=int(num_rows), replace=True)
            ustar_fullres_df_sample.sort_index(inplace=True)
        return ustar_fullres_df_sample

# def format_ustar_thres_overall_plot(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#
#     ax.tick_params(axis='both', labelbottom=False, labelleft=False,
#                    bottom=False, left=False)
#
#     # x_axis = ax.axes.get_xaxis()
#     # x_label = x_axis.get_label()
#     # x_label.set_visible(False)
#     # y_axis = ax.axes.get_yaxis()
#     # y_label = y_axis.get_label()
#     # y_label.set_visible(False)
#     return None

# def format_season_plots(season_key, ax, x_col, y_col, label_color, fontsize, df):
#     # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params
#
#     x_axis = ax.axes.get_xaxis()
#     x_label = x_axis.get_label()
#     y_axis = ax.axes.get_yaxis()
#     y_label = y_axis.get_label()
#
#     # Hide / show axis labels and tick labels
#     if season_key == 1:
#         # x_label.set_visible(False)
#         # y_label.set_visible(False)
#         # ax.set_ylabel('{} {}'.format(y_col[0], y_col[1]), color=label_color, fontsize=fontsize)
#         ax.tick_params(axis='both', labelbottom=True, labelleft=True)
#
#     elif season_key == 2:
#         # x_label.set_visible(False)
#         # y_label.set_visible(False)
#         ax.tick_params(axis='both', labelbottom=True, labelleft=True)
#
#     elif season_key == 3:
#         ax.set_xlabel('{} ({})'.format(x_col[0], x_col[1]), color=label_color, fontsize=fontsize)
#         # ax.set_ylabel('{} {}'.format(y_col[0], y_col[1]), color=label_color, fontsize=fontsize)
#         ax.tick_params(axis='both', labelbottom=True, labelleft=True)
#
#     elif season_key == 4:
#         ax.set_xlabel('{} ({})'.format(x_col[0], x_col[1]), color=label_color, fontsize=fontsize)
#         # y_label.set_visible(False)
#         ax.tick_params(axis='both', labelbottom=True, labelleft=True)
#
#     # if season_key == 1:
#     counts_col = (y_col[0] + '_COUNTS', y_col[1] + '_COUNTS')
#     min_counts = int(df[counts_col].min())
#     max_counts = int(df[counts_col].max())
#
#     ax.text(0.95, 0.95, "{}-{} values / avg".format(min_counts, max_counts),
#             horizontalalignment='right', verticalalignment='top',
#             transform=ax.transAxes, size=FONTSIZE_INFO_DEFAULT,
#             color=FONTCOLOR_LABELS_AXIS, backgroundcolor='none')

# def get_ustar_params(self):
#     params = {
#         'col_dict_tuples': self.selected_file_colnames_tuples_dict,
#         'data_df': self.data_df,
#         'fig': self.fig_an_ref_uStarDet,
#         'FLUX_data_col': self.drp_an_ref_uStarDet_flux.currentText(),
#         'FLUX_perc_thres': float(self.lne_an_ref_uStarDet_fluxPercThres.text()),
#         'TA_data_col': self.drp_an_ref_uStarDet_TA.currentText(),
#         'TA_numClasses': int(self.lne_an_ref_uStarDet_numClasses_TA.text()),
#         'USTAR_data_col': self.drp_an_ref_uStarDet_uStar.currentText(),
#         'USTAR_numClasses': int(self.lne_an_ref_uStarDet_numClasses_uStar.text()),
#         'col_list_pretty': self.selected_file_colnames_pretty_strings_list,
#         'defineNighttimeAs': self.drp_an_ref_uStarDet_defineNighttime.currentText(),
#         'axes_dict': self.axesDict_an_ref_uStarDet,
#         'seasons_type': self.drp_an_ref_uStarDet_seasons.currentText(),
#         'define_overall_thres': self.drp_an_ref_uStarDet_defineOverallThres.currentText(),
#         'define_season_thres': self.drp_an_ref_uStarDet_defineSeasonThres.currentText(),
#         'ta_ustar_corr_thres': float(self.lne_an_ref_uStarDet_TAvsUSTAR_corrThres.text()),
#         'set_to_ustar_data_percentile': float(self.lne_an_ref_uStarDet_setToUstarDataPercentile.text()),
#         'allowed_min_ustar_thres': float(self.lne_an_ref_uStarDet_USTAR_allowedMinThres.text()),
#         'num_bootstrap_runs': int(self.lne_an_ref_uStarDet_numBootstrapRuns.text())
#     }
#     return params

# def update_refinements_with_template(self, template):
#     if template == 'Custom':
#         pass
#
#     # elif template == 'Reichstein et al. (2005)':
#     #     self.lne_TA_numClasses.setText('6')
#     #     self.lne_uStar_numClasses.setText('20')
#     #     # self.drp_defineNighttime == 'Short-Wave Radiation < 20 W m-2'
#     #     self.drp_seasons.setCurrentText('Four 3-month seasons')  ## check if works
#     #     self.lne_flux_plateau_thres_perc.setText('95')
#     #     self.drp_define_overall_thres.setCurrentText('Maximum Of Season Thresholds')
#     #     self.drp_define_season_thres.setCurrentText('Median of Class Thresholds')
#
#     elif template == 'Papale et al. (2006)':
#         self.lne_num_bootstrap_runs.setText('100')
#         self.lne_TA_numClasses.setText('6')
#         self.lne_uStar_numClasses.setText('20')
#         # self.drp_defineNighttime == 'Short-Wave Radiation < 20 W m-2'
#         self.drp_seasons.setCurrentText('Four 3-month seasons: DJF, MAM, JJA, SON')
#         self.lne_flux_plateau_thres_perc.setText('99')
#         self.drp_define_overall_thres.setCurrentText('Maximum Of Season Thresholds')
#         self.drp_define_season_thres.setCurrentText('Median of Class Thresholds')
#         self.lne_set_to_ustar_data_percentile.setText('0.9')
#         self.lne_ta_ustar_corr_thres.setText('0.4')
#         self.lne_allowed_min_ustar_threshold.setText('0.01')  # for short vegetation

# # Group by season data pools
# season_grouped = self.results_df.groupby(self.season_data_pool_col)
# for season_key, season_df in season_grouped:
#
#     # Get threshold for season
#     # Since we already grouped by data pools and then by seasons, there must be
#     # only one unique value in the season threshold column in season_df: the
#     # threshold for this data pool and this season.
#     this_season_thres = season_df[self.thres_season_col].unique()
#     this_year_thres = season_df[self.thres_year_col].unique()
#     if len(this_season_thres) == 1:
#
#         print(f"data pool: {season_key}    season: {season_key}    "
#               f"season ustar: {this_season_thres[0]}")
#
#         dtpl_season_filter = data_df[self.season_data_pool_col] == season_key
#         data_df.loc[dtpl_season_filter, self.out_season_thres_col] = this_season_thres[0]
#         data_df.loc[dtpl_season_filter, self.out_year_thres_col] = this_year_thres[0]
#     else:
#         print('-season-threshold-not-unique-')

# # Flag, overall thresholds per data pool
# data_df.loc[data_df[self.ustar_data_col] > data_df[self.out_year_thres_col],
#             self.out_year_qcflag_col] = 0  # Good data
# data_df.loc[data_df[self.ustar_data_col] < data_df[self.out_year_thres_col],
#             self.out_year_qcflag_col] = 2  # Hard flag
