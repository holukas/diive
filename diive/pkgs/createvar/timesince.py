"""
Output time since (number of values) last occurrence

For example, number of values since:
    - the last time rain was recorded
    - air temperature was above 20°C

"""
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PyQt5 import QtGui

import gui.base
import gui.elements
import gui.plotfuncs
import logger
from help import abbreviations
from modboxes.plots.styles.LightTheme import *
from pkgs.dfun.frames import export_to_main


class addContent(gui.base.buildTab):
    """Build new tab and populate with contents"""

    def __init__(self, app_obj, title, tab_id):
        super().__init__(app_obj, title, tab_id, tab_template='SVP')

        # Tab icon
        self.TabWidget.setTabIcon(self.tab_ix, QtGui.QIcon(app_obj.ctx.tab_icon_create_variable))

        # Add settings menu contents
        self.lne_upper_limit, self.lne_lower_limit, self.btn_calc, self.btn_add_as_new_var = \
            self.add_settings_fields()

        # # Add variables required in settings
        # self.populate_settings_fields()

        # Add available variables to variable lists
        gui.base.buildTab.populate_variable_list(obj=self)

        # # Add plots
        # self.populate_plot_area()

    # def populate_plot_area(self):
    #     self.axes_dict = self.add_axes()

    # def populate_settings_fields(self):
    #     pass

    # def add_axes(self):
    #     gs = gridspec.GridSpec(1, 1)  # rows, cols
    #     gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
    #     ax_main = self.fig.add_subplot(gs[0, 0])
    #     axes_dict = {'ax_main': ax_main}
    #     for key, ax in axes_dict.items():
    #         gui.plotfuncs.default_format(ax=ax, txt_xlabel=False)
    #     return axes_dict

    def add_settings_fields(self):
        gui.elements.add_header_to_grid_top(layout=self.sett_layout, txt='Settings')

        # Based on
        lne_upper_limit = gui.elements.add_label_linedit_pair_to_grid(txt='Smaller Than',
                                                                      css_ids=['', 'cyan'],
                                                                      layout=self.sett_layout,
                                                                      orientation='horiz',
                                                                      row=1, col=0)

        lne_lower_limit = gui.elements.add_label_linedit_pair_to_grid(txt='Larger Than',
                                                                      css_ids=['', 'cyan'],
                                                                      layout=self.sett_layout,
                                                                      orientation='horiz',
                                                                      row=2, col=0)

        btn_calc = gui.elements.add_button_to_grid(grid_layout=self.sett_layout,
                                                   txt='Calculate Time Since', css_id='',
                                                   row=3, col=0, rowspan=1, colspan=2)

        btn_add_as_new_var = gui.elements.add_button_to_grid(grid_layout=self.sett_layout,
                                                             txt='+ Add timesince Variable', css_id='',
                                                             row=4, col=0, rowspan=1, colspan=2)

        self.sett_layout.setRowStretch(5, 1)
        return lne_upper_limit, lne_lower_limit, btn_calc, btn_add_as_new_var


class Run(addContent):
    target_loaded = False
    ready_to_export = False
    marker_filter = None  # Filter to mark detected outliers

    class_id = abbreviations.create_variable_time_since
    sub_outdir = "create_var_time_since"

    def __init__(self, app_obj, title, tab_id):
        super().__init__(app_obj, title, tab_id)
        logger.log(name='>>> Starting Outlier Removal: Absolute Limit', dict={}, highlight=True)  # Log info
        self.sub_outdir = self.project_outdir / self.sub_outdir
        self.class_df = pd.DataFrame()
        self.target_col = None
        # self.set_colnames()
        gui.base.buildTab.update_btn_status(obj=self, target=True, marker=False, export=True)
        self.axes_dict = self.make_axes_dict()

    def select_target(self):
        """Select target var from list"""
        self.timesince_var_exists = False
        self.set_target_col()
        self.class_df = self.init_class_df()
        # self.init_new_cols()
        self.update_fields()
        self.plot_data()
        self.target_loaded = True
        self.ready_to_export = False
        gui.base.buildTab.update_btn_status(obj=self, target=True, marker=False, export=True)

    def calc(self):
        # Needed in case executed several times in a row, reset aux cols:
        self.class_df = self.class_df[[self.target_col]]
        self.set_colnames()
        self.init_new_cols()
        upper_lim, lower_lim = self.get_settings_from_fields()
        self.calc_upper_lower_lim(upper_lim=upper_lim, lower_lim=lower_lim)
        self.generate_timesince_var()
        self.timesince_var_exists = True
        self.mark_in_plot()
        self.ready_to_export = True
        gui.base.buildTab.update_btn_status(obj=self, target=True, marker=False, export=True)

    def init_class_df(self):
        return self.tab_data_df[[self.target_col]].copy()

    def init_new_cols(self):
        self.class_df[self.timesince_col] = np.nan
        self.class_df[self.upper_lim_col] = np.nan
        self.class_df[self.lower_lim_col] = np.nan

    def set_target_col(self):
        """Get column name of target var from var list"""
        target_var = self.lst_varlist_available.selectedIndexes()
        target_var_ix = target_var[0].row()
        self.target_col = self.col_dict_tuples[target_var_ix]

    def set_colnames(self):
        # Marker for values in the outlier range
        self.timesince_col = (f"{self.target_col[0]}{self.class_id}", f"[records_since]")
        self.upper_lim_col = (f"_upper_limit", '[aux]')
        self.lower_lim_col = (f"_lower_limit", '[aux]')
        self.flag_col = (f"_is_outside_range", '[1=True]')

    def update_fields(self):
        self.lne_upper_limit.setText(str(self.class_df[self.target_col].quantile(1)))
        self.lne_lower_limit.setText(str(self.class_df[self.target_col].quantile(.9)))

    def get_settings_from_fields(self):
        upper_lim = float(self.lne_upper_limit.text())
        lower_lim = float(self.lne_lower_limit.text())
        return upper_lim, lower_lim

    def calc_upper_lower_lim(self, upper_lim, lower_lim):
        self.class_df[self.upper_lim_col] = upper_lim
        self.class_df[self.lower_lim_col] = lower_lim

    def generate_timesince_var(self):
        """Detect all values that are within the specified limit range, use 0/1 to mark values."""
        filter_inrange = (self.class_df[self.target_col] > self.class_df[self.lower_lim_col]) & \
                         (self.class_df[self.target_col] < self.class_df[self.upper_lim_col])
        self.class_df.loc[filter_inrange, self.flag_col] = 0  # Inside range
        self.class_df.loc[~filter_inrange, self.flag_col] = 1  # Outside range, note: this also counts NaNs as 1

        # Set all NaN values to 1
        # OLD: Set all NaN values to 0, necessary for correct summations of values outside range
        # OLD: Otherwise, time periods with gaps would also be counted as "outside range", i.e. 1.
        self.class_df.loc[self.class_df[self.target_col].isnull(), self.flag_col] = 1

        # fantastic: https://stackoverflow.com/questions/27626542/counting-consecutive-positive-value-in-python-array
        y = self.class_df[self.flag_col]
        yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

        self.class_df.loc[:, self.timesince_col] = yy

    def mark_in_plot(self):
        # self.marker_filter = self.class_df[self.qcflag_col] == True
        self.ready_to_export = False
        gui.base.buildTab.update_btn_status(obj=self, target=True, marker=False, export=True)
        self.plot_data()

    def prepare_export(self):
        return self.class_df[[self.timesince_col]].copy()

    def get_selected(self, main_df):
        """Return modified class data back to main data"""
        export_df = self.prepare_export()
        main_df = export_to_main(main_df=main_df,
                                 export_df=export_df,
                                 tab_data_df=self.tab_data_df)  # Return results to main
        return main_df

    def plot_data(self):
        # Delete all axes in figure
        for ax in self.fig.axes:
            self.fig.delaxes(ax)

        axes_dict = self.make_axes_dict()

        legend_lns = []
        for key, ax in axes_dict.items():
            if key == 'ax_main':
                col = self.target_col
                color = '#546E7A'
                bg_color = '#90A4AE'
                title = "Target Variable"
            elif key == 'ax_timesince':
                if self.timesince_var_exists:  # Check if var already exists
                    col = self.timesince_col
                    color = '#4DB6AC'
                    bg_color = '#B2DFDB'
                    title = "Time Since"
                else:
                    continue
            else:
                col = color = bg_color = title = '-col-not-found-'

            ax.set_title(title, loc='left', fontdict={'fontweight': 'bold',
                                                      'fontsize': FONTSIZE_HEADER_AXIS_LARGE})

            ax.plot_date(x=self.class_df.index, y=self.class_df[col],
                         color=color, alpha=1, ls='-',
                         marker='o', markeredgecolor='none', ms=4, zorder=98, label=f"{col[0]} {col[1]}")

            ax.text(0.01, 0.97, f"{col[0]}    {col[1]}", weight='bold',
                    size=FONTSIZE_HEADER_AXIS_LARGE, color=FONTCOLOR_HEADER_AXIS,
                    backgroundcolor=bg_color, transform=ax.transAxes, alpha=1,
                    horizontalalignment='left', verticalalignment='top', zorder=99)

            ax.text(0.97, 0.03, f"MEAN: {self.class_df[col].mean():.2f}", weight='bold',
                    size=FONTSIZE_HEADER_AXIS_LARGE, color=FONTCOLOR_HEADER_AXIS,
                    backgroundcolor=bg_color, transform=ax.transAxes, alpha=1,
                    horizontalalignment='right', verticalalignment='bottom', zorder=99)

            gui.plotfuncs.default_grid(ax=ax)
            gui.plotfuncs.default_legend(ax=ax, from_line_collection=False, line_collection=legend_lns)

            locator = mdates.AutoDateLocator(minticks=3, maxticks=30)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def make_axes_dict(self):
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        ax_main = self.fig.add_subplot(gs[0, 0])
        ax_timesince = self.fig.add_subplot(gs[1, 0], sharex=ax_main)
        axes_dict = {'ax_main': ax_main, 'ax_timesince': ax_timesince}
        for key, ax in axes_dict.items():
            gui.plotfuncs.default_format(ax=ax, txt_xlabel=False)
        return axes_dict
