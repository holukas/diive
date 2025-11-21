import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas
import pandas as pd

import selfheating_newcols


class ScalingFactors(selfheating_newcols.NewCols):

    def __init__(self, plot_df, outplot, classvar_col, userconfig):
        self.userconfig = userconfig
        self.plot_df = plot_df
        self.classvar_col = classvar_col
        self.axes_dict, self.fig = self.prepare_axes()
        self.outplot = outplot
        self.run()

    def plot_scatter(self, ax):
        marker_cycle = {'day': 'o', 'night': 's'}
        color_cycle = {'day': '#FF9800',
                       'night': '#3F51B5'}  # Orange 500 / Indigo 500; https://www.materialui.co/colors

        xmax = ymax = -9999
        xmin = ymin = 9999

        # Separate groups for daytime and nighttime data
        _daynight_grouped = self.plot_df.groupby('DAYTIME')
        for _daynight_group_ix, _daynight_group_df in _daynight_grouped:
            _time = 'day' if _daynight_group_ix == 1 else 'night'
            ax.plot(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_MEDIAN'],
                    color=color_cycle[_time],
                    alpha=1, ls='-',
                    marker=marker_cycle[_time], markeredgecolor='none', ms=4, zorder=98,
                    label=f'{_time}: median from {self.pm_num_bootstrap_runs}x bootstrap, '
                          f'incl. 1-99% and interquartile range ({_daynight_group_df["NUMVALS_AVG"].mean():.0f} values per data point)')
            ax.fill_between(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_Q99'],
                            _daynight_group_df['SF_Q01'],
                            color=color_cycle[_time], alpha=0.1, lw=0)
            ax.fill_between(_daynight_group_df['GROUP_CLASSVAR_MIN'], _daynight_group_df['SF_Q75'],
                            _daynight_group_df['SF_Q25'],
                            color=color_cycle[_time], alpha=0.2, lw=0)

            #     fit_line = ax.plot(px, nom, c='black', lw=2, zorder=99, alpha=1, label="prediction for higher classes with 95% region")
            #     fit_confidence_intervals = ax.fill_between(px, nom - 1.96 * std, nom + 1.96 * std, alpha=.2, color='#90A4AE', zorder=1)  # uncertainty lines (95% confidence)

            xmax = _daynight_group_df['GROUP_CLASSVAR_MIN'].max() if _daynight_group_df[
                                                                         'GROUP_CLASSVAR_MIN'].max() > xmax else xmax
            xmin = _daynight_group_df['GROUP_CLASSVAR_MIN'].min() if _daynight_group_df[
                                                                         'GROUP_CLASSVAR_MIN'].min() < xmin else xmin
            ymax = _daynight_group_df['SF_Q75'].max() if _daynight_group_df['SF_Q75'].max() > ymax else ymax
            ymin = _daynight_group_df['SF_Q25'].min() if _daynight_group_df['SF_Q25'].min() < ymin else ymin
            ymax = _daynight_group_df['SF_Q99'].max() if _daynight_group_df['SF_Q99'].max() > ymax else ymax
            ymin = _daynight_group_df['SF_Q01'].min() if _daynight_group_df['SF_Q01'].min() < ymin else ymin

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Scaling factors per {self.classvar_col} class, for daytime and nighttime")
        ax.set_xlabel('class variable (units)', color='black', fontsize=12, fontweight='bold')
        ax.set_ylabel('scaling factor ξ', color='black', fontsize=12, fontweight='bold')
        ax.axhline(0, lw=1, color='black')
        ax.legend()

    def run(self):
        self.plot_scatter(ax=self.axes_dict['ax1'])
        savefig(fig=self.fig, outfile=self.outplot)

    def prepare_axes(self):
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(12, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        axes_dict = {'ax1': ax1}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


class DielCyclesFlux(selfheating_newcols.NewCols):

    def __init__(self, df, outdir,
                 uncorrected_flux_col, corrected_flux_col, true_flux_col=None):

        self.plot_df = df[[uncorrected_flux_col, corrected_flux_col]].copy()
        if true_flux_col:
            self.plot_df[true_flux_col] = df[true_flux_col]

        self.plot_df = self.plot_df.dropna()  # Keep overlapping
        self.uncorrected_flux = self.plot_df[uncorrected_flux_col]
        self.corrected_flux = self.plot_df[corrected_flux_col]
        if true_flux_col:
            self.true_flux = self.plot_df[true_flux_col]

        self.true_flux_col = true_flux_col

        self.axes_dict, self.fig = self.prepare_axes()
        self.outfile = outdir / "4__fluxes_diel_cycles.png"
        self.run()

    def run(self):
        print("Plotting DielCyclesFlux ...")

        # IRGA75 (uncorrected)
        plot_diel_cycle(title="OPEN-PATH CO2 flux (uncorrected)",
                        series=self.uncorrected_flux,
                        ax=self.axes_dict['ax1'])
        if self.true_flux_col:
            plot_diel_cycle(title="Difference:\nOPEN-PATH (uncorrected) - ENCLOSED-PATH (true flux)",
                            series=self.uncorrected_flux.sub(self.true_flux),
                            ax=self.axes_dict['ax3'])

        # IRGA72
        if self.true_flux_col:
            plot_diel_cycle(title="ENCLOSED-PATH CO2 flux (true flux)",
                            series=self.true_flux,
                            ax=self.axes_dict['ax4'])
            plot_diel_cycle(series=self.uncorrected_flux,
                            ax=self.axes_dict['ax4'], ls=':')
            plot_diel_cycle(title="Difference:\nENCLOSED-PATH - OPEN-PATH (uncorrected) ",
                            series=self.true_flux.sub(self.uncorrected_flux),
                            ax=self.axes_dict['ax5'])

        # IRGA75 (corrected)
        plot_diel_cycle(title="OPEN-PATH CO2 flux (corrected)",
                        series=self.corrected_flux,
                        ax=self.axes_dict['ax7'])
        plot_diel_cycle(series=self.uncorrected_flux,
                        ax=self.axes_dict['ax7'], ls=':')
        plot_diel_cycle(title="Difference:\nOPEN-PATH (corrected) - OPEN-PATH (uncorrected) ",
                        series=self.corrected_flux.sub(self.uncorrected_flux),
                        ax=self.axes_dict['ax8'])
        if self.true_flux_col:
            plot_diel_cycle(title="Difference:\nOPEN-PATH (corrected) - ENCLOSED-PATH (true flux)",
                            series=self.corrected_flux.sub(self.true_flux),
                            ax=self.axes_dict['ax9'])

        savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(3, 3)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(12, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')  # Hide axis
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0], sharey=ax1)
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')  # Hide axis
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])
        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 'ax4': ax4,
                     'ax5': ax5, 'ax6': ax6, 'ax7': ax7, 'ax8': ax8,
                     'ax9': ax9}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


class SeriesFlux(selfheating_newcols.NewCols):

    def __init__(self, outdir, daytime,
                 uncorrected_flux, corrected_flux, true_flux=pd.Series()):
        self.uncorrected_flux = uncorrected_flux
        self.corrected_flux = corrected_flux
        self.true_flux = true_flux
        self.daytime = daytime
        self.axes_dict, self.fig = self.prepare_axes()
        self.outfile = outdir / "2__fluxes_series.png"
        self.run()

    def run(self):
        print("Plotting SeriesFlux ...")

        plot_series(title="OPEN-PATH CO2 flux (uncorrected)",
                    series=self.uncorrected_flux,
                    ax=self.axes_dict['ax1'])

        if not self.true_flux.empty:
            plot_series(title="ENCLOSED-PATH CO2 flux (true flux)",
                        series=self.true_flux,
                        ax=self.axes_dict['ax2'])

        plot_series(title="OPEN-PATH CO2 flux (corrected)",
                    series=self.corrected_flux,
                    ax=self.axes_dict['ax3'])

        savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(3, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(9, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharey=ax1)
        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


class CumulativeFlux(selfheating_newcols.NewCols):

    def __init__(self, df, outdir, daytime_col,
                 uncorrected_flux_col, corrected_flux_col, true_flux_col=None):

        self.daytime_col = daytime_col

        self.plot_df = df[[daytime_col, uncorrected_flux_col, corrected_flux_col]].copy()
        if true_flux_col:
            self.plot_df[true_flux_col] = df[true_flux_col]

        self.plot_df = self.plot_df.dropna()  # Keep overlapping
        self.uncorrected_flux = self.plot_df[uncorrected_flux_col]
        self.corrected_flux = self.plot_df[corrected_flux_col]
        if true_flux_col:
            self.true_flux = self.plot_df[true_flux_col]

        self.true_flux_col = true_flux_col

        self.axes_dict, self.fig = self.prepare_axes()
        self.outfile = outdir / "5__fluxes_cumulative.png"
        self.run()

    def run(self):
        print("Plotting CumulativeFlux ...")

        plot_cumulative(title="Daytime", which='daytime',
                        ax=self.axes_dict['ax1'],
                        subset=self.plot_df,
                        daytime_col=self.daytime_col)

        plot_cumulative(title="Nighttime", which='nighttime',
                        ax=self.axes_dict['ax2'],
                        subset=self.plot_df,
                        daytime_col=self.daytime_col)

        plot_cumulative(title="Daytime & Nighttime", which='all',
                        ax=self.axes_dict['ax3'],
                        subset=self.plot_df,
                        daytime_col=self.daytime_col)

        savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(3, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(9, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


class DielCyclesVars(selfheating_newcols.NewCols):

    def __init__(self, plot_df, outdir):
        self.plot_df = plot_df
        self.axes_dict, self.fig = self.prepare_axes()
        self.outfile = outdir / "3__vars_diel_cycles.png"
        self.run()

    def run(self):
        print("Plotting DielCyclesVars ...")

        plot_diel_cycle(title="ra: Aerodynamic Resistance (s m-1)",
                        series=self.plot_df[self.ra_col],
                        ax=self.axes_dict['ax1'])

        plot_diel_cycle(title="rho_d: Dry Air Density (kg m-3)",
                        series=self.plot_df[self.rho_d_col],
                        ax=self.axes_dict['ax2'])

        plot_diel_cycle(title="TS (°C)",
                        series=self.plot_df[self.ts_col],
                        ax=self.axes_dict['ax5'])

        plot_diel_cycle(title="FCT_unsc (µmol m-2 s-1)",
                        series=self.plot_df[self.fct_unsc_gf_col],
                        ax=self.axes_dict['ax6'])

        plot_diel_cycle(title="SF (#)",
                        series=self.plot_df[self.sf_gf_col],
                        ax=self.axes_dict['ax7'])

        plot_diel_cycle(title="FCT (µmol m-2 s-1)",
                        series=self.plot_df[self.fct_col],
                        ax=self.axes_dict['ax8'])

        savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(2, 4)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(16, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])

        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3, 'ax4': ax4,
                     'ax5': ax5, 'ax6': ax6, 'ax7': ax7, 'ax8': ax8}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


class SeriesVars(selfheating_newcols.NewCols):

    def __init__(self, plot_df, outdir):
        self.plot_df = plot_df
        self.axes_dict, self.fig = self.prepare_axes()
        self.outfile = outdir / "1__vars_series.png"
        self.run()

    def run(self):
        print("Plotting SeriesVars ...")

        plot_series(title="ra: Aerodynamic Resistance (s m-1)",
                    series=self.plot_df[self.ra_col],
                    ax=self.axes_dict['ax1'])

        plot_series(title="rho_d: Dry Air Density (kg m-3)",
                    series=self.plot_df[self.rho_d_col],
                    ax=self.axes_dict['ax2'])

        plot_series(title="TS: Instr. Surface Temperature (°C)",
                    series=self.plot_df[self.ts_col],
                    ax=self.axes_dict['ax5'])

        plot_series(title="FCT_unsc (µmol m-2 s-1)",
                    series=self.plot_df[self.fct_unsc_gf_col],
                    ax=self.axes_dict['ax6'])

        plot_series(title="SF (#)",
                    series=self.plot_df[self.sf_gf_col],
                    ax=self.axes_dict['ax7'])

        plot_series(title="FCT (µmol m-2 s-1)",
                    series=self.plot_df[self.fct_col],
                    ax=self.axes_dict['ax8'])

        savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(2, 5)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.Figure(facecolor='white', figsize=(24, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])
        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3,
                     'ax4': ax4, 'ax5': ax5, 'ax6': ax6,
                     'ax7': ax7, 'ax8': ax8}
        for key, ax in axes_dict.items():
            default_format(ax=ax)
        plt.show()
        return axes_dict, fig


def default_format(ax, fontsize=9, label_color='black',
                   txt_xlabel=False, txt_ylabel=False, txt_ylabel_units=False,
                   width=0.5, length=3, direction='in', colors='black', facecolor='white'):
    """ Apply default format to plot. """
    ax.set_facecolor(facecolor)
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize)
    format_spines(ax=ax, color=colors, lw=1)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')
    return None


def format_spines(ax, color, lw):
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)
    return None


def savefig(fig, outfile):
    print(f"--> Saving plot {outfile} ...")
    fig.savefig(outfile, format='png', bbox_inches='tight', facecolor='w',
                transparent=True, dpi=150)


def plot_series(ax, series, title):
    ax.plot_date(x=series.index, y=series,
                 ms=2, alpha=.3, ls='-', marker='o', markeredgecolor='none')
    ax.set_title(title, fontsize=9, fontweight='bold', y=1)


def plot_diel_cycle(ax, series, title=None, ls='-', lw=1, legend=True):
    # Calculate the hourly mean per month
    diel_cycle_df = series.groupby([series.index.month, series.index.hour]).mean().unstack()
    diel_cycle_df.T.plot(ls=ls, ax=ax, colormap='jet', label="X", lw=lw)
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold', y=1)
    if legend:
        ax.legend(ncol=2, labelspacing=0.1, prop={'size': 5})
    if (series.min() < 0) & (series.max() > 0):
        ax.axhline(0, lw=1, color='black')


def plot_cumulative(ax: plt.axis, title: str, which: str, subset: pandas.DataFrame,
                    daytime_col: tuple):
    subset = subset.dropna()

    if which == 'daytime':
        subset = subset.loc[subset[daytime_col] == 1]
    elif which == 'nighttime':
        subset = subset.loc[subset[daytime_col] == 0]
    elif which == 'all':
        pass

    # # Convert to gC m-2
    # _subset[self.op_co2_flux_QC01_nocorr_col] = \
    #     _subset[self.op_co2_flux_QC01_nocorr_col].multiply(0.02161926)
    # _subset[self.cp_co2_flux_QC01_col] = \
    #     _subset[self.cp_co2_flux_QC01_col].multiply(0.02161926)
    # _subset[self.op_co2_flux_corr_jar09_col] = \
    #     _subset[self.op_co2_flux_corr_jar09_col].multiply(0.02161926)

    subset = subset.cumsum()
    for col in subset.columns:
        if col == daytime_col:
            continue
        ax.plot_date(x=subset.index, y=subset[col], label=col, ms=3, alpha=.5)
    ax.set_title(title, fontsize=9, fontweight='bold', y=1)
    ax.legend()

# def plot_cumulative_diff(self, ax: plt.axis, title: str, which: str):
#     _subset = self.plot_df[self.cols].copy()
#     _subset = _subset.dropna()
#
#     if which == 'daytime':
#         _subset = _subset.loc[_subset[self.daytime_col] == 1]
#     elif which == 'nighttime':
#         _subset = _subset.loc[_subset[self.daytime_col] == 0]
#     elif which == 'all':
#         pass
#
#     _diff = _subset[self.op_co2_flux_corr_jar09_col].sub(_subset[self.cp_co2_flux_QC01_col])
#     _diff=_diff.abs()
#     _subset = _diff.cumsum()
#     ax.plot_date(x=_subset.index, y=_subset,
#                  label="diff", ms=3, alpha=.5)
#     ax.set_title(title, fontsize=9, fontweight='bold', y=1)
#     ax.legend()
