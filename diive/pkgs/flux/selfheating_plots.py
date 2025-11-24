import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas
import pandas as pd

import selfheating_newcols


class DielCyclesFlux(selfheating_newcols.NewCols):

    def __init__(self, df,
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
        self.run()
        self.fig.show()

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

    def prepare_axes(self):
        gs = gridspec.GridSpec(3, 3)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.figure(facecolor='white', figsize=(12, 9))
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
        return axes_dict, fig


class SeriesFlux(selfheating_newcols.NewCols):

    def __init__(self, daytime,
                 uncorrected_flux, corrected_flux, true_flux=pd.Series()):
        self.uncorrected_flux = uncorrected_flux
        self.corrected_flux = corrected_flux
        self.true_flux = true_flux
        self.daytime = daytime


    def run(self):
        print("Plotting SeriesFlux ...")

        gs = gridspec.GridSpec(3, 1)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.figure(facecolor='white', figsize=(9, 9))
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharey=ax1)
        axes_dict = {'ax1': ax1, 'ax2': ax2, 'ax3': ax3}
        for key, ax in axes_dict.items():
            default_format(ax=ax)

        plot_series(title="OPEN-PATH CO2 flux (uncorrected)",
                    series=self.uncorrected_flux,
                    ax=axes_dict['ax1'])

        if not self.true_flux.empty:
            plot_series(title="ENCLOSED-PATH CO2 flux (true flux)",
                        series=self.true_flux,
                        ax=axes_dict['ax2'])

        plot_series(title="OPEN-PATH CO2 flux (corrected)",
                    series=self.corrected_flux,
                    ax=axes_dict['ax3'])

        # savefig(fig=self.fig, outfile=self.outfile)
        fig.show()





class DielCyclesVars(selfheating_newcols.NewCols):

    def __init__(self, plot_df):
        self.plot_df = plot_df
        self.axes_dict, self.fig = self.prepare_axes()
        self.run()
        self.fig.show()

    def run(self):
        print("Plotting DielCyclesVars ...")

        plot_diel_cycle(title="ra: Aerodynamic Resistance (s m-1)",
                        series=self.plot_df[self.aerodynamic_resistance],
                        ax=self.axes_dict['ax1'])

        plot_diel_cycle(title="rho_d: Dry Air Density (kg m-3)",
                        series=self.plot_df[self.dry_air_density],
                        ax=self.axes_dict['ax2'])

        plot_diel_cycle(title="TS (°C)",
                        series=self.plot_df[self.t_instrument_surface],
                        ax=self.axes_dict['ax5'])

        plot_diel_cycle(title="FCT_unsc (µmol m-2 s-1)",
                        series=self.plot_df[self.fct_unsc_gf],
                        ax=self.axes_dict['ax6'])

        plot_diel_cycle(title="SF (#)",
                        series=self.plot_df[self.sf_gf],
                        ax=self.axes_dict['ax7'])

        plot_diel_cycle(title="FCT (µmol m-2 s-1)",
                        series=self.plot_df[self.fct],
                        ax=self.axes_dict['ax8'])

        # savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(2, 4)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.figure(facecolor='white', figsize=(16, 9))
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
        return axes_dict, fig


class SeriesVars(selfheating_newcols.NewCols):

    def __init__(self, plot_df):
        self.plot_df = plot_df
        self.axes_dict, self.fig = self.prepare_axes()
        self.run()
        self.fig.show()

    def run(self):
        print("Plotting SeriesVars ...")

        plot_series(title="ra: Aerodynamic Resistance (s m-1)",
                    series=self.plot_df[self.aerodynamic_resistance],
                    ax=self.axes_dict['ax1'])

        plot_series(title="rho_d: Dry Air Density (kg m-3)",
                    series=self.plot_df[self.dry_air_density],
                    ax=self.axes_dict['ax2'])

        plot_series(title="TS: Instr. Surface Temperature (°C)",
                    series=self.plot_df[self.t_instrument_surface],
                    ax=self.axes_dict['ax5'])

        plot_series(title="FCT_unsc (µmol m-2 s-1)",
                    series=self.plot_df[self.fct_unsc_gf],
                    ax=self.axes_dict['ax6'])

        plot_series(title="SF (#)",
                    series=self.plot_df[self.sf_gf],
                    ax=self.axes_dict['ax7'])

        plot_series(title="FCT (µmol m-2 s-1)",
                    series=self.plot_df[self.fct],
                    ax=self.axes_dict['ax8'])

        # savefig(fig=self.fig, outfile=self.outfile)

    def prepare_axes(self):
        gs = gridspec.GridSpec(2, 5)  # rows, cols
        gs.update(wspace=0.2, hspace=0.2, left=0.03, right=0.96, top=0.96, bottom=0.03)
        fig = plt.figure(facecolor='white', figsize=(24, 9))
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
        # plt.show()
        return axes_dict, fig













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
