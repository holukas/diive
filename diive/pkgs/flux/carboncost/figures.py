"""

====================
CARBON COST: FIGURES
====================



"""

# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from common.plots import plotfuncs
from common.plots.fitplot import fitplot
from common.plots.rectangle import rectangle


# def fig_gpp_reco_nee_vs_vpd(results, linecrossings_aggs, thres_chd, num_chds):
#     """Assemble figure"""
#     # Prepare figure
#     fig = plt.figure(figsize=(20, 9))
#     gs = gridspec.GridSpec(1, 2)  # rows, cols
#     gs.update(wspace=.2, hspace=0, left=.1, right=.9, top=.9, bottom=.1)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     # fig.show()
#
#     # fig_gpp_reco_vs_vpd(ax=ax1, results=results, linecrossings_aggs=linecrossings_aggs)
#     # _plot_nee_vs_vpd(ax=ax2, results=results, linecrossings_aggs=linecrossings_aggs,
#     #                  thres_chd=thres_chd, num_chds=num_chds)
#     #
#     fig.show()

def plot_nee_vs_vpd(ax, results):
    """Plot NEE vs VPD"""

    # TODO hier weiter

    bts_results = results['bts_results'][0]  # 0 means non-bootstrapped data

    # NEE
    line_xy_nee, line_fit_nee, line_fit_ci_nee, line_fit_pb_nee = \
        fitplot(ax=ax, label='NEE',
                x=bts_results['nee']['x'],
                y=bts_results['nee']['y'],
                fitx=bts_results['nee']['fit_df']['fit_x'],
                fity=bts_results['nee']['fit_df']['nom'],
                fity_ci95_lower=bts_results['nee']['fit_df']['nom_lower_ci95'],
                fity_ci95_upper=bts_results['nee']['fit_df']['nom_upper_ci95'],
                fity_pb_lower=bts_results['nee']['fit_df']['lower_predband'],
                fity_pb_upper=bts_results['nee']['fit_df']['upper_predband'],
                color='#2196F3')  # color blue 500

    # ax.axvline(thres_chd, color='black', lw=1, ls='-', zorder=99, label="CHD threshold")
    # ax.fill_between([thres_chd,
    #                  results['nee']['fit_df']['fit_x'].max()],
    #                 0, 1,
    #                 color='#f44336', alpha=0.1, transform=ax.get_xaxis_transform(),
    #                 label=f"CHDs ({num_chds} days)", zorder=1)

    ax.axhline(0, lw=1, color='black', zorder=12)


def plot_gpp_reco_vs_vpd(ax, results: dict):
    """Create figure showing GPP and RECO vs VPD"""

    bts_results = results['bts_results'][0]  # 0 means non-bootstrapped data
    bts_linecrossings_aggs = results['bts_linecrossings_aggs']

    # GPP
    _numvals_per_bin = bts_results['gpp']['numvals_per_bin']
    line_xy_gpp, line_fit_gpp, line_fit_ci_gpp, line_fit_pb_gpp = \
        fitplot(ax=ax, label=f"GPP ({_numvals_per_bin['min']:.0f} - {_numvals_per_bin['max']:.0f} values per bin)",
                x=bts_results['gpp']['x'],
                y=bts_results['gpp']['y'],
                fitx=bts_results['gpp']['fit_df']['fit_x'],
                fity=bts_results['gpp']['fit_df']['nom'],
                fity_ci95_lower=bts_results['gpp']['fit_df']['nom_lower_ci95'],
                fity_ci95_upper=bts_results['gpp']['fit_df']['nom_upper_ci95'],
                fity_pb_lower=bts_results['gpp']['fit_df']['lower_predband'],
                fity_pb_upper=bts_results['gpp']['fit_df']['upper_predband'],
                color='#2196F3')  # color blue 500

    # RECO
    _numvals_per_bin = bts_results['reco']['numvals_per_bin']
    line_xy_reco, line_fit_reco, line_fit_ci_reco, line_fit_pb_reco = \
        fitplot(ax=ax, label=f"RECO ({_numvals_per_bin['min']:.0f} - {_numvals_per_bin['max']:.0f} values per bin)",
                x=bts_results['reco']['x'],
                y=bts_results['reco']['y'],
                fitx=bts_results['reco']['fit_df']['fit_x'],
                fity=bts_results['reco']['fit_df']['nom'],
                fity_ci95_lower=bts_results['reco']['fit_df']['nom_lower_ci95'],
                fity_ci95_upper=bts_results['reco']['fit_df']['nom_upper_ci95'],
                fity_pb_lower=bts_results['reco']['fit_df']['lower_predband'],
                fity_pb_upper=bts_results['reco']['fit_df']['upper_predband'],
                color='#E53935')  # color red 600

    # Actual non-bootstrapped line crossing, the point where RECO = GPP
    line_equilibrium = ax.scatter(bts_results['linecrossing_vals']['x_col'],
                                  bts_results['linecrossing_vals']['gpp_nom'],
                                  edgecolor='none', color='black', alpha=1, s=90,
                                  label='flux equilibrium, RECO = GPP', zorder=99, marker='s')

    # Add rectangle (bootstrapped results)
    rect = rectangle(ax=ax,
                     rect_lower_left_x=bts_linecrossings_aggs['x_min'],
                     rect_lower_left_y=bts_linecrossings_aggs['y_gpp_min'],
                     rect_width=bts_linecrossings_aggs['x_max'] - bts_linecrossings_aggs['x_min'],
                     rect_height=bts_linecrossings_aggs['y_gpp_max'] - bts_linecrossings_aggs['y_gpp_min'],
                     label="equlibrium range (bootstrapped)")

    # Format
    xlabel = "classes of daily VPD maxima (hPa)"
    ylabel = r"daytime median flux ($\mu mol \/\ CO_2 \/\ m^{-2} \/\ s^{-1}$)"
    # ylabel = "daytime median flux" + " (gC $\mathregular{m^{-2} \ d^{-1}}$  --> UNITS ???)"
    plotfuncs.default_format(ax=ax, txt_xlabel=xlabel, txt_ylabel=ylabel,
                             fontsize=12, width=1)

    # Custom legend
    # Assign two of the handles to the same legend entry by putting them in a tuple
    # and using a generic handler map (which would be used for any additional
    # tuples of handles like (p1, p3)).
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    l = ax.legend(
        [
            line_xy_gpp,
            line_xy_reco,
            (line_fit_gpp, line_fit_reco),
            (line_fit_ci_gpp, line_fit_ci_reco),
            (line_fit_pb_gpp, line_fit_pb_reco),
            line_equilibrium,
            rect
        ],
        [
            line_xy_gpp.get_label(),
            line_xy_reco.get_label(),
            line_fit_gpp.get_label(),
            line_fit_ci_gpp.get_label(),
            line_fit_pb_gpp.get_label(),
            line_equilibrium.get_label(),
            rect.get_label()
        ],
        scatterpoints=1,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)})
