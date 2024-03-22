"""
HEATMAP
=======
"""
import copy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.plotfuncs import default_format
from diive.pkgs.analyses.quantilexyaggz import QuantileXYAggZ


class HeatmapBase:

    def __init__(self,
                 fig=None,
                 ax=None,
                 title: str = None,
                 cb_digits_after_comma: int = 2,
                 cb_label_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ax_labels_fontsize: float = theme.AX_LABELS_FONTSIZE,
                 ticks_labels_fontsize: float = theme.TICKS_LABELS_FONTSIZE,
                 minyticks: int = 3,
                 maxyticks: int = 10,
                 cmap: str = 'RdYlBu_r',
                 color_bad: str = 'grey',
                 figsize: tuple = (10, 9)):
        self.fig = fig
        self.ax = ax
        self.title = title
        self.cb_digits_after_comma = cb_digits_after_comma
        self.cb_label_fontsize = cb_label_fontsize
        self.axlabels_fontsize = ax_labels_fontsize
        self.ticks_labelsize = ticks_labels_fontsize
        self.minyticks = minyticks
        self.maxyticks = maxyticks
        self.cmap = cmap
        self.color_bad = color_bad
        self.figsize = figsize

        # Create axis if none is given
        if not ax:
            self.fig, self.ax = self._create_ax()

    def _create_ax(self):
        """Create figure and axis"""
        # Figure setup
        fig = plt.figure(facecolor='white', figsize=self.figsize)
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        return fig, ax

    def format(self, ax, plot, xlabel: str, ylabel: str, zlabel: str,
               tickpos: list, ticklabels: list, cb_digits_after_comma: int = 2,
               labelsize: float = None):
        # title = self.title if self.title else f"{self.series.name} in {self.series.index.freqstr} time resolution"
        # self.ax.set_title(title, color='black')
        # self.ax.set_title(self.title, color='black', size=theme.FONTSIZE_HEADER_AXIS)
        labelsize = labelsize if labelsize else self.cb_label_fontsize
        ax.set_xticks(tickpos)
        ax.set_yticks(tickpos)
        ax.set_xticklabels(ticklabels)
        ax.set_yticklabels(ticklabels)
        # nice_date_ticks(ax=self.ax, minticks=self.minyticks, maxticks=self.maxyticks, which='y')
        cb = plt.colorbar(plot, ax=ax, format=f"%.{int(cb_digits_after_comma)}f",
                          label=zlabel)
        cb.ax.tick_params(labelsize=labelsize)
        cb.set_label(label=zlabel, size=labelsize)

        # cbytick_obj = plt.getp(cb.axes_dict, 'yticklabels')  # Set y tick label color
        # plt.setp(cbytick_obj, color='black', fontsize=theme.FONTSIZE_HEADER_AXIS)
        default_format(ax=ax,
                       ax_xlabel_txt=xlabel,
                       ax_ylabel_txt=ylabel,
                       ax_labels_fontsize=labelsize,
                       ticks_direction='out',
                       ticks_length=8,
                       ticks_width=2,
                       ticks_labels_fontsize=labelsize)

        from diive.core.plotting.plotfuncs import format_spines
        format_spines(ax=ax, color='black', lw=2)

    @staticmethod
    def _set_cmap(cmap, color_bad, z):
        """Set colormap and color of missing values"""
        # Colormap
        cmap = copy.copy(plt.get_cmap(cmap))  # Needed for now, https://github.com/matplotlib/matplotlib/issues/17634
        cmap.set_bad(color=color_bad, alpha=1.)  # Set missing data to specific color
        z = np.ma.masked_invalid(z)  # Mask NaN as missing
        return cmap, z


class HeatmapPivotXYZ(HeatmapBase):

    def __init__(self, pivotdf: DataFrame, verbose: bool = False):
        """
        Plot heatmap of pivoted dataframe with dataframe columns on x-axis,
        dataframe index on y-axis and dataframe values as z colors

        Args:
            pivotdf: Pivoted dataframe containing values for x in columns,
                y in index and z in values
            verbose: More text output to the console if *True*

        """
        super().__init__()
        self.pivotdf = pivotdf

        self.verbose = verbose

        if self.verbose:
            print(f"Plotting heatmap  ...")

        self.x, self.y, self.z = self._setdata()

    def plot(self,
             ax=None,
             cmap=None,
             xlabel: str = None,
             ylabel: str = None,
             zlabel: str = None,
             tickpos: list = None,
             ticklabels: list = None,
             cb_digits_after_comma: int = 2):

        if not ax:
            ax = self.ax
            showplot = True
        else:
            showplot = False

        xlabel = xlabel if xlabel else self.pivotdf.columns.name
        ylabel = ylabel if ylabel else self.pivotdf.index.name
        zlabel = zlabel if zlabel else "Value"
        if not tickpos:
            tickpos = list(self.x)
        if not ticklabels:
            ticklabels = list(self.x)

        cmap = cmap if cmap else self.cmap
        cmap, z = self._set_cmap(cmap=cmap, color_bad=self.color_bad, z=self.z)
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)

        p = ax.pcolormesh(self.x, self.y, z,
                          linewidths=1, cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          zorder=99)



        self.format(ax=ax, plot=p,
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    tickpos=tickpos, ticklabels=ticklabels,
                    cb_digits_after_comma=cb_digits_after_comma,
                    labelsize=20)
        if showplot:
            self.fig.show()

    def _setdata(self):
        x = self.pivotdf.columns.values
        y = self.pivotdf.index.values
        z = self.pivotdf.values
        return x, y, z


def example():
    from diive.core.io.files import load_pickle

    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    nee_orig = 'NEE_CUT_REF_orig'
    nep_col = 'NEP'
    ta_col = 'Tair_f'
    gpp_dt_col = 'GPP_DT_CUT_REF'
    reco_dt_col = 'Reco_DT_CUT_REF'
    gpp_nt_col = 'GPP_CUT_REF_f'
    reco_nt_col = 'Reco_CUT_REF'
    ratio_dt_gpp_reco = 'RATIO_DT_GPP_RECO'
    rh_col = 'RH'
    swin_col = 'Rg_f'

    # Load data, using pickle for fast loading
    source_file = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\__manuscripts\11.01_NEP-Penalty_CH-DAV_1997-2022.09 (2023)\data\CH-DAV_FP2022.5_1997-2022_ID20230206154316_30MIN.diive.csv.pickle"
    df_orig = load_pickle(filepath=source_file)
    # df_orig = df_orig.loc[df_orig.index.year >= 2019].copy()

    # Data between May and Sep
    df_orig = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()

    # Subset
    df = df_orig[[nee_col, vpd_col, ta_col, gpp_dt_col, reco_dt_col, swin_col]].copy()
    df[ratio_dt_gpp_reco] = df[gpp_dt_col].divide(df[reco_dt_col])

    # Convert units
    df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[nep_col] = df[nee_col].multiply(-1)  # Convert NEE to NEP, net uptake is now positive
    df[gpp_dt_col] = df[gpp_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[reco_dt_col] = df[reco_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1

    xcol = ta_col
    ycol = vpd_col
    zcol = reco_dt_col

    # Aggregation to daily values
    df = df.groupby(df.index.date).agg(
        {
            xcol: ['min', 'max'],
            ycol: ['min', 'max'],
            zcol: 'sum'
        }
    )

    from diive.core.dfun.frames import flatten_multiindex_all_df_cols
    df = flatten_multiindex_all_df_cols(df=df)
    x = f"{xcol}_max"
    y = f"{ycol}_max"
    z = f"{zcol}_sum"

    pivotdf = QuantileXYAggZ(x=df[x],
                             y=df[y],
                             z=df[z],
                             n_quantiles=20,
                             min_n_vals_per_bin=5,
                             binagg_z='mean')

    print(pivotdf)

    hm = HeatmapPivotXYZ(pivotdf=pivotdf)
    hm.plot(
        cb_digits_after_comma=0,
        # xlabel=r'Percentile of daily maximum TA ($\mathrm{°C}$)',
        # ylabel=r'Percentile of daily maximum VPD ($\mathrm{kPa}$)',
        # zlabel=r'Net ecosystem productivity ($\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$)'
    )


if __name__ == '__main__':
    example()
