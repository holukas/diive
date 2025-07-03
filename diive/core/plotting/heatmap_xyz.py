"""
HEATMAP
=======
"""
from poetry.console.commands import self

import diive as dv
import pandas as pd
import numpy as np
from diive.core.plotting.heatmap_base import HeatmapBase
from typing import Literal

class HeatmapXYZ(HeatmapBase):

    def __init__(self,
                 x: pd.Series,
                 y: pd.Series,
                 z: pd.Series,
                 n_bins: int = 10,
                 min_n_vals_per_bin: int = 1,
                 binagg_z: Literal['mean', 'min', 'max', 'median', 'count', 'sum'] = 'mean',
                 xlabel: str = None,
                 ylabel: str = None,
                 xtickpos: list = None,
                 xticklabels: list = None,
                 ytickpos: list = None,
                 yticklabels: list = None,
                 **kwargs):
        """
        Plot heatmap of pivoted dataframe with dataframe columns on x-axis,
        dataframe index on y-axis and dataframe values as z colors

        Args:
            pivotdf: Pivoted dataframe containing values for x in columns,
                y in index and z in values
            verbose: More text output to the console if *True*

        """
        super().__init__(**kwargs)
        # self.pivotdf = pivotdf
        self.x = x
        self.y = y
        self.z = z
        self.n_bins = n_bins
        self.min_n_vals_per_bin = min_n_vals_per_bin
        self.binagg_z = binagg_z

        self.xlabel = self.x.name if not xlabel else xlabel
        self.ylabel = self.y.name if not ylabel else ylabel
        self.xtickpos = xtickpos
        self.xticklabels = xticklabels
        self.ytickpos = ytickpos
        self.yticklabels = yticklabels

        self._prepare_data()

    def _prepare_data(self):

        q = dv.qga(
            x=self.x,
            y=self.y,
            z=self.z,
            n_quantiles=self.n_bins,
            min_n_vals_per_bin=self.min_n_vals_per_bin,
            binagg_z=self.binagg_z
        )
        q.run()

        # Pivoted dataframe, 2D grid (matrix)
        pivot_df = q.df_wide

        # Extract x, y, and z values for pcolormesh
        x_coords = pivot_df.columns.values
        y_coords = pivot_df.index.values
        z_values = pivot_df.values

        # For pcolormesh, it's generally better to define the *edges* of the cells
        # rather than their centers. We can get these calculating the differences
        # between the coordinate values and extending by half the step at the ends.
        # Assuming regular spacing, which is true for 0, 25, 50, 75.
        dx = np.diff(x_coords)[-1]
        dy = np.diff(y_coords)[-1]

        # Create cell boundaries for x and y
        x_edges = np.append(x_coords, x_coords[-1] + dx)
        y_edges = np.append(y_coords, y_coords[-1] + dy)

        self.x = x_edges
        self.y = y_edges
        self.z = z_values

    def plot(self):
        p = self.plot_pcolormesh(shading='flat')

        if self.show_values:
            self.show_vals_in_plot()

        # xlabel = self.pivotdf.columns.name
        # xlabel = xlabel if xlabel else self.pivotdf.columns.name
        # ylabel = self.pivotdf.index.name
        # ylabel = ylabel if ylabel else self.pivotdf.index.name
        zlabel = "Value"
        # zlabel = zlabel if zlabel else "Value"
        if self.xtickpos:
            self.ax.set_xticks(self.xtickpos)
            if self.xticklabels:
                self.ax.set_xticklabels(self.xticklabels)
        if self.ytickpos:
            self.ax.set_yticks(self.ytickpos)
            if self.yticklabels:
                self.ax.setyticklabels(self.yticklabels)

        self.format(
            plot=p,
            ax_xlabel_txt=self.xlabel,
            ax_ylabel_txt=self.ylabel,
            # zlabel=zlabel,
            # tickpos=tickpos,
            # ticklabels=ticklabels,
            # cb_digits_after_comma=cb_digits_after_comma,
            # labelsize=20
        )
        # if showplot:
        #     self.fig.show()


def _example():
    import diive as dv

    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    nep_col = 'NEP'
    ta_col = 'Tair_f'
    # nee_orig = 'NEE_CUT_REF_orig'
    # gpp_dt_col = 'GPP_DT_CUT_REF'
    # reco_dt_col = 'Reco_DT_CUT_REF'
    # gpp_nt_col = 'GPP_CUT_REF_f'
    # reco_nt_col = 'Reco_CUT_REF'
    # ratio_dt_gpp_reco = 'RATIO_DT_GPP_RECO'
    # rh_col = 'RH'
    # swin_col = 'Rg_f'

    # Load data, using parquet for fast loading
    df_orig = dv.load_exampledata_parquet()
    # df_orig = df_orig.loc[df_orig.index.year >= 2019].copy()

    # Data between May and Sep
    df_orig = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()

    # Subset
    df = df_orig[[nee_col, vpd_col, ta_col]].copy()
    # df[ratio_dt_gpp_reco] = df[gpp_dt_col].divide(df[reco_dt_col])

    # Convert units
    df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[nep_col] = df[nee_col].multiply(-1)  # Convert NEE to NEP, net uptake is now positive
    # df[gpp_dt_col] = df[gpp_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    # df[reco_dt_col] = df[reco_dt_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1

    xcol = ta_col
    ycol = vpd_col
    zcol = nep_col

    # Aggregation to daily values
    df = df.groupby(df.index.date).agg(
        {
            xcol: ['min', 'max', 'mean'],
            ycol: ['min', 'max', 'mean'],
            zcol: 'sum'
        }
    )

    from diive.core.dfun.frames import flatten_multiindex_all_df_cols
    df = flatten_multiindex_all_df_cols(df=df)
    x = f"{xcol}_mean"
    y = f"{ycol}_mean"
    z = f"{zcol}_sum"



    hm = dv.heatmapxyz(
        x=df[x],
        y=df[y],
        z=df[z],
        n_bins=5,
        min_n_vals_per_bin=1,
        binagg_z='count',
        # x=df[x],
        # y=df[y],
        # z=df[z],
        # xtickpos=[0, 25, 50, 75],
        # ytickpos=[0, 25, 50, 75],
        # xticklabels=['0-25', '25-50', '50-75', '75-100'],
        cb_digits_after_comma=0
    )
    hm.show(
        # cb_digits_after_comma=0,
        # tickpos=[10, 50, 90],
        # tickpos=[16, 25, 50, 75, 84],
        # ticklabels=['10', '50', '90']
        # ticklabels=['16', '25', '50', '75', '84']
        # xlabel=r'Percentile of daily maximum TA ($\mathrm{°C}$)',
        # ylabel=r'Percentile of daily maximum VPD ($\mathrm{kPa}$)',
        # zlabel=r'Net ecosystem productivity ($\mathrm{gCO_{2}\ m^{-2}\ d^{-1}}$)'
    )


if __name__ == '__main__':
    _example()
