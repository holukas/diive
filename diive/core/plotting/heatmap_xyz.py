"""
HEATMAP
=======
"""

import numpy as np
import pandas as pd

from diive.core.plotting.heatmap_base import HeatmapBase


class HeatmapXYZ(HeatmapBase):

    def __init__(self,
                 x: pd.Series,
                 y: pd.Series,
                 z: pd.Series,
                 xlabel: str = None,
                 ylabel: str = None,
                 zlabel: str = None,
                 xtickpos: list = None,
                 xticklabels: list = None,
                 ytickpos: list = None,
                 yticklabels: list = None,
                 **kwargs):
        """
        Initializes the HeatmapXYZ class for plotting heatmaps from X, Y, and Z Series data.

        This class extends HeatmapBase to create a heatmap where the x-axis, y-axis,
        and color values (z) are derived directly from input pandas Series. It pivots
        the data to create the 2D grid for the heatmap.

        Args:
            x: A pandas Series representing the x-coordinates (columns in the pivoted data).
               Its name will be used as the x-axis label if `xlabel` is None.
            y: A pandas Series representing the y-coordinates (index in the pivoted data).
               Its name will be used as the y-axis label if `ylabel` is None.
            z: A pandas Series representing the values (colors) for the heatmap cells.
               Its name will be used as the colorbar label if `zlabel` is None.
            xlabel: Optional string for the x-axis label. If None, `x.name` is used.
            ylabel: Optional string for the y-axis label. If None, `y.name` is used.
            zlabel: Optional string for the colorbar label. If None, `z.name` is used.
            xtickpos: Optional list of x-axis tick positions.
            xticklabels: Optional list of x-axis tick labels, corresponding to `xtickpos`.
            ytickpos: Optional list of y-axis tick positions.
            yticklabels: Optional list of y-axis tick labels, corresponding to `ytickpos`.
            **kwargs: Additional keyword arguments passed to the `HeatmapBase` constructor.
        """
        super().__init__(heatmaptype='xyz', **kwargs)
        self.x = x
        self.y = y
        self.z = z

        self.xlabel = self.x.name if not xlabel else xlabel
        self.ylabel = self.y.name if not ylabel else ylabel
        self.zlabel = self.z.name if not zlabel else zlabel
        self.xtickpos = xtickpos
        self.xticklabels = xticklabels
        self.ytickpos = ytickpos
        self.yticklabels = yticklabels

        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare input x, y, and z Series into a pivoted DataFrame
        suitable for heatmap plotting.

        Creates a DataFrame from the three input Series,
        then pivots it to form a 2D grid. Calculates the cell
        edges for `pcolormesh` to ensure proper rendering with 'flat' shading.
        The processed x, y, and z arrays are stored as instance variables.
        """

        data = {
            self.x.name: self.x,
            self.y.name: self.y,
            self.z.name: self.z
        }
        df = pd.DataFrame.from_dict(data, orient='columns')

        pivot_df = pd.pivot_table(df, index=self.y.name, columns=self.x.name, values=self.z.name)

        # Extract x, y, and z values for pcolormesh
        x_coords = pivot_df.columns.values
        y_coords = pivot_df.index.values
        z_values = pivot_df.values

        # For pcolormesh, it's generally better to define the *edges* of the cells
        # rather than their centers. Needed for 'shading=flat'
        dx = np.diff(x_coords)[-1]
        dy = np.diff(y_coords)[-1]

        # Create cell boundaries for x and y
        x_edges = np.append(x_coords, x_coords[-1] + dx)
        y_edges = np.append(y_coords, y_coords[-1] + dy)

        self.x = x_edges
        self.y = y_edges
        self.z = z_values

    def plot(self):
        """
        Generate HeatmapXYZ plot.

        This method orchestrates the plotting process by calling `plot_pcolormesh`,
        optionally displaying values on the heatmap, and applying axis formatting
        including custom tick positions and labels if provided.
        """
        p = self.plot_pcolormesh(shading='flat')

        if self.show_values:
            self.show_vals_in_plot()

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
        )


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

    q = dv.qga(
        x=df[x],
        y=df[y],
        z=df[z],
        n_quantiles=10,
        min_n_vals_per_bin=1,
        binagg_z='mean'
    )
    q.run()

    # Pivoted dataframe, 2D grid (matrix)
    df_long = q.df_long

    hm = dv.heatmapxyz(
        x=df_long['BIN_Tair_f_mean'],
        y=df_long['BIN_VPD_f_mean'],
        z=df_long['NEP_sum'],
        show_values=True,
        show_values_n_dec_places=0,
        show_values_fontsize=9,
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
