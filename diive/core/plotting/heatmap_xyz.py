"""
HEATMAP — XYZ variant
=====================

Contains :class:`HeatmapXYZ`, a heatmap that accepts three parallel Series
(x, y, z) and pivots them into a 2-D colour grid.

Unlike :class:`~diive.core.plotting.heatmap_datetime.HeatmapDateTime` (which
reshapes a time series by date and time-of-day), ``HeatmapXYZ`` is fully
generic: x and y can be any numeric or categorical coordinates, and z is the
value mapped to colour.  The typical use-case is visualising the output of
:class:`~diive.pkgs.analyses.gridaggregator.GridAggregator` — e.g. mean NEP
binned by temperature and VPD.

**Important:** ``HeatmapXYZ`` expects **pre-aggregated input** where each unique
(x, y) pair appears exactly once. Use :attr:`GridAggregator.df_agg_long` (not
``df_long`` which contains raw observations) as the source.

Top-level aliases:
    - ``dv.heatmap_xyz(x, y, z, ...)`` — direct Series input
    - ``dv.heatmap_xyz.from_gridaggregator(q, x_col, y_col, z_col, ...)`` — from GridAggregator output

Example (recommended with GridAggregator)::

    import diive as dv
    df = dv.load_exampledata_parquet()
    q = dv.ga(x=df['Tair_f'], y=df['VPD_f'], z=df['NEE_CUT_REF_f'],
              binning_type='quantiles', n_bins=10,
              min_n_vals_per_bin=1, aggfunc='mean')
    hm = dv.heatmap_xyz.from_gridaggregator(q, 'Tair_f', 'VPD_f', 'NEE_CUT_REF_f',
                                           show_values=True, show_values_n_dec_places=2)
    hm.show()

References:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_levels.html
"""

import numpy as np
import pandas as pd

from diive.core.plotting.heatmap_base import HeatmapBase


class HeatmapXYZ(HeatmapBase):
    """Heatmap built from three parallel Series: x coordinates, y coordinates, and z values.

    **Important:** Input Series must be **pre-aggregated**, with exactly one value per unique
    (x, y) coordinate pair. The typical source is :attr:`GridAggregator.df_agg_long`.

    The three Series are combined into a long-format DataFrame and pivoted into
    a 2-D grid (y rows × x columns) whose cells are colour-coded by the z value.
    Cell boundaries are computed automatically from the coordinate spacing so
    that ``pcolormesh`` renders each cell at the correct position.

    Top-level alias: ``dv.heatmap_xyz(x, y, z, ...)``

    Example::

        import diive as dv
        hm = dv.heatmap_xyz(x=series_x, y=series_y, z=series_z,
                           show_values=True, show_values_n_dec_places=1)
        hm.show()
    """

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
        """Creates the heatmap object and prepares the data grid.

        Args:
            x: Series of x-coordinates (becomes the column axis after pivoting).
               Must have a non-``None`` name that is unique across x, y, and z.
               The name is used as the x-axis label when ``xlabel`` is *None*.
            y: Series of y-coordinates (becomes the row axis after pivoting).
               Same naming requirements as ``x``.
            z: Series of cell values mapped to colour.
               Same naming requirements as ``x``.
            xlabel: x-axis label.  When *None* ``x.name`` is used.
            ylabel: y-axis label.  When *None* ``y.name`` is used.
            zlabel: Colorbar label.  When *None* ``z.name`` is used.
            xtickpos: Explicit x-axis tick positions.  *None* = auto.
            xticklabels: Tick labels matching ``xtickpos``.  Ignored when
                         ``xtickpos`` is *None*.
            ytickpos: Explicit y-axis tick positions.  *None* = auto.
            yticklabels: Tick labels matching ``ytickpos``.  Ignored when
                         ``ytickpos`` is *None*.
            **kwargs: All keyword arguments accepted by
                      :class:`~diive.core.plotting.heatmap_base.HeatmapBase`,
                      e.g. ``figsize``, ``cmap``, ``vmin``/``vmax``,
                      ``show_values``, ``verbose``.

        Raises:
            ValueError: If x, y, or z has a ``None`` name.
            ValueError: If x, y, and z do not all have distinct names.
            ValueError: If x, y, and z do not have the same length.
        """
        # Validate Series names before anything else so errors are clear
        if x.name is None or y.name is None or z.name is None:
            raise ValueError(
                "x, y, and z Series must all have a non-None name. "
                f"Got: x.name={x.name!r}, y.name={y.name!r}, z.name={z.name!r}."
            )
        if len({x.name, y.name, z.name}) < 3:
            raise ValueError(
                "x, y, and z Series must all have unique names. "
                f"Got: x.name={x.name!r}, y.name={y.name!r}, z.name={z.name!r}."
            )
        if not (len(x) == len(y) == len(z)):
            raise ValueError(
                f"x, y, and z must have the same length "
                f"(got {len(x)}, {len(y)}, {len(z)})."
            )

        super().__init__(heatmaptype='xyz', **kwargs)
        self.x = x
        self.y = y
        self.z = z

        # Use is None so that an explicit empty string is honoured
        self.xlabel = x.name if xlabel is None else xlabel
        self.ylabel = y.name if ylabel is None else ylabel
        self.zlabel = z.name if zlabel is None else zlabel

        self.xtickpos = xtickpos
        self.xticklabels = xticklabels
        self.ytickpos = ytickpos
        self.yticklabels = yticklabels

        self.p = None  # Collects the pcolormesh plot object

        self._prepare_data()

    @classmethod
    def from_gridaggregator(cls,
                            gridagg,
                            x_col: str,
                            y_col: str,
                            z_col: str,
                            xlabel: str = None,
                            ylabel: str = None,
                            zlabel: str = None,
                            **kwargs) -> 'HeatmapXYZ':
        """Create HeatmapXYZ from GridAggregator output.

        Convenience constructor that automatically extracts pre-aggregated data from
        ``GridAggregator.df_agg_long`` and handles bin column naming conventions,
        eliminating the need for manual DataFrame extraction.

        Args:
            gridagg: GridAggregator instance with binned and aggregated output
            x_col: Original x series name used when creating GridAggregator
                   (e.g., 'Tair_f', not 'BIN_Tair_f'). Will be automatically
                   converted to the bin column name 'BIN_Tair_f'.
            y_col: Original y series name used when creating GridAggregator
            z_col: Original z series name used when creating GridAggregator.
                   The aggregated z-values in df_agg_long keep the original name
                   (no BIN_ prefix).
            xlabel: x-axis label. When *None*, uses ``x_col``.
            ylabel: y-axis label. When *None*, uses ``y_col``.
            zlabel: Colorbar label. When *None*, uses ``z_col``.
            **kwargs: All keyword arguments accepted by :meth:`HeatmapXYZ.__init__`,
                      e.g. ``figsize``, ``cmap``, ``vmin``/``vmax``,
                      ``show_values``, ``verbose``.

        Returns:
            HeatmapXYZ instance ready to plot.

        Raises:
            AttributeError: If ``gridagg.df_agg_long`` is not available.
            KeyError: If ``x_col``, ``y_col``, or ``z_col`` not found in df_agg_long.

        Example::

            import diive as dv

            # Create and aggregate data
            q = dv.ga(x=df['Tair_f'], y=df['VPD_f'], z=df['NEE_CUT_REF_f'],
                      binning_type='quantiles', n_bins=10, aggfunc='mean')

            # Create heatmap directly from GridAggregator output
            hm = dv.heatmap_xyz.from_gridaggregator(
                q, 'Tair_f', 'VPD_f', 'NEE_CUT_REF_f',
                show_values=True, show_values_n_dec_places=2
            )
            hm.show()
        """
        # Extract pre-aggregated long-format DataFrame
        df_agg = gridagg.df_agg_long

        # Construct bin column names from original series names
        x_bin_col = f'BIN_{x_col}'
        y_bin_col = f'BIN_{y_col}'

        # Extract Series from aggregated DataFrame
        x = df_agg[x_bin_col].copy()
        x.name = x_col  # Use original name for validation in __init__

        y = df_agg[y_bin_col].copy()
        y.name = y_col

        z = df_agg[z_col].copy()
        z.name = z_col

        # Infer axis labels from column names if not provided
        xlabel = xlabel if xlabel is not None else x_col
        ylabel = ylabel if ylabel is not None else y_col
        zlabel = zlabel if zlabel is not None else z_col

        return cls(x=x, y=y, z=z, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

    def _prepare_data(self):
        """Pivots the x/y/z Series into a 2-D grid and computes cell boundaries.

        **Expected input:** Three Series with one value per unique (x, y) coordinate pair
        (i.e., pre-aggregated data from :attr:`GridAggregator.df_agg_long`).

        Steps:

        1. Combines ``self.x``, ``self.y``, and ``self.z`` into a long-format
           DataFrame and pivots it so that unique x values form columns and
           unique y values form rows.  For pre-aggregated input, this is a
           no-op reshape that creates the required 2-D grid structure.
        2. Extracts the numeric coordinate arrays and the 2-D value matrix.
        3. Computes cell **boundary** arrays for ``pcolormesh`` (which requires
           boundaries, not centres): the median step size across all unique
           coordinates is used to append one extra boundary beyond the last
           data point.  A fallback of ``1.0`` is used when only a single unique
           coordinate value is present.

        After this method ``self.x``, ``self.y``, and ``self.z`` are replaced
        with numpy arrays suitable for ``plot_pcolormesh``.
        """
        data = {
            self.x.name: self.x,
            self.y.name: self.y,
            self.z.name: self.z,
        }
        df = pd.DataFrame.from_dict(data, orient='columns')
        pivot_df = pd.pivot_table(df, index=self.y.name, columns=self.x.name, values=self.z.name)

        x_coords = pivot_df.columns.values
        y_coords = pivot_df.index.values
        z_values = pivot_df.values

        # Compute cell step using the median diff so non-uniform grids are
        # handled gracefully.  Fall back to 1.0 when only one unique value exists.
        x_diffs = np.diff(x_coords)
        dx = float(np.median(x_diffs)) if len(x_diffs) > 0 else 1.0
        y_diffs = np.diff(y_coords)
        dy = float(np.median(y_diffs)) if len(y_diffs) > 0 else 1.0

        # Extend boundary arrays by one step beyond the last data point
        x_edges = np.append(x_coords, x_coords[-1] + dx)
        y_edges = np.append(y_coords, y_coords[-1] + dy)

        self.x = x_edges
        self.y = y_edges
        self.z = z_values

    def plot(self):
        """Renders the heatmap and applies axis formatting.

        Steps:

        1. Calls :meth:`~diive.core.plotting.heatmap_base.HeatmapBase.plot_pcolormesh`
           with ``shading='flat'`` to fill each cell with the colour of its
           lower-left boundary value.
        2. If ``show_values=True``, overlays each cell with its numeric z-value
           via :meth:`~diive.core.plotting.heatmap_base.HeatmapBase.show_vals_in_plot`.
        3. Applies custom tick positions and labels on x and/or y when
           ``xtickpos`` / ``ytickpos`` were provided.
        4. Calls :meth:`~diive.core.plotting.heatmap_base.HeatmapBase.format` to
           attach the colorbar, set axis labels, and style the spines.
        """
        self.p = self.plot_pcolormesh(shading='flat')

        if self.show_values:
            self.show_vals_in_plot()

        # Use is not None so an explicitly passed empty list is also honoured
        if self.xtickpos is not None:
            self.ax.set_xticks(self.xtickpos)
            if self.xticklabels is not None:
                self.ax.set_xticklabels(self.xticklabels)
        if self.ytickpos is not None:
            self.ax.set_yticks(self.ytickpos)
            if self.yticklabels is not None:
                self.ax.set_yticklabels(self.yticklabels)

        self.format(
            plot=self.p,
            ax_xlabel_txt=self.xlabel,
            ax_ylabel_txt=self.ylabel,
        )


def _example():
    import diive as dv

    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    nep_col = 'NEP'
    ta_col = 'Tair_f'

    # Load data, using parquet for fast loading
    df_orig = dv.load_exampledata_parquet()

    # Data between May and Sep
    df_orig = df_orig.loc[(df_orig.index.month >= 5) & (df_orig.index.month <= 9)].copy()

    # Subset
    df = df_orig[[nee_col, vpd_col, ta_col]].copy()

    # Convert units
    df[vpd_col] = df[vpd_col].multiply(0.1)  # hPa --> kPa
    df[nee_col] = df[nee_col].multiply(0.0792171)  # umol CO2 m-2 s-1 --> g CO2 m-2 30min-1
    df[nep_col] = df[nee_col].multiply(-1)  # Convert NEE to NEP, net uptake is now positive

    xcol = ta_col
    ycol = vpd_col
    zcol = nep_col

    # Aggregation to daily values
    df = df.groupby(df.index.date).agg(
        {
            xcol: ['min', 'max', 'mean'],
            ycol: ['min', 'max', 'mean'],
            zcol: 'mean'
        }
    )

    from diive.core.dfun.frames import flatten_multiindex_all_df_cols
    df = flatten_multiindex_all_df_cols(df=df)
    x = f"{xcol}_mean"
    y = f"{ycol}_mean"
    z = f"{zcol}_mean"

    q = dv.ga(
        x=df[x],
        y=df[y],
        z=df[z],
        binning_type='quantiles',
        n_bins=10,
        min_n_vals_per_bin=1,
        aggfunc='mean'
    )

    # Pre-aggregated dataframe (one row per bin)
    df_agg = q.df_agg_long

    hm = dv.heatmap_xyz(
        x=df_agg['BIN_Tair_f_mean'],
        y=df_agg['BIN_VPD_f_mean'],
        z=df_agg['NEP_mean'],
        show_values=True,
        show_values_n_dec_places=2,
        show_values_fontsize=8,
        cb_digits_after_comma=0
    )
    hm.show()


if __name__ == '__main__':
    _example()
