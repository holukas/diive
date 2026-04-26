"""Hexbin scatter plot for aggregating flux values into 2D bins of driver variables.

HexbinPlot visualizes the relationship between two driver variables (e.g., soil temperature
and water-filled pore space) and a flux variable by aggregating flux values into hexagonal
bins. This is useful for identifying patterns in high-frequency or high-volume data.

**Important:** Input Series must have no NaN values in x and y; z may contain NaNs (ignored
during aggregation).

Top-level alias: ``dv.hexbinplot(x, y, z, ...)``

Example with percentile normalization::

    import diive as dv
    df = dv.load_exampledata_parquet()
    hm = dv.hexbinplot(
        x=df['Tair_f'],
        y=df['VPD_f'],
        z=df['NEE_CUT_REF_f'],
        normalize_axes=True,  # Convert drivers to 0-100 percentile scale
        gridsize=11,          # Number of hexagon bins
        xlabel='Air temperature (percentile)',
        ylabel='Vapor pressure deficit (percentile)',
        zlabel='Net ecosystem exchange'
    )
    hm.show()

Example with absolute values::

    hm = dv.hexbinplot(
        x=df['Tair_f'],
        y=df['VPD_f'],
        z=df['NEE_CUT_REF_f'],
        normalize_axes=False,  # Use original values
        gridsize=11,
        xlabel='Air temperature (°C)',
        ylabel='Vapor pressure deficit (hPa)',
        zlabel='NEE (µmol m⁻² s⁻¹)'
    )
    hm.show()
"""

import numpy as np

from diive.core.plotting.heatmap_base import HeatmapBase


class HexbinPlot(HeatmapBase):
    """Hexbin scatter plot aggregating z-values into 2D bins of driver variables.

    Creates a hexagonal binning plot where flux values (z) are aggregated within bins
    defined by two driver variables (x, y). This pattern is useful for visualizing
    high-density scatter data and identifying relationships between drivers and fluxes.

    **Important:** Input Series must have one value per observation (not pre-aggregated).
    X and Y axes must have no NaN values; Z may contain NaNs (ignored during aggregation).

    Top-level alias: ``dv.hexbinplot(x, y, z, ...)``

    Example:
        See `examples/visualization/hexbin.py` for complete examples
        including percentile normalization, mean aggregation, and value overlay.
    """

    def __init__(self,
                 x,
                 y,
                 z,
                 xlabel=None,
                 ylabel=None,
                 zlabel=None,
                 gridsize=11,
                 reduce_C_function=np.median,
                 normalize_axes=False,
                 mincnt=0,
                 edgecolors='none',
                 vmin=None,
                 vmax=None,
                 show_values=False,
                 show_values_n_dec_places=2,
                 show_values_fontsize=8,
                 show_values_color='black',
                 **kwargs):
        """
        Args:
            x: pandas Series with driver variable (x-axis)
            y: pandas Series with driver variable (y-axis)
            z: pandas Series with flux values to aggregate (color scale)
            xlabel: Label for x-axis (auto-inferred from x.name if None)
            ylabel: Label for y-axis (auto-inferred from y.name if None)
            zlabel: Label for colorbar (auto-inferred from z.name if None)
            gridsize: Number of hexagon bins (default 11, matches matplotlib.hexbin)
            reduce_C_function: Aggregation function for z-values (default np.median)
                Can be np.mean, np.sum, etc.
            normalize_axes: If True, convert x/y to percentile ranks (0-100 scale)
                (default False, use original values)
            mincnt: Minimum number of data points per hexagon (default 0)
            edgecolors: Hexagon edge color (default 'none')
            vmin: Minimum value for color scale (default None, auto-scaled)
            vmax: Maximum value for color scale (default None, auto-scaled)
            show_values: If True, overlay aggregated z-values on hexagons (default False)
            show_values_n_dec_places: Number of decimal places for displayed values (default 2)
            show_values_fontsize: Font size for displayed values (default 8)
            show_values_color: Text color for displayed values (default 'black')
            **kwargs: Additional arguments passed to HeatmapBase
                (figsize, cmap, title, cb_digits_after_comma, verbose, etc.)
                cb_digits_after_comma: Decimal places for colorbar labels (default 2)

        Raises:
            ValueError: If Series have mismatched lengths or no names
            ValueError: If x or y contain NaN values
        """
        # Validate inputs
        if len(x) != len(y) or len(y) != len(z):
            raise ValueError(f"Series must have same length: x={len(x)}, y={len(y)}, z={len(z)}")

        if x.name is None or y.name is None or z.name is None:
            raise ValueError(f"All Series must have names. Got: x.name={x.name}, y.name={y.name}, z.name={z.name}")

        if x.isnull().any() or y.isnull().any():
            raise ValueError("X and Y Series cannot contain NaN values (required for hexbin)")

        # Warn if z has NaNs
        if z.isnull().any():
            n_nan = z.isnull().sum()
            if hasattr(self, 'verbose') and kwargs.get('verbose', False):
                print(f"Info: Z Series contains {n_nan} NaN values (will be ignored during aggregation)")

        # Store parameters before parent init (except show_values which will be set after)
        self.gridsize = gridsize
        self.reduce_C_function = reduce_C_function
        self.normalize_axes = normalize_axes
        self.mincnt = mincnt
        self.edgecolors = edgecolors
        self.vmin = vmin
        self.vmax = vmax

        # Set default labels before parent init
        if xlabel is None:
            xlabel = x.name
        if ylabel is None:
            ylabel = y.name
        if zlabel is None:
            zlabel = z.name

        # Determine colorbar extension based on vmin/vmax vs data range
        z_min = z.min()
        z_max = z.max()
        cb_extend = 'neither'
        if vmin is not None and vmax is not None:
            if vmin > z_min and vmax < z_max:
                cb_extend = 'both'
            elif vmin > z_min:
                cb_extend = 'min'
            elif vmax < z_max:
                cb_extend = 'max'
        elif vmin is not None and vmin > z_min:
            cb_extend = 'min'
        elif vmax is not None and vmax < z_max:
            cb_extend = 'max'

        # Call parent init with vmin/vmax and cb_extend (this sets self.x, self.y, self.z = None)
        super().__init__(heatmaptype='hexbin', vmin=vmin, vmax=vmax, cb_extend=cb_extend, **kwargs)

        # Now assign data AFTER parent init (so they don't get overwritten)
        self.x_orig = x.copy()
        self.y_orig = y.copy()
        self.z_orig = z.copy()

        # Normalize if requested
        if normalize_axes:
            self.x = self._percentile_normalize(x)
            self.y = self._percentile_normalize(y)
        else:
            self.x = x.copy()
            self.y = y.copy()

        self.z = z.copy()

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        # Set show_values parameters AFTER parent init (parent class resets them)
        self.show_values = show_values
        self.show_values_n_dec_places = show_values_n_dec_places
        self.show_values_fontsize = show_values_fontsize
        self.show_values_color = show_values_color

        self.p = None  # Hexbin collection object

    def show_vals_in_plot(self):
        """Overlay aggregated z-values on hexagon centers.

        Extracts the hexagon centers from the plotted hexagons and places
        text annotations showing aggregated values.
        """
        # Get the aggregated values (C array)
        array = self.p.get_array()

        if array is None or len(array) == 0:
            return  # No data to display

        # Get the individual polygon paths from the PolyCollection
        # For hexbin, we need to extract vertices from each polygon
        offsets = self.p.get_offsets()

        # If offsets work, use them as centers
        if offsets is not None and len(offsets) > 0:
            centers = offsets
        else:
            # Fallback: extract from polygon vertices
            try:
                paths = self.p.get_paths()
                centers = []

                # If there's only one path, it might be a compound path
                if len(paths) == 1:
                    # Extract individual polygons from the compound path
                    path = paths[0]
                    codes = path.codes
                    vertices = path.vertices

                    # Find MOVETO commands which indicate new polygons
                    if codes is not None:
                        polygon_starts = [i for i, code in enumerate(codes) if code == 1]  # MOVETO = 1
                        polygon_starts.append(len(vertices))  # Add end marker

                        for j in range(len(polygon_starts) - 1):
                            start = polygon_starts[j]
                            end = polygon_starts[j + 1]
                            hex_vertices = vertices[start:end]
                            if len(hex_vertices) > 0:
                                center = hex_vertices.mean(axis=0)
                                centers.append(center)
                    else:
                        # No codes, try to split by fixed size (hexagon = 6 vertices + 1 close)
                        vertices_per_hex = 7
                        for j in range(0, len(vertices), vertices_per_hex):
                            hex_vertices = vertices[j:j + vertices_per_hex]
                            if len(hex_vertices) > 0:
                                center = hex_vertices.mean(axis=0)
                                centers.append(center)
                else:
                    # Multiple paths, each should be a hexagon
                    for path in paths:
                        center = path.vertices.mean(axis=0)
                        centers.append(center)

                centers = np.array(centers) if centers else np.array([])
            except:
                return  # If extraction fails, don't display values

        # Place text at each hexagon center
        if len(centers) > 0:
            for center, val in zip(centers, array):
                if not np.isnan(val):  # Skip NaN values
                    x_center, y_center = center
                    val_str = f"{val:.{self.show_values_n_dec_places}f}"
                    self.ax.text(
                        x_center, y_center, val_str,
                        ha='center', va='center',
                        fontsize=self.show_values_fontsize,
                        color=self.show_values_color,
                        zorder=10
                    )

    @staticmethod
    def _percentile_normalize(series):
        """Convert Series values to percentile ranks (0-100 scale).

        Args:
            series: pandas Series with numeric values

        Returns:
            pandas Series with percentile ranks in range [0, 100]
        """
        # Use rank with pct=True to get percentiles (0-1), then scale to 0-100
        percentiles = series.rank(pct=True) * 100
        percentiles.name = series.name
        return percentiles

    def plot(self):
        """Create hexbin plot and apply formatting.

        Steps:
        1. Call matplotlib.hexbin() with x, y, z and aggregation function
        2. Set equal aspect ratio for hexagons
        3. Format axes (labels, ticks, limits)
        4. Apply HeatmapBase styling (title, colorbar, spines, grid, etc.)
        """
        # Create hexbin plot (convert Series to numpy arrays)
        self.p = self.ax.hexbin(
            self.x.values, self.y.values,
            C=self.z.values,
            gridsize=self.gridsize,
            reduce_C_function=self.reduce_C_function,
            mincnt=self.mincnt,
            cmap=self.cmap,
            edgecolors=self.edgecolors,
            linewidths=1,  # Hexagon border line width
            vmin=self.vmin,
            vmax=self.vmax,
            zorder=0
        )

        # Set linewidth for all lines (axes, spines, etc.)
        self.p.set_linewidth(1)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1)

        # Set equal aspect ratio to make hexagons appear as regular hexagons (not skewed)
        # Use 'datalim' to adjust data limits so hexagons maintain 1:1 aspect in data space
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.apply_aspect()

        # Format axes
        self.ax.set_xlabel(self.xlabel, fontsize=self.axlabels_fontsize)
        self.ax.set_ylabel(self.ylabel, fontsize=self.axlabels_fontsize)
        self.ax.xaxis.set_tick_params(labelsize=self.ticks_labelsize)
        self.ax.yaxis.set_tick_params(labelsize=self.ticks_labelsize)

        # Overlay values on hexagons if requested
        if self.show_values:
            self.show_vals_in_plot()

        # Apply base formatting (title, colorbar, spines, grid, etc.)
        # HeatmapBase.format() will handle the colorbar creation
        self.format(plot=self.p, ax_xlabel_txt=self.xlabel, ax_ylabel_txt=self.ylabel)
