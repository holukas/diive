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

import warnings

import numpy as np

from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.styles import LightTheme as theme
from diive.core.utils.console import info


class HexbinPlot(HeatmapBase):
    """Hexbin scatter plot aggregating z-values into 2D bins of driver variables.

    Creates a hexagonal binning plot where flux values (z) are aggregated within bins
    defined by two driver variables (x, y). This pattern is useful for visualizing
    high-density scatter data and identifying relationships between drivers and fluxes.

    **Important:** Input Series must have one value per observation (not pre-aggregated).
    X and Y axes must have no NaN values; Z may contain NaNs (ignored during aggregation).

    Top-level alias: ``dv.hexbinplot(x, y, z, ...)``

    See Also:
        examples/visualization/hexbin.py — Hexbin variations (percentile normalization, aggregation, overlay)
    """

    def __init__(self,
                 x,
                 y,
                 z,
                 gridsize: int = 11,
                 reduce_C_function=np.median,
                 normalize_axes: bool = False,
                 mincnt: int = 0,
                 edgecolors: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 zlabel: str = None,
                 verbose: bool = False):
        """Prepare 2D scatter data for hexbin plotting (Phase 1 of two-phase design).

        Args:
            x: pandas Series with driver variable (x-axis). Must have no NaN values
            y: pandas Series with driver variable (y-axis). Must have no NaN values
            z: pandas Series with flux values to aggregate (color scale). NaNs ignored
            gridsize: Number of hexagon bins (default 11, matches matplotlib.hexbin)
            reduce_C_function: Aggregation function for z-values in each hexagon
                (default np.median). Can be np.mean, np.sum, etc.
            normalize_axes: If True, convert x/y to percentile ranks (0-100 scale)
                (default False, use original values)
            mincnt: Minimum number of data points per hexagon (default 0)
            edgecolors: Hexagon edge color (default 'none')
            xlabel: Label for x-axis (auto-inferred from x.name if None)
            ylabel: Label for y-axis (auto-inferred from y.name if None)
            zlabel: Label for colorbar (auto-inferred from z.name if None)
            verbose: Print progress and diagnostic messages (default False)

        Raises:
            ValueError: If Series have mismatched lengths or no names
            ValueError: If x or y contain NaN values

        See Also:
            plot : Render the hexbin plot with matplotlib styling options
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
            if verbose:
                info(f"Z Series contains {n_nan} NaN values (will be ignored during aggregation)", verbose=verbose)

        # Call parent init with only heatmaptype and verbose
        super().__init__(heatmaptype='hexbin', verbose=verbose)

        # Store data computation parameters
        self.gridsize = gridsize
        self.reduce_C_function = reduce_C_function
        self.normalize_axes = normalize_axes
        self.mincnt = mincnt

        # Styling belongs in plot(); these are kept here only as deprecated
        # pass-throughs (labels still auto-default from the data's .name).
        if any(v is not None for v in (edgecolors, xlabel, ylabel, zlabel)):
            warnings.warn("HexbinPlot: `edgecolors`/`xlabel`/`ylabel`/`zlabel` in the constructor "
                          "are deprecated; pass them to plot() instead.", DeprecationWarning, stacklevel=2)
        self.edgecolors = edgecolors

        # Store original Series
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

        # Set default labels
        self.xlabel = xlabel if xlabel is not None else x.name
        self.ylabel = ylabel if ylabel is not None else y.name
        self.zlabel = zlabel if zlabel is not None else z.name

        self.p = None  # Hexbin collection object (created in plot())

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

    def plot(self,
             ax=None,
             fig=None,
             figsize: tuple = None,
             figdpi: int = 72,
             title: str = None,
             vmin: float = None,
             vmax: float = None,
             cmap: str = 'RdYlBu_r',
             zlabel: str = None,
             xlabel: str = None,
             ylabel: str = None,
             edgecolors: str = None,
             cb_digits_after_comma: int = 2,
             cb_labelsize: float = None,
             cb_extend: str = None,
             axlabels_fontsize: float = None,
             ticks_labelsize: float = None,
             minticks: int = 3,
             maxticks: int = 10,
             color_bad: str = 'grey',
             show_colormap: bool = True,
             show_less_xticklabels: bool = False,
             show_values: bool = False,
             show_values_fontsize: float = None,
             show_values_n_dec_places: int = 0,
             show_values_color: str = 'black',
             show_grid: bool = False):
        """Render HexbinPlot with matplotlib styling (Phase 2 of two-phase design).

        All styling and presentation parameters go here. Can be called multiple times
        on the same HexbinPlot object to plot on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure
            fig: Existing matplotlib Figure. If None and ax is None, creates new figure
            figsize: Figure size as (width, height) in inches. Only used when ax is None
            figdpi: Figure DPI. Only used when ax is None (default 72)
            title: Plot title (auto-generated if None)
            vmin: Minimum color value (auto from data if None)
            vmax: Maximum color value (auto from data if None)
            cmap: Colormap name (default: 'RdYlBu_r')
            zlabel: Colorbar label (e.g., '°C', 'µmol m⁻²s⁻¹')
            cb_digits_after_comma: Decimal places on colorbar labels (default 2)
            cb_labelsize: Font size for colorbar tick labels
            cb_extend: Colorbar extension arrows ('neither', 'both', 'min', 'max')
            axlabels_fontsize: Font size for axis labels
            ticks_labelsize: Font size for tick labels
            minticks: Minimum major ticks on axes (default 3)
            maxticks: Maximum major ticks on axes (default 10)
            color_bad: Color for NaN cells (default 'grey')
            show_colormap: Whether to show colorbar (default True)
            show_less_xticklabels: Hide every second x-tick label (default False)
            show_values: Overlay numeric values on hexagons (default False)
            show_values_fontsize: Font size for value overlay text
            show_values_n_dec_places: Decimal places for value overlay (default 0)
            show_values_color: Text color for value overlay (default 'black')
            show_grid: Show gridlines (default False)

        Returns:
            None (displays plot if ax=None, otherwise renders on provided axes)
        """
        # Use the provided styling, or fall back to the (deprecated) __init__
        # value. Labels auto-default from the data's .name (set in __init__).
        if zlabel is None:
            zlabel = self.zlabel
        if xlabel is None:
            xlabel = self.xlabel
        if ylabel is None:
            ylabel = self.ylabel
        if edgecolors is None:
            edgecolors = self.edgecolors if self.edgecolors is not None else 'none'

        # Use theme defaults if not provided
        if cb_labelsize is None:
            cb_labelsize = theme.AX_LABELS_FONTSIZE
        if axlabels_fontsize is None:
            axlabels_fontsize = theme.AX_LABELS_FONTSIZE
        if ticks_labelsize is None:
            ticks_labelsize = theme.TICKS_LABELS_FONTSIZE
        if show_values_fontsize is None:
            show_values_fontsize = theme.AX_LABELS_FONTSIZE

        # Determine colorbar extension based on vmin/vmax vs data range if not provided
        if cb_extend is None:
            z_min = self.z.min()
            z_max = self.z.max()
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

        # Call parent plot() to create figure/axes and apply styling
        super().plot(
            ax=ax,
            fig=fig,
            figsize=figsize,
            figdpi=figdpi,
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            zlabel=zlabel,
            cb_digits_after_comma=cb_digits_after_comma,
            cb_labelsize=cb_labelsize,
            cb_extend=cb_extend,
            axlabels_fontsize=axlabels_fontsize,
            ticks_labelsize=ticks_labelsize,
            minticks=minticks,
            maxticks=maxticks,
            color_bad=color_bad,
            show_colormap=show_colormap,
            show_less_xticklabels=show_less_xticklabels,
            show_values=show_values,
            show_values_fontsize=show_values_fontsize,
            show_values_n_dec_places=show_values_n_dec_places,
            show_grid=show_grid
        )

        # Store styling parameters for show_vals_in_plot()
        self.show_values = show_values
        self.show_values_n_dec_places = show_values_n_dec_places
        self.show_values_fontsize = show_values_fontsize
        self.show_values_color = show_values_color

        # Domain-specific rendering (hexbin plot)
        self.p = self.ax.hexbin(
            self.x.to_numpy(), self.y.to_numpy(),
            C=self.z.to_numpy(),
            gridsize=self.gridsize,
            reduce_C_function=self.reduce_C_function,
            mincnt=self.mincnt,
            cmap=cmap,
            edgecolors=edgecolors,
            linewidths=1,
            vmin=vmin,
            vmax=vmax,
            zorder=0
        )

        # Set equal aspect ratio (hexagons appear as regular hexagons, not skewed)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.apply_aspect()

        # Format axes with styling
        self.ax.set_xlabel(xlabel, fontsize=axlabels_fontsize)
        self.ax.set_ylabel(ylabel, fontsize=axlabels_fontsize)
        self.ax.xaxis.set_tick_params(labelsize=ticks_labelsize)
        self.ax.yaxis.set_tick_params(labelsize=ticks_labelsize)

        # Overlay values on hexagons if requested
        if show_values:
            self.show_vals_in_plot()

        # Apply base formatting (title, colorbar, spines, grid, etc.)
        self.format(plot=self.p, ax_xlabel_txt=xlabel, ax_ylabel_txt=ylabel)
