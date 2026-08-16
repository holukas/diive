"""Hexbin scatter plot for aggregating flux values into 2D bins of driver variables.

HexbinPlot visualizes the relationship between two driver variables (e.g., soil temperature
and water-filled pore space) and a flux variable by aggregating flux values into hexagonal
bins. This is useful for identifying patterns in high-frequency or high-volume data.

**Important:** Input Series must have no NaN values in x and y; z may contain NaNs (ignored
during aggregation).

Public name: ``dv.plotting.HexbinPlot(x, y, z, ...)``. ``normalize_axes=True`` puts both
drivers on a 0-100 percentile scale; see the class for a sample.
"""

import warnings

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from diive.core.plotting.heatmap_base import HeatmapBase
from diive.core.plotting.styles import LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle
from diive.core.utils.console import info


class HexbinPlot(HeatmapBase):
    """Hexbin scatter plot aggregating z-values into 2D bins of driver variables.

    Creates a hexagonal binning plot where flux values (z) are aggregated within bins
    defined by two driver variables (x, y). This pattern is useful for visualizing
    high-density scatter data and identifying relationships between drivers and fluxes.

    **Important:** Input Series must have one value per observation (not pre-aggregated).
    X and Y axes must have no NaN values; Z may contain NaNs (ignored during aggregation).

    Example:
        >>> import diive as dv, pandas as pd, numpy as np
        >>> ta, vpd, nee = (pd.Series(np.arange(100.), name=n) for n in ('TA', 'VPD', 'NEE'))
        >>> hm = dv.plotting.HexbinPlot(x=ta, y=vpd, z=nee, gridsize=11, normalize_axes=True)
        >>> ax = hm.plot(zlabel='Net ecosystem exchange')

    See Also:
        examples/visualization/plot_hexbin_basic.py — Hexbin variations (percentile normalization, aggregation, overlay)
    """

    def __init__(self,
                 x,
                 y,
                 z,
                 gridsize: int = 11,
                 reduce_C_function=np.median,
                 normalize_axes: bool = False,
                 mincnt: int = 1,
                 extent: tuple = None,
                 edgecolors: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 zlabel: str = None,
                 verbose: bool = False):
        """Prepare 2D scatter data for hexbin plotting (Phase 1 of two-phase design).

        Args:
            x: pandas Series with driver variable (x-axis). Must have no NaN values
            y: pandas Series with driver variable (y-axis). Must have no NaN values
            z: pandas Series with flux values to aggregate (color scale). NaNs ignored.
                x, y and z are paired on their index; three Series carrying the
                same labels in a different order are re-aligned, not zipped by
                position. Identical indexes (the usual case: three columns of one
                dataframe) are used as they are
            gridsize: Number of hexagon bins (default 11, matches matplotlib.hexbin)
            reduce_C_function: Aggregation function for z-values in each hexagon
                (default np.median). Can be np.mean, np.sum, etc.
            normalize_axes: If True, convert x/y to percentile ranks (0-100 scale)
                (default False, use original values)
            mincnt: Minimum number of data points a hexagon must hold to be drawn
                (default 1, i.e. only cells that actually contain data). Must be
                >= 1: matplotlib's cutoff is ``len(values) >= mincnt``, so
                ``mincnt=0`` passes *empty* input to ``reduce_C_function`` and is
                rejected here — see Raises
            extent: ``(xmin, xmax, ymin, ymax)`` the hexagon grid is built over,
                in the units of the *plotted* x/y — percentile ranks when
                ``normalize_axes=True``. Default None derives the extent from
                this dataset's own x/y range, so two subsets of one record
                (daytime vs nighttime, before vs after) get **different**
                hexagons and cell *i* is not the same region in both. Pass the
                same extent to every subset to compare them cell by cell
            edgecolors: Hexagon edge color (default 'none')
            xlabel: Label for x-axis (auto-inferred from x.name if None)
            ylabel: Label for y-axis (auto-inferred from y.name if None)
            zlabel: Label for colorbar (auto-inferred from z.name if None)
            verbose: Print progress and diagnostic messages (default False)

        Raises:
            ValueError: If Series have mismatched lengths or no names
            ValueError: If the three indexes differ and do not describe the same
                records, so aligning them would change the number of observations
            ValueError: If x or y contain NaN values
            ValueError: If mincnt < 1. An empty cell must render as empty, and no
                reducer makes ``mincnt=0`` correct: ``np.sum`` returns 0.0 so the
                empty cell is painted as a measured zero, ``np.max``/``np.min``
                raise, and ``np.mean``/``np.median`` return NaN (dropped by
                matplotlib anyway) at the cost of one RuntimeWarning per empty cell

        See Also:
            plot : Render the hexbin plot with matplotlib styling options
        """
        # Validate inputs
        if len(x) != len(y) or len(y) != len(z):
            raise ValueError(f"Series must have same length: x={len(x)}, y={len(y)}, z={len(z)}")

        if x.name is None or y.name is None or z.name is None:
            raise ValueError(f"All Series must have names. Got: x.name={x.name}, y.name={y.name}, z.name={z.name}")

        # The three roles used to be zipped by position (each taken through .to_numpy()),
        # so Series carrying the same labels in a different order were mispaired without
        # a word. Align on the index, with the internal keys ScatterXY and GridAggregator
        # use so a shared Series name cannot collapse two roles into one column. Identical
        # indexes skip this: the concat is a no-op there, and a legitimately duplicated
        # index (repeated timestamps) is then still paired the way the caller wrote it.
        if not (x.index.equals(y.index) and x.index.equals(z.index)):
            aligned = pd.concat([x, y, z], axis=1, keys=['_x', '_y', '_z'])
            if len(aligned) != len(x):
                raise ValueError(f"x, y and z have different indexes that do not describe the "
                                 f"same {len(x)} records: aligning them yields {len(aligned)} "
                                 f"rows. Pass Series that share an index (e.g. three columns of "
                                 f"one dataframe), or reset the index on all three to pair them "
                                 f"by position.")
            x = aligned['_x'].rename(x.name)
            y = aligned['_y'].rename(y.name)
            z = aligned['_z'].rename(z.name)

        if x.isnull().any() or y.isnull().any():
            raise ValueError("X and Y Series cannot contain NaN values (required for hexbin)")

        # matplotlib's cutoff is `len(values) >= mincnt`, so anything below 1 hands empty
        # input to reduce_C_function: np.sum paints an empty cell as a measured 0.0,
        # np.max/np.min raise, np.mean/np.median warn per empty cell. No reducer makes it
        # right, so reject it rather than document a trap.
        if mincnt < 1:
            raise ValueError(f"mincnt must be >= 1, got {mincnt}. A value below 1 includes "
                             f"hexagons holding no data, which passes empty input to "
                             f"reduce_C_function (np.sum then draws them as measured zeros, "
                             f"np.max raises). Use mincnt=1 to draw every cell that contains "
                             f"data.")

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
        self.extent = extent

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
            except Exception:
                return  # If extraction fails, don't display values

        # Place text at each hexagon center
        if len(centers) > 0:
            for center, val in zip(centers, array, strict=False):
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

    def _auto_cb_extend(self, vmin, vmax) -> str:
        """Pick the colorbar extension arrows from the drawn hexagon values.

        Reads the aggregated values off the plotted collection, so the arrows
        describe what the colour scale actually clips. Call after the hexbin is
        drawn.

        Args:
            vmin: Lower bound of the colour scale, or None for auto.
            vmax: Upper bound of the colour scale, or None for auto.

        Returns:
            str: One of ``'neither'``, ``'min'``, ``'max'``, ``'both'``.
        """
        agg = np.ma.compressed(np.ma.masked_invalid(self.p.get_array()))
        if agg.size == 0:
            return 'neither'
        clips_low = vmin is not None and vmin > agg.min()
        clips_high = vmax is not None and vmax < agg.max()
        if clips_low and clips_high:
            return 'both'
        if clips_low:
            return 'min'
        if clips_high:
            return 'max'
        return 'neither'

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
             format_style: FormatStyle = None,
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
             minticks: int = None,
             maxticks: int = None,
             color_bad: str = 'grey',
             show_colormap: bool = True,
             show_less_xticklabels: bool = False,
             show_values: bool = False,
             show_values_fontsize: float = None,
             show_values_n_dec_places: int = 0,
             show_values_color: str = 'black'):
        """Render HexbinPlot with matplotlib styling (Phase 2 of two-phase design).

        All styling and presentation parameters go here. Can be called multiple times
        on the same HexbinPlot object to plot on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure
            fig: Existing matplotlib Figure. If None and ax is None, creates new figure
            figsize: Figure size as (width, height) in inches. Only used when ax is None
            figdpi: Figure DPI. Only used when ax is None (default 72)
            format_style: Shared chrome (title/labels/fonts/ticks/spines/grid) via
                :class:`~diive.core.plotting.styles.format.FormatStyle`. None = house
                style (grid off). The ``xlabel``/``ylabel`` below still feed the
                axis labels as caller defaults; a passed format_style overrides
                them. The colorbar stays ``cb_*``/``zlabel``-controlled.
            vmin: Minimum color value (auto from data if None)
            vmax: Maximum color value (auto from data if None)
            cmap: Colormap name (default: 'RdYlBu_r')
            zlabel: Colorbar label (e.g., '°C', 'µmol m⁻²s⁻¹')
            cb_digits_after_comma: Decimal places on colorbar labels (default 2)
            cb_labelsize: Font size for colorbar tick labels
            cb_extend: Colorbar extension arrows ('neither', 'both', 'min', 'max').
                Default None derives them from ``vmin``/``vmax`` against the range
                of the *aggregated* hexagon values, which is what the colorbar maps
            minticks: Minimum major ticks per axis. Default None keeps matplotlib's
                own tick density, which adapts to the figure size
            maxticks: Maximum major ticks per axis (counted within the axis view).
                Default None keeps matplotlib's own tick density
            color_bad: Ignored by hexbin, kept for signature parity with the other
                heatmaps. ``ax.hexbin`` drops every cell whose aggregate is NaN
                (matplotlib's ``good_idxs = ~np.isnan(accum)``), so a hexbin has no
                bad cells to colour — a missing cell is blank, not ``color_bad``
            show_colormap: Whether to show colorbar (default True)
            show_less_xticklabels: Hide every second x-tick label (default False)
            show_values: Overlay numeric values on hexagons (default False)
            show_values_fontsize: Font size for value overlay text
            show_values_n_dec_places: Decimal places for value overlay (default 0)
            show_values_color: Text color for value overlay (default 'black')

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
        if show_values_fontsize is None:
            show_values_fontsize = theme.AX_LABELS_FONTSIZE

        # Call parent plot() to create figure/axes and apply styling
        super().plot(
            ax=ax,
            fig=fig,
            figsize=figsize,
            figdpi=figdpi,
            format_style=format_style,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            zlabel=zlabel,
            cb_digits_after_comma=cb_digits_after_comma,
            cb_labelsize=cb_labelsize,
            cb_extend=cb_extend,
            color_bad=color_bad,
            show_colormap=show_colormap,
            show_less_xticklabels=show_less_xticklabels,
            show_values=show_values,
            show_values_fontsize=show_values_fontsize,
            show_values_n_dec_places=show_values_n_dec_places
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
            extent=self.extent,
            cmap=cmap,
            edgecolors=edgecolors,
            linewidths=1,
            vmin=vmin,
            vmax=vmax,
            zorder=0
        )

        # Equal aspect only yields regular hexagons when both axes share a scale,
        # i.e. percentile-normalized (0-100). For raw drivers with different
        # units/ranges (e.g. air temperature vs VPD) it stretches the hexagons,
        # so leave the aspect automatic — matplotlib then tiles near-regular
        # hexagons that fill the axes box.
        if self.normalize_axes:
            self.ax.set_aspect('equal', adjustable='datalim')
            self.ax.apply_aspect()

        # The colorbar maps the per-hexagon aggregate, not the raw z, and the two ranges
        # differ in both directions (np.median narrows them, np.sum widens them far past
        # the raw range). Deriving the arrows from raw z therefore both invented arrows
        # for data that is not clipped and omitted them for data that is. The aggregate
        # only exists once ax.hexbin has run, so resolve the auto case here — before
        # format(), which is what attaches the colorbar.
        if cb_extend is None:
            self.cb_extend = self._auto_cb_extend(vmin=vmin, vmax=vmax)

        # Overlay values on hexagons if requested
        if show_values:
            self.show_vals_in_plot()

        # Apply base formatting (title, axis labels + fonts, ticks, spines, grid,
        # colorbar) via the shared FormatStyle path. The hexbin's axis labels flow
        # through as the caller defaults, so a passed format_style can override them.
        self.format(plot=self.p, ax_xlabel_txt=xlabel, ax_ylabel_txt=ylabel)

        # `minticks`/`maxticks` were forwarded to HeatmapBase, which only stores them for
        # `nice_date_ticks` — a date-axis routine hexbin never reaches, so both were inert.
        # Hexbin's axes are plain numeric driver axes, so honour them here. Left at None
        # matplotlib keeps its own locator, which sizes the tick count to the figure.
        if minticks is not None or maxticks is not None:
            for axis in (self.ax.xaxis, self.ax.yaxis):
                # One locator per axis: a Locator binds to the axis it is set on.
                axis.set_major_locator(MaxNLocator(
                    nbins=(maxticks - 1) if maxticks is not None else 'auto',
                    min_n_ticks=minticks if minticks is not None else 2,
                    steps=[1, 2, 2.5, 5, 10]))

        # HeatmapBase only stores this flag; each subclass applies it. Hexbin accepted
        # and documented it but never did, so it silently had no effect. Must run after
        # format(), which is what sets the tick labels. Same block as HeatmapDateTime.
        if self.show_less_xticklabels:
            for i, label in enumerate(self.ax.get_xticklabels()):
                if i % 2 != 0:
                    label.set_visible(False)
