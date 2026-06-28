"""
FORMAT_STYLE: SHARED PLOT FORMATTING
====================================

A single, reusable description of how a diive plot's *chrome* looks — title,
axis labels (and units), font sizes, text/spine/tick colours, grid, legend, and
the optional zero reference line. One :class:`FormatStyle` instance can be built
once and handed to any plot class's ``plot(..., format_style=...)`` method, so
every visualization shares the same look and the same parameter vocabulary.

Every field left at *None* falls back to the standard defined in
:mod:`diive.core.plotting.styles.LightTheme`, so the default ``FormatStyle()``
*is* the diive house style. Override only what you need:

    style = FormatStyle(title="Air temperature", yunits="(°C)", show_legend=False)
    dv.plotting.TimeSeries(series).plot(format_style=style)

The actual matplotlib calls live in :meth:`FormatStyle.apply`, which reuses the
shared helpers in :mod:`diive.core.plotting.plotfuncs` so there is one
implementation of "the standard look" for the whole package.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pandas import DataFrame, Series

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.styles import LightTheme as theme


@dataclass
class FormatStyle:
    """Shared, reusable formatting for any diive plot.

    Holds only the cross-cutting *chrome* common to every plot type (title,
    labels, units, font sizes, colours, grid, legend, zero line). Per-plot
    rendering choices (line width, marker, colormap, ...) stay as arguments on
    each ``plot()`` method — they are not the same across plot types.

    All ``None`` font-size/colour fields resolve to the standard values in
    :mod:`~diive.core.plotting.styles.LightTheme` when :meth:`apply` runs, so an
    unmodified ``FormatStyle()`` reproduces the diive house style.

    Text content (``title``/``xlabel``/``ylabel``/``zlabel``) left at *None*
    means "let the plot decide" — the caller passes its own auto value (e.g. the
    series name) as the ``default_*`` argument of :meth:`apply`.

    Args:
        title: Plot title. *None* -> caller's default; ``""`` -> no title.
        xlabel: X-axis label. *None* -> caller's default.
        ylabel: Y-axis label. *None* -> caller's default.
        zlabel: Colorbar / z-axis label (only used by plots that have one).
        xunits: Units appended to the x-axis label, e.g. ``"(°C)"``.
        yunits: Units appended to the y-axis label.
        title_fontsize: Title font size. *None* -> ``theme.FONTSIZE_TITLE``.
        axlabel_fontsize: Axis-label font size. *None* -> ``theme.FONTSIZE_AXLABEL``.
        ticks_fontsize: Tick-label font size. *None* -> ``theme.FONTSIZE_TICKS``.
        legend_fontsize: Legend font size. *None* -> ``theme.FONTSIZE_LEGEND``.
        title_fontweight: Font weight for the title (default ``'bold'``).
        axlabel_fontweight: Font weight for the axis labels (default ``'normal'``).
        text_color: Colour of title + axis labels. *None* -> ``theme.COLOR_TEXT``.
        chrome_color: Colour of ticks + spines. *None* -> ``theme.COLOR_CHROME``.
        facecolor: Axes background colour. *None* -> ``theme.COLOR_FACE``.
        grid_color: Gridline colour. *None* -> ``theme.COLOR_LINE_GRID``.
        spine_linewidth: Spine line width. *None* -> ``theme.LINEWIDTH_SPINES``.
        ticks_direction: Tick direction, ``'in'`` (default) / ``'out'`` / ``'inout'``.
        ticks_length: Tick length. *None* -> ``theme.TICKS_LENGTH``.
        ticks_width: Tick width. *None* -> ``theme.TICKS_WIDTH``.
        show_grid: Draw the grid (default *True*).
        show_legend: Draw a legend when the axes has labelled artists (default *True*).
        legend_loc: Legend location (matplotlib ``loc``, default ``'best'``).
        legend_ncol: Number of legend columns (default 1).
        show_zeroline: Draw a horizontal line at y=0 when the data straddles zero
            (default *True*; only drawn when :meth:`apply` is given the data).
    """

    # --- text content (None -> caller-supplied default / auto) ---
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    xunits: str | None = None
    yunits: str | None = None

    # --- font sizes (None -> theme standard) ---
    title_fontsize: float | None = None
    axlabel_fontsize: float | None = None
    ticks_fontsize: float | None = None
    legend_fontsize: float | None = None

    # --- font weights ---
    title_fontweight: str = theme.FONTWEIGHT_TITLE
    axlabel_fontweight: str = theme.FONTWEIGHT_AXLABEL

    # --- colours (None -> theme standard) ---
    text_color: str | None = None
    chrome_color: str | None = None
    facecolor: str | None = None
    grid_color: str | None = None

    # --- spine / tick geometry ---
    spine_linewidth: float | None = None
    ticks_direction: str = theme.TICKS_DIRECTION
    ticks_length: float | None = None
    ticks_width: float | None = None

    # --- toggles ---
    show_grid: bool = True
    show_legend: bool = True
    legend_loc: int | str = 'best'
    legend_ncol: int = 1
    show_zeroline: bool = True

    def merged(self, **overrides) -> "FormatStyle":
        """Return a copy with the given non-*None* fields overridden.

        Used by ``plot()`` methods to fold legacy flat keyword arguments
        (``title=``, ``xlabel=``, ...) onto a caller-supplied style without
        mutating it. Keys whose value is *None* are ignored, so an unset legacy
        argument never clobbers a value already on the style.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self

    def apply(self,
              ax,
              default_title: str = None,
              default_xlabel: str = None,
              default_ylabel: str = None,
              zeroline_data: Series | DataFrame = None) -> None:
        """Apply this style to a matplotlib axes.

        Resolves every *None* field to its :mod:`LightTheme` default, then sets
        the facecolor, ticks, spines, axis labels (with units), title, grid,
        optional zero line, and legend using the shared
        :mod:`~diive.core.plotting.plotfuncs` helpers.

        Args:
            ax: The matplotlib axes to format.
            default_title: Title to use when ``self.title`` is *None*
                (e.g. the series name). ``self.title == ""`` suppresses it.
            default_xlabel: X-label to use when ``self.xlabel`` is *None*.
            default_ylabel: Y-label to use when ``self.ylabel`` is *None*.
            zeroline_data: Data used to decide whether to draw the y=0 line
                (only drawn when ``show_zeroline`` and the data straddles zero).
        """
        # Resolve None -> theme standard.
        title_fs = self.title_fontsize if self.title_fontsize is not None else theme.FONTSIZE_TITLE
        axlabel_fs = self.axlabel_fontsize if self.axlabel_fontsize is not None else theme.FONTSIZE_AXLABEL
        ticks_fs = self.ticks_fontsize if self.ticks_fontsize is not None else theme.FONTSIZE_TICKS
        legend_fs = self.legend_fontsize if self.legend_fontsize is not None else theme.FONTSIZE_LEGEND
        text_color = self.text_color if self.text_color is not None else theme.COLOR_TEXT
        chrome_color = self.chrome_color if self.chrome_color is not None else theme.COLOR_CHROME
        facecolor = self.facecolor if self.facecolor is not None else theme.COLOR_FACE
        grid_color = self.grid_color if self.grid_color is not None else theme.COLOR_LINE_GRID
        ticks_length = self.ticks_length if self.ticks_length is not None else theme.TICKS_LENGTH
        ticks_width = self.ticks_width if self.ticks_width is not None else theme.TICKS_WIDTH

        # Background
        ax.set_facecolor(facecolor)

        # Ticks + spines
        pf.format_ticks(ax=ax, width=ticks_width, length=ticks_length,
                        direction=self.ticks_direction, color=chrome_color, labelsize=ticks_fs)
        pf.format_spines(ax=ax, color=chrome_color, lw=self.spine_linewidth)

        # Axis labels (+ optional units)
        xlabel = self.xlabel if self.xlabel is not None else default_xlabel
        ylabel = self.ylabel if self.ylabel is not None else default_ylabel
        if xlabel and self.xunits:
            xlabel = f"{xlabel} {self.xunits}"
        if ylabel and self.yunits:
            ylabel = f"{ylabel} {self.yunits}"
        ax.set_xlabel(xlabel or "", color=text_color, fontsize=axlabel_fs, fontweight=self.axlabel_fontweight)
        ax.set_ylabel(ylabel or "", color=text_color, fontsize=axlabel_fs, fontweight=self.axlabel_fontweight)

        # Title (None -> caller default; "" -> suppressed)
        title = self.title if self.title is not None else default_title
        if title:
            ax.set_title(title, color=text_color, fontsize=title_fs, fontweight=self.title_fontweight)

        # Grid
        if self.show_grid:
            ax.grid(True, ls='--', color=grid_color, lw=theme.LINEWIDTH_SPINES, zorder=0)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)

        # Zero reference line (only when data crosses zero)
        if self.show_zeroline and zeroline_data is not None:
            pf.add_zeroline_y(data=zeroline_data, ax=ax)

        # Legend (only if there are labelled artists)
        if self.show_legend and ax.get_legend_handles_labels()[0]:
            pf.default_legend(ax=ax, loc=self.legend_loc, ncol=self.legend_ncol,
                              textsize=legend_fs, textcolor=text_color)
