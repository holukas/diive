"""
WATERFALL: CUMULATIVE WATERFALL PLOT
====================================

Financial-style waterfall chart for sequential contributions to a running total,
e.g. daily CO2 uptake/release building up a seasonal or annual flux budget.

Part of the diive library: https://github.com/holukas/diive
"""

from pandas import Series

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle


class WaterfallPlot:
    """Plot a financial-style waterfall chart of sequential contributions to a running total.

    Each period is drawn as a floating bar starting where the previous period's
    running total ended, so the cumulative sum builds up across the series. Bars
    are colored by sign (e.g. CO2 uptake vs. release), with thin connector lines
    between consecutive bars and the final running total annotated.

    Designed for daily CO2 flux budgets: pass the flux at its native resolution
    (e.g. half-hourly NEE) and it aggregates to one bar per period internally.

    Two-phase design: data preparation (__init__) is separate from rendering (plot).

    Args:
        series: Flux time series with a datetime index. Aggregated to one value
            per period internally (see `resample`).
        series_units: Units string for the y-axis (e.g. r'($\\mathrm{gC\\ m^{-2}}$)').
            LaTeX is supported.
        resample: Pandas offset alias to aggregate the series before plotting
            (default 'D' for daily). Set to None to plot the series as-is, e.g.
            when it is already aggregated to one value per period.
        agg: Aggregation function passed to the resampler (default 'sum').
        uptake_is_negative: Sign convention for coloring. With the default True
            (NEE convention), negative values are treated as uptake (sink) and
            colored blue, positive values as release (source) and colored red.
            Set to False when the data is sign-flipped so positive means uptake.

    Methods:
        plot : Render the waterfall chart with styling options
        get : Return the axis

    See Also:
        examples/visualization/plot_waterfall.py — Daily CO2 uptake waterfall
    """

    def __init__(self,
                 series: Series,
                 series_units: str = None,
                 resample: str = 'D',
                 agg: str = 'sum',
                 uptake_is_negative: bool = True):
        """Aggregate the series and compute the running totals. See the class docstring for parameters."""
        self.series_units = series_units
        self.varname = series.name
        self.uptake_is_negative = uptake_is_negative

        s = series.dropna()
        # Aggregate to one bar per period unless the caller opted out.
        self.contributions = s.resample(resample).agg(agg).dropna() if resample else s.copy()

        # Each bar floats from the previous running total to the new one.
        self.cumulative = self.contributions.cumsum()
        self.bar_bottoms = self.cumulative.shift(1).fillna(0.0)

        # Figure/axis are created in plot() (phase 2), or supplied by the caller.
        self.fig = None
        self.ax = None
        self._own_fig = False

    def _apply_format(self, style: FormatStyle):
        auto_title = (f"Cumulative waterfall ({self.contributions.index.min().date()} to "
                      f"{self.contributions.index.max().date()})")

        # When this object owns the figure the title is a FIGHEADER suptitle, not
        # an axes title — let the shared chrome draw everything else but the title,
        # which we place as a suptitle here to preserve the figure-header look.
        if self._own_fig:
            shown_title = style.title if style.title is not None else auto_title
            if shown_title:
                self.fig.suptitle(shown_title, fontsize=theme.FIGHEADER_FONTSIZE)
            style.merged(title="").apply(
                ax=self.ax, default_xlabel="Date",
                default_ylabel=f"Cumulative {self.varname}", zeroline_data=self.cumulative)
        else:
            style.apply(ax=self.ax, default_title=auto_title, default_xlabel="Date",
                        default_ylabel=f"Cumulative {self.varname}", zeroline_data=self.cumulative)

        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

    def get(self):
        """Return axis"""
        return self.ax

    def plot(self, ax=None, format_style: FormatStyle = None, showplot: bool = True,
             digits_after_comma: int = 0,
             color_uptake: str = '#2196F3', color_release: str = '#F44336',
             bar_width: float = None, show_connectors: bool = True):
        """Render the waterfall chart (Phase 2 of the two-phase design).

        Chrome (title, labels, units, font sizes, colours, grid, legend, zero line)
        comes from a shared :class:`~diive.plotting.FormatStyle`, so it looks and is
        configured like every other diive plot. The bar/connector/annotation
        rendering arguments stay here.

        Args:
            ax: Matplotlib axes to plot on. If None, a new figure is created and shown.
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome.
                When None the diive house style is used. The ``series_units`` passed at
                construction is folded onto it as the y-axis units for backward
                compatibility.
            showplot: Show the figure (only when this object created the figure).
            digits_after_comma: Decimals for the final-total annotation.
            color_uptake: Bar color for uptake (sink) periods.
            color_release: Bar color for release (source) periods.
            bar_width: Bar width in days. Defaults to ~80% of the median spacing
                between periods.
            show_connectors: Draw thin lines linking consecutive bars.

        Returns:
            The matplotlib axis.
        """
        # Fold the legacy series_units onto the (copied) style so old call sites keep
        # working without mutating a caller-supplied FormatStyle.
        style = (format_style or FormatStyle()).merged(yunits=self.series_units)
        if ax is None:
            self.fig, self.ax = pf.create_ax()
            self._own_fig = True
        else:
            self.ax = ax
            self.fig = ax.get_figure()
            self._own_fig = False
        self.ax.xaxis.axis_date()

        # Uptake/release split depends on the sign convention.
        uptake_mask = self.contributions < 0 if self.uptake_is_negative else self.contributions > 0
        colors = uptake_mask.map({True: color_uptake, False: color_release})

        # Default bar width to ~80% of the median period spacing (in days).
        if bar_width is None:
            spacing_days = self.contributions.index.to_series().diff().dt.total_seconds().median() / 86400
            bar_width = (spacing_days if spacing_days and spacing_days > 0 else 1.0) * 0.8

        self.ax.bar(self.contributions.index, self.contributions.values,
                    bottom=self.bar_bottoms.values,
                    width=bar_width, color=colors.values, edgecolor='none',
                    zorder=10, label='_nolegend_')

        # Connectors run from each bar's running total to the next bar's start.
        if show_connectors and len(self.cumulative) > 1:
            for x0, x1, y in zip(self.cumulative.index[:-1], self.cumulative.index[1:],
                                 self.cumulative.values[:-1]):
                self.ax.plot([x0, x1], [y, y], color=theme.COLOR_LINE_ZERO,
                             lw=theme.LINEWIDTH_SPINES, alpha=0.4, zorder=9)

        # Annotate the final running total.
        x_last = self.cumulative.index[-1]
        y_last = self.cumulative.iloc[-1]
        self.ax.plot(x_last, y_last, marker='o', ms=8, color=theme.COLOR_LINE_ZERO, zorder=11)
        self.ax.text(x_last, y_last, f"  {y_last:.{digits_after_comma}f}",
                     size=12, color=theme.COLOR_LINE_ZERO,
                     horizontalalignment='left', verticalalignment='center', zorder=11)

        self._apply_format(style)

        if self._own_fig and showplot:
            self.fig.show()

        return self.ax
