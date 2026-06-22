"""
PLOTTING: CUMULATIVE SUMS
=========================

Cumulative-sum plots: per-year overlays (:class:`CumulativeYear`) and per-column
cumulatives (:class:`Cumulative`).

Part of the diive library: https://github.com/holukas/diive
"""
import warnings

from pandas import Series
from pandas.core.interchange.dataframe_protocol import DataFrame

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle
from diive.core.times.times import remove_after_date, keep_years, doy_cumulatives_per_year, doy_mean_cumulative


class CumulativeYear:
    """Plot yearly cumulative sums with reference mean and standard deviation.

    Visualizes cumulative sums for each year with optional reference period
    mean and ±1 standard deviation band for comparison.

    See Also:
        examples/visualization/timeseries_and_cumulative.py — Yearly cumulative plots with reference bands
    """

    def __init__(self,
                 series: Series,
                 series_units: str = None,
                 yearly_end_date: str = None,
                 start_year: int = None,
                 end_year: int = None,
                 show_reference: bool = False,
                 excl_years_from_reference: list = None,
                 highlight_year: int = None,
                 highlight_year_color: str = None):
        """

        Args:
            series: Data for cumulative
            highlight_year_color: Deprecated — pass this styling option to plot() instead.
            yearly_end_date: Calculate cumulatives up to this date, given as string,
                e.g. "08-11" means that cumulatives for each year are calculated until
                    11 August of each year.
            cumulative_for:
                - "all" for cumulative sum over all years
                - "year" for yearly cumulative sums
        """

        self.series = series.copy()
        self.series_units = series_units
        self.varname = series.name
        self.start_year = start_year if start_year else self.series.index.year.min()
        self.end_year = end_year if end_year else self.series.index.year.max()
        self.yearly_end_date = yearly_end_date
        self.show_reference = show_reference
        self.excl_years_from_reference = excl_years_from_reference
        self.highlight_year = highlight_year
        # Styling belongs in plot(); kept here only as a deprecated pass-through.
        if highlight_year_color is not None:
            warnings.warn("CumulativeYear: `highlight_year_color` in the constructor is deprecated; "
                          "pass it to plot() instead.", DeprecationWarning, stacklevel=2)
        self.highlight_year_color = highlight_year_color

        self.series = self.series.dropna()
        # self.series_full = self.series.copy()

        self.series = keep_years(data=self.series, start_year=self.start_year, end_year=self.end_year)

        if self.yearly_end_date:
            self.series = remove_after_date(data=self.series, yearly_end_date=yearly_end_date)

        self.uniq_years = self.series.index.year.unique()  # Unique years

        # self.series_long_df = calc_doy_timefraction(input_series=self.series)
        self.cumulatives_per_year_df = doy_cumulatives_per_year(series=self.series)

        # Remove data for DOY 366
        self.cumulatives_per_year_df = self.cumulatives_per_year_df.loc[self.cumulatives_per_year_df.index < 366]

        # Figure/axis are created in plot() (phase 2), or supplied by the caller.
        self.fig = None
        self.ax = None
        self._own_fig = False


    def _add_reference(self, digits_after_comma):
        # Calculate reference
        self.mean_doy_cumulative_df = doy_mean_cumulative(cumulatives_per_year_df=self.cumulatives_per_year_df,
                                                          excl_years_from_reference=self.excl_years_from_reference)

        # label = f"{year}: {cumulative_df[year].dropna().iloc[-1]:.2f}"
        mean_end = self.mean_doy_cumulative_df['MEAN_DOY_TIME'].iloc[-1]
        self.ax.plot(self.mean_doy_cumulative_df.index.to_numpy(),
                     self.mean_doy_cumulative_df['MEAN_DOY_TIME'].values,
                     color='black', alpha=1,
                     ls='-', lw=theme.WIDTH_LINE_WIDER,
                     marker='', markeredgecolor='none', ms=0,
                     zorder=99, label=f'mean {mean_end:.{digits_after_comma}f}')
        # self.ax.fill_between(mean_cumulative_df.index.to_numpy(),
        #                      mean_cumulative_df['MEAN+1.96_SD'].values,
        #                      mean_cumulative_df['MEAN-1.96_SD'].values,
        #                      alpha=.005, zorder=0, color='black', edgecolor='none',
        #                      label="XXX")
        self.ax.fill_between(self.mean_doy_cumulative_df.index.to_numpy(),
                             self.mean_doy_cumulative_df['MEAN+SD'].values,
                             self.mean_doy_cumulative_df['MEAN-SD'].values,
                             alpha=.1, zorder=0, color='black', edgecolor='none',
                             label="mean±1sd")


    def _apply_format(self, style: FormatStyle):
        # The figure-level dynamic title uses the larger FIGHEADER_FONTSIZE and may
        # live on fig.suptitle, so it stays here rather than going through style.apply.
        title = f"Cumulatives per year ({self.uniq_years.min()}-{self.uniq_years.max()}), " \
                f"until DOY {int(self.cumulatives_per_year_df.index[-1])}"
        if self._own_fig:
            self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
        else:
            self.ax.set_title(title, fontsize=theme.FIGHEADER_FONTSIZE)

        ymin = self.cumulatives_per_year_df.min().min()
        ymax = self.cumulatives_per_year_df.max().max()
        self.ax.set_ylim(ymin, ymax)

        # Shared chrome (facecolor/ticks/spines/labels/units/grid/zeroline). Title is
        # owned above and the per-year legend (custom ncol + labelspacing) below, so
        # suppress both on the applied copy; keep style.show_legend to gate the legend.
        chrome = style.merged(title="")
        chrome.show_legend = False
        chrome.apply(ax=self.ax, default_xlabel='Month', default_ylabel=self.varname,
                     zeroline_data=self.cumulatives_per_year_df)

        if style.show_legend and self.ax.get_legend_handles_labels()[0]:
            n_legend_cols = pf.n_legend_cols(n_legend_entries=len(self.uniq_years))
            pf.default_legend(ax=self.ax,
                              labelspacing=0.2,
                              ncol=n_legend_cols)

        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='month')

        # self.fig.tight_layout()


    def get(self):
        """Return axis"""
        return self.ax


    def plot(self, ax=None, format_style: FormatStyle = None, showplot: bool = True,
             digits_after_comma: int = 2, highlight_year_color: str = None):
        """Plot one cumulative-sum curve per year on a shared day-of-year axis.

        Args:
            ax: Matplotlib axes to draw on; a standalone figure is created if None.
            format_style: Shared chrome (title/labels/fonts/grid/legend). None = house style.
            showplot: If True and a new figure was created, show it.
            digits_after_comma: Decimal places for the end-of-year total in each legend label.
            highlight_year_color: Colour for the highlighted year; defaults to red.
        """
        # Fold the legacy units kwarg onto the (copied) style; the dynamic title and
        # the per-year legend stay class-owned, so suppress those on the applied style.
        style = (format_style or FormatStyle()).merged(yunits=self.series_units)
        # Phase 2: use the caller's axes, or create a standalone figure.
        if ax is None:
            self.fig, self.ax = pf.create_ax()
            self._own_fig = True
        else:
            self.ax = ax
            self.fig = ax.get_figure()
            self._own_fig = False
        self.ax.xaxis.axis_date()

        # Resolve highlight color: plot() arg wins, then the (deprecated) constructor
        # value, then the default.
        hl_color = highlight_year_color or self.highlight_year_color or '#F44336'

        color_list = theme.colorwheel_36()  # get some colors

        # Plot yearly cumulatives
        for ix, year in enumerate(self.cumulatives_per_year_df.columns):
            label = f"{year}: {self.cumulatives_per_year_df[year].dropna().iloc[-1]:.{digits_after_comma}f}"
            lw = theme.WIDTH_LINE_WIDER if year == self.highlight_year else theme.WIDTH_LINE_DEFAULT
            color = hl_color if year == self.highlight_year else color_list[ix]

            self.ax.plot(self.cumulatives_per_year_df.index,
                         self.cumulatives_per_year_df[year],
                         color=color, alpha=1,
                         ls='-', lw=lw,
                         marker='', markeredgecolor='none', ms=0,
                         zorder=99, label=label)

        # Show reference
        if self.show_reference:
            self._add_reference(digits_after_comma=digits_after_comma)

        self._apply_format(style)

        if self._own_fig and showplot:
            self.fig.show()

        return self.ax


class Cumulative:
    """Plot cumulative sums across all data.

    Visualizes cumulative sums for each column in the input DataFrame
    with values displayed at the end of each series.

    Args:
        df: Dataframe, cumulatives are plotted for each column.
        units: Units shown in plot, LaTeX format is supported.
        start_year: Start year of shown data.
        end_year: End year of shown data.

    See Also:
        examples/visualization/timeseries_and_cumulative.py — Cumulative plots with multiple scenarios
        examples/flux/uncertainty.py — Cumulative uncertainty visualization
    """

    def __init__(self,
                 df: DataFrame,
                 units: str = None,
                 start_year: int = None,
                 end_year: int = None):
        """Compute cumulative sums per column. See the class docstring for parameters."""
        self.df = df
        self.units = units
        self.start_year = start_year if start_year else self.df.index.year.min()
        self.end_year = end_year if end_year else self.df.index.year.max()

        if self.start_year | self.end_year:
            self.df = keep_years(data=self.df, start_year=self.start_year, end_year=self.end_year)

        self.show_grid = True
        self.show_legend = True
        self.ylabel = None
        self.show_title = True

        self.cumulative = self.df.cumsum()

        # Figure/axis are created in plot() (phase 2), or supplied by the caller.
        self.fig = None
        self.ax = None
        self._own_fig = False

    def _apply_format(self, style: FormatStyle):
        # The dynamic title uses the larger FIGHEADER_FONTSIZE and may live on
        # fig.suptitle, so it stays here rather than going through style.apply.
        if self.show_title:
            title = f"Cumulatives ({self.cumulative.index.min()}-{self.cumulative.index.max()})"
            if self._own_fig:
                self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
            else:
                self.ax.set_title(title, fontsize=theme.FIGHEADER_FONTSIZE)
        ymin = self.cumulative.min().min()
        ymax = self.cumulative.max().max()
        ymax = ymax * 1.05 if ymax > 0 else ymax * 0.95
        ymin = ymin * 0.95 if ymin > 0 else ymin * 1.05
        self.ax.set_ylim(ymin, ymax)
        ax_ylabel_txt = self.ylabel if self.ylabel else "Cumulative"

        # Shared chrome (facecolor/ticks/spines/labels/units/grid/zeroline). Title is
        # owned above and the multi-column legend (custom ncol + labelspacing) below, so
        # suppress both on the applied copy; keep style.show_legend to gate the legend.
        chrome = style.merged(title="")
        chrome.show_legend = False
        chrome.apply(ax=self.ax, default_xlabel="Date", default_ylabel=ax_ylabel_txt,
                     zeroline_data=self.cumulative)

        if style.show_legend:
            n_legend_cols = pf.n_legend_cols(n_legend_entries=len(self.cumulative.columns))
            pf.default_legend(ax=self.ax,
                              labelspacing=0.2,
                              ncol=n_legend_cols)
        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        # self.fig.tight_layout()

    def get_ax(self):
        """Return axis"""
        return self.ax

    def plot(self, ax=None, format_style: FormatStyle = None, showplot: bool = True,
             digits_after_comma: int = 0,
             show_title: bool = True,
             fill: bool = False
             ):
        """Plot the cumulative sum of each column, with the final total in each legend label.

        Args:
            ax: Matplotlib axes to draw on; a standalone figure is created if None.
            format_style: Shared chrome (title/labels/fonts/grid/legend). None = house style.
            showplot: If True and a new figure was created, show it.
            digits_after_comma: Decimal places for the end-of-series total in each legend label.
            show_title: If True, draw the auto-generated title.
            fill: If True, shade the area between each curve and zero.
        """
        # Units come from the constructor; all other chrome comes from format_style.
        style = (format_style or FormatStyle()).merged(yunits=self.units)

        # Phase 2: use the caller's axes, or create a standalone figure.
        if ax is None:
            self.fig, self.ax = pf.create_ax()
            self._own_fig = True
        else:
            self.ax = ax
            self.fig = ax.get_figure()
            self._own_fig = False
        self.ax.xaxis.axis_date()

        # show_title has no FormatStyle field (the dynamic title is class-owned), so it
        # stays on the instance and is read by _apply_format.
        self.show_title = show_title

        color_list = theme.colorwheel_36()  # get some colors

        # Plot yearly cumulatives
        for ix, col in enumerate(self.cumulative):
            series = self.cumulative[col]
            label = f"{col}: {series.dropna().iloc[-1]:.{digits_after_comma}f}"
            lw = theme.WIDTH_LINE_DEFAULT
            color = color_list[ix]

            self.ax.plot(series.index, series,
                         color=color, alpha=1,
                         ls='-', lw=lw,
                         marker='', markeredgecolor='none', ms=0,
                         zorder=99, label=label)

            # Faintly shade the area between the curve and the zero line.
            if fill:
                self.ax.fill_between(series.index, series.to_numpy(), 0,
                                     color=color, alpha=0.12, edgecolor='none',
                                     zorder=1)

            x = series.index[-1]
            y = series.iloc[-1]

            self.ax.plot(x, y, color=color, alpha=1, marker="o", markeredgecolor=color, ms=10)

            self.ax.text(x, y, f"  {y:.{digits_after_comma}f}",
                         size=12, color=color,
                         backgroundcolor='none', alpha=1,
                         horizontalalignment='left', verticalalignment='center', zorder=99)

        self._apply_format(style)

        if self._own_fig and showplot:
            self.fig.show()

        return self.ax
