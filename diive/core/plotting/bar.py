"""
PLOTTING: LONG-TERM ANNUAL ANOMALIES BAR PLOT
=============================================

Bar plot of yearly anomalies relative to a long-term reference period.

Part of the diive library: https://github.com/holukas/diive
"""
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.styles.format import FormatStyle


class LongtermAnomaliesYear:
    """Calculate and plot long-term anomaly for a variable, per year, compared to a reference period.

    Visualizes yearly anomalies as red/blue bars (above/below reference mean),
    with reference period mean ± standard deviation band for context.

    Two-phase design: separate data preparation (__init__) from rendering (plot).
    Phase 1 creates the plotter with data; Phase 2 renders with styling options.

    Args:
        series: Time series for anomalies with one value per year (pandas Series with year index)
        reference_start_year: First year of the reference period (int)
        reference_end_year: Last year of the reference period (int)
        series_label: Description label for the variable (displayed in title and legend)
        series_units: Units string for the variable (displayed on y-axis, e.g., '(°C)')

    Methods:
        plot : Render anomaly bar chart with styling options

    Example:
        See `examples/visualization/plot_other_plots.py` for complete example.
    """

    # Internal key for the data column of the working frame. Keying it by the
    # caller's Series name lets a variable called e.g. 'reference_mean' overwrite
    # the data before the anomaly is computed (same reason as ScatterXY/GridAggregator).
    _VALUECOL = '_values'

    def __init__(self,
                 series: Series,
                 reference_start_year: int,
                 reference_end_year: int,
                 series_label: str = None,
                 series_units: str = None):
        """
        Prepare long-term anomaly data for plotting.

        Args:
            series: Data to plot (pandas Series with year index)
            reference_start_year: First year of reference period for anomaly calculation
            reference_end_year: Last year of reference period for anomaly calculation
            series_label: Label for the variable (used in plot title and text)
            series_units: Units string (e.g., '(°C)', appended to y-axis label)

        See Also:
            plot : Render the anomaly chart with matplotlib styling options
        """
        self.series = series.copy()
        self.series_units = series_units
        self.series_label = series_label
        self.reference_start_year = reference_start_year
        self.reference_end_year = reference_end_year

        # Without a single measured year there is nothing to anomalise. Caught here
        # because the year lattice below derives its bounds from min()/max(), which
        # fail with "cannot convert float NaN to integer" - an internal detail that
        # names neither the class nor the empty input.
        if self.series.dropna().empty:
            raise ValueError(f"LongtermAnomaliesYear needs at least one year of data, "
                             f"the given series holds none "
                             f"(length {len(self.series)}, all missing).")

        # Chronological order is required: the bars are drawn in frame order and
        # the "last 10 years" annotation is a tail() of the same frame.
        self.series = self.series.sort_index(ascending=True)
        self.data_first_year = self.series.index.min()
        self.data_last_year = self.series.index.max()

        # Complete the year lattice. The bars are drawn with `plot.bar`, which is
        # categorical, so a year the record does not cover takes up no axis width
        # at all: a 12-year outage was one bar-width jump between two evenly spaced
        # ticks, while the title below asserts the full first-to-last span. Years
        # nothing was measured in become NaN bars, i.e. visible holes. The reference
        # mean and sd are computed over the values and skip NaN, so the injected
        # years leave them untouched.
        self.series = self.series.reindex(
            pd.Index(range(int(self.data_first_year), int(self.data_last_year) + 1),
                     name=self.series.index.name))

        self._anomalies_df = self._calc_reference()

    @property
    def anomalies_df(self) -> pd.DataFrame:
        """Results frame (copy), with the data column back under the caller's Series name."""
        return self._anomalies_df.rename(columns={self._VALUECOL: self.series.name})

    def _annotate_reference(self):
        """Draw the domain-specific reference-statistics info box (not shared chrome)."""
        ref_mean = self._anomalies_df['reference_mean'].iloc[-1]
        ref_sd = self._anomalies_df['reference_sd'].iloc[-1]
        ref_n_years = (self.reference_end_year - self.reference_start_year) + 1
        last10 = self._anomalies_df[self._VALUECOL].tail(10)
        last10_mean = last10.mean()
        last10_std = last10.std()

        self.ax.text(0.98, 0.02, f"reference period mean: {ref_mean:.2f}±{ref_sd:.2f}sd "
                                 f"({self.reference_start_year}-{self.reference_end_year}, "
                                 f"{ref_n_years} years)\n"
                                 f"last 10 years mean: {last10_mean:.2f}±{last10_std:.2f}sd "
                                 f"({last10.index[0]}-{last10.index[-1]})",
                     size=11, color='#2C3E50', backgroundcolor='white', transform=self.ax.transAxes,
                     alpha=0.9, horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#CCCCCC', linewidth=0.5))

        # X-axis tick configuration (avoid overcrowding for long records)
        nbins = 50 if len(self.series) > 50 else len(self.series)
        self.ax.locator_params(axis='x', nbins=nbins)
        self.ax.set_xlim(-1, len(self.series))

    def _calc_reference(self):
        anomalies_df = self.series.rename(self._VALUECOL).to_frame()

        ref_subset = self.series.loc[(self.series.index >= self.reference_start_year)
                                     & (self.series.index <= self.reference_end_year)]
        # ref_subset = self.series.between(self.reference_start_ix, self.reference_end_ix)
        # A reference period holding no measurement leaves mean and sd NaN, which
        # makes every anomaly NaN: an empty chart carrying a title that asserts the
        # record's full span and an annotation reading "nan+/-nansd". Reachable with
        # the period reversed, or landing wholly inside an outage of the record.
        if ref_subset.dropna().empty:
            raise ValueError(f"Reference period {self.reference_start_year}-{self.reference_end_year} "
                             f"holds no data, so no anomaly can be calculated "
                             f"(record covers {self.data_first_year}-{self.data_last_year}).")
        anomalies_df['reference_mean'] = ref_subset.mean()
        anomalies_df['reference_sd'] = ref_subset.std()
        anomalies_df['anomaly'] = anomalies_df[self._VALUECOL].sub(anomalies_df['reference_mean'])
        anomalies_df['anomaly_above'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] >= 0]
        anomalies_df['anomaly_below'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] < 0]
        return anomalies_df

    def plot(self, ax=None, format_style: FormatStyle = None):
        """
        Render long-term anomaly bar chart with matplotlib styling (Phase 2 of two-phase design).

        Chrome (title, labels, units, font sizes, colours, grid, ticks, spines, zero
        line) comes from a shared :class:`~diive.plotting.FormatStyle` so it looks and
        is configured the same way as every other diive plot. The red/blue above/below
        bar colouring is data encoding and stays here. Can be called multiple times on
        the same object to draw on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure and displays it
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome.
                When None the diive house style is used.

        Returns:
            None (displays plot if ax=None, otherwise renders on provided axes)

        Example:
            >>> import diive as dv, pandas as pd, matplotlib.pyplot as plt
            >>> annual = pd.Series([5.1, 5.4, 6.0, 6.3], index=[2021, 2022, 2023, 2024], name='TA')
            >>> anomaly = dv.plotting.LongtermAnomaliesYear(annual, 2021, 2022)
            >>> anomaly.plot(ax=plt.subplots()[1], format_style=dv.plotting.FormatStyle(title='TA'))
        """
        style = format_style or FormatStyle()

        # Create axis if not provided (Phase 2 only)
        if ax:
            # If ax is given, plot directly to ax, no fig needed
            self.fig = None
            self.ax = ax
            self.showplot = False
        else:
            # If no ax is given, create fig and ax and then show the plot
            self.fig, self.ax = pf.create_ax()
            self.showplot = True

        # Publication-ready colors for above/below anomalies (data encoding)
        color_above = '#EF5350'  # Red for above-reference
        color_below = '#42A5F5'  # Blue for below-reference

        # Plot bars
        self._anomalies_df['anomaly_above'].plot.bar(
            color=color_above,
            ax=self.ax,
            legend=False,
            width=0.7,
            alpha=0.9
        )
        self._anomalies_df['anomaly_below'].plot.bar(
            color=color_below,
            ax=self.ax,
            legend=False,
            width=0.7,
            alpha=0.9
        )

        # Shared formatting layer: title/labels/units/fonts/grid/ticks/spines/zeroline.
        default_title = f"{self.series_label} anomaly per year ({self.data_first_year}-{self.data_last_year})"
        default_ylabel = f"{self.series_label} anomaly" + (f" {self.series_units}" if self.series_units else "")
        style.apply(ax=self.ax, default_title=default_title, default_xlabel='Year',
                    default_ylabel=default_ylabel, zeroline_data=self._anomalies_df['anomaly'])

        # Domain-specific annotation + categorical x-axis tweaks (not shared chrome).
        self._annotate_reference()

        if self.showplot:
            self.fig.patch.set_facecolor('white')
            self.fig.tight_layout(pad=1.2)
            self.fig.show()

    def get(self):
        """Return axis"""
        return self.ax
