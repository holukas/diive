import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme


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
        See `examples/core/visualization/plot_other_plots.py` for complete example.
    """

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

        self.series.sort_index(ascending=True)
        self.data_first_year = self.series.index.min()
        self.data_last_year = self.series.index.max()

        self.anomalies_df = self._calc_reference()

    def _apply_format(self, title: str = None):
        """Format matplotlib plot with modern scientific design principles."""
        # Publication-ready title styling
        if title is None:
            title = f"{self.series_label} anomaly per year ({self.data_first_year}-{self.data_last_year})"
        if self.fig is not None:
            self.fig.suptitle(title, fontsize=16, fontweight=500, color='#2C3E50', y=0.98)

        # Calculate reference statistics
        ref_mean = self.anomalies_df['reference_mean'].iloc[-1]
        ref_sd = self.anomalies_df['reference_sd'].iloc[-1]
        ref_n_years = (self.reference_end_year - self.reference_start_year) + 1
        last10 = self.anomalies_df[self.series.name].tail(10)
        last10_mean = last10.mean()
        last10_std = last10.std()

        # Publication-ready info text
        self.ax.text(0.98, 0.02, f"reference period mean: {ref_mean:.2f}±{ref_sd:.2f}sd "
                                 f"({self.reference_start_year}-{self.reference_end_year}, "
                                 f"{ref_n_years} years)\n"
                                 f"last 10 years mean: {last10_mean:.2f}±{last10_std:.2f}sd "
                                 f"({last10.index[0]}-{last10.index[-1]})",
                     size=11, color='#2C3E50', backgroundcolor='white', transform=self.ax.transAxes,
                     alpha=0.9, horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#CCCCCC', linewidth=0.5))

        # X-axis tick configuration
        nbins = 50 if len(self.series) > 50 else len(self.series)
        self.ax.locator_params(axis='x', nbins=nbins)

        # Publication-ready axis labels and formatting
        self.ax.set_xlabel('Year', fontsize=13, fontweight=600, color='#2C3E50', labelpad=10)
        ylabel_text = f"{self.series_label} anomaly" + (f" {self.series_units}" if self.series_units else "")
        self.ax.set_ylabel(ylabel_text, fontsize=13, fontweight=600, color='#2C3E50', labelpad=10)

        # Publication-ready gridline styling (subtle, y-axis only)
        self.ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.7, color='#CCCCCC')
        self.ax.set_axisbelow(True)

        # Publication-ready spine styling (all spines visible)
        for spine in ['top', 'right', 'left', 'bottom']:
            self.ax.spines[spine].set_color('#2C3E50')
            self.ax.spines[spine].set_linewidth(1.2)

        # Zero line reference
        self.ax.axhline(0, lw=1.0, color='#2C3E50', linestyle='-', alpha=0.6, zorder=0)

        # Publication-ready tick styling
        self.ax.tick_params(axis='both', which='major', labelsize=12, colors='#2C3E50',
                           length=6, width=1.0, pad=6)
        self.ax.tick_params(axis='both', which='minor', length=4, width=0.7)

        self.ax.set_xlim(-1, len(self.series))
        if self.fig is not None:
            self.fig.tight_layout(pad=1.2)

    def _calc_reference(self):
        anomalies_df = pd.DataFrame(self.series)

        ref_subset = self.series.loc[(self.series.index >= self.reference_start_year)
                                     & (self.series.index <= self.reference_end_year)]
        # ref_subset = self.series.between(self.reference_start_ix, self.reference_end_ix)
        anomalies_df['reference_mean'] = ref_subset.mean()
        anomalies_df['reference_sd'] = ref_subset.std()
        anomalies_df['anomaly'] = anomalies_df[self.series.name].sub(anomalies_df['reference_mean'])
        anomalies_df['anomaly_above'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] >= 0]
        anomalies_df['anomaly_below'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] < 0]
        return anomalies_df

    def plot(self, ax=None, title: str = None):
        """
        Render long-term anomaly bar chart with matplotlib styling (Phase 2 of two-phase design).

        All styling and presentation parameters go here. Can be called multiple times
        on the same LongtermAnomaliesYear object to plot on different axes with different styling.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure and displays it
            title: Figure title (default: auto-generated from series_label and year range)

        Returns:
            None (displays plot if ax=None, otherwise renders on provided axes)

        Example:
            >>> anomaly = dv.plot_longterm_anomalies_year(series=data, reference_start_year=2015)
            >>> anomaly.plot(title='Custom Title')  # New figure with custom title
            >>> anomaly.plot(ax=ax1, title='Subplot Title')  # Plot on existing axis
        """
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

        # Publication-ready colors for above/below anomalies
        color_above = '#EF5350'  # Red for above-reference
        color_below = '#42A5F5'  # Blue for below-reference

        # Plot bars
        self.anomalies_df['anomaly_above'].plot.bar(
            color=color_above,
            ax=self.ax,
            legend=False,
            width=0.7,
            alpha=0.9
        )
        self.anomalies_df['anomaly_below'].plot.bar(
            color=color_below,
            ax=self.ax,
            legend=False,
            width=0.7,
            alpha=0.9
        )

        # Apply formatting
        self._apply_format(title=title)

        # Set white background
        self.ax.set_facecolor('white')
        if self.showplot:
            self.fig.patch.set_facecolor('white')
            self.fig.show()

    def get(self):
        """Return axis"""
        return self.ax
