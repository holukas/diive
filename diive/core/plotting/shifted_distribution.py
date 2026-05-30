"""
ShiftedDistributionPlot: SHIFTED DISTRIBUTION PLOT
===================================================

Visualize how a variable's distribution has shifted between a reference
period and a comparison period, with color-coded zones derived from the
reference period's standard deviation (Hansen et al. methodology).

Part of the diive library: https://github.com/holukas/diive
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from sklearn.neighbors import KernelDensity
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.plotfuncs import default_format


class ShiftedDistributionPlot:
    """Shifted distribution plot comparing two time periods.

    Shows how a variable's distribution has shifted between a reference period
    (gray hatched outline) and a comparison period (color-coded zones). Zone
    boundaries are computed from the reference period's mean and standard
    deviation: cold/hot at ±1σ, extremely cold/hot at ±3σ.

    Args:
        series: Time-indexed Series with the variable to plot.
        ref_period: (start, end) date strings for the reference period.
        comp_period: (start, end) date strings for the comparison period.

    Call `plot()` to render with styling options (including ``zone_labels`` and
    ``zone_colors``).

    See Also:
        examples/visualization/plot_shifted_distribution.py
    """

    _DEFAULT_LABELS = ['Extremely cold', 'Cold', 'Normal', 'Hot', 'Extremely hot']
    _DEFAULT_COLORS = ['#1565C0', '#64B5F6', '#90A4AE', '#FF7043', '#B71C1C']

    def __init__(
        self,
        series: Series,
        ref_period: tuple,
        comp_period: tuple,
        zone_labels: list = None,
        zone_colors: list = None,
    ):
        self.series = series
        self.ref_period = ref_period
        self.comp_period = comp_period
        # Styling belongs in plot(); kept here only as deprecated pass-throughs.
        if zone_labels is not None or zone_colors is not None:
            warnings.warn("ShiftedDistributionPlot: `zone_labels`/`zone_colors` in the constructor "
                          "are deprecated; pass them to plot() instead.", DeprecationWarning, stacklevel=2)
        self.zone_labels = zone_labels
        self.zone_colors = zone_colors

        self.fig = None
        self.ax = None

        self._ref_data = series.loc[ref_period[0]:ref_period[1]].dropna().values
        self._comp_data = series.loc[comp_period[0]:comp_period[1]].dropna().values

        ref_mean = self._ref_data.mean()
        ref_std = self._ref_data.std()

        # 4 cut points → 5 zones: extremely low | low | normal | high | extremely high
        self.breakpoints = [
            ref_mean - 3 * ref_std,
            ref_mean - 1 * ref_std,
            ref_mean + 1 * ref_std,
            ref_mean + 3 * ref_std,
        ]

        # Evaluation grid spans both periods with a small margin
        all_vals = np.concatenate([self._ref_data, self._comp_data])
        margin = ref_std
        self._x = np.linspace(all_vals.min() - margin, all_vals.max() + margin, 1000)

        self._ref_kde = self._fit_kde(self._ref_data)
        self._comp_kde = self._fit_kde(self._comp_data)

    def _fit_kde(self, data: np.ndarray) -> np.ndarray:
        bw = 1.06 * data.std() * len(data) ** (-0.2)  # Silverman's rule
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(data.reshape(-1, 1))
        log_dens = kde.score_samples(self._x.reshape(-1, 1))
        return np.exp(log_dens)

    def get_fig(self):
        return self.fig

    def get_ax(self):
        return self.ax

    def plot(
        self,
        ax=None,
        title: str = None,
        xlabel: str = None,
        ref_label: str = None,
        comp_label: str = None,
        show_legend: bool = True,
        show_title: bool = True,
        show_xaxis: bool = True,
        show_yaxis: bool = True,
        figsize: tuple = (16, 7),
        zone_labels: list = None,
        zone_colors: list = None,
    ):
        """Render the shifted distribution plot.

        Args:
            ax: Matplotlib axes (creates new figure if None).
            title: Plot title. Defaults to series name.
            xlabel: X-axis label. Defaults to series name.
            ref_label: Legend label for reference period.
            comp_label: Legend label for comparison period.
            show_legend: Show legend (default True).
            show_title: Show title (default True).
            show_xaxis: Show x-axis spine, ticks, and tick labels (default True).
            show_yaxis: Show y-axis spine, ticks, and tick labels (default True).
            figsize: Figure size when ax is None.
            zone_labels: 5 zone labels from lowest to highest. Defaults to temperature labels.
            zone_colors: 5 fill colors for the zones (lowest to highest).
        """
        # Resolve styling: plot() arg wins, then the (deprecated) constructor value,
        # then the class defaults.
        zone_labels = zone_labels or self.zone_labels or self._DEFAULT_LABELS
        zone_colors = zone_colors or self.zone_colors or self._DEFAULT_COLORS

        self.ax = ax
        self.fig, self.ax, showplot = pf.setup_figax(ax=self.ax, figsize=figsize)

        x = self._x
        bp = self.breakpoints
        zone_edges = [x[0]] + list(bp) + [x[-1]]

        # Reference period: gray hatched outline drawn first (behind colored zones)
        _ref_label = ref_label or f"Reference ({self.ref_period[0]} - {self.ref_period[1]})"
        self.ax.fill_between(
            x, self._ref_kde,
            facecolor='none', edgecolor='#546E7A', linewidth=0,
            hatch='///', alpha=0.55, label=_ref_label, zorder=1,
        )
        self.ax.plot(x, self._ref_kde, color='#546E7A', linewidth=1.2, alpha=0.7, zorder=1)

        # Comparison period: filled colored zones on top
        for i in range(5):
            mask = (x >= zone_edges[i]) & (x <= zone_edges[i + 1])
            if mask.any():
                self.ax.fill_between(
                    x[mask], self._comp_kde[mask],
                    color=zone_colors[i], alpha=0.5, linewidth=0, zorder=2,
                )

        # Thin outline on comparison KDE
        _comp_label = comp_label or f"Comparison ({self.comp_period[0]} - {self.comp_period[1]})"
        self.ax.plot(x, self._comp_kde, color='#37474F', linewidth=0.8, alpha=0.4, zorder=3)

        # Thin dashed lines at breakpoints
        for bp_val in bp:
            self.ax.axvline(bp_val, color='white', linewidth=1.0, alpha=0.7, linestyle='--', zorder=5)

        _xlabel = xlabel or (str(self.series.name) if self.series.name else "")
        default_format(
            ax=self.ax,
            ax_xlabel_txt=_xlabel,
            ax_ylabel_txt="Density",
            ticks_width=1.5,
            ticks_length=5,
            ticks_direction='in',
            showgrid=False,
        )
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(right=False, top=False)
        self.ax.set_ylim(bottom=0)

        if not show_xaxis:
            self.ax.spines['bottom'].set_visible(False)
            self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
            self.ax.set_xlabel('')

        if not show_yaxis:
            self.ax.spines['left'].set_visible(False)
            self.ax.tick_params(axis='y', left=False, labelleft=False)
            self.ax.set_ylabel('')

        # Zone labels: text annotations just above the top spine, in data-x / axes-y coords
        trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)
        label_positions = [(zone_edges[i] + zone_edges[i + 1]) / 2 for i in range(5)]
        for pos, label, color in zip(label_positions, zone_labels, zone_colors):
            self.ax.text(
                pos, 1.01, label,
                transform=trans, color=color, fontsize=11, fontweight='bold',
                ha='center', va='bottom', clip_on=False,
            )

        if show_title:
            _title = title or f"Shifted distribution: {self.series.name}"
            self.ax.set_title(_title, fontsize=16, fontweight='bold', loc='left', pad=28)

        if show_legend:
            ref_patch = Patch(facecolor='none', edgecolor='#546E7A', hatch='///', label=_ref_label)
            comp_patch = Patch(facecolor='#90A4AE', label=_comp_label)
            self.ax.legend(
                handles=[ref_patch, comp_patch],
                fontsize=11, framealpha=0.0, edgecolor='none',
                loc='upper left', bbox_to_anchor=(0.01, 0.99),
            )

        if showplot:
            self.fig.show()
