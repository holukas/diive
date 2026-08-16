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
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from sklearn.neighbors import KernelDensity
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.core.plotting.styles import LightTheme as theme
from diive.core.plotting.styles.format import FormatStyle


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
        """Fit the reference/comparison KDEs and zone breakpoints. See the class docstring
        for parameters (``zone_labels``/``zone_colors`` here are deprecated — pass them to :meth:`plot`)."""
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
        self._artists = []  # What this instance drew on `self.ax`, taken back on a repeat plot()

        self._ref_data = series.loc[ref_period[0]:ref_period[1]].dropna().values
        self._comp_data = series.loc[comp_period[0]:comp_period[1]].dropna().values

        # A period holding no records has no mean and no spread. Asking numpy for them
        # anyway only emits "Mean of empty slice" warnings on the way to the same NaN.
        ref_mean = self._ref_data.mean() if self._ref_data.size else np.nan
        # Sample sd (ddof=1), as everywhere else in diive: a reference period is a sample,
        # and the population sd understates its spread by sqrt(n/(n-1)) -- 41% at n=2, 1.7%
        # even at n=30 -- which narrows every zone and over-counts extremes. A single record
        # has no sample sd, so it keeps 0.0 and stays the spike it is rather than losing its
        # zones to NaN.
        if self._ref_data.size > 1:
            ref_std = self._ref_data.std(ddof=1)
        else:
            ref_std = 0.0 if self._ref_data.size else np.nan

        # 4 cut points → 5 zones: extremely low | low | normal | high | extremely high
        self.breakpoints = [
            ref_mean - 3 * ref_std,
            ref_mean - 1 * ref_std,
            ref_mean + 1 * ref_std,
            ref_mean + 3 * ref_std,
        ]

        # Evaluation grid spans both periods with a small margin. The reference SD is
        # that margin, but it is zero for a constant reference and NaN for an empty one,
        # either of which would collapse the grid onto a single point (or onto NaN), so
        # fall back to a margin scaled to the data instead.
        all_vals = np.concatenate([self._ref_data, self._comp_data])
        if all_vals.size == 0:
            self._x = None  # Neither period holds a record: nothing to evaluate on
        else:
            lo, hi = float(all_vals.min()), float(all_vals.max())
            margin = ref_std if np.isfinite(ref_std) and ref_std > 0 else 0.1 * max(abs(lo), abs(hi), 1.0)
            self._x = np.linspace(lo - margin, hi + margin, 1000)

        self._ref_kde = self._fit_kde(self._ref_data)
        self._comp_kde = self._fit_kde(self._comp_data)

    def _fit_kde(self, data: np.ndarray) -> np.ndarray | None:
        if data.size == 0 or self._x is None:
            return None  # No records, hence no density; plot() labels the period instead
        bw = 1.06 * data.std() * len(data) ** (-0.2)  # Silverman's rule
        if bw <= 0:
            # A constant period, or one holding a single record, has zero spread, so
            # Silverman's rule gives a zero bandwidth that KernelDensity rejects. The
            # distribution is real — a spike — so draw it as one, with a kernel narrow
            # against the plotted range, rather than refusing the whole plot.
            bw = (self._x[-1] - self._x[0]) / 200
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(data.reshape(-1, 1))
        log_dens = kde.score_samples(self._x.reshape(-1, 1))
        return np.exp(log_dens)

    def _remember_artists(self, n_before: tuple):
        """Record what this plot() added to the axes, so a repeat call can take it back."""
        n_collections, n_lines, n_texts = n_before
        self._artists = [*self.ax.collections[n_collections:],
                         *self.ax.lines[n_lines:],
                         *self.ax.texts[n_texts:]]

    def get_fig(self):
        """Return the matplotlib Figure (available after :meth:`plot`)."""
        return self.fig

    def get_ax(self):
        """Return the matplotlib Axes (available after :meth:`plot`)."""
        return self.ax

    def plot(
        self,
        ax=None,
        format_style: FormatStyle = None,
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
            format_style: A :class:`~diive.plotting.FormatStyle` describing the chrome
                (axis labels/font sizes/colours/ticks/grid). When None the diive house
                style is used, with the grid off — pass ``show_grid=True`` to draw it.
                The y-label defaults to ``"Density"``; the title and the legend keep this
                plot's own left-aligned placement, so ``legend_loc``/``legend_ncol`` do
                not apply.
            ref_label: Legend label for reference period.
            comp_label: Legend label for comparison period.
            show_legend: Show legend (default True).
            show_title: Show title (default True).
            show_xaxis: Show x-axis spine, ticks, and tick labels (default True).
            show_yaxis: Show y-axis spine, ticks, and tick labels (default True).
            figsize: Figure size when ax is None.
            zone_labels: Exactly 5 zone labels from lowest to highest. Defaults to
                temperature labels. Any other length raises ``ValueError``.
            zone_colors: Exactly 5 fill colors for the zones (lowest to highest).

        Calling this again on the same axes replaces the previous rendering; calling it
        on a different axes leaves the earlier one drawn.
        """
        # Resolve styling: plot() arg wins, then the (deprecated) constructor value,
        # then the class defaults.
        zone_labels = zone_labels or self.zone_labels or self._DEFAULT_LABELS
        zone_colors = zone_colors or self.zone_colors or self._DEFAULT_COLORS

        # The plot has exactly five zones, so anything else is caller error: too few
        # colours used to raise IndexError halfway through drawing, too few labels left
        # zones unlabelled and a longer list was dropped, all without a word.
        for _name, _value in (('zone_labels', zone_labels), ('zone_colors', zone_colors)):
            if len(_value) != len(self._DEFAULT_LABELS):
                raise ValueError(
                    f"ShiftedDistributionPlot: `{_name}` needs exactly {len(self._DEFAULT_LABELS)} "
                    f"entries, one per zone from lowest to highest, but got {len(_value)}."
                )

        # The dashed breakpoint markers already carry the vertical structure, so a second
        # family of dashed verticals would only compete with them: the grid is off in this
        # plot's *default* style rather than forced off over the caller's, so a caller
        # asking for show_grid=True still gets it (same pattern as the heatmaps).
        style = format_style or FormatStyle(show_grid=False)

        self.ax = ax
        self.fig, self.ax, showplot = pf.setup_figax(ax=self.ax, figsize=figsize)

        # A second plot() on the *same* axes replaces this plot instead of stacking a
        # second copy over it; on a different axes both stay, which is what the two-phase
        # contract promises. Only this instance's own artists go -- everything else on the
        # axes is the caller's, and one the caller has already cleared away has no axes.
        for artist in self._artists:
            if artist.axes is self.ax:
                artist.remove()
        self._artists = []
        _n_before = (len(self.ax.collections), len(self.ax.lines), len(self.ax.texts))

        _ref_label = ref_label or f"Reference ({self.ref_period[0]} - {self.ref_period[1]})"
        _comp_label = comp_label or f"Comparison ({self.comp_period[0]} - {self.comp_period[1]})"

        if self._x is None:
            # Neither period holds a record, so there is no density and no zone
            # structure. Say so on the axes rather than raising.
            self.ax.text(0.5, 0.5, "No data in reference or comparison period",
                         transform=self.ax.transAxes, ha='center', va='center',
                         fontsize=12, color=theme.COLOR_TEXT)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self._remember_artists(_n_before)
            if showplot:
                self.fig.show()
            return

        x = self._x
        bp = self.breakpoints
        # An empty reference period leaves the breakpoints NaN, so there are no zones.
        has_zones = bool(np.isfinite(bp).all())

        # A +-3 SD breakpoint falls outside the evaluation grid whenever the reference is
        # skewed or bounded (RH against 100, precipitation against 0). Left unclipped the
        # edges stop being monotonic, so that zone's interval is inverted: it paints
        # nothing while its label sits over the neighbour it does not describe. Clip the
        # drawn edges into the grid; an inverted interval means the zone lies entirely off
        # the plotted range, so it is neither painted nor labelled.
        zone_edges, zone_in_view = [], []
        if has_zones:
            raw_edges = [x[0]] + list(bp) + [x[-1]]
            zone_edges = [x[0]] + list(np.clip(bp, x[0], x[-1])) + [x[-1]]
            zone_in_view = [raw_edges[i] <= raw_edges[i + 1] for i in range(5)]

        # Reference period: gray hatched outline drawn first (behind colored zones)
        if self._ref_kde is None:
            _ref_label = f"{_ref_label}: no data"
        else:
            self.ax.fill_between(
                x, self._ref_kde,
                facecolor='none', edgecolor='#546E7A', linewidth=0,
                hatch='///', alpha=0.55, label=_ref_label, zorder=1,
            )
            self.ax.plot(x, self._ref_kde, color='#546E7A', linewidth=1.2, alpha=0.7, zorder=1)

        # Comparison period: filled colored zones on top
        if self._comp_kde is None:
            _comp_label = f"{_comp_label}: no data"
        elif has_zones:
            for i in range(5):
                mask = (x >= zone_edges[i]) & (x <= zone_edges[i + 1])
                if zone_in_view[i] and mask.any():
                    self.ax.fill_between(
                        x[mask], self._comp_kde[mask],
                        color=zone_colors[i], alpha=0.5, linewidth=0, zorder=2,
                    )
        else:
            # Without a reference there is nothing to grade the comparison against, so
            # it is filled unzoned in the neutral colour instead of not at all.
            self.ax.fill_between(x, self._comp_kde, color=zone_colors[2],
                                 alpha=0.5, linewidth=0, zorder=2)

        # Thin outline on comparison KDE
        if self._comp_kde is not None:
            self.ax.plot(x, self._comp_kde, color='#37474F', linewidth=0.8, alpha=0.4, zorder=3)

        # Thin dashed lines at breakpoints
        if has_zones:
            for bp_val in bp:
                # A breakpoint off the grid marks nothing that is drawn; it would only
                # stretch the x-axis into empty space.
                if x[0] <= bp_val <= x[-1]:
                    self.ax.axvline(bp_val, color='white', linewidth=1.0, alpha=0.7,
                                    linestyle='--', zorder=5)

        # Shared chrome: facecolor/ticks/spines/axis labels/grid. The title and the legend
        # are re-drawn below with this plot's own placement, so they are the only two
        # fields the class overrides — everything else, the y-label included, stays the
        # caller's. "Density" is passed as the *default* y-label so a caller-set
        # FormatStyle.ylabel still wins. No zeroline_data is passed, so show_zeroline is
        # inert here either way.
        _default_xlabel = str(self.series.name) if self.series.name else ""
        chrome = style.merged(title="", show_legend=False)
        chrome.apply(ax=self.ax, default_xlabel=_default_xlabel, default_ylabel="Density")
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
        if has_zones:
            trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)
            for i, label, color in zip(range(5), zone_labels, zone_colors, strict=True):
                if not zone_in_view[i]:
                    continue
                self.ax.text(
                    (zone_edges[i] + zone_edges[i + 1]) / 2, 1.01, label,
                    transform=trans, color=color, fontsize=11, fontweight='bold',
                    ha='center', va='bottom', clip_on=False,
                )

        # Title font/colour/weight come from the style; the left-aligned, padded
        # placement is this plot's own layout (not part of the shared chrome). An
        # empty style.title ("") still suppresses the title.
        if show_title and style.title != "":
            _title = style.title if style.title is not None else f"Shifted distribution: {self.series.name}"
            title_fs = style.title_fontsize if style.title_fontsize is not None else theme.FONTSIZE_TITLE
            title_color = style.text_color if style.text_color is not None else theme.COLOR_TEXT
            self.ax.set_title(_title, fontsize=title_fs, fontweight=style.title_fontweight,
                              color=title_color, loc='left', pad=28)

        # Custom patch legend with this plot's own anchor; only the font size is shared.
        if show_legend and style.show_legend:
            legend_fs = style.legend_fontsize if style.legend_fontsize is not None else theme.FONTSIZE_LEGEND
            ref_patch = Patch(facecolor='none', edgecolor='#546E7A', hatch='///', label=_ref_label)
            comp_patch = Patch(facecolor='#90A4AE', label=_comp_label)
            self.ax.legend(
                handles=[ref_patch, comp_patch],
                fontsize=legend_fs, framealpha=0.0, edgecolor='none',
                loc='upper left', bbox_to_anchor=(0.01, 0.99),
            )

        self._remember_artists(_n_before)

        if showplot:
            self.fig.show()
