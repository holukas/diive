"""
PLOTTING: COMPOUND EXTREMES
============================

Quadrant scatter of two standardized drivers (z-scores), with points coloured and
marked by compound-extreme category and dashed threshold lines marking the extreme
quadrants. Visualizes the output of
:class:`diive.analysis.compoundextremes.CompoundExtremes` (after Wang et al., Fig. 2).

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.analysis.compoundextremes import (CAT_NONE, CAT_VAR1, CAT_VAR2,
                                             CAT_COMPOUND, CATEGORY_ORDER)
from diive.core.plotting.styles.format import FormatStyle

# Default per-category styling, keyed by the CompoundExtremes category codes.
# Colours/markers follow Wang et al. Fig. 2: none = black dots, single-driver
# extremes = orange triangles / dark-red squares, compound = red diamonds.
DEFAULT_CATEGORY_STYLES = {
    CAT_NONE: {'color': '#212121', 'marker': 'o'},
    CAT_VAR1: {'color': '#FB8C00', 'marker': '^'},
    CAT_VAR2: {'color': '#8E2A2A', 'marker': 's'},
    CAT_COMPOUND: {'color': '#E53935', 'marker': 'D'},
}

# Fallback palette/markers for arbitrary (non-code) category keys.
_FALLBACK_COLORS = ['#212121', '#FB8C00', '#8E2A2A', '#E53935', '#2196F3', '#43A047']
_FALLBACK_MARKERS = ['o', '^', 's', 'D', 'v', 'P']


class CompoundExtremesPlot:
    """Quadrant scatter of compound extremes. See :meth:`__init__`."""

    def __init__(
            self,
            x: Series,
            y: Series,
            category: Series,
            category_styles: dict = None,
            category_order: list = None,
            labels: Series = None,
            threshold_x: float = 2.0,
            threshold_y: float = -2.0,
    ):
        """Scatter of var1 vs var2 z-scores, coloured by extreme category.

        Two-phase design: pass data and classification here, style and render in
        :meth:`plot`.

        Args:
            x: var1 z-scores (e.g. VPD), used for the x-axis.
            y: var2 z-scores (e.g. soil water content), used for the y-axis.
            category: Category per period — either CompoundExtremes codes
                ('none'/'var1'/'var2'/'compound') or arbitrary group labels.
                Aligned to *x*/*y* by index.
            category_styles: Optional {key: {'color':, 'marker':, 'label':}} override.
                Missing entries fall back to the defaults (for code keys) or a palette.
                'label' sets the legend text (default: the key, or the code's role).
            category_order: Draw/legend order of categories (default: the standard
                none -> var1 -> var2 -> compound order, then any extra keys).
            labels: Optional per-point annotations (e.g. '2022-08'), aligned by index.
            threshold_x: x-position of the vertical dashed threshold line (None to skip).
            threshold_y: y-position of the horizontal dashed threshold line (None to skip).

        See Also:
            CompoundExtremesPlot.from_compound_extremes : build directly from a
                :class:`~diive.analysis.compoundextremes.CompoundExtremes` instance.
            examples/visualization/plot_compound_extremes.py : worked examples.
        """
        self.xname = str(x.name) if x.name is not None else 'var1'
        self.yname = str(y.name) if y.name is not None else 'var2'
        self.threshold_x = threshold_x
        self.threshold_y = threshold_y

        df = pd.concat([x.rename('_x'), y.rename('_y'), category.rename('_cat')], axis=1)
        if labels is not None:
            df['_label'] = labels
        else:
            df['_label'] = ''
        self.df = df.dropna(subset=['_x', '_y', '_cat'])

        present = list(self.df['_cat'].unique())
        if category_order is None:
            ordered = [c for c in CATEGORY_ORDER if c in present]
            ordered += [c for c in present if c not in ordered]
            self.category_order = ordered
        else:
            self.category_order = [c for c in category_order if c in present]

        self.category_styles = self._resolve_styles(category_styles)

        self.fig = None
        self.ax = None

    @classmethod
    def from_compound_extremes(cls, ce, annotate_labels: bool = True):
        """Build the plot straight from a fitted :class:`CompoundExtremes`.

        Pulls the z-score columns, category codes, per-category labels, period
        annotations, and threshold positions (signed by each variable's extreme
        direction) from the analysis instance.

        Args:
            ce: A fitted ``CompoundExtremes`` instance.
            annotate_labels: Carry the period names ('2022-08', ...) for annotation.
        """
        res = ce.results
        styles = {code: {'label': label} for code, label in ce.label_map.items()}
        thr_x = ce.var1_threshold if ce.var1_extreme == 'high' else -ce.var1_threshold
        thr_y = ce.var2_threshold if ce.var2_extreme == 'high' else -ce.var2_threshold
        return cls(
            x=res[ce.var1_z_col],
            y=res[ce.var2_z_col],
            category=res['CATEGORY'],
            category_styles=styles,
            labels=res['PERIOD'] if annotate_labels else None,
            threshold_x=thr_x,
            threshold_y=thr_y,
        )

    def _resolve_styles(self, overrides: dict) -> dict:
        overrides = overrides or {}
        resolved = {}
        for i, key in enumerate(self.category_order):
            base = dict(DEFAULT_CATEGORY_STYLES.get(key, {}))
            if not base:
                base = {'color': _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)],
                        'marker': _FALLBACK_MARKERS[i % len(_FALLBACK_MARKERS)]}
            base.setdefault('label', str(key))
            base.update(overrides.get(key, {}))
            resolved[key] = base
        return resolved

    def plot(self,
             ax=None,
             format_style: FormatStyle = None,
             figsize: tuple = (9, 8),
             figdpi: int = 100,
             markersize: float = 55,
             alpha: float = 0.9,
             edgecolor: str = 'none',
             annotate: bool = True,
             annotate_categories: list = None,
             annotate_fontsize: float = 8.0,
             annotate_offset: tuple = (4.0, 2.0),
             threshold_color: str = '#9E9E9E',
             legend: bool = True,
             legend_title: str = 'Extreme',
             showplot: bool = False):
        """Render the quadrant scatter (Phase 2).

        Args:
            ax: Target matplotlib axes. If None, a new figure is created.
            format_style: Shared :class:`~diive.plotting.FormatStyle` for chrome
                (title/labels/fonts/grid/legend). Axis labels default to the z-score
                column names.
            figsize: Figure size in inches when *ax* is None.
            figdpi: Figure DPI when *ax* is None.
            markersize: Marker area (matplotlib ``s``).
            alpha: Marker opacity.
            edgecolor: Marker edge colour (default 'none').
            annotate: Annotate points with their *labels* (e.g. period names).
            annotate_categories: Restrict annotation to these category keys (default:
                every category except 'none', so only extremes are labelled).
            annotate_fontsize: Annotation font size.
            annotate_offset: (dx, dy) point offset for annotation text.
            threshold_color: Colour of the dashed threshold lines.
            legend: Draw the category legend.
            legend_title: Legend title.
            showplot: Call ``fig.show()`` when a new figure was created.
        """
        style = format_style or FormatStyle()

        if ax is None:
            self.fig, self.ax = pf.create_ax(figsize=figsize)
            self.fig.set_dpi(figdpi)
            standalone = True
        else:
            self.ax = ax
            self.fig = ax.figure
            standalone = False
        ax = self.ax

        # Threshold (quadrant) lines first, so markers draw on top.
        if self.threshold_x is not None:
            ax.axvline(self.threshold_x, color=threshold_color, linestyle='--', linewidth=1.2, zorder=1)
        if self.threshold_y is not None:
            ax.axhline(self.threshold_y, color=threshold_color, linestyle='--', linewidth=1.2, zorder=1)

        if annotate_categories is None:
            annotate_categories = [c for c in self.category_order if c != CAT_NONE]

        for key in self.category_order:
            sub = self.df[self.df['_cat'] == key]
            if sub.empty:
                continue
            st = self.category_styles[key]
            ax.scatter(sub['_x'], sub['_y'], s=markersize, c=st['color'],
                       marker=st['marker'], alpha=alpha, edgecolors=edgecolor,
                       label=st['label'], zorder=3)
            if annotate and key in annotate_categories:
                for xv, yv, lab in zip(sub['_x'], sub['_y'], sub['_label']):
                    if lab:
                        ax.annotate(lab, (xv, yv), textcoords='offset points',
                                    xytext=annotate_offset, fontsize=annotate_fontsize,
                                    color=st['color'], zorder=4)

        style.apply(ax=ax, default_title="Compound extremes",
                    default_xlabel=self.xname, default_ylabel=self.yname)

        if legend:
            ax.legend(title=legend_title, loc='lower left', frameon=False)

        if standalone:
            self.fig.tight_layout()
            if showplot:
                self.fig.show()
