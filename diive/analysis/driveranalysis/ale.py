"""
DRIVERANALYSIS: ACCUMULATED LOCAL EFFECTS (ALE)
===============================================

Dependency-free Accumulated Local Effects for response-curve attribution.

ALE (Apley & Zhu, 2020) estimates the local effect of a feature on a model's
prediction by averaging the change in prediction across narrow feature bins. It
is correlation-robust: unlike partial dependence plots (PDP), it does not
extrapolate the model into regions of feature space where no data exist, which
makes it the appropriate response-curve tool for the strongly correlated drivers
typical of eddy-covariance data (e.g. SW_IN, TA, VPD all covary).

ALE describes how a *model* responds to a feature. It is an association-level
diagnostic and must never be presented as causal.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Material Design palette (see CLAUDE.md plotting conventions).
_MD_BLUE = '#2196F3'
_MD_BLUE_BG = '#BBDEFB'
_MD_GREY = '#455A64'


@dataclass
class AleCurve:
    """One-dimensional Accumulated Local Effects curve for a single feature.

    Attributes:
        feature: Name of the feature.
        bin_edges: Quantile bin edges (knot positions) at which the centered ALE
            is defined. Length ``n_bins + 1``.
        ale: Centered accumulated local effects, one value per bin edge. The mean
            effect is zero by construction, so values are read as a deviation
            from the average prediction (in target units).
        bin_counts: Number of observations falling in each bin. Length ``n_bins``.
        size: Total number of observations used.
    """
    feature: str
    bin_edges: np.ndarray
    ale: np.ndarray
    bin_counts: np.ndarray
    size: int

    @property
    def bin_centers(self) -> np.ndarray:
        """Midpoints between successive bin edges (one per bin)."""
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

    @property
    def ale_range(self) -> float:
        """Magnitude of the effect: ``max(ale) - min(ale)`` (target units)."""
        if self.ale.size == 0:
            return 0.0
        return float(np.nanmax(self.ale) - np.nanmin(self.ale))

    def direction(self, flat_threshold: float) -> str:
        """Classify the curve shape.

        Args:
            flat_threshold: ALE ranges at or below this value are reported as
                ``'flat'`` (no meaningful response, regardless of importance).

        Returns:
            One of ``'+'`` (increasing), ``'-'`` (decreasing), ``'∩'`` (unimodal
            peak), ``'∪'`` (unimodal trough), ``'flat'``, or ``'nonmonotonic'``.
        """
        if self.ale.size < 2 or self.ale_range <= flat_threshold:
            return 'flat'
        diffs = np.diff(self.ale)
        # Tolerance scaled to the curve range so near-zero wiggles are ignored.
        tol = 0.02 * self.ale_range
        up = diffs > tol
        down = diffs < -tol
        if not down.any():
            return '+'
        if not up.any():
            return '-'
        # A single sign change in the slope => unimodal.
        sign = np.sign(diffs[np.abs(diffs) > tol])
        n_sign_changes = int((np.diff(sign) != 0).sum()) if sign.size else 0
        if n_sign_changes == 1:
            return '∩' if sign[0] > 0 else '∪'
        return 'nonmonotonic'

    def plot(self, ax=None, color: str = _MD_BLUE, label: str = None,
             title: str = None, showplot: bool = False, **style):
        """Render the ALE curve (two-phase: styling lives here).

        Args:
            ax: Existing axes to draw on. A new figure is created when ``None``.
            color: Line color (Material Design blue by default).
            label: Legend label; defaults to the feature name.
            title: Axes title.
            showplot: Call ``fig.show()`` when a new figure was created.
            **style: Passed through to ``ax.plot``.
        """
        import matplotlib.pyplot as plt
        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(9, 6))
            created = True
        label = self.feature if label is None else label
        ax.plot(self.bin_edges, self.ale, color=color, marker='o', ms=4,
                label=label, zorder=5, **style)
        # Rug of bin edges shows where the data actually constrain the curve.
        ax.plot(self.bin_edges, np.full_like(self.bin_edges, self.ale.min()),
                marker='|', ls='none', color=_MD_GREY, alpha=0.5, zorder=4)
        ax.axhline(0, color=_MD_GREY, lw=0.8, ls='--', zorder=2)
        ax.set_xlabel(self.feature)
        ax.set_ylabel('ALE (effect on prediction, target units)')
        ax.set_title(title if title else f"ALE: {self.feature}")
        ax.legend(loc='best', fontsize=10)
        if created:
            ax.grid(True, ls='--', alpha=0.4, zorder=0)
            if showplot:
                fig.show()
        return ax


@dataclass
class Ale2DResult:
    """Two-dimensional (second-order) ALE for a pair of features.

    The second-order ALE isolates the *interaction* effect: main effects of each
    feature are removed, so a flat surface means the two features act additively.

    Attributes:
        f1, f2: Feature names (x and y axes).
        x_edges, y_edges: Quantile bin edges for each feature.
        ale: 2D array of centered second-order effects, shape
            ``(len(y_edges), len(x_edges))``.
    """
    f1: str
    f2: str
    x_edges: np.ndarray
    y_edges: np.ndarray
    ale: np.ndarray

    @property
    def interaction_strength(self) -> float:
        """Magnitude of the interaction surface (``max - min``)."""
        if self.ale.size == 0:
            return 0.0
        return float(np.nanmax(self.ale) - np.nanmin(self.ale))

    def plot(self, ax=None, cmap: str = 'RdBu_r', title: str = None,
             showplot: bool = False):
        """Render the 2D ALE surface as a diverging heatmap."""
        created = False
        if ax is None:
            from diive.core.plotting import plotfuncs as pf
            fig, ax = pf.create_ax(figsize=(8, 7))
            created = True
        vmax = max(abs(np.nanmin(self.ale)), abs(np.nanmax(self.ale)), 1e-12)
        mesh = ax.pcolormesh(self.x_edges, self.y_edges, self.ale,
                             cmap=cmap, vmin=-vmax, vmax=vmax, shading='auto')
        ax.figure.colorbar(mesh, ax=ax, label='2nd-order ALE (interaction)')
        ax.set_xlabel(self.f1)
        ax.set_ylabel(self.f2)
        ax.set_title(title if title else f"2D ALE: {self.f1} x {self.f2}")
        if created and showplot:
            ax.figure.show()
        return ax


def _quantile_edges(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Quantile-based, deduplicated bin edges robust to skewed distributions."""
    probs = np.linspace(0, 1, grid_size + 1)
    edges = np.quantile(values, probs)
    edges = np.unique(edges)
    if edges.size < 2:
        # Degenerate (near-constant) feature: fabricate a tiny span.
        lo = float(values.min())
        edges = np.array([lo, lo + 1e-9])
    return edges


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    # Pass the DataFrame as-is so models fitted with feature names stay happy.
    return np.asarray(model.predict(X), dtype=float)


def accumulated_local_effects(model, X: pd.DataFrame, feature: str,
                              grid_size: int = 20) -> AleCurve:
    """Compute the 1D Accumulated Local Effects curve for ``feature``.

    Correlation-robust alternative to partial dependence: predictions are only
    ever evaluated at the edges of bins that actually contain data, so the model
    is never extrapolated into empty regions of feature space.

    Args:
        model: Fitted regressor exposing ``.predict``.
        X: Feature matrix the model was trained on (a ``DataFrame``).
        feature: Column in ``X`` to compute the effect for.
        grid_size: Target number of quantile bins (the realized number may be
            smaller for low-cardinality features).

    Returns:
        AleCurve with centered effects defined at the bin edges.
    """
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not found in X columns.")

    # Bin the feature by quantiles, so each bin holds roughly equal data mass.
    x = X[feature].to_numpy(dtype=float)
    edges = _quantile_edges(x, grid_size)
    n_bins = edges.size - 1

    # Assign each row to a bin (last bin is right-inclusive).
    idx = np.searchsorted(edges, x, side='left')
    idx = np.clip(idx, 1, n_bins)

    # The ALE core idea: within each bin, take the rows that ACTUALLY fall there,
    # move the feature from the bin's lower edge to its upper edge, and measure how
    # the prediction changes. Averaging over the rows that live in the bin is what
    # keeps ALE honest under correlation — unlike PDP, we never ask the model about
    # feature combinations that don't occur in the data.
    local_delta = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for k in range(1, n_bins + 1):
        mask = idx == k
        counts[k - 1] = int(mask.sum())
        if not mask.any():
            continue
        X_lo = X.loc[mask].copy()
        X_hi = X_lo.copy()
        X_lo[feature] = edges[k - 1]   # feature pinned to bin's lower edge
        X_hi[feature] = edges[k]       # ... and to its upper edge
        # Mean per-bin effect = average prediction change across the move.
        local_delta[k - 1] = float(np.mean(_predict(model, X_hi) - _predict(model, X_lo)))

    # "Accumulate" = cumulative-sum the per-bin effects into a running curve
    # defined at the bin edges (the first edge is the zero reference point).
    ale_uncentered = np.concatenate([[0.0], np.cumsum(local_delta)])

    # Center the curve so its data-weighted mean is zero. Then a value reads as a
    # deviation from the average prediction (in target units), not an absolute.
    if counts.sum() > 0:
        seg_mid = (ale_uncentered[:-1] + ale_uncentered[1:]) / 2.0
        weighted_mean = float(np.sum(seg_mid * counts) / counts.sum())
    else:
        weighted_mean = 0.0
    ale = ale_uncentered - weighted_mean

    return AleCurve(feature=feature, bin_edges=edges, ale=ale,
                    bin_counts=counts, size=int(x.size))


def accumulated_local_effects_2d(model, X: pd.DataFrame, f1: str, f2: str,
                                 grid_size: int = 10) -> Ale2DResult:
    """Compute the second-order (interaction) ALE surface for two features.

    Implements the Apley & Zhu second-order estimator: per cell, the local
    second difference of the prediction is averaged, accumulated in both
    directions, then double-centered to remove first-order (main) effects.

    Args:
        model: Fitted regressor exposing ``.predict``.
        X: Feature matrix the model was trained on.
        f1: Feature mapped to the x-axis.
        f2: Feature mapped to the y-axis.
        grid_size: Target number of quantile bins per feature.

    Returns:
        Ale2DResult with the centered interaction surface.
    """
    for f in (f1, f2):
        if f not in X.columns:
            raise KeyError(f"Feature '{f}' not found in X columns.")

    x = X[f1].to_numpy(dtype=float)
    y = X[f2].to_numpy(dtype=float)
    xe = _quantile_edges(x, grid_size)
    ye = _quantile_edges(y, grid_size)
    nx, ny = xe.size - 1, ye.size - 1

    xi = np.clip(np.searchsorted(xe, x, side='left'), 1, nx)
    yi = np.clip(np.searchsorted(ye, y, side='left'), 1, ny)

    # 2D analogue of the 1D move: for the rows in each (x,y) cell, evaluate the
    # prediction at all FOUR corners and take the "second difference"
    # (ur - ul - lr + ll). That combination cancels each feature's solo effect,
    # leaving only what the two features do *jointly* — the interaction.
    delta = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=int)
    for a in range(1, nx + 1):
        for b in range(1, ny + 1):
            mask = (xi == a) & (yi == b)
            counts[b - 1, a - 1] = int(mask.sum())
            if not mask.any():
                continue
            base = X.loc[mask].copy()
            # Four corners of the cell (ll=low/low, ur=high/high, etc.).
            ll = base.copy(); ll[f1] = xe[a - 1]; ll[f2] = ye[b - 1]
            lr = base.copy(); lr[f1] = xe[a];     lr[f2] = ye[b - 1]
            ul = base.copy(); ul[f1] = xe[a - 1]; ul[f2] = ye[b]
            ur = base.copy(); ur[f1] = xe[a];     ur[f2] = ye[b]
            second_diff = (_predict(model, ur) - _predict(model, ul)
                           - _predict(model, lr) + _predict(model, ll))
            delta[b - 1, a - 1] = float(np.mean(second_diff))

    # Accumulate in both directions.
    acc = np.cumsum(np.cumsum(delta, axis=0), axis=1)
    # Pad to edge positions (prepend a zero row and column).
    acc = np.pad(acc, ((1, 0), (1, 0)), mode='constant')

    # Double-center: remove main effects via row/column means, restore grand mean.
    row_mean = np.nanmean(acc, axis=1, keepdims=True)
    col_mean = np.nanmean(acc, axis=0, keepdims=True)
    grand = float(np.nanmean(acc))
    ale = acc - row_mean - col_mean + grand

    return Ale2DResult(f1=f1, f2=f2, x_edges=xe, y_edges=ye, ale=ale)
