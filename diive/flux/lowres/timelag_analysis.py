"""
TIME LAG ANALYSIS: FLUX-CONCENTRATION PAIRING DETECTION
========================================================

Analyze and visualize time lags between gas concentrations and flux measurements.

Part of the diive library: https://github.com/holukas/diive

The module performs:
- Histogram-based lag distribution analysis
- Gradient-based peak range detection
- EddyPro-compatible lag range adjustment
- Multi-panel visualization with overview and zoomed views

Key Features:
    * Automated fringe bin exclusion to remove non-physical lag accumulations
    * Configurable gradient thresholds for sensitive edge detection
    * Reference lag window visualization for quality assessment
    * Batch processing of multiple gas species
    * Result caching for efficient reanalysis

Example:
    Load flux data and analyze CO2 time lags::

        from diive.core.io.files import load_parquet
        from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

        df = load_parquet('FLUXES_L0_ALL.parquet')
        analysis = TimeLagAnalysis(df, ignore_fringe_bins=[5, 10])
        analysis.plot_gas('CO2', outdir='output/')

References:
    Time lag detection is essential for accurate flux measurement in eddy covariance systems.
    EddyPro uses discrete 0.05s steps for lag scanning. This module provides tools to
    detect optimal lags from measured distributions and format them for EddyPro input.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import warnings

from diive.core.io.files import load_parquet
from diive.core.plotting.plotfuncs import default_format
from diive.core.utils.console import info, warn
from diive.analysis.histogram import Histogram


# Apply modern plot style
plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#dee2e6',
    'axes.grid': True,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlepad': 6,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.framealpha': 0.97,
    'legend.edgecolor': '#dee2e6',
    'legend.fontsize': 9,
    'grid.alpha': 0.25,
    'grid.color': '#dee2e6',
    'grid.linestyle': '-',
})


class TimeLagAnalysis:
    """
    Analyze and visualize time lags in eddy covariance flux measurements.

    Detects optimal time lags for gas concentrations (CO2, H2O) relative to wind
    measurements using histogram-based analysis with gradient-based peak detection.
    Generates 4-panel visualizations showing overview and zoomed histogram/time series.

    Key Features:
        - Automated fringe bin exclusion for non-physical lag accumulations
        - Gradient-based peak range detection with configurable sensitivity
        - EddyPro-compatible lag range adjustment (0.05s discrete steps)
        - Dynamic tick spacing to prevent label overlap
        - Result caching for efficient reanalysis

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with TLAG columns (e.g., 'CO2_TLAG_ACTUAL') and datetime index
    ignore_fringe_bins : list, optional
        Bin indices to exclude from histogram. Default: None
    lag_window_min, lag_window_max : float
        Reference acceptable lag range for visualization (seconds)
    histogram_startbin, histogram_endbin : int
        Bin range for display and analysis. Default: 0 to 10
    gradient_threshold : float
        Edge detection sensitivity; lower = stricter. Default: 0.15
    zoom_margin : list, optional
        [before_peak, after_peak] offsets for zoomed view (seconds). Default: [0.5, 0.8]

    Example
    -------
    See `examples/flux/lowres/flux_timelag_analysis.py` for complete examples.
    """

    def __init__(self,
                 df,
                 ignore_fringe_bins=None,
                 lag_window_min=0.10,
                 lag_window_max=1.00,
                 histogram_startbin=0,
                 histogram_endbin=10,
                 gradient_threshold=0.15,
                 zoom_margin=None):
        """
        Initialize TimeLagAnalysis with data and parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with TLAG columns (e.g., 'CO2_TLAG_ACTUAL', 'H2O_TLAG_ACTUAL')
            and a datetime index. Expected to contain columns matching pattern
            '*_TLAG_ACTUAL' for analysis.
        ignore_fringe_bins : list, optional
            Fringe bin indices to exclude from histogram computation. Fringe bins
            tend to accumulate non-physical lag values and should be excluded.
            Default: [5, 10]
        lag_window_min : float
            Lower bound of physically acceptable lag range in seconds. Used for
            reference line visualization (typically ~0.10s). Default: 0.10
        lag_window_max : float
            Upper bound of physically acceptable lag range in seconds. Used for
            reference line visualization (typically ~1.00s). Default: 1.00
        histogram_startbin : int
            First histogram bin index to display in plots. Default: 0
        histogram_endbin : int
            Last histogram bin index to display in plots. Default: 10
        gradient_threshold : float
            Threshold for gradient-based peak edge detection (0-1 range).
            Lower values = stricter edge detection. Controls where peak edges are
            identified based on histogram gradient magnitude. Default: 0.15
        zoom_margin : list, optional
            Zoom range offsets [before_peak, after_peak] in seconds for zoomed
            subplots. Defines how far left/right of peak to display in zoom views.
            Default: [0.5, 1.5]
        """
        self.df = df
        self.ignore_fringe_bins = ignore_fringe_bins or [5, 10]
        self.lag_window_min = lag_window_min
        self.lag_window_max = lag_window_max
        self.histogram_startbin = histogram_startbin
        self.histogram_endbin = histogram_endbin
        self.gradient_threshold = gradient_threshold
        self.zoom_margin = zoom_margin or [0.5, 1.5]

        # Extract and prepare time lag data
        tlag_cols = [c for c in df.columns if "TLAG" in c]
        self.tlag_actual_cols = [c for c in tlag_cols if c.endswith("_ACTUAL")]
        self.tlag_actual = df[self.tlag_actual_cols].copy()

        # Cache analysis results to avoid recomputation
        self._analysis_cache = {}

    def _extract_gas_from_column(self, col):
        """Extract gas name from column name (e.g., 'CO2_TLAG_ACTUAL' -> 'CO2')."""
        return col.split('_')[0]

    @staticmethod
    def _check_tick_overlap(nbins, axis_range, label_type='numeric', axis_width_inches=4.5):
        """
        Dynamically adjust nbins to maximize ticks while preventing overlap.

        Estimates label width in pixels and calculates how many labels fit in the
        available axis space, returning the maximum non-overlapping tick count.

        Parameters
        ----------
        nbins : int
            Initial estimated number of ticks
        axis_range : float
            Data range on the axis
        label_type : str
            Type of labels: 'numeric' (lag values), 'count' (integers), or 'date' (dates)
        axis_width_inches : float
            Estimated axis width in inches (default 4.5 for typical plots)

        Returns
        -------
        int
            Adjusted nbins that maximizes ticks while preventing overlap
        """
        if axis_range <= 0:
            return max(5, nbins)

        # Standard assumptions for label width in pixels (at 100 DPI, 9pt font)
        dpi = 100
        label_width_pixels = {
            'numeric': 25,      # "0.50", "-0.10" - ~25px wide
            'count': 30,        # "1000", "1500" - ~30px wide
            'date': 70,         # "2024-01-15" - ~70px wide
        }

        # Get estimated label width
        label_px = label_width_pixels.get(label_type, 30)

        # Add padding: 1.5x label width to account for spacing between labels
        total_space_per_label = label_px * 1.5

        # Calculate available axis space in pixels
        available_space_px = axis_width_inches * dpi

        # Calculate maximum number of labels that fit
        max_nbins_from_space = max(5, int(available_space_px / total_space_per_label))

        # Return the minimum to ensure no overlap, but maximize density
        return min(nbins, max_nbins_from_space)

    @staticmethod
    def adjust_range_for_eddypro(min_lag, max_lag, step=0.05):
        """
        Adjust detected lag range for EddyPro's discrete lag stepping behavior.

        EddyPro's covariance maximization searches at fixed step intervals (default 0.05s)
        and excludes the specified boundaries (inclusive lower, exclusive upper).
        To ensure a desired range [min_lag, max_lag] is actually used in EddyPro,
        the input must be expanded by one step on each side.

        **Example:**
        To ensure EddyPro uses the range [0.10-1.00] seconds::

            min_detected = 0.10
            max_detected = 1.00
            min_input, max_input = adjust_range_for_eddypro(min_detected, max_detected)
            # Result: min_input=0.05, max_input=1.05
            # EddyPro will check: 0.05, 0.10, 0.15, ..., 0.95, 1.00
            # (0.10 and 1.00 are included as desired)

        Parameters
        ----------
        min_lag : float
            Detected minimum lag value (seconds)
        max_lag : float
            Detected maximum lag value (seconds)
        step : float
            EddyPro's lag step size in seconds. Standard for EddyPro is 0.05s.
            Default: 0.05

        Returns
        -------
        eddypro_min : float
            Adjusted minimum for EddyPro input
        eddypro_max : float
            Adjusted maximum for EddyPro input

        Notes
        -----
        This is a utility function independent of class state and does not require
        an instance. Can be called as TimeLagAnalysis.adjust_range_for_eddypro().
        """
        eddypro_min = min_lag - step
        eddypro_max = max_lag + step
        return eddypro_min, eddypro_max

    @staticmethod
    def detect_peak_range(histogram_results, peakbins, gradient_threshold=0.15):
        """
        Detect the range around a histogram peak using gradient-based edge detection.

        Identifies peak boundaries by finding where the histogram gradient (first derivative)
        magnitude drops below a threshold. This method is robust to peak shape variations
        and works well for both symmetric and skewed distributions.

        **Algorithm:**
        1. Normalize counts to 0-1 range
        2. Compute first derivative (gradient) of normalized counts
        3. Locate peak bin (closest bin to peakbins[0])
        4. Search left: find first bin where |gradient| < threshold
        5. Search right: find first bin where |gradient| < threshold
        6. Return boundaries at these indices

        **Gradient threshold interpretation:**
        - 0.15 (default): detects moderate slope changes, suited for moderately sharp peaks
        - < 0.10: stricter, captures narrower peaks, sensitive to noise
        - > 0.20: lenient, captures broader tails, includes more uncertainty range

        Parameters
        ----------
        histogram_results : pd.DataFrame
            Histogram bin data with required columns:
            - 'BIN_START_INCL' : bin boundaries (inclusive)
            - 'COUNTS' : bin counts/frequencies
        peakbins : array-like
            Peak bin values where peakbins[0] is the primary peak lag (seconds)
        gradient_threshold : float
            Threshold magnitude for gradient-based edge detection (0-1 range).
            Represents normalized gradient magnitude below which edges are detected.
            Lower values = stricter edge detection, narrower range.
            Default: 0.15

        Returns
        -------
        min_lag : float
            Detected minimum lag boundary (seconds)
        max_lag : float
            Detected maximum lag boundary (seconds)

        Notes
        -----
        This is a utility function independent of class state and does not require
        an instance. Can be called as TimeLagAnalysis.detect_peak_range(...).

        The method assumes peakbins[0] is valid and falls within histogram_results bins.
        Edge cases: if gradient never drops below threshold, returns peak_idx boundaries.
        """
        bins = histogram_results['BIN_START_INCL'].values
        counts = histogram_results['COUNTS'].values

        # Normalize counts for gradient calculation
        max_count = counts.max()
        normalized_counts = counts / max_count if max_count > 0 else counts

        # Calculate gradient (first derivative)
        gradient = np.gradient(normalized_counts)

        # Find the peak index
        peak_idx = np.argmin(np.abs(bins - peakbins[0]))

        # Search left for edge: where gradient magnitude drops below threshold
        left_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if np.abs(gradient[i]) < gradient_threshold:
                left_idx = i
                break

        # Search right for edge: where gradient magnitude drops below threshold
        right_idx = peak_idx
        for i in range(peak_idx + 1, len(gradient)):
            if np.abs(gradient[i]) < gradient_threshold:
                right_idx = i
                break

        min_lag = bins[left_idx]
        max_lag = bins[right_idx]

        return min_lag, max_lag

    def analyze_gas(self, gas):
        """
        Analyze time lags for a specific gas species.

        Performs complete lag analysis workflow:
        1. Extracts time lag series for specified gas (*_TLAG_ACTUAL column)
        2. Creates histogram using unique value binning
        3. Excludes specified fringe bins (non-physical accumulations)
        4. Filters histogram to display range (startbin-endbin)
        5. Detects peak lag using histogram maximum
        6. Identifies peak range boundaries via gradient-based edge detection
        7. Adjusts range for EddyPro's discrete 0.05s steps

        Results are cached internally to avoid recomputation on repeated calls.

        Parameters
        ----------
        gas : str
            Gas name extracted from column prefix. Expected to match a column
            named '{gas}_TLAG_ACTUAL' (e.g., 'CO2', 'H2O'). Case-sensitive.

        Returns
        -------
        dict
            Complete analysis results with keys:
            - 'gas' : str – Gas name
            - 'series' : pd.Series – Full time lag series for gas
            - 'histogram_results' : pd.DataFrame – Histogram bins, counts
            - 'peakbins' : array – Peak bin values from Histogram class
            - 'peak' : float – Primary peak lag value (seconds)
            - 'peak_min', 'peak_max' : float – Detected peak range boundaries
            - 'eddypro_min', 'eddypro_max' : float – EddyPro-adjusted boundaries
            - 'first_date', 'last_date' : date – Data time span

        Raises
        ------
        ValueError
            If column '{gas}_TLAG_ACTUAL' not found in input dataframe.

        Notes
        -----
        Results are cached in self._analysis_cache. Subsequent calls with same gas
        return cached result without recomputation.
        """
        if gas in self._analysis_cache:
            return self._analysis_cache[gas]

        gascol = f'{gas}_TLAG_ACTUAL'
        if gascol not in self.tlag_actual.columns:
            raise ValueError(f"Column {gascol} not found in data")

        series = self.tlag_actual[gascol].copy()

        # Create histogram using Histogram class
        hist = Histogram(
            series=series,
            method='uniques',
            ignore_fringe_bins=self.ignore_fringe_bins
        )
        results = hist.results
        peakbins = hist.peakbins

        # Filter histogram bins to display range
        locs = ((results['BIN_START_INCL'] >= self.histogram_startbin) &
                (results['BIN_START_INCL'] <= self.histogram_endbin))
        results = results[locs].copy()

        # Detect peak and ranges
        peak = peakbins[0]
        peak_min, peak_max = self.detect_peak_range(
            results, peakbins, self.gradient_threshold
        )
        eddypro_min, eddypro_max = self.adjust_range_for_eddypro(
            peak_min, peak_max, step=0.05
        )

        analysis = {
            'gas': gas,
            'series': series,
            'histogram_results': results,
            'peakbins': peakbins,
            'peak': peak,
            'peak_min': peak_min,
            'peak_max': peak_max,
            'eddypro_min': eddypro_min,
            'eddypro_max': eddypro_max,
            'first_date': self.df.index[0].date(),
            'last_date': self.df.index[-1].date(),
        }

        self._analysis_cache[gas] = analysis
        return analysis

    def plot_gas(self, gas, outdir=None, figsize=(18, 9), show=True):
        """
        Create comprehensive visualization of time lag analysis for a gas species.

        Generates a 4-panel figure showing overview and zoomed perspectives:

        **Layout (2x2 grid):**
        - Top-left: Overview histogram with all lag bins and detection ranges
        - Top-right: Zoomed histogram centered on detected peak
        - Bottom-left: Time series of all lag observations over study period
        - Bottom-right: Zoomed time series around detected peak range

        **Overlays (all panels):**
        - Black line: Detected peak lag value
        - Teal shaded region + boundaries: Detected peak range (gradient-based)
        - Orange dashed lines + shaded region: EddyPro-adjusted input range
        - Purple dashed lines: Reference acceptable lag window

        **Legend** (overview histogram only):
        Shows peak value, detected range, EddyPro input range, and reference window.

        Automatically calls self.analyze_gas() if needed, using cached results if available.

        Parameters
        ----------
        gas : str
            Gas name (e.g., 'CO2', 'H2O'). Must match column prefix in data.
        outdir : str, optional
            Output directory for saving figure. If provided, saves as
            '{outdir}/02_{gas}_TLAG_ACTUAL_{first_date}_{last_date}.png' at 150 dpi.
            If None, figure not saved. Default: None
        figsize : tuple
            Figure size as (width, height) in inches. Default: (18, 9)
        show : bool
            If True, calls fig.show() to display the figure. If False, figure
            is created but not displayed (useful for batch processing).
            Default: True

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object. Can be further modified or saved.

        Raises
        ------
        ValueError
            If gas analysis fails (e.g., column not found). Error message printed
            to console but does not interrupt calling function.

        Notes
        -----
        Figure styling is applied via matplotlib rcParams set at module import.
        Colors and fonts are pre-configured and consistent across all plots.

        Examples
        --------
        Create and save visualization for CO2::

            analysis = TimeLagAnalysis(df)
            fig = analysis.plot_gas('CO2', outdir='output/', show=False)

        Create multiple gas plots in batch::

            for gas in ['CO2', 'H2O']:
                analysis.plot_gas(gas, outdir='output/', show=False)
        """
        analysis = self.analyze_gas(gas)

        series = analysis['series']
        results = analysis['histogram_results']
        peak = analysis['peak']
        peak_min = analysis['peak_min']
        peak_max = analysis['peak_max']
        eddypro_min = analysis['eddypro_min']
        eddypro_max = analysis['eddypro_max']
        first_date = analysis['first_date']
        last_date = analysis['last_date']
        gascol = f'{gas}_TLAG_ACTUAL'

        hist_bins = results['BIN_START_INCL'].copy()
        hist_counts = results['COUNTS'].copy()
        bin_width = 0.05
        bar_args = dict(width=bin_width, align='edge')
        zoom_min = peak - self.zoom_margin[0]
        zoom_max = peak + self.zoom_margin[1]

        # Auto-detect optimal number of ticks based on display ranges
        # Histogram x-axis (lag bins)
        overview_range = self.histogram_endbin - self.histogram_startbin + bin_width
        overview_nbins_x = max(10, int(overview_range / bin_width))
        overview_nbins_x = self._check_tick_overlap(overview_nbins_x, overview_range, 'numeric')

        zoom_range = zoom_max - zoom_min
        zoom_nbins_x = max(10, int(zoom_range / bin_width))
        zoom_nbins_x = self._check_tick_overlap(zoom_nbins_x, zoom_range, 'numeric')

        # Histogram y-axis (counts)
        max_count = hist_counts.max()
        overview_nbins_y = max(5, int(max_count / max(1, max_count // 5)))
        overview_nbins_y = self._check_tick_overlap(overview_nbins_y, max_count, 'count')

        # Time series y-axis (lag range)
        overview_lag_range = 10 - 0
        overview_nbins_y_ts = max(5, int(overview_lag_range / 0.5))
        overview_nbins_y_ts = self._check_tick_overlap(overview_nbins_y_ts, overview_lag_range, 'numeric')

        zoom_lag_range = zoom_max - zoom_min
        zoom_nbins_y_ts = max(5, int(zoom_lag_range / 0.25))
        zoom_nbins_y_ts = self._check_tick_overlap(zoom_nbins_y_ts, zoom_lag_range, 'numeric')

        # Time series x-axis (date range) - reasonable default for dates
        time_range_days = (series.index[-1] - series.index[0]).days
        nbins_x_ts = max(5, min(12, int(time_range_days / max(1, time_range_days // 6))))
        nbins_x_ts = self._check_tick_overlap(nbins_x_ts, time_range_days, 'date')

        # Create figure and layout
        fig = plt.figure(facecolor='#f8f9fa', figsize=figsize)

        # Header band
        fig.text(0.5, 0.97, f"Time Lag Analysis  ·  {gascol}",
                 ha='center', va='top', fontsize=18, fontweight='bold', color='#212529')
        fig.text(0.5, 0.935, f"{first_date}  –  {last_date}",
                 ha='center', va='top', fontsize=11, color='#6c757d')

        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            left=0.06, right=0.97, top=0.89, bottom=0.07,
            hspace=0.32, wspace=0.22
        )
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax_z = fig.add_subplot(gs[0, 1])
        ax2_z = fig.add_subplot(gs[1, 1])

        # Helper function to add range markers
        # Note: Lines mark inclusive range:
        # - peak_min at left edge (bin start)
        # - peak_max at right edge (bin start + bin width)
        def _add_range_markers(a, orient='v'):
            vline = a.axvline if orient == 'v' else a.axhline
            vspan = a.axvspan if orient == 'v' else a.axhspan
            vline(peak_min, color='#17a2b8', linewidth=2.0, alpha=0.85, zorder=4)
            vline(peak_max + bin_width, color='#17a2b8', linewidth=2.0, alpha=0.85, zorder=4)
            vspan(peak_min, peak_max + bin_width, alpha=0.12, color='#17a2b8', zorder=2)
            vline(eddypro_min, color='#fd7e14', linestyle=':', linewidth=2.0, alpha=0.8, zorder=3)
            vline(eddypro_max + bin_width, color='#fd7e14', linestyle=':', linewidth=2.0, alpha=0.8, zorder=3)
            vspan(eddypro_min, eddypro_max + bin_width, alpha=0.07, color='#fd7e14', zorder=1)
            vline(self.lag_window_min, color='#6f42c1', linestyle='--', linewidth=1.4, alpha=0.6, zorder=3)
            vline(self.lag_window_max + bin_width, color='#6f42c1', linestyle='--', linewidth=1.4, alpha=0.6, zorder=3)

        # Overview histogram (top-left)
        # Draw histogram bars, highlighting peak bin
        peak_bin_idx = np.where(hist_bins == peak)[0][0] if peak in hist_bins.to_numpy() else 0
        colors = ['#212529' if idx == peak_bin_idx else '#6c757d' for idx in range(len(hist_bins))]
        ax.bar(x=hist_bins, height=hist_counts, color=colors, zorder=90, **bar_args)
        _add_range_markers(ax, orient='v')

        legend_handles = [
            Patch(facecolor='#212529', label=f'Peak: {peak:.2f}s'),
            Patch(facecolor='#17a2b8', alpha=0.4, label=f'Detected: {peak_min:.2f}–{peak_max:.2f}s'),
            Patch(facecolor='#fd7e14', alpha=0.35, label=f'EddyPro: {eddypro_min:.2f}–{eddypro_max:.2f}s'),
            Line2D([0], [0], color='#6f42c1', linewidth=1.4, linestyle='--',
                   label=f'Window: {self.lag_window_min:.2f}–{self.lag_window_max:.2f}s'),
        ]
        legend = ax.legend(handles=legend_handles, loc='upper right', frameon=True, fancybox=False)
        legend.get_frame().set_linewidth(0.8)

        default_format(ax=ax, ax_xlabel_txt="Lag (s)", ax_ylabel_txt="Counts")
        ax.set_title("Overview — Histogram")
        ax.locator_params(axis='x', nbins=overview_nbins_x)
        ax.locator_params(axis='y', nbins=overview_nbins_y)
        ax.tick_params(axis='both', direction='out', top=False, right=False)
        ax.minorticks_off()

        # Overview time series (bottom-left)
        ax2.plot(series.index, series, alpha=0.55, c='#0d6efd', marker='.', ms=3.5, ls='none')
        _add_range_markers(ax2, orient='h')

        default_format(ax=ax2, ax_xlabel_txt="Date", ax_ylabel_txt="Lag (s)")
        ax2.set_title("Overview — Time Series")
        ax2.set_ylim([0, 10])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="'set_params\\(\\)' not defined for locator")
            ax2.locator_params(axis='x', nbins=nbins_x_ts)
        ax2.locator_params(axis='y', nbins=overview_nbins_y_ts)
        ax2.tick_params(axis='both', direction='out', top=False, right=False)
        ax2.minorticks_off()

        # Zoomed histogram (top-right)
        colors = ['#212529' if idx == peak_bin_idx else '#6c757d' for idx in range(len(hist_bins))]
        ax_z.bar(x=hist_bins, height=hist_counts, color=colors, zorder=90, **bar_args)
        _add_range_markers(ax_z, orient='v')

        default_format(ax=ax_z, ax_xlabel_txt="Lag (s)", ax_ylabel_txt="Counts")
        ax_z.set_title(f"Zoom [{self.zoom_margin[0]}s, +{self.zoom_margin[1]}s] — Histogram")
        ax_z.set_xlim(zoom_min, zoom_max)
        ax_z.locator_params(axis='x', nbins=zoom_nbins_x)
        ax_z.locator_params(axis='y', nbins=overview_nbins_y)
        ax_z.tick_params(axis='both', direction='out', top=False, right=False)
        ax_z.minorticks_off()

        # Zoomed time series (bottom-right)
        ax2_z.plot(series.index, series, alpha=0.55, c='#0d6efd', marker='.', ms=3.5, ls='none')
        _add_range_markers(ax2_z, orient='h')

        default_format(ax=ax2_z, ax_xlabel_txt="Date", ax_ylabel_txt="Lag (s)")
        ax2_z.set_title(f"Zoom [{self.zoom_margin[0]}s, +{self.zoom_margin[1]}s] — Time Series")
        ax2_z.set_ylim(zoom_min, zoom_max)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="'set_params\\(\\)' not defined for locator")
            ax2_z.locator_params(axis='x', nbins=nbins_x_ts)
        ax2_z.locator_params(axis='y', nbins=zoom_nbins_y_ts)
        ax2_z.tick_params(axis='both', direction='out', top=False, right=False)
        ax2_z.minorticks_off()

        if outdir:
            outfile = f"{outdir}/02_{gascol}_{first_date}_{last_date}.png"
            fig.savefig(outfile, dpi=150, bbox_inches='tight')
            info(f"Saved: {outfile}")

        if show:
            fig.show()

        return fig

    def analyze_all_gases(self, gases=None):
        """
        Analyze multiple gas species in batch mode.

        Automatically detects all available gases (based on *_TLAG_ACTUAL column names)
        if not explicitly specified. Calls analyze_gas() for each gas and returns
        consolidated results. Errors in individual gas analyses are caught and reported
        without interrupting the batch process.

        Parameters
        ----------
        gases : list, optional
            Explicit list of gas names to analyze (e.g., ['CO2', 'H2O']).
            Names must match column prefixes in data (case-sensitive).
            If None, automatically detects all gases from available TLAG_ACTUAL columns.
            Default: None (auto-detect)

        Returns
        -------
        dict
            Analysis results keyed by gas name. Each key maps to the dict returned
            by analyze_gas(). If a gas fails to analyze, it is excluded from results
            and a warning message is printed.

        Examples
        --------
        Auto-detect all available gases::

            all_results = analysis.analyze_all_gases()

        Analyze specific gases only::

            results = analysis.analyze_all_gases(gases=['CO2', 'H2O'])

        Notes
        -----
        Failed analyses (e.g., missing columns) print warnings but do not raise
        exceptions. This allows batch processing to continue even if some gases
        cannot be analyzed.
        """
        if gases is None:
            gases = sorted(set(
                self._extract_gas_from_column(c) for c in self.tlag_actual_cols
            ))

        results = {}
        for gas in gases:
            try:
                results[gas] = self.analyze_gas(gas)
            except ValueError as e:
                warn(f"Could not analyze {gas}: {e}")

        return results

    def plot_all_gases(self, gases=None, outdir=None, figsize=(18, 9), show=True):
        """
        Create and optionally save visualizations for multiple gas species.

        Calls plot_gas() for each specified or auto-detected gas. Results in one
        4-panel figure per gas. All parameters are passed through to plot_gas().
        Errors in individual plots are caught and reported without interrupting
        the batch process.

        Parameters
        ----------
        gases : list, optional
            Explicit list of gas names to plot (e.g., ['CO2', 'H2O']).
            Names must match column prefixes in data (case-sensitive).
            If None, automatically detects all gases from available TLAG_ACTUAL columns.
            Default: None (auto-detect)
        outdir : str, optional
            Output directory for saving figures. Each figure saved as
            '{outdir}/02_{gas}_TLAG_ACTUAL_{first_date}_{last_date}.png' at 150 dpi.
            If None, figures not saved (only displayed if show=True).
            Default: None
        figsize : tuple
            Figure size as (width, height) in inches. Applied to all plots.
            Default: (18, 9)
        show : bool
            If True, each figure is displayed via fig.show(). Useful for interactive
            analysis. If False, figures created but not displayed (for batch processing).
            Default: True

        Returns
        -------
        dict
            Matplotlib figure objects keyed by gas name. Each value is the Figure
            returned by plot_gas(). If a gas fails to plot, it is excluded from
            results and a warning message is printed.

        Examples
        --------
        Plot all detected gases and save to output directory::

            figs = analysis.plot_all_gases(outdir='output/')

        Plot specific gases without displaying::

            figs = analysis.plot_all_gases(
                gases=['CO2', 'H2O'],
                outdir='output/',
                show=False
            )

        Notes
        -----
        If outdir is provided, creates figures even if show=False. This is useful
        for batch processing without blocking on figure display.

        Failed plots (e.g., missing columns) print warnings but do not raise
        exceptions, allowing batch processing to continue.
        """
        if gases is None:
            gases = sorted(set(
                self._extract_gas_from_column(c) for c in self.tlag_actual_cols
            ))

        figs = {}
        for gas in gases:
            try:
                figs[gas] = self.plot_gas(gas, outdir=outdir, figsize=figsize, show=show)
            except ValueError as e:
                warn(f"Could not plot {gas}: {e}")

        return figs
