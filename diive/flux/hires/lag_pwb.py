"""
LAG_PWB: PRE-WHITENING WITH BLOCK-BOOTSTRAP TIME LAG DETECTION
===============================================================

In eddy covariance systems, gas analyzers connected via a sample tube receive
the air parcel slightly later than the sonic anemometer measures the vertical
wind at the intake. This offset -- the time lag -- must be detected and
corrected before computing flux covariances.  For reactive or trace gases such
as N2O and CH4, the gas-wind cross-correlation is often too weak to pinpoint
the lag reliably with conventional covariance-maximization (CM) methods.

This module implements the pre-whitening with block-bootstrap (PWB) procedure
introduced by Vitale et al. (2024) to overcome that limitation.  The algorithm
works in two stages:

**Pre-whitening** removes serial autocorrelation from each time series before
computing the cross-correlation function (CCF).  Turbulent wind and trace-gas
series have strong autocorrelation that broadens and distorts the CCF peak; an
AR(p) filter estimated by AIC makes the residuals approximately white noise,
sharpening the peak and enabling a cleaner lag estimate.

**Block-bootstrap** quantifies the uncertainty of the detected lag.  Instead
of relying on a single CCF computed from the full averaging period, the method
draws N_B resampled series (using non-overlapping blocks of length L = 20 s to
preserve local autocorrelation structure), detects the peak lag in each, and
summarises the resulting distribution with a mode (the lag estimate) and a 95%
Highest Density Interval (HDI).  A narrow HDI (< 0.5 s) indicates that
repeated resampling consistently finds the same lag -- the S1 reliability
criterion from the paper.

**Four CCF combinations** (following RFlux v3.2.0) are evaluated when the
sonic temperature T_SONIC is provided.  For each combination a different AR
filter is applied before computing the CCF between the scalar and either W or
T_SONIC.  Because T_SONIC and W are correlated through buoyant turbulence, the
T_SONIC-based combinations often reveal a cleaner lag peak for gases with a
weak scalar x W signal:

    cw  -- scalar x W,       scalar AR filter   (R: bootccf_cw / x1, z1)
    ct  -- scalar x T_SONIC, scalar AR filter   (R: bootccf_ct / x1, y1)
    wc  -- scalar x W,       W AR filter        (R: bootccf_wc / x3, z3)
    tc  -- scalar x T_SONIC, T_SONIC AR filter  (R: bootccf_tc / x2, y2)

The combination that produces the highest smoothed CCF peak at its mode lag is
selected as the most informative for that averaging period
(R: which.max(abs(corr_est_s))).

Implementation notes:
- An ADF unit-root test (R: egcm::bvr.test, same H0 direction) is applied to
  each aligned series before AR fitting.  If any series has a unit root
  (p >= 0.01), all series are first-differenced.  For turbulent EC data this
  virtually never triggers but handles pathological non-stationary periods.
- Three separate AR(p) models are fitted (scalar, W, T_SONIC) using AIC with
  max_order = floor(100 * log10(N)), matching R's formula.  For N = 36000
  records this gives max_order = 455, which is large enough for AIC to capture
  the long-range autocorrelation typical of turbulent series.
- Linear interpolation (na.approx) is applied to input NaN before AR fitting.
- Forward-fill + backward-fill (na.locf, two-pass) fills the NaN edges of each
  per-replicate smoothed CCF before the peak search so that lags at the
  boundary of the search window can still be selected.
- Providing var_tsonic is strongly recommended for trace gases (N2O, CH4);
  without it, only the two scalar x W combinations (cw and wc) are evaluated.

Input data requirement:
    All classes in this module require **wind-rotation-corrected**
    high-frequency data (double rotation or planar-fit, e.g. EddyPro "Advanced"
    rotated output).  Wind rotation removes the mean vertical-wind offset so that
    W contains only turbulent fluctuations; a non-zero mean W corrupts the
    cross-correlation and yields an unreliable lag estimate.

Batch processing across many files:
    ``PwbBatchDetection`` (also in this module) wraps ``PreWhiteningBootstrap``
    and distributes files across CPU cores via ``ProcessPoolExecutor``, collects
    results into a single DataFrame, applies PWBOPT post-processing
    (S1/S2/S3 selection and optional HDI pre-filter), fills remaining NaN lags
    via ``fill_tlag_gaps()``, and produces batch summary figures.

    The module is also directly executable as a CLI tool -- see the CLI section
    below for full usage.

CLI usage
---------
Run the batch detector from the command line::

    python -m diive.pkgs.flux.hires.lag_pwb --help

or with the installed console-script alias (after ``uv sync`` or ``pip install``)::

    uv run diive-tlag-pwb-batch --help

**Required flags**

``--input-dir PATH``
    Directory containing EddyPro rotated ``.txt`` files.
``--output-dir PATH``
    Output directory for the results CSV, checkpoint, and optional plots.
``--scalar LABEL:column``
    Gas label and column name, e.g. ``CH4:ch4``.  Repeat for each gas.

**Column mapping**

``--col-w NAME``
    Column name for the vertical wind component W.  Default: ``w``.
``--col-tsonic NAME``
    Column name for sonic temperature T_SONIC.  Enables the full 4-combination
    RFlux v3.2.0 logic (strongly recommended for trace gases).
``--usecols I [I ...]``
    0-based column indices to read from each file.  Default: ``0 1 2 3 4 5``.
``--col-names N [N ...]``
    Column names assigned to the selected columns.
    Default: ``u v w ts ch4 n2o``.
``--skiprows N``
    Number of metadata rows before the column-name header row.
    EddyPro default: 9.
``--na-values V [V ...]``
    Strings treated as NaN.  Default: ``-9999``, ``-9999.0``,
    ``-9999.0000000000000``.

**Detection parameters**

``--hz N``
    Sampling frequency in Hz.  Default: 20.
``--lag-max S``
    CCF search half-width in seconds.  Default: 10.0.
``--n-bootstrap N``
    Number of block-bootstrap replicates.  Default: 99.
    Use 9 for fast testing; 99-999 for production.
``--block-length S``
    Bootstrap block length in seconds.  Default: 20.0.
``--min-valid-frac F``
    Minimum fraction of non-NaN values required for a series to be processed.
    Default: 0.3.

**PWBOPT post-processing thresholds**

``--hdi-thresh S``
    S1 threshold: periods with HDI range < this value (seconds) are flagged
    S1_optimal (reliable detection).  Default: 0.5.
``--dev-thresh S``
    S2 threshold: periods with HDI >= hdi-thresh but lag within this distance
    (seconds) of the preceding optimal are flagged S2_optimal.  Default: 0.5.
``--hdi-prefilter S``
    Pre-filter: lags with HDI range > this value are set to NaN before PWBOPT
    so that S2 cannot accept wide-uncertainty detections.  Set to 0 to
    disable.  Default: 1.0.

**Execution**

``--n-workers N``
    Number of parallel worker processes.  Default: all available CPU cores.
``--file-pattern PATTERN``
    Glob pattern used to collect input files.  Default: ``*.txt``.
``--save-plots``
    Save one diagnostic PNG per averaging period per scalar to
    ``<output-dir>/plots/``.

**EddyPro 10-column rotated file layout (0-indexed)**::

    col  0:u   1:v   2:w   3:ts   4:co2   5:h2o   6:ch4   7:4th_gas   8:air_t   9:air_p

    To select u, v, w, ts, ch4, 4th_gas (as N2O):
        --usecols 0 1 2 3 6 7 --col-names u v w ts ch4 n2o

**Example — bash (multi-line)**::

    uv run diive-tlag-pwb-batch \\
        --input-dir  /path/to/hires_files \\
        --output-dir /path/to/results \\
        --scalar CH4:ch4 --scalar N2O:n2o \\
        --col-w w --col-tsonic ts \\
        --usecols 0 1 2 3 6 7 --col-names u v w ts ch4 n2o \\
        --skiprows 9 --hz 20 --lag-max 10.0 \\
        --n-bootstrap 99 --block-length 20.0 --min-valid-frac 0.3 \\
        --hdi-thresh 0.5 --dev-thresh 0.5 --hdi-prefilter 1.0 \\
        --n-workers 16 --save-plots

**Example — PowerShell / Windows (one-liner)**::

    uv run diive-tlag-pwb-batch --input-dir "F:\\path\\to\\input" --output-dir "F:\\path\\to\\output" --scalar CH4:ch4 --scalar N2O:n2o --col-w w --col-tsonic ts --usecols 0 1 2 3 6 7 --col-names u v w ts ch4 n2o --skiprows 9 --hz 20 --lag-max 10.0 --n-bootstrap 99 --block-length 20.0 --min-valid-frac 0.3 --hdi-thresh 0.5 --dev-thresh 0.5 --hdi-prefilter 1.0 --n-workers 16 --save-plots

Windows note:
    ``ProcessPoolExecutor`` uses the *spawn* start method on Windows.  Any
    script that instantiates ``PwbBatchDetection`` must guard its entry point::

        if __name__ == '__main__':
            det = PwbBatchDetection(...)
            results = det.run()

References:
    Vitale D, Fratini G, Helfter C, Hortnagl L, et al. (2024) A pre-whitening
    with block-bootstrap cross-correlation procedure for temporal alignment of
    data sampled by eddy covariance systems. Environmental and Ecological
    Statistics 31:219-244. doi:10.1007/s10651-024-00615-9

Part of the diive library: https://github.com/holukas/diive
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

# Suppress the runpy double-import warning that fires in every worker process
# when diive.__init__ has already imported this module before -m re-executes it.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import correlate as _signal_correlate, lfilter
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller

from diive.core.plotting.plotfuncs import default_format
from diive.core.plotting.styles import LightTheme as theme

# Default NA strings matching EddyPro rotated-file conventions
_DEFAULT_NA_VALUES = ['-9999', '-9999.0', '-9999.0000000000000']

# Smoothing widths for the full-data PW CCF diagnostic panels (older R convention).
_SMOOTH_WIDTH_CCF = 13
_SMOOTH_WIDTH_CCOV = 3


def _na_approx(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of NaN, matching R's zoo::na.approx(na.rm=FALSE).

    Interior NaN are linearly interpolated between their neighbours.  Leading
    and trailing NaN are filled with the nearest valid boundary value (constant
    extrapolation), which matches numpy.interp's default clamping behaviour and
    pandas interpolate(limit_direction='both').
    """
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    out = x.copy()
    # np.interp clamps to boundary values for indices outside valid range,
    # matching R's na.approx edge behaviour.
    out[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return out


def _na_locf_1d(x: np.ndarray) -> np.ndarray:
    """
    Forward-fill then backward-fill NaN, matching R's zoo::na.locf two-pass.

    Forward fill carries the last valid value rightward (fills trailing NaN).
    Backward fill carries the next valid value leftward (fills leading NaN).
    """
    nans = np.isnan(x)
    if not nans.any():
        return x
    col = np.arange(len(x))
    out = x.copy()
    # Forward fill: for each NaN at position j, use the last valid index before j
    idx = np.where(~nans, col, 0)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(nans, out[idx], out)
    # Backward fill: remaining left-edge NaN get the next valid value after them
    nans2 = np.isnan(out)
    if nans2.any():
        idx2 = np.where(~nans2, col, len(x) - 1)
        np.minimum.accumulate(idx2[::-1], out=idx2[::-1])
        out = np.where(nans2, out[idx2], out)
    return out


class PreWhiteningBootstrap:
    """
    Detect the tube-delay time lag between a scalar gas concentration and the
    vertical wind component using the pre-whitening with block-bootstrap (PWB)
    cross-correlation procedure (Vitale et al. 2024, RFlux v3.2.0).

    .. important::
        Input data must be **wind-rotation-corrected** (double rotation or
        planar-fit, e.g. from EddyPro "Advanced" rotated output).  Wind
        rotation removes the mean vertical-wind offset so that W consists
        only of turbulent fluctuations.  A non-zero mean W biases the
        cross-correlation and produces an unreliable lag estimate.

    **Why pre-whitening?**  Turbulent wind and scalar series carry strong
    autocorrelation that broadens the raw CCF peak.  An AR(p) filter estimated
    by AIC transforms each series into approximate white noise; the CCF of the
    filtered residuals has a sharper peak, making the lag easier to locate.

    **Why block-bootstrap?**  A single CCF can be noisy.  The bootstrap draws
    N_B resampled series (in non-overlapping blocks of L = 20 s to preserve
    local autocorrelation) and detects the peak lag in each.  The mode of the
    resulting N_B lag estimates is the final lag; the 95% HDI (Highest Density
    Interval, shortest interval containing 95% of the distribution) measures
    how consistently the resampling agrees on that lag.  A narrow HDI means
    the lag is robust; a wide HDI signals an unreliable estimate.

    **Why four combinations?**  When *var_tsonic* is provided, four
    pre-whitened CCF variants are computed.  T_SONIC and W co-vary through
    buoyant turbulent structures; for trace gases (N2O, CH4) the scalar x W
    signal is often too weak to produce a clear peak, whereas scalar x T_SONIC
    combinations can reveal the same tube-delay lag more clearly.  The
    combination that produces the highest smoothed CCF value at its mode lag is
    selected (R: which.max(abs(corr_est_s))):

        cw  -- scalar x W,       scalar AR filter
        ct  -- scalar x T_SONIC, scalar AR filter
        wc  -- scalar x W,       W AR filter
        tc  -- scalar x T_SONIC, T_SONIC AR filter

    Without *var_tsonic*, only cw and wc are evaluated.

    **Two lag estimates are returned:**

    - *tlag_pw* -- Pre-whitening lag: argmax of the smoothed scalar-AR CCF on
      the full data.  Kept for diagnostic purposes (panel 1 of the plot).
    - *tlag_s* -- Bootstrap lag (the primary output): mode of detected lags
      across N_B bootstrap samples for the best-performing combination.

    **Reliability criterion (S1):** HDI range < 0.5 s.  Use *is_reliable* to
    check; unreliable segments should fall back to a carry-forward strategy
    (PWBOPT S2/S3, see PwboptLagDetection).

    The *plot()* method shows three diagnostic panels:

    - Panel 1 (left):  Pre-whitened CCF (scalar AR, full data) with Bartlett
      significance bands and the detected-lag marker.
    - Panel 2 (middle): Raw cross-covariance with smoothed overlay.
    - Panel 3 (right): Bootstrap lag histogram with 95% HDI and mode marker for
      the selected combination.

    Attributes:
        df (DataFrame): Input high-frequency data.
        var_w (str): Column name for vertical wind component W.
        var_scalar (str): Column name for scalar gas concentration.
        var_tsonic (str | None): Column name for sonic temperature T_SONIC.
            Strongly recommended for trace gases; enables the 4-combination
            RFlux v3.2.0 logic.
        hz (int): Acquisition frequency in Hz.
        lag_max_s (float): Half-width of lag search window in seconds.
        n_bootstrap (int): Number of block-bootstrap samples (N_B).
        block_length_s (float): Bootstrap block length in seconds (L).
        segment_name (str): Label for this averaging segment.

    See Also
    --------
    MaxCovariance : Time lag detection via covariance maximisation (CM method).

    Example
    -------
    See `examples/flux/hires/flux_lag_pwb.py` for a complete example.
    """

    def __init__(
            self,
            df: DataFrame,
            var_w: str,
            var_scalar: str,
            var_tsonic: str = None,
            hz: int = 20,
            lag_max_s: float = 10.0,
            n_bootstrap: int = 99,
            block_length_s: float = 20.0,
            segment_name: str = 'segment',
    ):
        """
        Store data and parameters. No computation is performed here.

        Positive lag means *var_scalar* lags behind *var_w* (arrives later,
        physical tube delay). Negative lag means *var_scalar* arrives before
        *var_w*.

        Args:
            df (DataFrame): High-frequency eddy covariance data.
            var_w (str): Column name for vertical wind component W.
            var_scalar (str): Column name for scalar gas concentration (CO2,
                CH4, N2O, ...).
            var_tsonic (str | None): Column name for sonic temperature T_SONIC.
                When supplied, the full 4-combination RFlux v3.2.0 logic is
                used, which is essential for trace gases with a weak W
                cross-correlation.  Defaults to None.
            hz (int): Acquisition frequency in samples per second (10 or 20).
                Defaults to 20.
            lag_max_s (float): Half-width of lag search window in seconds
                (R: LAG.MAX = mfreq * 10 = 10 s). Defaults to 10.0.
            n_bootstrap (int): Number of block-bootstrap samples (N_B in the
                paper). Defaults to 99.
            block_length_s (float): Bootstrap block length in seconds (L = 20 s
                in the paper). Defaults to 20.0.
            segment_name (str): Identifier for this averaging period. Defaults
                to 'segment'.
        """
        self.df = df
        self.var_w = var_w
        self.var_scalar = var_scalar
        self.var_tsonic = var_tsonic
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.segment_name = segment_name

        # Search window half-width in records (R: LAG.MAX = mfreq * 10)
        self._lag_max_records = int(round(lag_max_s * hz))
        # Block length in records (paper: L = 20 s, Section 2.2)
        self._block_length_records = int(round(block_length_s * hz))
        # Bootstrap CCF smoothing width: hz/2 + 1 (R: wdt = floor(mfreq/2) + 1)
        self._smooth_width_bootstrap = hz // 2 + 1

        # --- Results populated by run() ---
        # PW results (full-data scalar-AR CCF, for diagnostic panels)
        self._tlag_pw_records: int | None = None
        self._tlag_opt_records: int | None = None
        self._tlag_lmax: list | None = None
        self._tlag_lmin: list | None = None
        self._corr_est: float | None = None
        self._n_eff: int | None = None
        # PWB results (from the selected combination's bootstrap)
        self._tlag_records: int | None = None
        self._hdi_lo_s: float | None = None
        self._hdi_hi_s: float | None = None
        self._bootstrap_lags: np.ndarray | None = None
        # AR model info
        self._ar_order: int | None = None  # scalar AR order (primary)
        self._ar_orders: dict | None = None  # all fitted AR orders
        self._best_combination: str | None = None
        # Arrays kept for plotting
        self._lags_axis: np.ndarray | None = None
        self._raw_ccov: np.ndarray | None = None
        self._smooth_raw_ccov: np.ndarray | None = None
        self._pw_ccf: np.ndarray | None = None
        self._smooth_pw_ccf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties -- PW results
    # ------------------------------------------------------------------

    @property
    def tlag_pw_records(self) -> int:
        """PW time lag in records (argmax of smoothed pre-whitened CCF)."""
        if self._tlag_pw_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_pw_records

    @property
    def tlag_pw_s(self) -> float:
        """PW time lag in seconds."""
        return self.tlag_pw_records / self.hz

    @property
    def tlag_opt_records(self) -> int:
        """Optimal PW lag in records (after decision rules)."""
        if self._tlag_opt_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_opt_records

    @property
    def tlag_opt_s(self) -> float:
        """Optimal PW lag in seconds."""
        return self.tlag_opt_records / self.hz

    @property
    def corr_est(self) -> float:
        """Un-smoothed CCF value at tlag_opt."""
        if self._corr_est is None:
            raise RuntimeError("Call run() first.")
        return self._corr_est

    @property
    def cv5pct(self) -> float:
        """5% Bartlett significance threshold: 1.96 / sqrt(N)."""
        if self._n_eff is None:
            raise RuntimeError("Call run() first.")
        return 1.96 / np.sqrt(self._n_eff)

    @property
    def cv1pct(self) -> float:
        """1% Bartlett significance threshold: 2.57 / sqrt(N)."""
        if self._n_eff is None:
            raise RuntimeError("Call run() first.")
        return 2.57 / np.sqrt(self._n_eff)

    # ------------------------------------------------------------------
    # Properties -- PWB results
    # ------------------------------------------------------------------

    @property
    def tlag_records(self) -> int:
        """Bootstrap (PWB) time lag in records (mode over N_B samples)."""
        if self._tlag_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_records

    @property
    def tlag_s(self) -> float:
        """Bootstrap (PWB) time lag in seconds."""
        return self.tlag_records / self.hz

    @property
    def hdi_lo_s(self) -> float:
        """Lower bound of 95% HDI in seconds."""
        if self._hdi_lo_s is None:
            raise RuntimeError("Call run() first.")
        return self._hdi_lo_s

    @property
    def hdi_hi_s(self) -> float:
        """Upper bound of 95% HDI in seconds."""
        if self._hdi_hi_s is None:
            raise RuntimeError("Call run() first.")
        return self._hdi_hi_s

    @property
    def hdi_range_s(self) -> float:
        """Width of 95% HDI in seconds (uncertainty measure)."""
        return self.hdi_hi_s - self.hdi_lo_s

    @property
    def is_reliable(self) -> bool:
        """True if HDI range < 0.5 s (S1 reliability criterion, paper Section 2.3)."""
        return self.hdi_range_s < 0.5

    @property
    def results(self) -> dict:
        """All key outputs as a dictionary."""
        return {
            'segment_name': self.segment_name,
            # PW results (full-data scalar-AR CCF, for diagnostics)
            'tlag_pw_records': self.tlag_pw_records,
            'tlag_pw_s': self.tlag_pw_s,
            'opt_tlag_records': self.tlag_opt_records,
            'opt_tlag_s': self.tlag_opt_s,
            'tlag_lmax': self._tlag_lmax,
            'tlag_lmin': self._tlag_lmin,
            'corr_est': self.corr_est,
            'cv5pct': self.cv5pct,
            'cv1pct': self.cv1pct,
            'ar_order': self._ar_order,  # scalar AR order (primary)
            'ar_orders': self._ar_orders,  # all fitted AR orders
            'best_combination': self._best_combination,
            # PWB results (from the selected combination)
            'tlag_records': self.tlag_records,
            'tlag_s': self.tlag_s,
            'hdi_lo_s': self.hdi_lo_s,
            'hdi_hi_s': self.hdi_hi_s,
            'hdi_range_s': self.hdi_range_s,
            'is_reliable': self.is_reliable,
            'n_bootstrap': self.n_bootstrap,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute the full PWB pipeline following RFlux v3.2.0.

        Steps
        -----
        1. Load all input series and linearly interpolate NaN (R: na.approx).
           Drop rows where any series remains NaN after interpolation.
        1b. ADF unit-root test on each series (R: egcm::bvr.test).  If any
           series has a unit root (p >= 0.01), first-difference all series.
        2. Fit separate AR(p) models to the scalar, W, and T_SONIC (when
           provided) using AIC selection with max_order = floor(100*log10(N))
           (R: ar(x, aic=TRUE, order.max=floor(10^2*log10(length(x))))).
        3. Apply each AR filter to all relevant series to produce the
           pre-whitened filtered arrays (R: x1/y1/z1/x2/y2/x3/z3).
        4. Compute raw cross-covariance on the original data (diagnostic panel 2).
        5. Compute full-data PW CCF using the scalar AR filter on W and scalar
           (R: ccf_pww). Detect tlag_pw / tlag_opt for diagnostic panel 1.
        6. Block-bootstrap each combination (2 or 4):
             cw  -- scalar x W,       scalar AR (R: bootccf_cw)
             wc  -- scalar x W,       W AR      (R: bootccf_wc)
             ct  -- scalar x T_SONIC, scalar AR (R: bootccf_ct)  [if var_tsonic]
             tc  -- scalar x T_SONIC, T_SONIC AR(R: bootccf_tc)  [if var_tsonic]
           Per replicate: smooth CCF with width=hz/2+1, apply na.locf to edges,
           find the peak lag index.  Also compute the mean bootstrap CCF,
           smoothed with the same width + na.locf (R: ccf_cw / ccfs_cw).
        7. Select the combination with the highest |avg_smooth_ccf| at its mode
           lag (R: corr_est_s -> which.max -> corr_ind).
        8. Summarise the winning combination's bootstrap distribution:
           mode -> tlag_s, 95% HDI -> hdi_lo_s / hdi_hi_s.
        """
        # ---- Step 1: load, interpolate NaN, align ----
        w_raw = _na_approx(self.df[self.var_w].values.astype(float))
        s_raw = _na_approx(self.df[self.var_scalar].values.astype(float))
        has_tsonic = self.var_tsonic is not None

        if has_tsonic:
            t_raw = _na_approx(self.df[self.var_tsonic].values.astype(float))
            valid = ~np.isnan(w_raw) & ~np.isnan(s_raw) & ~np.isnan(t_raw)
        else:
            t_raw = None
            valid = ~np.isnan(w_raw) & ~np.isnan(s_raw)

        w = w_raw[valid]
        s = s_raw[valid]
        t = t_raw[valid] if has_tsonic else None

        # ---- Step 1b: stationarity check (R: egcm::bvr.test) ----
        # ADF unit-root test on each aligned series.  If ANY series has a unit
        # root (p >= 0.01), first-difference ALL series before AR fitting.
        # This matches R's logic exactly: stationarity of all three is required
        # to use the original series; a single failure triggers differencing.
        # For turbulent EC data the test virtually always passes.  The rare
        # failures occur during sensor drift, rain events, or other artefacts.
        series_to_test = [s, w] + ([t] if has_tsonic else [])
        if not all(self._is_stationary(x) for x in series_to_test):
            s = np.diff(s)
            w = np.diff(w)
            if has_tsonic:
                t = np.diff(t)

        self._lags_axis = np.arange(-self._lag_max_records, self._lag_max_records + 1)

        # ---- Step 2: fit separate AR models ----
        # R: ar.resx = ar(x=scalar, ...), ar.resz = ar(z=W, ...), ar.resy = ar(y=T_SONIC, ...)
        phi_s, p_s = self._fit_ar_model(s)  # scalar AR
        phi_w, p_w = self._fit_ar_model(w)  # W AR
        self._ar_order = p_s
        self._ar_orders = {'scalar': p_s, 'w': p_w}
        if has_tsonic:
            phi_t, p_t = self._fit_ar_model(t)  # T_SONIC AR
            self._ar_orders['tsonic'] = p_t

        # ---- Step 3: filtered series ----
        # Scalar AR applied to scalar (x1), W (z1), and T_SONIC (y1)
        s_fa = self._apply_ar_filter(phi_s, s)  # scalar filt by scalar AR  (R: x1)
        w_fa = self._apply_ar_filter(phi_s, w)  # W filt by scalar AR       (R: z1)
        # W AR applied to scalar (x3) and W (z3)
        s_fw = self._apply_ar_filter(phi_w, s)  # scalar filt by W AR       (R: x3)
        w_fw = self._apply_ar_filter(phi_w, w)  # W filt by W AR            (R: z3)
        if has_tsonic:
            t_fa = self._apply_ar_filter(phi_s, t)  # T_SONIC filt by scalar AR (R: y1)
            s_ft = self._apply_ar_filter(phi_t, s)  # scalar filt by T_SONIC AR (R: x2)
            t_ft = self._apply_ar_filter(phi_t, t)  # T_SONIC filt by T_SONIC AR(R: y2)

        # ---- Step 4: raw cross-covariance (diagnostic panel 2) ----
        self._raw_ccov = self._compute_ccov(w, s)
        self._smooth_raw_ccov = self._smooth_series(self._raw_ccov, _SMOOTH_WIDTH_CCOV)

        # ---- Step 5: full-data PW CCF, scalar AR (diagnostic panel 1) ----
        # R: ccf_pww = ccf(x1, z1)  ->  scalar AR filter, scalar x W pair
        self._pw_ccf = self._compute_ccf(w_fa, s_fa)
        self._smooth_pw_ccf = self._smooth_series(self._pw_ccf, _SMOOTH_WIDTH_CCF)
        # R uses length(x1) including leading NaN for the Bartlett denominator
        self._n_eff = len(s_fa)

        self._tlag_pw_records, self._tlag_opt_records, \
            self._tlag_lmax, self._tlag_lmin, self._corr_est = \
            self._compute_tlag_opt()

        # ---- Step 6: block-bootstrap each combination ----
        # x argument = leading signal (W or T_SONIC), y = scalar (delayed by tube)
        combinations = {
            'cw': self._run_combination_bootstrap(w_fa, s_fa),  # scalar x W, scalar AR
            'wc': self._run_combination_bootstrap(w_fw, s_fw),  # scalar x W, W AR
        }
        if has_tsonic:
            combinations['ct'] = self._run_combination_bootstrap(t_fa, s_fa)  # scalar x T_SONIC, scalar AR
            combinations['tc'] = self._run_combination_bootstrap(t_ft, s_ft)  # scalar x T_SONIC, T_SONIC AR

        # ---- Step 7: select the winning combination ----
        best_key = self._select_best_combination(combinations)
        self._best_combination = best_key
        best = combinations[best_key]

        # ---- Step 8: summarise from the winning combination ----
        self._bootstrap_lags = best['lags']
        tlag_mode, hdi_lo, hdi_hi = self._mode_and_hdi(best['lags'])
        self._tlag_records = int(tlag_mode)
        self._hdi_lo_s = hdi_lo
        self._hdi_hi_s = hdi_hi

    # ------------------------------------------------------------------
    # Private: pre-whitening  (AR fitting and filtering)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stationary(x: np.ndarray, alpha: float = 0.01) -> bool:
        """
        ADF unit-root test: return True if the series is stationary (p < alpha).

        Matches the direction of R's egcm::bvr.test used in tlag_detection.R:
        H0 = unit root (non-stationary); p < alpha rejects H0, confirming
        stationarity.  maxlag=1 is sufficient to discriminate turbulent EC series
        (p ~ 0) from non-stationary series (p ~ 0.5) and costs only ~4 ms for
        N = 36000 vs ~1500 ms with autolag='AIC'.
        """
        return adfuller(x, maxlag=1, autolag=None)[1] < alpha

    def _fit_ar_model(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Fit an AR(p) model to x and return the AR coefficients and selected order.

        The order p is selected by AIC from candidates 1..max_order, where
        max_order = floor(100 * log10(N)) matches R's formula
        (order.max = floor(10^2 * log10(length(x)))).  For a 30-minute 20 Hz
        record (N = 36000) this gives max_order = 455.  The large upper bound
        allows AIC to pick up the long-range autocorrelation structures found in
        turbulent wind and trace-gas series; capping at a smaller value (e.g. 45)
        forces AIC to under-fit and leaves residual autocorrelation that distorts
        the CCF peak.

        Levinson-Durbin recursion computes all orders 1..max_order in O(max_order^2)
        after a single FFT-based autocorrelation pass, so the large max_order is
        numerically feasible.

        Returns:
            phi (np.ndarray): AR coefficients for the selected order p.
            p (int): Selected AR order (0 if the series is already white noise).
        """
        x_clean = x[~np.isnan(x)]
        x_centered = x_clean - x_clean.mean()
        n = len(x_centered)

        # R: order.max = floor(10^2 * log10(length(x)))
        max_lag = int(np.floor(100 * np.log10(n)))

        # Biased autocorrelation via FFT
        nfft = 1 << (2 * n).bit_length()
        xf = np.fft.rfft(x_centered, n=nfft)
        acf = np.fft.irfft(xf * np.conj(xf), n=nfft)[:max_lag + 1].real / n

        if acf[0] <= 0:
            return np.array([], dtype=float), 0

        best_aic = n * np.log(acf[0])
        best_phi = np.array([], dtype=float)
        best_p = 0

        phi = np.array([acf[1] / acf[0]], dtype=float)
        sigma2 = acf[0] * (1.0 - phi[0] ** 2)

        if sigma2 > 0:
            aic = n * np.log(sigma2) + 2.0 * 1
            if aic < best_aic:
                best_aic, best_phi, best_p = aic, phi.copy(), 1

        for p in range(2, max_lag + 1):
            if sigma2 <= 0:
                break
            kappa = (acf[p] - phi @ acf[1:p][::-1]) / sigma2
            phi = np.append(phi - kappa * phi[::-1], kappa)
            sigma2 *= 1.0 - kappa ** 2
            if sigma2 > 0:
                aic = n * np.log(sigma2) + 2.0 * p
                if aic < best_aic:
                    best_aic, best_phi, best_p = aic, phi.copy(), p

        return best_phi, best_p

    def _apply_ar_filter(self, phi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Apply AR(p) filter to x to produce the pre-whitened series x_tilde.

        Matches R: stats::filter(x, filter=c(1,-phi), method="convolution", sides=1).
        The first p output values are set to NaN because lfilter seeds its delay
        line with zeros (not actual past data); R's stats::filter also marks them NA.
        These NaN values are removed by na.omit inside _compute_ccf (R line: ccf(...,
        na.action=na.omit)) or zeroed out before block-bootstrap sampling.
        """
        x_centered = x - np.nanmean(x)
        if len(phi) == 0:
            return x_centered

        b = np.concatenate([[1.0], -phi])
        x_filled = np.where(np.isnan(x_centered), 0.0, x_centered)
        x_tilde = lfilter(b, [1.0], x_filled)
        x_tilde[:len(phi)] = np.nan
        return x_tilde

    # ------------------------------------------------------------------
    # Private: tlag_pw / tlag_opt decision logic
    # ------------------------------------------------------------------

    def _compute_tlag_opt(self) -> tuple[int, int, list, list, float]:
        """
        Detect tlag_pw and refine to tlag_opt from the full-data PW CCF.

        tlag_pw is the argmax of the smoothed PW CCF (scalar AR filter).
        The four sequential if-rules search local extrema of the raw CCov near
        the PW peak; Rule 4 always fires when any extremum exists, so in
        practice tlag_opt == tlag_pw.  The local extrema are retained in the
        results dict for diagnostic use.
        """
        tl0 = int(np.nanargmax(np.abs(self._smooth_pw_ccf)))
        tlag_pw = tl0 - self._lag_max_records

        win_start = max(0, tl0 - 12)
        win_end = min(tl0 + 13, 2 * self._lag_max_records)
        window = self._smooth_raw_ccov[win_start:win_end]

        offset = win_start - self._lag_max_records
        tlag_lmax = self._local_extrema(window, 'max', offset)
        tlag_lmin = self._local_extrema(window, 'min', offset)

        tlag_opt = tlag_pw
        if len(tlag_lmax) == 0 and len(tlag_lmin) == 0:
            tlag_opt = tlag_pw
        if len(tlag_lmax) == 1 and len(tlag_lmin) == 0:
            tlag_opt = tlag_lmax[0]
        if len(tlag_lmax) == 0 and len(tlag_lmin) == 1:
            tlag_opt = tlag_lmin[0]
        if len(tlag_lmax) >= 1 or len(tlag_lmin) >= 1:
            tlag_opt = tlag_pw

        opt_idx = tlag_opt + self._lag_max_records
        corr_est = float(self._pw_ccf[opt_idx]) if 0 <= opt_idx < len(self._pw_ccf) else 0.0

        return tlag_pw, tlag_opt, tlag_lmax, tlag_lmin, corr_est

    @staticmethod
    def _local_extrema(arr: np.ndarray, kind: str, offset: int) -> list[int]:
        """
        Find local maxima or minima in arr and return their lag values.

        Matches R tlag_detection.R helper functions local_max / local_min:
            local_max(x) = which(x[i] > x[i-1] AND x[i] > x[i+1])
        """
        mid = arr[1:-1]
        if kind == 'max':
            mask = (mid > arr[:-2]) & (mid > arr[2:])
        else:
            mask = (mid < arr[:-2]) & (mid < arr[2:])
        return (np.where(mask)[0] + 1 + offset).tolist()

    # ------------------------------------------------------------------
    # Private: combination bootstrap (RFlux v3.2.0 multi-combination logic)
    # ------------------------------------------------------------------

    def _run_combination_bootstrap(
            self, x_pw: np.ndarray, y_pw: np.ndarray
    ) -> dict:
        """
        Run block bootstrap for one pre-whitened (x, y) pair and summarise.

        x_pw is the pre-whitened leading series -- either W or T_SONIC after AR
        filtering.  y_pw is the pre-whitened scalar (tube-delayed) after the same
        AR filtering.  The naming convention for each combination is:

            cw: x_pw = W filtered by scalar AR,       y_pw = scalar filtered by scalar AR
            wc: x_pw = W filtered by W AR,             y_pw = scalar filtered by W AR
            ct: x_pw = T_SONIC filtered by scalar AR,  y_pw = scalar filtered by scalar AR
            tc: x_pw = T_SONIC filtered by T_SONIC AR, y_pw = scalar filtered by T_SONIC AR

        Leading NaN left by the AR filter (order-p initialisation artifact) are
        zeroed so that block 0 -- which contains those positions -- is harmless
        when sampled; it is also excluded from the bootstrap draw.

        Returns a dict with:
            lags           -- shape (N_B,) peak lags per bootstrap sample, in records
            mode_lag       -- most frequent peak lag across all N_B samples, in records
            mean_smooth_ccf-- mean bootstrap CCF smoothed with na.locf, used by
                              _select_best_combination (R: ccfs_cw, ccfs_ct, etc.)
        """
        x0 = np.where(np.isnan(x_pw), 0.0, x_pw)
        y0 = np.where(np.isnan(y_pw), 0.0, y_pw)
        boot_lags, mean_smooth_ccf = self._block_bootstrap(x0, y0)
        mode_lag = self._map_estimate(boot_lags)  # KDE MAP, matching R's bayestestR::map_estimate
        return {'lags': boot_lags, 'mode_lag': mode_lag, 'mean_smooth_ccf': mean_smooth_ccf}

    @staticmethod
    def _select_best_combination(combinations: dict) -> str:
        """
        Return the key of the combination with the highest |avg_smooth_ccf| at its mode lag.

        The selection criterion is the absolute value of the mean bootstrap CCF
        (smoothed) evaluated at the mode lag detected by that combination.  A
        higher value indicates that the cross-correlation peak is more prominent
        relative to background noise, making that combination the most informative
        signal for this averaging period.  For strong-flux scalars (CO2) the
        scalar x W combinations (cw/wc) typically win; for weak-flux trace gases
        (N2O, CH4) the T_SONIC combinations (ct/tc) often show a cleaner peak.

        Mirrors R tlag_detection.R lines 151-158:
            corr_est_s <- c(ccfs_ct[maps[1]], ccfs_cw[maps[2]], ...)
            corr_ind   <- which.max(abs(corr_est_s))
        """
        best_corr = -np.inf
        best_key = None
        for key, combo in combinations.items():
            mode_lag = combo['mode_lag']
            avg_ccf = combo['mean_smooth_ccf']
            lag_max = (len(avg_ccf) - 1) // 2
            idx = mode_lag + lag_max
            if 0 <= idx < len(avg_ccf) and not np.isnan(avg_ccf[idx]):
                corr = abs(avg_ccf[idx])
                if corr > best_corr:
                    best_corr = corr
                    best_key = key
        return best_key if best_key is not None else next(iter(combinations))

    def _block_bootstrap(
            self, x_pw: np.ndarray, y_pw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw N_B block-bootstrap samples and return peak lags + mean smooth CCF.

        Procedure (paper Section 2.2 + RFlux v3.2.0):
        1. Divide into n_blocks non-overlapping blocks of length L.
        2. Draw N_B sets of block indices (block 0 excluded — AR init artifact).
        3. Assemble N_B bootstrap series via fancy indexing.
        4. Batch FFT CCF for all N_B samples simultaneously.
        5. Compute mean CCF across N_B samples, smooth with width=hz/2+1,
           apply na.locf (R: ccfs_cw = rollapply(mean_ccf, width=wdt) + na.locf).
        6. Per-replicate: smooth each CCF with the same width, apply na.locf
           to fill NaN edges (R: rollapply + na.locf per replicate in bootccf_*).
        7. Find the peak lag per replicate via argmax(abs(smooth_ccf)).

        Returns:
            boot_lags       -- shape (N_B,) peak lags in records
            mean_smooth_ccf -- shape (2*lag_max+1,) mean bootstrap CCF, smoothed
        """
        n = len(x_pw)
        L = self._block_length_records
        n_blocks = n // L
        N_B = self.n_bootstrap

        valid_blocks = np.arange(1, n_blocks) if n_blocks > 1 else np.arange(n_blocks)

        x_blocks = x_pw[:n_blocks * L].reshape(n_blocks, L)
        y_blocks = y_pw[:n_blocks * L].reshape(n_blocks, L)
        all_chosen = np.random.choice(valid_blocks, size=(N_B, n_blocks), replace=True)
        x_boot = x_blocks[all_chosen].reshape(N_B, -1)
        y_boot = y_blocks[all_chosen].reshape(N_B, -1)

        all_ccf = self._batch_ccf_fft(x_boot, y_boot)

        # Mean CCF across all N_B samples, smoothed then na.locf to fill edge NaN.
        # Used by _select_best_combination to compare combinations (R: ccfs_cw etc.).
        mean_smooth_ccf = _na_locf_1d(
            self._smooth_series(np.mean(all_ccf, axis=0), self._smooth_width_bootstrap)
        )

        # Per-replicate: smooth each CCF, fill edge NaN with na.locf so that lags
        # at the boundary of the search window are not excluded from the argmax search
        # (without na.locf, nanargmax would skip those positions and could miss an
        # edge-lag peak -- R applies zoo::na.locf per replicate for the same reason).
        all_smooth = self._na_locf_rows(
            self._smooth_rows(all_ccf, self._smooth_width_bootstrap)
        )
        peak_indices = np.nanargmax(np.abs(all_smooth), axis=1)
        boot_lags = peak_indices.astype(int) - self._lag_max_records

        return boot_lags, mean_smooth_ccf

    def _batch_ccf_fft(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Batch FFT-based normalised cross-correlation for all N_B bootstrap samples.

        Args:
            X: Leading-signal bootstrap arrays, shape (N_B, N). Zero-filled, no NaN.
            Y: Scalar bootstrap arrays, shape (N_B, N). Same.

        Returns:
            Normalised CCF windows of shape (N_B, 2*lag_max+1), ordered
            [lag=-lag_max, ..., lag=0, ..., lag=+lag_max]. Positive lag = scalar
            delayed (tube delay).
        """
        N_B, N = X.shape
        lag_max = self._lag_max_records

        X_c = X - X.mean(axis=1, keepdims=True)
        Y_c = Y - Y.mean(axis=1, keepdims=True)

        n_fft = 1 << (2 * N - 2).bit_length()

        FX = np.fft.rfft(X_c, n=n_fft, axis=1)
        FY = np.fft.rfft(Y_c, n=n_fft, axis=1)

        ccf_full = np.fft.irfft(FY * np.conj(FX), n=n_fft, axis=1)

        norm = np.sqrt((X_c ** 2).sum(axis=1) * (Y_c ** 2).sum(axis=1))
        norm = np.where(norm == 0.0, 1.0, norm)
        ccf_full /= norm[:, np.newaxis]

        neg = ccf_full[:, n_fft - lag_max:]
        pos = ccf_full[:, :lag_max + 1]
        return np.concatenate([neg, pos], axis=1)

    # ------------------------------------------------------------------
    # Private: cross-correlation and cross-covariance
    # ------------------------------------------------------------------

    def _compute_ccf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Normalised cross-correlation for lags [-lag_max, +lag_max].

        Sign convention: positive lag = scalar (y) arrives later than x (tube delay).
        Leading NaN from AR filtering are trimmed together (R: na.action=na.omit).
        """
        is_nan = np.isnan(x) | np.isnan(y)
        if np.any(is_nan):
            first_valid = int(np.argmax(~is_nan))
            x = x[first_valid:]
            y = y[first_valid:]

        x_c = np.where(np.isnan(x - np.nanmean(x)), 0.0, x - np.nanmean(x))
        y_c = np.where(np.isnan(y - np.nanmean(y)), 0.0, y - np.nanmean(y))

        full = _signal_correlate(y_c, x_c, mode='full', method='fft')

        denom = np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2))
        if denom == 0:
            full[:] = 0.0
        else:
            full /= denom

        mid = len(full) // 2
        return full[mid - self._lag_max_records: mid + self._lag_max_records + 1]

    def _compute_ccov(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Un-normalised cross-covariance for lags [-lag_max, +lag_max].

        Computed on the original (not pre-whitened) data. Used in
        _compute_tlag_opt to search for local extrema near the PW peak.
        Biased estimator (divide by N), matching R's ccf(type="covariance").
        """
        x_c = np.where(np.isnan(x), 0.0, x - np.nanmean(x))
        y_c = np.where(np.isnan(y), 0.0, y - np.nanmean(y))

        full = _signal_correlate(y_c, x_c, mode='full', method='fft')
        n = max(len(x_c), len(y_c))
        full /= n

        mid = len(full) // 2
        return full[mid - self._lag_max_records: mid + self._lag_max_records + 1]

    @staticmethod
    def _smooth_series(series: np.ndarray, width: int) -> np.ndarray:
        """
        Centered rolling mean of given width with NaN at edges (min_periods=width).
        Matches R: rollapply(series, width=width, FUN="mean", fill=NA).
        """
        M = len(series)
        half = width // 2
        cs = np.empty(M + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(series, out=cs[1:])
        result = np.full(M, np.nan, dtype=np.float64)
        result[half:M - half] = (cs[width:] - cs[:M - width + 1]) / width
        return result

    @staticmethod
    def _smooth_rows(arr: np.ndarray, width: int) -> np.ndarray:
        """
        Vectorised centered rolling mean for every row of a 2-D array.
        NaN at the first and last half positions (min_periods=width behaviour).
        """
        if width <= 1:
            return arr.copy()

        N_B, M = arr.shape
        half = width // 2

        cs = np.empty((N_B, M + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(arr, axis=1, out=cs[:, 1:])

        result = np.full((N_B, M), np.nan, dtype=np.float64)
        result[:, half:M - half] = (cs[:, width:] - cs[:, :M - width + 1]) / width
        return result

    @staticmethod
    def _na_locf_rows(arr: np.ndarray) -> np.ndarray:
        """
        Apply forward-fill then backward-fill NaN to every row of a 2-D array.

        The centered rolling mean used for smoothing leaves NaN at the first and
        last (width//2) positions of each row.  If those positions correspond to
        the search-window edges (+/- lag_max), a true lag peak there would be
        invisible to nanargmax.  Filling the edges with the nearest non-NaN value
        ensures every lag position is a candidate -- matching R's two-pass
        zoo::na.locf applied per bootstrap replicate.

        Matches R: zoo::na.locf(zoo::na.locf(..., na.rm=FALSE), fromLast=TRUE).

        Implementation uses vectorised numpy accumulate over all N_B rows
        simultaneously instead of a Python for loop.
        """
        N_B, M = arr.shape
        col = np.arange(M)  # (M,) broadcasts to (N_B, M)
        row = np.arange(N_B)[:, np.newaxis]  # (N_B, 1) for 2-D fancy indexing

        out = arr.copy()

        # Forward fill: carry last valid value rightward (fills right-edge NaN).
        # idx[i, j] = j when out[i, j] is valid, else 0; after maximum.accumulate
        # idx[i, j] holds the column of the last valid value at or before j.
        nan_mask = np.isnan(out)
        idx = np.where(~nan_mask, col, 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = np.where(nan_mask, out[row, idx], out)

        # Backward fill: carry next valid value leftward (fills left-edge NaN).
        # Achieved by running maximum.accumulate on the column-reversed array.
        nan_mask = np.isnan(out)
        if nan_mask.any():
            idx = np.where(~nan_mask, col, M - 1)
            np.minimum.accumulate(idx[:, ::-1], axis=1, out=idx[:, ::-1])
            out = np.where(nan_mask, out[row, idx], out)

        return out

    # ------------------------------------------------------------------
    # Private: mode and HDI  (paper Section 2.2, eqs. 6-7)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_estimate(samples: np.ndarray) -> int:
        """
        MAP (mode) estimate via KDE with tiny jitter, matching R's
        bayestestR::map_estimate used in tlag_detection.R line 146.

        R adds small Gaussian noise to discrete integer lag indices to break
        exact ties before fitting a kernel density, then rounds the KDE peak
        to the nearest integer.  For a unimodal distribution -- which arises
        whenever the lag signal is clear -- this is identical to
        scipy.stats.mode.  The two differ only when two adjacent integer lags
        have exactly equal counts (bimodal tie); in that case the KDE
        interpolates between them while mode picks the lower value.  That
        edge case coincides with a wide HDI (unreliable result) where the
        exact mode value is inconsequential.

        Args:
            samples: Bootstrap lag distribution in integer records.

        Returns:
            MAP estimate rounded to the nearest integer record.
        """
        if len(np.unique(samples)) == 1:
            # All samples identical: gaussian_kde would have zero bandwidth.
            return int(samples[0])
        jittered = samples.astype(float) + np.random.normal(0, 0.0001, len(samples))
        kde = gaussian_kde(jittered)
        x_grid = np.linspace(jittered.min(), jittered.max(), 512)
        return int(round(float(x_grid[np.argmax(kde(x_grid))])))

    def _mode_and_hdi(self, lags_records: np.ndarray) -> tuple[int, float, float]:
        """
        Summarise the N_B bootstrap lag distribution (paper eq. 7).

        The mode (MAP estimate via KDE) is the most probable lag across the N_B
        bootstrap replicates.  The 95% HDI is the shortest interval containing
        95% of the distribution; its width is the reliability metric (Section 2.3).
        """
        tlag_mode = self._map_estimate(lags_records)
        lags_s = lags_records / self.hz
        hdi_lo, hdi_hi = self._hdi(lags_s, credible_mass=0.95)
        return tlag_mode, float(hdi_lo), float(hdi_hi)

    @staticmethod
    def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> tuple[float, float]:
        """
        Highest Density Interval: shortest interval containing credible_mass of samples.
        """
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        n_included = int(np.floor(credible_mass * n))
        n_intervals = n - n_included
        if n_intervals <= 0:
            return float(sorted_s[0]), float(sorted_s[-1])
        widths = sorted_s[n_included:] - sorted_s[:n_intervals]
        min_idx = int(np.argmin(widths))
        return float(sorted_s[min_idx]), float(sorted_s[min_idx + n_included])

    # ------------------------------------------------------------------
    # Plotting  (Phase 2 -- styling only, no computation)
    # ------------------------------------------------------------------

    def plot(
            self,
            ax: plt.Axes = None,
            title: str = None,
            showplot: bool = True,
            outpath: str = None,
            outname: str = None,
    ) -> Figure:
        """
        Three-panel diagnostic figure.

        Panel 1 (left) -- Pre-whitened CCF (scalar AR, full data):
            Grey stems + smoothed black line (width=13) + Bartlett significance
            bands + red tlag_opt marker + significance annotation.

        Panel 2 (middle) -- Raw cross-covariance:
            Grey stems + smoothed cyan line (width=3) + red tlag_opt marker.

        Panel 3 (right) -- Bootstrap lag distribution (best combination):
            Histogram of N_B detected lags + 95% HDI shading + mode marker.
            Title includes the selected combination name.

        Args:
            ax: Not used (three-panel figure always created). Kept for API
                consistency with the diive plotting pattern.
            title (str): Optional figure suptitle.
            showplot (bool): If True, display the figure. Defaults to True.
            outpath (str): Directory path for saving. Optional.
            outname (str): File name (with extension) for saving. Optional.

        Returns:
            Figure: The matplotlib figure.
        """
        if self._tlag_records is None:
            raise RuntimeError("Call run() before plot().")

        fig = plt.figure(facecolor='white', figsize=(18, 6), layout='constrained')
        gs = gridspec.GridSpec(1, 3, figure=fig)
        ax_ccf = fig.add_subplot(gs[0, 0])
        ax_ccov = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[0, 2])

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')

        self._plot_pw_ccf(ax_ccf)
        self._plot_raw_ccov(ax_ccov)
        self._plot_bootstrap_histogram(ax_hist)

        if outpath and outname:
            fig.savefig(f"{outpath}/{outname}", format='png', bbox_inches='tight',
                        facecolor='w', transparent=True, dpi=100)
            plt.close(fig)

        if showplot:
            fig.show()

        return fig

    def _plot_pw_ccf(self, ax: plt.Axes):
        """Panel 1: pre-whitened CCF (scalar AR, full data)."""
        lags_s = self._lags_axis / self.hz
        cv5 = self.cv5pct
        cv1 = self.cv1pct

        ax.vlines(lags_s, 0, self._pw_ccf,
                  colors='#808080', linewidth=0.6, alpha=0.7)
        ax.plot(lags_s, self._smooth_pw_ccf,
                color='black', linewidth=2, label='smoothed CCF')

        ax.axhline(cv5, color='steelblue', linestyle=':', linewidth=1,
                   label='+/-1.96/sqrt(n)  (5%)')
        ax.axhline(-cv5, color='steelblue', linestyle=':', linewidth=1)
        ax.axhline(cv1, color='steelblue', linestyle='--', linewidth=1,
                   label='+/-2.57/sqrt(n)  (1%)')
        ax.axhline(-cv1, color='steelblue', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(self.tlag_opt_s, color='red', linestyle=':', linewidth=1.5)

        abs_est = abs(self._corr_est)
        if abs_est >= cv1:
            sig_txt = (f'Detected peak at {self.tlag_opt_s:.3f} s'
                       f'  stat. sign. at 0.01 level')
        elif abs_est >= cv5:
            sig_txt = (f'Detected peak at {self.tlag_opt_s:.3f} s'
                       f'  stat. sign. at 0.05 level')
        else:
            sig_txt = 'Detected peak not stat. significant'

        ax.set_title(sig_txt, fontsize=9)
        default_format(ax=ax, ax_labels_fontsize=theme.AX_LABELS_FONTSIZE,
                       ax_xlabel_txt='lag', ax_ylabel_txt='cross-correlation',
                       txt_ylabel_units='s')
        ax.legend(frameon=False, fontsize=8)

    def _plot_raw_ccov(self, ax: plt.Axes):
        """Panel 2: raw cross-covariance."""
        lags_s = self._lags_axis / self.hz

        ax.vlines(lags_s, 0, self._raw_ccov,
                  colors='#808080', linewidth=0.6, alpha=0.7)
        ax.plot(lags_s, self._smooth_raw_ccov,
                color='cyan', linewidth=1.5, label='smoothed cross-cov')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(self.tlag_opt_s, color='red', linestyle=':', linewidth=1.5,
                   label=f'tlag_opt = {self.tlag_opt_s:.2f} s')

        ax.set_title(f'Optimal time lag at {self.tlag_opt_s:.2f} s', fontsize=9)
        default_format(ax=ax, ax_labels_fontsize=theme.AX_LABELS_FONTSIZE,
                       ax_xlabel_txt='lag', ax_ylabel_txt='cross-covariance',
                       txt_ylabel_units='s')
        ax.legend(frameon=False, fontsize=8)

    def _plot_bootstrap_histogram(self, ax: plt.Axes):
        """Panel 3: bootstrap lag distribution from the selected combination."""
        lags_s = self._bootstrap_lags / self.hz
        n_bins = max(10, self.n_bootstrap // 5)

        ax.hist(lags_s, bins=n_bins, color='#808080', alpha=0.7,
                edgecolor='none', label='bootstrap lags')
        ax.axvspan(self._hdi_lo_s, self._hdi_hi_s, alpha=0.35, color='steelblue',
                   label=f'95% HDI  [{self._hdi_lo_s:.2f}, {self._hdi_hi_s:.2f}] s')
        ax.axvline(self.tlag_s, color='red', linewidth=2,
                   label=f'mode = {self.tlag_s:.2f} s')

        reliability = 'reliable' if self.is_reliable else 'UNRELIABLE'
        combo_label = f'combo={self._best_combination}' if self._best_combination else ''
        ax.set_title(
            f'{self.segment_name} | {reliability} '
            f'(HDI={self.hdi_range_s:.2f} s, {combo_label})',
            fontsize=9
        )
        default_format(ax=ax, ax_labels_fontsize=theme.AX_LABELS_FONTSIZE,
                       ax_xlabel_txt='time lag', ax_ylabel_txt='count',
                       txt_ylabel_units='s')
        ax.legend(frameon=False, fontsize=8)


class PwboptLagPlot:
    """
    Scatter + KDE comparison figure for two PWBOPT lag time series.

    Reproduces the publication-style figure from the Vitale et al. (2024)
    analysis workflow.  For each gas, a scatter of optimal lags over time
    (x-axis: period index or timestamps) sits alongside a normalized KDE
    sidebar, enabling simultaneous inspection of temporal variability and the
    overall lag distribution.  Two methods or strategies are placed side by
    side in each row for direct comparison.

    Typical use cases:

    - **Strategy comparison**: standard PWBOPT (left) vs pre-filtered PWBOPT
      (right) -- using columns produced by ``flux_lag_pwbopt.py``.
    - **Method comparison**: CM covariance-max lags (left) vs PWB lags (right).

    Layout (2 rows x 5 cols via GridSpec, width_ratios=[5,1,0.25,5,1])::

        (a) scatter_A | kde_A | gap | (b) scatter_B | kde_B   <- gas 1
        (c) scatter_A | kde_A | gap | (d) scatter_B | kde_B   <- gas 2

    Follows the diive two-phase plotting pattern:

    - **Phase 1** (``__init__``): store data and parameters; pre-compute
      x-axis values.  No computation that depends on styling.
    - **Phase 2** (``plot``): all layout, KDE computation, and styling.
      Can be called multiple times with different arguments.

    Parameters
    ----------
    results : DataFrame
        Per-period results table, one row per averaging period.  Must contain
        the lag columns listed in ``scalars``.
    scalars : dict
        ``{gas_label: {'col_a': col_name_A, 'col_b': col_name_B}}``.
        ``col_a`` is plotted on the left pair of panels; ``col_b`` on the right.
        Supports 1 or 2 gases (e.g. ``{'CH4': {...}, 'N2O': {...}}``).
    label_a : str
        Legend label for method or strategy A (left panels).
    label_b : str
        Legend label for method or strategy B (right panels).
    color_a : str
        Scatter and KDE color for A.  Defaults to Wong blue ``'#0072B2'``.
    color_b : str
        Scatter and KDE color for B.  Defaults to coral ``'#E05C2A'``.
    timestamp_col : str, optional
        Column in ``results`` containing datetime-like values used as the
        x-axis.  If ``None`` or not present, a numeric period index is used.
    jitter : float
        Half-width of uniform random jitter applied to lag values so that
        overlapping points are visible.  Defaults to 0.06 s.
    ylim : tuple of float, optional
        Shared y-axis limits ``(lo, hi)`` for all scatter panels.  If
        ``None``, Matplotlib auto-scales.

    See Also
    --------
    PreWhiteningBootstrap : Single-period PWB detection producing the
        ``tlag_s`` and ``hdi_range_s`` columns fed into this figure.

    Example
    -------
    See ``examples/flux/hires/flux_lag_pwbopt.py`` for a complete example.
    """

    def __init__(
            self,
            results: DataFrame,
            scalars: dict,
            label_a: str,
            label_b: str,
            color_a: str = '#0072B2',
            color_b: str = '#E05C2A',
            timestamp_col: str = None,
            jitter: float = 0.06,
            ylim: tuple = None,
    ):
        """
        Store data and parameters.  No layout or computation is performed here.

        Args:
            results (DataFrame): Per-period results table.
            scalars (dict): ``{gas_label: {'col_a': str, 'col_b': str}}``.
            label_a (str): Legend label for method / strategy A.
            label_b (str): Legend label for method / strategy B.
            color_a (str): Color for A panels. Defaults to ``'#0072B2'``.
            color_b (str): Color for B panels. Defaults to ``'#E05C2A'``.
            timestamp_col (str, optional): Datetime column for the x-axis.
                Falls back to integer period index when not provided.
            jitter (float): Lag jitter half-width in seconds. Defaults to 0.06.
            ylim (tuple, optional): Shared y-axis limits ``(lo, hi)``.
        """
        self.results = results
        self.scalars = scalars
        self.label_a = label_a
        self.label_b = label_b
        self.color_a = color_a
        self.color_b = color_b
        self.timestamp_col = timestamp_col
        self.jitter = jitter
        self.ylim = ylim

        # Pre-compute x-axis values used in every panel
        if timestamp_col and timestamp_col in results.columns:
            self._xvals = pd.to_datetime(results[timestamp_col]).values
            self._use_dates = True
        else:
            self._xvals = np.arange(len(results))
            self._use_dates = False

    # ------------------------------------------------------------------
    # Public interface (Phase 2)
    # ------------------------------------------------------------------

    def plot(
            self,
            title: str = None,
            events: list = None,
            showplot: bool = True,
            outpath: str = None,
            outname: str = None,
    ) -> Figure:
        """
        Draw the scatter + KDE comparison figure.

        Args:
            title (str, optional): Figure-level suptitle.
            events (list of dict, optional): Site events drawn as vertical
                lines on the scatter panels.  Each dict must have keys
                ``'date'`` (str, parseable by ``pd.Timestamp``), ``'label'``
                (str), ``'ls'`` (linestyle str), ``'color'`` (str), and
                ``'side'`` (``'left'`` or ``'right'``).  Events are only
                drawn when ``timestamp_col`` was supplied.  Example::

                    events = [
                        {'date': '2021-08-20', 'label': 'Tillage',
                         'ls': '-', 'color': 'black', 'side': 'right'},
                    ]

            showplot (bool): Display the figure interactively. Defaults to True.
            outpath (str, optional): Directory for saving the figure.
            outname (str, optional): File name (with extension) for saving.

        Returns:
            Figure: The matplotlib figure.
        """
        gas_labels = list(self.scalars.keys())
        n_rows = len(gas_labels)
        # Extra row added at the bottom when exactly 2 gases are present:
        # left panel = overlay of both gases over time; right = cross-scatter.
        _has_gas_comparison = (n_rows == 2)
        n_rows_fig = n_rows + (1 if _has_gas_comparison else 0)
        rng = np.random.default_rng(42)
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        panel_count = 0

        # Row height and inter-row spacing.  With a comparison row present the
        # last gas row is no longer the visual bottom, so x-axis labels shift
        # down to the comparison row.  The extra hspace prevents the last gas
        # row's tick labels from colliding with the comparison row's titles.
        fig_h = n_rows_fig * 8.0 / 2.54  # 8 cm per row (was 7 cm)
        _hspace = 0.42 if _has_gas_comparison else 0.25
        fig = plt.figure(figsize=(28 / 2.54, fig_h), facecolor='white')

        if title:
            fig.suptitle(title, fontsize=11, fontweight='bold', y=1.00)

        gs = gridspec.GridSpec(
            n_rows_fig, 5, figure=fig,
            hspace=_hspace, wspace=0.06,
            left=0.06, right=0.98,
            top=0.88, bottom=0.08,
            width_ratios=[5, 1, 0.25, 5, 1],
        )

        ax_master = None  # first scatter axis; all others share its x and y

        # Compute effective y limits before building axes so they can be
        # applied once to ax_master (all sharey axes inherit the same range).
        # When self.ylim is given, use it directly.  Otherwise compute a tight
        # range from every plotted column: data extent + 10% padding on each
        # side, with a minimum of 0.5 s headroom so the axis never collapses.
        if self.ylim is not None:
            _effective_ylim = self.ylim
        else:
            _all_vals = []
            for _cols in self.scalars.values():
                for _col in (_cols['col_a'], _cols['col_b']):
                    if _col in self.results.columns:
                        _all_vals.extend(
                            self.results[_col].dropna().tolist())
            if _all_vals:
                _lo = float(min(_all_vals))
                _hi = float(max(_all_vals))
                _pad = max((_hi - _lo) * 0.10, 0.5)
                _effective_ylim = (_lo - _pad, _hi + _pad)
            else:
                _effective_ylim = None

        for row, gas in enumerate(gas_labels):
            cols = self.scalars[gas]
            col_a = cols['col_a']
            col_b = cols['col_b']
            # When a comparison row follows below, the last gas row is not the
            # visual bottom: suppress its x-axis labels so they do not collide
            # with the comparison row's panel titles.
            is_bottom = (row == n_rows - 1) and not _has_gas_comparison
            is_top = (row == 0)

            # Build four axes for this row.  Scatter panels share both x and y
            # with ax_master so that zooming/panning is synchronised across rows
            # and across the left/right scatter columns.
            if ax_master is None:
                ax_s_a = fig.add_subplot(gs[row, 0])
                ax_master = ax_s_a
                # Apply once; all sharey axes inherit the same range
                if _effective_ylim:
                    ax_master.set_ylim(_effective_ylim)
            else:
                ax_s_a = fig.add_subplot(gs[row, 0],
                                         sharex=ax_master, sharey=ax_master)
            ax_k_a = fig.add_subplot(gs[row, 1], sharey=ax_s_a)
            ax_s_b = fig.add_subplot(gs[row, 3],
                                     sharex=ax_master, sharey=ax_master)
            ax_k_b = fig.add_subplot(gs[row, 4], sharey=ax_s_b)

            for side, (ax_s, ax_k, col, color, method_label) in enumerate([
                (ax_s_a, ax_k_a, col_a, self.color_a, self.label_a),
                (ax_s_b, ax_k_b, col_b, self.color_b, self.label_b),
            ]):
                letter = _letters[panel_count]
                panel_count += 1
                show_ylabel = (side == 0)  # y-label only on the left column

                panel_title = f'{gas} — {method_label}'

                # Guard: column not present (e.g. PWBOPT not yet run)
                if col not in self.results.columns:
                    ax_s.text(0.5, 0.5, f'column not found:\n{col}',
                              transform=ax_s.transAxes, ha='center', va='center',
                              color='grey', fontsize=8, style='italic')
                    self._style_scatter(ax_s, show_ylabel=show_ylabel,
                                        show_xlabel=is_bottom,
                                        use_dates=self._use_dates)
                    self._style_kde(ax_k, show_xlabel=is_bottom)
                    continue

                series = self.results[col].dropna()
                if series.empty:
                    self._style_scatter(ax_s, show_ylabel=show_ylabel,
                                        show_xlabel=is_bottom,
                                        use_dates=self._use_dates)
                    self._style_kde(ax_k, show_xlabel=is_bottom)
                    continue

                xs = self._xvals[series.index]

                # Jittered scatter -- small uniform noise so stacked points separate
                jitter_v = rng.uniform(-self.jitter, self.jitter,
                                       size=len(series))
                ax_s.scatter(xs, series.values + jitter_v,
                             marker='o', color=color,
                             s=6, alpha=0.25, linewidths=0, zorder=3)

                # KDE-estimated mode + normalized KDE sidebar
                if len(series) > 5:
                    mode_val, kde_y, kde_d = self._kde_and_mode(series.values)
                    dn = kde_d / kde_d.max()
                    ax_s.axhline(mode_val, color='black',
                                 linewidth=1.2, linestyle='--', zorder=5)
                    ax_k.axhline(mode_val, color='black',
                                 linewidth=1.0, linestyle='--', zorder=5)
                    ax_k.fill_betweenx(kde_y, dn, color=color, alpha=0.15)
                    ax_k.plot(dn, kde_y, color=color, linewidth=1.2)

                # Vertical event lines (only when x-axis contains real dates)
                if self._use_dates and events:
                    self._draw_events(ax_s, events, show_labels=is_top)

                # Panel letter + title in top-left corner, outside the axes
                ax_s.text(0.01, 1.10, f'({letter}) {panel_title}',
                          transform=ax_s.transAxes,
                          fontsize=9, fontweight='bold',
                          va='top', ha='left', color='black')

                self._style_scatter(ax_s, show_ylabel=show_ylabel,
                                    show_xlabel=is_bottom,
                                    use_dates=self._use_dates)
                self._style_kde(ax_k, show_xlabel=is_bottom)

        # Inter-gas comparison row (only when exactly 2 gases are configured)
        if _has_gas_comparison:
            # Overlay (cols 0-1) shares the x-axis with the main scatter panels
            # so period index / timestamps stay synchronised on pan and zoom.
            ax_overlay = fig.add_subplot(gs[n_rows, 0:2],
                                         sharex=ax_master if ax_master else None)
            ax_scatter = fig.add_subplot(gs[n_rows, 3:5])
            self._plot_gas_comparison(ax_overlay, ax_scatter,
                                      gas_labels, _effective_ylim)

        # Shared legend centred above the figure
        handles = [
            plt.scatter([], [], marker='o', color=self.color_a,
                        s=20, linewidths=0, label=self.label_a),
            plt.scatter([], [], marker='o', color=self.color_b,
                        s=20, linewidths=0, label=self.label_b),
            plt.Line2D([0], [0], color='black', linewidth=1.2,
                       linestyle='--', label='KDE mode'),
        ]
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, 1.00), ncol=len(handles),
                   frameon=False, fontsize=9,
                   handletextpad=0.4, borderpad=0.5, columnspacing=1.5)

        if outpath and outname:
            fig.savefig(f'{outpath}/{outname}', dpi=150,
                        bbox_inches='tight', facecolor='white')

        if showplot:
            fig.show()

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kde_and_mode(values: np.ndarray, n: int = 500):
        """
        Gaussian KDE over the value range and the KDE-estimated mode.

        The mode is rounded to the nearest 0.05 s for clean annotation,
        matching the R visualization scripts (plot.py line: round(.../0.05)*0.05).

        Args:
            values: 1-D array of lag values (NaN already removed by caller).
            n: Number of points for the KDE grid. Defaults to 500.

        Returns:
            Tuple (mode_val, kde_y, kde_d) where kde_y is the evaluation grid
            and kde_d is the density array (not normalised; caller divides by max).
        """
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if lo >= hi:
            # Degenerate: all values identical -- return a trivial distribution
            return lo, np.array([lo - 0.5, lo + 0.5]), np.array([0.0, 1.0])
        pad = (hi - lo) * 0.12
        kde_y = np.linspace(lo - pad, hi + pad, n)
        kde_d = gaussian_kde(values, bw_method='scott')(kde_y)
        mode_val = round(float(kde_y[np.argmax(kde_d)]) / 0.05) * 0.05
        return mode_val, kde_y, kde_d

    @staticmethod
    def _draw_events(ax: plt.Axes, events: list, show_labels: bool):
        """
        Draw vertical lines at site-event dates with optional rotated labels.

        Args:
            ax: Scatter axes to annotate.
            events: List of dicts with keys 'date', 'label', 'ls', 'color',
                'side' ('left' or 'right').
            show_labels: When True, draw rotated text label next to each line.
        """
        for ev in events:
            dt = pd.Timestamp(ev['date'])
            ax.axvline(dt, color=ev.get('color', 'black'),
                       linewidth=1.0, linestyle=ev.get('ls', '-'), zorder=6)
            if show_labels:
                side = ev.get('side', 'right')
                offset = (pd.Timedelta(days=1) if side == 'right'
                          else pd.Timedelta(days=-1))
                ax.text(dt + offset, 0.98, ev.get('label', ''),
                        fontsize=8, va='top',
                        ha='left' if side == 'right' else 'right',
                        color=ev.get('color', 'black'),
                        transform=ax.get_xaxis_transform(),
                        rotation=90, clip_on=True)

    @staticmethod
    def _style_scatter(ax: plt.Axes, show_ylabel: bool,
                       show_xlabel: bool, use_dates: bool):
        """Apply shared scatter-panel styling (spines, grid, labels)."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.4, linestyle='--')
        if show_ylabel:
            ax.set_ylabel('Time lag (s)', labelpad=4)
        else:
            ax.tick_params(axis='y', labelleft=False)
        if show_xlabel:
            if use_dates:
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator))
            ax.set_xlabel('Date' if use_dates else 'Period index', labelpad=3)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    @staticmethod
    def _style_kde(ax: plt.Axes, show_xlabel: bool):
        """Apply shared KDE-panel styling (spines, ticks, limits)."""
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1.0]))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(['0', '.5', '1']))
        ax.set_xlabel('KDE', labelpad=3)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.4, linestyle='--')
        ax.grid(axis='x', visible=False)
        if not show_xlabel:
            ax.tick_params(axis='x', labelbottom=False)

    def _plot_gas_comparison(
            self,
            ax_overlay: plt.Axes,
            ax_scatter: plt.Axes,
            gas_labels: list,
            ylim: tuple,
    ):
        """
        Draw two inter-gas comparison panels for the bottom row.

        Both panels compare gas1 vs gas2 using the standard strategy (col_a)
        for the overlay and both strategies for the scatter.

        Left panel -- **Overlay**: gas1 and gas2 optimal lags plotted on the
        same time axis as jittered scatter with KDE mode lines.  Answers
        "do the two gases track each other over time?"

        Right panel -- **Cross-scatter**: gas1 lag on x vs gas2 lag on y with
        a 1:1 identity line.  Both strategies are shown (col_a = color_a,
        col_b = color_b).  Answers "is there a consistent offset between the
        two gases?" and "how tightly correlated are their lags?"

        Args:
            ax_overlay: Axes for the time-series overlay panel.
            ax_scatter: Axes for the cross-scatter panel.
            gas_labels: List of exactly two gas name strings.
            ylim: Effective y-axis limits shared with the main panels, or
                None if auto-scaling is active.
        """
        gas1, gas2 = gas_labels[0], gas_labels[1]
        cols1 = self.scalars[gas1]
        cols2 = self.scalars[gas2]
        rng = np.random.default_rng(42)

        # In the inter-gas panels color_a identifies gas1, color_b identifies
        # gas2 -- consistent with the top rows where these colors anchor each
        # left / right panel pair.
        _c1, _c2 = self.color_a, self.color_b

        # ---- Overlay: gas1 and gas2 on the same time axis (pre-filtered only) ----
        # col_b = pre-filtered PWBOPT -- used exclusively so the overlay shows
        # the cleaner, more conservative lag series for both gases.
        for col, color, gas in [
            (cols1['col_b'], _c1, gas1),
            (cols2['col_b'], _c2, gas2),
        ]:
            if col not in self.results.columns:
                continue
            series = self.results[col].dropna()
            if series.empty:
                continue
            xs = self._xvals[series.index]
            jitter_v = rng.uniform(-self.jitter, self.jitter, size=len(series))
            ax_overlay.scatter(xs, series.values + jitter_v,
                               marker='o', color=color,
                               s=6, alpha=0.25, linewidths=0, zorder=3,
                               label=gas)
            if len(series) > 5:
                mode_val, _, _ = self._kde_and_mode(series.values)
                ax_overlay.axhline(mode_val, color=color,
                                   linewidth=1.2, linestyle='--', zorder=5)

        if ylim:
            ax_overlay.set_ylim(ylim)
        if self._use_dates:
            locator = mdates.AutoDateLocator()
            ax_overlay.xaxis.set_major_locator(locator)
            ax_overlay.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator))
        ax_overlay.set_xlabel('Date' if self._use_dates else 'Period index',
                              labelpad=3)
        ax_overlay.set_ylabel('Time lag (s)', labelpad=4)
        ax_overlay.set_title(f'Gas lag overlay ({self.label_b})', fontsize=9)
        ax_overlay.spines['top'].set_visible(False)
        ax_overlay.spines['right'].set_visible(False)
        ax_overlay.grid(axis='y', color='#DDDDDD', linewidth=0.4, linestyle='--')
        ax_overlay.legend(frameon=False, fontsize=8, loc='upper right')

        # ---- Cross-scatter: gas1 lag vs gas2 lag (pre-filtered only) --------
        # Only col_b (pre-filtered) is shown -- the cleaner series for
        # assessing whether the two gases share the same inlet lag.
        col1, col2 = cols1['col_b'], cols2['col_b']
        if (col1 in self.results.columns and col2 in self.results.columns):
            s1 = self.results[col1]
            s2 = self.results[col2]
            mask = s1.notna() & s2.notna()
            if mask.sum() >= 2:
                ax_scatter.scatter(s1[mask], s2[mask],
                                   marker='o', color=self.color_b,
                                   s=14, alpha=0.4, linewidths=0, zorder=3,
                                   label=self.label_b)

        # 1:1 identity line spanning the pre-filtered data range
        if ylim:
            lo, hi = ylim
        else:
            _all = []
            for _col in (cols1['col_b'], cols2['col_b']):
                if _col in self.results.columns:
                    _all.extend(self.results[_col].dropna().tolist())
            if _all:
                _pad = max((max(_all) - min(_all)) * 0.10, 0.5)
                lo, hi = min(_all) - _pad, max(_all) + _pad
            else:
                lo, hi = 0.0, 5.0

        ax_scatter.plot([lo, hi], [lo, hi],
                        color='black', linewidth=1.0, linestyle='-',
                        alpha=0.35, zorder=2, label='1:1')
        ax_scatter.set_xlim(lo, hi)
        ax_scatter.set_ylim(lo, hi)
        ax_scatter.set_aspect('equal', adjustable='box')
        ax_scatter.set_xlabel(f'{gas1} time lag (s)', labelpad=3)
        ax_scatter.set_ylabel(f'{gas2} time lag (s)', labelpad=3)
        ax_scatter.set_title('Cross-gas lag comparison', fontsize=9)
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)
        ax_scatter.grid(color='#DDDDDD', linewidth=0.4, linestyle='--')
        ax_scatter.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope to be picklable by
# ProcessPoolExecutor (Windows spawn: each worker re-imports this module)
# ---------------------------------------------------------------------------

def _pwb_file_worker(args: tuple) -> dict:
    """Process one averaging-period file and return one result-row dict."""
    (filepath, scalars, col_w, col_tsonic,
     hz, lag_max_s, n_bootstrap, block_length_s,
     usecols, col_names, skiprows, na_values,
     min_valid_frac, plot_dir, save_plots) = args

    period_name = Path(filepath).name
    row: dict = {'period': period_name}

    try:
        df = pd.read_csv(
            filepath,
            skiprows=skiprows + 1,
            header=None,
            sep=r'\s+',
            na_values=na_values,
            low_memory=False,
        )
        df = df.iloc[:, usecols].copy()
        df.columns = col_names[:len(df.columns)]
    except Exception:
        return row

    if len(df) < 25:
        return row

    w_arr = np.asarray(df[col_w], dtype=float)
    if (np.mean(~np.isnan(w_arr)) < min_valid_frac
            or np.nanstd(w_arr) < np.finfo(float).eps):
        return row

    for scalar_label, scalar_col in scalars.items():
        prefix = scalar_label.lower()
        _nan_keys = (
            'tlag_s', 'hdi_lo_s', 'hdi_hi_s',
            'hdi_range_s', 'tlag_pw_s', 'corr_est', 'ar_order',
        )
        nan_row = {f'{prefix}_{k}': np.nan for k in _nan_keys}

        if scalar_col not in df.columns:
            row.update(nan_row)
            continue

        s_arr = np.asarray(df[scalar_col], dtype=float)
        if (np.mean(~np.isnan(s_arr)) < min_valid_frac
                or np.nanstd(s_arr) < np.finfo(float).eps):
            row.update(nan_row)
            continue

        has_ts = col_tsonic is not None and col_tsonic in df.columns
        col_map = {col_w: 'W', scalar_col: scalar_label}
        keep_cols = [col_w, scalar_col]
        if has_ts:
            col_map[col_tsonic] = 'T_SONIC'
            keep_cols.append(col_tsonic)

        try:
            pwb = PreWhiteningBootstrap(
                df=df[keep_cols].rename(columns=col_map),
                var_w='W',
                var_scalar=scalar_label,
                var_tsonic='T_SONIC' if has_ts else None,
                hz=hz,
                lag_max_s=lag_max_s,
                n_bootstrap=n_bootstrap,
                block_length_s=block_length_s,
                segment_name=period_name,
            )
            pwb.run()
            res = pwb.results

            row[f'{prefix}_tlag_s'] = res['tlag_s']
            row[f'{prefix}_hdi_lo_s'] = res['hdi_lo_s']
            row[f'{prefix}_hdi_hi_s'] = res['hdi_hi_s']
            row[f'{prefix}_hdi_range_s'] = res['hdi_range_s']
            row[f'{prefix}_tlag_pw_s'] = res['tlag_pw_s']
            row[f'{prefix}_corr_est'] = res['corr_est']
            row[f'{prefix}_ar_order'] = res['ar_order']

            if save_plots and plot_dir:
                fig = pwb.plot(
                    title=f'{period_name} | {scalar_label}',
                    showplot=False,
                )
                fig.savefig(
                    Path(plot_dir) / f'{Path(period_name).stem}_{prefix}.png',
                    dpi=100, bbox_inches='tight',
                )
                plt.close(fig)

        except Exception:
            row.update(nan_row)

    return row


# ---------------------------------------------------------------------------
# PwbBatchDetection
# ---------------------------------------------------------------------------

class PwbBatchDetection:
    """
    Parallel batch PWB time-lag detection across many averaging-period files.

    .. important::
        Input files must contain **wind-rotation-corrected** data (double
        rotation or planar-fit applied, e.g. EddyPro "Advanced" rotated
        output).  Wind rotation removes the mean vertical-wind offset so that
        W contains only turbulent fluctuations.  Without rotation the
        cross-correlation is biased and the detected lag is unreliable.

    Distributes ``PreWhiteningBootstrap`` across CPU cores using
    ``ProcessPoolExecutor``.  Results accumulate into a DataFrame; an optional
    checkpoint CSV is written after every completed file (silently skipped when
    the file is locked, e.g. open in Excel) so that a crash can be diagnosed.

    PWBOPT post-processing (S1/S2/S3 selection and optional HDI pre-filter)
    from Vitale et al. (2024) Section 2.3 is available via the static methods
    ``apply_pwbopt()``, ``apply_hdi_prefilter()``, and ``fill_tlag_gaps()``.

    The module is also directly executable as a CLI tool — see the module
    docstring for the full flag reference and usage examples.

    Windows note: guard the calling script with ``if __name__ == '__main__':``
    because ``ProcessPoolExecutor`` uses the *spawn* start method on Windows.

    Parameters
    ----------
    files:
        Paths to averaging-period files (one file per 30-min period).
    scalars:
        ``{gas_label: column_name}`` mapping, e.g.
        ``{'CH4': 'ch4', 'N2O': 'n2o'}``.
    col_w:
        Column name for vertical wind W after loading and renaming.
    col_tsonic:
        Column name for sonic temperature T_SONIC, or ``None`` for
        2-combination mode (cw and wc only).
    hz:
        Sampling frequency in Hz. Defaults to 20.
    lag_max_s:
        CCF search half-width in seconds. Defaults to 10.0.
    n_bootstrap:
        Number of block-bootstrap replicates. Defaults to 99.
    block_length_s:
        Bootstrap block length in seconds. Defaults to 20.0.
    usecols:
        0-based column indices to select from each file.
    col_names:
        Column names to assign after selecting *usecols*.
    skiprows:
        Metadata rows before the column-name row (EddyPro default: 9).
    na_values:
        Strings to treat as NaN. Defaults to EddyPro conventions.
    min_valid_frac:
        Minimum fraction of non-NaN values for a series to be processed.
        Defaults to 0.3.
    output_dir:
        Optional directory for checkpoint CSV and diagnostic plots.
    save_plots:
        Save one diagnostic PNG per period per scalar. Requires *output_dir*.
    n_workers:
        Number of parallel worker processes. Defaults to ``os.cpu_count()``.

    Example
    -------
    See ``examples/flux/hires/flux_lag_pwb_batch.py`` for a complete example.

    See Also
    --------
    PreWhiteningBootstrap : Single-period PWB detection.
    """

    def __init__(
            self,
            files: list,
            scalars: dict,
            col_w: str,
            col_tsonic: str | None = None,
            hz: int = 20,
            lag_max_s: float = 10.0,
            n_bootstrap: int = 99,
            block_length_s: float = 20.0,
            usecols: list | None = None,
            col_names: list | None = None,
            skiprows: int = 9,
            na_values: list | None = None,
            min_valid_frac: float = 0.3,
            output_dir: Path | None = None,
            save_plots: bool = False,
            n_workers: int | None = None,
    ):
        if usecols is None or col_names is None:
            raise ValueError("usecols and col_names must both be provided.")
        if len(usecols) != len(col_names):
            raise ValueError("usecols and col_names must have the same length.")

        self.files = [Path(f) for f in files]
        self.scalars = scalars
        self.col_w = col_w
        self.col_tsonic = col_tsonic
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.usecols = usecols
        self.col_names = col_names
        self.skiprows = skiprows
        self.na_values = na_values if na_values is not None else _DEFAULT_NA_VALUES
        self.min_valid_frac = min_valid_frac
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_plots = save_plots
        self.n_workers = n_workers or os.cpu_count()

        self._results: DataFrame | None = None

    @property
    def results(self) -> DataFrame:
        """Accumulated results DataFrame (available after ``run()``)."""
        if self._results is None:
            raise RuntimeError("Call run() first.")
        return self._results

    def run(self, on_progress: Callable | None = None) -> DataFrame:
        """
        Execute PWB detection across all files in parallel.

        Results are collected in completion order then sorted back to the
        original file order.

        Args:
            on_progress: Optional callback ``f(completed, total, row)`` called
                each time a file finishes.

        Returns:
            DataFrame with one row per file.
        """
        plot_dir = None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_plots:
                plot_dir = self.output_dir / 'plots'
                plot_dir.mkdir(exist_ok=True)

        worker_args = [
            (
                str(f),
                self.scalars, self.col_w, self.col_tsonic,
                self.hz, self.lag_max_s, self.n_bootstrap, self.block_length_s,
                self.usecols, self.col_names, self.skiprows, self.na_values,
                self.min_valid_frac,
                str(plot_dir) if plot_dir else None,
                self.save_plots,
            )
            for f in self.files
        ]

        rows: list[dict] = []
        total = len(worker_args)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(_pwb_file_worker, args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_file):
                row = future.result()
                rows.append(row)
                completed += 1

                if self.output_dir:
                    try:
                        pd.DataFrame(rows).to_csv(
                            self.output_dir / 'tlag_results_checkpoint.csv',
                            index=False,
                        )
                    except PermissionError:
                        pass  # file locked (e.g. open in Excel); skip checkpoint

                if on_progress is not None:
                    on_progress(completed, total, row)

        name_to_idx = {Path(f).name: i for i, f in enumerate(self.files)}
        rows.sort(key=lambda r: name_to_idx.get(r.get('period', ''), total))

        self._results = pd.DataFrame(rows)
        if self.output_dir:
            self._results.to_csv(
                self.output_dir / 'tlag_results.csv',
                index=False,
            )

        return self._results

    # ------------------------------------------------------------------
    # PWBOPT post-processing (static: usable standalone or via instance)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_pwbopt(
            tlag_s,
            hdi_range_s,
            hdi_thresh: float = 0.5,
            dev_thresh: float = 0.5,
    ) -> DataFrame:
        """
        Apply PWBOPT S1/S2/S3 selection to a sequence of PWB lag estimates.

        S1 -- HDI range < *hdi_thresh* -> reliable, accept directly.
        S2 -- uncertain but within *dev_thresh* of the preceding optimal lag
              -> accept for temporal continuity.
        S3 -- unreliable -> carry forward the last known optimal lag.

        Args:
            tlag_s: Detected PWB lags in seconds (NaN where detection failed).
            hdi_range_s: 95% HDI range in seconds per period.
            hdi_thresh: S1 threshold in seconds. Default: 0.5.
            dev_thresh: S2 max deviation from the preceding optimal. Default: 0.5.

        Returns:
            DataFrame with columns ``pwbopt_s`` (optimal lag, s) and ``flag``.
        """
        tlag_s = np.asarray(tlag_s, dtype=float)
        hdi_range_s = np.asarray(hdi_range_s, dtype=float)
        n = len(tlag_s)
        flags = ['S3_unreliable'] * n
        optimal = np.full(n, np.nan)
        last_optimal = np.nan

        for i in range(n):
            tl = tlag_s[i]
            hdi = hdi_range_s[i]

            if np.isnan(tl) or np.isnan(hdi):
                optimal[i] = last_optimal
                continue

            if hdi < hdi_thresh:
                flags[i] = 'S1_optimal'
                optimal[i] = tl
                last_optimal = tl
            elif not np.isnan(last_optimal) and abs(tl - last_optimal) <= dev_thresh:
                flags[i] = 'S2_optimal'
                optimal[i] = tl
                last_optimal = tl
            else:
                optimal[i] = last_optimal

        return pd.DataFrame({'pwbopt_s': optimal, 'flag': flags})

    @staticmethod
    def fill_tlag_gaps(
            pwbopt_s,
            tlag_s_raw=None,
            fallback: float | None = None,
    ) -> np.ndarray:
        """
        Fill NaN values in a PWBOPT lag series so every averaging period has
        a usable time lag for flux covariance calculation.

        PWBOPT carries the last known optimal lag forward in time.  Periods
        *before* the first S1/S2 detection have nothing to carry forward and
        remain NaN.  This method fills them with a three-step strategy:

        1. **Backward fill** — propagates the first reliable lag backward to
           cover the leading NaN periods.
        2. **Median of raw lags** — if the entire series is NaN (no S1/S2
           detection at all), the median of all raw detected lags is used.
        3. **Explicit fallback** — constant value used as last resort (e.g.
           the nominal tube-delay for the gas/site).

        Args:
            pwbopt_s: Optimal lag series from ``apply_pwbopt()``.
            tlag_s_raw: Raw PWB lags before PWBOPT.  Used to compute the
                median fallback.  Ignored when ``None``.
            fallback: Constant lag in seconds used as last resort.

        Returns:
            Array of the same length as *pwbopt_s*, NaN-free when a finite
            value can be found through any of the three strategies.
        """
        result = pd.Series(np.asarray(pwbopt_s, dtype=float))
        result = result.bfill()

        if result.isna().any() and tlag_s_raw is not None:
            raw = np.asarray(tlag_s_raw, dtype=float)
            median_raw = np.nanmedian(raw)
            if np.isfinite(median_raw):
                result = result.fillna(median_raw)

        if result.isna().any() and fallback is not None:
            result = result.fillna(fallback)

        return result.to_numpy()

    @staticmethod
    def apply_hdi_prefilter(
            tlag_s,
            hdi_range_s,
            threshold: float = 1.0,
    ) -> np.ndarray:
        """
        Replace lags whose HDI exceeds *threshold* with NaN before PWBOPT.

        More conservative than standard PWBOPT: wide-HDI detections are
        discarded before S1/S2/S3 runs, so S2 cannot accept them even if they
        happen to lie close to the preceding optimal lag.

        Args:
            tlag_s: Detected lags in seconds.
            hdi_range_s: 95% HDI range in seconds.
            threshold: Lags with HDI range wider than this are set to NaN.

        Returns:
            Array of pre-filtered lags (NaN where HDI > threshold).
        """
        tlag_filtered = np.asarray(tlag_s, dtype=float).copy()
        hdi = np.asarray(hdi_range_s, dtype=float)
        tlag_filtered[(hdi > threshold) & ~np.isnan(hdi)] = np.nan
        return tlag_filtered

    @staticmethod
    def plot_summary(
            results: DataFrame,
            scalars: dict,
            hdi_thresh: float = 0.5,
            hdi_prefilter: float = 1.0,
            lag_max_s: float = 10.0,
            output_dir: Path | None = None,
            showplot: bool = False,
    ) -> None:
        """
        Generate batch-level summary figures after PWBOPT post-processing.

        Produces one 5-panel figure per scalar and one scatter + KDE comparison
        figure across all scalars (``PwboptLagPlot``).  Panel layout:

        1. Detected lags coloured by S1/S2/S3 flag (scatter, no lines) + mode.
        2. Final gap-filled lags: S1/S2 anchor points (filled markers) +
           pre-filtered final lag as open black circles + mode.
        3. 95% HDI range bars with S1 and pre-filter threshold lines.
        4. Flag bars per period: standard vs. pre-filtered side by side.
        5. Histogram of all detected lags with mode marker.

        Expects *results* to already contain the PWBOPT columns added by
        ``apply_pwbopt()`` / ``apply_hdi_prefilter()`` / ``fill_tlag_gaps()``.

        Args:
            results: Per-period results DataFrame (one row per file).
            scalars: ``{gas_label: column_name}`` mapping used during detection.
            hdi_thresh: S1 HDI threshold in seconds (for plot label).
            hdi_prefilter: HDI pre-filter threshold in seconds (0 = disabled).
            lag_max_s: CCF half-width in seconds — sets y-axis limits.
            output_dir: Directory for saved PNGs.  ``None`` skips saving.
            showplot: Call ``plt.show()`` after each figure.
        """
        import matplotlib.patches as mpatches

        FLAG_COLORS = {
            'S1_optimal': '#2ca02c',
            'S2_optimal': '#ff7f0e',
            'S3_unreliable': '#d62728',
        }
        px = np.arange(len(results))
        out = Path(output_dir) if output_dir else None

        for scalar_label in scalars:
            prefix = scalar_label.lower()
            tlag_col = f'{prefix}_tlag_s'
            hdi_col = f'{prefix}_hdi_range_s'
            flag_std_col = f'{prefix}_flag_std'
            flag_pf_col = f'{prefix}_flag_pf'
            opt_pf_col = f'{prefix}_pwbopt_s_pf'
            final_pf_col = f'{prefix}_tlag_final_pf_s'

            if tlag_col not in results.columns or flag_std_col not in results.columns:
                continue

            tlag = results[tlag_col].values.astype(float)
            hdi = results[hdi_col].values.astype(float)
            flag_std = results[flag_std_col].values
            has_pf = flag_pf_col in results.columns
            has_final_pf = final_pf_col in results.columns
            valid_lags = tlag[~np.isnan(tlag)]

            # Mode: exact value counts on lags rounded to 1/hz resolution
            if len(valid_lags) > 0:
                mode_lag = (pd.Series(np.round(valid_lags, 2))
                            .value_counts().idxmax())
            else:
                mode_lag = np.nan

            fig, axes = plt.subplots(
                5, 1, figsize=(14, 17),
                gridspec_kw={'height_ratios': [3, 2.5, 2, 1.5, 2]},
            )
            fig.suptitle(
                f'{scalar_label} -- PWB lag pipeline (PWBOPT strategy comparison)',
                fontsize=12,
            )

            # Panel 1: raw detected lags coloured by S1/S2/S3, no connecting lines
            ax = axes[0]
            ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
            for flag, color in FLAG_COLORS.items():
                mask = flag_std == flag
                ax.scatter(px[mask], tlag[mask], color=color, s=50, zorder=3,
                           label=flag)
            if not np.isnan(mode_lag):
                ax.axhline(mode_lag, color='#9467bd', linewidth=1.2,
                           linestyle='-.', zorder=2,
                           label=f'mode = {mode_lag:.2f} s')
            ax.set_ylabel('Time lag (s)')
            ax.set_title('Detected lags per period (coloured by standard PWBOPT flag)')
            ax.legend(frameon=False, fontsize=8, ncol=4)
            ax.set_ylim(-lag_max_s - 0.5, lag_max_s + 0.5)

            # Panel 2: final (gap-filled) lags — S1/S2 anchor points +
            #          pre-filtered final lag as open black circles
            ax = axes[1]
            ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
            for flag in ('S1_optimal', 'S2_optimal'):
                mask = flag_std == flag
                ax.scatter(px[mask], tlag[mask], color=FLAG_COLORS[flag], s=50,
                           zorder=4, label=flag)
            if has_final_pf:
                final_pf = results[final_pf_col].values.astype(float)
                ax.scatter(px, final_pf, color='none', edgecolors='black',
                           linewidths=1.0, s=50, zorder=3,
                           label='Final lag — pre-filtered')
            if not np.isnan(mode_lag):
                ax.axhline(mode_lag, color='#9467bd', linewidth=1.2,
                           linestyle='-.', zorder=2,
                           label=f'mode = {mode_lag:.2f} s')
            ax.set_ylabel('Time lag (s)')
            ax.set_title('Final (gap-filled) lags used for flux calculation')
            ax.legend(frameon=False, fontsize=8, ncol=3)
            ax.set_ylim(-lag_max_s - 0.5, lag_max_s + 0.5)

            # Panel 3: HDI range
            ax = axes[2]
            ax.bar(px, hdi, color='#aec7e8', edgecolor='none', label='HDI range')
            ax.axhline(hdi_thresh, color='#2ca02c', linewidth=1.5, linestyle='--',
                       label=f'S1 threshold ({hdi_thresh} s)')
            if hdi_prefilter > 0:
                ax.axhline(hdi_prefilter, color='steelblue', linewidth=1.5,
                           linestyle=':',
                           label=f'Pre-filter threshold ({hdi_prefilter} s)')
            ax.set_ylabel('95% HDI range (s)')
            ax.set_title('Bootstrap uncertainty (HDI range) per period')
            ax.legend(frameon=False, fontsize=8)

            # Panel 4: flag bars — standard (left) vs pre-filtered (right)
            ax = axes[3]
            bar_w = 0.4
            flag_cols = [(flag_std_col, 0)]
            if has_pf:
                flag_cols.append((flag_pf_col, 1))
            for flag_col, offset in flag_cols:
                for p in px:
                    flag = results[flag_col].iloc[p]
                    ax.bar(p + (offset - 0.5) * bar_w, 1, bar_w,
                           color=FLAG_COLORS.get(flag, '#aaaaaa'), alpha=0.85)
            patches = [mpatches.Patch(color=c, label=f)
                       for f, c in FLAG_COLORS.items()]
            ax.legend(handles=patches, frameon=False, fontsize=8)
            ax.set_yticks([])
            ax.set_title(
                'Flag per period: standard (left bar) vs. pre-filtered (right bar)'
                if has_pf else 'Flag per period'
            )

            # Panel 5: histogram of detected lags
            ax = axes[4]
            ax.set_xlabel('Time lag (s)')
            ax.set_title(f'Distribution of detected lags  (n={len(valid_lags)})')
            if len(valid_lags) > 0:
                n_bins = min(50, max(10, len(valid_lags) // 4))
                ax.hist(valid_lags, bins=n_bins, range=(-lag_max_s, lag_max_s),
                        color='#aec7e8', edgecolor='white', linewidth=0.4)
                ax.axvline(0, color='#888888', linewidth=0.8, linestyle='-')
                ax.axvline(mode_lag, color='#9467bd', linewidth=1.5,
                           linestyle='-.', label=f'mode = {mode_lag:.2f} s')
                ax.legend(frameon=False, fontsize=8)
            ax.set_ylabel('Count')

            plt.tight_layout()
            if out:
                fig.savefig(out / f'summary_{prefix}.png', dpi=100,
                            bbox_inches='tight')
            if showplot:
                plt.show()
            else:
                plt.close(fig)

        # PwboptLagPlot: scatter + KDE — only when both std and pf columns exist
        scalars_plot = {}
        for label in scalars:
            pfx = label.lower()
            col_a = f'{pfx}_pwbopt_s_std'
            col_b = f'{pfx}_pwbopt_s_pf'
            if col_a in results.columns and col_b in results.columns:
                scalars_plot[label] = {'col_a': col_a, 'col_b': col_b}

        if scalars_plot:
            lag_plot = PwboptLagPlot(
                results=results,
                scalars=scalars_plot,
                label_a='PWBOPT standard',
                label_b='PWBOPT pre-filtered',
                color_a='#0072B2',
                color_b='#E05C2A',
            )
            lag_plot.plot(
                title='PWB optimal lag: standard vs. pre-filtered PWBOPT',
                showplot=showplot,
                outpath=str(out) if out else None,
                outname='summary_lag_comparison.png',
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m diive.pkgs.flux.hires.lag_pwb',
        description=(
            'Parallel PWB time-lag detection across EddyPro high-frequency files.\n'
            'Alias: uv run diive-tlag-pwb-batch'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument('--input-dir', required=True,
                   help='Directory containing EddyPro rotated .txt files.')
    p.add_argument('--file-pattern', default='*.txt',
                   help='Glob pattern for input files.')
    p.add_argument('--output-dir', required=True,
                   help='Directory for results CSV, checkpoint, and optional plots.')
    # Scalars
    p.add_argument('--scalar', dest='scalars', action='append',
                   metavar='LABEL:column', required=True,
                   help='Gas label and column name, e.g. CH4:ch4. Repeat for each gas.')
    # Column mapping
    p.add_argument('--col-w', default='w',
                   help='Column name for vertical wind W.')
    p.add_argument('--col-tsonic', default=None,
                   help='Column name for sonic temperature T_SONIC (enables 4-combination mode).')
    p.add_argument('--usecols', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5],
                   help='0-based column indices to read from each file.')
    p.add_argument('--col-names', nargs='+', default=['u', 'v', 'w', 'ts', 'ch4', 'n2o'],
                   help='Column names to assign after selecting --usecols.')
    p.add_argument('--skiprows', type=int, default=9,
                   help='Metadata rows before the column-name row.')
    p.add_argument('--na-values', nargs='+',
                   default=['-9999', '-9999.0', '-9999.0000000000000'],
                   help='Strings to treat as NaN.')
    # PWB parameters
    p.add_argument('--hz', type=int, default=20,
                   help='Sampling frequency in Hz.')
    p.add_argument('--lag-max', type=float, default=10.0,
                   help='CCF search half-width in seconds.')
    p.add_argument('--n-bootstrap', type=int, default=99,
                   help='Number of block-bootstrap replicates.')
    p.add_argument('--block-length', type=float, default=20.0,
                   help='Bootstrap block length in seconds.')
    p.add_argument('--min-valid-frac', type=float, default=0.3,
                   help='Minimum non-NaN fraction for a series to be processed.')
    # PWBOPT thresholds
    p.add_argument('--hdi-thresh', type=float, default=0.5,
                   help='S1 HDI threshold in seconds.')
    p.add_argument('--dev-thresh', type=float, default=0.5,
                   help='S2 deviation threshold in seconds.')
    p.add_argument('--hdi-prefilter', type=float, default=1.0,
                   help='HDI pre-filter threshold in seconds (0 = disabled).')
    # Execution
    p.add_argument('--n-workers', type=int, default=None,
                   help='Parallel worker processes (default: os.cpu_count()).')
    p.add_argument('--save-plots', action='store_true',
                   help='Save one diagnostic PNG per period per scalar.')
    return p


def _cli_main():
    import sys
    from rich.console import Console as _Console
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    console = _Console(log_path=False)

    args = _build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f'ERROR: --input-dir not found: {input_dir}', file=sys.stderr)
        sys.exit(1)

    files = sorted(input_dir.glob(args.file_pattern))
    if not files:
        print(f'ERROR: no files matching {args.file_pattern!r} in {input_dir}',
              file=sys.stderr)
        sys.exit(1)

    scalars = {}
    for token in args.scalars:
        if ':' not in token:
            print(f'ERROR: --scalar must be LABEL:column, got {token!r}',
                  file=sys.stderr)
            sys.exit(1)
        label, col = token.split(':', 1)
        scalars[label] = col

    det = PwbBatchDetection(
        files=files,
        scalars=scalars,
        col_w=args.col_w,
        col_tsonic=args.col_tsonic,
        hz=args.hz,
        lag_max_s=args.lag_max,
        n_bootstrap=args.n_bootstrap,
        block_length_s=args.block_length,
        usecols=args.usecols,
        col_names=args.col_names,
        skiprows=args.skiprows,
        na_values=args.na_values,
        min_valid_frac=args.min_valid_frac,
        output_dir=Path(args.output_dir),
        save_plots=args.save_plots,
        n_workers=args.n_workers,
    )

    msg = (f'PWB batch detection  {len(files)} files  '
           f'{det.n_workers} workers  -> {args.output_dir}')

    console.print(f'\n[bold]{msg}[/bold]\n')

    def _fmt(row, gas):
        pfx = gas.lower()
        v = row.get(f'{pfx}_tlag_s')
        h = row.get(f'{pfx}_hdi_range_s')
        if v is None or v != v:
            return f'[dim]{gas}=--[/dim]'
        hdi_color = ('green' if h == h and h < 0.5
                     else ('yellow' if h == h and h < 1.0 else 'red'))
        return f'{gas}=[bold]{v:.2f}s[/bold] HDI=[{hdi_color}]{h:.2f}[/{hdi_color}]'

    prog = Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=8,
    )
    task_id = prog.add_task(
        f'[cyan]{det.n_workers} workers[/cyan]', total=len(files))

    with prog:
        def _cb(done, total, row):
            period = row.get('period', '')
            period_short = Path(period).stem.split('_')[0]
            parts = '  '.join(_fmt(row, g) for g in scalars)
            console.log(f'[dim]{period_short}[/dim]  {parts}')
            prog.update(task_id, completed=done,
                        description=f'[cyan]{det.n_workers} workers[/cyan]  [dim]{period_short}[/dim]')

        results = det.run(on_progress=_cb)

    console.print(f'\n[green]Done — {len(results)} periods.[/green]')

    # PWBOPT post-processing
    for label in scalars:
        pfx = label.lower()
        tc, hc = f'{pfx}_tlag_s', f'{pfx}_hdi_range_s'
        if tc not in results.columns:
            continue
        std = PwbBatchDetection.apply_pwbopt(
            results[tc], results[hc], args.hdi_thresh, args.dev_thresh)
        results[f'{pfx}_pwbopt_s_std'] = std['pwbopt_s']
        results[f'{pfx}_flag_std'] = std['flag']

        if args.hdi_prefilter > 0:
            tpf = PwbBatchDetection.apply_hdi_prefilter(
                results[tc], results[hc], args.hdi_prefilter)
            pf = PwbBatchDetection.apply_pwbopt(
                tpf, results[hc], args.hdi_thresh, args.dev_thresh)
            results[f'{pfx}_pwbopt_s_pf'] = pf['pwbopt_s']
            results[f'{pfx}_flag_pf'] = pf['flag']

        # Fill leading/trailing NaN lags (raw lags supply the median fallback)
        raw_tlag = results[tc]
        results[f'{pfx}_tlag_final_s'] = PwbBatchDetection.fill_tlag_gaps(
            results[f'{pfx}_pwbopt_s_std'], tlag_s_raw=raw_tlag)
        if f'{pfx}_pwbopt_s_pf' in results.columns:
            results[f'{pfx}_tlag_final_pf_s'] = PwbBatchDetection.fill_tlag_gaps(
                results[f'{pfx}_pwbopt_s_pf'], tlag_s_raw=raw_tlag)

    PwbBatchDetection.plot_summary(
        results=results,
        scalars=scalars,
        hdi_thresh=args.hdi_thresh,
        hdi_prefilter=args.hdi_prefilter,
        lag_max_s=args.lag_max,
        output_dir=Path(args.output_dir),
        showplot=False,
    )

    out_csv = Path(args.output_dir) / 'tlag_results.csv'
    results.to_csv(out_csv, index=False)
    print(f'Results saved to: {out_csv}')


if __name__ == '__main__':
    _cli_main()
