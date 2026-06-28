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
draws N_B resampled series (moving-block resampling with overlapping blocks of
length L to preserve local autocorrelation structure, matching R's
``tsboot(sim="fixed", l=LAG.MAX*2)``), detects the peak lag in each, and
summarises the resulting distribution with a mode (the lag estimate) and a 95%
Highest Density Interval (HDI).  A narrow HDI (< 0.5 s) indicates that
repeated resampling consistently finds the same lag -- the S1 reliability
criterion from the paper.

**Four CCF combinations** (following RFlux v3.2.0) are always evaluated; the
sonic temperature T_SONIC is required.  For each combination a different AR
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
- A Breitung (2002) variance-ratio unit-root test (R: egcm::bvr.test) is applied
  to each aligned series before AR fitting.  If any series has a unit root
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
- var_tsonic is required: the 4-combination RFlux v3.2.0 logic always runs.
  T_SONIC combinations are especially valuable for trace gases (N2O, CH4) with
  a weak scalar x W signal.
- Optional lws/uws restrict the bootstrap peak search to an asymmetric window
  [lws, uws] in seconds (RFlux's lws/uws). The CCF is still computed
  symmetrically over +/-lag_max; only the argmax is windowed, so the negative
  half beyond the window is computed-but-diagnostic.  Both None (default) =
  full symmetric search, byte-identical to prior behaviour.  A positive-only
  window (e.g. lws=0, uws=25) keeps only physical tube-delay lags -- useful for
  a long-inlet gas such as H2O.  See window_to_lag_params in
  detect_and_remove_tlag.py for the window -> (lag_max, block) mapping used by
  the pipeline.

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

    python -m diive.flux.hires.lag_pwb --help

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
``--col-tsonic NAME`` (required)
    Column name for sonic temperature T_SONIC.  Enables the full 4-combination
    RFlux v3.2.0 logic (especially valuable for weak-flux trace gases).
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

import datetime as dt
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
from scipy.signal import correlate as _signal_correlate, detrend as _detrend, lfilter
from scipy.stats import gaussian_kde

from diive.core.plotting.plotfuncs import default_format
from diive.core.plotting.styles import LightTheme as theme
from diive.core.utils.console import warn

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
    N_B resampled series (moving-block resampling with overlapping blocks of
    length L to preserve local autocorrelation, matching R's
    ``tsboot(sim="fixed", l=LAG.MAX*2)``) and detects the peak lag in each.  The mode of the
    resulting N_B lag estimates is the final lag; the 95% HDI (Highest Density
    Interval, shortest interval containing 95% of the distribution) measures
    how consistently the resampling agrees on that lag.  A narrow HDI means
    the lag is robust; a wide HDI signals an unreliable estimate.

    **Why four combinations?**  *var_tsonic* is required, so four pre-whitened
    CCF variants are always computed (matching R and the paper).  T_SONIC and W
    co-vary through buoyant turbulent structures; for trace gases (N2O, CH4) the
    scalar x W signal is often too weak to produce a clear peak, whereas
    scalar x T_SONIC combinations can reveal the same tube-delay lag more
    clearly.  The combination that produces the highest smoothed CCF value at
    its mode lag is selected (R: which.max(abs(corr_est_s))):

        cw  -- scalar x W,       scalar AR filter
        ct  -- scalar x T_SONIC, scalar AR filter
        wc  -- scalar x W,       W AR filter
        tc  -- scalar x T_SONIC, T_SONIC AR filter

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
        var_tsonic (str): Column name for sonic temperature T_SONIC (required).
            Enables the 4-combination RFlux v3.2.0 logic, which always runs.
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
            var_tsonic: str,
            hz: int = 20,
            lag_max_s: float = 10.0,
            n_bootstrap: int = 99,
            block_length_s: float | None = None,
            wdt: int = 5,
            random_state: int | np.random.Generator | None = None,
            segment_name: str = 'segment',
            lws: float | None = None,
            uws: float | None = None,
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
            var_tsonic (str): Column name for sonic temperature T_SONIC.
                Required: the full 4-combination RFlux v3.2.0 logic always runs
                (matching R and the paper, which compute all four combinations).
            hz (int): Acquisition frequency in samples per second (10 or 20).
                Defaults to 20.
            lag_max_s (float): Half-width of lag search window in seconds
                (R: LAG.MAX = mfreq * 10 = 10 s). Defaults to 10.0.
            n_bootstrap (int): Number of block-bootstrap samples (N_B in the
                paper). Defaults to 99.
            block_length_s (float | None): Bootstrap block length in seconds (L).
                When None (default), set to ``2 * lag_max_s`` following R's
                ``l = LAG.MAX * 2``.  The paper treats L and LAG.MAX as
                independent (L = 20 s, LAG.MAX = 10 s); R couples them, and the
                default here follows R.  An explicit value inconsistent with
                ``2 * lag_max_s`` emits a warning.
            wdt (int): Bootstrap CCF smoothing width.  ``wdt=5`` matches R's
                default; the paper equation 6 specifies ``hz/2 + 1``.  Set
                ``wdt = hz // 2 + 1`` for paper-equivalent behaviour.
                Defaults to 5.
            random_state (int | np.random.Generator | None): Seed or generator
                for the block-bootstrap resampling and the MAP jitter.  Pass an
                int (or Generator) for reproducible results; None (default)
                leaves the bootstrap non-reproducible.
            segment_name (str): Identifier for this averaging period. Defaults
                to 'segment'.
            lws (float | None): Lower limit of an optional asymmetric search
                window in seconds.  uws (float | None): upper limit.  When
                either is set, the bootstrap peak search and the diagnostic
                ``tlag_pw`` are restricted to lags within ``[lws, uws]`` (the
                unset bound falls back to ``-lag_max_s`` / ``+lag_max_s``).
                Use e.g. ``lws=0, uws=25`` to keep only physical positive
                tube-delay lags up to 25 s -- useful for a long-inlet gas such
                as H2O whose lag is large and where unphysical negative-lag
                peaks would otherwise compete.  Both ``None`` (default) ->
                full symmetric window, identical to prior behaviour.  Mirrors
                RFlux's lws/uws, but here the window constrains the *returned*
                PWB lag (R computes a windowed variant separately).
        """
        self.df = df
        self.var_w = var_w
        self.var_scalar = var_scalar
        self.var_tsonic = var_tsonic
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap

        # Block length defaults to R's l = LAG.MAX * 2 (coupled to lag_max_s).
        # A block LONGER than the coupling is fine -- it preserves more
        # autocorrelation structure and is intentional when a per-gas window
        # floors the block (window_to_lag_params). Only a block SHORTER than
        # 2*lag_max risks being too small to contain the lag, so warn just then.
        if block_length_s is None:
            block_length_s = 2.0 * lag_max_s
        elif block_length_s < 2.0 * lag_max_s - 1e-9:
            warn(f"block_length_s={block_length_s} s is shorter than R's coupled "
                 f"default 2 * lag_max_s = {2.0 * lag_max_s} s.")
        self.block_length_s = block_length_s
        self.wdt = wdt
        self.segment_name = segment_name

        self._rng = np.random.default_rng(random_state)

        # Search window half-width in records (R: LAG.MAX = mfreq * 10)
        self._lag_max_records = int(round(lag_max_s * hz))
        # Block length in records (R: l = LAG.MAX * 2)
        self._block_length_records = int(round(block_length_s * hz))

        # Optional asymmetric search window [lws, uws] in seconds. None on both
        # -> the full symmetric window, so _windowed_argmax reduces to a plain
        # argmax over the whole CCF (no behaviour change vs. prior versions).
        self.lws = lws
        self.uws = uws
        if lws is not None or uws is not None:
            lo_s = -lag_max_s if lws is None else lws
            hi_s = lag_max_s if uws is None else uws
            lo_rec = max(-self._lag_max_records, int(round(lo_s * hz)))
            hi_rec = min(self._lag_max_records, int(round(hi_s * hz)))
            if hi_rec <= lo_rec:
                raise ValueError(
                    f"Empty search window: [{lo_s}, {hi_s}] s resolves to no "
                    f"lags within +/-lag_max_s={lag_max_s} s.")
            self._win_lo_idx = lo_rec + self._lag_max_records
            self._win_hi_idx = hi_rec + self._lag_max_records  # inclusive
        else:
            self._win_lo_idx = 0
            self._win_hi_idx = 2 * self._lag_max_records  # inclusive (full)

        # --- Results populated by run() ---
        # PW results (full-data scalar-AR CCF, for diagnostic panels)
        self._tlag_pw_records: int | None = None
        self._corr_pw: float | None = None
        self._n_eff: int | None = None
        # PWB results (from the selected combination's bootstrap)
        self._tlag_records: int | None = None
        self._cov_pwb: float | None = None
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
        """PW time lag in records (argmax of the unsmoothed pre-whitened CCF)."""
        if self._tlag_pw_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_pw_records

    @property
    def tlag_pw_s(self) -> float:
        """PW time lag in seconds."""
        return self.tlag_pw_records / self.hz

    @property
    def corr_pw(self) -> float:
        """Un-smoothed PW CCF value at tlag_pw (R: cor_pww)."""
        if self._corr_pw is None:
            raise RuntimeError("Call run() first.")
        return self._corr_pw

    @property
    def cv_99(self) -> float:
        """Bartlett 99% significance threshold (R: 3.291 / sqrt(N * 13))."""
        if self._n_eff is None:
            raise RuntimeError("Call run() first.")
        return 3.291 / np.sqrt(self._n_eff * 13)

    # ------------------------------------------------------------------
    # Properties -- PWB results
    # ------------------------------------------------------------------

    @property
    def tlag_records(self) -> int:
        """Bootstrap (PWB) time lag in records (mode over N_B samples).

        This is the raw mode position even when it falls on the search-window
        edge (see ``is_edge_pinned``); the *reported* lag ``tlag_s`` is NaN in
        that case.  Kept raw for diagnostics (e.g. the per-chunk plot).
        """
        if self._tlag_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_records

    @property
    def is_edge_pinned(self) -> bool:
        """True if the bootstrap mode lag sits on the search-window boundary.

        A peak at the window edge means the true peak lies at or beyond the
        window limit -- the lag is undetermined, not measured.  A chunk with no
        real W-scalar coupling produces a flat, noisy cross-correlation whose
        ``argmax`` is pushed to the edge by the na.locf edge-fill; every
        bootstrap replicate then agrees there, faking a zero-width HDI.  Such
        edge detections are treated as failures: ``tlag_s`` and the HDI are NaN
        and ``is_reliable`` is False, so the lag is never carried into PWBOPT or
        applied -- matching EddyPro, which also discards boundary lags.  The
        edge is each gas's *own* window (``lws``/``uws``), not the global
        ``+/-lag_max_s``.
        """
        if self._tlag_records is None:
            raise RuntimeError("Call run() first.")
        lo = self._win_lo_idx - self._lag_max_records
        hi = self._win_hi_idx - self._lag_max_records
        return self._tlag_records <= lo or self._tlag_records >= hi

    @property
    def tlag_s(self) -> float:
        """Bootstrap (PWB) time lag in seconds (NaN if edge-pinned = failed)."""
        if self.is_edge_pinned:
            return np.nan
        return self.tlag_records / self.hz

    @property
    def cov_pwb(self) -> float:
        """Raw cross-covariance at the selected PWB lag (R: cov_pwb = ccf_mcw[peak_ref])."""
        if self._cov_pwb is None:
            raise RuntimeError("Call run() first.")
        return self._cov_pwb

    @property
    def hdi_lo_s(self) -> float:
        """Lower bound of 95% HDI in seconds (NaN if edge-pinned = failed)."""
        if self._hdi_lo_s is None:
            raise RuntimeError("Call run() first.")
        return np.nan if self.is_edge_pinned else self._hdi_lo_s

    @property
    def hdi_hi_s(self) -> float:
        """Upper bound of 95% HDI in seconds (NaN if edge-pinned = failed)."""
        if self._hdi_hi_s is None:
            raise RuntimeError("Call run() first.")
        return np.nan if self.is_edge_pinned else self._hdi_hi_s

    @property
    def hdi_range_s(self) -> float:
        """Width of 95% HDI in seconds (uncertainty measure)."""
        return self.hdi_hi_s - self.hdi_lo_s

    @property
    def is_reliable(self) -> bool:
        """True if HDI range < 0.5 s (S1 reliability criterion, paper Section 2.3).

        Always False for an edge-pinned detection (a failed detection).
        """
        if self.is_edge_pinned:
            return False
        return self.hdi_range_s < 0.5

    @property
    def results(self) -> dict:
        """All key outputs as a dictionary."""
        return {
            'segment_name': self.segment_name,
            # PW results (full-data scalar-AR CCF, for diagnostics)
            'tlag_pw_records': self.tlag_pw_records,
            'tlag_pw_s': self.tlag_pw_s,
            'corr_pw': self.corr_pw,
            'ar_order': self._ar_order,  # scalar AR order (primary)
            'ar_orders': self._ar_orders,  # all fitted AR orders
            'best_combination': self._best_combination,
            # PWB results (from the selected combination)
            'tlag_records': self.tlag_records,
            'tlag_s': self.tlag_s,
            'cov_pwb': self.cov_pwb,
            'hdi_lo_s': self.hdi_lo_s,
            'hdi_hi_s': self.hdi_hi_s,
            'hdi_range_s': self.hdi_range_s,
            'is_reliable': self.is_reliable,
            'n_bootstrap': self.n_bootstrap,
            'lws': self.lws,
            'uws': self.uws,
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
        1b. Breitung (2002) variance-ratio unit-root test on each series
           (R: egcm::bvr.test).  If any series has a unit root (p >= 0.01),
           first-difference all series.
        2. Fit separate AR(p) models to the scalar, W, and T_SONIC using AIC
           selection with max_order = floor(100*log10(N))
           (R: ar(x, aic=TRUE, order.max=floor(10^2*log10(length(x))))).
        3. Apply each AR filter to all relevant series to produce the
           pre-whitened filtered arrays (R: x1/y1/z1/x2/y2/x3/z3).
        4. Compute raw cross-covariance on the linearly detrended data
           (R: ccf(detrend(scalar), detrend(w), type="covariance")).
        5. Compute full-data PW CCF using the scalar AR filter on W and scalar
           (R: ccf_pww). Detect tlag_pw as argmax of the unsmoothed PW CCF
           (R: tl_pww <- which.max(abs(ccf_pww))).
        6. Block-bootstrap each of the four combinations:
             cw  -- scalar x W,       scalar AR (R: bootccf_cw)
             wc  -- scalar x W,       W AR      (R: bootccf_wc)
             ct  -- scalar x T_SONIC, scalar AR (R: bootccf_ct)
             tc  -- scalar x T_SONIC, T_SONIC AR(R: bootccf_tc)
           Per replicate: smooth CCF with width=wdt, apply na.locf to edges,
           find the peak lag index.  Also compute the mean bootstrap CCF,
           smoothed with the same width + na.locf (R: ccf_cw / ccfs_cw).
        7. Select the combination with the highest |avg_smooth_ccf| at its mode
           lag (R: corr_est_s -> which.max -> corr_ind).
        8. Summarise the winning combination's bootstrap distribution:
           mode -> tlag_s, 95% HDI -> hdi_lo_s / hdi_hi_s, and read the raw
           cross-covariance at the selected lag (R: cov_pwb = ccf_mcw[peak_ref]).
        """
        # ---- Step 1: load, interpolate NaN, align ----
        w_raw = _na_approx(self.df[self.var_w].values.astype(float))
        s_raw = _na_approx(self.df[self.var_scalar].values.astype(float))
        t_raw = _na_approx(self.df[self.var_tsonic].values.astype(float))
        valid = ~np.isnan(w_raw) & ~np.isnan(s_raw) & ~np.isnan(t_raw)

        w = w_raw[valid]
        s = s_raw[valid]
        t = t_raw[valid]

        # ---- Step 1b: stationarity check (R: egcm::bvr.test) ----
        # Breitung variance-ratio unit-root test on each aligned series.  If ANY
        # series has a unit root (p >= 0.01), first-difference ALL series before
        # AR fitting.  This matches R's logic exactly: stationarity of all three
        # is required to use the original series; a single failure triggers
        # differencing.  For turbulent EC data the test virtually always passes.
        # The rare failures occur during sensor drift, rain events, or artefacts.
        if not all(self._is_stationary(x) for x in (s, w, t)):
            s = np.diff(s)
            w = np.diff(w)
            t = np.diff(t)

        self._lags_axis = np.arange(-self._lag_max_records, self._lag_max_records + 1)

        # ---- Step 2: fit separate AR models ----
        # R: ar.resx = ar(x=scalar, ...), ar.resz = ar(z=W, ...), ar.resy = ar(y=T_SONIC, ...)
        phi_s, p_s = self._fit_ar_model(s)  # scalar AR
        phi_w, p_w = self._fit_ar_model(w)  # W AR
        phi_t, p_t = self._fit_ar_model(t)  # T_SONIC AR
        self._ar_order = p_s
        self._ar_orders = {'scalar': p_s, 'w': p_w, 'tsonic': p_t}

        # ---- Step 3: filtered series ----
        # Scalar AR applied to scalar (x1), W (z1), and T_SONIC (y1)
        s_fa = self._apply_ar_filter(phi_s, s)  # scalar filt by scalar AR  (R: x1)
        w_fa = self._apply_ar_filter(phi_s, w)  # W filt by scalar AR       (R: z1)
        # W AR applied to scalar (x3) and W (z3)
        s_fw = self._apply_ar_filter(phi_w, s)  # scalar filt by W AR       (R: x3)
        w_fw = self._apply_ar_filter(phi_w, w)  # W filt by W AR            (R: z3)
        t_fa = self._apply_ar_filter(phi_s, t)  # T_SONIC filt by scalar AR (R: y1)
        s_ft = self._apply_ar_filter(phi_t, s)  # scalar filt by T_SONIC AR (R: x2)
        t_ft = self._apply_ar_filter(phi_t, t)  # T_SONIC filt by T_SONIC AR(R: y2)

        # ---- Step 4: raw cross-covariance (diagnostic panel 2) ----
        # R: ccf(detrend(scalar), detrend(w), type="covariance") -- linear detrend
        # first (w and s are NaN-free here after na.approx + valid masking).
        self._raw_ccov = self._compute_ccov(_detrend(w, type='linear'),
                                            _detrend(s, type='linear'))
        self._smooth_raw_ccov = self._smooth_series(self._raw_ccov, _SMOOTH_WIDTH_CCOV)

        # ---- Step 5: full-data PW CCF, scalar AR (diagnostic panel 1) ----
        # R: ccf_pww = ccf(x1, z1)  ->  scalar AR filter, scalar x W pair
        self._pw_ccf = self._compute_ccf(w_fa, s_fa)
        self._smooth_pw_ccf = self._smooth_series(self._pw_ccf, _SMOOTH_WIDTH_CCF)
        # R uses length(x1) including leading NaN for the Bartlett denominator
        self._n_eff = len(s_fa)

        # tlag_pw = argmax of the UNSMOOTHED PW CCF (R: tl_pww <- which.max(abs(ccf_pww)))
        tl0 = self._windowed_argmax(self._pw_ccf)
        self._tlag_pw_records = tl0 - self._lag_max_records
        self._corr_pw = float(self._pw_ccf[tl0])

        # ---- Step 6: block-bootstrap each combination ----
        # x argument = leading signal (W or T_SONIC), y = scalar (delayed by tube)
        combinations = {
            'cw': self._run_combination_bootstrap(w_fa, s_fa),  # scalar x W, scalar AR
            'wc': self._run_combination_bootstrap(w_fw, s_fw),  # scalar x W, W AR
            'ct': self._run_combination_bootstrap(t_fa, s_fa),  # scalar x T_SONIC, scalar AR
            'tc': self._run_combination_bootstrap(t_ft, s_ft),  # scalar x T_SONIC, T_SONIC AR
        }

        # ---- Step 7: select the winning combination ----
        best_key = self._select_best_combination(combinations)
        self._best_combination = best_key
        best = combinations[best_key]

        # ---- Step 8: summarise from the winning combination ----
        # Reuse the SAME MAP estimate that won the selection (R uses one `maps`
        # value for selection, the reported lag, and cov_pwb).  Recomputing it
        # here would draw fresh jitter and could report a lag different from the
        # one that was actually selected.
        self._bootstrap_lags = best['lags']
        self._tlag_records = int(best['mode_lag'])
        hdi_lo, hdi_hi = self._hdi(best['lags'] / self.hz, credible_mass=0.95)
        self._hdi_lo_s = float(hdi_lo)
        self._hdi_hi_s = float(hdi_hi)

        # Raw cross-covariance at the selected PWB lag (R: cov_pwb = ccf_mcw[peak_ref])
        cov_idx = self._tlag_records + self._lag_max_records
        self._cov_pwb = (float(self._raw_ccov[cov_idx])
                         if 0 <= cov_idx < len(self._raw_ccov) else np.nan)

    # ------------------------------------------------------------------
    # Private: pre-whitening  (AR fitting and filtering)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stationary(x: np.ndarray, alpha: float = 0.01) -> bool:
        """
        Breitung (2002) variance-ratio unit-root test: return True if x is
        stationary.

        Matches R's egcm::bvr.test used in tlag_detection.R.  The statistic is

            rho = sum(S_t^2) / (n^2 * sum(e_t^2)),   e_t = x_t - mean(x),
                                                     S_t = cumsum(e_t)

        with H0 = unit root (non-stationary).  Small rho rejects H0, confirming
        stationarity.  R compares ``bvr.test(...)$p.val < 0.01``; because 0.01
        is itself a tabulated quantile, that is equivalent to ``rho < CV_1PCT``,
        where CV_1PCT is the 1% entry of egcm's ``bvr_qtab``.  For any EC
        averaging period (n >> 1250) egcm's ``quantile_table_interpolate`` clamps
        the sample size to the n=1250 column (``findInterval`` saturates and the
        cross-column branch is skipped), whose 1% value is 0.00537748.  The 1%
        critical value varies only marginally across n (0.00538-0.00586 for
        n=1250 down to 25), so this constant is exact for realistic EC data.
        For turbulent EC series rho ~ 1e-5, so the test passes overwhelmingly;
        non-stationary series give rho of order 0.05-0.1.
        """
        # egcm bvr_qtab 1% quantile, n=1250 column (the clamp target for large n).
        _CV_1PCT = 0.00537748023783321
        if alpha != 0.01:
            raise ValueError("Breitung VR critical value tabulated only for alpha=0.01.")
        e = np.asarray(x, dtype=float)
        e = e - e.mean()
        n = len(e)
        sse = np.sum(e ** 2)
        if sse == 0.0:
            return True  # constant series: no unit root
        s_cum = np.cumsum(e)
        rho = np.sum(s_cum ** 2) / (n ** 2 * sse)
        return rho < _CV_1PCT

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

        # R: order.max = floor(10^2 * log10(length(x))); bound by n-1 so that
        # acf[max_lag] stays in range for short segments.
        max_lag = min(int(np.floor(100 * np.log10(n))), n - 1)

        # Biased autocorrelation via FFT. Only acf[0..max_lag] is used, so a pad
        # to n + max_lag (not 2*n) keeps those lags free of circular wraparound.
        nfft = 1 << (n + max_lag - 1).bit_length()
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
        mode_lag = self._map_estimate(boot_lags, self._rng)  # KDE MAP, matching R's bayestestR::map_estimate
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

        Procedure (RFlux v3.2.0, R tsboot(sim="fixed", l=LAG.MAX*2)):
        1. Moving-block (overlapping) bootstrap: every position 0..n-L is a
           valid block start; draw N_B sets of block starts with replacement.
        2. Assemble N_B bootstrap series by concatenating consecutive blocks of
           length L from each start, then truncate to length n.
        3. Batch FFT CCF for all N_B samples simultaneously.
        4. Compute mean CCF across N_B samples, smooth with width=wdt,
           apply na.locf (R: ccfs_cw = rollapply(mean_ccf, width=wdt) + na.locf).
        5. Per-replicate: smooth each CCF with the same width, apply na.locf
           to fill NaN edges (R: rollapply + na.locf per replicate in bootccf_*).
        6. Find the peak lag per replicate via argmax(abs(smooth_ccf)).

        Returns:
            boot_lags       -- shape (N_B,) peak lags in records
            mean_smooth_ccf -- shape (2*lag_max+1,) mean bootstrap CCF, smoothed
        """
        n = len(x_pw)
        L = self._block_length_records
        N_B = self.n_bootstrap

        # R tsboot sim="fixed": overlapping moving-block bootstrap. Every index in
        # [0, n-L] is a valid block start. Concatenate ceil(n/L) consecutive blocks
        # of length L per sample, then truncate to the original length n.
        n_starts = max(1, n - L + 1)
        n_blocks_per_sample = -(-n // L)  # ceil(n / L)
        starts = self._rng.integers(0, n_starts, size=(N_B, n_blocks_per_sample))
        offsets = np.arange(L)
        idx = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :])
        idx = idx.reshape(N_B, -1)[:, :n]
        idx = np.minimum(idx, n - 1)  # guard the final partial block

        x_boot = x_pw[idx]
        y_boot = y_pw[idx]

        all_ccf = self._batch_ccf_fft(x_boot, y_boot)

        # Mean CCF across all N_B samples, smoothed then na.locf to fill edge NaN.
        # Used by _select_best_combination to compare combinations (R: ccfs_cw etc.).
        mean_smooth_ccf = _na_locf_1d(
            self._smooth_series(np.mean(all_ccf, axis=0), self.wdt)
        )

        # Per-replicate: smooth each CCF, fill edge NaN with na.locf so that lags
        # at the boundary of the search window are not excluded from the argmax search
        # (without na.locf, nanargmax would skip those positions and could miss an
        # edge-lag peak -- R applies zoo::na.locf per replicate for the same reason).
        all_smooth = self._na_locf_rows(
            self._smooth_rows(all_ccf, self.wdt)
        )
        peak_indices = self._windowed_argmax(all_smooth)
        boot_lags = peak_indices.astype(int) - self._lag_max_records

        return boot_lags, mean_smooth_ccf

    def _windowed_argmax(self, mat: np.ndarray):
        """Argmax of ``|mat|`` restricted to the lws/uws window.

        ``mat`` is a signed CCF array, shape ``(N,)`` or ``(N_B, N)`` with
        ``N = 2*lag_max_records + 1``.  Returns the peak index (or per-row
        indices) *into the full array*, so the caller subtracts
        ``lag_max_records`` to get the lag in records.  With no window set the
        slice spans the whole array, so this is identical to a plain argmax
        over ``|mat|`` -- the default path is unchanged.
        """
        lo, hi = self._win_lo_idx, self._win_hi_idx
        a = np.abs(mat)
        if a.ndim == 1:
            return int(np.nanargmax(a[lo:hi + 1])) + lo
        return np.nanargmax(a[:, lo:hi + 1], axis=1) + lo

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

        # Only lags in [-lag_max, +lag_max] are kept, so circular wraparound is
        # harmless as long as n_fft >= N + lag_max (the contaminating terms fall
        # outside the linear-correlation support). This is half the size of the
        # full 2*N-1 zero-pad and gives identical values in the kept window.
        n_fft = 1 << (N + lag_max - 1).bit_length()

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

        Computed on the original (not pre-whitened) data. Used to read the raw
        cross-covariance at the selected lag (R: cov_pwb = ccf_mcw[peak_ref]).
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
    def _map_estimate(samples: np.ndarray, rng: np.random.Generator) -> int:
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
            rng:     Generator used for the tie-breaking jitter (reproducibility).

        Returns:
            MAP estimate rounded to the nearest integer record.
        """
        if len(np.unique(samples)) == 1:
            # All samples identical: gaussian_kde would have zero bandwidth.
            return int(samples[0])
        jittered = samples.astype(float) + rng.normal(0, 0.0001, len(samples))
        kde = gaussian_kde(jittered)
        x_grid = np.linspace(jittered.min(), jittered.max(), 512)
        return int(round(float(x_grid[np.argmax(kde(x_grid))])))

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
            title: str = None,
            showplot: bool = True,
            outpath: str = None,
            outname: str = None,
    ) -> Figure:
        """
        Three-panel diagnostic figure.

        Panel 1 (left) -- Pre-whitened CCF (scalar AR, full data):
            Grey stems + smoothed black line + Bartlett significance band
            (R: +/-3.291/sqrt(n*13)) + red tlag marker + significance annotation.

        Panel 2 (middle) -- Raw cross-covariance:
            Grey stems + smoothed cyan line + red tlag marker.

        Panel 3 (right) -- Bootstrap lag distribution (best combination):
            Histogram of N_B detected lags + 95% HDI shading + mode marker.
            Title includes the selected combination name.

        Args:
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

        if showplot:
            fig.show()
        elif outpath and outname:
            plt.close(fig)

        return fig

    def _plot_pw_ccf(self, ax: plt.Axes):
        """Panel 1: pre-whitened CCF (scalar AR, full data)."""
        lags_s = self._lags_axis / self.hz
        cv = self.cv_99  # R: +/-3.291/sqrt(n*13)

        ax.vlines(lags_s, 0, self._pw_ccf,
                  colors='#808080', linewidth=0.6, alpha=0.7)
        ax.plot(lags_s, self._smooth_pw_ccf,
                color='black', linewidth=2, label='smoothed CCF')

        ax.axhline(cv, color='steelblue', linestyle='--', linewidth=1,
                   label='+/-3.291/sqrt(n*13)')
        ax.axhline(-cv, color='steelblue', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(self.tlag_pw_s, color='red', linestyle=':', linewidth=1.5)

        if abs(self.corr_pw) >= cv:
            sig_txt = (f'PW peak at {self.tlag_pw_s:.3f} s'
                       f'  stat. significant')
        else:
            sig_txt = 'PW peak not stat. significant'

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
        ax.axvline(self.tlag_pw_s, color='red', linestyle=':', linewidth=1.5,
                   label=f'tlag_pw = {self.tlag_pw_s:.2f} s')

        ax.set_title(f'PW time lag at {self.tlag_pw_s:.2f} s', fontsize=9)
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
        elif outpath and outname:
            plt.close(fig)

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

def _format_width(file_date_format: str) -> int:
    """
    Number of characters a fixed-width strptime format renders to.

    Measured against a reference datetime whose every field is two digits
    (month 12, day 23, ...) so fixed-width directives render at full width.
    """
    return len(dt.datetime(2222, 12, 23, 13, 24, 25).strftime(file_date_format))


def _find_timestamp_offset(filename: str, file_date_format: str | None) -> int | None:
    """
    Locate the datetime token inside a filename by sliding-window ``strptime``.

    Slides a window of the format's rendered width across *filename* and returns
    the **leftmost** offset at which ``strptime`` succeeds, or ``None`` if no
    substring parses (or no format is given).  This offset is determined once
    from the first file and then reused for every subsequent file (see
    ``PwbBatchDetection``), so the search cost is paid only once per batch.
    """
    if not file_date_format:
        return None
    n = _format_width(file_date_format)
    for i in range(len(filename) - n + 1):
        try:
            dt.datetime.strptime(filename[i:i + n], file_date_format)
            return i
        except ValueError:
            continue
    return None


def _parse_file_timestamp(
        filename: str, file_date_format: str | None, offset: int | None
) -> pd.Timestamp:
    """
    Parse a timestamp from *filename* at a known *offset*.

    *file_date_format* is a ``datetime.strptime`` pattern for the datetime token
    (diive ``file_date_format`` convention), e.g. ``'%Y%m%d-%H%M'`` for
    ``'20210820-0930_rotated.txt'``.  *offset* is the start position previously
    located by :func:`_find_timestamp_offset` on the first file.  Returns
    ``pd.NaT`` when no format/offset is given or the slice does not parse.
    """
    if not file_date_format or offset is None:
        return pd.NaT
    n = _format_width(file_date_format)
    try:
        return pd.Timestamp(
            dt.datetime.strptime(filename[offset:offset + n], file_date_format))
    except (ValueError, TypeError):
        return pd.NaT


def _pwb_file_worker(args: tuple) -> dict:
    """Process one averaging-period file and return one result-row dict."""
    (filepath, scalars, col_w, col_tsonic,
     hz, lag_max_s, n_bootstrap, block_length_s,
     usecols, col_names, skiprows, na_values,
     min_valid_frac, plot_dir, save_plots, seed, strict,
     file_date_format, ts_offset) = args

    period_name = Path(filepath).name
    row: dict = {'period': period_name,
                 'timestamp': _parse_file_timestamp(period_name, file_date_format, ts_offset)}

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
    except Exception as e:
        if strict:
            raise
        row['_error'] = repr(e)
        return row

    if len(df) < 25:
        row['_error'] = f'too few records ({len(df)} < 25)'
        return row

    w_arr = np.asarray(df[col_w], dtype=float)
    if (np.mean(~np.isnan(w_arr)) < min_valid_frac
            or np.nanstd(w_arr) < np.finfo(float).eps):
        row['_error'] = 'W below min_valid_frac or constant'
        return row

    for scalar_idx, (scalar_label, scalar_col) in enumerate(scalars.items()):
        prefix = scalar_label.lower()
        _nan_keys = (
            'tlag_s', 'hdi_lo_s', 'hdi_hi_s',
            'hdi_range_s', 'tlag_pw_s', 'corr_pw', 'cov_pwb', 'ar_order',
        )
        nan_row = {f'{prefix}_{k}': np.nan for k in _nan_keys}

        if scalar_col not in df.columns:
            row.update(nan_row)
            row[f'{prefix}_error'] = f'scalar column {scalar_col!r} not found'
            continue
        if col_tsonic not in df.columns:
            row.update(nan_row)
            row[f'{prefix}_error'] = f'T_SONIC column {col_tsonic!r} not found'
            continue

        s_arr = np.asarray(df[scalar_col], dtype=float)
        if (np.mean(~np.isnan(s_arr)) < min_valid_frac
                or np.nanstd(s_arr) < np.finfo(float).eps):
            row.update(nan_row)
            row[f'{prefix}_error'] = 'scalar below min_valid_frac or constant'
            continue

        col_map = {col_w: 'W', scalar_col: scalar_label, col_tsonic: 'T_SONIC'}
        keep_cols = [col_w, scalar_col, col_tsonic]

        # Deterministic per-(file, scalar) seed so batch results are reproducible
        # regardless of worker completion order.
        sub_seed = None if seed is None else int(seed) + scalar_idx

        try:
            pwb = PreWhiteningBootstrap(
                df=df[keep_cols].rename(columns=col_map),
                var_w='W',
                var_scalar=scalar_label,
                var_tsonic='T_SONIC',
                hz=hz,
                lag_max_s=lag_max_s,
                n_bootstrap=n_bootstrap,
                block_length_s=block_length_s,
                random_state=sub_seed,
                segment_name=period_name,
            )
            pwb.run()
            res = pwb.results

            row[f'{prefix}_tlag_s'] = res['tlag_s']
            row[f'{prefix}_hdi_lo_s'] = res['hdi_lo_s']
            row[f'{prefix}_hdi_hi_s'] = res['hdi_hi_s']
            row[f'{prefix}_hdi_range_s'] = res['hdi_range_s']
            row[f'{prefix}_tlag_pw_s'] = res['tlag_pw_s']
            row[f'{prefix}_corr_pw'] = res['corr_pw']
            row[f'{prefix}_cov_pwb'] = res['cov_pwb']
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

        except Exception as e:
            if strict:
                raise
            row.update(nan_row)
            row[f'{prefix}_error'] = repr(e)

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
        Column name for sonic temperature T_SONIC (required). T_SONIC enables
        the full 4-combination RFlux v3.2.0 logic.
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
    random_state:
        Base seed for reproducibility. Each file gets a deterministic derived
        seed so results are independent of worker completion order. Defaults to
        ``None`` (non-reproducible).
    strict:
        If True, a worker re-raises the first exception instead of recording it
        in a ``*_error`` column. Defaults to False.
    file_date_format:
        Optional ``datetime.strptime`` pattern for the datetime token embedded in
        each filename (diive ``file_date_format`` convention), e.g.
        ``'%Y%m%d-%H%M'`` for ``'20210814-1100_raw_dataset_..._adv.txt'``.  The
        token may appear anywhere in the name: its position is located **once**,
        on the first file, by sliding a fixed-width window and taking the
        leftmost substring that parses.  That offset is then applied to every
        subsequent file — so all files must share the same naming pattern (same
        datetime position), which is the caller's responsibility.  When provided,
        a ``timestamp`` column is added to the results and used as the x-axis of
        the summary plots.  Defaults to ``None`` (results carry ``NaT`` and plots
        fall back to the integer period index).  The pattern must use fixed-width
        directives (``%Y %m %d %H %M %S``); variable-width tokens such as ``%-d``
        or month names are not supported.

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
            col_tsonic: str,
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
            random_state: int | None = None,
            strict: bool = False,
            file_date_format: str | None = None,
    ):
        """Set up batch PWB time-lag detection. See the class docstring."""
        if usecols is None or col_names is None:
            raise ValueError("usecols and col_names must both be provided.")
        if len(usecols) != len(col_names):
            raise ValueError("usecols and col_names must have the same length.")
        if not col_tsonic:
            raise ValueError("col_tsonic is required (T_SONIC enables the "
                             "4-combination RFlux v3.2.0 logic).")

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
        self.random_state = random_state
        self.strict = strict
        self.file_date_format = file_date_format

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

        # Locate the datetime token once, on the first file, then reuse the same
        # offset for all files. The user is responsible for ensuring every file
        # follows the same naming pattern (same datetime position).
        ts_offset = None
        if self.file_date_format and self.files:
            ts_offset = _find_timestamp_offset(
                self.files[0].name, self.file_date_format)
            if ts_offset is None:
                warn(f"Could not locate a {self.file_date_format!r} timestamp in "
                     f"the first filename {self.files[0].name!r}; the 'timestamp' "
                     f"column will be NaT for all files.")

        worker_args = [
            (
                str(f),
                self.scalars, self.col_w, self.col_tsonic,
                self.hz, self.lag_max_s, self.n_bootstrap, self.block_length_s,
                self.usecols, self.col_names, self.skiprows, self.na_values,
                self.min_valid_frac,
                str(plot_dir) if plot_dir else None,
                self.save_plots,
                None if self.random_state is None else self.random_state + i,
                self.strict,
                self.file_date_format, ts_offset,
            )
            for i, f in enumerate(self.files)
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

        Edge-pinned detections (lag at the search-window boundary) are already
        NaN here: ``PreWhiteningBootstrap`` rejects them at detection time (an
        edge lag is a failed detection and is never applied, matching EddyPro),
        so this method only sees usable lags or NaN gaps.

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
                median fallback.  Ignored when ``None``.  Edge-pinned detections
                are already NaN (rejected at detection time), so they do not
                pollute the median.
            fallback: Constant lag in seconds used as last resort.

        Returns:
            Array of the same length as *pwbopt_s*, NaN-free when a finite
            value can be found through any of the three strategies.
        """
        result = pd.Series(np.asarray(pwbopt_s, dtype=float))
        result = result.bfill()

        if result.isna().any() and tlag_s_raw is not None:
            raw = np.asarray(tlag_s_raw, dtype=float)
            median_raw = np.nanmedian(raw) if np.any(~np.isnan(raw)) else np.nan
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
        out = Path(output_dir) if output_dir else None

        # X-axis: real timestamps when a parseable 'timestamp' column exists,
        # otherwise the integer period index (backward-compatible fallback).
        use_dates = ('timestamp' in results.columns
                     and pd.to_datetime(results['timestamp'], errors='coerce').notna().any())
        if use_dates:
            px = pd.to_datetime(results['timestamp'], errors='coerce').values
            # Bar width in matplotlib date units (days): 80% of the median spacing.
            _spacing = np.median(np.diff(mdates.date2num(px))) if len(px) > 1 else 1.0
            bar_full = float(_spacing) * 0.8
            xlabel = 'Time'
        else:
            px = np.arange(len(results))
            bar_full = 0.8
            xlabel = 'Period index'

        def _format_xaxis(ax):
            """Date formatting for time-series panels when timestamps are used."""
            if use_dates:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

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
            _format_xaxis(ax)

            # Panel 2: the lag ACTUALLY applied (final pre-filtered, gap-filled)
            # drawn as a solid black line ON TOP of the raw per-chunk detections
            # that PWBOPT accepted (S1/S2). The coloured dots are the *raw*
            # detections (the input), NOT the applied lag — only the black line
            # is removed from the data. Keeping the dots faint and small and the
            # final lag as a bold named line makes that unambiguous (a wide-HDI
            # raw detection that dips away from the black line was pre-filtered
            # out and replaced, so it was never applied).
            ax = axes[1]
            ax.axhline(0, color='#888888', linewidth=0.8, linestyle='-', zorder=1)
            for flag, lbl in (('S1_optimal', 'Raw detection · S1 reliable'),
                              ('S2_optimal', 'Raw detection · S2 carried')):
                mask = flag_std == flag
                ax.scatter(px[mask], tlag[mask], color=FLAG_COLORS[flag], s=20,
                           alpha=0.45, linewidths=0, zorder=3, label=lbl)
            if has_final_pf:
                final_pf = results[final_pf_col].values.astype(float)
                ax.plot(px, final_pf, color='black', linewidth=1.4, zorder=5,
                        label=f'APPLIED lag for flux ({final_pf_col})')
            if not np.isnan(mode_lag):
                ax.axhline(mode_lag, color='#9467bd', linewidth=1.2,
                           linestyle='-.', zorder=2,
                           label=f'mode = {mode_lag:.2f} s')
            ax.set_ylabel('Time lag (s)')
            ax.set_title('Applied lag = black line (final, gap-filled). '
                         'Coloured points are raw detections, NOT the applied '
                         'lag.')
            ax.legend(frameon=False, fontsize=8, ncol=2)
            ax.set_ylim(-lag_max_s - 0.5, lag_max_s + 0.5)
            _format_xaxis(ax)

            # Panel 3: HDI range
            ax = axes[2]
            ax.bar(px, hdi, width=bar_full, color='#aec7e8', edgecolor='none',
                   label='HDI range')
            ax.axhline(hdi_thresh, color='#2ca02c', linewidth=1.5, linestyle='--',
                       label=f'S1 threshold ({hdi_thresh} s)')
            if hdi_prefilter > 0:
                ax.axhline(hdi_prefilter, color='steelblue', linewidth=1.5,
                           linestyle=':',
                           label=f'Pre-filter threshold ({hdi_prefilter} s)')
            ax.set_ylabel('95% HDI range (s)')
            ax.set_title('Bootstrap uncertainty (HDI range) per period')
            ax.legend(frameon=False, fontsize=8)
            _format_xaxis(ax)

            # Panel 4: flag bars — standard (left) vs pre-filtered (right).
            # Iterate integer positions for .iloc; x-position comes from px so the
            # bars line up with the timestamp (or index) axis of the other panels.
            ax = axes[3]
            bar_w = bar_full / 2.0
            x_num = mdates.date2num(px) if use_dates else px
            flag_cols = [(flag_std_col, 0)]
            if has_pf:
                flag_cols.append((flag_pf_col, 1))
            for flag_col, offset in flag_cols:
                for i in range(len(results)):
                    flag = results[flag_col].iloc[i]
                    ax.bar(x_num[i] + (offset - 0.5) * bar_w, 1, bar_w,
                           color=FLAG_COLORS.get(flag, '#aaaaaa'), alpha=0.85)
            patches = [mpatches.Patch(color=c, label=f)
                       for f, c in FLAG_COLORS.items()]
            ax.legend(handles=patches, frameon=False, fontsize=8)
            ax.set_yticks([])
            ax.set_xlabel(xlabel)
            ax.set_title(
                'Flag per period: standard (left bar) vs. pre-filtered (right bar)'
                if has_pf else 'Flag per period'
            )
            _format_xaxis(ax)

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
                timestamp_col='timestamp' if use_dates else None,
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
        prog='python -m diive.flux.hires.lag_pwb',
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
    p.add_argument('--col-tsonic', required=True,
                   help='Column name for sonic temperature T_SONIC (required; '
                        'enables 4-combination RFlux v3.2.0 mode).')
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
    p.add_argument('--random-state', type=int, default=None,
                   help='Base seed for reproducible bootstrap results.')
    p.add_argument('--file-date-format', default=None,
                   help="strptime pattern for the leading datetime in each "
                        "filename, e.g. '%%Y%%m%%d-%%H%%M' for "
                        "'20210820-0930_rotated.txt'. Adds a 'timestamp' column "
                        "and uses dates as the summary-plot x-axis.")
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
        random_state=args.random_state,
        file_date_format=args.file_date_format,
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
