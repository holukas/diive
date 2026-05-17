"""
LAG_PWB: PRE-WHITENING WITH BLOCK-BOOTSTRAP TIME LAG DETECTION
==============================================================

Detect the time lag between a scalar gas concentration and the vertical wind
component using the pre-whitening with block-bootstrap (PWB) cross-correlation
procedure from Vitale et al. 2024.

The pre-whitening (PW) step is a direct Python port of the R function
tlag_detection() in the RFlux package (github.com/icos-etc/RFlux), including
the local-extrema decision logic, na.omit handling, and smoothing widths.
The block-bootstrap extension follows Section 2.2 of the paper.

References:
    Vitale D, Fratini G, Helfter C, Hortnagl L, et al. (2024) A pre-whitening
    with block-bootstrap cross-correlation procedure for temporal alignment of
    data sampled by eddy covariance systems. Environmental and Ecological
    Statistics 31:219-244. doi:10.1007/s10651-024-00615-9

Part of the diive library: https://github.com/holukas/diive
"""

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import correlate as _signal_correlate, lfilter
from scipy.stats import gaussian_kde, mode
from statsmodels.tsa.ar_model import ar_select_order

from diive.core.plotting.plotfuncs import default_format
from diive.core.plotting.styles import LightTheme as theme

# Smoothing widths are hardcoded in tlag_detection.R and must not be changed.
# R line 11: rollapply(cross_cov$acf, width=3, ...)   -> raw cross-covariance
# R line 35: rollapply(cross_cor$acf, width=13, ...)  -> pre-whitened CCF
_SMOOTH_WIDTH_CCF = 13
_SMOOTH_WIDTH_CCOV = 3


class PreWhiteningBootstrap:
    """
    Detect time lag between a scalar and vertical wind via pre-whitening with
    block-bootstrap cross-correlation (PWB), following Vitale et al. (2024).

    Unlike covariance maximisation (MaxCovariance), this method is robust for
    low-magnitude fluxes (e.g. CH4, N2O) where the cross-covariance function
    lacks a distinct peak. Pre-whitening removes serial correlation that inflates
    spurious CCF peaks, and block-bootstrapping quantifies detection uncertainty
    so unreliable lags can be flagged and replaced.

    Two complementary lag estimates are produced:

    - **tlag_pw** — Pre-whitening lag (PW): argmax of the smoothed pre-whitened
      CCF on the full data, refined by the R tlag_detection.R decision rules
      (tlag_opt). This is the fast, single-pass estimate.
    - **tlag_s** — Bootstrap lag (PWB): mode of lags detected across N_B
      block-bootstrap samples. More robust for low-magnitude fluxes.

    The 95% HDI of the bootstrap lag distribution quantifies uncertainty.
    Lags with HDI range < 0.5 s are considered reliable (S1 criterion).

    The plot() method reproduces the two R tlag_detection.R diagnostic panels
    plus a third panel for the PWB bootstrap distribution:

    - Panel 1 (left):  Pre-whitened CCF — grey stems, smoothed black line,
      significance bands, detected-lag marker  (R panel 1)
    - Panel 2 (middle): Raw cross-covariance — grey stems, smoothed cyan line,
      detected-lag marker  (R panel 2)
    - Panel 3 (right): Bootstrap lag histogram with 95% HDI  (PWB-specific)

    Attributes:
        df (DataFrame): Input high-frequency data.
        var_w (str): Column name for vertical wind component W.
        var_scalar (str): Column name for scalar gas concentration.
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
        self.hz = hz
        self.lag_max_s = lag_max_s
        self.n_bootstrap = n_bootstrap
        self.block_length_s = block_length_s
        self.segment_name = segment_name

        # Search window half-width in records (R: LAG.MAX = mfreq * 10)
        self._lag_max_records = int(round(lag_max_s * hz))
        # Block length in records (paper: L = 20 s, Section 2.2)
        self._block_length_records = int(round(block_length_s * hz))
        # Bootstrap CCF smoothing width: hz/2 + 1 timesteps (paper Section 2.2, eq. 6)
        self._smooth_width_bootstrap = hz // 2 + 1

        # --- Results populated by run() ---
        # PW results (from full-data pre-whitened CCF, matching tlag_detection.R)
        self._tlag_pw_records: int | None = None    # argmax of smoothed PW CCF
        self._tlag_opt_records: int | None = None   # after R decision rules (= tlag_pw)
        self._tlag_lmax: list | None = None         # local maxima lags in CCov window
        self._tlag_lmin: list | None = None         # local minima lags in CCov window
        self._corr_est: float | None = None         # un-smoothed CCF at tlag_opt
        self._n_eff: int | None = None              # full series length for Bartlett thresholds
        # PWB results (from block bootstrap)
        self._tlag_records: int | None = None       # mode of bootstrap lags
        self._hdi_lo_s: float | None = None
        self._hdi_hi_s: float | None = None
        self._bootstrap_lags: np.ndarray | None = None
        # Arrays kept for plotting
        self._ar_order: int | None = None
        self._lags_axis: np.ndarray | None = None
        self._raw_ccov: np.ndarray | None = None
        self._smooth_raw_ccov: np.ndarray | None = None
        self._pw_ccf: np.ndarray | None = None
        self._smooth_pw_ccf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties — PW results
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
        """Optimal PW lag in records (after R tlag_detection.R decision rules)."""
        if self._tlag_opt_records is None:
            raise RuntimeError("Call run() first.")
        return self._tlag_opt_records

    @property
    def tlag_opt_s(self) -> float:
        """Optimal PW lag in seconds."""
        return self.tlag_opt_records / self.hz

    @property
    def corr_est(self) -> float:
        """Un-smoothed CCF value at tlag_opt (R: corr_est)."""
        if self._corr_est is None:
            raise RuntimeError("Call run() first.")
        return self._corr_est

    @property
    def cv5pct(self) -> float:
        """5% Bartlett significance threshold: 1.96 / sqrt(N) (R: cv5pct)."""
        if self._n_eff is None:
            raise RuntimeError("Call run() first.")
        return 1.96 / np.sqrt(self._n_eff)

    @property
    def cv1pct(self) -> float:
        """1% Bartlett significance threshold: 2.57 / sqrt(N) (R: cv1pct)."""
        if self._n_eff is None:
            raise RuntimeError("Call run() first.")
        return 2.57 / np.sqrt(self._n_eff)

    # ------------------------------------------------------------------
    # Properties — PWB results
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
        """All key outputs as a dictionary (mirrors R return list)."""
        return {
            'segment_name': self.segment_name,
            # PW results (matching R tlag_detection.R return values)
            'tlag_pw_records': self.tlag_pw_records,
            'tlag_pw_s': self.tlag_pw_s,
            'opt_tlag_records': self.tlag_opt_records,
            'opt_tlag_s': self.tlag_opt_s,
            'tlag_lmax': self._tlag_lmax,
            'tlag_lmin': self._tlag_lmin,
            'corr_est': self.corr_est,
            'cv5pct': self.cv5pct,
            'cv1pct': self.cv1pct,
            'ar_order': self._ar_order,
            # PWB results (bootstrap extension, paper Section 2.2)
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
        Execute the full PWB pipeline.

        Follows the structure of tlag_detection.R with the block-bootstrap
        extension described in paper Section 2.2.

        Steps
        -----
        1. Raw cross-covariance on original data (used for local-extrema search
           around the PW peak). R lines 10-11.
        2. Pre-whitening: fit AR(p) to W, apply the same filter to both W and
           the scalar S to remove serial autocorrelation. R lines 13-32.
        3. Pre-whitened CCF on the na.omit-trimmed filtered series. R line 33.
        4. Smooth the PW CCF with a centered rolling mean (width=13). R line 35.
        5. Detect tlag_pw (argmax of smoothed PW CCF) and refine to tlag_opt
           via the four sequential if-rules from R lines 37-47. Compute corr_est
           (un-smoothed CCF at tlag_opt) for the significance test.
        6. Block bootstrap: draw N_B block samples from the pre-whitened series,
           compute and smooth the CCF for each, record the peak lag. Paper eq. 6.
        7. Summarise the N_B bootstrap lags as mode (tlag_s) + 95% HDI. Paper eq. 7.
        """
        w = self.df[self.var_w].values.astype(float)
        s = self.df[self.var_scalar].values.astype(float)

        # Lag axis in records: [-lag_max, ..., 0, ..., +lag_max]
        # Positive = scalar arrives later than wind (tube delay)
        self._lags_axis = np.arange(-self._lag_max_records, self._lag_max_records + 1)

        # ---- Step 1: raw cross-covariance on the original, unfiltered data ----
        # Computed here, BEFORE pre-whitening, so it reflects the original signal.
        # Used later in _compute_tlag_opt to search for local extrema near the
        # PW-detected peak (R lines 10-11).
        self._raw_ccov = self._compute_ccov(w, s)
        self._smooth_raw_ccov = self._smooth_series(self._raw_ccov, _SMOOTH_WIDTH_CCOV)

        # ---- Step 2: pre-whitening ----
        # Fit AR(p) to W only; apply the same filter coefficients to both W and S.
        # This removes the dominant autocorrelation structure from both series so
        # that the subsequent CCF is not dominated by spurious within-series peaks
        # (paper Section 2.1; R lines 28-31).
        phi, self._ar_order = self._fit_ar_model(w)
        x_pw = self._apply_ar_filter(phi, w)
        y_pw = self._apply_ar_filter(phi, s)

        # ---- Step 3: pre-whitened CCF ----
        # Leading NaN (from AR filter initialisation) are trimmed before computing
        # the correlation, matching R's na.action=na.omit inside ccf() (R line 33).
        self._pw_ccf = self._compute_ccf(x_pw, y_pw)

        # ---- Step 4: smooth the PW CCF ----
        # Centered rolling mean, width=13, matching R line 35.
        # Edge positions with < 13 neighbours remain NaN (min_periods=13), which
        # matches R's rollapply(..., fill=NA) behaviour.
        self._smooth_pw_ccf = self._smooth_series(self._pw_ccf, _SMOOTH_WIDTH_CCF)

        # Bartlett significance threshold denominator: R uses length(x) = full
        # array length including the leading NaN from the AR filter, not the
        # trimmed (na.omit) length.  Matches R line 53: 1.96/sqrt(length(x)).
        self._n_eff = len(x_pw)

        # ---- Step 5: PW lag detection and tlag_opt decision ----
        # Identifies tlag_pw (argmax of smoothed PW CCF), searches the raw CCov
        # for local extrema near that peak, and applies the four sequential
        # if-rules from R lines 37-47 to arrive at tlag_opt.
        self._tlag_pw_records, self._tlag_opt_records, \
            self._tlag_lmax, self._tlag_lmin, self._corr_est = \
            self._compute_tlag_opt()

        # ---- Step 6: block bootstrap ----
        # Replace the leading NaN (AR initialisation) with 0 so that block
        # boundaries never contain NaN.  Block 0 (which holds those zeroed
        # positions) is excluded from sampling to avoid any residual bias.
        x_pw0 = np.where(np.isnan(x_pw), 0.0, x_pw)
        y_pw0 = np.where(np.isnan(y_pw), 0.0, y_pw)
        self._bootstrap_lags = self._block_bootstrap(x_pw0, y_pw0)

        # ---- Step 7: mode + 95% HDI ----
        # Mode gives the most frequently detected lag across N_B bootstrap samples
        # (paper eq. 7).  The HDI is the shortest interval containing 95% of the
        # bootstrap distribution; its width is the uncertainty metric (Section 2.3).
        tlag_mode, hdi_lo, hdi_hi = self._mode_and_hdi(self._bootstrap_lags)
        self._tlag_records = int(tlag_mode)
        self._hdi_lo_s = hdi_lo
        self._hdi_hi_s = hdi_hi

    # ------------------------------------------------------------------
    # Private: pre-whitening  (R lines 13-32)
    # ------------------------------------------------------------------

    def _fit_ar_model(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Fit AR(p) model to x with AIC-selected order.

        The R script supports two modes (R lines 28-29):
            AIC=TRUE  -> ar.ols(x, aic=TRUE)              (AIC selection)
            AIC=FALSE -> ar.ols(x, aic=FALSE, order.max=10*log10(n))  (fixed order, R default)

        This implementation always uses AIC selection, which the paper recommends
        as it adapts the model complexity to the data.  The maximum candidate order
        is 10*log10(n), matching R's order.max formula in both cases.
        """
        x_clean = x[~np.isnan(x)]
        x_centered = x_clean - np.nanmean(x_clean)

        # Maximum candidate AR order: 10*log10(n), matching R: order.max=10*log(length(x),10)
        max_lag = min(int(10 * np.log10(len(x_centered))), 100)

        # ar_select_order performs an AIC grid search over orders 1..max_lag and
        # returns the order with the lowest AIC, equivalent to R's ar.ols(aic=TRUE).
        sel = ar_select_order(x_centered, maxlag=max_lag, ic='aic', old_names=False)
        model = sel.model.fit()

        # params[0] is the intercept; params[1:] are the AR coefficients phi_1..phi_p
        phi = np.asarray(model.params[1:])
        return phi, len(phi)

    def _apply_ar_filter(self, phi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Apply AR(p) filter to x to produce the pre-whitened series x_tilde.

        The filter computes the one-step-ahead prediction error:
            x_tilde[t] = x[t] - phi[0]*x[t-1] - ... - phi[p-1]*x[t-p]

        This matches R: stats::filter(x, filter=c(1,-phi), method="convolution", sides=1)
        from tlag_detection.R lines 30-31.

        scipy.signal.lfilter with b=[1, -phi_1, ..., -phi_p] and a=[1] is the
        direct Python equivalent of R's one-sided convolution filter.

        The first p output values are set to NaN because lfilter seeds the delay
        line with zeros (not actual past data), so those values are unreliable.
        R's stats::filter also marks the first p values as NA.  They are removed
        by na.omit inside _compute_ccf, matching R line 33.
        """
        x_centered = x - np.nanmean(x)
        if len(phi) == 0:
            # AR(0) selected: series is already white, no filtering needed
            return x_centered

        # Build FIR numerator: [1, -phi_1, -phi_2, ..., -phi_p]
        b = np.concatenate([[1.0], -phi])

        # Replace any input NaN with 0 before lfilter (NaN would propagate)
        x_filled = np.where(np.isnan(x_centered), 0.0, x_centered)
        x_tilde = lfilter(b, [1.0], x_filled)

        # First p values are contaminated by the zero initial conditions of lfilter.
        # Mark them NaN so that _compute_ccf trims them via na.omit (R line 33).
        x_tilde[:len(phi)] = np.nan
        return x_tilde

    # ------------------------------------------------------------------
    # Private: tlag_pw / tlag_opt decision logic  (R lines 37-47)
    # ------------------------------------------------------------------

    def _compute_tlag_opt(self) -> tuple[int, int, list, list, float]:
        """
        Detect tlag_pw and refine to tlag_opt.

        Direct port of tlag_detection.R lines 37-47.

        tlag_pw is the lag at the absolute maximum of the smoothed PW CCF.
        tlag_opt is intended to refine this estimate using local extrema of the
        raw cross-covariance near the PW peak.  However, the four sequential
        if-rules in R always end with Rule 4 overwriting Rules 2 and 3, so in
        practice tlag_opt == tlag_pw for all possible inputs.  The local extrema
        (tlag_lmax, tlag_lmin) are returned in the results dict to match R's
        return list, but they do not influence the detected lag.
        """
        # Index of the absolute maximum in the smoothed PW CCF (R: which.max(abs(scross_cor)))
        # R returns 1-based; Python's nanargmax returns 0-based.
        tl0 = int(np.nanargmax(np.abs(self._smooth_pw_ccf)))

        # Convert 0-based array index to lag in records.
        # R formula (1-based): tlag_pw = tl0 - LAG.MAX - 1
        # Python equivalent (0-based): tlag_pw = tl0 - lag_max
        tlag_pw = tl0 - self._lag_max_records

        # Search window: +-12 records around tl0 in the smoothed raw CCov.
        # R line 40: scross_cov[max(1, tl0-12) : min(tl0+12, LAG.MAX*2)]  (1-based inclusive)
        # Converted to Python 0-based exclusive slicing:
        #   start = max(0, tl0-12)      [same value because tl0_R = tl0_P + 1 cancels -1]
        #   end   = min(tl0+13, 2*lag_max)  [R caps at LAG.MAX*2, excluding the very last element]
        win_start = max(0, tl0 - 12)
        win_end = min(tl0 + 13, 2 * self._lag_max_records)
        window = self._smooth_raw_ccov[win_start:win_end]

        # Convert local-extrema positions within the window to lags in records.
        # Element at 0-based window index i has array index win_start+i,
        # which corresponds to lag = (win_start + i) - lag_max = i + offset.
        offset = win_start - self._lag_max_records
        tlag_lmax = self._local_extrema(window, 'max', offset)
        tlag_lmin = self._local_extrema(window, 'min', offset)

        # Four sequential if-rules from R lines 43-46 (NOT if-elif — exact port).
        # Rule 4 always fires when any extremum exists, overwriting Rules 2 and 3,
        # so tlag_opt == tlag_pw in every case.
        tlag_opt = tlag_pw                                        # Rule 1 default
        if len(tlag_lmax) == 0 and len(tlag_lmin) == 0:
            tlag_opt = tlag_pw                                    # Rule 1: no extrema found
        if len(tlag_lmax) == 1 and len(tlag_lmin) == 0:
            tlag_opt = tlag_lmax[0]                               # Rule 2 (overwritten by Rule 4)
        if len(tlag_lmax) == 0 and len(tlag_lmin) == 1:
            tlag_opt = tlag_lmin[0]                               # Rule 3 (overwritten by Rule 4)
        if len(tlag_lmax) >= 1 or len(tlag_lmin) >= 1:
            tlag_opt = tlag_pw                                    # Rule 4: always reverts to tlag_pw

        # Un-smoothed CCF value at tlag_opt, used for the significance test.
        # R line 47: corr_est <- cross_cor$acf[tlag_opt + LAG.MAX + 1]  (1-based index)
        # Python:    opt_idx  = tlag_opt + lag_max_records               (0-based index)
        opt_idx = tlag_opt + self._lag_max_records
        corr_est = float(self._pw_ccf[opt_idx]) if 0 <= opt_idx < len(self._pw_ccf) else 0.0

        return tlag_pw, tlag_opt, tlag_lmax, tlag_lmin, corr_est

    @staticmethod
    def _local_extrema(arr: np.ndarray, kind: str, offset: int) -> list[int]:
        """
        Find local maxima or minima in arr and return their lag values.

        Matches R tlag_detection.R lines 5-6:
            local_max <- function(x) which(x - shift(x,1) > 0 & x - shift(x,1,type='lead') > 0)
            local_min <- function(x) which(x - shift(x,1) < 0 & x - shift(x,1,type='lead') < 0)

        R's shift(x, 1) shifts values forward (lag): shift(x,1)[i] = x[i-1].
        R's shift(x, 1, type='lead') shifts values back: shift(x,1,type='lead')[i] = x[i+1].
        So a local maximum satisfies: x[i] > x[i-1] AND x[i] > x[i+1].

        Args:
            arr: Input 1-D array (window of smoothed cross-covariance).
            kind: 'max' for local maxima, 'min' for local minima.
            offset: Converts 0-based window index i to lag in records: lag = i + offset.

        Returns:
            List of lag values (in records) where local extrema occur.
        """
        result = []
        for i in range(1, len(arr) - 1):
            if kind == 'max':
                if arr[i] - arr[i - 1] > 0 and arr[i] - arr[i + 1] > 0:
                    result.append(i + offset)
            else:
                if arr[i] - arr[i - 1] < 0 and arr[i] - arr[i + 1] < 0:
                    result.append(i + offset)
        return result

    # ------------------------------------------------------------------
    # Private: block bootstrap  (paper Section 2.2, eq. 6)
    # ------------------------------------------------------------------

    def _block_bootstrap(self, x_pw: np.ndarray, y_pw: np.ndarray) -> np.ndarray:
        """
        Draw N_B block-bootstrap samples and return the detected lag per sample.

        Procedure (paper Section 2.2):
        1. Divide the pre-whitened series into n_blocks non-overlapping blocks
           of length L = block_length_records.
        2. Draw all N_B sets of block indices at once (vectorised, no Python loop).
        3. Assemble all N_B bootstrap series in one operation via fancy indexing.
        4. Compute the CCF for all N_B samples simultaneously via batch FFT
           (_batch_ccf_fft): O(N_B × N log N) instead of N_B × O(N²).
        5. Smooth all CCFs at once with a vectorised rolling mean (_smooth_rows):
           O(N_B × M) via cumulative sum, no pandas overhead.
        6. Find the peak lag for each bootstrap sample with a single np.nanargmax
           call across the entire (N_B × M) array.

        Block 0 (the very first block) is excluded from sampling because the AR
        filter sets its first p values to 0 (they were NaN before the caller
        replaced them), which would bias the CCF.

        Memory note: two (N_B × N) float64 arrays are allocated (~28 MB each for
        N_B=99, N=36000), plus one (N_B × n_fft) array for the FFT (~100 MB).
        All temporaries are released before returning.
        """
        n = len(x_pw)
        L = self._block_length_records    # block length in records
        n_blocks = n // L                 # number of complete blocks
        N_B = self.n_bootstrap

        # Skip block 0 (contaminated by AR filter zero-initialisation)
        valid_blocks = np.arange(1, n_blocks) if n_blocks > 1 else np.arange(n_blocks)

        # ---- Pre-compute block views: shape (n_blocks, L) ----
        # Avoids repeated slicing inside the loop; trailing records (< L) discarded.
        x_blocks = x_pw[:n_blocks * L].reshape(n_blocks, L)
        y_blocks = y_pw[:n_blocks * L].reshape(n_blocks, L)

        # ---- Draw all N_B sets of block indices at once: shape (N_B, n_blocks) ----
        all_chosen = np.random.choice(valid_blocks, size=(N_B, n_blocks), replace=True)

        # ---- Assemble all bootstrap series via fancy indexing: shape (N_B, N) ----
        # x_blocks[all_chosen] is (N_B, n_blocks, L); reshape collapses blocks × L -> N.
        x_boot = x_blocks[all_chosen].reshape(N_B, -1)
        y_boot = y_blocks[all_chosen].reshape(N_B, -1)

        # ---- Batch FFT CCF: shape (N_B, 2*lag_max+1) ----
        all_ccf = self._batch_ccf_fft(x_boot, y_boot)

        # ---- Batch smoothing: vectorised rolling mean, NaN at edges ----
        all_smooth = self._smooth_rows(all_ccf, self._smooth_width_bootstrap)

        # ---- Batch argmax across all bootstrap samples ----
        peak_indices = np.nanargmax(np.abs(all_smooth), axis=1)  # shape (N_B,)
        return peak_indices.astype(int) - self._lag_max_records

    def _batch_ccf_fft(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Batch FFT-based normalised cross-correlation for all N_B bootstrap samples.

        Replaces N_B sequential O(N²) np.correlate calls with a single batch of
        FFT operations that scales as O(N_B × N log N).

        For N=36000 and N_B=99 the FFT approach is roughly 2 000× faster than
        the direct convolution used by np.correlate(mode='full').

        Args:
            X: Pre-whitened wind bootstrap arrays, shape (N_B, N). Zero-filled,
               no NaN (caller replaced NaN with 0 before bootstrapping).
            Y: Pre-whitened scalar bootstrap arrays, shape (N_B, N). Same.

        Returns:
            Normalised CCF windows of shape (N_B, 2*lag_max+1), ordered
            [lag=-lag_max, ..., lag=0, ..., lag=+lag_max].  Same sign convention
            as _compute_ccf: positive lag = scalar delayed (tube delay).
        """
        N_B, N = X.shape
        lag_max = self._lag_max_records

        # Centre each bootstrap sample (remove per-row mean)
        X_c = X - X.mean(axis=1, keepdims=True)
        Y_c = Y - Y.mean(axis=1, keepdims=True)

        # FFT size: next power of 2 >= 2N-1 ensures linear (non-cyclic) correlation.
        # For N=36000: n_fft = 2^17 = 131072.
        n_fft = 1 << (2 * N - 2).bit_length()

        # Batch real FFT along axis=1: shape (N_B, n_fft//2 + 1) complex
        FX = np.fft.rfft(X_c, n=n_fft, axis=1)
        FY = np.fft.rfft(Y_c, n=n_fft, axis=1)

        # Cross-correlation via IFFT of the spectral product.
        # irfft(FY * conj(FX))[k] = sum_n Y[n]*X[n-k], matching np.correlate(y,x,'full').
        ccf_full = np.fft.irfft(FY * np.conj(FX), n=n_fft, axis=1)  # (N_B, n_fft)

        # Normalise to correlation coefficient in [-1, +1]
        norm = np.sqrt((X_c ** 2).sum(axis=1) * (Y_c ** 2).sum(axis=1))  # (N_B,)
        norm = np.where(norm == 0.0, 1.0, norm)
        ccf_full /= norm[:, np.newaxis]

        # Extract the lag window [-lag_max, +lag_max] from the FFT output layout:
        #   ccf_full[:, k]          = lag +k  (k = 0 .. lag_max)
        #   ccf_full[:, n_fft - k]  = lag -k  (k = 1 .. lag_max)
        # Concatenate negative lags (ascending) then non-negative lags.
        neg = ccf_full[:, n_fft - lag_max:]   # lags -lag_max .. -1  (N_B, lag_max)
        pos = ccf_full[:, :lag_max + 1]       # lags 0 .. +lag_max   (N_B, lag_max+1)
        return np.concatenate([neg, pos], axis=1)  # (N_B, 2*lag_max+1)

    # ------------------------------------------------------------------
    # Private: cross-correlation and cross-covariance
    # ------------------------------------------------------------------

    def _compute_ccf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Normalised cross-correlation for lags [-lag_max, +lag_max].

        Matches R: ccf(x, y, na.action=na.omit, lag.max=LAG.MAX, type="correlation")
        from tlag_detection.R line 33.

        Sign convention (positive = tube delay):
            np.correlate(y_c, x_c)[mid+k] = sum(y[t] * x[t-k])
            At positive k: y is shifted right relative to x, meaning y arrives
            k records later — the physical tube delay.
            This matches R's ccf(x=wind, y=scalar) convention, where positive lag
            means x (wind) leads y (scalar).

        na.omit handling:
            After the AR filter, both x and y have NaN at the same leading positions
            (first p records).  These are trimmed together before computing the
            correlation, exactly as R's na.action=na.omit does inside ccf().
        """
        # Remove leading NaN (AR filter initialisation) from both series simultaneously.
        # R: ccf(..., na.action=na.omit) trims any position where either x or y is NA.
        is_nan = np.isnan(x) | np.isnan(y)
        if np.any(is_nan):
            first_valid = int(np.argmax(~is_nan))  # index of first fully valid pair
            x = x[first_valid:]
            y = y[first_valid:]

        # Centre both series; replace any residual NaN with 0 (guard only)
        x_c = np.where(np.isnan(x - np.nanmean(x)), 0.0, x - np.nanmean(x))
        y_c = np.where(np.isnan(y - np.nanmean(y)), 0.0, y - np.nanmean(y))

        # Full cross-correlation: correlate(y, x) so that positive lag k = scalar delayed.
        # FFT-based (_signal_correlate method='fft') is O(N log N) vs O(N²) for np.correlate.
        full = _signal_correlate(y_c, x_c, mode='full', method='fft')

        # Normalise to correlation coefficient in [-1, +1]
        denom = np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2))
        if denom == 0:
            full[:] = 0.0
        else:
            full /= denom

        # Extract the central window [-lag_max, +lag_max]
        mid = len(full) // 2
        return full[mid - self._lag_max_records: mid + self._lag_max_records + 1]

    def _compute_ccov(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Un-normalised cross-covariance for lags [-lag_max, +lag_max].

        Matches R: ccf(x, y, lag.max=LAG.MAX, type="covariance")
        from tlag_detection.R line 10.

        Computed on the ORIGINAL (not pre-whitened) data.  The result is stored
        and later used in _compute_tlag_opt to search for local extrema of the
        raw signal near the PW-detected peak.

        The biased estimator (divide by N, not by N-k) is used, matching R's
        ccf() default.  Same positive-lag sign convention as _compute_ccf.
        """
        # Centre and fill NaN with 0 (NaN in raw data would propagate)
        x_c = np.where(np.isnan(x), 0.0, x - np.nanmean(x))
        y_c = np.where(np.isnan(y), 0.0, y - np.nanmean(y))

        # FFT-based: O(N log N) vs O(N²) for np.correlate.
        full = _signal_correlate(y_c, x_c, mode='full', method='fft')
        # Biased covariance: divide by N (not lag-dependent), matching R's ccf(type="covariance")
        n = max(len(x_c), len(y_c))
        full /= n

        mid = len(full) // 2
        return full[mid - self._lag_max_records: mid + self._lag_max_records + 1]

    @staticmethod
    def _smooth_series(series: np.ndarray, width: int) -> np.ndarray:
        """
        Centered rolling mean of given width.

        Matches R: rollapply(series, width=width, FUN="mean", fill=NA)
        from tlag_detection.R lines 11 (width=3) and 35 (width=13).

        Used only on the full-data PW CCF and raw CCov (called twice per period).
        min_periods=width ensures that edge positions where a full window of
        width neighbours is not available are left as NaN, exactly replicating
        R's fill=NA behaviour.  np.nanargmax in the callers then correctly
        skips these NaN edges when searching for the CCF peak.

        Note: NOT used in the bootstrap path — _smooth_rows handles that
        with a vectorised numpy cumsum that avoids per-iteration pandas overhead.
        """
        return (
            pd.Series(series)
            .rolling(window=width, center=True, min_periods=width)
            .mean()
            .values
        )

    @staticmethod
    def _smooth_rows(arr: np.ndarray, width: int) -> np.ndarray:
        """
        Vectorised centered rolling mean for every row of a 2-D array.

        Used inside _block_bootstrap to smooth all N_B CCF arrays at once
        with a single O(N_B × M) cumulative-sum pass — no pandas, no Python loop.

        Replicates the min_periods=width edge behaviour of _smooth_series:
        positions within half = width//2 records of either edge are set to NaN
        because a full window of width neighbours is not available there.

        Args:
            arr: Shape (N_B, M). No NaN expected (bootstrap arrays are zero-filled).
            width: Rolling-mean window width (must be >= 1).

        Returns:
            Shape (N_B, M) float64 array; NaN at the first and last half positions.
        """
        if width <= 1:
            return arr.copy()

        N_B, M = arr.shape
        half = width // 2

        # Padded cumulative sum: cs[:, j] = sum(arr[:, 0:j])  (cs[:, 0] = 0)
        cs = np.empty((N_B, M + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(arr, axis=1, out=cs[:, 1:])

        # Interior positions [half, M-half): full window of `width` values available.
        # result[:, i] = (cs[:, i+half+1] - cs[:, i-half]) / width
        # Vectorised: upper slice cs[:, width:] and lower cs[:, :M-width+1]
        # each have exactly M - 2*half = M - width + 1 elements.
        result = np.full((N_B, M), np.nan, dtype=np.float64)
        result[:, half:M - half] = (cs[:, width:] - cs[:, :M - width + 1]) / width
        return result

    # ------------------------------------------------------------------
    # Private: mode and HDI  (paper Section 2.2, eqs. 6-7)
    # ------------------------------------------------------------------

    def _mode_and_hdi(self, lags_records: np.ndarray) -> tuple[int, float, float]:
        """
        Summarise the N_B bootstrap lag distribution (paper eq. 7).

        The mode is the most frequently detected lag across the N_B bootstrap
        replicates and is the final PWB lag estimate (TL^PWB, paper eq. 6).

        The 95% HDI is the shortest interval that contains 95% of the bootstrap
        distribution.  Its width is the reliability metric: HDI < 0.5 s means
        the lag detection is considered reliable (S1 criterion, Section 2.3).
        """
        # Mode of the integer bootstrap lag distribution (in records)
        tlag_mode = int(mode(lags_records, keepdims=True).mode[0])
        # Convert lag distribution to seconds for the HDI
        lags_s = lags_records / self.hz
        hdi_lo, hdi_hi = self._hdi(lags_s, credible_mass=0.95)
        return tlag_mode, float(hdi_lo), float(hdi_hi)

    @staticmethod
    def _hdi(samples: np.ndarray, credible_mass: float = 0.95) -> tuple[float, float]:
        """
        Highest Density Interval (HDI): shortest interval containing credible_mass
        of the sample distribution.

        Algorithm: sort the samples, then slide a window of length
        n_included = floor(credible_mass * n) across the sorted array and pick
        the window with the smallest width.  O(n log n) due to the sort.
        No external Bayesian library required.
        """
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        n_included = int(np.floor(credible_mass * n))
        n_intervals = n - n_included
        if n_intervals <= 0:
            # All samples fit inside one interval
            return float(sorted_s[0]), float(sorted_s[-1])
        # Width of every candidate interval of length n_included
        widths = sorted_s[n_included:] - sorted_s[:n_intervals]
        min_idx = int(np.argmin(widths))
        return float(sorted_s[min_idx]), float(sorted_s[min_idx + n_included])

    # ------------------------------------------------------------------
    # Plotting  (Phase 2 — styling only, no computation)
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

        Panel 1 (left) — Pre-whitened CCF, matching R tlag_detection.R panel 1:
            Grey stems + smoothed black line (width=13) + ±1.96/sqrt(n) dotted
            and ±2.57/sqrt(n) dashed significance bands + red tlag_opt marker
            + significance annotation in title.

        Panel 2 (middle) — Raw cross-covariance, matching R panel 2:
            Grey stems + smoothed cyan line (width=3) + red tlag_opt marker.

        Panel 3 (right) — Bootstrap lag distribution (PWB-specific):
            Histogram of N_B detected lags + 95% HDI shading + mode marker.

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
        ax_ccf = fig.add_subplot(gs[0, 0])    # Panel 1: PW CCF  (R panel 1)
        ax_ccov = fig.add_subplot(gs[0, 1])   # Panel 2: raw CCov (R panel 2)
        ax_hist = fig.add_subplot(gs[0, 2])   # Panel 3: bootstrap histogram

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
        """
        Panel 1: pre-whitened CCF.

        Reproduces R tlag_detection.R plot panel 1 (top panel):
            - Grey stems:    R plot(..., type="h", col="grey68")
            - Smoothed line: R lines(..., col=1, lwd=2)
            - 5% threshold:  R abline(h=+-1.96/sqrt(n), col=4, lty=3)   dotted blue
            - 1% threshold:  R abline(h=+-2.57/sqrt(n), col=4, lty=2)   dashed blue
            - Lag marker:    R abline(v=tlag_opt, col="red", lty=3)
            - Annotation:    R mtext(...) with significance level (R lines 59-61)
        """
        lags_s = self._lags_axis / self.hz
        cv5 = self.cv5pct   # Bartlett 5% threshold: 1.96 / sqrt(N)
        cv1 = self.cv1pct   # Bartlett 1% threshold: 2.57 / sqrt(N)

        # Grey stems for each lag (R: type="h", col="grey68")
        ax.vlines(lags_s, 0, self._pw_ccf,
                  colors='#808080', linewidth=0.6, alpha=0.7)

        # Smoothed PW CCF overlaid in black (R: lines(..., col=1, lwd=2))
        ax.plot(lags_s, self._smooth_pw_ccf,
                color='black', linewidth=2, label='smoothed CCF')

        # Bartlett significance bands (R: abline lines 53-56)
        ax.axhline(cv5,  color='steelblue', linestyle=':', linewidth=1,
                   label=f'+/-1.96/sqrt(n)  (5%)')
        ax.axhline(-cv5, color='steelblue', linestyle=':', linewidth=1)
        ax.axhline(cv1,  color='steelblue', linestyle='--', linewidth=1,
                   label=f'+/-2.57/sqrt(n)  (1%)')
        ax.axhline(-cv1, color='steelblue', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.5)

        # Red vertical line at the detected lag (R: abline(v=tlag_opt, col="red", lty=3))
        ax.axvline(self.tlag_opt_s, color='red', linestyle=':', linewidth=1.5)

        # Significance annotation in panel title (R: mtext lines 59-61).
        # corr_est is the un-smoothed CCF at tlag_opt; compare against Bartlett thresholds.
        # Case 3 ("not significant") omits the lag position, matching R exactly.
        abs_est = abs(self._corr_est)
        if abs_est >= cv1:
            sig_txt = (f'Detected peak at {self.tlag_opt_s:.3f} s'
                       f'  stat. sign. at 0.01 level')
        elif abs_est >= cv5:
            sig_txt = (f'Detected peak at {self.tlag_opt_s:.3f} s'
                       f'  stat. sign. at 0.05 level')
        else:
            sig_txt = 'Detected peak not stat. significant'  # R: "Detected peak not stat. significant"

        ax.set_title(sig_txt, fontsize=9)
        default_format(ax=ax, ax_labels_fontsize=theme.AX_LABELS_FONTSIZE,
                       ax_xlabel_txt='lag', ax_ylabel_txt='cross-correlation',
                       txt_ylabel_units='s')
        ax.legend(frameon=False, fontsize=8)

    def _plot_raw_ccov(self, ax: plt.Axes):
        """
        Panel 2: raw cross-covariance (original, non-pre-whitened data).

        Reproduces R tlag_detection.R plot panel 2 (bottom panel):
            - Grey stems:    R plot(..., type="h", col="grey68")
            - Smoothed line: R lines(..., col="cyan")
            - Lag marker:    R abline(v=tlag_opt, col="red", lty=3)
            - Title:         R mtext("Optimal Time Lag at X timesteps")
        """
        lags_s = self._lags_axis / self.hz

        # Grey stems (R: type="h", col="grey68")
        ax.vlines(lags_s, 0, self._raw_ccov,
                  colors='#808080', linewidth=0.6, alpha=0.7)

        # Smoothed cross-covariance in cyan (R: lines(..., col="cyan"))
        ax.plot(lags_s, self._smooth_raw_ccov,
                color='cyan', linewidth=1.5, label='smoothed cross-cov')
        ax.axhline(0, color='black', linewidth=0.5)

        # Red vertical line at the detected lag (R: abline(v=tlag_opt, col="red", lty=3))
        ax.axvline(self.tlag_opt_s, color='red', linestyle=':', linewidth=1.5,
                   label=f'tlag_opt = {self.tlag_opt_s:.2f} s')

        # Panel title (R: mtext("Optimal Time Lag at X timesteps"))
        ax.set_title(f'Optimal time lag at {self.tlag_opt_s:.2f} s', fontsize=9)

        default_format(ax=ax, ax_labels_fontsize=theme.AX_LABELS_FONTSIZE,
                       ax_xlabel_txt='lag', ax_ylabel_txt='cross-covariance',
                       txt_ylabel_units='s')
        ax.legend(frameon=False, fontsize=8)

    def _plot_bootstrap_histogram(self, ax: plt.Axes):
        """
        Panel 3: bootstrap lag distribution.

        Not present in the R script; specific to the PWB extension (paper Section 2.2,
        analogous to Fig. 2 right column which shows the bootstrap CCF distribution).

        Shows the histogram of N_B detected lags, the 95% HDI shaded in blue,
        and the mode (final PWB lag estimate) as a red vertical line.
        """
        lags_s = self._bootstrap_lags / self.hz
        n_bins = max(10, self.n_bootstrap // 5)

        ax.hist(lags_s, bins=n_bins, color='#808080', alpha=0.7,
                edgecolor='none', label='bootstrap lags')

        # 95% HDI shading: shortest interval containing 95% of the bootstrap distribution
        ax.axvspan(self._hdi_lo_s, self._hdi_hi_s, alpha=0.35, color='steelblue',
                   label=f'95% HDI  [{self._hdi_lo_s:.2f}, {self._hdi_hi_s:.2f}] s')

        # Mode = final PWB lag estimate (TL^PWB, paper eq. 6)
        ax.axvline(self.tlag_s, color='red', linewidth=2,
                   label=f'mode = {self.tlag_s:.2f} s')

        # Reliability flag: HDI range < 0.5 s = S1 criterion (paper Section 2.3)
        reliability = 'reliable' if self.is_reliable else 'UNRELIABLE'
        ax.set_title(
            f'{self.segment_name} | {reliability} '
            f'(HDI range = {self.hdi_range_s:.2f} s)',
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
      (right) — using columns produced by ``flux_lag_pwbopt.py``.
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
        rng = np.random.default_rng(42)
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        panel_count = 0

        fig_h = n_rows * 7.0 / 2.54          # 7 cm per row, matching original FIG_H
        fig = plt.figure(figsize=(28 / 2.54, fig_h), facecolor='white')

        if title:
            fig.suptitle(title, fontsize=11, fontweight='bold', y=1.00)

        gs = gridspec.GridSpec(
            n_rows, 5, figure=fig,
            hspace=0.18, wspace=0.06,
            left=0.06, right=0.98,
            top=0.88, bottom=0.14,
            width_ratios=[5, 1, 0.25, 5, 1],
        )

        ax_master = None  # first scatter axis; all others share its x and y

        for row, gas in enumerate(gas_labels):
            cols = self.scalars[gas]
            col_a = cols['col_a']
            col_b = cols['col_b']
            is_bottom = (row == n_rows - 1)
            is_top = (row == 0)

            # Build four axes for this row.  Scatter panels share both x and y
            # with ax_master so that zooming/panning is synchronised across rows
            # and across the left/right scatter columns.
            if ax_master is None:
                ax_s_a = fig.add_subplot(gs[row, 0])
                ax_master = ax_s_a
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

                # Jittered scatter — small uniform noise so stacked points separate
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

                if self.ylim:
                    ax_s.set_ylim(self.ylim)

                self._style_scatter(ax_s, show_ylabel=show_ylabel,
                                    show_xlabel=is_bottom,
                                    use_dates=self._use_dates)
                self._style_kde(ax_k, show_xlabel=is_bottom)

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
            # Degenerate: all values identical — return a trivial distribution
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
