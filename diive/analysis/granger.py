"""
GRANGER CAUSALITY: PREDICTIVE CAUSALITY TESTING
================================================

Test whether one time series helps predict another (Granger causality).

Part of the diive library: https://github.com/holukas/diive
"""

import pandas as pd
from pandas import Series, DataFrame
from statsmodels.tsa.stattools import grangercausalitytests

from diive.core.utils.console import console as _console


class GrangerCausality:
    """
    Test if one time series Granger-causes another.

    Granger causality tests whether past values of variable X improve predictions
    of variable Y beyond what Y's own past values provide. This is predictive
    causality, not true causality (correlation or reverse causation possible).

    Parameters:
        x: Time series that may cause y (pandas Series)
        y: Time series potentially caused by x (pandas Series)
        max_lag: Maximum lag order to test (default: 5)
        verbose: Print detailed results (default: True)

    Example:
        See `examples/analysis/analysis_granger.py` for complete examples.

        Quick start:

        >>> gc = GrangerCausality(x=radiation, y=co2_flux, max_lag=10)
        >>> gc.report()  # Print results
        >>> results = gc.results  # Access raw results
    """

    def __init__(self, x: Series, y: Series, max_lag: int = 5, verbose: bool = True):
        """Initialize Granger causality test."""
        self.x = x.copy().dropna()
        self.y = y.copy().dropna()
        self.max_lag = max_lag
        self.verbose = verbose

        # Align series by index
        common_idx = self.x.index.intersection(self.y.index)
        self.x = self.x.loc[common_idx]
        self.y = self.y.loc[common_idx]

        if len(self.x) < max_lag + 2:
            raise ValueError(f"Need at least {max_lag + 2} records, got {len(self.x)}")

        # Prepare data: combine as [y, x] for statsmodels
        self.data = pd.DataFrame({
            'y': self.y,
            'x': self.x
        }).dropna()

        # Run test
        self._run_test()

    def _run_test(self):
        """Run Granger causality test across lags."""
        try:
            # Note: statsmodels deprecated the `verbose` kwarg (removed in 0.15)
            # and the test is silent by default; we don't pass it.
            self.results = grangercausalitytests(
                self.data[['y', 'x']],
                self.max_lag,
            )
        except Exception as e:
            raise ValueError(f"Granger causality test failed: {e}")

    def p_values(self) -> DataFrame:
        """
        Extract p-values for each lag.

        Returns:
            DataFrame with p-values for each test (Lag, Ssr_ftest, Ssr_chi2test, etc.)
        """
        p_vals = []
        for lag in range(1, self.max_lag + 1):
            res_dict = self.results[lag][0]
            # Extract p-value from F-test (Ssr_ftest)
            # res_dict['ssr_ftest'] is a tuple: (test_stat, p_value, dof_num, dof_denom)
            p_val = res_dict['ssr_ftest'][1]
            p_vals.append({'Lag': lag, 'p_value': p_val})
        return pd.DataFrame(p_vals)

    def significant_lag(self, alpha: float = 0.05) -> int or None:
        """
        Find first lag where X Granger-causes Y (p < alpha).

        Args:
            alpha: Significance level (default: 0.05)

        Returns:
            Lag order if significant, None otherwise
        """
        p_df = self.p_values()
        sig = p_df[p_df['p_value'] < alpha]
        return sig['Lag'].iloc[0] if len(sig) > 0 else None

    def report(self, alpha: float = 0.05):
        """
        Print summary of Granger causality results.

        Interpretation: A p-value < alpha at a given lag means past values of X
        significantly improve predictions of Y beyond what Y's own past provides.
        """
        _console.print(f"\n{'=' * 70}")
        _console.print(f"GRANGER CAUSALITY TEST")
        _console.print(f"{'=' * 70}")
        _console.print(f"X variable: {self.x.name}")
        _console.print(f"Y variable: {self.y.name}")
        _console.print(f"Records: {len(self.data)}")
        _console.print(f"Max lag: {self.max_lag}")
        _console.print(f"Significance level: {alpha}")
        _console.print(f"{'=' * 70}\n")

        p_df = self.p_values()
        _console.print("p-values by lag:")
        for _, row in p_df.iterrows():
            sig_marker = "*" if row['p_value'] < alpha else " "
            _console.print(f"  Lag {int(row['Lag'])}: p={row['p_value']:.4f} {sig_marker}")

        sig_lag = self.significant_lag(alpha)
        if sig_lag:
            _console.print(f"\n[+] X GRANGER-CAUSES Y at lag {sig_lag} (p < {alpha})")
        else:
            _console.print(f"\n[-] X does NOT Granger-cause Y (all p >= {alpha})")

        _console.print(f"{'=' * 70}\n")
