"""
Tests for the self-heating (SCOP) physics.

Part of the diive library: https://github.com/holukas/diive
"""
import unittest

import numpy as np
import pandas as pd


def _physics(rho_v_value, method):
    """One ScopPhysics run on a small synthetic record, everything but humidity fixed."""
    from diive.flux.lowres.selfheating import ScopPhysics
    n = 480  # 10 days at 30 min
    idx = pd.date_range('2023-06-01 00:15', periods=n, freq='30min', name='TIMESTAMP_MIDDLE')
    hr = idx.hour + idx.minute / 60
    ta = pd.Series(15 + 8 * np.sin(2 * np.pi * (hr - 9) / 24), index=idx)
    physics = ScopPhysics(
        ta=ta,
        gas_density=pd.Series(1.7e7, index=idx),      # umol m-3
        rho_a=pd.Series(1.2, index=idx),              # kg m-3
        rho_v=pd.Series(rho_v_value, index=idx),      # kg m-3
        u=pd.Series(2.5, index=idx),                  # m s-1
        c_p=pd.Series(1005.0, index=idx),             # J K-1 kg-1
        ustar=pd.Series(0.4, index=idx),              # m s-1
        lat=47.478333, lon=8.364389, utc_offset=1,
    )
    physics.run(correction_method_base=method, gapfill=False)
    return physics.fct_unsc


class TestWaterVapourDilutionIsAppliedByEveryMethod(unittest.TestCase):
    """BUR08 must carry the (1 + 1.6077 rho_v/rho_d) factor, like BUR06 and JAR09.

    In Burba et al. (2008) the instrument-surface heat fluxes are *added to* the
    ambient sensible heat flux (Method 4) and the total enters the WPL equation,
    whose sensible-heat term carries that factor; their poster states the
    already-corrected form used here verbatim. BUR08 used to omit it, so choosing
    a correction_method_base silently changed two things at once: the
    surface-temperature model and whether the factor applied at all.
    """

    RHO_V = 0.012  # kg m-3, ~12 g m-3, a normal summer humidity

    def _expected_factor(self):
        rho_d = 1.2 - self.RHO_V  # dry air density = rho_a - rho_v
        return 1 + 1.6077 * (self.RHO_V / rho_d)

    def test_every_method_scales_with_humidity_by_the_same_factor(self):
        for method in ('BUR08', 'BUR06', 'JAR09'):
            with self.subTest(method=method):
                dry = _physics(0.0, method)
                humid = _physics(self.RHO_V, method)
                ratio = (humid / dry).dropna()
                self.assertGreater(len(ratio), 0)
                np.testing.assert_allclose(ratio.to_numpy(),
                                           self._expected_factor(),
                                           rtol=1e-9)

    def test_the_factor_is_not_negligible(self):
        # ~1% at ordinary humidity - small, but systematic and one-signed.
        self.assertGreater(self._expected_factor(), 1.015)


if __name__ == '__main__':
    unittest.main()
