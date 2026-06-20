"""
Tests for random uncertainty estimation (RandomUncertaintyPAS20).

Part of the diive library: https://github.com/holukas/diive
"""
import numpy as np
import pandas as pd

import diive as dv


def _subset():
    df = dv.load_exampledata_parquet()
    df = df.loc[(df.index.year == 2013) & (df.index.month == 3)].copy()
    sub = df[['NEE_CUT_REF_orig', 'NEE_CUT_REF_f', 'Tair_f', 'VPD_f', 'Rg_f']].copy()
    sub['VPD_f'] = sub['VPD_f'] * 0.1  # hPa -> kPa (diive convention)
    return sub


def test_cumulative_uncertainty_is_quadrature():
    # The cumulative random uncertainty is the quadrature (root-sum-of-squares)
    # combination of the independent per-record random uncertainties:
    #   UNC_CUMULATIVE[k] = sqrt( sum_{i<=k} randunc_i^2 ).
    ru = dv.flux.RandomUncertaintyPAS20(
        _subset(), 'NEE_CUT_REF_orig', 'NEE_CUT_REF_f', 'Tair_f', 'VPD_f', 'Rg_f')
    ru.run()
    cum = ru.randunc_results_cumulatives
    randunc = ru.randunc_results[ru.randunccol]

    expected = np.sqrt((randunc.astype(float) ** 2).cumsum())
    np.testing.assert_allclose(cum['UNC_CUMULATIVE'].to_numpy(),
                               expected.to_numpy(), rtol=1e-9, atol=1e-9)
    # Monotonic non-decreasing (variance only ever accumulates).
    assert (cum['UNC_CUMULATIVE'].diff().dropna() >= -1e-9).all()
    # Bounds are the cumulative flux +/- 1 sigma.
    np.testing.assert_allclose(
        cum['FLUX+UNC'].to_numpy(),
        (cum['NEE_CUT_REF_f'] + cum['UNC_CUMULATIVE']).to_numpy())
    np.testing.assert_allclose(
        cum['FLUX-UNC'].to_numpy(),
        (cum['NEE_CUT_REF_f'] - cum['UNC_CUMULATIVE']).to_numpy())


def test_cumulative_uncertainty_nan_does_not_poison():
    # A single missing per-record uncertainty must not nullify every later
    # cumulative value (the old ufloat-cumsum poisoned the whole tail).
    ru = dv.flux.RandomUncertaintyPAS20(
        _subset(), 'NEE_CUT_REF_orig', 'NEE_CUT_REF_f', 'Tair_f', 'VPD_f', 'Rg_f')
    ru._calc_random_uncertainty()
    col = ru._randunc_results.columns.get_loc(ru.randunccol)
    ru._randunc_results.iloc[10, col] = np.nan
    ru._calc_cumulative_uncertainty_propagation()

    unc_cum = ru.randunc_results_cumulatives['UNC_CUMULATIVE']
    # Only the single injected record may be undefined; the tail stays defined.
    assert int(unc_cum.isna().sum()) <= 1
    assert pd.notna(unc_cum.iloc[-1])
    assert (unc_cum.dropna().diff().dropna() >= -1e-9).all()
