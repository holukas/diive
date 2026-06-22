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


def _joint_subset():
    """A month with a measured NEE, its gap-filled REF and the 16th/84th USTAR
    percentile scenario fluxes — the joint-uncertainty inputs."""
    df = dv.load_exampledata_parquet()
    df = df.loc[(df.index.year == 2014) & (df.index.month == 7)].copy()
    return df[['NEE_CUT_REF_orig', 'NEE_CUT_REF_f',
               'NEE_CUT_16_f', 'NEE_CUT_84_f']].copy()


def _with_randunc(df):
    """Attach a NEE_CUT_REF_RANDUNC column (computed) to the joint subset."""
    sub = dv.load_exampledata_parquet()
    sub = sub.loc[df.index].copy()
    work = sub[['NEE_CUT_REF_orig', 'NEE_CUT_REF_f', 'Tair_f', 'VPD_f', 'Rg_f']].copy()
    work['VPD_f'] = work['VPD_f'] * 0.1
    ru = dv.flux.RandomUncertaintyPAS20(
        work, 'NEE_CUT_REF_orig', 'NEE_CUT_REF_f', 'Tair_f', 'VPD_f', 'Rg_f')
    ru.run()
    df = df.copy()
    df['NEE_CUT_REF_RANDUNC'] = ru.randunc_series
    return df


def test_joint_uncertainty_faithful_formula():
    # Faithful ONEFlux compute_join: sqrt(rand^2 + ((p84-p16)/2)^2), per record.
    df = _with_randunc(_joint_subset())
    ju = dv.flux.JointUncertaintyPAS20(
        df, 'NEE_CUT_REF_RANDUNC', 'NEE_CUT_16_f', 'NEE_CUT_84_f',
        fluxgapfilledcol='NEE_CUT_REF_f', divisor=2.0)
    ju.run()

    # Default output name strips _RANDUNC -> _JOINTUNC.
    assert ju.jointunccol == 'NEE_CUT_REF_JOINTUNC'

    expected = np.sqrt(
        df['NEE_CUT_REF_RANDUNC'].astype(float) ** 2
        + ((df['NEE_CUT_84_f'] - df['NEE_CUT_16_f']) / 2.0) ** 2)
    np.testing.assert_allclose(ju.jointunc_series.to_numpy(),
                               expected.to_numpy(), rtol=1e-12, atol=1e-12,
                               equal_nan=True)
    # Joint uncertainty is never below the random component (quadrature add).
    mask = ju.jointunc_series.notna() & df['NEE_CUT_REF_RANDUNC'].notna()
    assert (ju.jointunc_series[mask]
            >= df['NEE_CUT_REF_RANDUNC'][mask] - 1e-9).all()


def test_joint_uncertainty_pure_function_iqr_divisor():
    # The pure function honours the divisor (1.349 for the LE/H 25th/75th IQR).
    idx = pd.date_range('2020-01-01', periods=5, freq='30min')
    rand = pd.Series([1.0, 2.0, 0.0, np.nan, 3.0], index=idx)
    lower = pd.Series([0.0, 1.0, 2.0, 1.0, 1.0], index=idx)
    upper = pd.Series([1.349, 2.349, 2.0, 2.0, np.nan], index=idx)
    out = dv.flux.joint_uncertainty_pas20(rand, lower, upper,
                                          divisor=dv.flux.lowres.uncertainty.JOINT_DIVISOR_IQR)
    # row0: sqrt(1 + (1.349/1.349)^2) = sqrt(2)
    assert out.iloc[0] == np.sqrt(2.0)
    # row2: rand 0 -> joint == scenario term ((2-2)/1.349 = 0) -> 0
    assert out.iloc[2] == 0.0
    # NaN in any input propagates to NaN (ONEFlux INVALID_VALUE behaviour).
    assert np.isnan(out.iloc[3])  # rand NaN
    assert np.isnan(out.iloc[4])  # upper NaN


def test_joint_cumulative_components():
    # Cumulative random part is quadrature; scenario part is the running spread
    # of the cumulative scenario sums; total is their quadrature combination.
    df = _with_randunc(_joint_subset())
    ju = dv.flux.JointUncertaintyPAS20(
        df, 'NEE_CUT_REF_RANDUNC', 'NEE_CUT_16_f', 'NEE_CUT_84_f',
        fluxgapfilledcol='NEE_CUT_REF_f', divisor=2.0)
    ju.run()
    cum = ju.jointunc_results_cumulatives

    flux = df['NEE_CUT_REF_f'].astype(float)
    rand = df['NEE_CUT_REF_RANDUNC'].astype(float)
    exp_random = np.sqrt((rand ** 2).where(flux.notna()).cumsum())
    exp_scen = (df['NEE_CUT_84_f'].cumsum() - df['NEE_CUT_16_f'].cumsum()) / 2.0
    np.testing.assert_allclose(cum['UNC_RANDOM_CUMULATIVE'].to_numpy(),
                               exp_random.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(cum['UNC_SCENARIO_CUMULATIVE'].to_numpy(),
                               exp_scen.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(
        cum['UNC_CUMULATIVE'].to_numpy(),
        np.sqrt(exp_random ** 2 + exp_scen ** 2).to_numpy(),
        rtol=1e-9, atol=1e-9, equal_nan=True)
