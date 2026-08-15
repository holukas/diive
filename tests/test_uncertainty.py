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


def test_joint_cumulative_scenario_masked_to_available_flux():
    # The two cumulative scenario sums must skip the same records. Unmasked,
    # skipna cumsum let each percentile accumulate over its own record set: a
    # record with only one percentile (or with no flux to contribute to the
    # cumulative flux the band brackets) shifted one sum and not the other, and
    # the "spread" could even turn negative.
    idx = pd.date_range('2020-06-01 00:30', periods=5, freq='30min')
    df = pd.DataFrame(
        {'NEE_f': [1.0, 1.0, np.nan, 1.0, 1.0],  # record 2: no flux
         'NEE_16': [2.0, 2.0, 2.0, 2.0, 2.0],
         'NEE_84': [3.0, np.nan, 3.0, 3.0, 3.0],  # record 1: upper missing
         'NEE_RANDUNC': [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=idx)
    ju = dv.flux.JointUncertaintyPAS20(
        df, 'NEE_RANDUNC', 'NEE_16', 'NEE_84',
        fluxgapfilledcol='NEE_f', divisor=2.0)
    ju.run()
    scen = ju.jointunc_results_cumulatives['UNC_SCENARIO_CUMULATIVE']

    # Records 1 and 2 contribute to neither sum, so they are undefined (as the
    # random term and the cumulative flux already are at such records) and the
    # spread grows by (3-2)/2 per contributing record without poisoning the tail.
    # Unmasked this read [0.5, nan, 0.0, 0.5, 1.0]: record 2 entered both sums
    # while its flux did not, and record 1 entered only the lower sum, so the
    # spread came out too small from there on (with a narrower percentile band it
    # turns negative).
    np.testing.assert_allclose(scen.to_numpy(), [0.5, np.nan, np.nan, 1.0, 1.5],
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    assert int(scen.isna().sum()) == 2
    # A spread is never negative and, with both percentiles ordered, never shrinks.
    assert (scen.dropna() >= 0).all()
    assert (scen.dropna().diff().dropna() >= -1e-12).all()


# --- Random uncertainty method 4: symmetric neighbour window -------------------
# Method 4 (diive extension, not ONEFlux) takes the median uncertainty of the
# fluxes closest in magnitude. The neighbour slice used an exclusive stop of
# `cur_ix + 5`, which reached only 4 records above the current flux against 5
# below, biasing the median toward the lower fluxes.

def _method4_case():
    """21 records on an ascending flux ladder; only the middle one needs method 4.

    Uncertainty is 1.0 below the middle record and 100.0 above it, so a window
    that is short one record above returns a visibly different median.
    """
    n = 21
    idx = pd.date_range('2020-06-01 00:30', periods=n, freq='30min')
    df = pd.DataFrame({'NEE': np.nan,
                       'NEE_f': np.arange(n, dtype=float),
                       'TA': 10.0, 'VPD': 0.5, 'SW_IN': 100.0}, index=idx)
    ru = dv.flux.RandomUncertaintyPAS20(df, 'NEE', 'NEE_f', 'TA', 'VPD', 'SW_IN')
    ru._randunc_results['WINDOW_N_VALS_METHOD4'] = np.nan
    randunc = np.concatenate([np.full(10, 1.0), [np.nan], np.full(10, 100.0)])
    ru._randunc_results[ru.randunccol] = randunc
    return ru


def test_method4_neighbour_window_is_symmetric():
    ru = _method4_case()
    ru._method4()
    res = ru.randunc_results

    # 5 records below (1.0) and 5 above (100.0) -> median of ten values.
    assert res['WINDOW_N_VALS_METHOD4'].iloc[10] == 10
    assert res[ru.randunccol].iloc[10] == 50.5
    # Records that already had an uncertainty are untouched.
    assert res[ru.randunccol].iloc[9] == 1.0
    assert res[ru.randunccol].iloc[11] == 100.0


# --- MDS window: trimmed, not clipped -----------------------------------------
# The window positions used to be clipped into range (`np.clip(w, 0, n-1)`), so
# near the start of a record every out-of-range offset collapsed onto record 0
# and that one value entered the mean, the SD and the count hundreds of times.
# ONEFlux narrows the window bounds instead (`common.c:2525-2533`) and its
# diurnal method skips out-of-range positions outright (`:2630`).

_NPERDAY = 48


def _mds_synthetic():
    """40-day half-hourly record with a plausible diurnal + seasonal signal."""
    n = 40 * _NPERDAY
    t = np.arange(n)
    hr = (t % _NPERDAY) / 2.0
    swin = np.clip(600 * np.sin(2 * np.pi * (hr - 6) / 24), 0, None)
    ta = 10 + 8 * np.sin(2 * np.pi * (t / _NPERDAY) / 365) + 3 * np.sin(2 * np.pi * (hr - 9) / 24)
    vpd = np.clip(0.05 * swin + 0.3 * ta, 0, None)
    series = 0.02 * swin - 0.5 * ta + np.random.RandomState(0).normal(0, 0.5, n)
    return n, hr, swin, ta, vpd, series


def _mds_fill(gap_position):
    from diive.gapfilling.similarity import mds_gapfill_cascade
    n, hr, swin, ta, vpd, series = _mds_synthetic()
    tofill = series.copy()
    tofill[gap_position] = np.nan
    return n, mds_gapfill_cascade(tofill, swin, ta, vpd, hr, _NPERDAY)


def test_mds_count_never_exceeds_the_records_in_range():
    # A gap at position 2 reported 453 contributing records from a window that
    # cannot hold that many - they were duplicates of record 0.
    for pos in (2, 960, 40 * _NPERDAY - 3):
        n, res = _mds_fill(pos)
        half = res['time_window'][pos] * _NPERDAY / 2.0
        # Distinct positions the window can reach once it is trimmed to the record.
        in_range = min(n, int(pos + half)) - max(0, int(pos - half))
        assert int(res['count'][pos]) <= in_range, f"position {pos}"


def test_mds_edge_gap_does_not_reuse_the_edge_value():
    # Duplicates carry no spread, so clipping also collapsed the SD.
    _, res = _mds_fill(2)
    assert res['sd'][2] > 0.55


def test_mds_interior_gap_is_unaffected():
    # Interior windows never left the record, so nothing there may move.
    _, res = _mds_fill(960)
    assert np.isfinite(res['filled'][960])
    assert int(res['count'][960]) == 271
