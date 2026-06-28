import unittest

from diive.analysis.histogram import Histogram


class TestAnalyses(unittest.TestCase):

    def test_percentiles(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.quantiles import percentiles101
        df = load_exampledata_parquet()
        percentiles_df = percentiles101(series=df['Tair_f'], showplot=False, verbose=True)
        self.assertEqual(len(percentiles_df.columns), 2)
        self.assertEqual(len(percentiles_df.index), 101)
        self.assertEqual(percentiles_df.sum().sum(), 5521.9898)

    def test_gapfinder(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.gapfinder import GapFinder
        data_df = load_exampledata_parquet()
        series = data_df['NEE_CUT_REF_orig']
        gf = GapFinder(series=series, sort_results=True)
        gapfinder_df = gf.results
        self.assertEqual(len(gapfinder_df.columns), 4)
        self.assertEqual(len(gapfinder_df.index), 15602)
        self.assertEqual(gapfinder_df.iloc[0]['GAP_LENGTH'], 2633)
        self.assertEqual(gapfinder_df.iloc[1]['GAP_LENGTH'], 468)
        self.assertEqual(gapfinder_df.iloc[-1]['GAP_LENGTH'], 1)
        self.assertEqual(gapfinder_df['GAP_LENGTH'].sum(), 117099)

        # gap_at: a timestamp inside a gap returns that gap; outside returns the
        # nearest; None when there are no gaps.
        longest = gapfinder_df.iloc[0]
        mid = longest['GAP_START'] + (longest['GAP_END'] - longest['GAP_START']) / 2
        hit = gf.gap_at(mid)
        self.assertEqual(hit['GAP_START'], longest['GAP_START'])
        self.assertEqual(hit['GAP_END'], longest['GAP_END'])
        # A tz-aware timestamp is accepted (reduced to tz-naive wall time).
        import pandas as pd
        hit_tz = gf.gap_at(pd.Timestamp(mid).tz_localize('UTC'))
        self.assertEqual(hit_tz['GAP_START'], longest['GAP_START'])
        # No gaps -> None.
        from diive.analysis.gapfinder import GapFinder
        nogaps = GapFinder(series=series.fillna(0.0))
        self.assertIsNone(nogaps.gap_at(series.index[0]))

    def test_rank_drivers(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.correlation import rank_drivers
        df = load_exampledata_parquet()

        res = rank_drivers(df, target='NEE_CUT_REF_f', method='pearson', max_lag=0)
        self.assertListEqual(list(res.columns),
                             ['DRIVER', 'CORR', 'ABS_CORR', 'BEST_LAG', 'N'])
        self.assertNotIn('NEE_CUT_REF_f', res['DRIVER'].tolist())  # target excluded
        # Sorted by |corr| descending.
        vals = res['ABS_CORR'].to_numpy()
        self.assertTrue((vals[:-1] >= vals[1:]).all())
        # Strongest driver is strongly correlated; lags are 0 with no scan.
        self.assertGreater(res.iloc[0]['ABS_CORR'], 0.5)
        self.assertTrue((res['BEST_LAG'] == 0).all())

        # Lag scan: best lag stays within the scanned window; explicit features.
        res2 = rank_drivers(df, target='NEE_CUT_REF_f', method='pearson', max_lag=4,
                            features=['Tair_f', 'VPD_f', 'Rg_f'])
        self.assertEqual(len(res2), 3)
        self.assertLessEqual(int(res2['BEST_LAG'].abs().max()), 4)

        # Validation.
        with self.assertRaises(ValueError):
            rank_drivers(df, target='NOT_A_COLUMN')
        with self.assertRaises(ValueError):
            rank_drivers(df, target='NEE_CUT_REF_f', method='kendall')

    def test_spectrogram(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.harmonic import spectrogram
        nee = load_exampledata_parquet()['NEE_CUT_REF_f'].loc['2015-01-01':'2015-12-31']
        spec = spectrogram(nee, nperseg=512, noverlap=256)
        self.assertEqual(set(spec), {'frequencies', 'times', 'power', 'power_db'})
        # power is 2D [n_frequencies, n_times]; axes line up with the arrays.
        self.assertEqual(spec['power'].shape,
                         (len(spec['frequencies']), len(spec['times'])))
        self.assertEqual(spec['power_db'].shape, spec['power'].shape)
        # nperseg is clamped to the series length (no crash on short input).
        short = spectrogram(nee.iloc[:100], nperseg=512)
        self.assertGreater(short['power'].shape[1], 0)

    def test_seasonal_trend_decomposition(self):
        import diive as dv
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.seasonaltrend import SeasonalTrendDecomposition
        daily = dv.times.resample_to_daily_agg(
            load_exampledata_parquet()['Tair_f'], agg='mean').dropna()

        # STL — regression test: it used to always raise on real data (period was
        # never passed to statsmodels, and fit() got an unsupported `weights`).
        std = SeasonalTrendDecomposition(
            daily, method='stl', seasonal_period=365,
            robust=False, seasonal_jump=12, trend_jump=12)
        tr, se, re = std.trend, std.seasonal, std.residual
        recon = (tr + se + re).dropna()
        # Additive components reconstruct the input.
        self.assertLess((recon - daily.loc[recon.index]).abs().max(), 1e-6)
        self.assertGreater(std.seasonality_strength, 0.5)  # strong annual cycle

        # Classical and harmonic methods also run and return aligned components.
        for method in ('classical', 'harmonic'):
            s = SeasonalTrendDecomposition(daily, method=method, seasonal_period=365)
            self.assertEqual(len(s.seasonal), len(daily))

    def test_gapstats_gap_at(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.gapfinder import GapStats
        series = load_exampledata_parquet()['NEE_CUT_REF_orig']
        gs = GapStats(series=series, long_gap_records=48)
        longest = gs.long_gaps.iloc[0]
        hit = gs.gap_at(longest['GAP_START'])
        self.assertEqual(hit['GAP_START'], longest['GAP_START'])
        # GapStats rows carry YEAR / MONTH enrichment.
        self.assertIn('YEAR', hit.index)
        self.assertIn('MONTH', hit.index)

    def test_sorting_bins_method(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis.decoupling import SortingBinsMethod
        vpd_col = 'VPD_f'  # Vapor pressure deficit
        ta_col = 'Tair_f'  # Air temperature
        swin_col = 'Rg_f'  # Radiation used to detect daytime data
        # Load 10-year dataset of half-hourly measurements
        df = load_exampledata_parquet()
        # Keep data between June and September
        df = df.loc[(df.index.month >= 6) & (df.index.month <= 9)].copy()
        # Keep daytime data (radiation > 50 W m-2) and data when air temperatures was > 5°C
        daytime_locs = (df[swin_col] > 50) & (df[ta_col] > 0)
        df = df[daytime_locs].copy()
        # Rename variables
        rename_dict = {
            ta_col: 'air_temperature',
            vpd_col: 'vapor_pressure_deficit',
            swin_col: 'short-wave_incoming_radiation'
        }
        df = df.rename(columns=rename_dict, inplace=False)
        # Use new column names
        ta_col = 'air_temperature'
        vpd_col = 'vapor_pressure_deficit'
        swin_col = 'short-wave_incoming_radiation'
        # Make subset
        df = df[[ta_col, vpd_col, swin_col]].copy()
        sbm = SortingBinsMethod(df=df,
                                zvar=ta_col,
                                xvar=swin_col,
                                yvar=vpd_col,
                                n_bins_z=5,
                                n_bins_x=10,
                                conversion=None,
                                agg='median')
        sbm.calcbins()
        sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True)
        binmedians = sbm.get_binaggs()
        keys = []
        for group_key, group_df in binmedians.items():
            keys.append(group_key)
        self.assertEqual(len(keys), 5)
        self.assertEqual(len(binmedians['20.9'].columns), 13)
        self.assertEqual(len(binmedians['20.9'].index), 10)
        self.assertEqual(binmedians['20.9'].drop('group_short-wave_incoming_radiation', axis=1).sum().sum(),
                         25880.624680000004)

    def test_daily_correlation(self):
        from diive.configs.exampledata import load_exampledata_parquet
        from diive.analysis import daily_correlation
        from diive.variables import potrad
        data_df = load_exampledata_parquet()
        # Use only year 2022
        data_df = data_df.loc[data_df.index.year == 2022].copy()
        # Observed radiation time series for example (1) and (2)
        rg_series = data_df['Rg_f'].copy()
        # Observed air temperature time series for example (2)
        ta_series = data_df['Tair_f'].copy()
        # Observed net ecosystem exhange of CO2 time series from eddy covariance measurements for example (3)
        nee_series = data_df['NEE_CUT_REF_f'].copy()
        # Calculate potential radiation SW_IN_POT to use as reference
        reference = potrad(timestamp_index=rg_series.index,
                           lat=47.286417,
                           lon=7.733750,
                           utc_offset=1)
        # Calculate daily correlation between Rg_f and SW_IN_POT
        daycorrs = daily_correlation(s1=rg_series, s2=reference, mincorr=0.8).result
        self.assertEqual(daycorrs.sum(), 337.3189145385522)
        # Calculate daily correlation between Tair_f and NEE_CUT_REF_f
        daycorrs = daily_correlation(s1=ta_series, s2=nee_series).result
        self.assertEqual(daycorrs.sum(), -167.25042524807637)
        self.assertEqual(daycorrs.min(), -0.9450031804629302)
        self.assertEqual(daycorrs.max(), 0.7109706199504967)


    def test_histogram(self):
        from diive.configs.exampledata import load_exampledata_parquet
        data_df = load_exampledata_parquet()
        series = data_df['NEE_CUT_REF_f'].copy()

        hist = Histogram(series=series, method='n_bins', n_bins=10, ignore_fringe_bins=None)
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -40.811)
        self.assertEqual(bin_starts.mean(), -11.065549999999998)
        self.assertEqual(bin_starts.count(), 10)
        checkix = results.index[results['BIN_START_INCL'] == -1.1503999999999976].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 112210)
        self.assertEqual(hist.peakbins, [-1.1503999999999976, -7.7605, -14.3706, 5.459699999999998, -20.9807])

        hist = Histogram(series=series, method='n_bins', n_bins=10, ignore_fringe_bins=[1, 3])
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -34.2009)
        self.assertEqual(bin_starts.iloc[-1], -1.1503999999999976)
        self.assertEqual(bin_starts.mean(), -17.67565)
        self.assertEqual(bin_starts.count(), 6)
        checkix = results.index[results['BIN_START_INCL'] == -1.1503999999999976].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 112210)
        self.assertEqual(hist.peakbins, [-1.1503999999999976, -7.7605, -14.3706, -20.9807, -27.5908])

        hist = Histogram(series=series, method='uniques', n_bins=10, ignore_fringe_bins=[1, 3])
        results = hist.results
        bin_starts = hist.results['BIN_START_INCL'].copy()
        self.assertEqual(bin_starts.iloc[0], -39.817)
        self.assertEqual(bin_starts.iloc[-1], 22.276)
        self.assertEqual(bin_starts.mean(), -3.605244071740962)
        self.assertEqual(bin_starts.count(), 24923)
        checkix = results.index[results['BIN_START_INCL'] == -8.062].tolist()
        self.assertEqual(len(checkix), 1)
        checkix = int(checkix[0])
        self.assertEqual(results.iloc[checkix]['COUNTS'], 7)
        self.assertEqual(hist.peakbins, [1.148, 1.241, 1.929, 1.324, 1.632])

    def test_profile_dataframe(self):
        import numpy as np
        import pandas as pd
        from diive.analysis.profile import (
            PROFILE_COLUMNS, count_gaps, dataframe_overview, profile_dataframe)

        # Deterministic synthetic frame: 30-min index, a duplicated timestamp,
        # numeric / constant / non-numeric / all-missing columns with known gaps.
        idx = pd.date_range('2020-01-01', periods=10, freq='30min')
        idx = idx.append(pd.DatetimeIndex([idx[-1]]))  # one duplicate timestamp
        df = pd.DataFrame(index=idx)
        # NUM: two separate NaN runs (positions 2 and 5-6) -> 2 gaps; one zero.
        df['NUM'] = [1.0, 0.0, np.nan, 3.0, 4.0, np.nan, np.nan, 7.0, 8.0, 9.0, 9.0]
        df['CONST'] = 5.0                  # constant numeric column
        df['CAT'] = list('aabbccddeeff'[:11])  # non-numeric (object)
        df['EMPTY'] = np.nan               # all missing

        prof = profile_dataframe(df)
        self.assertListEqual(list(prof.columns), PROFILE_COLUMNS)
        self.assertEqual(len(prof), 4)
        # Row order matches df column order; VARIABLE is a string.
        self.assertListEqual(prof['VARIABLE'].tolist(), ['NUM', 'CONST', 'CAT', 'EMPTY'])

        num = prof.set_index('VARIABLE').loc['NUM']
        self.assertEqual(num['COUNT'], 8)
        self.assertEqual(num['MISSING'], 3)
        self.assertEqual(num['N_GAPS'], 2)           # two consecutive-NaN runs
        self.assertEqual(num['N_ZEROS'], 1)
        self.assertFalse(num['CONSTANT'])
        self.assertEqual(num['MIN'], 0.0)
        self.assertEqual(num['MAX'], 9.0)
        self.assertEqual(num['MEDIAN'], np.median([1, 0, 3, 4, 7, 8, 9, 9]))

        const = prof.set_index('VARIABLE').loc['CONST']
        self.assertTrue(const['CONSTANT'])
        self.assertEqual(const['N_UNIQUE'], 1)

        # Non-numeric: numeric summaries are NaN, no zeros counted.
        cat = prof.set_index('VARIABLE').loc['CAT']
        self.assertEqual(cat['N_ZEROS'], 0)
        self.assertTrue(np.isnan(cat['MEAN']))
        self.assertTrue(np.isnan(cat['MIN']))

        # All-missing: count 0, 100% missing, single gap spanning everything.
        empty = prof.set_index('VARIABLE').loc['EMPTY']
        self.assertEqual(empty['COUNT'], 0)
        self.assertEqual(empty['MISSING_PERC'], 100.0)
        self.assertEqual(empty['N_GAPS'], 1)
        self.assertTrue(empty['CONSTANT'])           # <=1 unique non-missing value
        self.assertTrue(np.isnan(empty['MEAN']))

        # count_gaps directly: no gaps, single value, edge runs.
        self.assertEqual(count_gaps(pd.Series([1.0, 2.0, 3.0])), 0)
        self.assertEqual(count_gaps(pd.Series([np.nan])), 1)
        self.assertEqual(count_gaps(pd.Series([np.nan, 1.0, np.nan, np.nan])), 2)

        ov = dataframe_overview(df)
        self.assertEqual(ov['n_rows'], 11)
        self.assertEqual(ov['n_cols'], 4)
        self.assertEqual(ov['n_cells'], 44)
        self.assertEqual(ov['duplicate_timestamps'], 1)
        # The duplicate timestamp makes the index irregular -> freq uninferable.
        self.assertIsNone(ov['freq'])
        self.assertEqual(ov['start'], idx.min())
        self.assertEqual(ov['end'], idx.max())
        # Missing cells: NUM(3) + EMPTY(11) = 14 of 44.
        self.assertEqual(ov['missing_cells'], 14)
        self.assertAlmostEqual(ov['missing_perc'], round(100 * 14 / 44, 2))
        self.assertGreater(ov['memory_bytes'], 0)

    def test_profile_dataframe_exampledata(self):
        import diive as dv
        from diive.analysis.profile import PROFILE_COLUMNS
        from diive.configs.exampledata import load_exampledata_parquet
        df = load_exampledata_parquet()

        prof = dv.analysis.profile_dataframe(df)
        self.assertListEqual(list(prof.columns), PROFILE_COLUMNS)
        self.assertEqual(len(prof), df.shape[1])     # one row per variable
        # COUNT + MISSING reconcile with the frame for every variable.
        self.assertTrue((prof['COUNT'] + prof['MISSING'] == len(df)).all())

        ov = dv.analysis.dataframe_overview(df)
        self.assertEqual(ov['n_rows'], len(df))
        self.assertEqual(ov['n_cols'], df.shape[1])
        self.assertEqual(ov['freq'], '30min')


    def test_compound_extremes(self):
        import numpy as np
        import pandas as pd
        import diive as dv
        from diive.analysis.compoundextremes import (CompoundExtremes, CAT_NONE,
                                                     CAT_VAR1, CAT_VAR2, CAT_COMPOUND)
        from diive.configs.exampledata import load_exampledata_parquet
        df = load_exampledata_parquet()
        vpd = df['VPD_f'].copy()
        swc = df['SWC_FF0_0.15_1'].copy()

        # Monthly, deseasonalized: high VPD / low SWC, threshold 2 sigma.
        ce = CompoundExtremes(var1=vpd, var2=swc, agg='monthly', threshold=2.0,
                              var1_extreme='high', var2_extreme='low',
                              standardize_by='season', var1_label='Air', var2_label='Soil')
        res = ce.results
        self.assertEqual(len(res), 120)  # 10 years x 12 months
        # Deseasonalized z-scores have ~zero mean per calendar month.
        self.assertAlmostEqual(res['VPD_f_Z'].mean(), 0.0, places=6)
        counts = ce.counts
        self.assertEqual(counts['None'], 117)
        self.assertEqual(counts['Air'], 3)
        self.assertEqual(counts['Soil'], 0)
        self.assertEqual(counts['Compound'], 0)
        # counts reconcile with the per-period rows.
        self.assertEqual(counts.sum(), len(res))
        # Extreme flags agree with the z-score thresholds + directions.
        self.assertTrue((res.loc[res['VAR1_EXTREME'], 'VPD_f_Z'] >= 2.0).all())
        self.assertTrue((res.loc[res['VAR2_EXTREME'], 'SWC_FF0_0.15_1_Z'] <= -2.0).all())
        # Category codes follow the flags.
        air = res[res['CATEGORY'] == CAT_VAR1]
        self.assertTrue((air['VAR1_EXTREME'] & ~air['VAR2_EXTREME']).all())
        self.assertListEqual(sorted(res['CATEGORY'].unique()), sorted({CAT_NONE, CAT_VAR1}))
        # PERIOD labels are %Y-%m for monthly.
        self.assertRegex(res['PERIOD'].iloc[0], r'^\d{4}-\d{2}$')

        # Whole-record standardization flags a different set (seasonal cycle kept).
        ce_rec = CompoundExtremes(var1=vpd, var2=swc, agg='monthly', threshold=2.0,
                                  standardize_by='record', var1_label='Air', var2_label='Soil')
        rec = ce_rec.counts
        self.assertEqual(rec['None'], 113)
        self.assertEqual(rec['Air'], 3)
        self.assertEqual(rec['Soil'], 4)

        # Daily resolution, deseasonalized by day-of-year.
        ce_day = CompoundExtremes(var1=vpd, var2=swc, agg='daily', threshold=2.0,
                                  standardize_by='season', var1_label='Air', var2_label='Soil')
        self.assertEqual(ce_day.counts['Air'], 83)
        self.assertEqual(ce_day.counts['Soil'], 7)
        self.assertRegex(ce_day.results['PERIOD'].iloc[0], r'^\d{4}-\d{2}-\d{2}$')

        # Synthetic series with a guaranteed compound period exercises CAT_COMPOUND.
        idx = pd.date_range('2010-01-01', periods=120, freq='MS', name='TIMESTAMP_MIDDLE')
        rng = np.random.default_rng(0)
        v1 = pd.Series(rng.normal(0, 1, 120), index=idx, name='A')
        v2 = pd.Series(rng.normal(0, 1, 120), index=idx, name='B')
        v1.iloc[60] += 10.0   # force high A
        v2.iloc[60] -= 10.0   # force low B at the same month -> compound
        ce_syn = CompoundExtremes(var1=v1, var2=v2, agg='monthly', threshold=2.0,
                                  var1_extreme='high', var2_extreme='low',
                                  standardize_by='record')
        self.assertEqual(ce_syn.results['CATEGORY'].iloc[60], CAT_COMPOUND)
        self.assertEqual(ce_syn.counts['Compound'], 1)

        # var1/var2 with identical names is rejected.
        with self.assertRaises(ValueError):
            CompoundExtremes(var1=vpd.rename('X'), var2=swc.rename('X'))


if __name__ == '__main__':
    unittest.main()
