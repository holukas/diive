"""
USTAR REDDYPROC DETECTION: U* THRESHOLD AS IN REDDYPROC
=======================================================

Faithful port of the u* threshold estimation of REddyProc 1.3.4
(``sEddyProc_sEstUstarThold`` and ``sEddyProc_sEstimateUstarScenarios``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from diive.core.utils.console import detail, info, success

# REddyProc defaults: usControlUstarSubsetting (DP:629-678) and
# usControlUstarEst (DP:564-623). REddyProc warns when they are changed.
N_TA_CLASSES = 7
N_USTAR_CLASSES = 20
SWIN_NIGHT_THRESHOLD = 10.0
MIN_RECORDS_TEMP = 100
MIN_RECORDS_SEASON = 160
MIN_RECORDS_YEAR = 3000
PLATEAU_FWD = 10
PLATEAU_CRIT = 0.95
CORR_CHECK = 0.5
FIRST_USTAR_MEAN_CHECK = 0.2
MIN_VALID_TEMP_CLASSES_PROP = 0.2
MIN_VALID_BOOT_PROP = 0.4
MIN_N_USTAR_PLATEAU = 3


def season_factor_month(times: pd.DatetimeIndex,
                        start_months=(3, 6, 9, 12)) -> np.ndarray:
    """REddyProc ``usCreateSeasonFactorMonth`` (DP:684-741).

    Seasons start in the given months of every calendar year present in
    ``times``. A season runs until the next start, across the year boundary,
    so with the default December of year Y and January-February of Y + 1
    form season ``Y * 1000 + 12``. If January is not a start month, the
    first year's January and February form a season of their own
    (``Y * 1000 + 1``). The last season has no end.

    Args:
        times: Mid-period timestamps (REddyProc's ``sDateTime``).
        start_months: Months (1-12) in which a season starts.

    Returns:
        Integer season code ``year * 1000 + start month`` per record.
    """
    month = times.month.to_numpy().astype(np.int64)
    year = times.year.to_numpy().astype(np.int64)
    sm = sorted(set(int(m) for m in start_months))
    years = sorted(set(year.tolist()))
    starts = [(y, m) for y in years for m in sm]
    if starts[0][1] != 1:
        starts.insert(0, (years[0], 1))
    codes = np.array([y * 1000 + m for y, m in starts], dtype=np.int64)
    ym = year * 1000 + month
    season = np.zeros(len(ym), dtype=np.int64)
    for i in range(len(codes) - 1):
        season[(ym >= codes[i]) & (ym < codes[i + 1])] = codes[i]
    season[ym >= codes[-1]] = codes[-1]
    return season


def _year_of_seasons(season_idx: np.ndarray, t_sec: np.ndarray,
                     n_seasons: int) -> np.ndarray:
    """REddyProc ``usGetYearOfSeason`` (DP:891-912): per season, the year of
    the midpoint between its first and its last record in data order."""
    from diive.flux.lowres._ustar_rp_kernels import first_last
    first, last = first_last(season_idx, n_seasons)
    x0 = t_sec[first]
    center = x0 + (t_sec[last] - x0) / 2
    return np.floor(center).astype('int64').astype('datetime64[s]').astype(
        'datetime64[Y]').astype(np.int64) + 1970


class UstarReddyProcDetection:
    """u* threshold detection as in REddyProc 1.3.4 (moving point test, Fw2).

    Port of ``sEddyProc_sEstUstarThold`` and of the bootstrap
    ``sEddyProc_sEstimateUstarScenarios`` in REddyProc's
    ``EddyUStarFilterDP.R`` (cited as DP:line below). It follows the idea of
    Papale et al. (2006) with REddyProc's own binning and aggregation, which
    differ from ONEFlux ``ustar_mp`` (see
    :class:`diive.flux.UstarMovingPointDetection`).

    Algorithm (REddyProc defaults):

    1. Night records: NEE, TA, USTAR and SW_IN finite and SW_IN < 10 W m-2
       (DP:1174-1195).
    2. Seasons start in March, June, September and December. December joins
       January and February of the next year. The first year's January and
       February form a season of their own (DP:684-741). A season belongs to
       the year of the midpoint between its first and last record
       (DP:891-912).
    3. Per season with at least 160 night records and at least 100 per
       temperature class: 7 temperature classes of equal size, equal
       temperatures (within 1e-8) kept together (DP:968-1000). A class is used
       only if ``|cor(USTAR, TA)| < 0.5`` (DP:499-506). Within it, 20 u*
       classes of at least ``floor(n / 20)`` records, equal u* (within 1e-14)
       kept together, a short rest merged into the last class, ids above 20
       set to 20 (DP:924-1053). If the first class has a mean u* above
       0.2 m s-1 the temperature class gives no threshold (DP:532-538).
    4. Moving point test Fw2 (DP:1129-1167): the threshold is the mean u* of
       the first class whose mean NEE, and that of the next class, reach 95 %
       of the mean NEE of the following 10 classes. At least 3 classes must
       follow.
    5. Season: median over the temperature classes, missing if fewer than
       20 % of them gave a value (DP:251-256). Year: maximum over its seasons
       (DP:269-276). A year with fewer than 3000 night records, or without any
       season value, is estimated from all its night records pooled as one
       season (DP:288-331). Single value: median over the years (DP:333).
       Years without a value get the single value, seasons without a value
       get their year's value (DP:339-347).
    6. Bootstrap (:meth:`bootstrap`, DP:1264-1401): resample the records of
       every season with replacement (all records, before the night filter),
       repeat the estimate, and report the 5, 50 and 95 % quantiles (R type 7)
       over the original and the resampled estimates. Columns with fewer than
       40 % finite values give no quantiles.

    REddyProc's change-point option (``isUsingCPTSeveralT``, which needs the R
    package ``segmented``) is not ported.

    Timestamps: REddyProc works with the period middle (``sDateTime``, the
    END stamp minus half a period). The input index is taken as diive's
    TIMESTAMP_MIDDLE. An index named ``TIMESTAMP_END`` or ``TIMESTAMP_START``
    is moved to the middle by half the median record interval first. The
    input must be a regular time series without missing timestamps, as
    REddyProc requires: records with missing values count for the year of a
    season and are resampled in the bootstrap.

    Reproduced REddyProc behaviour (do not "fix"):

    - In the bootstrap, a season's year comes from the first and last record
      of the resample (DP:112, called through DP:1337). A season that
      crosses the year boundary can therefore belong to the other year in a
      resample.
    - The check that a resample has the same years as the original reads a
      column that does not exist (``res$uStarTh$year``, DP:1348) and always
      passes. Year values are taken by position (DP:1350).
    - ``seasonFactor = dsBootWithinSeason$season`` (DP:1339) works only by
      R's partial matching of ``$season`` to the column ``seasonFactor``.
    - Temperature classes that fail the correlation check report u* class 0
      for their records (DP:480), not NA.
    - Where a temperature class has no value change after its nominal start,
      R stops with an error (DP:992-996). This port raises ``ValueError``.

    Numerics: means are R's long-double ``mean()`` (emulated bit for bit, see
    :mod:`diive.flux.lowres._ustar_rp_numerics`), sorting is R's stable
    ``order()``, quantiles use R's type 7 formula. Measured agreement with
    native REddyProc 1.3.4 is in the parity harness
    ``diive-data/references/ustar_parity/reddyproc/``.

    Reference:
        Papale, D. et al. (2006). Towards a standardized processing of Net
        Ecosystem Exchange measured with eddy covariance technique:
        algorithms and uncertainty estimation. Biogeosciences, 3, 571-583.
        https://doi.org/10.5194/bg-3-571-2006

        Wutzler, T. et al. (2018). Basic and extensible post-processing of
        eddy covariance flux data with REddyProc. Biogeosciences, 15,
        5015-5030. https://doi.org/10.5194/bg-15-5015-2018

    Example:
        >>> import diive as dv
        >>> from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
        >>> df = dv.load_exampledata_parquet_lae()
        >>> det = UstarReddyProcDetection(
        ...     df, nee_col='NEE_L3.1_L3.2_QCF_IRGA72', ta_col='TA_T1_47_1_gfXG_IRGA72',
        ...     ustar_col='USTAR_IRGA72', swin_col='SW_IN_T1_47_1_gfXG_IRGA72').run()
        >>> round(det.single_, 4)
        0.4117

        ``det.bootstrap(seed=42)`` then fills ``det.distribution_`` and
        ``det.scenarios_annual_``.
    """

    def __init__(self,
                 df: DataFrame,
                 nee_col: str,
                 ta_col: str,
                 ustar_col: str,
                 swin_col: str,
                 season_start_months=(3, 6, 9, 12),
                 verbose: int = 1):
        """
        Args:
            df: Data with a regular DatetimeIndex (TIMESTAMP_MIDDLE, or an
                index named TIMESTAMP_END or TIMESTAMP_START). Several years
                are allowed. Seasons then cross the year boundaries.
            nee_col: NEE before u* filtering (umol m-2 s-1).
            ta_col: Air temperature (degC).
            ustar_col: Friction velocity (m s-1).
            swin_col: Incoming shortwave radiation (W m-2), for the night
                filter.
            season_start_months: Months in which a season starts
                (REddyProc ``usCreateSeasonFactorMonth`` ``startMonth``).
            verbose: Console verbosity (0 silent, 1 errors, 2 progress, 3 debug).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("'df' must have a DatetimeIndex.")
        if len(df) < 2:
            raise ValueError("'df' needs at least two records.")
        self.verbose = verbose
        self.cols = dict(nee=nee_col, ta=ta_col, ustar=ustar_col, swin=swin_col)
        missing = [c for c in self.cols.values() if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in 'df': {missing}")
        data = df[[nee_col, ta_col, ustar_col, swin_col]].astype(float)
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        self.index = data.index
        self.mid = self._middle_timestamps(data.index)
        self.nee = data[nee_col].to_numpy(dtype=float).copy()
        self.ta = data[ta_col].to_numpy(dtype=float).copy()
        self.ustar = data[ustar_col].to_numpy(dtype=float).copy()
        self.swin = data[swin_col].to_numpy(dtype=float).copy()
        self.season_codes = season_factor_month(self.mid, season_start_months)
        self.seasons = np.unique(self.season_codes)
        self.season_idx = np.searchsorted(self.seasons, self.season_codes)
        # seconds since the epoch, as R's as.numeric(POSIXct)
        self.t_sec = ((self.mid - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(float)
        self._res0 = None
        self._season_pos = None
        self._reset_results()

    def _reset_results(self):
        self.records_ = None
        self.bin_averages_ = None
        self.temp_class_thresholds_ = None
        self.temp_class_thresholds_pooled_ = None
        self.seasons_ = None
        self.years_ = None
        self.single_ = None
        self.thresholds_ = None
        self.boot_samples_ = None
        self.distribution_ = None
        self.scenarios_annual_ = None
        self.scenarios_annual_by_year_ = None

    @staticmethod
    def _middle_timestamps(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        # A time zone would move the year boundaries; REddyProc gets local
        # wall-clock time, so keep that.
        if index.tz is not None:
            index = index.tz_localize(None)
        name = str(index.name) if index.name is not None else ''
        if name not in ('TIMESTAMP_END', 'TIMESTAMP_START'):
            return pd.DatetimeIndex(index)
        half = pd.Series(index).diff().median() / 2
        return pd.DatetimeIndex(index - half if name == 'TIMESTAMP_END' else index + half)

    # ------------------------------------------------------------------ core
    def _estimate(self, pos: np.ndarray | None, details: bool) -> dict:
        """``usEstUstarThreshold`` (DP:103-398) on the records at ``pos``
        (0-based, in that order; ``None`` for all records in time order)."""
        from diive.flux.lowres import _ustar_rp_kernels as K
        from diive.flux.lowres._ustar_rp_numerics import r_median_finite

        if pos is None:
            nee, ta, ust, sw = self.nee, self.ta, self.ustar, self.swin
            sidx, tsec = self.season_idx, self.t_sec
        else:
            nee, ta, ust, sw = self.nee[pos], self.ta[pos], self.ustar[pos], self.swin[pos]
            sidx, tsec = self.season_idx[pos], self.t_sec[pos]
        n_seasons = len(self.seasons)
        season_year = _year_of_seasons(sidx, tsec, n_seasons)
        # DP:196-197 usGetValidUstarIndices
        valid = (np.isfinite(nee) & np.isfinite(ta) & np.isfinite(ust) & np.isfinite(sw)
                 & (sw < SWIN_NIGHT_THRESHOLD))
        vpos = np.flatnonzero(valid)
        if vpos.size == 0:
            raise ValueError("sEstUstarThreshold: no finite records in dataset")
        v_sidx = sidx[vpos]
        n_rec_season = np.bincount(v_sidx, minlength=n_seasons)

        def run_subset(p):
            out = K.estimate_subset(
                nee[p], ust[p], ta[p], N_TA_CLASSES, N_USTAR_CLASSES, MIN_RECORDS_SEASON,
                MIN_RECORDS_TEMP, CORR_CHECK, FIRST_USTAR_MEAN_CHECK, PLATEAU_FWD,
                PLATEAU_CRIT, MIN_N_USTAR_PLATEAU)
            if out[-1] == K.ERR_TEMP_BREAK:
                raise ValueError(
                    "REddyProc stops here: no temperature change after the nominal start of a "
                    "temperature class (.binWithEqualValuesBalanced, 'NA/NaN argument').")
            return out

        def aggregate(th):
            # DP:251-256 median over temperature classes, NA below 20 %
            if np.isfinite(th).sum() / th.size < MIN_VALID_TEMP_CLASSES_PROP:
                return np.nan
            return r_median_finite(th)

        temp_th = np.full((n_seasons, N_TA_CLASSES), np.nan)
        season_est = np.full(n_seasons, np.nan)
        tbin = np.full(len(nee), -1, dtype=np.int64)
        ubin = np.full(len(nee), -1, dtype=np.int64)
        bins = []
        for s in range(n_seasons):
            p = vpos[v_sidx == s]
            th, tb, ub, b_ust, b_nee, b_n, n_bins, _ = run_subset(p)
            temp_th[s] = th
            season_est[s] = aggregate(th)
            if details:
                tbin[p] = tb
                ubin[p] = ub
                for k in range(N_TA_CLASSES):
                    for c in range(n_bins[k]):
                        bins.append((self.seasons[s], k + 1, c + 1, b_ust[k, c], b_nee[k, c],
                                     b_n[k, c]))
        # DP:269-276 year = max over seasons; DP:209-213 years with few records
        years = np.unique(season_year)
        n_rec_year = np.array([n_rec_season[season_year == y].sum() for y in years])
        max_season = np.array([np.nanmax(season_est[season_year == y])
                               if np.isfinite(season_est[season_year == y]).any() else np.nan
                               for y in years])
        aggr = max_season.copy()
        pooled_years = set(years[~np.isfinite(aggr) & (n_rec_year > 0)].tolist())
        pooled_years |= set(years[n_rec_year < MIN_RECORDS_YEAR].tolist())
        pooled = np.full(len(years), np.nan)
        pooled_temp_th = {}
        if pooled_years:
            v_year = season_year[v_sidx]
            in_pooled = np.isin(v_year, list(pooled_years))
            if not in_pooled.any():
                raise ValueError(
                    f"Expected valid uStar records for year {sorted(pooled_years)}, but got no "
                    "records.")
            # DP:296-326: pooled estimate per year, records in data order
            for j, y in enumerate(years):
                if y in pooled_years:
                    p = vpos[in_pooled & (v_year == y)]
                    if p.size:
                        th = run_subset(p)[0]
                        pooled_temp_th[y] = th
                        pooled[j] = aggregate(th)
        aggr[np.isfinite(pooled)] = pooled[np.isfinite(pooled)]
        # DP:333 single = median over years; DP:339-347 fill NA years / seasons
        single = r_median_finite(aggr)
        year_ustar = np.where(np.isfinite(aggr), aggr, single)
        year_of = {y: year_ustar[j] for j, y in enumerate(years)}
        season_ustar = np.array([season_est[s] if np.isfinite(season_est[s])
                                 else year_of[season_year[s]] for s in range(n_seasons)])
        # Row order of the season table: merge() sorts by seasonYear, stable in
        # the season order (DP:343-345)
        s_order = np.argsort(season_year, kind='stable')
        res = dict(single=single, years=years, year_ustar=year_ustar,
                   season_order=s_order, season_year=season_year, season_ustar=season_ustar)
        if details:
            res.update(temp_th=temp_th, season_est=season_est, n_rec_season=n_rec_season,
                       n_rec_year=n_rec_year, max_season=max_season, pooled=pooled,
                       pooled_temp_th=pooled_temp_th, valid=valid, tbin=tbin, ubin=ubin, bins=bins)
        return res

    # ------------------------------------------------------------------ API
    def run(self) -> "UstarReddyProcDetection":
        """Estimate the thresholds on the data (``sEstUstarThold``)."""
        info("u* threshold detection (REddyProc 1.3.4 port) ...", verbose=self.verbose)
        res = self._estimate(None, details=True)
        self._res0 = res
        seasons = self.seasons
        sy = res['season_year']
        self.records_ = pd.DataFrame({
            'SEASON': self.season_codes,
            'SEASON_YEAR': sy[self.season_idx],
            'NIGHT_VALID': res['valid'],
            'TEMP_CLASS': pd.arrays.IntegerArray(res['tbin'], res['tbin'] < 0),
            'USTAR_CLASS': pd.arrays.IntegerArray(res['ubin'], res['ubin'] < 0),
        }, index=self.index)
        self.bin_averages_ = pd.DataFrame(
            res['bins'], columns=['season', 'temp_class', 'ustar_class', 'ustar_mean',
                                  'nee_mean', 'n_records'])
        self.temp_class_thresholds_ = pd.DataFrame(
            res['temp_th'].T, index=pd.Index(np.arange(1, N_TA_CLASSES + 1), name='temp_class'),
            columns=pd.Index(seasons, name='season'))
        self.temp_class_thresholds_pooled_ = pd.DataFrame(
            {y: th for y, th in res['pooled_temp_th'].items()},
            index=pd.Index(np.arange(1, N_TA_CLASSES + 1), name='temp_class'))
        self.temp_class_thresholds_pooled_.columns.name = 'season_year'
        order = res['season_order']
        self.seasons_ = pd.DataFrame({
            'season_year': sy[order],
            'season': seasons[order],
            'n_records': res['n_rec_season'][order],
            'ustar_season_est': res['season_est'][order],
            'ustar': res['season_ustar'][order],
        })
        self.years_ = pd.DataFrame({
            'season_year': res['years'],
            'n_records': res['n_rec_year'],
            'ustar_max_season': res['max_season'],
            'ustar_pooled': res['pooled'],
            'ustar': res['year_ustar'],
        })
        self.single_ = float(res['single'])
        self.thresholds_ = self._threshold_table(res)
        detail(f"single u* threshold: {self.single_:.4f} m s-1", verbose=self.verbose)
        success("u* threshold detection (REddyProc) done.", verbose=self.verbose)
        return self

    def _threshold_table(self, res: dict) -> DataFrame:
        """``uStarTh`` layout (DP:349-361): single, years, seasons."""
        order = res['season_order']
        n_y = len(res['years'])
        n_s = len(order)
        return pd.DataFrame({
            'aggregation_mode': ['single'] + ['year'] * n_y + ['season'] * n_s,
            'season_year': pd.array([pd.NA] + list(res['years'])
                                    + list(res['season_year'][order]), dtype='Int64'),
            'season': pd.array([pd.NA] * (1 + n_y) + list(self.seasons[order]), dtype='Int64'),
            'ustar': np.concatenate([[res['single']], res['year_ustar'],
                                     res['season_ustar'][order]]),
        })

    def resample_indices(self, rng: np.random.Generator) -> np.ndarray:
        """One bootstrap resample as REddyProc draws it (DP:1330-1335): every
        season in turn, its records drawn with replacement. Returns 0-based
        record positions in resampled order."""
        if self._season_pos is None:
            self._season_pos = [np.flatnonzero(self.season_idx == s)
                                for s in range(len(self.seasons))]
        return np.concatenate([p[rng.integers(0, p.size, p.size)] for p in self._season_pos])

    def bootstrap(self,
                  n_sample: int = 200,
                  probs=(0.05, 0.5, 0.95),
                  seed: int | None = None,
                  rng: np.random.Generator | None = None,
                  indices=None) -> "UstarReddyProcDetection":
        """Bootstrap distribution of the thresholds (``sEstimateUstarScenarios``).

        Args:
            n_sample: Number of rows of the distribution, the original estimate
                included (REddyProc ``nSample``), so ``n_sample - 1`` resamples.
            probs: Quantiles to report.
            seed: Seed for a new ``numpy.random.default_rng`` (ignored if
                ``rng`` or ``indices`` is given). numpy's generator does not
                reproduce R's random numbers.
            rng: numpy Generator to draw the resamples with.
            indices: The resamples themselves, sequences of 0-based record
                positions in resampled order (2-D array or list of arrays).
                Their number plus one replaces ``n_sample``. Use this to repeat
                R's resamples exactly (R's row indices minus one). The order
                matters: a season's year comes from its first and last record
                in the resample, and records with equal TA or u* keep it.

        Results: ``boot_samples_`` (one row per sample, row 0 the original
        estimate), ``distribution_`` (``thresholds_`` plus the quantiles),
        ``scenarios_annual_`` (per season, its year's ``uStar``, ``U05``,
        ``U50``, ``U95``, REddyProc ``usGetAnnualSeasonUStarMap``) and
        ``scenarios_annual_by_year_`` (the same per year).
        """
        if self._res0 is None:
            self.run()
        res0 = self._res0
        if indices is not None:
            samples = [np.asarray(ix, dtype=np.int64) for ix in indices]
            n_sample = len(samples) + 1
        else:
            if rng is None:
                rng = np.random.default_rng(seed)
            samples = None
        info(f"u* bootstrap (REddyProc), {n_sample - 1} resamples ...", verbose=self.verbose)
        years0 = res0['years']
        order0 = res0['season_order']
        seasons0 = self.seasons[order0]
        n_y0 = len(years0)
        n_s0 = len(seasons0)
        stat = np.full((n_sample, 1 + n_y0 + n_s0), np.nan)
        stat[0] = np.concatenate([[res0['single']], res0['year_ustar'],
                                  res0['season_ustar'][order0]])
        for b in range(1, n_sample):
            pos = samples[b - 1] if samples is not None else self.resample_indices(rng)
            r = self._estimate(pos, details=False)
            order = r['season_order']
            flat = np.concatenate([[r['single']], r['year_ustar'], r['season_ustar'][order]])
            stat[b, 0] = flat[0]
            # DP:1348-1351: the year check never fails (res$uStarTh$year is NULL),
            # values are taken by row position; positions past the end are NA
            take = flat[1:1 + n_y0]
            stat[b, 1:1 + take.size] = take
            # DP:1352-1357: seasons only if rows and season labels agree
            if (len(flat) == stat.shape[1]
                    and np.array_equal(self.seasons[order], seasons0)):
                stat[b, 1 + n_y0:] = flat[1 + n_y0:]
        cols = (['single'] + [f'year_{y}' for y in years0]
                + [f'season_{s}' for s in seasons0])
        self.boot_samples_ = pd.DataFrame(stat, columns=cols)
        self.boot_samples_.index.name = 'sample'
        q = self.quantiles(stat, probs)
        qnames = [_quantile_name(p) for p in probs]
        dist = self.thresholds_.copy()
        for j, name in enumerate(qnames):
            dist[name] = q[:, j]
        self.distribution_ = dist
        # DP:1198-1218 usGetAnnualSeasonUStarMap: each season gets its year's row
        years_rows = dist[dist['aggregation_mode'] == 'year'].set_index('season_year')
        seas = dist[dist['aggregation_mode'] == 'season']
        scen = years_rows.loc[seas['season_year'].to_numpy(), ['ustar'] + qnames]
        scen.columns = ['uStar'] + qnames
        scen.insert(0, 'season', seas['season'].to_numpy())
        self.scenarios_annual_ = scen.reset_index(drop=True)
        by_year = years_rows[['ustar'] + qnames].copy()
        by_year.columns = ['uStar'] + qnames
        self.scenarios_annual_by_year_ = by_year
        success("u* bootstrap (REddyProc) done.", verbose=self.verbose)
        return self

    @staticmethod
    def quantiles(stat: np.ndarray, probs) -> np.ndarray:
        """Quantiles per column as DP:1383-1387: R type 7 with ``na.rm``, NA
        where fewer than 40 % of the rows are finite."""
        from diive.flux.lowres._ustar_rp_numerics import r_quantile7
        stat = np.asarray(stat, dtype=float)
        out = np.full((stat.shape[1], len(probs)), np.nan)
        for j in range(stat.shape[1]):
            if np.isfinite(stat[:, j]).sum() / stat.shape[0] < MIN_VALID_BOOT_PROP:
                continue
            out[j] = r_quantile7(stat[:, j], probs)
        return out


def _quantile_name(p: float) -> str:
    """Column name as REddyProc writes it: ``0.05`` -> ``U05`` (DP:1213-1214)."""
    pct = f"{100 * p:.7g}"
    return 'U' + pct.rjust(2, '0')
