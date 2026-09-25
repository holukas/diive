"""
TEST_METEOSCREENING_WORKFLOW: END-TO-END METEOSCREENING UP TO THE UPLOAD FRAME
==============================================================================

Runs the flow of ``notebooks/DatabaseInfluxStepwiseMeteoScreening.ipynb`` from a
frame shaped like ``InfluxIO.download()`` output to the frame the notebook passes
to ``InfluxIO.upload_singlevar()``, and checks that frame against an independent
computation on the raw records. No database is touched: the download is replaced
by a synthetic frame and the upload by the checks it makes before connecting.

Mixed time resolutions are screened on a grid at the finest resolution, with each
raw record only at its own END slot and every other slot empty (NaN). Resampling
weights each record by its original resolution. The reference below computes that
time-weighted aggregate directly from the raw records.

The unit tests in ``test_meteoscreening.py``, ``test_resampling.py`` and
``test_time.py`` cover the pieces. This module exists because a timestamp shift
(30-min data uploaded half a period early) passed all of them.

The second half checks the mixed-resolution rules end to end, in both orders
(10-min then 1-min, 1-min then 10-min): rolling-window and difference tests run
per resolution period, the fine grid is the greatest common divisor of the
resolutions, overlapping transitions are not counted twice, re-finalizing
restores records, daily resampling, warnings for dates that match no record, and
QCF reports that count records rather than empty grid slots.

Part of the diive library: https://github.com/holukas/diive
"""
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

import diive.core.utils.console as console_module
import diive.preprocessing.qaqc.qcf as qcf_module
from diive.core.io.db.influx.common import TAGS
from diive.core.io.db.influx.influxio import InfluxIO
from diive.core.times.times import DetectFrequency
from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb

FIELD = 'TA_T1_2_1'
SITE, LAT, LON, UTC_OFFSET = 'ch-xyz', 47.29, 7.73, 1
TARGET = '30min'
P = pd.Timedelta(TARGET)
START = pd.Timestamp('2024-07-01 00:00')  # first raw END is START + raw freq

# Screening settings. abslim catches the 80 spikes. Hampel runs on double
# differences, where a spike of +S gives 2S at the spike and -S at its two
# neighbours; n_sigma is set so the band (~24 * 0.49 = 12 for noise sd 0.2) lies
# between S=8 and 2S, i.e. only the spike itself is rejected. zscore then finds
# nothing more, so every rejection in the result is one this module planted.
ABSLIM = dict(minval=0, maxval=40)
ABSLIM_SPIKE = 80.0
EDGE_VALUE = 99.0
HAMPEL_SPIKE = 8.0
HAMPEL_N_SIGMA = 24
ZSCORE_THRES = 4.5
CAP = 19.0  # correction_setto_max_threshold, clips daytime peaks
MINCOUNTS_PERC = .25
NEW_TAGS = {'freq': TARGET, 'data_version': 'meteoscreening_diive'}

# Hampel as the notebook calls it
HAMPEL_NOTEBOOK = dict(window_length='7D', n_sigma_daytime=5.5, n_sigma_nighttime=5.5,
                       use_differencing=True, separate_day_night=True, repeat=True,
                       showplot=False)


@dataclass
class Scenario:
    id: str
    segments: list  # [(raw freq, time span), ...] in time order
    agg: str = 'mean'
    abslim_spikes: list = dc_field(default_factory=lambda: ['2024-07-01 13:00', '2024-07-03 02:00'])
    hampel_spikes: list = dc_field(default_factory=lambda: ['2024-07-02 03:30', '2024-07-02 14:00'])
    manual_single: str = '2024-07-01 20:00'
    manual_range: tuple = ('2024-07-02 06:00', '2024-07-02 06:45')  # start is shifted by one raw period
    missing: list = dc_field(default_factory=list)  # END timestamps or [start, end] missing from the download
    edge_minutes: int = 0  # records within this many minutes of either end are rejected
    hampel: bool = True


def _single(freq, **kw):
    return Scenario(segments=[(freq, '3D')], **kw)


# 10-min era, then 1-min era from END 2024-07-03 00:01. The manual removal of one
# record hits the 10-min era, the range the 1-min era. The Hampel spikes are in the
# 1-min era. Hampel runs on each resolution period separately; its window of one
# day of fine-grid slots (1440) is longer than the 10-min era (288 records).
MIXED = dict(segments=[('10min', '2D'), ('1min', '2D')],
             abslim_spikes=['2024-07-01 13:00', '2024-07-04 02:00'],
             hampel_spikes=['2024-07-03 03:30', '2024-07-03 14:00'],
             manual_single='2024-07-01 20:00',
             manual_range=('2024-07-04 06:00', '2024-07-04 06:45'))

# The reverse order: 1-min era, then 10-min era from END 2024-07-03 00:10. The
# single manual removal hits the 1-min era, the range the 10-min era.
MIXED_REV = dict(segments=[('1min', '2D'), ('10min', '2D')],
                 abslim_spikes=['2024-07-01 13:00', '2024-07-04 02:00'],
                 hampel_spikes=['2024-07-01 14:00', '2024-07-02 03:30'],
                 manual_single='2024-07-01 20:00',
                 manual_range=('2024-07-04 06:00', '2024-07-04 06:45'))

ORDERS = ['10min_then_1min', '1min_then_10min']
MIXED_BY_ORDER = {'10min_then_1min': MIXED, '1min_then_10min': MIXED_REV}

# 10-min era, then 15-min era from END 2024-07-03 00:15. Half of the 15-min
# records (END at :15 and :45) are not on a 10-min grid; the fine grid is 5 min,
# the greatest common divisor of both resolutions. No Hampel: a record-count
# window in fine-grid slots means different spans in the two eras.
OFFGRID = dict(segments=[('10min', '2D'), ('15min', '2D')],
               abslim_spikes=['2024-07-01 13:00', '2024-07-04 02:00'],
               hampel_spikes=[], hampel=False,
               manual_single='2024-07-01 20:00',
               manual_range=('2024-07-04 06:00', '2024-07-04 06:45'))

# 10-min era up to END 2024-07-02 12:10, then 1-min. The half hour (12:00, 12:30]
# holds the 10-min record ending 12:10 and the five 1-min records ending 12:21 to
# 12:25, i.e. coverage 0.5. The half hour ending 2024-07-02 08:30 holds a single
# 10-min record (coverage 1/3, kept), the one ending 2024-07-03 08:30 only five
# 1-min records (coverage 1/6, below MINCOUNTS_PERC).
PARTIAL = dict(segments=[('10min', '1D 12:10:00'), ('1min', '1D 11:50:00')],
               abslim_spikes=['2024-07-01 13:00', '2024-07-03 02:00'],
               hampel_spikes=['2024-07-03 03:30', '2024-07-03 14:00'],
               manual_single='2024-07-01 20:00',
               manual_range=('2024-07-03 06:00', '2024-07-03 06:45'),
               missing=[['2024-07-02 08:10', '2024-07-02 08:20'],
                        ['2024-07-02 12:11', '2024-07-02 12:20'],
                        ['2024-07-02 12:26', '2024-07-02 12:30'],
                        ['2024-07-03 08:01', '2024-07-03 08:25']])

SCENARIOS = [
    _single('1min', id='raw_1min'),
    _single('10min', id='raw_10min'),
    _single('30min', id='raw_30min'),
    Scenario(id='mixed_1min_10min_sum', agg='sum', **MIXED),
    Scenario(id='mixed_1min_10min_mean', agg='mean', **MIXED),
    Scenario(id='mixed_partial_period_mean', agg='mean', **PARTIAL),
    Scenario(id='mixed_partial_period_sum', agg='sum', **PARTIAL),
    Scenario(id='mixed_1min_then_10min_mean', agg='mean', **MIXED_REV),
    Scenario(id='mixed_1min_then_10min_sum', agg='sum', **MIXED_REV),
    Scenario(id='mixed_10min_15min_mean', agg='mean', **OFFGRID),
    Scenario(id='mixed_10min_15min_sum', agg='sum', **OFFGRID),
    _single('10min', id='gap_10min', missing=['2024-07-02 10:00']),
    _single('30min', id='gap_30min', missing=['2024-07-02 10:00']),
    _single('1min', id='rejected_edges_1min', edge_minutes=40),
    _single('30min', id='rejected_edges_30min', edge_minutes=30),
]


@dataclass
class Result:
    sc: Scenario
    raw: pd.DataFrame  # the data_detailed frame as handed to the class
    duration: pd.Series  # raw END -> length of the period that record covers
    rejected: pd.DatetimeIndex  # raw END timestamps this module made the screening reject
    manual: pd.DatetimeIndex
    mscr: StepwiseMeteoScreeningDb
    out: pd.DataFrame  # the frame the notebook uploads


def _tagvalues(freq: str) -> dict:
    return dict(site='CH-XYZ', varname=FIELD, units='degC', raw_varname='TA_raw', raw_units='degC',
                hpos='T1', vpos='2', repl='1', data_raw_freq=freq, freq=freq, filegroup='meteo',
                config_filetype='TEST', data_version='raw', gain='1', offset='0')


def _download_frame(sc: Scenario) -> tuple[pd.DataFrame, pd.Series]:
    """Frame in the shape InfluxIO.download() returns for one field: TIMESTAMP_END
    index without freq, all TAGS columns as strings, the field column last, and no
    rows for records missing in the database."""
    parts, t = [], START
    for freq, span in sc.segments:
        idx = pd.date_range(t + pd.Timedelta(freq), t + pd.Timedelta(span), freq=freq)
        part = pd.DataFrame(_tagvalues(freq), index=idx)
        part['_duration'] = pd.Timedelta(freq)
        parts.append(part)
        t = idx[-1]
    df = pd.concat(parts)
    hour = df.index.hour.to_numpy() + df.index.minute.to_numpy() / 60
    rng = np.random.default_rng(42)
    df[FIELD] = 15 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.2, len(df))
    for spec in sc.missing:
        start, end = (spec, spec) if isinstance(spec, str) else spec
        df = df.drop(df.loc[start:end].index)
    df.index = pd.DatetimeIndex(df.index.to_numpy(), name='TIMESTAMP_END')  # no freq, as downloaded
    duration = df.pop('_duration')
    return df[TAGS + [FIELD]], duration


def _screening(raw: pd.DataFrame) -> StepwiseMeteoScreeningDb:
    return StepwiseMeteoScreeningDb(site=SITE, data_detailed={FIELD: raw}, fields=[FIELD],
                                    site_lat=LAT, site_lon=LON, utc_offset=UTC_OFFSET)


def _run(sc: Scenario) -> Result:
    raw, duration = _download_frame(sc)
    ends = raw.index

    abslim = pd.DatetimeIndex(sc.abslim_spikes)
    raw.loc[abslim, FIELD] = ABSLIM_SPIKE
    hampel = pd.DatetimeIndex(sc.hampel_spikes)
    raw.loc[hampel, FIELD] += HAMPEL_SPIKE
    edges = pd.DatetimeIndex([])
    if sc.edge_minutes:
        span = pd.Timedelta(minutes=sc.edge_minutes)
        edges = ends[(ends <= ends[0] + span - duration.iloc[0]) | (ends > ends[-1] - span)]
        raw.loc[edges, FIELD] = EDGE_VALUE

    # Manual removal takes END timestamps: one record, and a [start, end] range
    # that starts one raw period after the given time so it begins on a record.
    first_freq = duration.loc[pd.Timestamp(sc.manual_range[0]):].iloc[0]
    range_start = pd.Timestamp(sc.manual_range[0]) + first_freq
    range_end = pd.Timestamp(sc.manual_range[1])
    manual = ends[(ends == pd.Timestamp(sc.manual_single))
                  | ((ends >= range_start) & (ends <= range_end))]
    remove_dates = [sc.manual_single, [str(range_start), str(range_end)]]

    # --- The notebook flow -------------------------------------------------------
    mscr = _screening(raw)
    mscr.start_outlier_detection()
    # abslim first: Hampel would also reject the neighbours of an 80 spike.
    mscr.flag_outliers_abslim_test(**ABSLIM, showplot=False)
    mscr.addflag()
    mscr.flag_manualremoval_test(remove_dates=remove_dates, showplot=False)
    mscr.addflag()
    if sc.hampel:
        records_per_day = int(pd.Timedelta('1D') / pd.Timedelta(mscr.series_hires_orig[FIELD].index.freq))
        mscr.flag_outliers_hampel_test(window_length=records_per_day, n_sigma=HAMPEL_N_SIGMA,
                                       use_differencing=True, separate_day_night=True, repeat=True,
                                       showplot=False)
        mscr.addflag()
    mscr.flag_outliers_zscore_test(thres_zscore=ZSCORE_THRES, separate_day_night=True, repeat=True,
                                   showplot=False)
    mscr.addflag()
    mscr.flag_missingvals_test()
    mscr.finalize_outlier_detection()
    mscr.correction_setto_max_threshold(threshold=CAP, showplot=False)
    mscr.resample(to_freqstr=TARGET, agg=sc.agg, mincounts_perc=MINCOUNTS_PERC)
    out = mscr.resampled_detailed[FIELD]

    rejected = abslim.union(hampel).union(edges).union(manual)
    return Result(sc=sc, raw=raw, duration=duration, rejected=rejected, manual=manual,
                  mscr=mscr, out=out)


_RESULTS = {}


def _cached(sid: str) -> Result:
    """Each scenario runs once per session; the dedicated tests reuse the fixture's runs."""
    if sid not in _RESULTS:
        _RESULTS[sid] = _run(next(s for s in SCENARIOS if s.id == sid))
    return _RESULTS[sid]


@pytest.fixture(scope='module', params=[s.id for s in SCENARIOS], ids=[s.id for s in SCENARIOS])
def res(request) -> Result:
    return _cached(request.param)


# --- Independent reference computed on the raw records -----------------------------

def _step(duration: pd.Series) -> pd.Timedelta:
    """Fine grid step: the greatest common divisor of the raw resolutions and of
    the distances between END timestamps. That is the finest resolution when all
    records lie on its grid, and 5 min for 10-min then 15-min records."""
    secs = np.concatenate([duration.dt.total_seconds().to_numpy(),
                           (duration.index - duration.index[0]).total_seconds().to_numpy()])
    return pd.Timedelta(seconds=int(np.gcd.reduce(secs.astype(np.int64))))


def _weighted_aggregate(values: pd.Series, duration: pd.Series, agg: str,
                        period: pd.Timedelta = P) -> tuple[pd.Series, pd.Series]:
    """Aggregate of *values* (raw END index, no NaN) per target period END T, over
    the records with END in (T - period, T].

    Each record weighs its own duration in fine-grid slots (10 for a 10-min record
    on a 1-min grid). mean = sum(w * v) / sum(w), sum = sum(v), coverage = sum(w)
    over the slots per period; periods below MINCOUNTS_PERC are NaN. With one
    resolution this is the plain mean and the record count.

    Returns the aggregate and the coverage, both only for periods with records.
    """
    step = _step(duration)
    w = duration.reindex(values.index) / step
    target = values.index.ceil(period)
    wsum = w.groupby(target).sum()
    coverage = wsum / (period / step)
    if agg == 'sum':
        aggregate = values.groupby(target).sum()
    else:
        aggregate = (values * w).groupby(target).sum() / wsum
    return aggregate.where(coverage >= MINCOUNTS_PERC), coverage


def _target_index(r: Result) -> pd.DatetimeIndex:
    return pd.date_range(r.raw.index[0].ceil(P), r.raw.index[-1].ceil(P), freq=P)


def _expected(r: Result) -> pd.Series:
    """Aggregate of the kept raw records per target period, over the full screened
    range; NaN where no record is kept or coverage is too low."""
    kept = r.raw[FIELD].drop(r.rejected).clip(upper=CAP)
    aggregate, coverage = _weighted_aggregate(kept, r.duration, r.sc.agg)
    # The data are designed to stay clear of the mincounts limit, where rounding
    # of the minimum record count could decide: each period is either at most a
    # sixth or at least a third covered.
    assert not ((coverage > 1 / 6 + 1e-9) & (coverage < 1 / 3 - 1e-9)).any(), \
        'test data hit the mincounts limit'
    return aggregate.reindex(_target_index(r))


def _record_slots(r: Result) -> pd.DatetimeIndex:
    """TIMESTAMP_MIDDLE slot of each raw record on the fine grid: its END slot."""
    return r.raw.index - _step(r.duration) / 2


# --- Assertions -----------------------------------------------------------------

def test_index_is_timestamp_end_at_target_freq(res):
    idx = res.out.index
    assert idx.name == 'TIMESTAMP_END'
    assert to_offset(idx.freqstr) == to_offset(TARGET)
    assert idx.is_monotonic_increasing and not idx.has_duplicates
    assert (idx.to_series().diff().dropna() == P).all()
    # The notebook's own check before uploading
    assert to_offset(DetectFrequency(index=idx, verbose=False).get()) == to_offset(TARGET)


def test_values_equal_aggregate_of_raw_records_in_period(res):
    expected = _expected(res)
    got = res.out[FIELD]
    pd.testing.assert_index_equal(got.index, expected.index, check_names=False)
    pd.testing.assert_series_equal(got, expected, check_names=False, check_freq=False,
                                   rtol=1e-9, atol=1e-9)


def test_each_raw_record_once_at_its_end_slot(res):
    """Every kept raw record appears exactly once in the cleaned series, at its END
    slot with its (corrected) value; the planted rejections are NaN, and so is
    every slot that held no raw record."""
    cleaned = res.mscr.series_hires_cleaned[FIELD]
    step = _step(res.duration)
    assert pd.Timedelta(cleaned.index.freq) == step
    slots = _record_slots(res)
    assert slots.isin(cleaned.index).all()

    got = cleaned.reindex(slots).to_numpy()
    want = res.raw[FIELD].clip(upper=CAP).to_numpy()
    is_rejected = res.raw.index.isin(res.rejected)
    ends = res.raw.index
    lost = ends[~is_rejected & np.isnan(got)]
    changed = ends[~is_rejected & ~np.isnan(got) & ~np.isclose(got, want)]
    still_there = ends[is_rejected & ~np.isnan(got)]
    assert lost.empty, f'raw records lost: {list(lost[:5])}'
    assert changed.empty, f'raw records changed: {list(changed[:5])}'
    assert still_there.empty, f'planted rejections still present: {list(still_there[:5])}'

    empty = cleaned.drop(slots)
    filled = empty.index[empty.notna()]
    assert filled.empty, f'{len(filled)} slots without a raw record hold values: {list(filled[:5])}'

    # The original resolution is kept for record slots only (resampling weights by it).
    freq_sec = res.mscr.data_detailed[FIELD]['FREQ_AUTO_SEC']
    np.testing.assert_array_equal(freq_sec.reindex(slots).to_numpy(),
                                  res.duration.dt.total_seconds().to_numpy())
    assert freq_sec.drop(slots).isna().all()

    # And every raw record falls into a period of the upload frame.
    assert res.raw.index.ceil(P).isin(res.out.index).all()


def test_frame_spans_screened_period(res):
    """The pre-upload delete runs from the first to the last index entry, so the
    frame must cover the whole screened range even when edge records are rejected."""
    idx = res.out.index
    assert idx[0] == res.raw.index[0].ceil(P)
    assert idx[-1] == res.raw.index[-1].ceil(P)
    if res.sc.edge_minutes:
        assert np.isnan(res.out[FIELD].iloc[0]) and np.isnan(res.out[FIELD].iloc[-1])
    # Delete bounds as upload_singlevar builds them (both ends inclusive)
    utc = InfluxIO._format_utc_offset(UTC_OFFSET)
    start = pd.Timestamp(InfluxIO._convert_datestr_to_iso8601(str(idx[0]), UTC_OFFSET))
    stop = pd.Timestamp(InfluxIO._convert_datestr_to_iso8601(str(idx[-1]), UTC_OFFSET))
    assert start == idx[0].tz_localize(utc) and stop == idx[-1].tz_localize(utc)


def test_one_data_column_plus_complete_tags(res):
    out = res.out
    assert sorted(out.columns) == sorted([FIELD] + TAGS)
    for tag in TAGS:
        values = out[tag]
        assert values.notna().all(), tag
        assert not values.astype(str).str.lower().isin(['nan', 'none', '']).any(), tag
        assert values.nunique() == 1, tag


def test_freq_and_data_version_tags(res):
    out = res.out
    for tag, value in NEW_TAGS.items():
        assert (out[tag] == value).all(), tag
    # All other tags come through from the download; a tag with several values
    # (data_raw_freq in mixed data) is joined with commas.
    for tag in TAGS:
        if tag in NEW_TAGS:
            continue
        assert set(out[tag].iloc[0].split(',')) == set(res.raw[tag].unique()), tag


def test_manual_removal_by_end_timestamp(res):
    assert len(res.manual) >= 2
    cleaned = res.mscr.series_hires_cleaned[FIELD]
    slots = res.manual - _step(res.duration) / 2
    assert cleaned.reindex(slots).isna().all(), 'manually removed records still in cleaned series'
    # The removed values are nowhere else either (values at CAP are ambiguous).
    removed = res.raw.loc[res.manual, FIELD]
    removed = removed[removed < CAP].to_numpy()
    assert not np.isin(cleaned.dropna().to_numpy(), removed).any()
    # The aggregate of each affected period is computed without the removed records.
    periods = res.manual.ceil(P).unique()
    pd.testing.assert_series_equal(res.out.loc[periods, FIELD], _expected(res).loc[periods],
                                   check_names=False, check_freq=False, rtol=1e-9, atol=1e-9)


def test_passes_upload_prechecks(res):
    """What upload_singlevar checks before it opens a client (not called here)."""
    var_df = res.out.copy()
    cols = var_df.columns.to_list()
    assert [t for t in TAGS if t not in cols] == []
    assert [c for c in cols if c not in TAGS] == [FIELD]
    assert len(set(var_df['data_version'].tolist())) == 1
    localized = var_df.index.tz_localize(InfluxIO._format_utc_offset(UTC_OFFSET))
    assert localized.notna().all() and len(localized) == len(var_df)


def test_caller_frame_untouched():
    sc = SCENARIOS[1]
    raw, _ = _download_frame(sc)
    before = raw.copy()
    _screening(raw)
    pd.testing.assert_frame_equal(raw, before)


# --- Mixed resolutions: records of different length in one period -----------------

def test_partial_period_is_time_weighted():
    """(12:00, 12:30] holds one 10-min record (END 12:10) and five 1-min records
    (END 12:21 to 12:25): coverage 15/30, mean (10 * v10 + sum(v1)) / 15,
    sum v10 + sum(v1)."""
    r = _cached('mixed_partial_period_mean')
    mscr = r.mscr
    values = r.raw[FIELD].clip(upper=CAP)
    t = pd.Timestamp('2024-07-02 12:30')
    in_period = values.loc[t - P + pd.Timedelta('1s'):t]
    v10 = values[pd.Timestamp('2024-07-02 12:10')]
    v1 = values.loc['2024-07-02 12:21':'2024-07-02 12:25']
    assert list(in_period.index) == [pd.Timestamp('2024-07-02 12:10')] + list(v1.index)
    assert (r.duration[v1.index] == pd.Timedelta('1min')).all() and len(v1) == 5

    def resampled(agg, mincounts_perc=MINCOUNTS_PERC) -> pd.Series:
        mscr.resample(to_freqstr=TARGET, agg=agg, mincounts_perc=mincounts_perc)
        return mscr.resampled_detailed[FIELD][FIELD]

    assert np.isclose(resampled('mean')[t], (10 * v10 + v1.sum()) / 15, rtol=1e-12)
    assert np.isclose(resampled('sum')[t], v10 + v1.sum(), rtol=1e-12)
    # Coverage 0.5: kept when half the period is required, NaN above that.
    assert not np.isnan(resampled('mean', mincounts_perc=.45)[t])
    assert np.isnan(resampled('mean', mincounts_perc=.55)[t])

    out = resampled('mean')
    # A single 10-min record covers a third of its half hour: kept, as its own value.
    single_10min = pd.Timestamp('2024-07-02 08:30')
    assert list(r.raw.loc[single_10min - P + pd.Timedelta('1s'):single_10min].index) == [single_10min]
    assert np.isclose(out[single_10min], values[single_10min], rtol=1e-12)
    # Five 1-min records cover a sixth: below MINCOUNTS_PERC.
    five_1min = pd.Timestamp('2024-07-03 08:30')
    assert len(r.raw.loc[five_1min - P + pd.Timedelta('1s'):five_1min]) == 5
    assert np.isnan(out[five_1min])


@pytest.mark.parametrize('sid', ['mixed_1min_10min_mean', 'mixed_1min_10min_sum'])
def test_mixed_manual_removal_removes_whole_coarse_record(sid):
    """L177: removing a 10-min record by its END timestamp removes the whole record,
    and its value is absent from the 30-min aggregate."""
    r = _cached(sid)
    end = pd.Timestamp(r.sc.manual_single)
    assert r.duration[end] == pd.Timedelta('10min')
    cleaned = r.mscr.series_hires_cleaned[FIELD]
    # The END slot and the nine empty slots before it hold nothing.
    assert cleaned.loc[end - pd.Timedelta('10min'):end].isna().all()
    # Exact match: a surviving copy would carry the same bits (isclose finds noise neighbours).
    assert not (cleaned == r.raw.loc[end, FIELD]).any()
    # The half hour is aggregated from the other two 10-min records only.
    others = r.raw[FIELD].loc[[end - pd.Timedelta('20min'), end - pd.Timedelta('10min')]].clip(upper=CAP)
    want = others.sum() if r.sc.agg == 'sum' else others.mean()
    assert np.isclose(r.out.loc[end.ceil(P), FIELD], want, rtol=1e-12)


@pytest.mark.parametrize('segments', [[('1min', '4D')], [('10min', '2D'), ('1min', '2D')],
                                      [('1min', '2D'), ('10min', '2D')]],
                         ids=['single_1min', 'mixed_10min_1min', 'mixed_1min_10min'])
def test_hampel_notebook_settings_on_clean_data(segments):
    """L178: Hampel on double differences with the notebook settings rejects
    almost nothing of clean data, also when the resolution changes.

    Measured: 0 rejections in all three cases. In the mixed cases each resolution
    period is tested on its own grid, so the 10-min records are judged against
    each other. With coarse records back-filled onto the fine grid, runs of
    identical copies made most differences zero and the same call rejected 1664
    of the 3168 records. The bound (0.2% of the records, 6 of the 3168 mixed
    records) leaves room for noise while failing on anything like that.
    """
    sc = Scenario(id='clean', segments=segments)
    raw, duration = _download_frame(sc)  # no spikes: _run plants them
    mscr = _screening(raw)
    mscr.start_outlier_detection()
    mscr.flag_outliers_hampel_test(**HAMPEL_NOTEBOOK)
    flag = mscr.outlier_detection[FIELD].last_flag
    n_rejected = int((flag.reindex(raw.index - _step(duration) / 2) == 2).sum())
    assert n_rejected <= 0.002 * len(raw), f'{n_rejected} of {len(raw)} clean records rejected'


# --- Empty slots stay empty through every step ----------------------------------------

def test_corrections_leave_empty_slots_empty():
    """Corrections that write values (nighttime zero offset sets nighttime to 0,
    thresholds, set-to-value over a range) must not create values in slots of the
    fine grid that hold no raw record, and resampling must see only the records."""
    sc = Scenario(id='empty_slots', **MIXED)
    raw, duration = _download_frame(sc)
    slots = raw.index - _step(duration) / 2
    mscr = _screening(raw)

    def check(stage):
        cleaned = mscr.series_hires_cleaned[FIELD]
        empty = cleaned.drop(slots)
        filled = empty.index[empty.notna()]
        assert filled.empty, f'{stage}: {len(filled)} empty slots hold values, e.g. {list(filled[:3])}'
        assert cleaned.reindex(slots).notna().all(), f'{stage}: raw records lost'

    check('start')
    mscr.correction_remove_nighttime_zero_offset(showplot=False)
    check('nighttime zero offset')
    mscr.start_outlier_detection()
    sod_series = mscr.outlier_detection[FIELD].series_hires_cleaned
    assert sod_series.drop(slots).isna().all(), 'outlier detection starts with filled empty slots'
    mscr.flag_outliers_abslim_test(minval=-50, maxval=50, showplot=False)
    mscr.addflag()
    mscr.finalize_outlier_detection()
    check('finalize')
    mscr.correction_setto_min_threshold(threshold=1, showplot=False)
    check('min threshold')
    mscr.correction_setto_max_threshold(threshold=CAP, showplot=False)
    check('max threshold')
    # A 10-min record by its END, and a range of 10-min records with empty slots between.
    single, rng = '2024-07-01 20:10', ['2024-07-02 06:00', '2024-07-02 07:00']
    mscr.correction_setto_value(dates=[single, rng], value=3.7, verbose=0)
    check('set to value')
    cleaned = mscr.series_hires_cleaned[FIELD]
    set_ends = raw.loc[rng[0]:rng[1]].index.union([pd.Timestamp(single)])
    assert len(set_ends) == 8
    assert (cleaned.reindex(set_ends - _step(duration) / 2) == 3.7).all()

    values = pd.Series(cleaned.reindex(slots).to_numpy(), index=raw.index)
    target = pd.date_range(raw.index[0].ceil(P), raw.index[-1].ceil(P), freq=P)
    for agg in ('mean', 'sum'):
        mscr.resample(to_freqstr=TARGET, agg=agg, mincounts_perc=MINCOUNTS_PERC)
        expected, _ = _weighted_aggregate(values, duration, agg)
        pd.testing.assert_series_equal(mscr.resampled_detailed[FIELD][FIELD], expected.reindex(target),
                                       check_names=False, check_freq=False, rtol=1e-9, atol=1e-9)


# --- Mixed resolutions, round 2 ------------------------------------------------------

def _frame(parts) -> pd.DataFrame:
    """Download-shaped frame from ``[(END timestamps, raw freq, values), ...]``."""
    frames = []
    for ends, freq, values in parts:
        part = pd.DataFrame(_tagvalues(freq), index=ends)
        part[FIELD] = np.asarray(values, dtype=float)
        frames.append(part)
    df = pd.concat(frames)
    df.index = pd.DatetimeIndex(df.index.to_numpy(), name='TIMESTAMP_END')
    return df[TAGS + [FIELD]]


def _part(df: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.DataFrame:
    """The records of *df* at *ends*, as a download of that part alone."""
    part = df.loc[ends].copy()
    part.index.name = 'TIMESTAMP_END'
    return part


def _slots(mscr: StepwiseMeteoScreeningDb, ends: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """TIMESTAMP_MIDDLE slots of the records ending at *ends* on the screening grid."""
    return ends - pd.Timedelta(mscr.series_hires_orig[FIELD].index.freq) / 2


class _Output:
    """What diive printed while a ``_captured()`` block ran."""

    def __init__(self):
        self.lines: list[str] = []
        self.pywarnings: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(' '.join(str(a) for a in args))

    def log(self, *args, **kwargs):
        self.print(*args)

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)

    @property
    def warnings(self) -> list[str]:
        """Lines printed by ``console.warn()``, plus Python warnings."""
        return [line for line in self.lines if '[yellow]!' in line] + self.pywarnings


@contextmanager
def _captured():
    """Record diive's console output. The helpers (``warn`` etc.) print to the
    console module's current console, while ``qcf`` keeps the console it imported;
    the two differ once ``refresh_console()`` ran (tests/test_console.py), so
    mirror both."""
    out = _Output()
    consoles = list({id(c): c for c in (console_module.console, qcf_module._console)}.values())
    for c in consoles:
        c.add_mirror(out)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            yield out
        out.pywarnings.extend(str(w.message) for w in caught)
    finally:
        for c in consoles:
            c.remove_mirror(out)


# --- Item 1: rolling-window and difference tests run per resolution period ----------

# 14 days of 10-min records and one day of 1-min records, in either order. White
# noise (sd 0.2) around a level that steps from 10 to 13 seven days into the 10-min
# era, and six +3.5 spikes in that era, each at least three days from the step. A
# '7D' window around a spike sees one level, and every test below catches the
# spikes in the 10-min records screened alone. A window stretched to the 1-min
# grid's 10080 slots, counted in 10-min records, spans the step, and the spikes
# vanish in its spread.
LEVEL_STEP = 3.0
LEVEL_SPIKE = 3.5
LEVEL_SPIKE_DAYS = ['1D 03:00:00', '2D 13:00:00', '3D 22:00:00', '11D 02:00:00', '12D 12:00:00', '13D 20:00:00']

PER_PERIOD_TESTS = {
    'hampel_differencing_7D': lambda m: m.flag_outliers_hampel_test(**HAMPEL_NOTEBOOK),
    'hampel_7D': lambda m: m.flag_outliers_hampel_test(
        window_length='7D', n_sigma=5.5, use_differencing=False, separate_day_night=False,
        repeat=True, showplot=False),
    'zscore_increments': lambda m: m.flag_outliers_increments_zcore_test(
        thres_zscore=5, repeat=False, showplot=False),
    'localsd_7D': lambda m: m.flag_outliers_localsd_test(
        n_sd=5.5, winsize='7D', separate_day_night=False, repeat=False, showplot=False),
    'zscore_rolling_7D': lambda m: m.flag_outliers_zscore_rolling_test(
        thres_zscore=4.5, winsize='7D', repeat=True, showplot=False),
}


@lru_cache(maxsize=None)
def _level_step_data(order: str) -> tuple:
    """Returns the mixed frame, the END timestamps of its 10-min and of its 1-min
    records, and the END timestamps of the spikes. Callers must not modify the frame."""
    segments = [('10min', 14), ('1min', 1)] if order == '10min_then_1min' else [('1min', 1), ('10min', 14)]
    ends, t = {}, START
    for freq, days in segments:
        ends[freq] = pd.date_range(t + pd.Timedelta(freq), t + pd.Timedelta(days=days), freq=freq)
        t = ends[freq][-1]
    coarse, fine = ends['10min'], ends['1min']
    spikes = pd.DatetimeIndex([coarse[0].normalize() + pd.Timedelta(d) for d in LEVEL_SPIKE_DAYS])
    rng = np.random.default_rng(7)
    parts = []
    for freq, _ in segments:
        e = ends[freq]
        level = np.where(e > coarse[0] + pd.Timedelta('7D'), LEVEL_STEP, 0.0)
        values = 10 + level + rng.normal(0, 0.2, len(e)) + np.where(e.isin(spikes), LEVEL_SPIKE, 0.0)
        parts.append((e, freq, values))
    return _frame(parts), coarse, fine, spikes


@pytest.mark.parametrize('test', list(PER_PERIOD_TESTS))
@pytest.mark.parametrize('order', ORDERS)
def test_windowed_tests_run_per_resolution_period(order, test):
    """Item 1: each resolution period is screened on its own grid, so the 10-min
    records get exactly the flags they get when screened alone as 10-min data, and
    the 1-min records those of the 1-min records alone. That covers both halves of
    the fix: difference tests (Hampel with differencing, increments) judge coarse
    records, and a time-span window ('7D') spans seven days in each era. The flag
    is the pending test: addflag() removes what it rejected in both eras."""
    df, coarse, fine, spikes = _level_step_data(order)
    run = PER_PERIOD_TESTS[test]

    def screened(frame):
        m = _screening(frame)
        m.start_outlier_detection()
        run(m)
        return m

    def flags(m, ends):
        return m.outlier_detection[FIELD].last_flag.reindex(_slots(m, ends)).to_numpy()

    mixed = screened(df)
    want_coarse = flags(screened(_part(df, coarse)), coarse)
    want_fine = flags(screened(_part(df, fine)), fine)

    # The test data are meaningful: screened alone, the 10-min spikes are caught.
    n_caught = int((want_coarse[coarse.isin(spikes)] == 2).sum())
    assert n_caught >= len(spikes) - 1, f'{test} catches only {n_caught} of {len(spikes)} spikes alone'

    np.testing.assert_array_equal(flags(mixed, coarse), want_coarse,
                                  err_msg=f'{test}: 10-min records flagged differently than alone')
    np.testing.assert_array_equal(flags(mixed, fine), want_fine,
                                  err_msg=f'{test}: 1-min records flagged differently than alone')

    mixed.addflag()
    cleaned = mixed.outlier_detection[FIELD].series_hires_cleaned
    ends = coarse.append(fine)
    rejected = np.concatenate([want_coarse, want_fine]) == 2
    assert cleaned.reindex(_slots(mixed, ends[rejected])).isna().all()
    assert cleaned.reindex(_slots(mixed, ends[~rejected])).notna().all()


# --- Item 3: overlapping transition ----------------------------------------------

def _overlap_frame(last_fine_end: str, first_coarse_end: str, last_coarse_end: str) -> pd.DataFrame:
    """1-min records of value 0 from END 2024-07-01 00:01, then 10-min records of
    value 10 whose first one overlaps the last 1-min records."""
    fine = pd.date_range('2024-07-01 00:01', last_fine_end, freq='1min')
    coarse = pd.date_range(first_coarse_end, last_coarse_end, freq='10min')
    return _frame([(fine, '1min', np.zeros(len(fine))), (coarse, '10min', np.full(len(coarse), 10.0))])


def _resampled_mean(df: pd.DataFrame, mincounts_perc: float) -> pd.Series:
    mscr = _screening(df)
    mscr.resample(to_freqstr=TARGET, agg='mean', mincounts_perc=mincounts_perc)
    return mscr.resampled_detailed[FIELD][FIELD]


def test_overlapping_transition_counts_time_once():
    """Item 3: 1-min records up to END 00:05, then a 10-min record ending 00:10,
    which covers (00:00, 00:10] and so overlaps five 1-min records. Only its
    non-overlapping part (00:05, 00:10] counts: the half hour (00:00, 00:30] is
    (5 * 0 + 5 * 10 + 20 * 10) / 30, exactly fully covered. Counting the whole
    record gives (5 * 0 + 30 * 10) / 35 at 35/30 coverage."""
    df = _overlap_frame('2024-07-02 00:05', '2024-07-02 00:10', '2024-07-03 00:00')
    t = pd.Timestamp('2024-07-02 00:30')
    for perc in (MINCOUNTS_PERC, 1.0):
        out = _resampled_mean(df, mincounts_perc=perc)
        assert np.isclose(out[t], 250 / 30, rtol=1e-12), f'mincounts_perc={perc}: {out[t]}'
        # The periods around it hold one resolution only.
        assert out[t - P] == 0 and out[t + P] == 10


def test_overlapping_transition_coverage_at_most_full():
    """Item 3: 1-min records up to END 00:33, then 10-min records at :05, :15, ...
    The one ending 00:35 covers (00:25, 00:35], but only (00:33, 00:35] is new, so
    the half hour (00:30, 01:00] holds 3 + 2 + 10 + 10 = 25 of 30 minutes, mean
    (2 * 10 + 20 * 10) / 25. Counting the whole record covers it 33/30, more than
    completely, and keeps it at mincounts_perc=0.9."""
    df = _overlap_frame('2024-07-02 00:33', '2024-07-02 00:35', '2024-07-03 00:05')
    t = pd.Timestamp('2024-07-02 01:00')
    assert np.isnan(_resampled_mean(df, mincounts_perc=.9)[t])
    assert np.isclose(_resampled_mean(df, mincounts_perc=.8)[t], 220 / 25, rtol=1e-12)


# --- Items 4 and 5: finalize ---------------------------------------------------------

STRICT = dict(daytime_accept_qcf_below=0)  # rejects every daytime record


def _finalized(raw: pd.DataFrame, *finalize_kwargs) -> StepwiseMeteoScreeningDb:
    """Flag the abslim spikes, correct, then finalize once per kwargs dict."""
    mscr = _screening(raw)
    mscr.start_outlier_detection()
    mscr.flag_outliers_abslim_test(**ABSLIM, showplot=False)
    mscr.addflag()
    mscr.correction_setto_max_threshold(threshold=CAP, showplot=False)
    for kwargs in finalize_kwargs:
        mscr.finalize_outlier_detection(**kwargs)
    return mscr


def _spiked_download(order: str) -> tuple[pd.DataFrame, pd.Series, Scenario]:
    sc = Scenario(id=f'finalize_{order}', **MIXED_BY_ORDER[order])
    raw, duration = _download_frame(sc)
    raw.loc[pd.DatetimeIndex(sc.abslim_spikes), FIELD] = ABSLIM_SPIKE
    return raw, duration, sc


@pytest.mark.parametrize('order', ORDERS)
def test_refinalize_with_looser_thresholds_restores_records(order):
    """Item 5: finalize starts from the corrected, unmasked series, so a looser run
    after a stricter one gives what the looser run alone gives, corrections kept."""
    raw, _, sc = _spiked_download(order)
    strict = _finalized(raw, STRICT)
    strict_then_loose = _finalized(raw, STRICT, {})
    loose = _finalized(raw, {})

    n_strict = int(strict.series_hires_cleaned[FIELD].notna().sum())
    n_loose = int(loose.series_hires_cleaned[FIELD].notna().sum())
    assert n_loose == len(raw) - len(sc.abslim_spikes)
    assert n_strict < n_loose / 2, 'the strict run should remove the daytime records'

    pd.testing.assert_series_equal(strict_then_loose.series_hires_cleaned[FIELD],
                                   loose.series_hires_cleaned[FIELD])
    pd.testing.assert_series_equal(strict_then_loose.outlier_detection_qcf[FIELD].flagqcf,
                                   loose.outlier_detection_qcf[FIELD].flagqcf)
    assert strict_then_loose.series_hires_cleaned[FIELD].max() <= CAP, 'correction lost'


@pytest.mark.parametrize('order', ORDERS)
def test_daytime_of_coarse_records_from_their_true_middle(order):
    """Item 4: the QCF day/night split places each record at its true middle (END
    minus half its own resolution), so the strict run rejects the same records as
    each era screened alone at its own resolution."""
    raw, duration, _ = _spiked_download(order)
    mixed = _finalized(raw, STRICT)

    def rejected(mscr, ends):
        cleaned = mscr.series_hires_cleaned[FIELD].reindex(_slots(mscr, ends))
        return ends[cleaned.isna().to_numpy()]

    for freq in ('10min', '1min'):
        ends = raw.index[(duration == pd.Timedelta(freq)).to_numpy()]
        want = rejected(_finalized(_part(raw, ends), STRICT), ends)
        got = rejected(mixed, ends)
        assert len(want) > 0
        assert got.equals(want), (f'{freq} era: rejected only in mixed {list(got.difference(want)[:5])}, '
                                  f'only alone {list(want.difference(got)[:5])}')


# --- Item 6: daily resampling ------------------------------------------------------

@pytest.mark.parametrize('order', ORDERS)
def test_resample_to_one_day(order):
    """Item 6: a Day-based target works for mean and sum, time-weighted as for 30 min."""
    r = _run(Scenario(id=f'daily_{order}', **MIXED_BY_ORDER[order]))
    day = pd.Timedelta('1D')
    kept = r.raw[FIELD].drop(r.rejected).clip(upper=CAP)
    target = pd.date_range(r.raw.index[0].ceil(day), r.raw.index[-1].ceil(day), freq=day)
    for agg in ('mean', 'sum'):
        r.mscr.resample(to_freqstr='1D', agg=agg, mincounts_perc=MINCOUNTS_PERC)
        out = r.mscr.resampled_detailed[FIELD]
        assert out.index.name == 'TIMESTAMP_END'
        assert (out.index.to_series().diff().dropna() == day).all()
        assert (out['freq'] == '1D').all()
        want, _ = _weighted_aggregate(kept, r.duration, agg, period=day)
        pd.testing.assert_series_equal(out[FIELD], want.reindex(target), check_names=False,
                                       check_freq=False, rtol=1e-9, atol=1e-9)


# --- Item 7: dates that match no record ---------------------------------------------

# END timestamps in the 10-min era of MIXED. 20:05 and the range both lie inside
# the record ending 20:10.
NO_RECORD = {
    'inside_coarse_record': '2024-07-01 20:05',
    'range_inside_coarse_record': ['2024-07-01 20:01', '2024-07-01 20:09'],
    'outside_data': '2024-08-01 20:00',
}


@pytest.mark.parametrize('case', list(NO_RECORD))
def test_date_matching_no_record_warns(case):
    """Item 7: manual removal and set-to-value warn about an entry that matches no
    record, and change nothing; an entry that matches a record does not warn."""
    raw, _ = _download_frame(Scenario(id='dates', **MIXED))
    entry = NO_RECORD[case]
    first = entry if isinstance(entry, str) else entry[0]
    mscr = _screening(raw)
    mscr.start_outlier_detection()

    with _captured() as out:
        mscr.flag_manualremoval_test(remove_dates=[entry])
    assert any(first in w for w in out.warnings), f'manual removal: no warning naming {first}'
    assert not (mscr.outlier_detection[FIELD].last_flag == 2).any()

    before = mscr.series_hires_cleaned[FIELD].copy()
    with _captured() as out:
        mscr.correction_setto_value(dates=[entry], value=3.7)
    assert any(first in w for w in out.warnings), f'set to value: no warning naming {first}'
    pd.testing.assert_series_equal(mscr.series_hires_cleaned[FIELD], before)

    # Control: the record ending 20:10 exists.
    with _captured() as out:
        mscr.flag_manualremoval_test(remove_dates=['2024-07-01 20:10'])
        mscr.correction_setto_value(dates=['2024-07-01 20:10'], value=3.7)
    assert out.warnings == []
    assert int((mscr.outlier_detection[FIELD].last_flag == 2).sum()) == 1


# --- Item 8: QCF reports count records, not empty slots ------------------------------

MISSING_1MIN = {'10min_then_1min': '2024-07-03 12:00', '1min_then_10min': '2024-07-02 12:00'}


def _report_number(text: str, label: str) -> int:
    m = re.search(rf'{label}[^:\n]*:\s*(\d+)', text)
    assert m, f'{label!r} not in report'
    return int(m.group(1))


@pytest.mark.parametrize('order', ORDERS)
def test_qcf_reports_count_records_not_empty_slots(order):
    """Item 8: one 1-min record is missing from the download. The reports count it
    as the only missing record; the empty slots of the fine grid within 10-min
    records are no records at all: not potential, not missing, QCF NaN."""
    missing_end = pd.Timestamp(MISSING_1MIN[order])
    sc = Scenario(id=f'qcf_{order}', missing=[str(missing_end)], **MIXED_BY_ORDER[order])
    raw, _ = _download_frame(sc)
    raw.loc[pd.DatetimeIndex(sc.abslim_spikes), FIELD] = ABSLIM_SPIKE
    n = len(raw)
    mscr = _screening(raw)
    mscr.start_outlier_detection()
    mscr.flag_outliers_abslim_test(**ABSLIM, showplot=False)
    mscr.addflag()
    mscr.flag_missingvals_test()
    mscr.finalize_outlier_detection()
    qcf = mscr.outlier_detection_qcf[FIELD]

    potential = _slots(mscr, raw.index.union([missing_end]))
    empty = qcf.flags.index.difference(potential)
    assert len(empty) > 0
    flags = qcf.flags
    assert flags[qcf.flagqcfcol].reindex(potential).notna().all()
    # Rejected: the abslim spikes and the missing record.
    assert flags[qcf.flagqcfcol].reindex(potential).eq(2).sum() == len(sc.abslim_spikes) + 1
    for col in (qcf.flagqcfcol, qcf.sumflagscol, qcf.sumhardflagscol, qcf.sumsoftflagscol):
        assert flags[col].reindex(empty).isna().all(), f'{col} not NaN at empty slots'
    assert mscr.data_detailed[FIELD][qcf.flagqcfcol].reindex(empty).isna().all()

    table, _ = qcf.screening_report()
    overall = table[table['period'] == 'OVERALL']
    assert (overall['n_potential'] == n + 1).all()
    assert (overall['n_measured'] == n).all()
    assert overall['n_rejected'].iloc[-1] == len(sc.abslim_spikes)
    first = table.groupby('period')['n_potential'].first()
    assert first['DAYTIME'] + first['NIGHTTIME'] == n + 1

    with _captured() as out:
        qcf.report_qcf_series()
    assert _report_number(out.text, 'Potential records') == n + 1
    assert _report_number(out.text, 'Measured records') == n
    assert _report_number(out.text, 'Missing records') == 1

    with _captured() as out:
        qcf.report_qcf_evolution()
    assert _report_number(out.text, 'Measured records') == n

    # Report 1A (all records): the missing-values test passes n records, fails one,
    # and has no NaN flag (an empty slot would be one).
    with _captured() as out:
        qcf.report_qcf_flags()
    lines = out.text.splitlines()
    start = next(i for i, line in enumerate(lines) if 'REPORT 1A' in line)
    header = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith('_MISSING'))
    row = next(lines[i] for i in range(header, len(lines)) if 'OVERALL' in lines[i])
    n_pass, _, n_fail, n_nan = (int(x) for x in re.findall(r'(\d+) \(\s*[\d.]+%\)', row))
    assert (n_pass, n_fail, n_nan) == (n, 1, 0), row
