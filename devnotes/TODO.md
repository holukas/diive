# To do: planned features

Features diive does not support yet, collected for future releases. Bugs go in
`CODE_REVIEW_FINDINGS.md`. When an item here is done, remove it and add a line to
`CHANGELOG.md`.

## Time series with monthly, weekly or yearly records

**Status:** not supported (decided 2026-10-20, before v0.92.0), parked for much later (2026-10-24). Records must be
daily or finer.

**What happens now:** loading such data fails (L320).
- `DetectFrequency` reads `'MS'` (month start) as milliseconds and regularizes to a
  1 ms grid, which runs out of memory.
- `'ME'`, `'W'` and `'YS'`/`'YE'` raise ValueError.
- This affects `load_parquet`, `TimestampSanitizer`, the file readers and GUI
  File > Open.
- `resample_series_to_freq` to `'MS'` or `'ME'` also raises
  (`core/times/resampling.py`).

**What support needs:**
- Keep calendar offsets as pandas returns them in `DetectFrequency`. Convert only
  fixed steps (Tick offsets) through `pd.Timedelta`.
- Regularization (`_regularize`, `continuous_timestamp_freq`) already starts a
  calendar grid at the first record. Check the rest of the time handling:
  `TimestampSanitizer`, `common_time_grid`, the timestamp conversions between
  START, MIDDLE and END, and `describe_time_resolution`.
- Decide which tools make sense for such data. Many do not: day/night split,
  diel cycle, half-hourly heatmaps, u* filtering, gap-filling windows.
- Tests with monthly, weekly and yearly data for loading, saving and the GUI open
  dialog.

## Footprint

**Status:** idea (2026-10-30). Footprint estimates, e.g. the flux footprint prediction of
Kljun et al. (2015), and a QC flag for records with too little footprint over the target area.

## Wind-direction sector flags

**Status:** idea (2026-10-30). Flag records from excluded wind sectors, e.g. tower shading or
land outside the target area (Hammerle et al. 2007).

## Water flux partitioning

**Status:** idea (2026-10-30). Partition ET into transpiration and evaporation, e.g. TEA
(Nelson et al. 2020).

## Energy balance closure

**Status:** idea (2026-10-30). Energy balance ratio and residual per period.

## Artificial gap scenarios

**Status:** idea (2026-10-30). Insert realistic gaps (lengths, day and night) into measured data
and score the gap-filling methods on them.

## GHG budget

**Status:** idea (2026-10-30). Combine CO2, CH4 and N2O into one budget, with GWP100 and GWP*
(Allen et al. 2016).

## Phenology metrics from GPP and RECO

**Status:** idea (2026-10-30). Start, end, length and peak of the growing and respiratory
seasons from smoothed (Savitzky-Golay) daily GPP and RECO (Liu et al. 2020).
