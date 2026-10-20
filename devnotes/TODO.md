# To do: planned features

Features diive does not support yet, collected for future releases. Bugs go in
`CODE_REVIEW_FINDINGS.md`. When an item here is done, remove it and add a line to
`CHANGELOG.md`.

## Time series with monthly, weekly or yearly records

**Status:** not supported (decided 2026-10-20, before v0.92.0). Records must be
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
