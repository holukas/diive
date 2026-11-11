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

**Status:** model done (2026-11-04, [#444](https://github.com/holukas/diive/issues/444)),
flag open. `ffp_footprint` and `ffp_climatology` (`flux/lowres/footprint.py`) reproduce the FFP
of Kljun et al. (2015) bitwise, harness `diive-data/references/footprint_parity/`. Still open: a
QC flag for records with too little footprint over the target area, and a GUI tab
([#447](https://github.com/holukas/diive/issues/447)).

### Target-area flag

**Status:** planned (2026-11-11, [#448](https://github.com/holukas/diive/issues/448)). Only the site team knows the target area, so diive never
defines or guesses it. The input is optional: without it the footprint runs as now and no flag
is made; with it, each record gets the share of its footprint inside the target area and a flag.

- **Input**, one of, from simple to detailed:
  1. Fetch per wind sector: distance to the edge of the target area per direction, e.g.
     `{0: 300, 30: 250, ...}` in m. Needs no maps; every site team can give it.
  2. A polygon around the target area, in m relative to the tower or in coordinates with the
     tower position (e.g. read from a GeoJSON the researchers draw).
  3. A raster mask (target or not) on the footprint grid, e.g. from a land-cover map.
- **Output:** the footprint share inside the target area per record, and a QC flag where the
  share is below a threshold (configurable, e.g. 80 %), usable in the QCF like the other tests.
- **No input given:** no flag and no warning beyond saying that no target area is defined.
- **GUI** (#447): optional fields for the fetch per sector or a GeoJSON file, with the footprint
  climatology drawn over the target area so researchers can check their definition.

### Footprint in complex terrain (CH-DAV)

**Status:** idea (2026-11-06). A footprint that accounts for topography and, later, individual
trees, using swisstopo data. FFP cannot be adjusted for this: it parameterises backward
Lagrangian simulations over flat, homogeneous terrain, and none of its inputs carries slope,
streamline curvature or flow separation. Over a forested hill, LES-based footprints grow on the
upwind slope and can extend downwind on the lee slope (Fu et al. 2025, JGR Atmos.,
10.1029/2025jd043591). Three stages, each usable on its own:

1. **FFP with terrain and canopy maps (weeks, in diive).** swissALTI3D (DTM 0.5 m) and
   swissSURFACE3D (LiDAR, DSM) give canopy height, d and z0 per wind sector, the effective
   measurement height along the upwind path and slope-corrected area weights. Compute the
   footprint in the planar-fit frame of the fluxes. Still FFP, but with site maps instead of one
   site value: for representativeness, sector filtering and remote-sensing comparison.
2. **3D footprint from a flow model (months, separate repository).** LES or RANS over the
   swisstopo terrain with a canopy layer, plus backward Lagrangian particles from the sonic
   (PALM has topography, LiDAR canopy and a particle model). A case matrix, e.g. 12 wind
   sectors x 3-4 stability classes plus drainage flow, gives a look-up table for the
   climatology. Needs HPC (CSCS). diive would only read the resulting footprint tables.
3. **Single trees and an "FFP for slopes" (research).** Plant area density per voxel from the
   point cloud (Beer-Lambert, Boudreault et al. 2015) in the LES. Tree-scale structure acts up
   to 2-5 canopy heights, so it matters for a tower over a 25-30 m canopy. An LES/LS database
   over idealised slopes could give correction functions for FFP's scaled footprint.

**Validation needed:** a tracer release upwind (Arriga et al. 2017); above- and below-canopy
fluxes at CH-DAV (Krebs et al. 2026); LES flow statistics (u*, sigma_w, wind direction) against
the tower. **Limit:** at night, drainage flow and advection make the tower flux non-local; no
footprint model fixes that (Etzold et al. 2010 for CH-LAE). Best done with a boundary-layer
or LES group.

## Wet-night XGBoost in the GUI

**Status:** open (2026-11-10). `run_level41_xgb_wetnight` and `run_chain` (`FluxConfig.gapfill_xgb_wetnight`, `wetnight_wet`, ...) fill the wet night gaps of open-path
analysers (#445), and the code generator renders it, but the GUI flux chain tab does not offer
it. Needs a checkbox at Level 4.1, a picker for the wet column (and the optional dry column and
temperature), and `xgbwn` as a choice for the gap-filled NEE of Level 4.2.

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

## Parallel MDS scenarios

**Status:** idea (2026-10-31), for later. `run_level41_mds` fills each u* scenario one after the
other (43 with the REF). The scenarios are independent, so running them in parallel processes
would cut Level 4.1 by roughly the number of cores, with bitwise identical results. Adds to the
compiled MDS loop (#443).
