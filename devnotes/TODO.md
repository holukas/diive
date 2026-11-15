# To do: planned features

Features diive does not support yet, collected for future releases. Bugs go in
`CODE_REVIEW_FINDINGS.md`. When an item here is done, remove it and add a line to
`CHANGELOG.md`.

Items are grouped into bundles that can be worked on together. Bundles are numbered 1, 2, ...
and items 1.1, 1.2, ... When an item is done, remove it and keep the other numbers as they are,
so references to them stay valid. New items get the next free number in their bundle.

Overview:

1. Footprint and wind sectors: 1.1 footprint tab from loaded data, 1.2 target-area flag,
   1.3 wind-direction sector flags, 1.4 footprint in complex terrain
2. GUI: 2.1 wet-night XGBoost in the flux chain tab, 2.2 command palette, 2.3 linked view for
   QC, 2.4 app-wide job list, 2.5 static matplotlib views in a worker process
3. Gap-filling: 3.1 artificial gap scenarios, 3.2 parallel MDS scenarios
4. Ecosystem analyses: 4.1 energy balance closure, 4.2 water flux partitioning, 4.3 GHG budget,
   4.4 phenology metrics
5. Parked: 5.1 monthly, weekly or yearly records
6. NEE partitioning: 6.1 Keenan daytime, 6.2 neural networks
7. Derived meteo variables: 7.1 dewpoint, 7.2 wilting point, 7.3 PET and aridity index
8. Plotting: 8.1 overview plots, 8.2 endless colour wheel, 8.3 warming stripes, 8.4 borderless
   heatmap, 8.5 LakesideSunset colormap
9. QC options: 9.1 negative daytime SW_IN, 9.2 meteo screening from file, 9.3 upper-only or
   lower-only filtering
10. Analysis tools: 10.1 aggregator, 10.2 volatility, 10.3 points of interest, 10.4 xAI,
    10.5 optimum range, 10.6 causal discovery

Bundles 6 to 10 are one line per GitHub issue. The detail is in the issue.

## 1. Footprint and wind sectors

**Bundle status:** model done (2026-11-04, [#444](https://github.com/holukas/diive/issues/444)).
`ffp_footprint` and `ffp_climatology` (`flux/lowres/footprint.py`) reproduce the FFP of Kljun et
al. (2015) bitwise, harness `diive-data/references/footprint_parity/`. The GUI tab Flux ▸ Flux
footprint (2026-11-14) draws the footprint of typed-in values. 1.2 and 1.3 both take input per
wind sector, so they could share one input format.

### 1.1 Footprint tab from loaded data

**Status:** open ([#447](https://github.com/holukas/diive/issues/447)). The tab still lacks the
footprint of a selected record and the climatology of a period from the loaded data. The 3-D view
is undecided. The GUI part of 1.2 also goes into this tab.

### 1.2 Target-area flag

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
- **GUI** (#447, with 1.1): optional fields for the fetch per sector or a GeoJSON file, with the
  footprint climatology drawn over the target area so researchers can check their definition.

### 1.3 Wind-direction sector flags

**Status:** idea (2026-10-30, [#121](https://github.com/holukas/diive/issues/121)). Flag records from excluded wind sectors, e.g. tower shading or
land outside the target area (Hammerle et al. 2007).

### 1.4 Footprint in complex terrain (CH-DAV)

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

## 2. GUI

**Bundle status:** 2.1 closes a gap between library and GUI. 2.2 to 2.5 come from a review of
GUI layout studies (2026-11-14) and change no results. 2.4 and 2.5 both build on the runners in
`gui/widgets/worker.py`.

### 2.1 Wet-night XGBoost in the flux chain tab

**Status:** open (2026-11-10). `run_level41_xgb_wetnight` and `run_chain` (`FluxConfig.gapfill_xgb_wetnight`, `wetnight_wet`, ...) fill the wet night gaps of open-path
analysers (#445), and the code generator renders it, but the GUI flux chain tab does not offer
it. Needs a checkbox at Level 4.1, a picker for the wet column (and the optional dry column and
temperature), and `xgbwn` as a choice for the gap-filled NEE of Level 4.2.

### 2.2 Command palette

**Status:** idea (2026-11-14). The GUI has more than 60 tabs and no way to find one by name. A
search field (Ctrl K) over the tab registry (`MENU_TABS` in `gui/registry.py`) would open any tab
from a few typed letters. Later it could also search actions inside tabs.

### 2.3 Linked view for QC

**Status:** idea (2026-11-14). Show a flux (e.g. NEE) and its drivers (SW_IN, TA, VPD) in stacked
panels with one shared time axis. A crosshair reads every panel at the same timestamp, and zoom
and pan move all panels together. No tab does this yet: the existing `sharex` plots show one
variable before and after a step. Start with matplotlib (`sharex`, `widgets/hover.py`).

### 2.4 App-wide job list

**Status:** idea (2026-11-14). Runners belong to their tab, so nothing shows which tab is running
what. A small panel or popup over `WorkerRunner`, `LatestRunner` and `ProcessLatestRunner` would
list running and finished jobs and link back to the tab with the result. Most useful for long
Flux chain runs.

### 2.5 Static matplotlib views in a worker process

**Status:** idea (2026-11-14), measure first. Render a bare `Figure` with the Agg canvas in a
worker process and show it as an image, with a placeholder while it renders. This would remove
the remaining freezes on tab opens and Overview clicks (0.3 to 1.7 s, see
`GUI_PERFORMANCE.md`). Check against P18 first, which names the current costs. Views drawn as
images lose hover and zoom, so this fits only views without them, and it changes the
synchronous `canvas.draw()` rule for those views.

## 3. Gap-filling

### 3.1 Artificial gap scenarios

**Status:** idea (2026-10-30, [#440](https://github.com/holukas/diive/issues/440)). Insert realistic gaps (lengths, day and night) into measured data
and score the gap-filling methods on them.

### 3.2 Parallel MDS scenarios

**Status:** idea (2026-10-31), for later. `run_level41_mds` fills each u* scenario one after the
other (43 with the REF). The scenarios are independent, so running them in parallel processes
would cut Level 4.1 by roughly the number of cores, with bitwise identical results. Adds to the
compiled MDS loop (#443).

## 4. Ecosystem analyses

New analyses on processed fluxes. Each item stands on its own.

### 4.1 Energy balance closure

**Status:** idea (2026-10-30). Energy balance ratio and residual per period.

### 4.2 Water flux partitioning

**Status:** idea (2026-10-30, [#439](https://github.com/holukas/diive/issues/439)). Partition ET into transpiration and evaporation, e.g. TEA
(Nelson et al. 2020).

### 4.3 GHG budget

**Status:** idea (2026-10-30, [#441](https://github.com/holukas/diive/issues/441)). Combine CO2, CH4 and N2O into one budget, with GWP100 and GWP*
(Allen et al. 2016).

### 4.4 Phenology metrics from GPP and RECO

**Status:** idea (2026-10-30, [#272](https://github.com/holukas/diive/issues/272)). Start, end, length and peak of the growing and respiratory
seasons from smoothed (Savitzky-Golay) daily GPP and RECO (Liu et al. 2020).

## 5. Parked

### 5.1 Time series with monthly, weekly or yearly records

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

## 6. NEE partitioning

- 6.1 Daytime partitioning after Keenan et al. (2019) ([#436](https://github.com/holukas/diive/issues/436))
- 6.2 Partitioning with neural networks after Tramontana et al. (2020) ([#442](https://github.com/holukas/diive/issues/442))

## 7. Derived meteo variables

- 7.1 Dewpoint ([#130](https://github.com/holukas/diive/issues/130))
- 7.2 Wilting point ([#224](https://github.com/holukas/diive/issues/224))
- 7.3 Potential evapotranspiration and aridity index ([#399](https://github.com/holukas/diive/issues/399))

## 8. Plotting

- 8.1 Overview plots ([#44](https://github.com/holukas/diive/issues/44))
- 8.2 Endless colour wheel ([#277](https://github.com/holukas/diive/issues/277))
- 8.3 Warming stripes ([#289](https://github.com/holukas/diive/issues/289))
- 8.4 Borderless heatmap, bug ([#323](https://github.com/holukas/diive/issues/323))
- 8.5 LakesideSunset colormap ([#334](https://github.com/holukas/diive/issues/334))

## 9. QC options

- 9.1 Negative daytime values, e.g. SW_IN ([#32](https://github.com/holukas/diive/issues/32))
- 9.2 Meteo screening from a file ([#262](https://github.com/holukas/diive/issues/262))
- 9.3 Quality flags with upper-only or lower-only filtering ([#336](https://github.com/holukas/diive/issues/336))

## 10. Analysis tools

- 10.1 Aggregator ([#55](https://github.com/holukas/diive/issues/55))
- 10.2 Volatility indicator ([#287](https://github.com/holukas/diive/issues/287))
- 10.3 Find points of interest ([#335](https://github.com/holukas/diive/issues/335))
- 10.4 Expand xAI ([#403](https://github.com/holukas/diive/issues/403))
- 10.5 Refine the optimum range ([#404](https://github.com/holukas/diive/issues/404))
- 10.6 Optional causal-discovery layer in DriverAnalysis ([#424](https://github.com/holukas/diive/issues/424))
