# GUI Performance

Started 2026-09-24 · diive v0.91.1 · branch `indev` (at `329c2612`)

Working document for slow spots in the desktop GUI (`diive/gui/`): what is measured, what is
still open, what was done, and what was tried and rejected. Each fix lands as its own commit and
is recorded in the entry it closes.

Status: `[ ]` open · `[x]` done (with commits) · `[-]` won't do (with the reason).

---

## How to measure

Run `devnotes/gui_timing_pass.py` (see its docstring). It opens the GUI offscreen on the full
bundled example data (CH-DAV, 175,296 half-hourly records × 37 variables, 10 years) in a
1600×1000 window and reports, per action:

- **freeze:** the longest GUI-thread block, i.e. the largest gap between ticks of a 5 ms
  `QTimer`. This is what the user feels.
- **ready:** wall time until the GUI is idle again, including background work.

Pitfalls learned the hard way:

- **Resizing offscreen:** `win.resize(...)` plus `processEvents()` does not reliably reach
  the plot canvas, so a resize can look free when it is not. Resize the `FigureCanvasQTAgg`
  widget itself and drain posted events (`QCoreApplication.sendPostedEvents()`).
- **Measuring a busy machine:** background agents or test runs on the same machine inflated
  the first baseline 2 to 5 times. Measure with nothing else running.
- **The `__main__` guard:** any script that opens the GUI needs it. The Seasonal trend tab
  computes in a spawned worker process, which re-imports the main module; without the guard
  the whole script runs a second time inside the worker.
- **Segfault at the end:** the pass can end with a segmentation fault (exit 139) after it
  printed the SLOWEST list, in `shutdown_process_pool` or `os._exit`. Seen on 2026-10-24 with
  the code of before and after that day, so the results are complete. Not investigated.
- **The splash offscreen:** `QSplashScreen.show()` blocks about 1 s on the offscreen platform,
  so offscreen "splash painted" times are about 1 s later than on a real screen.

## Current timings (2026-10-25, after the P3-P16 round)

Measured while another session worked on the same machine, so the numbers are noisy. The pass
opens the window directly, not through `run()`, so the background `scipy.stats` import and the
flux pre-import (D8) do not show here.

| Action | Freeze | Ready |
|---|---|---|
| Load example data + first Overview render | 2.8 s | 2.9 s |
| Data change with 60+ tabs open (only the Overview renders) | 1.7 s | 1.7 s |
| Overview: click a variable | 0.8–1.7 s | 0.8–1.7 s |
| Open Select variables (first menu tab, see P17) | 1.5 s | 1.5 s |
| Open Flux post-processing chain (first open, no pre-import here) | 1.3 s | 1.3 s |
| Open Gaps & coverage | 0.9 s | 1.0 s |
| Open Stepwise screening | 0.7 s | 0.7 s |
| Theme apply (no token change) / open Appearance | 0.6 s / 0.02 s | 0.7 s / 0.03 s |
| Open Time series / Heatmap date/time | 0.55 s | 0.6 s |
| Open Spectrogram, Cumulative year, Waterfall, Heatmap x/y/z | 0.4–0.5 s | 0.4–0.5 s |
| Open Ridgeline / Scatter XY / Cumulative / Diel cycle | 0.17–0.35 s | 0.19–0.37 s |
| Open Shifted distribution | 0.09 s | 0.10 s |

Before this work (same data, measured more loosely): an Overview click took 9–12 s, a data
change with 62 tabs open 39 s, a theme change 8–14 s, and each resize step of a window drag
about 1.6 s.

---

## Open

### [ ] P17. Offscreen only: the first menu tab grows the tab bar by 2 px
Offscreen, adding the first closable tab makes the tab bar 2 px taller for good (Qt's default
close button, then ours), so the Overview relayouts once (about 1 s, now when switching back to
it). On the Windows platform the bar stays 52 px, so users do not see it. It still shows in the
timing pass as "Open Select variables". Options: a `min-height` on the tab QSS in `theme.py`, or
creating our close button in a `_StudioTabBar.tabInserted` override.

### [ ] P18. Overview line render and layout
Agg strokes the thinned time series at 2 columns per pixel: 330 ms for Rg_f, 144 ms for NEE per
draw, on every select and pan step. Less would change pixels (1 column per pixel showed streaks
in D1). Constrained layout is still about 165 ms per select, mostly tick and text extents.

### [ ] P19. Library costs behind the Overview and previews
`dv.sstats` about 50 ms (nine separate `quantile` calls, about 30 `.loc` setitems; the Overview
uses 10 of 30 metrics). `HistogramPlot` KDE about 47 ms plus 21 tick objects the Overview
replaces. `HeatmapDateTime` `TimestampSanitizer` about 27 ms. `WaterfallPlot` 30-55 ms.

### [ ] P20. Smaller leftovers
- Events filter: a keystroke that changes the matches moves cards between parents (about 120-180
  ms at 100 events). `FlowLayout._do_layout` could skip hidden items (check its other users).
- About 180 `setStyleSheet` calls remain. Repeated ones: `_apply_style` in plotting,
  `_screening_base`, `combine_variables` (on theme change), the `_refresh_availability`
  functions, `metadata_explorer`, `stepwise_cards._restyle`, `flux_pipeline_rail._restyle`.
- Restored projects: each plot tab still renders once on the data push before its restored
  render. Restoring hidden tabs lazily would need `_open_tabs_state` changes.
- Spectrogram: the gouraud triangles in view still take about 0.22 s per draw.
- Hover: about 2 ms per move is matplotlib drawing the tooltip text.
- Spearman ranking in the Driver explorer: `spearmanr` itself dominates.

---

## Done

### [x] D8. Round P3-P16 (2026-10-25)
Seven agents on disjoint files, A/B measured back to back while other work ran.
- **P16 canvas relayout** (`mpl_canvas.py`, `app.py`): a resize back to the size of the last
  render and solve does nothing, the first resize after a render defers its solve to the next
  render, a hidden canvas owes its relayout until shown, and `_open_menu_tab` shows the tab
  before pushing data, so it renders once at its real size. Six plot-tab opens behind the
  Overview 7.3 s -> 2.2 s, Appearance 1.0 s -> 0.04 s. P3 (Select variables) was this.
- **P6/P8** (`app.py`): `scipy.stats` imported in the startup worker (first render 1.9 -> 1.3 s
  through `run()`). After startup a daemon thread imports `diive.flux.postprocessingchain` and
  `diive.flux.partitioning`; tab opens join it (flux chain first open 1.0 s -> 0.25 s). Every
  session now loads the ML stack in the background. Event flags rebuild only when an event
  changes, not on a visibility toggle.
- **P7** (`diive/__init__.py`, `core/dfun/stats.py`): top-level utilities and `__version__`
  resolve lazily, `scipy.stats` imports inside the functions that need it. `import diive`
  1.0 s -> 0.02 s, the splash about 0.9 s sooner.
- **P5** (`overview.py`, `resampling.diel_cycle`): tick memo per view (`_TicksPerView`), legend
  'best' search only on changed inputs (`_PlacedLegend`), `fig.delaxes` before reset, the hero
  band settles before the first render (one draw at startup instead of four), chips reused.
  `diel_cycle` groups by integer time of day, bit-identical, 40 -> 14 ms. Pixel-identical.
- **P4/P15** (`theme.py`, `settings.py`, `sub_tabs.py`, `events.py`): `app.setStyleSheet` only
  when the text changes (theme apply without token change 1.7 s -> 0.01 s), swatches and
  subtab pills restyle only on change, Events cards cached and reused.
- **P13/P14** (`hover.py`, `variable_panel.py`, `variable_delegate.py`): hover blits only the
  old and new tooltip boxes and skips moves on the same sample (line 11 -> 3.7 ms, heatmap
  5.6 -> 1.0 ms per move). Lists reorder with one `sortItems` and uniform item sizes, the
  delegate caches fonts and colours (filter at 3000 columns 2.9 s -> 54 ms).
- **P9** (`plotting.py`, previews): hidden plot tabs render when shown (theme change with 5
  hidden plot tabs 0.6 s -> 0.02 s), restored tabs no longer queue an extra render, the Time
  series line is thinned at 4 columns per pixel and re-thinned on zoom, pan and resize,
  heatmaps and previews draw `as_image=True` (MDS heatmaps 0.5 -> 0.25 s).
- **P10/P11** (`scatter.py`, `spectrogram.py`, `seasonaltrend.py`): the colour-coded scatter
  draws each opaque marker as one edge-free polygon (draw 560 -> 210 ms). This gives up D7's
  pixel identity for that case: up to 9/255 levels in antialiased outlines, kept on purpose.
  Spectrogram draws only the mesh in view (pixel-identical, 355 -> 245 ms), Seasonal trend
  memoizes its date ticks per view (pixel-identical).
- **P12** (`analysis/correlation.py`): `rank_drivers` scans plain arrays with the same
  `np.corrcoef` / `spearmanr` calls, results identical, lag scans 1.5-3.7x faster.

### [x] D0. Ridgeline and Shifted distribution tabs (P1, P2)
Both spent their time in scikit-learn's `KernelDensity.score_samples`, which sums every record at
every grid point. The new `diive/core/plotting/kde.py` (`gaussian_kde_on_grid`) gives the same
Gaussian density with a direct sum limited to 40 bandwidths or, when cheaper, a binned 17-term
Taylor/Hermite series summed by FFT. Curves agree with scikit-learn to about 1e-12 of the peak
(pixel-identical in 20 of 25 compared Ridgeline figures, one antialiasing level in 1-2 pixels in
the others). `RidgeLinePlot` also computed each density twice (y-limit pass and drawing), now
once. Timing pass 2026-10-24, old and new code back to back: Ridgeline open 10.0 s -> 0.37 s
freeze, Shifted distribution 4.5 s -> 0.30 s.

### [x] D1. Overview click, render and pan
An Overview click went from 9–12 s to about 1.5 s, the first render from about 9 s to 2.8 s,
a pan step from about 1.5 s to 0.2–0.4 s.

- `5a2e2248` Waterfall bars and connectors as two collections instead of ~7,300 artists.
- `255f4993` Diel cycle drawn with matplotlib, not pandas: pandas hid the time series' tick
  labels, which cost a second full draw per render and zoom.
- `514e409b` No per-record markers on long series (isolated values keep one).
- `d325588c` Diel cycle and histogram recomputed once a pan or zoom settles.
- `87fb4397` Histogram KDE evaluated faster from all points (repeated values weighted, zero
  kernel terms skipped); matches scipy to 3.4e-12.
- `e154d731`, `a8fc8de3` Heatmap drawn as an image (`as_image=True`).
- `85c0b687`, `d2158d6d` Time series thinned to screen columns (`decimate_line`); the hover
  still reads every record.
- `a0db8f75`, `f2934f82`, `1cc070b5`, `1253fe58`, `4cf44775` Smaller cuts: no unit conversion
  for the waterfall, faster heatmap grid, fewer layout solves and draws, one draw per click.
- `0e1d3be0` Cumulative fill as one path (much faster on gappy series; the fill colour now
  shows instead of grey streaks).
- `a4a90230` Panel decorations kept inside the panels, so the layout works down to 800×550.

### [x] D2. Data changes reach only the visible tab
`39967489`. A data change with 62 tabs open went from 39 s to the Overview's own render; hidden
tabs catch up when shown, and queued renders merge into one.

### [x] D3. Resize
`63e201fd` debounces the layout solve; `c28ab5e8`, `6d3a99b9` defer the render itself, so a
drag paints the last frame scaled and renders once at the end. A 10-step drag went from about
16 s to about 2.7 s.

### [x] D4. Controls that recomputed per step or keystroke
`34c186ab` (`Debouncer`), `f0972271` (Appearance, 3D surface, Select records, Gaps),
`f4f43e2b` (Combine variables colormap).

### [x] D5. Analysis tabs off the GUI thread
`ee0670e1` (`LatestRunner`), `1a3de683` (explorer worker path), `b99f4638` Driver explorer,
`2e2477a8` Seasonal trend, `ef0252a0` Spectrogram, `4e72c591` Data profile.
`9aeabb2f`, `60c041f0`, `e6afac60` move Seasonal trend into a worker process, because
statsmodels' STL fit holds the GIL and froze the window even on a thread.

### [x] D6. Startup and file I/O
`3972734d` menu tabs imported on first open; `8ace65b4` splash before the heavy imports (splash
about 3 s sooner); `6de22c20`, `82055391` project open/save and export on a worker;
`94dea143` parquet preview reads only the first rows; `9ef9dbd4` one Overview render when
opening a project instead of three.

### [x] D7. Scatter and legend
`496dfa1e` uncoloured scatter uses matplotlib's single-marker path (175k points: 2.5 s to
0.4 s); `c68bd099` colour-coded scatter skips fully covered markers, pixel-identical (3.1 s to
0.55 s); `c4dd288c` hover caches scatter pixel positions; `16d4a50a` legend `loc='best'` search
on one array (0.2 s to 0.02 s, same placement).

---

## Rejected approaches

### [-] Subsampling the histogram KDE
Would be faster, but the KDE must use every point (user decision). D1 made the full KDE fast
instead.

### [-] Grouping colour-coded scatter points by colormap entry
Stamping each colour group with `draw_markers` changes which marker is on top, because points
are drawn in data order: about 127k of 640k pixels differ, and it was no faster than D7.

### [-] Drawing uncoloured scatter points as one `Line2D`
Would change the artist type for library users and hover. The fast path was reached with the
`PathCollection` kept (D7).

### [-] Fixed legend location, or 'best' from a subsample
Loses the exact placement and the re-placement on zoom. D7 keeps matplotlib's exact search.

### [-] Killing the worker process to cancel a stale Seasonal trend job
A restart costs about 2 s and a warm robust fit about 3.7 s, so at most about 2 s could be saved,
at the price of a slow next job. Stale results are dropped instead.
