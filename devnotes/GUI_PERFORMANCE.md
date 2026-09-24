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
- **The splash offscreen:** `QSplashScreen.show()` blocks about 1 s on the offscreen platform,
  so offscreen "splash painted" times are about 1 s later than on a real screen.

## Current timings (2026-09-24, `329c2612`)

| Action | Freeze | Ready |
|---|---|---|
| Open Ridgeline plot tab | 8.2 s | 8.9 s |
| Open Shifted distribution plot tab | 4.2 s | 4.6 s |
| Load example data + first Overview render | 2.8 s | 3.7 s |
| Data change with 62 tabs open (only the Overview renders) | 2.2 s | 2.2 s |
| Overview: click a variable | 1.5–2.1 s | 1.5–2.1 s |
| Open Select variables | 1.8 s | 3.4 s |
| Open Flux processing chain (first open, lazy imports) | 1.3 s | 1.3 s |
| Theme change / open Appearance tab | 1.3 s | 1.7 / 2.3 s |
| Open Driver explorer / Stepwise screening / Spectrogram | 0.9–1.0 s | 1.7–2.2 s |
| Open Time series, Diel cycle, Cumulative, Cumulative year plot tabs | 0.7–0.8 s | 1.4–1.8 s |
| Open Absolute limits filter / Gaps & coverage | 0.7–0.8 s | 0.7–0.8 s |
| Open Seasonal trend (first run starts the worker process) | 0.4 s | 4.1 s |
| About 35 other tabs (outlier, correction, flux, gap-filling, …) | under 0.2 s | under 0.3 s |

Before this work (same data, measured more loosely): an Overview click took 9–12 s, a data
change with 62 tabs open 39 s, a theme change 8–14 s, and each resize step of a window drag
about 1.6 s.

---

## Open

### [ ] P1. Ridgeline plot tab takes 8.2 s to open
`diive/core/plotting/ridgeline.py`, `diive/gui/tabs/plotting.py`. The worst item by far, not
yet looked at. Profile first; likely per-group density estimates or artists at full resolution.

### [ ] P2. Shifted distribution plot tab takes 4.2 s to open
`diive/core/plotting/` (`ShiftedDistributionPlot`). Not yet looked at. Profile first.

### [ ] P3. Opening Select variables freezes for 1.8 s
`diive/gui/tabs/variable_selector.py`. Odd for a list picker: it opts into the full record
(`wants_full_data`) and may trigger a push or rebuild it doesn't need. Cause unknown.

### [ ] P4. A theme change still restyles the whole app (1.3 s); opening Appearance costs the same
`diive/gui/theme.py` `ThemeManager.apply()` always calls `app.setStyleSheet`, which re-polishes
every widget. The list width and heatmap colormap are not part of the stylesheet, so a change to
either needs only the `changed` signal. Check why opening the Appearance tab triggers a restyle.

### [ ] P5. Overview: a variable click still takes 1.5–2.1 s; the first render 2.8 s
`diive/gui/tabs/overview.py`. Remaining parts, measured by the agent that did the last round:
constrained layout about 320 ms per click; `dv.sstats` for the stats band 60–110 ms;
`diel_cycle` (`core/times/resampling.py`) builds `datetime.time` objects, about 56 ms per call
(once per click and once per zoom settle); `loc='best'` of the diel and histogram legends about
35 ms per pan step.

### [ ] P6. Opening a project blocks for one Overview render (about 2.6 s)
`diive/gui/app.py`. The file read runs on a worker; the render that follows does not. Tied to P5.

### [ ] P7. `import diive` takes about 1.9 s before the splash can show
`diive/__init__.py`. `diive.gui` sits inside the package, so this import always runs first.
Largest costs: `sstats` pulling in `scipy.stats` (about 0.8 s) and the example-data module
pulling in pandas (about 0.5 s).

### [ ] P8. First open of the Flux processing chain tab costs 1.3 s
Menu tabs are imported on first open (`registry.LazyTab`), so this tab pays for importing the
flux chain modules then. The cost moved here from startup on purpose. Could be hidden by
pre-importing in the background after startup.

### [ ] P9. Plot tabs and preview heatmaps render at full resolution
The Time series, Diel cycle, Cumulative and Cumulative year tabs freeze 0.7–0.8 s on open. The
Overview's speed-ups are not used elsewhere yet: `HeatmapDateTime.plot(as_image=True)` is used
only by the Overview, while the Heatmap plot tab and the previews in Combine variables, the
derived-variable tabs, the flux chain and the gap-filling tabs still draw a `pcolormesh`;
`decimate_line` (`core/plotting/plotfuncs.py`) is used only for the Overview time series.

### [ ] P10. Colour-coded scatter in the GUI still takes about 1.2 s
`diive/core/plotting/scatter.py` `_OccludedMarkerCollection`. At GUI size fewer markers are
fully covered than in a small figure (52k of 175k are drawn), and drawing those takes about 1 s.

### [ ] P11. Seasonal trend and Spectrogram redraws
The Seasonal trend redraw costs 0.37–0.42 s (constrained layout and ticks on four date axes).
The Spectrogram's `shading="gouraud"` mesh costs 0.4–0.8 s per draw; drawing only the rows up to
max cycles/day would roughly halve it, but raising the limit would then need a rebuild.

### [ ] P12. Driver explorer: 0.36 s left
Mostly drawing the scatter markers; the legend search is now about 0.02 s.

### [ ] P13. Hover repaints the whole canvas on every mouse move
`diive/gui/widgets/hover.py`: it restores the full-figure background and blits `fig.bbox`, so Qt
repaints the entire canvas per move. Blitting only the old and new annotation areas, and
coalescing mouse moves, would cut that. Line lookups on lines that are not thinned still scan
the full array per move.

### [ ] P14. Variable lists are rebuilt row by row
`diive/gui/widgets/variable_panel.py`: every metadata change removes and re-adds all rows
(`takeItem`, quadratic), with no `setUniformItemSizes`; the delegate builds fonts and colours on
every paint. Cheap at 37 columns, noticeable above a few hundred.

### [ ] P15. Stylesheet churn and card rebuilds
About 175 per-widget `setStyleSheet` calls; `SubTabs.add_page` restyles all buttons on every add;
the Overview hero band rebuilds styled chips per click; the Events tab rebuilds every card on each
keystroke in its filter; `_on_events_changed` rebuilds every event flag on a visibility-only
toggle.

---

## Done

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
