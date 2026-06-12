# diive GUI — User Manual

> ⚠️ **Work in progress.** The diive desktop GUI is under active development —
> features and layout may change, and you may hit rough edges. Feedback welcome.

A point-and-click desktop app for exploring time series data with diive.

---

## Install & launch

```bash
pip install 'diive[gui]'     # or: uv sync --extra gui
diive-gui
```

A splash screen with a loading spinner appears while the app starts up. diive
then **reopens the project you had open last**; if you haven't saved one yet, the
bundled example dataset (CH-DAV, 37 variables) loads automatically so you can try
everything right away. (You can see the splash again any time via **Help ▸ About**.)

---

## The basics

### Loading your own data

**File ▸ Open data file…**

1. **Browse…** and pick one *or more* files.
2. Choose the **file type** (EddyPro FLUXNET 30-min is starred ★ at the top; Parquet
   is auto-selected for `.parquet`).
3. Check the **preview** of the first parsed rows, then **Load**.

Selecting several files of the same type **merges** them into one dataset. The
filetype you used is remembered for next time.

### Focusing on a date range

**Data ▸ Select date range…** (Ctrl+R) narrows the dataset to a *from / to*
window. Pick the two dates (the pickers start at the full span and can't go
outside it) and **OK** — every tab, plot, and processing step then works on just
that window, and saving writes only that window.

This is **non-destructive**: the full record is kept in the background, so
**Data ▸ Reset to full range** brings everything back at any time. Features you
engineer while a range is active are kept too, and reappear when you reset.

### Saving your work as a project

A **project** keeps everything together so you can pick up exactly where you left
off. **File ▸ Save project as…** (Ctrl+Shift+S) asks for a name and a location and
writes a `<name>.diive` folder containing:

- the dataset (including features you engineered) as parquet,
- all your **metadata** — variable tags, notes, and the full origin/processing
  history,
- the **project settings** (author, description, site details, and your sticky
  **notes**) and the active **date range**,
- the **open tabs** (your workspace — the same tabs, in order, with pins, reopen
  when you load the project, each restored to its **selected variable(s) and
  settings**; the tab you had active regains focus),
- the **Overview** state (its selected variable and any variable subset),
- a `__diive__` marker that identifies the folder as a diive project.

**File ▸ Open project…** (Ctrl+Shift+O) loads a `.diive` folder and restores all of
the above. Once a project is open, **Ctrl+S (File ▸ Save project)** updates it in
place; the window title shows the project name. diive reopens your most recent
project automatically the next time you launch it.

### Saving just the data

**File ▸ Save data as parquet…** writes only the current dataset — including any
features you engineered — as a **diive-format parquet** file: a single header row
and a properly named timestamp index (`TIMESTAMP_MIDDLE` / `TIMESTAMP_END` /
`TIMESTAMP_START`; you're asked which if it isn't already set). These files load
straight back into diive (GUI or library).

### The variable list (left side, every tab)

- **Filter box** — type to narrow the list. Matching is fuzzy and ignores
  underscores/case: `gpp16` finds `GPP_CUT_16_f`.
- **Tag pills** — variables are colour-tagged by kind: NEE/FC (green), GPP (blue),
  Reco (red), LE/ET (purple), radiation/Rg/SW_IN/PPFD/PAR/LW (orange), air
  temperature TA (deep orange), VPD (cyan), soil water content SWC (brown).
  Features you create get a pink **✦ NEW** pill.
- **Metadata indicators** — a gold **★** marks a favorite, and a small **●N**
  shows how many extra tags a variable has. **Favorites sort to the top** of the
  list. **Hover** any variable for a tooltip with its origin, where it came from,
  its tags, any note, and its full processing history.
- **Right-click a variable** for **Edit metadata…** (jumps to the Metadata
  explorer focused on that variable), **Rename…**, **Delete…**, **★ Mark
  favorite**, and **Add tag… / Remove tag**. **Rename** and **Delete** work in
  **every** tab's list: rename asks for a new name (and the variable keeps all
  its tags, notes, and history under the new name); delete drops it from the
  loaded dataset. Both are **non-destructive** to the source file — reload or
  reopen to restore. Full metadata editing (tags, a free-text note, history)
  lives in **Data ▸ Metadata explorer**.
- The list looks and behaves the same in every tab.

---

## Tabs

The window opens with **Overview** and **Log** (these two stay open — they have no
close button). Other tabs open from the menus and can be closed (×). Most can be
opened **multiple times** — you'll get *Heatmap 1*, *Heatmap 2*, etc. You can
**drag tabs to reorder** them and **double-click a tab (left button) to rename**
it.

**Pin a tab to freeze its data:** right-click a tab → **Pin tab (freeze data)**.
A pinned tab keeps the dataset it currently shows and ignores later changes (a new
file, a date-range change, added features) — handy for comparing while you change
the data elsewhere. A pin marker appears on the tab; right-click → **Unpin** to let
it follow the data again. (Overview and Log are always live.)

### Overview (first tab)

Click a variable to see, for that variable:
- a **figure** with several panels — full time series (the variable name sits in a
  badge in its top-left), cumulative sum (shaded to zero), a per-month diel cycle,
  the daily mean, and a date/time heatmap;
- a **ribbon of statistics** along the bottom (count, mean, SD, min/max,
  percentiles, …); **hover** any one for a short description of what it is.

### Plot menu (heatmaps · time series · diel cycle · cumulative · ridgeline · scatter · hexbin · histogram)

Each plot method opens as its own tab (the menu shows a small icon for each).

- **Heatmap date/time** — date × time-of-day grid.
- **Heatmap year/month** — one cell per year × month (pick the aggregation —
  mean, sum, … — and optionally show *ranks*).
- **Time series** — the variable as a line over time. Ctrl+click more variables
  to stack them in extra panels (shared time axis), each its own colour.
- **Diel cycle** — the mean daily cycle (value by time of day) with a ±SD band;
  optionally one curve per month.
- **Cumulative year** — one cumulative-sum curve per year (overlaid by day of
  year); optionally **highlight a year** (chosen from a dropdown of the years
  present in the data) and show a mean reference.
- **Scatter XY** — click two variables for X and Y (a third, optional, colours
  the points); optionally bin the x-axis and show a trend. One panel.
- **Hexbin** — like Scatter XY but for very dense data: the plane is tiled with
  hexagons coloured by how many points fall in each. Pick variables by **X / Y / Z**
  role (it needs all three; Z drives the colour).
- **Ridgeline** — one stacked density curve per period (group by month, week, or
  year); set the overlap, shading, and KDE bandwidth. One variable at a time.
- **Histogram** — the distribution of one variable: bars of counts, with the
  peak bin highlighted and a z-score scale along the top. Set the number of bins
  and toggle the counts, info box, z-score axis, title and grid.
- **Click** a variable to plot it.
- **Ctrl + click** more variables to compare them in extra panels (up to 5):
  - *Heatmaps* line up **side by side** (shared axes).
  - *Time series* **stack** top-to-bottom (shared time axis), each its own colour.
  - *(The ridgeline and histogram show a single variable, so Ctrl+click just switches it.)*
- **Ctrl + click** a shown variable again to remove its panel.
- Use the small toolbar (bottom-right of the plot) to **pan, zoom, and save** the
  figure. Zooming one panel zooms them all. Set **Save DPI** (next to the toolbar)
  before saving for a higher-resolution image than the screen.

**Settings (middle column).** Between the variable list and the plot is a panel
of controls for the plot. Adjust as many as you like, then click **Update plot**
(below the controls) to apply them all at once — the plot does not change while
you are still editing. (Clicking a *variable* in the list, by contrast, updates
the plot immediately.) Available controls:
- *Heatmap*: colormap (with a **Reverse colormap** toggle), min/max colour values,
  missing-value colour, orientation (vertical/horizontal), date-axis ticks, grid,
  colorbar (show, label, decimals, extend arrows), and optionally overlaying the
  numeric values on the cells.
- *Time series*: title, line width, opacity, point markers (and marker size),
  whether to connect across gaps, and the axis labels/units.
- *Diel cycle*: mean/±SD band, one curve per month (each month gets its own
  colour), legend position, and labels.
- *Scatter XY*: marker size and opacity, the Z colormap (with **Reverse** and a
  Z min/max), an optional title, and bin aggregation.
- *Axes* (time series, scatter, cumulative year; Y-only for the diel cycle): set
  **X/Y min/max** (blank = automatic), **log** scaling, **invert Y**, and add a
  **grid**.

  Line *colours* for time series come from **Settings ▸ Appearance** (so a
  variable keeps the same colour everywhere).

### Analyze ▸ Gaps & coverage

A dashboard for finding and inspecting **missing data**. Pick a variable on the
left (it opens on the one with the most gaps); the right side shows:

- **Stat cards** — overall missing %, number of gap periods, long gaps, the
  longest gap and its duration, and the worst month.
- A **gap map**: a daily-availability heatmap (green = data, red = missing) over
  a timeline where each gap is a spike (taller = longer).
- A **table of the longest gaps** (start, end, length, duration).

The map is **clickable**: click a row in the table to highlight that gap on the
timeline, or click anywhere on the timeline to jump to the nearest gap (it gets
highlighted and its table row is selected). Use **Long gap ≥ (records)** to set
what counts as a "long" gap (48 records = one day for half-hourly data).

### Analyze ▸ Driver explorer

Find **what relates to a variable** — useful before gap-filling or interpreting
a flux. Pick a **target** on the left; the table ranks every other variable by
how strongly it correlates with it (the **r** value is tinted green for positive,
red for negative, stronger = more saturated). **Click a driver** to see the
target-vs-driver scatter.

- **Method** — *Pearson* (straight-line relationships) or *Spearman* (any
  consistently increasing/decreasing relationship).
- **Max lag (records)** — scan a lead/lag window and report each driver's
  *strongest* lag (0 = same time only). A positive **Lag** means the driver
  leads the target (e.g. radiation now → flux a bit later). The scatter is shown
  at that lag.
- Changing the method or max lag takes effect on **Rank drivers**; picking a new
  target updates immediately.

### Analyze ▸ Seasonal-trend & anomalies

See whether a variable is **changing over the years** and which years stood out.
Pick a variable; it is split into:

- **Trend** — the slow, long-term direction.
- **Seasonal** — the repeating yearly cycle.
- **Residual** — what's left (noise, events).

Switch **View** to **Yearly anomalies** to see each year compared to a
**reference period** (red bars = above the reference mean, blue = below) — handy
for spotting warming or unusually wet/dry years. Set the reference years with the
two boxes.

- **Method** — *STL* (robust, recommended), *Classical*, or *Harmonic*.
- **Robust** (STL) — down-weights outliers; more faithful but slower.
- Method/Robust changes apply on **Update**; variable, view and reference years
  update immediately.
- Needs at least ~2 years of data for the decomposition (the anomaly view works
  with fewer).

### Analyze ▸ Spectrogram

See **when** a variable's cycles are strong. A spectrogram shows time along the
bottom, frequency (cycles per day) up the side, and colour for power — a bright
horizontal band at **1 cycle/day** is the daily rhythm, and it usually
strengthens in the growing season. An explanation is shown above the plot.

- **Window (records)** / **Overlap %** / **Window** — how the series is split for
  the analysis; a wider window gives finer frequency detail but blurs timing.
  These apply on **Update**.
- **Max cycles/day** sets how far up the frequency axis to look; **Colormap**
  changes the colours — both update immediately.

### Outliers ▸ Hampel filter

Detect spikes with the **Hampel filter** (a robust, median-based test). Pick a
variable; the top panel shows it with detected outliers marked, the bottom panel
shows the cleaned copy (outliers removed). Your original variable is never changed.

- **Window (records)** — how many neighbouring points define "local" (624 ≈ 13
  days at half-hourly sampling).
- **n sigma (global)** — how far from the local median counts as an outlier
  (lower = stricter, flags more).
- **Use double-differencing** — removes trends first (Papale 2006); leave on for
  most flux/meteo data.
- **Repeat until no more outliers** — re-runs until clean; a **progress bar**
  fills as each pass removes fewer points.
The **Preview** section holds two display options (they don't change the result):

- **Live preview** *(on by default)* — both panels update after every iteration:
  the top panel grows its outlier markers pass by pass (red/blue when day/night is
  on), and the bottom panel shows the cleaned series with the points removed *this
  pass* flagged by a red **X**. Detection runs faster with this **off** — it then
  renders once at the end.
- **Show limit lines** *(off by default)* — overlay the upper/lower limits that
  decide what's an outlier, as faint lines (upper dashed, lower dotted; red =
  daytime, blue = nighttime, slate when not separating), with matching legend
  entries. Every iteration's band accumulates in the top panel (a tightening
  envelope); the bottom panel shows just the current pass's band. Shown in data
  units even with double-differencing on. Toggling it re-renders the last result.
- **Separate daytime / nighttime** *(optional)* — use **different** thresholds for
  day and night (the point of separating: with the same value for both, the result
  is identical to not separating). Set **Daytime n sigma** / **Nighttime n sigma**
  (they start from the global value, then edit each); the coordinates default from
  **Settings ▸ Project settings**. Day and night outliers are then drawn in **red** and
  **blue**, and the status line reports how many of each were found.

**Detect outliers** runs the filter; the status line reports the total and the
number of iterations. **Add cleaned + flag to dataset** adds two new columns
(`{var}_HAMPEL` and its flag) to the variable list. **Copy Python** (quiet link at
the bottom) puts the equivalent diive script on the clipboard.

### Outliers ▸ Local SD filter

Flags points that deviate from a **rolling-window median** by more than a number
of standard deviations. Same two-panel preview, live preview, limit lines,
day/night colouring, **Add to dataset**, and **Copy Python** as the Hampel tab —
only the parameters differ:

- **Window (records)** — the rolling window for the median and SD (seeded to ~5%
  of the series length when data loads).
- **n SD (global)** — how many standard deviations from the median count as an
  outlier (lower = stricter).
- **Constant SD (whole series)** — use the SD of the entire series instead of a
  rolling SD within the window.
- **Separate daytime / nighttime** — set a window and n SD per period (these
  become `[daytime, nighttime]` lists for the library).

Adds `{var}_LOCALSD` and its flag to the variable list.

### Outliers ▸ Z-score filter

Flags points whose absolute **z-score** (deviation from the mean in units of
standard deviation) exceeds a threshold. Same two-panel preview, live preview,
limit lines, day/night colouring, **Add to dataset**, and **Copy Python** as the
Hampel tab — only the parameters differ:

- **Threshold (global)** — flag a point when its absolute z-score exceeds this
  value (lower = stricter).
- **Separate daytime / nighttime** — the z-score is then computed separately for
  daytime and nighttime records (each with its own mean and SD), with a
  **threshold per period** (seeded from the global value).

Adds `{var}_ZSCORE` and its flag to the variable list.

### Outliers ▸ Z-score (rolling) filter

Like the z-score filter, but the mean and standard deviation are computed in a
**rolling window** centred on each point, so the band adapts to local
variability — useful for non-stationary series. No day/night mode (the rolling
window already adapts). Parameters:

- **Threshold** — flag a point when its absolute rolling z-score exceeds this
  value (lower = stricter).
- **Window (records)** — the rolling window for the mean and SD (seeded to ~5% of
  the series length when data loads).

With **Show limit lines** on, the band's centre — the **rolling mean** the band is
built around — is drawn as a solid line between the upper/lower limits.

Adds `{var}_ZSCOREROLLING` and its flag to the variable list.

### Outliers ▸ Z-score (increments) filter

Targets **abrupt changes**: a point is flagged only when the z-scores of its
forward, backward, *and* combined increments all exceed the threshold. This
isolates spikes while tolerating gradual change. No day/night mode. **No
detection-limit band** (the **Show limit lines** option has no effect here): the
test is on increment z-scores rather than the values, and because a point is
flagged only when *all three* increments are extreme, the accepted region is a
union of intervals — not a single upper/lower envelope — so there is no
data-unit band to draw. Parameter:

- **Threshold** — the z-score threshold applied to each increment series; a point
  is flagged only when all three exceed it (lower = stricter).

Adds `{var}_ZSCOREINCREMENTS` and its flag to the variable list (the flag column
keeps the library's `FLAG_{var}_OUTLIER_INCRZ_TEST` name).

### Outliers ▸ Trim-low filter

Removes low outliers with a **symmetric trim**: it rejects the values below a
**lower limit**, then rejects an equal number of the **highest** values — keeping
the distribution balanced (the trimmed-mean rationale). Because some rejected
points sit *above* the limit, use a one-sided method (Absolute limits) instead if
you only want to drop the low extremes. **No detection-limit band** (the **Show
limit lines** option has no effect): the high tail is removed by position, so the
kept set is not a single upper/lower envelope. Parameters:

- **Lower limit** — values below this are rejected (plus the matching count of
  highest values). Seeded from the selected variable's minimum, so nothing is
  trimmed until you raise it.
- **Trim daytime only** / **Trim nighttime only** — *optional*. Leave both off
  (the default) to trim the **whole series** against one distribution. Tick one or
  both to restrict (and split) the trim to those periods, each screened against
  its own distribution.
- **Latitude / Longitude / UTC offset** — only used (and enabled) when a day/night
  box is ticked; seeded from **Settings ▸ Project settings**.

Adds `{var}_TRIMLOW` and its flag (`FLAG_{var}_OUTLIER_TRIMLOW_TEST`) to the
variable list.

### Outliers ▸ Absolute limits filter

The simplest, one-sided test: flag everything **outside a fixed range**. Set a
**Min** and **Max** and any value below the min or above the max is rejected — the
limits are drawn on the preview as the detection band. Use it for hard physical
constraints (e.g. relative humidity must be 0–100 %). Optionally split by day and
night to enforce different ranges per period. Same preview, day/night colouring,
**Add to dataset**, and **Copy Python** as the other tabs.

Adds `{var}_ABSLIM` and its flag to the variable list.

### Outliers ▸ Local Outlier Factor filter

A density-based test (`LocalOutlierFactor`): a point is an outlier when its local
density is substantially lower than that of its nearest neighbours. Parameters:

- **Neighbors** (`n_neighbors`) — how many nearest neighbours define the local
  density (default 20).
- **Contamination** — the expected fraction of outliers (default 0.01), or tick
  **Auto** to use the threshold from the LOF paper.

Adds `{var}_LOF` and its flag to the variable list.

### Outliers ▸ Manual removal

Remove **known** bad records by hand — no statistics. List the timestamps and/or
periods to drop (a calibration window, a sensor failure you already know about),
then **Flag listed dates**. There is no detection band and no day/night mode; the
selection is just the records you named.

Adds `{var}_MANUAL` and its flag to the variable list.

### Outliers ▸ Stepwise screening

Chain several outlier tests on one variable and see what each step removes. Unlike
the single-method tabs (one detector, one pass), each committed step runs on the
data the previous steps already cleaned, so spikes are peeled off progressively.
Build the chain step by step, inspect the per-step removals, and check the overall
**quality flag (QCF)** — computed separately from the accumulated per-test flags.
**Add cleaned + flags + QCF to dataset** appends the cleaned series, every step's
flag, and the QCF-filtered series.

### Data ▸ Select variables

Pick a subset of variables to focus the **Overview** list on. Click a variable on
the left (*Available*) to move it to the right (*Selected*); click one on the right
to remove it. **Add all →** (under the Available list) moves everything across;
**Clear** (under the Selected list) empties the selection. **Confirm → update
Overview** restricts the Overview's variable list to your selection (your data is
not changed — load new data or re-open to reset).

### Data ▸ Rename variables

Add a common **prefix and/or suffix** to **all** variables at once — e.g. tag
every column with a site code (`CH-DAV_…`) or a year (`…_2024`). Type a prefix
and/or suffix; the table **previews** the old → new names (changed ones in bold)
before anything happens. **Apply rename** commits it to the loaded dataset.

To rename just **one** variable, **double-click its name** in the table (same as
the right-click **Rename…** on any variable list). Renaming is non-destructive to
the source file, and each variable keeps its tags, notes, and history under the
new name.

### Data ▸ Metadata explorer

See and edit the metadata that travels with each variable — useful once a
variable has been through several steps (load → outlier filter → gap-fill → …)
and you want to know *where the current version came from*. Pick a variable on
the left; the right panel shows:

- **Origin** — *original* (straight from the file), *modified* (a transformed
  copy, e.g. outliers removed), or *derived* (computed from a parent, e.g. a
  flag), plus the **parent** variable it came from.
- **Tags** — toggle **★ Favorite**, and add/remove your own tags (each user tag
  gets its own colour automatically). Operations also add tags themselves (e.g.
  `hampel`, `flag`). **Clear this variable's tags & note** removes just this
  variable's custom tags and note.
- **Note** — free text describing the variable, up to **50 words** (a live
  counter shows the count; **Save note** greys out once saved and re-enables when
  you edit again).
- **History** — the ordered list of operations that produced the variable, each
  with its settings and time; for a loaded variable the first entry is its import.

**Right-click a variable** in this tab's list for **Remove all tags & note** to
clear just that one variable. **Clear all tags & notes** (bottom of the tab) does
the same for every variable in the current dataset at once (after a confirmation).
Either way, auto-assigned tags, origin, and history are kept.

The bundled **example data always opens clean** (no tags or notes) — tags and
notes are kept only for data you load yourself.

**Your tags and notes are saved between sessions, per dataset** — the same column
name in two different files keeps separate tags. Origin/history are recomputed
each session as you work.

### Data ▸ Feature engineering

Build new features (lags, rolling stats, differences, EMA, polynomials, STL,
timestamp parts, …) with diive's feature engineer:

1. **Click** variables to move them into *Selected features*.
2. Tick the stages you want and set their options. *Timestamp features* and
   *Continuous record number* work on the time index alone — they need **no**
   selected variable.
3. **Run feature engineering** — the new columns are listed under *Newly created
   features*.
4. **Add features to variable list** — the new columns appear everywhere with a
   **✦ NEW** pill and can be plotted like any other variable.

When you add new columns (here or from an Outliers tab), the **Overview jumps
straight to the new variable** — it clears any active filter, scrolls the new row
into view, and plots it — so you can see the result right away.

### Flux ▸ Flux processing chain

A guided workspace for the flux processing chain, covering **Input + Level 2 +
Level 3.1 + Level 3.2 + Level 3.3** (only Level 4.1 gap-filling is still to come).
Pick the flux column and site, choose which Level-2 quality tests to run, set the
Level-3.1 storage correction, optionally build a Level-3.2 outlier-detection chain
and apply Level-3.3 USTAR filtering, then run the chain — the accepted
(QCF-filtered) flux of the deepest level shows as a heatmap. **Copy Python** puts
the exact, reproducible diive script for what you did on the clipboard, so a
point-and-click run stays scriptable.

> Needs eddy-covariance input with the raw EddyPro columns (FC, USTAR, the
> `*_TEST` flags). The bundled CH-DAV example is a processed product and won't
> run the chain — load a level-1 EC dataset.

### Settings ▸ Project settings

Settings for the current project:

- **Your name** — the project author.
- **Description** — free-text notes about the project (purpose, data source,
  processing decisions, …).
- **Site details** — the measurement site's **name, latitude, longitude,
  elevation, and UTC offset**.

Fill in and **Save**. The site coordinates and UTC offset are reused wherever
diive needs them (e.g. the Hampel tab's daytime/nighttime split, the flux chain),
so you don't retype them per tool. Everything here is **remembered between
sessions** and **saved with the project** (so it travels inside a `.diive`
folder).

**Notes wall.** The right side of the tab is a pinboard of **sticky notes** for
free-form reminders. **+ Add note** drops a card you can type a **bold title** and
body into; **drag** it by its top bar to arrange it, drag the bottom-right corner
to **resize** it, click **●** to **recolour** it (sticky-note palette or a custom
colour), and **✕** to remove it. The notes save automatically with the project (and
between sessions), so they travel inside the `.diive` folder.

### Settings ▸ Appearance

Customise the **Studio** look (the GUI's single design — near-white surfaces,
pill-shaped tabs, a slim header with drop-down menus) with a **live preview** —
colours update across the whole app as you change them:
- pill colours, selection/hover colours, time-series line colours;
- the variable-list **width** (applies to every tab);
- **Reset to defaults** to undo.

### Log

Mirrors diive's console output (file loading, feature engineering progress, …) in
colour. **Save…** writes the log to a text file; **Clear** empties it.

---

## Tips

- **Editable fields are tinted** (light blue) so you can see at a glance what you
  can change; **hover over a setting** to see a tooltip describing what it does.
- **Hover over any plot** to see the value under the cursor in a small box — on
  line plots it snaps to the nearest data point (with a marker); on heatmaps it
  shows the cell's date, time, and value. Untick **Hover values** (bottom-right,
  next to the plot toolbar) to switch it off.
- Your appearance settings, site details, window size/position, last-used
  filetype, variable tags/notes, and most-recent project are **remembered**
  between sessions.
- The window sizes itself to your screen on first launch.
- A short loading cue appears on a variable while its plot is being drawn.
- Stuck or something looks off? Check the **Log** tab for messages.

---

*Part of the diive library — https://github.com/holukas/diive*
