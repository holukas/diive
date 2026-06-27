# diive GUI: User Manual

> ⚠️ **Work in progress.** The diive desktop GUI is under active development, so
> features and layout may change and you may hit rough edges. Feedback welcome.

A point-and-click desktop app for exploring time series data with diive.

---

## Install & launch

```bash
pip install 'diive[gui]'     # or: uv sync --extra gui
diive-gui
```

A splash screen with a loading spinner appears while the app starts. diive then
**reopens the project you had open last**. If you haven't saved one yet, the
bundled example dataset (CH-DAV, 37 variables) loads automatically so you can try
everything right away. You can see the splash again any time via **Help ▸ About**.

---

## The basics

### Loading your own data

**File ▸ Open data file…**

1. **Browse…** and pick one *or more* files.
2. Choose the **file type**. EddyPro FLUXNET 30-min is starred ★ at the top;
   Parquet is auto-selected for `.parquet`.
3. Check the **preview** of the first parsed rows, then **Load**.

Selecting several files of the same type **merges** them into one dataset. The
filetype you used is remembered for next time.

### Focusing on a date range

**Data ▸ Select date range…** (Ctrl+R) narrows the dataset to a *from / to*
window. Pick the two dates (the pickers start at the full span and can't go
outside it) and click **OK**. Every tab, plot, and processing step then works on
just that window, and saving writes only that window.

This is **non-destructive**: the full record is kept in the background, so
**Data ▸ Reset to full range** brings everything back at any time. Features you
engineer while a range is active are kept too, and reappear when you reset.

### Saving your work as a project

A **project** keeps everything together so you can pick up exactly where you left
off. **File ▸ Save project as…** (Ctrl+Shift+S) asks for a name and a location and
writes a `<name>.diive` folder containing:

- the dataset (including features you engineered) as parquet;
- all your **metadata**: variable tags, notes, and the full origin and processing
  history;
- the **project settings** (author, description, site details, and your sticky
  notes) and the active **date range**;
- the **open tabs**, meaning the same tabs reopen in order, with pins, each
  restored to its selected variable(s) and settings, and the tab you had active
  regains focus;
- the **Overview** state (its selected variable and any variable subset);
- a `__diive__` marker that identifies the folder as a diive project.

**File ▸ Open project…** (Ctrl+Shift+O) loads a `.diive` folder and restores all of
the above. Once a project is open, **Ctrl+S (File ▸ Save project)** updates it in
place, and the window title shows the project name. diive reopens your most recent
project automatically the next time you launch it.

### Saving just the data

**File ▸ Save data as parquet…** writes only the current dataset, including any
features you engineered, as a **diive-format parquet** file: a single header row
and a properly named timestamp index (`TIMESTAMP_MIDDLE`, `TIMESTAMP_END`, or
`TIMESTAMP_START`; you're asked which if it isn't already set). These files load
straight back into diive, GUI or library.

### The variable list (left side, every tab)

- **Filter box.** Type to narrow the list. Matching is fuzzy and ignores
  underscores and case: `gpp16` finds `GPP_CUT_16_f`.
- **Tag pills.** Variables are colour-tagged by kind: NEE/FC (green), GPP (blue),
  Reco (red), LE/ET (purple), radiation Rg/SW_IN/PPFD/PAR/LW (orange), air
  temperature TA (deep orange), VPD (cyan), soil water content SWC (brown).
  Features you create get a pink **✦ NEW** pill.
- **Metadata indicators.** A gold **★** marks a favorite, and a small **●N** shows
  how many extra tags a variable has. **Favorites sort to the top** of the list.
  **Hover** any variable for a tooltip with its origin, where it came from, its
  tags, any note, and its full processing history.
- **Right-click a variable** for **Edit metadata…** (jumps to the Metadata
  explorer focused on that variable), **Rename…**, **Delete…**, **★ Mark
  favorite**, and **Add tag… / Remove tag**. **Rename** and **Delete** work in
  **every** tab's list. Rename asks for a new name, and the variable keeps all its
  tags, notes, and history under the new name; delete drops it from the loaded
  dataset. Both are **non-destructive** to the source file, so reload or reopen to
  restore. Full metadata editing lives in **Data ▸ Metadata explorer**.
- The list looks and behaves the same in every tab.

---

## Working with tabs

The window opens with **Overview** and **Log**. These two stay open: they have no
close button. Other tabs open from the menus and can be closed (×). Most can be
opened **multiple times**, so you'll get *Heatmap 1*, *Heatmap 2*, and so on. You
can **drag tabs to reorder** them, and **double-click a tab (left button) to
rename** it.

**Pin a tab to freeze its data:** right-click a tab → **Pin tab (freeze data)**.
A pinned tab keeps the dataset it currently shows and ignores later changes (a new
file, a date-range change, added features), which is handy for comparing while you
change the data elsewhere. A pin marker appears on the tab; right-click → **Unpin**
to let it follow the data again. Overview and Log are always live.

The sections below are grouped by **menu**: each menu is a heading, and each of its
entries is a sub-heading.

---

## Overview

The first tab, focused on every load. Click a variable to see, for that variable:

- a **figure** with several panels (the variable name is the figure title): a tall
  **time series** across the top, a full-height **date/time heatmap** down the right,
  and a bottom strip of smaller panels — a **cumulative** sum, a per-month **diel
  cycle**, the **daily mean ± SD**, a **distribution** (histogram with a KDE curve and
  mean/median markers), and a **cumulative waterfall**. The cumulative and waterfall
  panels show their running total inside the axes.
- a **ribbon of statistics** along the bottom (count, mean, SD, min/max,
  percentiles, and more). **Hover** any one for a short description of what it is.

The datetime panels (time series, cumulative, daily mean, waterfall) share an x-axis,
so panning or zooming one zooms them all to the same period; the diel cycle and
heatmap live in their own domains and stay put.

---

## Plot

Each plot method opens as its own tab (the menu shows a small icon for each).
Selecting variables works the same way across the per-variable types:

- **Click** a variable to plot it.
- **Ctrl + click** more variables to compare them in extra panels (up to 5):
  - *Heatmaps* line up **side by side** (shared axes).
  - *Time series*, *diel cycles*, and the *cumulative* **stack** top-to-bottom
    (shared x-axis), each panel with its own settings sub-tab.
  - The ridgeline, histogram, shifted distribution, tree ring, waterfall, and 3D
    surface show a single variable, so Ctrl+click just switches it.
- **Ctrl + click** a shown variable again to remove its panel.
- *Scatter XY* and *Wind rose* instead assign variables to **roles** via the
  dropdowns in their settings (or by **dragging** a variable from the list onto a
  field) — see those sections below. Hexbin and Heatmap x/y/z still pick by clicking
  in order.
- Use the small toolbar (bottom-right of the plot) to pan, zoom, and save the
  figure. Zooming one panel zooms them all; the **Home** button resets to the full
  view. Set **Save DPI** (next to the toolbar) before saving for a
  higher-resolution image than the screen.

**Settings (middle column).** Between the variable list and the plot is a panel of
controls. Adjust as many as you like, then click **Update plot** (just below the tab
header) to apply them all at once: the plot does not change while you are still
editing. The button is greyed out until something changes, and applying the changes
**keeps your current pan/zoom** (it does not snap back to the full view). For the
comparison types, clicking a *variable* in the list updates the plot immediately and
shows it in full; for Scatter XY and the Wind rose, changing a role dropdown waits
for **Update plot** like any other setting.

**Copy Python.** Every plot tab has a **Copy Python** button (top-right) that copies
a runnable script reproducing the current plot with its settings. For tabs that
compare several stacked or side-by-side panels, the script reproduces the active
panel (the one whose settings are showing).

**Per-subplot settings.** When you compare several variables in stacked/side-by-side
panels, a row of **panel pills** appears above the settings (one per panel). Click a
pill to edit *that* panel's settings on their own — line width, colour, title, axes,
everything is independent per subplot. A newly added panel inherits the active
panel's look but is given a distinct line colour automatically.

**Axes** (time series, scatter, cumulative year; Y-only for the diel cycle): set
**X/Y min/max** (blank means automatic), **log** scaling, and **invert Y**. An
explicit limit you set here always wins over the kept pan/zoom. (The grid is toggled
once, in the **Format** group's *Show grid*.)

### Heatmap date/time

A date × time-of-day grid. Settings: colormap (with a **Reverse colormap** toggle),
min/max colour values, missing-value colour, orientation (vertical or horizontal),
date-axis ticks, grid, colorbar (show, label, decimals, extend arrows), and
optionally overlaying the numeric values on the cells.

### Heatmap year/month

One cell per year × month. Pick the aggregation (mean, sum, and so on) and
optionally show *ranks*. Same colormap, colour-range, and colorbar settings as the
date/time heatmap.

### Heatmap x/y/z

A heatmap of a **Z** value over a grid of two driver variables. Click three
variables in the list **in order**: the 1st is the **X** driver, the 2nd is the
**Y** driver, and the 3rd is the **Z** value. The raw data is binned into an X/Y
grid and each cell is coloured by the aggregated Z. Settings:

- **Binning** — *quantiles* (equal counts per bin) or *equal-width* bins, and the
  **number of bins** along each axis.
- **Aggregation** — how to combine the Z values in each cell (mean, median, min,
  max, sum, count).
- **Minimum values per bin** — cells with fewer points are left empty.

Plus the usual colormap, colour-range, colorbar, and cell-value settings. Unlike
**Hexbin**, which tiles the raw points into hexagons, this bins the two drivers into
a rectangular grid.

### Time series

The variable as a line over time. Ctrl+click more variables to stack them in extra
panels (shared time axis), each its own colour. Settings: title, line width,
opacity, point markers (and marker size), whether to connect across gaps, and the
axis labels and units, plus the shared **Axes** controls.

**Line colour.** Pick a colour for the line (and its markers): type a hex code or
matplotlib colour name, click one of the preset swatches, or use **Pick…** for a
colour dialog. `auto` (the default) uses the theme palette colour. With per-subplot
settings, each panel keeps its own colour.

**Colour by another variable.** Instead of a single colour, colour the line by a
second variable's value via a colormap — e.g. a flux line coloured by air
temperature. Choose the variable in the **Color by** dropdown (or **drag** it from
the variable list onto the field) and pick the **Color-by map** colormap; a colorbar
is added. `(none)` returns to the single colour above.

### Diel cycle

The daily cycle (value by time of day), optionally one curve per month. Settings:

- **Aggregation** — the central curve: mean, median, min, max, or the 25th/75th
  percentile.
- **Uncertainty band** — the shaded band around the curve: ±SD, ±SE, IQR
  (25–75 %), Min–Max, or None.
- **Curves** — *one curve per month* (each month its own colour, the seasonal
  pattern) or *one curve overall* (a single curve over all data).
- **Color scheme** — for the per-month curves: the default month palette or a
  colormap (Viridis, Spectral, Turbo, …).
- **Show markers** (and marker size), plus labels and legend position.

Ctrl+click more variables to stack them as extra panels, each with its own settings
sub-tab. Because all panels share the same months, the **legend is drawn once** and
its column count is set automatically (more columns for more months). The fixed 0 to
24 h x-axis is left alone; only the **Axes** Y controls apply.

### Cumulative

The running cumulative total across the whole record, drawn as one continuous
curve. Unlike **Cumulative year**, the sum is not reset at each calendar year, so
the curve keeps climbing (or falling) over the full period. Settings: units, the
number of decimals on the label, whether to show the title, and **shade to zero**
(fill between the curve and the zero line). Ctrl+click more variables to stack them
in extra panels.

### Cumulative year

One cumulative-sum curve per year, overlaid by day of year. The sum restarts each
calendar year (the difference from **Cumulative**, which runs straight through).
Optionally **highlight a year** (chosen from a dropdown of the years present in the
data) and show a mean reference. The shared **Axes** controls apply.

### Scatter XY

Assign **X**, **Y**, and an optional **Colour** variable in the dropdowns at the top
of the settings (or **drag** a variable from the list onto a field) — the same
variable may fill more than one role (e.g. colour the points by X). One panel.
Optionally bin the x-axis and show a trend. Settings: marker size and opacity, the
colour-variable colormap (with **Reverse** and a min/max), an optional title, and bin
aggregation, plus the shared **Axes** controls. Hovering a point shows its X, Y
(and colour) value.

### Hexbin

Like Scatter XY but for very dense data: the plane is tiled with hexagons coloured
by how many points fall in each. Pick variables by **X / Y / Z** role (it needs all
three; Z drives the colour). A **Reverse colormap** toggle is available.

### Ridgeline

One stacked density curve per period (group by month, week, or year). Set the
overlap, shading, and KDE bandwidth. One variable at a time.

### Histogram

The distribution of one variable: bars of counts, with the peak bin highlighted and
a z-score scale along the top. Set the number of bins and toggle the counts, info
box, z-score axis, title, and grid. One variable at a time.

### Shifted distribution

Compares one variable's distribution in a **reference period** against a
**comparison period** — for example an early span of years against a later one, to
see whether the distribution has shifted. The two periods are seeded from the years
present in the data and can be edited. One variable at a time. Settings: the zone
labels and the usual display toggles (legend, title, axes).

### 3D surface

The variable's date × time-of-day grid rendered as a rotatable, GPU-accelerated 3-D
relief, the three-dimensional analogue of the date/time heatmap. The diel band,
seasonal swings, and gaps become hills and valleys you can orbit. Controls:
**Colormap**, **Vertical exaggeration** (height of the relief; 0 is flat), **Smooth
shading**, **Show mesh** (overlay the grid lines), **Smooth terrain** (round the
surface into rolling hills by subdividing the mesh; 0 is off), and **Reset view**.
This needs the optional **`gui3d`** extra (PyVista/VTK); without it the tab shows
install instructions instead of failing.

### Wind rose

A variable aggregated into **wind-direction sectors** and drawn as a polar rose — for
example mean flux, temperature, or concentration by the direction the wind came from.
Assign the **Value**, the **Wind direction** column, and an optional **Colour
variable** in the dropdowns (or **drag** a variable from the list onto a field).
Settings:

- **Aggregation.** How to combine the values in each sector (mean, median, min, max,
  sum, std, count).
- **Sectors.** How many direction bins to split the compass into (default 8).
- **Colour aggregation / colormap / colour range.** When a colour variable is given,
  how to aggregate it per sector and how to map it to colour, plus the colorbar.
- **Max sector labels.** How many compass labels to draw around the rim.

To the right of the plot, a **per-sector results table** lists the aggregated value
(and count) for each direction; **Copy** puts it on the clipboard, and the same
breakdown is echoed to the **Log** tab.

### Tree ring

One variable drawn as concentric **annual rings** on a circular (polar) plot, one
ring per year, with the earliest year innermost. Two styles:

- **Filled** — a colour mesh, each ring coloured by the variable's value.
- **Line** — a radial trace per year that wiggles in and out with the value.

Settings: the **resample frequency** (how the data is aggregated around each ring),
month labels and month lines, and year labels and year separators. One variable at
a time.

### Waterfall

A variable's cumulative budget shown as a **waterfall**: each bar is one period's
net contribution, stacked onto the running total so the bars walk up and down to the
final sum. One variable at a time. Settings:

- **Resample period** and **aggregation** — the period each bar covers and how its
  values are combined.
- **Uptake is negative** — flip the orientation so that uptake (a negative flux)
  draws downward.
- **Bar colours and width**, **connector lines** between bars, and the **units**.

---

## Events

Mark **when something happened** at the site (fertilization, harvest, grazing, a
management step, a sensor swap) and overlay those markers on the plots. Each event
is stored as a `0/1` column (`EVENT_<name>`, where 1 means the event took place),
so it travels with the data and can be used like any other flag. When events are
visible, they are drawn on the time-series, cumulative, and daily panels (labels on
the time series) and along the y-axis of the date/time heatmap. Events are saved
with the project and between sessions, and are rebuilt on the current index when you
load new data.

### Add event…

Opens a dialog: a **name**, a **category** (the combo lists your existing
categories), one of three timing modes (a single date/time, a from/to period, or a
start plus a duration) with calendar pickers, and a **colour**.

### Show events on plots

The master visibility toggle (the same switch as the checkbox on the Events tab).

### Events

Opens the full manager: a reflowing board of event **cards** on a soft-grey board,
one per event, sorted by start date and accented by category colour. Each card shows
a category pill, the title, the date settings, a relative-time hint ("in 3 mo", "5 d
ago", "ongoing"), a mini bar showing where the event sits in the loaded record, and
a description preview. On each card:

- **double-click** to edit;
- the **trashcan** deletes it;
- the **⋯ menu** offers *Show on Overview* (zooms the Overview's linked panels onto
  the event), *Edit*, *Duplicate*, and *Shift 1 day later / earlier*.

Above the board: a **filter** field, a **Group** combo (None / Category / Year), a
**Density** combo (Comfortable / Compact), **Manage categories…**, and **Add
event…**. A dashed **＋ Add event** ghost card sits at the end of the board.

**Manage categories…** edits the category palette (add, rename, recolour, remove).
A category colour overrides the default on the cards and on the plot overlays.

---

## Analyze

### Data profile

A whole-dataset overview that answers "what did I just load?". A strip across the
top shows dataset-level facts: record count, number of variables, overall missing
%, duplicate timestamps and rows, inferred frequency, time span, and memory use.
Below it, a **sortable table** profiles every variable at once: type, valid count,
missing count and %, number of gaps, unique values, zeros, whether the column is
constant, and the numeric summaries (mean, SD, min, median, max). Gappy variables
are tinted red in proportion to their missing %, and constant columns are flagged
(a common "this variable is useless" signal). A **filter** box narrows the table by
substring. All profiling is the library's `dv.analysis.profile_dataframe` /
`dataframe_overview`.

### Gaps & coverage

A dashboard for finding and inspecting **missing data**. Pick a variable on the
left (it opens on the one with the most gaps); the right side shows:

- **Stat cards:** overall missing %, number of gap periods, long gaps, the longest
  gap and its duration, and the worst month.
- A **gap map:** a daily-availability heatmap (green is data, red is missing) over
  a timeline where each gap is a spike (taller means longer).
- A **table of the longest gaps** (start, end, length, duration).

The map is **clickable**: click a row in the table to highlight that gap on the
timeline, or click anywhere on the timeline to jump to the nearest gap (it gets
highlighted and its table row is selected). Use **Long gap ≥ (records)** to set
what counts as a "long" gap (48 records is one day for half-hourly data).

### Driver explorer

Find **what relates to a variable**, useful before gap-filling or when interpreting
a flux. Pick a **target** on the left; the table ranks every other variable by how
strongly it correlates with it (the **r** value is tinted green for positive, red
for negative, more saturated for stronger). **Click a driver** to see the
target-vs-driver scatter.

- **Method.** *Pearson* (straight-line relationships) or *Spearman* (any
  consistently increasing or decreasing relationship).
- **Max lag (records).** Scan a lead/lag window and report each driver's
  *strongest* lag (0 means same time only). A positive **Lag** means the driver
  leads the target (radiation now → flux a bit later). The scatter is shown at that
  lag.
- Changing the method or max lag takes effect on **Rank drivers**; picking a new
  target updates immediately.

### Compound extremes

Find **dry-hot extremes** by combining two drivers. Each month or day is turned into
a **z-score** (how far from normal it was) for both variables, and classified into
four types, shown as a quadrant scatter (after Wang et al.):

- **None** — neither variable was extreme (black dots).
- **Air** — high VPD only, atmospheric dryness (orange triangles).
- **Soil** — low soil water content only, soil dryness (dark-red squares).
- **Compound** — both at once (red diamonds), the most stressful case.

Pick **Variable 1** (e.g. VPD) and **Variable 2** (e.g. soil water content) on the
left; sensible columns are auto-selected with a ✓/✗ availability marker. The status
line and the scatter update on **Run**.

- **Variable 1 / 2 extreme.** Which tail counts as extreme — *high* (e.g. high VPD)
  or *low* (e.g. low soil water). Defaults: variable 1 high, variable 2 low.
- **Threshold (sigma).** How far from normal a z-score must be to count as extreme
  (default 2). The dashed lines on the plot mark this threshold.
- **Resolution.** Classify **monthly** or **daily** periods.
- **Standardize by.** *Deseasonalized* (recommended) compares each month/day against
  the same time of year across all years, so the normal seasonal cycle does not count
  as "extreme". *Whole-record* uses one average over everything (simpler, but summer
  months tend to flag).
- **Category labels.** Rename the single-variable categories (default *Air* / *Soil*)
  to match your drivers; they appear in the legend and point labels.
- **Copy Python** copies a runnable script that reproduces the classification and plot.

### Seasonal trend & anomalies

See whether a variable is **changing over the years** and which years stood out.
Pick a variable; it is split into:

- **Trend:** the slow, long-term direction.
- **Seasonal:** the repeating yearly cycle.
- **Residual:** what's left (noise, events).

Switch **View** to **Yearly anomalies** to see each year compared to a **reference
period** (red bars are above the reference mean, blue are below), handy for
spotting warming or unusually wet or dry years. Set the reference years with the
two boxes.

- **Method.** *STL* (robust, recommended), *Classical*, or *Harmonic*.
- **Robust** (STL) down-weights outliers; more faithful but slower.
- Method and Robust changes apply on **Update**; variable, view, and reference
  years update immediately.
- The decomposition needs at least about 2 years of data (the anomaly view works
  with fewer).

### Spectrogram

See **when** a variable's cycles are strong. A spectrogram shows time along the
bottom, frequency (cycles per day) up the side, and colour for power. A bright
horizontal band at **1 cycle/day** is the daily rhythm, and it usually strengthens
in the growing season. An explanation is shown above the plot.

- **Window (records)**, **Overlap %**, and **Window** set how the series is split
  for the analysis. A wider window gives finer frequency detail but blurs timing.
  These apply on **Update**.
- **Max cycles/day** sets how far up the frequency axis to look, and **Colormap**
  changes the colours. Both update immediately.

---

## Outliers

### Hampel filter

Detect spikes with the **Hampel filter**, a robust median-based test. Pick a
variable; the top panel shows it with detected outliers marked, the bottom panel
shows the cleaned copy (outliers removed). Your original variable is never changed.

- **Window (records).** How many neighbouring points define "local" (624 is about
  13 days at half-hourly sampling).
- **n sigma (global).** How far from the local median counts as an outlier (lower
  is stricter, flags more).
- **Use double-differencing.** Removes trends first (Papale 2006); leave it on for
  most flux and meteo data.
- **Repeat until no more outliers.** Re-runs until clean; a **progress bar** fills
  as each pass removes fewer points.

The **Preview** section holds two display options that don't change the result:

- **Live preview** *(on by default).* Both panels update after every iteration: the
  top panel grows its outlier markers pass by pass (red/blue when day/night is on),
  and the bottom panel shows the cleaned series with the points removed *this pass*
  flagged by a red **X**. Detection runs faster with this **off**, where it renders
  once at the end.
- **Show limit lines** *(off by default).* Overlay the upper and lower limits that
  decide what's an outlier, as faint lines (upper dashed, lower dotted; red is
  daytime, blue is nighttime, slate when not separating), with matching legend
  entries. Every iteration's band accumulates in the top panel (a tightening
  envelope); the bottom panel shows just the current pass's band. Shown in data
  units even with double-differencing on. Toggling it re-renders the last result.
- **Separate daytime / nighttime** *(optional).* Use **different** thresholds for
  day and night. That is the point of separating: with the same value for both, the
  result is identical to not separating. Set **Daytime n sigma** and **Nighttime n
  sigma** (they start from the global value, then edit each); the coordinates
  default from **Settings ▸ Project settings**. Day and night outliers are then
  drawn in red and blue, and the status line reports how many of each were found.

**Detect outliers** runs the filter; the status line reports the total and the
number of iterations. **Add cleaned + flag to dataset** adds two new columns
(`{var}_HAMPEL` and its flag) to the variable list. **Copy Python** (the quiet link
at the bottom) puts the equivalent diive script on the clipboard.

### Local SD filter

Flags points that deviate from a **rolling-window median** by more than a number of
standard deviations. It shares the Hampel tab's two-panel preview, live preview,
limit lines, day/night colouring, **Add to dataset**, and **Copy Python**. Only the
parameters differ:

- **Window (records).** The rolling window for the median and SD (seeded to about
  5% of the series length when data loads).
- **n SD (global).** How many standard deviations from the median count as an
  outlier (lower is stricter).
- **Constant SD (whole series).** Use the SD of the entire series instead of a
  rolling SD within the window.
- **Separate daytime / nighttime.** Set a window and n SD per period (these become
  `[daytime, nighttime]` lists for the library).

Adds `{var}_LOCALSD` and its flag to the variable list.

### Z-score filter

Flags points whose absolute **z-score** (deviation from the mean in units of
standard deviation) exceeds a threshold. Shares the Hampel tab's two-panel preview,
live preview, limit lines, day/night colouring, **Add to dataset**, and **Copy
Python**. Only the parameters differ:

- **Threshold (global).** Flag a point when its absolute z-score exceeds this value
  (lower is stricter).
- **Separate daytime / nighttime.** The z-score is then computed separately for
  daytime and nighttime records (each with its own mean and SD), with a **threshold
  per period** (seeded from the global value).

Adds `{var}_ZSCORE` and its flag to the variable list.

### Z-score (rolling) filter

Like the z-score filter, but the mean and standard deviation are computed in a
**rolling window** centred on each point, so the band adapts to local variability.
Useful for non-stationary series. There is no day/night mode (the rolling window
already adapts). Parameters:

- **Threshold.** Flag a point when its absolute rolling z-score exceeds this value
  (lower is stricter).
- **Window (records).** The rolling window for the mean and SD (seeded to about 5%
  of the series length when data loads).

With **Show limit lines** on, the band's centre (the **rolling mean** the band is
built around) is drawn as a solid line between the upper and lower limits.

Adds `{var}_ZSCOREROLLING` and its flag to the variable list.

### Z-score (increments) filter

Targets **abrupt changes**: a point is flagged only when the z-scores of its
forward, backward, *and* combined increments all exceed the threshold. This
isolates spikes while tolerating gradual change. There is no day/night mode, and
**no detection-limit band** (the **Show limit lines** option has no effect here):
the test is on increment z-scores rather than the values, and because a point is
flagged only when all three increments are extreme, the accepted region is a union
of intervals rather than a single upper/lower envelope, so there is no data-unit
band to draw. Parameter:

- **Threshold.** The z-score threshold applied to each increment series; a point is
  flagged only when all three exceed it (lower is stricter).

Adds `{var}_ZSCOREINCREMENTS` and its flag to the variable list (the flag column
keeps the library's `FLAG_{var}_OUTLIER_INCRZ_TEST` name).

### Trim-low filter

Removes low outliers with a **symmetric trim**: it rejects the values below a
**lower limit**, then rejects an equal number of the **highest** values, keeping
the distribution balanced (the trimmed-mean rationale). Because some rejected points
sit *above* the limit, use a one-sided method (Absolute limits) instead if you only
want to drop the low extremes. There is **no detection-limit band** (the **Show
limit lines** option has no effect): the high tail is removed by position, so the
kept set is not a single upper/lower envelope. Parameters:

- **Lower limit.** Values below this are rejected (plus the matching count of
  highest values). Seeded from the selected variable's minimum, so nothing is
  trimmed until you raise it.
- **Trim daytime only / Trim nighttime only** *(optional).* Leave both off (the
  default) to trim the **whole series** against one distribution. Tick one or both
  to restrict the trim to those periods, each screened against its own distribution.
- **Latitude / Longitude / UTC offset.** Used (and enabled) only when a day/night
  box is ticked; seeded from **Settings ▸ Project settings**.

Adds `{var}_TRIMLOW` and its flag (`FLAG_{var}_OUTLIER_TRIMLOW_TEST`) to the
variable list.

### Absolute limits filter

The simplest, one-sided test: flag everything **outside a fixed range**. Set a
**Min** and **Max**; any value below the min or above the max is rejected, and the
limits are drawn on the preview as the detection band. Use it for hard physical
constraints (relative humidity must be 0 to 100 %, for example). Optionally split
by day and night to enforce different ranges per period. Same preview, day/night
colouring, **Add to dataset**, and **Copy Python** as the other tabs.

Adds `{var}_ABSLIM` and its flag to the variable list.

### Local Outlier Factor filter

A density-based test (`LocalOutlierFactor`): a point is an outlier when its local
density is substantially lower than that of its nearest neighbours. Parameters:

- **Neighbors** (`n_neighbors`). How many nearest neighbours define the local
  density (default 20).
- **Contamination.** The expected fraction of outliers (default 0.01), or tick
  **Auto** to use the threshold from the LOF paper.

Adds `{var}_LOF` and its flag to the variable list.

### Manual removal

Remove **known** bad records by hand, with no statistics. List the timestamps
and/or periods to drop (a calibration window, a sensor failure you already know
about), then click **Flag listed dates**. There is no detection band and no
day/night mode; the selection is just the records you named.

Adds `{var}_MANUAL` and its flag to the variable list.

### Stepwise screening

Chain several outlier tests on one variable and see what each step removes. Unlike
the single-method tabs (one detector, one pass), each step runs on the data the
previous steps already cleaned, so spikes are peeled off progressively. Check the
overall **quality flag (QCF)**, computed separately from the accumulated per-test
flags.

The middle panel is a **segmented inspector** with three pages; only the active one
is shown so the plots stay large:

- **Outliers.** The chain as a vertical list of **method cards** (each shows all its
  settings and how many points it removed; reorder with ▲▼, edit, or delete).
  **＋ Add step** opens the method picker. The **Run outliers** button below the
  list applies the chain. Editing a step does *not* update the preview until you
  run (a `•` on the button marks unapplied edits).
- **Corrections.** High-resolution corrections applied to the QCF-filtered series
  (nighttime zero-offset, relative-humidity offset, set-to-min/max, set-to-value,
  set-exact-to-missing). The **Measurement** dropdown (auto-detected from the
  variable name, e.g. *SW - shortwave radiation*) decides which corrections are
  physically meaningful and shows only those. The **Run corrections** button applies
  them. (The same corrections are also available one-per-tab under the **Corrections**
  menu, described below.)
- **Report.** The per-step screening statistics (retained / rejected, day/night),
  with **Copy report**.

**Add cleaned + flags + QCF to dataset** appends every step's flag, the overall QCF
flag, the QCF-filtered series, and (if any corrections ran) the corrected series.
**Copy Python** copies a reproducible script for the whole chain.

---

## Corrections

The **Corrections** menu has one tab per correction; they all share the same layout.
**Click a variable** in the **Target** list on the left, set the options in the
middle, and the right side previews the **original** against the **corrected**
series. **Run correction** applies it, **Add corrected to dataset** keeps the result
(a new `{var}_…` column; your original is never changed), and **Copy Python** (top
bar) copies a reproducible script. Each correction is its own tab, so any correction
is available for any variable; the suggested use is just a hint.

### Remove nighttime zero offset

For variables that should read **zero at night** (shortwave radiation, PPFD). For
each day it works out the average of that day's nighttime values (the **offset**),
subtracts it from all of the day's records, then forces the nighttime to zero. It
needs the **site coordinates** (from **Settings ▸ Project settings**) to tell day
from night; the latitude / longitude / UTC offset are shown and pre-filled.

- **Clamp negative values to zero** *(on by default).* After the offset is removed
  and the night zeroed, set any remaining negative values (daytime included) to zero
  as well, enforcing the variable's physical floor of zero. **Uncheck** it to keep
  those negatives, for example to inspect them.

The preview shows the **four stages** stacked: the original series, the daily
offset, the series after subtracting the offset, and the final corrected series
(each with a zero line so dips below zero stand out). The stats band at the top
reports how many records were **below zero before** the correction (overall and at
night) and **after** it. The after counts show a green **0 ✓** once nothing remains
below zero. Adds `{var}_NIGHTOFFSET`.

### Remove relative humidity offset

For **relative humidity** that drifts above 100%. The daily mean of the values
exceeding 100% is removed as an offset and anything still over 100% is capped at 100.
Same layout as above (no site coordinates needed). Adds `{var}_RHOFFSET`.

### Set to max threshold / Set to min threshold

Cap or floor a variable at a known physical limit: every value **above** (max) or
**below** (min) the **Threshold** you set is replaced with the threshold. This works
on any variable. Adds `{var}_SETMAX` or `{var}_SETMIN`.

### Set to value

Overwrite every record inside one or more **date ranges** with a fixed **value**,
for example to blank out a period of known instrument trouble. Enter ranges separated
by `;`, using `..` for a range (`2022-04-01..2022-04-05`) or a single timestamp on
its own. Adds `{var}_SETVAL`.

### Set exact values to missing

Set records that **exactly equal** any of the values you list (comma-separated, e.g.
`0, -9999`) to missing (NaN). Useful for dropping a stuck sentinel value. Adds
`{var}_SETMISSING`.

---

## Data

### Select variables

Pick a subset of variables to focus the **Overview** list on. Click a variable on
the left (*Available*) to move it to the right (*Selected*); click one on the right
to remove it. **Add all →** (under the Available list) moves everything across, and
**Clear** (under the Selected list) empties the selection. **Confirm → update
Overview** restricts the Overview's variable list to your selection. Your data is
not changed (load new data or re-open to reset).

### Select records by condition

Build a filtered copy of one variable (the **target**) using the value of another
variable (the **condition**). Click a target on the left, then build an
**operation** in the settings: pick a condition variable, a **lower** / **upper**
range (untick a side for an open bound), the boundary inclusivity, and an
**action** — *Keep selected records* (drop everything except the in-range ones) or
*Remove selected records* (drop the in-range ones, keep the rest). **Select
records** applies the operation to a working selection.

Operations **stack**: change the condition, range, or action and apply again to
narrow further — for example *keep where Tair in [15, 20]* then *remove where VPD
in [10, 100]*. The preview updates after each step (the condition's band and the
last operation's selected points on top, the surviving records on the target
below). **Undo last** / **Reset** walk the chain back, and the condition may be the
target itself (filter a variable by its own value). **Add selection to dataset**
appends the result as a new `{target}_SEL` column (out-of-range records set to
missing; the time index is preserved). **Copy Python** yields a runnable script.

### Rename variables

Add a common **prefix and/or suffix** to **all** variables at once, for example to
tag every column with a site code (`CH-DAV_…`) or a year (`…_2024`). Type a prefix
and/or suffix; the table **previews** the old → new names (changed ones in bold)
before anything happens. **Apply rename** commits it to the loaded dataset.

To rename just **one** variable, **double-click its name** in the table (the same
as the right-click **Rename…** on any variable list). Renaming is non-destructive to
the source file, and each variable keeps its tags, notes, and history under the new
name.

### Metadata explorer

See and edit the metadata that travels with each variable, useful once a variable
has been through several steps (load → outlier filter → gap-fill → …) and you want
to know *where the current version came from*. Pick a variable on the left; the
right panel shows:

- **Origin.** *original* (straight from the file), *modified* (a transformed copy,
  e.g. outliers removed), or *derived* (computed from a parent, e.g. a flag), plus
  the **parent** variable it came from.
- **Tags.** Toggle **★ Favorite**, and add or remove your own tags (each user tag
  gets its own colour automatically). Operations also add tags themselves (e.g.
  `hampel`, `flag`). **Clear this variable's tags & note** removes just this
  variable's custom tags and note.
- **Note.** Free text describing the variable, up to **50 words** (a live counter
  shows the count; **Save note** greys out once saved and re-enables when you edit
  again).
- **History.** The ordered list of operations that produced the variable, each with
  its settings and time; for a loaded variable the first entry is its import.

**Right-click a variable** in this tab's list for **Remove all tags & note** to
clear just that one. **Clear all tags & notes** (bottom of the tab) does the same
for every variable in the current dataset at once, after a confirmation. Either way,
auto-assigned tags, origin, and history are kept.

The bundled **example data always opens clean** (no tags or notes); tags and notes
are kept only for data you load yourself.

**Your tags and notes are saved between sessions, per dataset**, so the same column
name in two different files keeps separate tags. Origin and history are recomputed
each session as you work.

### Feature engineering

Build new features (lags, rolling stats, differences, EMA, polynomials, STL,
timestamp parts, and more) with diive's feature engineer:

1. **Click** variables to move them into *Selected features*.
2. Tick the stages you want and set their options. *Timestamp features* and
   *Continuous record number* work on the time index alone, so they need **no**
   selected variable.
3. **Run feature engineering.** The new columns are listed under *Newly created
   features*.
4. **Add features to variable list.** The new columns appear everywhere with a
   **✦ NEW** pill and can be plotted like any other variable.

When you add new columns (here or from an Outliers tab), the **Overview jumps
straight to the new variable**: it clears any active filter, scrolls the new row
into view, and plots it, so you can see the result right away.

### Combine variables

Build a new variable by combining two existing ones, with all three shown as
date/time heatmaps side by side.

1. **Drag** a variable from the list onto **Heatmap 1**, and another onto
   **Heatmap 2** (only these two are drop targets). Each plots as you drop it.
2. Pick how to **Combine** them:
   - **Multiply / Add / Subtract / Divide** — element-wise arithmetic of
     heatmap 1 (a) and heatmap 2 (b).
   - **Fill gaps of a with b** — keep heatmap 1 and fill only its gaps with the
     matching values from heatmap 2.
3. **Keep overlapping data points only** (arithmetic methods): when ticked, a
   result is kept only where *both* variables have a value; when unticked, a
   missing value is treated as the operation's identity (0 for add/subtract, 1 for
   multiply/divide) so one-sided records survive. (It is disabled for *Fill gaps*,
   which is always a union.)
4. **Heatmap 3** previews the combined result and updates live as you change the
   method or the overlap option.
5. Edit the **Name** (a default is suggested) and click **Add … to dataset** to
   append the new column. **Copy Python** yields a runnable script.

---

## Flux

### Flux processing chain

A guided workspace for the flux processing chain, covering **Input + Level 2 +
Level 3.1 + Level 3.2 + Level 3.3 + Level 4.1** (gap-filling). Pick the flux column
and site, choose which Level-2 quality tests to run, set the Level-3.1 storage
correction, optionally build a Level-3.2 outlier-detection chain, apply Level-3.3
USTAR filtering, run gap-filling, then run the chain. The accepted (QCF-filtered)
flux of the deepest level shows as a heatmap. **Copy Python** puts the exact,
reproducible diive script for what you did on the clipboard, so a point-and-click
run stays scriptable.

Each level has its own **Add to dataset** button (next to that level's run button),
enabled once the level has run. It appends that level's output columns (its flags,
QCF flag, gap-filled series, and the QCF-filtered flux — level-qualified names like
`FC_L3.1_QCF` so they don't collide) to the main variable list, so you can pull
intermediate results out of the chain without waiting for the whole pipeline.

**Level 3.3: USTAR filtering** has two modes (a *Mode* dropdown):

- **Constant thresholds.** Type one or more known u\* thresholds (m s⁻¹) with
  optional labels; each becomes a filtering scenario.
- **Detect (moving point, Papale 2006).** Estimate the threshold from the data with
  a multi-year bootstrap. Pick the TA and SW_IN columns, set the bootstrap
  iterations and percentiles, and choose what to **Apply**:
  - **CUT** (constant): one threshold per percentile applied to the whole record
    (`CUT_16/50/84`).
  - **VUT** (per-year): each year filtered by its own threshold (`VUT_16/50/84`).
    diive's VUT is smoothed over a 3-year window for stability.

  CUT and VUT are alternative strategies, so pick one; the percentiles (16/50/84)
  are the uncertainty scenarios within it. Each scenario is gap-filled separately at
  Level 4.1.

> Needs eddy-covariance input with the raw EddyPro columns (FC, USTAR, the `*_TEST`
> flags). The bundled CH-DAV example is a processed product and won't run the chain,
> so load a level-1 EC dataset.

### USTAR detection

Detect the friction-velocity (u\*) threshold on its own, without running the whole
chain, useful for exploring thresholds or getting a number for elsewhere. Pick the
**NEE, air-temperature, USTAR and SW_IN** columns (auto-filled from the data),
optionally adjust the stratification (temperature and USTAR classes, forward-mode
order), and:

- leave **Multi-year bootstrap** unticked for a quick **single detection**, giving a
  threshold per season plus the annual value (the maximum across seasons); or
- tick it to run the **multi-year bootstrap**, which reports **VUT** (a threshold
  per year) and **CUT** (one constant threshold pooled across all years), each at the
  chosen percentiles, as a table and a chart.

The threshold detection follows the ONEFlux moving-point method (Papale et al.
2006). Results appear as a table on the left and a plot on the right.

### Time lag analysis

Find the **optimal time lag for each gas** in raw eddy-covariance data. The gas
analyser sits downstream of the sonic anemometer, so each gas signal arrives a
fraction of a second after the wind. This tab builds a histogram of the measured
lags for a gas (from the `*_TLAG_ACTUAL` columns), detects the **peak** lag and the
**range** around it with gradient-based edge detection, and formats an
**EddyPro-ready search window** to use as the covariance-maximisation window for
that gas. An explanation is shown above the plot, and a four-panel figure is drawn
on the right with a results readout below it.

- Pick the **gas channel** from the detected `*_TLAG_ACTUAL` columns.
- **Parameters:** fringe bins to ignore (edge bins accumulate non-physical lags),
  the reference acceptable lag **window min/max** (s), histogram start and end bins,
  the **gradient threshold** (edge-detection sensitivity; lower is stricter and
  gives a narrower range), and a zoom margin around the peak.
- **Analyze & plot** runs it. The figure marks the peak (black), the detected range
  (teal), the EddyPro window expanded by one 0.05 s step (orange), and your
  reference window (purple).

> Needs Level-0 data with `*_TLAG_ACTUAL` columns. If the active dataset has none,
> click **Load example TLAG data** to load a bundled dataset locally so you can try
> the feature.

### NEE partitioning

Split measured net ecosystem exchange (NEE) into its two component fluxes, **gross
primary production (GPP)** and **ecosystem respiration (RECO)**, using one of four
faithful method ports, each on its own tab:

- **Nighttime partitioning (ONEFlux).** Reichstein et al. (2005), per calendar year.
  Output columns end in `_NT_OF`.
- **Nighttime partitioning (REddyProc).** The REddyProc `sMRFluxPartition` variant
  (single temperature sensitivity for the whole record). Columns end in `_NT_RP`.
- **Daytime partitioning (REddyProc).** The light-response-curve method (Lasslop et
  al. 2010, REddyProc `partitionNEEGL`). Columns end in `_DT_RP`.
- **Daytime partitioning (ONEFlux).** The FLUXNET2015 light-response method (Lasslop
  et al. 2010, ONEFlux). Columns end in `_DT_OF`.

The two families get at the split differently. The **nighttime** methods read
respiration straight from nighttime NEE, when photosynthesis is switched off, fit its
temperature response, extrapolate that response into the daytime, and recover GPP as
the leftover. The **daytime** methods fit a light-response curve to daytime NEE and
separate GPP and RECO together, so they also draw on VPD and radiation. ONEFlux and
REddyProc are two established reference implementations of these algorithms. Running
more than one and comparing is a common sanity check, which is why each writes its own
column suffix and all four can sit in the dataset side by side.

A note on signs: NEE is negative when the ecosystem takes up carbon. GPP and RECO come
back as positive fluxes, related by NEE = RECO - GPP.

On each tab, pick the **input columns** (auto-filled from the data, with a green ✓ /
red ✗ marker showing whether the chosen column exists):

- The **nighttime** methods need *measured* and *gap-filled* NEE and air
  temperature, plus shortwave-in radiation for the day/night split.
- The **daytime** methods fit on *measured* NEE with *gap-filled* meteo drivers (air
  temperature, VPD, shortwave-in). The ONEFlux daytime variant also takes the
  measured air temperature and radiation; the REddyProc daytime variant has an
  optional **NEE SD** column (leave it as *(none)* to use the method's default
  uncertainty).

Set the **site coordinates** (latitude, and for the methods that need them,
longitude and UTC offset; all default from **Settings ▸ Project settings**). For the
daytime methods, the **VPD is in kPa** toggle says whether your VPD column is in kPa
(the diive convention, default) or hPa.

Click **Run partitioning** (it runs in the background; the daytime methods can take
a while, roughly tens of seconds per year). The preview shows daily-mean NEE, GPP,
and RECO over time plus their cumulative sums. **Add results to dataset** appends all
of the method's output columns (GPP, RECO, and the fitted parameters) to your
variables.

> The bundled CH-DAV example already has measured and gap-filled NEE, air
> temperature, VPD, and radiation columns, so you can try these tabs on it right
> away.

### Random uncertainty (PAS20)

Estimate the **random measurement uncertainty** of a flux — the ±1σ scatter you'd
expect from repeating the same half-hour under the same conditions. This is the
hierarchical 4-method approach of Pastorello et al. (2020), a faithful port of the
ONEFlux `randunc` reference. It is separate from the gap-filling (model) uncertainty.

Pick the **input columns** (auto-filled, with a green ✓ / red ✗ availability marker):
the *measured* flux, the *gap-filled* flux (used for the cumulative propagation), and
the three meteorological similarity drivers — **air temperature (TA)**, **VPD** and
**shortwave-in radiation (SW_IN)**. The **VPD is in kPa** toggle says whether your VPD
column is in kPa (the diive convention, default) or hPa.

Every record is assigned an uncertainty by the first of four methods that succeeds
(each is more permissive than the last, so no record is left without an estimate):

1. **Method 1 — direct standard deviation (ONEFlux).** For a record with a *measured*
   flux, look at all other measured fluxes in a sliding **±7-day, ±1-hour** window
   (same time of day) that occurred under *similar meteorological conditions* — the
   same similarity test as MDS gap-filling: TA within ±2.5 °C, VPD within ±5 hPa, and
   SW_IN within a radiation-dependent band (±20–50 W m⁻²). If **more than 5** such
   fluxes exist, the uncertainty is their standard deviation. This is the only method
   that measures uncertainty directly; methods 2–4 reuse these values.
2. **Method 2 — median of similar fluxes (ONEFlux).** For records method 1 couldn't
   do (gap-filled half-hours, or measured ones with too few similar fluxes), take the
   **median of the method-1 uncertainties** of fluxes of *similar magnitude* (within
   ±20 %, but at least ±2 µmol CO₂ m⁻² s⁻¹) in a **±14-day** window.
3. **Method 3 — median over the whole record (diive extension).** Like method 2 but
   with **no time window** — the median of method-1 uncertainties of all
   similar-magnitude fluxes across the entire record. Fills the few records method 2
   still couldn't.
4. **Method 4 — nearest fluxes (diive extension).** A last resort with no similarity
   restriction: sort all records by flux magnitude and take the **median uncertainty
   of the ~10 closest** in magnitude. Methods 3 and 4 are diive additions (ONEFlux
   leaves these records undefined) so that every record gets an estimate.

Click **Run uncertainty** (it runs in the background, with a progress bar over the
four methods). The hero band reports the **mean / median** per-record uncertainty,
the number of records covered, and the final **cumulative ±σ**. The preview shows
three panels:

- **top** — the flux with its ±σ band (daily means);
- **bottom left** — the cumulative flux with its propagated uncertainty bounds.
  Random errors are assumed independent, so the cumulative uncertainty is their
  quadrature sum (√Σσ²) — it grows much more slowly than the flux itself;
- **bottom right** — uncertainty vs. flux magnitude, the classic Hollinger &
  Richardson (2005) scaling (uncertainty rises with |flux|). This panel shows
  **method-1 records only** (the directly-measured uncertainties); the method 2–4
  fallbacks are medians of repeated values and would otherwise paint as misleading
  horizontal streaks.

**Add result to dataset** appends a single `{flux}_RANDUNC` column (the per-record
±σ). The status line and the **Method 1/2/3/4 records** counts tell you how much of
the record each method covered — a high method-1 share means most uncertainties are
directly measured rather than inferred.

> Tolerances and window sizes follow the ONEFlux reference and are not adjustable.
> The cumulative panel and the `{flux}_RANDUNC` column are also available from the
> library via `dv.flux.RandomUncertaintyPAS20`.

### Joint uncertainty (PAS20)

Combine the **random measurement uncertainty** (above) with the **filtering
uncertainty** — the spread you get from the different USTAR-threshold scenarios — into
one *joint* uncertainty per record. This is the ONEFlux `compute_join` calculation
(Pastorello et al. 2020): the two error sources are added in quadrature,

```
JOINTUNC = √( RANDUNC² + ((scenario_upper − scenario_lower) / divisor)² )
```

so the joint uncertainty is always at least as large as the random part alone. Run the
**Random uncertainty (PAS20)** tab first to produce the `{flux}_RANDUNC` column this
tab needs.

Set the **Scenario percentiles** — this picks both the divisor and which scenario
columns are auto-selected:

- **NEE — USTAR scenarios (16th / 84th ÷ 2).** For NEE, the lower/upper scenarios are
  the 16th and 84th USTAR-threshold percentile fluxes (e.g. `NEE_CUT_16` / `NEE_CUT_84`).
  These bracket ±1σ, so their range is divided by **2**.
- **Energy flux LE/H (25th / 75th ÷ IQR 1.349).** For the energy fluxes, the scenarios
  are the 25th/75th energy-balance-correction percentiles — the interquartile range,
  which is **1.349** standard deviations wide.

Then pick the **input columns** (auto-filled, with ✓ / ✗ availability markers): the
random-uncertainty column, the lower and upper scenario fluxes (biased to match the
random-uncertainty column's flux), and the gap-filled flux (the central line the band
brackets). Click **Run joint uncertainty**. The hero band reports the **mean joint**
uncertainty alongside its **random** and **scenario** components and the final
**cumulative ±σ**. The preview shows three panels:

- **top** — the flux with its ±σ joint band (daily means);
- **bottom left** — the **component decomposition**: the random, scenario and joint
  uncertainties side by side, so you can see which source dominates;
- **bottom right** — the cumulative flux with its propagated joint bounds. The random
  part is independent (quadrature, √Σσ²) while the scenario (threshold) choice is
  *fully correlated* across the record — the same threshold applies to every half-hour
  — so its cumulative term is the running spread of the cumulative scenario sums; the
  two are then combined in quadrature.

**Add result to dataset** appends a single `{flux}_JOINTUNC` column.

> The same calculation is available from the library via
> `dv.flux.JointUncertaintyPAS20`.

---

## Gap-filling

### XGBoost gap-filling

Fill the gaps in one variable with an **XGBoost** model (gradient-boosted trees),
trained on other variables you pick as predictors. This tab does gap-filling only:
it has **no feature-engineering options**. If you want engineered predictors (lags,
rolling means, and so on), build them first in **Data ▸ Feature engineering**, then
select them here.

The tab has two sub-tabs: **Model** (set up and run the gap-filling, see the
performance band, heatmaps and SHAP table) and **Results** (a fuller dashboard of
tables and plots, filled in after a run). The **Copy Python** button sits in the
title bar; **Run gap-filling** and **Add results to dataset** sit next to the
sub-tab pills.

Set up the run on the **Model** sub-tab, with **three variable lists** on the left:

1. **Target** (far left). **Click** a variable to make it the **target** (the one
   whose gaps get filled). The chosen target is highlighted and shown as *Target:* at
   the bottom.
2. **Available features** (middle). **Click** a variable to use it as a model
   predictor (it moves to the right). The target is automatically excluded here.
3. **Selected features** (right). Your chosen predictors; **click** one to drop it
   back.

The **settings** column holds the model controls (hover any field for a tooltip):

- **n_estimators / max_depth / learning_rate.** Model size and learning speed.
- **early_stopping.** Stop adding trees once the validation score stalls (0 is off).
- **test_size.** Fraction of complete records held out to score the model honestly.
- **random_state.** The reproducibility seed; spin it down to **none** to let
  XGBoost reseed every run (results then drift), or leave a fixed number (default 42)
  for repeatable results.
- **Negatives.** Keep predicted negative values (fluxes like NEE can be negative) or
  clip / NaN them for variables that can't be negative (VPD, SW_IN).
- **Reduce features (SHAP importance).** Optionally drop weak predictors before the
  final model; the **threshold factor** sets how strict that is.

Click **Run gap-filling** (it runs in the background). When it finishes:

- A **performance band** across the top shows the model's **held-out test** scores
  (R², RMSE, MAE, MAPE, MAXE) plus how many gaps were filled, how many of those used
  the **fallback** model, and how many features were used. The green **HELD-OUT
  TEST** chip is a reminder that these scores are measured on data the model didn't
  train on. **Fallback** fills happen when a predictor is missing at a gap, so the
  model falls back to time-of-year/-day only; a high fallback count means those fills
  are weak.
- Two **heatmaps** compare the **observed** series (with its gaps) against the
  **gap-filled** result, on a shared colour scale.
- On the right, a **SHAP feature importance** table ranks how much each predictor
  drove the model (over all complete observations). Each row shows the variable, its
  mean |SHAP| value, and a small bar for its relative magnitude.

Open the **Results** sub-tab for a fuller report of the same run:

- A top row of tables: **Model performance** (held-out-test vs in-sample scores
  across R²/RMSE/MAE/MedAE/MAPE/MAXE/MSE), **Configuration** (the exact settings, for
  reproducibility), **Feature reduction** (only if you enabled it; each feature's
  importance vs. the random-baseline threshold, with kept features in green, dropped
  in grey, and an ⓘ button explaining the threshold equation), and **Gap-fill
  quality** (the observed / model-filled / fallback-filled breakdown + coverage).
- A row of plots: **predicted vs. observed** (with a 1:1 line), the **SHAP
  importance** bar chart, the gap-filled **diel cycle** by month, and the **cumulative
  sum**. Hover any table value for an explanation.

**Copy Python** (top bar) puts a runnable diive script for the current
target/features/settings on the clipboard, so a point-and-click run stays
scriptable. **Add results to dataset** appends the gap-filled series (`{target}_gfXG`)
and its fill flag to your variables, where they can be plotted or saved like any
other column.

> The bundled CH-DAV example has continuous flux and meteo columns, so you can try
> this right away, for example target `NEE_CUT_REF_orig` with `Tair_f`, `VPD_f`,
> `Rg_f` as features.

**How the held-out test score is computed (and why the split is random).** The model
is scored on a **held-out test set**. By default, a random **25%** of the *complete*
records (rows where the target is observed and every feature is present) is set aside
and not used for training; the reported scores measure how well the model predicts
those withheld values. The split is **random**, so the test records are distributed
across the whole record rather than taken from one contiguous period.

This is an intentional choice and is appropriate for gap-filling. Each gap is filled
from the **predictor values at its own timestamp**, and gaps are interspersed with
observed data, so the training set spans the same period and conditions as the gaps.
A random hold-out approximates this situation (predicting individual missing records
from their drivers, with training data covering the same period), so the resulting
scores give a representative estimate of gap-filling performance.

A **block or temporal split** (withholding a contiguous month or year) addresses a
different question: how well the model transfers to a period it has not seen. That is
relevant for forecasting, or for assessing **drift** between years, but it is not the
gap-filling task, in which training always covers the same conditions as the
interspersed gaps. It would also conflate fill accuracy with changes in the
driver-target relationship between periods.

### Random Forest gap-filling

Fill the gaps in one variable with a **Random Forest** model, trained on the
predictors you pick. The tab works **exactly like the XGBoost gap-filling tab**
above (the same Model/Results sub-tabs, target/feature lists, performance band,
heatmaps, SHAP table, and Results dashboard); only the model controls differ:
n_estimators, max_depth (**none** = grow trees fully), min_samples_split,
min_samples_leaf, max_features (`all` / `sqrt` / `log2`), test size, random seed,
n_jobs and negative-value handling. **Add results to dataset** appends the
gap-filled series (`{target}_gfRF`) and its fill flag. Use this when you want a
less tuning-sensitive alternative to XGBoost.

### MDS gap-filling

Fill the gaps in one variable with **marginal distribution sampling (MDS)**, the
FLUXNET look-up-table method. Unlike XGBoost and Random Forest, MDS is **not a
trained regressor**: it has no SHAP importances, no held-out test split, and no
feature reduction. Instead it fills each gap from records with **similar
meteorological conditions** using three fixed drivers: shortwave-in radiation, air
temperature, and VPD.

Pick a **target** (flux) on the left and the **three driver columns** (SWIN / TA /
VPD combos, auto-seeded by name with ✓ / ✗ markers; gap-filled `_f` versions are
preferred). Set the **similarity tolerances** (SWIN low/high, TA in °C, VPD in kPa)
and the minimum number of similar records to average. Driver units matter: TA must be
in °C and VPD in kPa. diive does not check the units, so a column in the wrong unit
produces a wrong fill with no warning.

A **progress bar** tracks the quality levels as the fill runs (higher level = looser
meteorological match). The **Results** sub-tab shows the configuration, in-sample
scores, a **per-quality-level breakdown** table and bar plot, a predicted-vs-observed
scatter, the cumulative sum, and a colour-by-quality time series. The fill flag
(`FLAG_{var}_gfMDS_ISFILLED`) is 0 for observed and 1+ for the quality level at which
each gap was filled. **Add results to dataset** appends the gap-filled series and its
flag; **Copy Python** copies a reproducible script.

---

## Settings

### Project settings

Settings for the current project:

- **Your name.** The project author.
- **Description.** Free-text notes about the project: purpose, data source,
  processing decisions, and anything else worth recording.
- **Site details.** The measurement site's name, latitude, longitude, elevation, and
  UTC offset.

Fill in and **Save**. The site coordinates and UTC offset are reused wherever diive
needs them (the Hampel tab's daytime/nighttime split, the flux chain, and so on), so
you don't retype them per tool. Everything here is **remembered between sessions**
and **saved with the project**, so it travels inside a `.diive` folder.

**Notes wall.** The right side of the tab is a pinboard of **sticky notes** for
free-form reminders. **+ Add note** drops a card you can type a **bold title** and
body into; **drag** it by its top bar to arrange it, drag the bottom-right corner to
**resize** it, click **●** to **recolour** it (sticky-note palette or a custom
colour), and **✕** to remove it. The notes save automatically with the project (and
between sessions), so they travel inside the `.diive` folder.

### Appearance

Customise the **Studio** look (the GUI's single design: near-white surfaces,
pill-shaped tabs, a slim header with drop-down menus) with a **live preview**, so
colours update across the whole app as you change them:

- pill colours, selection and hover colours, time-series line colours;
- the variable-list **width** (applies to every tab);
- **Reset to defaults** to undo.

---

## Log

Mirrors diive's console output (file loading, feature engineering progress, and so
on) in colour. **Save…** writes the log to a text file; **Clear** empties it.

---

## Tips

- **Editable fields are tinted** (light blue) so you can see at a glance what you can
  change; **hover over a setting** for a tooltip describing what it does.
- **Hover over any plot** to see the value under the cursor in a small box. On line
  plots it snaps to the nearest data point (with a marker); on heatmaps it shows the
  cell's date, time, and value. Untick **Hover values** (bottom-right, next to the
  plot toolbar) to switch it off.
- Your appearance settings, site details, window size and position, last-used
  filetype, variable tags and notes, and most-recent project are **remembered**
  between sessions.
- The window opens **maximized** to make the most of your screen.
- A short loading cue appears on a variable while its plot is being drawn.
- Stuck, or something looks off? Check the **Log** tab for messages.

---

*Part of the diive library: https://github.com/holukas/diive*
