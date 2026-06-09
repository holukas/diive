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

A splash screen appears while the app starts up; the bundled example dataset
(CH-DAV, 37 variables) then loads automatically, so you can try everything right
away. (You can see the splash again any time via **Help ▸ About**.)

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

### Saving your data

**File ▸ Save data as parquet…** (Ctrl+S) writes the current dataset — including
any features you engineered — as a **diive-format parquet** file: a single header
row and a properly named timestamp index (`TIMESTAMP_MIDDLE` / `TIMESTAMP_END` /
`TIMESTAMP_START`; you're asked which if it isn't already set). These files load
straight back into diive (GUI or library).

### The variable list (left side, every tab)

- **Filter box** — type to narrow the list. Matching is fuzzy and ignores
  underscores/case: `gpp16` finds `GPP_CUT_16_f`.
- **Tag pills** — variables are colour-tagged by kind: NEE/FC (green), GPP (blue),
  Reco (red), LE/ET (purple), radiation (orange). Features you create get a pink
  **✦ NEW** pill.
- The list looks and behaves the same in every tab.

---

## Tabs

The window opens with **Overview** and **Log**. Other tabs open from the menus and
can be closed (×). Most can be opened **multiple times** — you'll get *Heatmap 1*,
*Heatmap 2*, etc.

### Overview (first tab)

Click a variable to see, for that variable:
- a **figure** with several panels — full time series, cumulative sum, mean diel
  cycle, daily-mean time series, and a date/time heatmap;
- a **strip of statistic cards** along the bottom (count, mean, SD, min/max,
  percentiles, …).

### Plot ▸ Heatmap date/time · Heatmap year/month · Time series · Diel cycle · Cumulative year · Ridgeline · Hexbin

Each plot method opens as its own tab (the menu shows a small icon for each).

- **Heatmap date/time** — date × time-of-day grid.
- **Heatmap year/month** — one cell per year × month (pick the aggregation —
  mean, sum, … — and optionally show *ranks*).
- **Diel cycle** — the mean daily cycle (value by time of day) with a ±SD band;
  optionally one curve per month.
- **Cumulative year** — one cumulative-sum curve per year (overlaid by day of
  year); optionally **highlight a year** (chosen from a dropdown of the years
  present in the data) and show a mean reference.
- **Scatter XY** — click two variables for X and Y (a third, optional, colours
  the points); optionally bin the x-axis and show a trend. One panel.
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

### Tools ▸ Gaps & coverage

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

### Tools ▸ Driver explorer

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

### Tools ▸ Seasonal-trend & anomalies

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

### Tools ▸ Spectrogram

See **when** a variable's cycles are strong. A spectrogram shows time along the
bottom, frequency (cycles per day) up the side, and colour for power — a bright
horizontal band at **1 cycle/day** is the daily rhythm, and it usually
strengthens in the growing season. An explanation is shown above the plot.

- **Window (records)** / **Overlap %** / **Window** — how the series is split for
  the analysis; a wider window gives finer frequency detail but blurs timing.
  These apply on **Update**.
- **Max cycles/day** sets how far up the frequency axis to look; **Colormap**
  changes the colours — both update immediately.

### Tools ▸ Feature engineering

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

### Tools ▸ Flux processing chain

A guided workspace for the flux processing chain *(early — currently the Input +
Level 2 steps)*. Pick the flux column and site, choose which Level-2 quality
tests to run, and **Run Level 2** — the accepted (QCF-filtered) flux shows as a
heatmap. **Copy Python** puts the exact, reproducible diive script for what you
did on the clipboard, so a point-and-click run stays scriptable.

> Needs eddy-covariance input with the raw EddyPro columns (FC, USTAR, the
> `*_TEST` flags). The bundled CH-DAV example is a processed product and won't
> run the chain — load a level-1 EC dataset.

### Settings ▸ Appearance

Customise the look with a **live preview** — colours update across the whole app as
you change them:
- pill colours, selection/hover colours, time-series line colours;
- the variable-list **width** (applies to every tab);
- **Reset to defaults** to undo.

### Log

Mirrors diive's console output (file loading, feature engineering progress, …) in
colour. Use **Save…** to write the log to a file.

---

## Tips

- **Editable fields are tinted** (light blue) so you can see at a glance what you
  can change; **hover over a setting** to see a tooltip describing what it does.
- **Hover over any plot** to see the value under the cursor in a small box — on
  line plots it snaps to the nearest data point (with a marker); on heatmaps it
  shows the cell's date, time, and value. Untick **Hover values** (bottom-right,
  next to the plot toolbar) to switch it off.
- Your appearance settings, window size/position, and last-used filetype are
  **remembered** between sessions.
- The window sizes itself to your screen on first launch.
- A short loading cue appears on a variable while its plot is being drawn.
- Stuck or something looks off? Check the **Log** tab for messages.

---

*Part of the diive library — https://github.com/holukas/diive*
