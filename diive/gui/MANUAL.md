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

On startup the bundled example dataset (CH-DAV, 37 variables) loads automatically,
so you can try everything right away.

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

### Plot ▸ Heatmap date/time · Heatmap year/month · Time series

Each plot method opens as its own tab.

- **Heatmap date/time** — date × time-of-day grid.
- **Heatmap year/month** — one cell per year × month (pick the aggregation —
  mean, sum, … — and optionally show *ranks*).
- **Click** a variable to plot it.
- **Ctrl + click** more variables to compare them in extra panels (up to 5):
  - *Heatmaps* line up **side by side** (shared axes).
  - *Time series* **stack** top-to-bottom (shared time axis), each its own colour.
- **Ctrl + click** a shown variable again to remove its panel.
- Use the small toolbar (bottom-right of the plot) to **pan, zoom, and save** the
  figure. Zooming one panel zooms them all.

**Settings (middle column) — live preview.** Between the variable list and the
plot is a panel of controls for the plot. Change one and the plot updates
immediately:
- *Heatmap*: colormap, min/max colour values, missing-value colour, orientation
  (vertical/horizontal), date-axis ticks, grid, colorbar (show, label, decimals,
  extend arrows), and optionally overlaying the numeric values on the cells.
- *Time series*: line width, opacity, point markers, whether to connect across
  gaps, and the axis labels/units.

  Line *colours* for time series come from **Settings ▸ Appearance** (so a
  variable keeps the same colour everywhere).

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
