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
- a **figure** with several panels — full time series, cumulative sum, mean daily
  cycle, and a date/time heatmap;
- a **strip of statistic cards** along the bottom (count, mean, SD, min/max,
  percentiles, …).

### Plot ▸ Heatmap / Time series

Each plot method opens as its own tab.

- **Click** a variable to plot it.
- **Ctrl + click** more variables to compare them in extra panels (up to 5):
  - *Heatmaps* line up **side by side** (shared date axis).
  - *Time series* **stack** top-to-bottom (shared time axis), each its own colour.
- **Ctrl + click** a shown variable again to remove its panel.
- Use the small toolbar (bottom-right of the plot) to **pan, zoom, and save** the
  figure. Zooming one panel zooms them all.

### Tools ▸ Feature engineering

Build new features (lags, rolling stats, differences, EMA, polynomials, STL,
timestamp parts, …) with diive's feature engineer:

1. **Click** variables to move them into *Selected features*.
2. Tick the stages you want and set their options.
3. **Run feature engineering**.
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

- Your appearance settings, window size/position, and last-used filetype are
  **remembered** between sessions.
- The window sizes itself to your screen on first launch.
- A short loading cue appears on a variable while its plot is being drawn.
- Stuck or something looks off? Check the **Log** tab for messages.

---

*Part of the diive library — https://github.com/holukas/diive*
