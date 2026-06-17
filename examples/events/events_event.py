"""
==================================
Time-Stamped Events on Time Series
==================================

Mark *events* on a time series and turn them into data. An event is anything you
want to flag on the time axis — a management intervention, an instrument swap, a
field campaign, a weather event, a maintenance day. Each event has a name, an
optional category, a start, and (for periods) an end.

This example shows the two reusable operations from ``dv.events``:

- ``event_to_flag`` turns an event into a **0/1 data column** aligned to your
  index (1 where the event was active, 0 otherwise) — a normal series you can
  save, plot, and process like any other variable.
- ``overlay_events`` draws events onto an existing matplotlib axes (a dashed line
  for an instant, a shaded span for a period), so any plot can show *when* things
  happened.

Categories are deliberately generic (``category1``, ``category2``, …): events can
mean anything, so you rename the categories to suit your study, or pass your own
``{category: colour}`` palette.

Best for: annotating time series with field/management/instrument events, and
encoding those annotations as analysable 0/1 columns.
"""

# %%
# Create Events
# ^^^^^^^^^^^^^
# An event is either an *instant* (a single point in time, ``end=None``) or a
# *period* (a start and an end). Categories drive the default colour.

import matplotlib.pyplot as plt

import diive as dv
from diive.events import Event, event_to_flag, overlay_events

# Load one year of example data
df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2022].copy()
nee = df['NEE_CUT_REF_f'].copy()  # net ecosystem exchange (CO2 flux)

# Instant events (single point in time)
instrument_swap = Event(
    name="Sensor swap",
    start="2022-05-12",
    category="category1",
    description="Replaced the gas analyzer.",
)

# Period events (start + end)
campaign = Event(
    name="Field campaign",
    start="2022-07-01",
    end="2022-07-21",
    category="category2",
    description="Three weeks of intensive sampling.",
)

# A period built from a start date plus a duration
import pandas as pd
maintenance_start = pd.Timestamp("2022-09-05")
maintenance = Event(
    name="Maintenance",
    start=maintenance_start,
    end=maintenance_start + pd.Timedelta(days=3),
    category="category3",
)

events = [instrument_swap, campaign, maintenance]

for ev in events:
    kind = "period" if ev.is_range else "instant"
    print(f"{ev.name:16s} {kind:8s} {ev.start.date()} -> "
          f"{ev.end.date() if ev.end is not None else '(instant)'}  "
          f"color={ev.resolved_color()}")

# %%
# Encode an Event as a 0/1 Data Column
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``event_to_flag`` aligns an event to the dataset's index. A period flags every
# record inside ``[start, end]``; an instant flags the single nearest record. The
# result is a normal series (named ``EVENT_<name>``) you can analyse or save.

campaign_flag = event_to_flag(campaign, df.index)
swap_flag = event_to_flag(instrument_swap, df.index)

print(f"\nFlag column name: {campaign_flag.name}")
print(f"Records during the field campaign: {int(campaign_flag.sum())}")
print(f"Records flagged for the instrument swap: {int(swap_flag.sum())}")  # exactly 1

# Use the flag like any other column, e.g. mean flux during vs. outside the campaign
during = nee[campaign_flag == 1].mean()
outside = nee[campaign_flag == 0].mean()
print(f"\nMean NEE during campaign:  {during:.2f} umol m-2 s-1")
print(f"Mean NEE outside campaign: {outside:.2f} umol m-2 s-1")

# %%
# Overlay Events on a Plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# ``overlay_events`` only *adds* artists (lines/spans + labels); it never clears or
# rescales the axes, so it composes on top of an already-drawn plot. Here we plot a
# daily-mean NEE series and mark the events on it.

daily = dv.times.resample_to_daily_agg(nee, agg='mean', mincounts_perc=0.5)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily.index, daily.values, color="#455A64", lw=1.0, label="NEE (daily mean)")
ax.axhline(0, color="#B0BEC5", lw=0.8, zorder=1)

overlay_events(ax, events, axis="x")

ax.set_ylabel("NEE (umol m-2 s-1)")
ax.set_title("Daily-mean NEE with site events overlaid")
ax.legend(loc="upper right")
fig.tight_layout()
# Disabled for the example gallery; set to plt.show() to display interactively.
# plt.show()

# %%
# Use Your Own Category Palette
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The default category colours are generic placeholders. To use your own palette
# without touching the events, pass a ``{category: hex}`` map to ``resolved_color``
# or ``overlay_events`` — it overrides the built-in defaults (case-insensitive).

my_palette = {
    "category1": "#E53935",  # red
    "category2": "#1E88E5",  # blue
    "category3": "#43A047",  # green
}

for ev in events:
    print(f"{ev.name:16s} -> {ev.resolved_color(colors=my_palette)}")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(daily.index, daily.values, color="#455A64", lw=1.0)
overlay_events(ax2, events, axis="x", colors=my_palette)
ax2.set_title("Same events, custom category palette")
fig2.tight_layout()
# plt.show()
