"""
EVENTS.EVENT: TIME-STAMPED EVENT MARKERS
========================================

A small domain model for *events* — things that happened at a point in time or
over a period (fertilization, harvest, grazing, a management intervention, an
instrument swap). Each event carries a name, an optional category, a start and an
optional end, and presentation hints (description, colour).

Two reusable operations live here (no GUI, no Qt):

- :func:`event_to_flag` turns an event into a **yes/no (0/1) data column** aligned
  to a timestamp index: 1 where the event was active, 0 otherwise. This is the
  data representation of an event — a normal series that can be saved, plotted and
  processed like any other variable.
- :func:`overlay_events` draws events onto an existing matplotlib axes (a vertical
  line for an instant, a shaded span for a period), so any time-series plot can
  show when events occurred. It works on the value-vs-time axes (``axis='x'``) and
  on the date/time heatmap, where the date is on the y-axis (``axis='y'``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.dates as mdates
import pandas as pd

#: Default colours for common management categories (Material Design 400-level),
#: so events read clearly without the caller having to pick a colour. Categories
#: not listed here cycle through ``_FALLBACK_PALETTE`` by insertion order.
CATEGORY_COLORS: dict[str, str] = {
    "fertilization": "#66BB6A",   # green 400
    "harvest": "#FFA726",         # orange 400
    "grazing": "#8D6E63",         # brown 400
    "tillage": "#A1887F",         # brown 300
    "sowing": "#9CCC65",          # light green 400
    "irrigation": "#29B6F6",      # light blue 400
    "cut": "#FFCA28",             # amber 400
    "management": "#AB47BC",      # purple 400
    "instrument": "#EF5350",      # red 400
    "disturbance": "#EC407A",     # pink 400
}

#: Cycled for events whose category has no entry in ``CATEGORY_COLORS``.
_FALLBACK_PALETTE: list[str] = [
    "#42A5F5", "#26A69A", "#FF7043", "#7E57C2", "#5C6BC0",
    "#26C6DA", "#9CCC65", "#FFEE58", "#BDBDBD", "#78909C",
]


def make_event_flag_name(name: str) -> str:
    """Column name for an event's 0/1 flag: ``EVENT_<sanitized name>``.

    Whitespace collapses to single underscores so the result is a clean column
    label; other characters are kept (diive column names are otherwise free-form).
    """
    cleaned = "_".join(str(name).split())
    return f"EVENT_{cleaned}" if cleaned else "EVENT"


@dataclass
class Event:
    """A single time-stamped event.

    An event is either an **instant** (``end is None`` — a single point in time,
    e.g. a fertilization on one day) or a **period** (``end`` set — e.g. a grazing
    interval or a start date plus a duration). ``start``/``end`` are coerced to
    ``pandas.Timestamp``; a period with ``end < start`` is rejected.

    Args:
        name: Short label shown on plots and used to build the flag column name.
        start: When the event began (or the instant it occurred).
        end: When the event ended; ``None`` for an instant event.
        category: Optional grouping (e.g. ``"fertilization"``) that drives the
            default colour via :data:`CATEGORY_COLORS`.
        description: Free-text note.
        color: Explicit colour; when ``None`` it is resolved from the category.
    """

    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None = None
    category: str = ""
    description: str = ""
    color: str | None = None

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        if self.end is not None:
            self.end = pd.Timestamp(self.end)
            if self.end < self.start:
                raise ValueError(
                    f"Event '{self.name}': end ({self.end}) is before start "
                    f"({self.start}).")

    @property
    def is_range(self) -> bool:
        """True for a period event (has an end), False for an instant."""
        return self.end is not None

    @property
    def duration(self) -> pd.Timedelta:
        """Length of the event (zero for an instant)."""
        return (self.end - self.start) if self.end is not None else pd.Timedelta(0)

    @property
    def flag_name(self) -> str:
        """The 0/1 flag column name for this event (see :func:`make_event_flag_name`)."""
        return make_event_flag_name(self.name)

    def resolved_color(self, fallback_index: int = 0) -> str:
        """The colour to draw this event in: explicit, else by category, else a
        cycled fallback (indexed by ``fallback_index`` so a list of events with no
        category still get distinct colours)."""
        if self.color:
            return self.color
        cat = (self.category or "").strip().lower()
        if cat in CATEGORY_COLORS:
            return CATEGORY_COLORS[cat]
        return _FALLBACK_PALETTE[fallback_index % len(_FALLBACK_PALETTE)]

    def to_dict(self) -> dict:
        """Serialise to a plain dict (ISO timestamps) for persistence."""
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end is not None else None,
            "category": self.category,
            "description": self.description,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Rebuild an :class:`Event` from :meth:`to_dict` output."""
        end = data.get("end")
        return cls(
            name=str(data.get("name", "")),
            start=pd.Timestamp(data["start"]),
            end=pd.Timestamp(end) if end else None,
            category=str(data.get("category", "")),
            description=str(data.get("description", "")),
            color=data.get("color") or None,
        )


def event_to_flag(event: Event, index: pd.DatetimeIndex,
                  name: str | None = None) -> pd.Series:
    """Build the **yes/no (0/1) flag series** for an event, aligned to ``index``.

    The result is 1 where the event was active and 0 everywhere else — a normal
    data column that can be saved, plotted, and processed like any other variable.

    - **Period event** (``event.end`` set): 1 for every record in
      ``[start, end]`` (inclusive).
    - **Instant event** (``event.end is None``): 1 at the single record nearest to
      ``event.start`` (so a point event always marks exactly one record, even when
      its timestamp falls between two records).

    Args:
        event: The event to encode.
        index: Target timestamp index (the dataset's index).
        name: Optional series name; defaults to ``event.flag_name``.

    Returns:
        Integer ``Series`` of 0/1 indexed by ``index``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    flag = pd.Series(0, index=index, dtype="int64",
                     name=name or event.flag_name)
    if len(index) == 0:
        return flag
    if event.is_range:
        mask = (index >= event.start) & (index <= event.end)
        flag.loc[mask] = 1
    else:
        # Mark the single record nearest the instant, so a point event always
        # lands on exactly one record regardless of sampling.
        pos = index.get_indexer([event.start], method="nearest")[0]
        if pos != -1:
            flag.iloc[pos] = 1
    return flag


def overlay_events(ax, events, *, axis: str = "x", show_labels: bool = True,
                   label_fontsize: int = 8, line_alpha: float = 0.9,
                   span_alpha: float = 0.16, linewidth: float = 1.3) -> None:
    """Draw events onto an existing matplotlib axes.

    Instant events become a dashed line; period events become a translucent shaded
    span bordered by thin start/end lines. Positions are converted with
    :func:`matplotlib.dates.date2num`, which matches the float units of any diive
    datetime axis — both the value-vs-time line panels (``axis='x'``) and the
    date/time heatmap, where the date is the y-axis (``axis='y'``).

    This only *adds* artists; it never clears or rescales the axes, so it composes
    on top of an already-rendered plot.

    Args:
        ax: Target axes (already drawn).
        events: Iterable of :class:`Event`.
        axis: ``'x'`` to draw against time on the x-axis (line/cumulative/daily
            panels), ``'y'`` to draw against date on the y-axis (heatmap).
        show_labels: Annotate each event with its name.
        label_fontsize: Font size for the labels.
        line_alpha: Opacity of instant/border lines.
        span_alpha: Opacity of the period shading.
        linewidth: Width of instant/border lines.
    """
    events = list(events)
    if not events:
        return
    vertical = axis == "x"  # draw against the x-axis (time runs horizontally)
    for i, ev in enumerate(events):
        color = ev.resolved_color(i)
        s = mdates.date2num(ev.start)
        if ev.is_range:
            e = mdates.date2num(ev.end)
            if vertical:
                ax.axvspan(s, e, facecolor=color, alpha=span_alpha,
                           edgecolor="none", zorder=2)
                ax.axvline(s, color=color, alpha=line_alpha, lw=linewidth, zorder=3)
                ax.axvline(e, color=color, alpha=line_alpha * 0.6, lw=linewidth,
                           ls=":", zorder=3)
            else:
                ax.axhspan(s, e, facecolor=color, alpha=span_alpha,
                           edgecolor="none", zorder=2)
                ax.axhline(s, color=color, alpha=line_alpha, lw=linewidth, zorder=3)
                ax.axhline(e, color=color, alpha=line_alpha * 0.6, lw=linewidth,
                           ls=":", zorder=3)
        else:
            if vertical:
                ax.axvline(s, color=color, alpha=line_alpha, lw=linewidth,
                           ls="--", zorder=3)
            else:
                ax.axhline(s, color=color, alpha=line_alpha, lw=linewidth,
                           ls="--", zorder=3)
        if show_labels and ev.name:
            _label_event(ax, s, ev, color, vertical, label_fontsize)


def _label_event(ax, pos: float, ev: Event, color: str, vertical: bool,
                 fontsize: int) -> None:
    """Place a small name tag at the start of an event, anchored to the axes edge
    so it stays visible as the data is zoomed/panned."""
    if vertical:
        ax.annotate(
            ev.name, xy=(pos, 1.0), xycoords=("data", "axes fraction"),
            xytext=(2, -2), textcoords="offset points", rotation=90,
            ha="left", va="top", fontsize=fontsize, color=color, zorder=4,
            clip_on=True)
    else:
        ax.annotate(
            ev.name, xy=(0.0, pos), xycoords=("axes fraction", "data"),
            xytext=(2, 2), textcoords="offset points",
            ha="left", va="bottom", fontsize=fontsize, color=color, zorder=4,
            clip_on=True)
