# Events Examples

Examples for marking time-stamped *events* on time series and encoding them as data.

1 example covering event creation, 0/1 flag columns, and plot overlays.

## Examples

- **events_event.py** — Create instant/period `Event`s, turn them into 0/1 data columns with `event_to_flag`, overlay them on a plot with `overlay_events`, and apply a custom category palette

## Common Patterns

**Encode an event as a 0/1 column:**

```python
from diive.events import Event, event_to_flag

campaign = Event("Field campaign", "2022-07-01", "2022-07-21", category="category2")
flag = event_to_flag(campaign, df.index)   # 1 during the event, 0 otherwise
```

**Overlay events on any plot:**

```python
from diive.events import overlay_events

overlay_events(ax, [campaign], axis="x")   # adds a shaded span + label; never rescales
```

## Running Examples

```bash
uv run python examples/events/events_event.py
```

## Related Classes

See `dv.events` for full API documentation:

- `Event` — A single instant or period time-stamped marker
- `event_to_flag` — Encode an event as a 0/1 series aligned to an index
- `overlay_events` — Draw events (lines/spans) onto a datetime axes
- `make_event_flag_name` — Build the `EVENT_<name>` column name
- `CATEGORY_COLORS` — Default generic category colour palette
