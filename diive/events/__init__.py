"""
EVENTS: TIME-STAMPED EVENT MARKERS
==================================

Represent things that happened at a point in time or over a period (fertilization,
harvest, grazing, management interventions) as 0/1 data columns and as overlays on
time-series plots.

Part of the diive library: https://github.com/holukas/diive
"""
from diive.events.event import (
    CATEGORY_COLORS,
    Event,
    event_to_flag,
    make_event_flag_name,
    overlay_events,
)

__all__ = [
    "CATEGORY_COLORS",
    "Event",
    "event_to_flag",
    "make_event_flag_name",
    "overlay_events",
]
