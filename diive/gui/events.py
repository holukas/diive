"""
GUI.EVENTS: APP-WIDE EVENT STORE
================================

A tiny app-wide holder for the project's *events* (fertilization, harvest,
grazing, management interventions, …) and a single ``visible`` toggle that decides
whether they are drawn on the plots.

Like ``site.manager`` / ``metadata_store.manager`` this is a thin GUI store with no
domain logic: the events themselves are the **library's** :class:`diive.events.Event`
objects, and the flag-column / overlay maths live in the library. The GUI just
keeps the live list, emits ``changed`` when it is edited, and serialises it for the
GUI preferences and for saved projects.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from diive.events import Event


class EventManager(QObject):
    """Live, app-wide list of :class:`diive.events.Event` plus a visibility flag.

    Access the singleton as ``events.manager``. Edit through :meth:`add`,
    :meth:`replace`, :meth:`remove`, :meth:`set_visible` so the ``changed`` signal
    fires and dependent plots refresh.
    """

    #: Emitted after any edit (add/replace/remove/clear/visibility) so dependent
    #: widgets can repaint.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []
        #: Master toggle: draw events on the plots, or not.
        self.visible: bool = True

    def add(self, event: Event) -> None:
        self.events.append(event)
        self.changed.emit()

    def replace(self, index: int, event: Event) -> None:
        """Replace the event at ``index`` (used when editing an existing event)."""
        if 0 <= index < len(self.events):
            self.events[index] = event
            self.changed.emit()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.events):
            del self.events[index]
            self.changed.emit()

    def clear(self) -> None:
        if self.events:
            self.events = []
            self.changed.emit()

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible != self.visible:
            self.visible = visible
            self.changed.emit()

    def as_dict(self) -> dict:
        """Serialise for persistence (GUI prefs and saved projects)."""
        return {
            "visible": self.visible,
            "events": [e.to_dict() for e in self.events],
        }

    def load_dict(self, data: dict) -> None:
        """Restore from :meth:`as_dict` output; malformed entries are skipped."""
        if not data:
            return
        self.visible = bool(data.get("visible", self.visible))
        loaded: list[Event] = []
        for entry in data.get("events", []) or []:
            try:
                loaded.append(Event.from_dict(entry))
            except Exception:
                continue  # tolerate older/garbled entries rather than fail the load
        self.events = loaded
        self.changed.emit()


#: Process-wide singleton; import as ``from diive.gui import events`` then
#: ``events.manager``.
manager = EventManager()
