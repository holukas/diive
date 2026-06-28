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

#: The category palette every session starts with: simple generic categories the
#: user renames/recolours/deletes to taste. There is always at least one (the
#: dialog refuses to delete the last), so events can always be categorised.
DEFAULT_CATEGORIES: dict[str, str] = {
    "category1": "#66BB6A",   # green 400
    "category2": "#FFA726",   # orange 400
    "category3": "#8D6E63",   # brown 400
}

#: Colours cycled when the user adds further categories. Mirrors the library's
#: ``dv.events.CATEGORY_COLORS`` order so ``categoryN`` here gets the same colour
#: the library would resolve it to by default (``add_category`` indexes by ``i-1``).
_ADD_PALETTE: list[str] = [
    "#66BB6A", "#FFA726", "#8D6E63", "#A1887F", "#9CCC65",
    "#29B6F6", "#FFCA28", "#AB47BC", "#EF5350", "#EC407A",
]


class EventManager(QObject):
    """Live, app-wide list of :class:`diive.events.Event` plus a visibility flag.

    Access the singleton as ``events.manager``. Edit through :meth:`add`,
    :meth:`replace`, :meth:`remove`, :meth:`set_visible` so the ``changed`` signal
    fires and dependent plots refresh.

    It also holds a user-managed **category palette** (``categories``: a
    ``{name: hex}`` map): a category's colour, editable in the Events tab's
    *Manage categories* dialog, overrides the library default for every event of
    that category — on the cards and on the plot overlays. Palette edits emit
    :attr:`categories_changed` (not :attr:`changed`) so they only trigger a
    repaint, never a flag-column rebuild.
    """

    #: Emitted after any edit (add/replace/remove/clear/visibility) so dependent
    #: widgets can repaint.
    changed = Signal()
    #: Emitted when only the category palette changed (colour/add/remove) — a
    #: presentation-only change that should repaint cards/overlays but not touch
    #: the events themselves or their flag columns.
    categories_changed = Signal()
    #: Requests the Overview to focus its time axis on a window ``(start, end)``
    #: (``end`` may be ``None`` for an instant). Wired in ``MainWindow`` so a card
    #: can say "show me where this event is" without referencing the main window.
    focus_requested = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []
        #: Master toggle: draw events on the plots, or not.
        self.visible: bool = True
        #: User category palette ``{name: hex}`` (insertion-ordered), seeded with
        #: :data:`DEFAULT_CATEGORIES`.
        self.categories: dict[str, str] = dict(DEFAULT_CATEGORIES)

    def add(self, event: Event) -> None:
        self._register_category(event)
        self.events.append(event)
        self.changed.emit()

    def replace(self, index: int, event: Event) -> None:
        """Replace the event at ``index`` (used when editing an existing event)."""
        if 0 <= index < len(self.events):
            self._register_category(event)
            self.events[index] = event
            self.changed.emit()

    def _register_category(self, event: Event) -> None:
        """Add an event's category to the palette if it's a new one, so a category
        typed when adding an event (e.g. via the editable combo) shows up in the
        *Manage categories* dialog. Resolves its colour from the event's own
        colour, else the library default. Silent (no signal) — the ``changed``
        emitted by the caller already repaints; this only keeps the palette in
        sync with the categories actually in play."""
        cat = (event.category or "").strip()
        if cat and cat not in self.categories:
            self.categories[cat] = event.resolved_color(0, colors=self.categories)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.events):
            del self.events[index]
            self.changed.emit()

    def duplicate(self, index: int) -> None:
        """Append a copy of the event at ``index`` (name suffixed ``" copy"``)."""
        if 0 <= index < len(self.events):
            src = self.events[index]
            self.events.append(Event(
                name=f"{src.name} copy", start=src.start, end=src.end,
                category=src.category, description=src.description,
                color=src.color))
            self.changed.emit()

    def shift(self, index: int, delta) -> None:
        """Move the event at ``index`` (and its end, if any) by ``delta``."""
        if 0 <= index < len(self.events):
            ev = self.events[index]
            self.events[index] = Event(
                name=ev.name, start=ev.start + delta,
                end=(ev.end + delta) if ev.end is not None else None,
                category=ev.category, description=ev.description, color=ev.color)
            self.changed.emit()

    def request_focus(self, start, end) -> None:
        """Ask the Overview to focus its time axis on this event's window."""
        self.focus_requested.emit(start, end)

    def clear(self) -> None:
        if self.events:
            self.events = []
            self.changed.emit()

    def set_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible != self.visible:
            self.visible = visible
            self.changed.emit()

    # --- category palette ---------------------------------------------
    def add_category(self) -> str:
        """Append a new ``categoryN`` (next free index) with a cycled colour and
        return its name. Emits :attr:`categories_changed`."""
        i = len(self.categories) + 1
        while f"category{i}" in self.categories:
            i += 1
        name = f"category{i}"
        self.categories[name] = _ADD_PALETTE[(i - 1) % len(_ADD_PALETTE)]
        self.categories_changed.emit()
        return name

    def set_category(self, name: str, color: str) -> None:
        """Define (or recolour) a category. Emits :attr:`categories_changed`."""
        name = str(name).strip()
        if not name:
            return
        self.categories[name] = str(color)
        self.categories_changed.emit()

    def rename_category(self, old: str, new: str) -> bool:
        """Rename a category in place (preserving its position and colour) and
        re-point every event that used it. No-op (returns False) on an empty,
        unchanged, missing, or already-taken name."""
        new = str(new).strip()
        if not new or old not in self.categories or new == old \
                or new in self.categories:
            return False
        # Rebuild preserving insertion order so the row keeps its place.
        self.categories = {(new if k == old else k): v
                           for k, v in self.categories.items()}
        for ev in self.events:
            if (ev.category or "").strip() == old:
                ev.category = new
        self.categories_changed.emit()
        return True

    def remove_category(self, name: str) -> None:
        """Drop a category. The **last remaining** category is never removed, so
        there is always at least one to assign events to."""
        if name in self.categories and len(self.categories) > 1:
            del self.categories[name]
            self.categories_changed.emit()

    def known_categories(self) -> dict[str, str]:
        """Effective ``{name: hex}`` palette: the managed categories plus any
        category currently used by an event but not in the palette (so the
        add-event combo still lists everything that's in play)."""
        palette: dict[str, str] = dict(self.categories)
        for ev in self.events:
            cat = (ev.category or "").strip()
            if cat and cat not in palette:
                palette[cat] = ev.resolved_color(0, colors=self.categories)
        return palette

    def as_dict(self) -> dict:
        """Serialise for persistence (GUI prefs and saved projects)."""
        return {
            "visible": self.visible,
            "events": [e.to_dict() for e in self.events],
            "categories": dict(self.categories),
        }

    def load_dict(self, data: dict) -> None:
        """Restore from :meth:`as_dict` output; malformed entries are skipped."""
        if not data:
            return
        self.visible = bool(data.get("visible", self.visible))
        cats = data.get("categories")
        # Use the saved palette when present; otherwise keep the seeded defaults
        # (older configs/projects predate the palette).
        if isinstance(cats, dict) and cats:
            self.categories = {str(k): str(v) for k, v in cats.items()}
        loaded: list[Event] = []
        for entry in data.get("events", []) or []:
            try:
                loaded.append(Event.from_dict(entry))
            except Exception:
                continue  # tolerate older/garbled entries rather than fail the load
        self.events = loaded
        for ev in self.events:
            self._register_category(ev)
        self.changed.emit()


#: Process-wide singleton; import as ``from diive.gui import events`` then
#: ``events.manager``.
manager = EventManager()
