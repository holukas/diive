"""
CORE.METADATA: PER-VARIABLE METADATA MODEL
==========================================

A small, headless model for the metadata that travels alongside a variable as it
is processed: a set of **tags** (some auto-assigned by the operation that
produced a column, some user-set like ``favorite``) plus a **provenance** chain
(origin — original / modified / derived; parent variable; operation + params).

A variable is often modified many times (load -> outlier filter -> gap-fill -> …),
each step producing a new column. The column *name* is the only built-in record
of where it came from, which is lossy; this model keeps the lineage explicit so
the current version's history can be inspected.

This module is pure domain logic — no Qt, no wall-clock calls (timestamps are
passed in by the caller). The GUI records into a :class:`MetadataStore` and
renders it; the library owns the model and the GUI<->tab payload contract
(:data:`ATTRS_KEY` / :func:`provenance_attr`).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Origin of a variable relative to the loaded dataset.
ORIGINAL = "original"   # straight from the loaded file
MODIFIED = "modified"   # a transformed copy of a parent (e.g. outliers removed)
DERIVED = "derived"     # computed *from* a parent (e.g. a flag, a feature)

#: Source of a tag — decides whether it round-trips through persistence.
USER = "user"
FUNCTION = "function"

#: Conventional user tag for marking a variable a favorite.
FAVORITE = "favorite"

#: Cap on a variable's free-text description (words). Enforced on write so the
#: stored value is always within the limit, wherever it came from.
MAX_DESCRIPTION_WORDS = 50


def truncate_words(text: str, n: int = MAX_DESCRIPTION_WORDS) -> str:
    """Trim ``text`` to at most ``n`` whitespace-separated words."""
    return " ".join((text or "").split()[:n])


def _coerce_tag_sources(tags) -> dict:
    """Normalize a persisted ``tags`` value to ``{tag: source}``.

    Accepts the current ``{tag: source}`` dict or an older bare list of tag
    strings (from a project saved before tag sources were tracked); list entries
    become user tags, since user tags were the only ones that round-tripped then.
    Unknown shapes yield an empty mapping rather than raising.
    """
    if isinstance(tags, dict):
        return {str(t): (s if s in (USER, FUNCTION) else USER)
                for t, s in tags.items()}
    if isinstance(tags, (list, tuple, set)):
        return {str(t): USER for t in tags}
    return {}

#: Key under which a DataFrame carries its provenance payload in ``df.attrs``.
#: Operation tabs set ``df.attrs[ATTRS_KEY]`` before emitting new columns; the
#: main window consumes it via :meth:`MetadataStore.from_attrs`.
ATTRS_KEY = "diive_metadata"


def provenance_attr(*, origin: str = MODIFIED, parent: str | None = None,
                    operation: str = "", params: dict | None = None,
                    tags: list[str] | None = None) -> dict:
    """Build one column's entry for a ``df.attrs[ATTRS_KEY]`` payload.

    Centralised so the GUI->tab contract has a single definition; the dict shape
    matches what :meth:`MetadataStore.from_attrs` consumes.
    """
    return {
        "origin": origin,
        "parent": parent,
        "operation": operation,
        "params": dict(params or {}),
        "tags": list(tags or []),
    }


@dataclass
class ProvenanceEntry:
    """One step in a variable's history."""

    operation: str
    params: dict = field(default_factory=dict)
    parent: str | None = None
    timestamp: str | None = None
    source: str = FUNCTION

    def describe(self) -> str:
        """One-line human description, e.g. ``Hampel (n_sigma=5.5, window=48)``."""
        text = self.operation or "operation"
        if self.params:
            kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
            text = f"{text} ({kv})"
        return text

    def to_dict(self) -> dict:
        """Serialize this provenance entry to a plain dict."""
        return {"operation": self.operation, "params": self.params,
                "parent": self.parent, "timestamp": self.timestamp,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceEntry":
        """Build a ProvenanceEntry from a serialized dict."""
        return cls(operation=d.get("operation", ""), params=dict(d.get("params") or {}),
                   parent=d.get("parent"), timestamp=d.get("timestamp"),
                   source=d.get("source", FUNCTION))


@dataclass
class VariableMetadata:
    """Tags + provenance for a single variable.

    ``tags`` is exposed as a set, but tag *sources* are tracked internally so the
    persistence layer can round-trip only the user-set ones.
    """

    name: str
    origin: str = ORIGINAL
    parents: list[str] = field(default_factory=list)
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    #: Free-text user note, capped at MAX_DESCRIPTION_WORDS words.
    description: str = ""
    #: tag -> source (USER / FUNCTION).
    _tag_sources: dict[str, str] = field(default_factory=dict)

    @property
    def tags(self) -> set[str]:
        """All tags on this variable as a set."""
        return set(self._tag_sources)

    def add_tag(self, tag: str, *, source: str = FUNCTION) -> None:
        """Add ``tag``. A user source is sticky — it is never downgraded to a
        function source by a later auto-tagging pass."""
        if self._tag_sources.get(tag) == USER:
            return
        self._tag_sources[tag] = source

    def remove_tag(self, tag: str) -> None:
        """Remove *tag* if present (no-op otherwise)."""
        self._tag_sources.pop(tag, None)

    def is_user_tag(self, tag: str) -> bool:
        """Return True if *tag* was set by the user (vs. auto-tagging)."""
        return self._tag_sources.get(tag) == USER

    def user_tags(self) -> list[str]:
        """Tags the user set (the only ones worth persisting)."""
        return [t for t, s in self._tag_sources.items() if s == USER]

    def to_dict(self) -> dict:
        """Full serialization (origin, parents, provenance, tags+sources, note)."""
        return {
            "name": self.name, "origin": self.origin, "parents": list(self.parents),
            "description": self.description,
            "tags": dict(self._tag_sources),
            "provenance": [p.to_dict() for p in self.provenance],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VariableMetadata":
        """Rebuild from :meth:`to_dict`, tolerant of older project layouts.

        Forward/back-compat: ``name`` may be absent (caller skips such records),
        and ``tags`` may be the current ``{tag: source}`` dict or an older bare
        list of tag strings (coerced to user tags, the only kind that persisted).
        """
        name = d.get("name") or d.get("var")  # older drafts keyed it as "var"
        if not name:
            raise ValueError("variable metadata entry has no name")
        md = cls(name=str(name), origin=d.get("origin", ORIGINAL),
                 parents=[str(p) for p in (d.get("parents") or [])],
                 description=d.get("description", ""))
        md._tag_sources = _coerce_tag_sources(d.get("tags"))
        md.provenance = [ProvenanceEntry.from_dict(p)
                         for p in (d.get("provenance") or []) if isinstance(p, dict)]
        return md


class MetadataStore:
    """Dict-like collection of :class:`VariableMetadata`, keyed by variable name."""

    def __init__(self) -> None:
        """Create an empty metadata store."""
        self._items: dict[str, VariableMetadata] = {}

    def __contains__(self, name: str) -> bool:
        return str(name) in self._items

    def peek(self, name: str) -> VariableMetadata | None:
        """Return the record for ``name`` without creating one (None if absent)."""
        return self._items.get(str(name))

    def get(self, name: str) -> VariableMetadata:
        """Return the record for ``name``, creating an ``original`` baseline if
        it does not exist yet."""
        name = str(name)
        md = self._items.get(name)
        if md is None:
            md = VariableMetadata(name=name)
            md.add_tag(ORIGINAL, source=FUNCTION)
            self._items[name] = md
        return md

    def record_original(self, names, *, operation: str | None = None,
                       timestamp: str | None = None) -> None:
        """Reset ``names`` to a clean ``original`` baseline (provenance cleared).

        Called on a fresh dataset load. Existing records are replaced, so stale
        provenance from a previous dataset does not linger; user tags are
        re-applied separately via :meth:`load_user_tags`.

        Pass ``operation`` (e.g. ``"Imported from <file>"``) to seed the history
        with the import as its first step — the caller supplies the ``timestamp``
        (the model takes no wall-clock readings).
        """
        for name in names:
            name = str(name)
            md = VariableMetadata(name=name)
            md.add_tag(ORIGINAL, source=FUNCTION)
            if operation:
                md.provenance.append(ProvenanceEntry(
                    operation=operation, timestamp=timestamp, source=FUNCTION))
            self._items[name] = md

    def record_derived(self, name: str, *, parent: str | None = None,
                       operation: str = "", params: dict | None = None,
                       tags: list[str] | None = None, origin: str = MODIFIED,
                       timestamp: str | None = None) -> VariableMetadata:
        """Mark ``name`` as a modified/derived column: set its origin, link its
        parent, append a provenance entry, and add the operation's tags.

        On first creation the column inherits its parent's full history, so the
        lineage is cumulative (e.g. ``FC`` -> ``FC_LOCALSD`` -> ``FC_LOCALSD_HAMPEL``
        shows the import, the Local SD step, and the Hampel step in order). The
        parent's history is *copied* (a snapshot taken now), not shared."""
        name = str(name)
        existing = self._items.get(name)
        md = existing or VariableMetadata(name=name)
        self._items[name] = md
        md.remove_tag(ORIGINAL)  # a transformed/derived column is not original
        md.origin = origin
        if parent and str(parent) not in md.parents:
            md.parents.append(str(parent))
        # Inherit the parent's lineage (history + ancestor links) once, when this
        # column is first recorded. Re-running an operation on an existing column
        # only appends its own new step.
        if existing is None and parent:
            pmd = self._items.get(str(parent))
            if pmd is not None:
                md.provenance = [
                    ProvenanceEntry.from_dict(e.to_dict()) for e in pmd.provenance]
                for ancestor in pmd.parents:
                    if ancestor not in md.parents:
                        md.parents.append(ancestor)
        md.provenance.append(ProvenanceEntry(
            operation=operation, params=dict(params or {}),
            parent=str(parent) if parent else None, timestamp=timestamp,
            source=FUNCTION))
        for tag in (tags or []):
            md.add_tag(tag, source=FUNCTION)
        return md

    def from_attrs(self, attrs: dict | None, *, timestamp: str | None = None) -> None:
        """Consume a ``df.attrs[ATTRS_KEY]`` payload, recording each column's
        provenance. Columns with no entry are ignored."""
        for col, spec in (attrs or {}).items():
            self.record_derived(
                col, parent=spec.get("parent"),
                operation=spec.get("operation", ""), params=spec.get("params"),
                tags=spec.get("tags"), origin=spec.get("origin", MODIFIED),
                timestamp=timestamp)

    def set_description(self, name: str, text: str) -> str:
        """Set a variable's free-text note (truncated to the word cap). Returns
        the stored value so the caller can reflect any truncation."""
        md = self.get(name)
        md.description = truncate_words(text)
        return md.description

    def descriptions(self) -> dict[str, str]:
        """``{name: description}`` for every variable with a non-empty note."""
        return {n: md.description for n, md in self._items.items() if md.description}

    def add_user_tag(self, name: str, tag: str) -> None:
        """Add a user-set *tag* to variable *name*."""
        self.get(name).add_tag(tag, source=USER)

    def remove_user_tag(self, name: str, tag: str) -> None:
        """Remove user *tag* from variable *name* (no-op if absent)."""
        md = self._items.get(str(name))
        if md is not None:
            md.remove_tag(tag)

    def drop(self, name: str) -> None:
        """Forget a deleted variable."""
        self._items.pop(str(name), None)

    def rename(self, mapping: dict) -> None:
        """Rename variables, preserving each one's metadata and lineage links.

        ``mapping`` is ``{old_name: new_name}``. Re-keys the store, updates each
        record's ``name``, and rewrites parent references (``parents`` lists and
        per-provenance ``parent``) so the renamed variables still point at each
        other. Entries not in ``mapping`` keep their names but still get their
        parent references remapped (a child of a renamed variable stays linked).
        """
        mapping = {str(o): str(n) for o, n in mapping.items() if str(o) != str(n)}
        if not mapping:
            return
        for md in self._items.values():
            md.name = mapping.get(md.name, md.name)
            md.parents = [mapping.get(p, p) for p in md.parents]
            for entry in md.provenance:
                if entry.parent is not None:
                    entry.parent = mapping.get(entry.parent, entry.parent)
        self._items = {md.name: md for md in self._items.values()}

    def user_tags(self) -> dict[str, list[str]]:
        """``{name: [user tags]}`` for every variable that has any — the
        persistence surface (function-set provenance tags do not round-trip)."""
        return {n: md.user_tags() for n, md in self._items.items() if md.user_tags()}

    def load_user_tags(self, data: dict | None) -> None:
        """Re-apply persisted user tags (keyed by variable name)."""
        if not data:
            return
        for name, tags in data.items():
            md = self.get(name)
            for tag in tags:
                md.add_tag(tag, source=USER)

    def clear_user_data(self) -> None:
        """Remove every user-set tag and description (the destructive counterpart
        to :meth:`load_user_data`). Origin, provenance, and function-set tags
        (e.g. ``hampel``, ``flag``) are kept."""
        for name in self._items:
            self.clear_variable_user_data(name)

    def clear_variable_user_data(self, name: str) -> None:
        """Remove one variable's user tags + description (keeps origin/provenance
        and function-set tags)."""
        md = self._items.get(str(name))
        if md is None:
            return
        for tag in md.user_tags():
            md.remove_tag(tag)
        md.description = ""

    def to_dict(self) -> dict:
        """Full serialization of every variable's metadata (for diive projects)."""
        return {"variables": [md.to_dict() for md in self._items.values()]}

    def load_dict(self, data: dict | None) -> None:
        """Replace the store contents from a :meth:`to_dict` payload.

        Malformed entries (e.g. from an older project layout missing required
        fields) are skipped rather than aborting the whole load, so a stale
        project still opens with whatever metadata is recoverable."""
        self._items = {}
        for vd in (data or {}).get("variables", []):
            if not isinstance(vd, dict):
                continue
            try:
                md = VariableMetadata.from_dict(vd)
            except (ValueError, TypeError, KeyError):
                continue
            self._items[md.name] = md

    def user_data(self) -> dict:
        """All persistable user content: ``{"tags": ..., "descriptions": ...}``."""
        return {"tags": self.user_tags(), "descriptions": self.descriptions()}

    def load_user_data(self, data: dict | None) -> None:
        """Re-apply persisted user content. Accepts the current
        ``{"tags": ..., "descriptions": ...}`` shape or the legacy flat
        ``{name: [tags]}`` (older configs)."""
        if not data:
            return
        if "tags" in data or "descriptions" in data:
            self.load_user_tags(data.get("tags") or {})
            for name, desc in (data.get("descriptions") or {}).items():
                self.set_description(name, desc)
        else:  # legacy flat tags dict
            self.load_user_tags(data)
