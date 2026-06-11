"""
GUI.METADATA_STORE: APP-WIDE VARIABLE METADATA SINGLETON
========================================================

A thin Qt wrapper around the library's :class:`diive.core.metadata.MetadataStore`
so any tab or widget can read/edit per-variable metadata (tags + provenance)
without threading it through every ``on_data_loaded`` call. Mirrors the
``theme.manager`` / ``site.manager`` pattern: access the singleton as
``metadata_store.manager``, read ``manager.store`` directly, and edit through the
convenience methods so the ``changed`` signal fires and dependent widgets repaint.

The model and all domain logic live in the library; this only adds the Qt signal
and a few editing shortcuts (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from diive.core.metadata import ATTRS_KEY, FAVORITE, MetadataStore

__all__ = ["ATTRS_KEY", "MetadataManager", "manager"]


class MetadataManager(QObject):
    """Live, app-wide holder of variable metadata; emits ``changed`` on edits."""

    #: Emitted after any edit so the variable lists / explorer repaint.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.store = MetadataStore()

    def notify(self) -> None:
        """Emit ``changed`` after the main window records load/feature provenance
        directly on ``store`` (kept out of the loop so a batch is one repaint)."""
        self.changed.emit()

    def add_user_tag(self, name: str, tag: str) -> None:
        self.store.add_user_tag(name, tag)
        self.changed.emit()

    def remove_user_tag(self, name: str, tag: str) -> None:
        self.store.remove_user_tag(name, tag)
        self.changed.emit()

    def toggle_user_tag(self, name: str, tag: str) -> None:
        md = self.store.get(name)
        if md.is_user_tag(tag):
            self.store.remove_user_tag(name, tag)
        else:
            self.store.add_user_tag(name, tag)
        self.changed.emit()

    def is_favorite(self, name: str) -> bool:
        md = self.store.peek(name)
        return md is not None and FAVORITE in md.tags


#: Process-wide singleton; import as ``from diive.gui import metadata_store`` then
#: ``metadata_store.manager``.
manager = MetadataManager()
