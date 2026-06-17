"""
GUI.TABS.BASE: TAB EXTENSION POINT
==================================

`DiiveTab` is the abstract base every GUI tab implements. The main window is
deliberately ignorant of concrete tabs: it reads the registry, instantiates
each `DiiveTab`, and adds `tab.widget()` under `tab.title`. Adding a new
feature area (e.g. the flux processing chain) means writing a subclass and
appending it to the registry -- the main window does not change.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from PySide6.QtWidgets import QWidget


class DiiveTab(ABC):
    """One top-level tab in the diive GUI.

    Subclasses build their UI in `build()` and return a short tab label from
    `title`. Construction is cheap; `build()` is where Qt widgets are created
    so the main window controls instantiation order.
    """

    #: Human-readable label shown on the tab. Override as a class attribute.
    title: str = "Tab"

    def __init__(self) -> None:
        self._widget: QWidget | None = None

    @abstractmethod
    def build(self) -> QWidget:
        """Create and return the root widget for this tab."""
        raise NotImplementedError

    def widget(self) -> QWidget:
        """Return the tab's root widget, building it once on first access."""
        if self._widget is None:
            self._widget = self.build()
        return self._widget

    def on_data_loaded(self, df, created: set | None = None) -> None:
        """Called by the main window when the dataset changes.

        Tabs that present data override this to refresh themselves (e.g. the
        plotting tab repopulates its variable list). `created` is the set of
        column names produced by the feature engineer (for "NEW" tagging).
        Default is a no-op so tabs that don't consume data need not implement it.
        """

    def save_state(self) -> dict:
        """Return a JSON-serializable snapshot of this tab's UI state (selected
        variables, control values, …) so a project can reopen it as it was.

        Default is empty. Override to capture inputs only — not heavy results
        (those re-compute on restore). Paired with :meth:`restore_state`.
        """
        return {}

    def restore_state(self, state: dict) -> None:
        """Re-apply a snapshot from :meth:`save_state`. Called after the tab has
        received the dataset, so its variable list is already populated. Must be
        tolerant of missing keys / unknown variables (skip rather than fail)."""

