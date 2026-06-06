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

    def on_data_loaded(self, df) -> None:
        """Called by the main window when a new dataset is loaded.

        Tabs that present data override this to refresh themselves (e.g. the
        plotting tab repopulates its variable list). Default is a no-op so
        tabs that don't consume data need not implement it.
        """

