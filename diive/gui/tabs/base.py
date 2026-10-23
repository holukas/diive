"""
GUI.TABS.BASE: TAB EXTENSION POINT
==================================

`DiiveTab` is the abstract base every GUI tab implements. The main window is
deliberately ignorant of concrete tabs: it reads the registry, instantiates
each `DiiveTab`, and adds `tab.widget()` under `tab.title`. Adding a new
feature area means writing a subclass and appending it to the registry -- the
main window does not change.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from PySide6.QtWidgets import QWidget
from shiboken6 import Shiboken


#: Key the main window sets in a frame's ``attrs`` when the user declines to add
#: it (Cancel in the "Replace variables" question).
ADD_DECLINED_KEY = "diive_add_declined"


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

    def __del__(self, _sbk=Shiboken) -> None:
        """Delete a dropped tab's widget tree on the C++ side, in one piece.

        Without this, a tab in a reference cycle (most are) leaves its widgets
        to the garbage collector, which clears the Python wrappers in no
        particular order. PySide's ``addLayout`` creates wrappers for the
        ``QWidgetItem``/``QSpacerItem`` entries of the added layout, and
        clearing such a wrapper hands its C++ item to Python, which deletes it
        while the layout still lists it. The layout deletes it again when the
        tree goes: a double free, 0xc0000374 inside ``gc.collect()`` (L399).
        Finalizers run before the collector clears anything, so deleting the
        tree here destroys every item once, through its layout, and leaves only
        invalid wrappers to clear.

        Only a root that Python owns is deleted (one nobody parented, e.g. a
        tab a test dropped). A root inside the main window is Qt's, and the
        main window deletes a closed tab's widget itself. ``_sbk`` is bound at
        definition time because module globals can be gone at shutdown.
        """
        w = getattr(self, "_widget", None)
        if w is not None and _sbk.isValid(w) and _sbk.ownedByPython(w):
            _sbk.delete(w)

    def _emit_features(self, df) -> bool:
        """Send `df` to the main window to add it to the dataset.

        Returns False when the user cancelled the "Replace variables" question.
        The tab's status line then says that nothing was added, and the caller
        leaves its Add button enabled.
        """
        df.attrs.pop(ADD_DECLINED_KEY, None)
        self.featuresCreated.emit(df)
        if not df.attrs.pop(ADD_DECLINED_KEY, False):
            return True
        status = getattr(self, "status", None)
        if status is not None:
            status.setText("Nothing added: the existing variables were kept.")
        return False

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

