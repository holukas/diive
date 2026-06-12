"""
GUI.WIDGETS.COMBO: COMBO-BOX POPUP FIX
======================================

On the frameless, translucent Studio window every ``QComboBox`` drop-down shows
its native window frame and drop shadow as ugly black bars above/below the list
(the same artefact the Studio ``QMenu``s avoid). :func:`install_combo_popup_fix`
installs one application-wide event filter that makes **every** combo popup
frameless + shadowless + translucent the moment its container is created — so all
dropdowns across the app are fixed from a single place, including ones added
later.

Pure presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QComboBox

_POPUP_CLASS = "QComboBoxPrivateContainer"


class _ComboPopupFixer(QObject):
    """Strips the native frame/shadow from every combo-box popup container."""

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        # The popup container is created as a child of the combo; catch it the
        # moment it's polished (before the first show) and re-flag its window.
        if event.type() == QEvent.Type.ChildPolished and isinstance(obj, QComboBox):
            child = event.child()
            if (child is not None and child.isWidgetType()
                    and child.metaObject().className() == _POPUP_CLASS
                    and not (child.windowFlags()
                             & Qt.WindowType.NoDropShadowWindowHint)):
                child.setWindowFlags(child.windowFlags()
                                     | Qt.WindowType.FramelessWindowHint
                                     | Qt.WindowType.NoDropShadowWindowHint)
                child.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        return False


def install_combo_popup_fix(app) -> _ComboPopupFixer:
    """Install the app-wide combo-popup fix and return the (app-parented) filter."""
    fixer = _ComboPopupFixer(app)  # parented to the app so it stays alive
    app.installEventFilter(fixer)
    return fixer
