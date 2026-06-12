"""
GUI.WIDGETS.MENU: STUDIO CONTEXT MENU
=====================================

A factory for ``QMenu``s styled to match the Studio look. A plain ``QMenu`` on the
frameless, translucent Studio window renders its popup with a black background /
shadow (unreadable); the Studio header menus avoid this by being frameless +
no-shadow + translucent with the ``#studiomenu`` object name (the QSS then rounds
them into a white card). :func:`studio_menu` packages that treatment so every
context menu (card actions, tab pin, variable-list right-click, …) looks the same.

Pure presentation.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu


def studio_menu(parent=None) -> QMenu:
    """A ``QMenu`` styled as a rounded white Studio card (no black popup frame)."""
    m = QMenu(parent)
    m.setObjectName("studiomenu")
    m.setWindowFlags(m.windowFlags()
                     | Qt.WindowType.FramelessWindowHint
                     | Qt.WindowType.NoDropShadowWindowHint)
    m.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return m
