"""
GUI.WIDGETS.TAB_CHROME: SHARED TAB CHROME HELPERS
=================================================

Tiny presentation helpers for the bits of tab chrome that were copy-pasted
verbatim across the result-producing tab templates (correction, ML gap-filling,
partitioning):

  * :func:`build_titlebar` — the tracked, bold tab-title row with optional
    right-aligned action widgets (Copy Python, Run, Add ...),
  * :func:`list_header` — a bold list title with a muted parenthetical hint
    (e.g. ``Target (click to set target)``).

Presentation only (fonts/colours/labels); no domain logic.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from diive.gui import theme


def build_titlebar(title: str, *trailing: QWidget) -> QHBoxLayout:
    """A tracked, bold tab-title row. ``trailing`` widgets are added right-aligned
    in order (after a stretch), e.g. a Copy-Python button or Run/Add buttons."""
    bar = QHBoxLayout()
    bar.setContentsMargins(10, 8, 10, 8)
    label = QLabel(theme.manager.label_text(title))
    label.setFont(theme.manager.tracked_font(point_delta=1.0))
    label.setStyleSheet("font-weight: bold;")
    bar.addWidget(label)
    bar.addStretch(1)
    for w in trailing:
        bar.addWidget(w)
    return bar


def list_header(title: str, hint: str) -> QLabel:
    """A bold list title with a muted parenthetical hint, matching the gap-filling
    tabs' ``Target (click to set target)`` header."""
    label = QLabel(f"<b>{title}</b> <span style='color:#90A4AE'>({hint})</span>")
    label.setWordWrap(True)
    return label
