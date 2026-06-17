"""
GUI.WIDGETS.FLUX_PIPELINE_RAIL: STAGE RAIL FOR THE FLUX PROCESSING CHAIN
=======================================================================

The flux processing chain is a pipeline of levels (Input → L2 → L3.1 → L3.2 →
L3.3 → L4.1). This widget renders that pipeline as a compact horizontal **rail**
of selectable stage cards joined by chevrons — a navigation + status surface, not
a parameter form. Each :class:`StageCard` shows a level badge, a title, and a
live **status pill** (e.g. ``4 tests``, ``2 scenarios``, ``rf · mds``, ``off``)
whose colour encodes the stage's state. Selecting a card tells the host tab to
swap that stage's controls into its inspector; a completed run lights every
reached card with a green ✓.

GUI-only presentation: dumb display widgets that emit ``clicked`` / ``selected``;
the tab computes each stage's status text/kind and drives selection + run-reach.
Colours read from ``theme.manager`` so they track the Studio look.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme

_DONE = "#2E9E5B"   # green — a stage reached by the last run
_MUTED = "#6B7780"

#: status-pill palette per kind -> (background, foreground)
_PILL = {
    "set":  ("#DCE3E8", "#2A3942"),   # configured / active — accent tint
    "todo": ("#F0F0EC", _MUTED),      # at defaults, nothing chosen yet
    "off":  ("#F0F0EC", "#A0A6AC"),   # optional level switched off
    "warn": ("#FDECEA", "#C0392B"),   # required but empty (e.g. L2 with 0 tests)
}


class StageCard(QFrame):
    """One pipeline stage: level badge + title + status pill (a dumb display)."""

    clicked = Signal()

    def __init__(self, badge: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected = False
        self._reached = False
        self._kind = "todo"
        self.setObjectName("stagecard")
        self.setFixedSize(150, 86)
        self.setCursor(Qt.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(11, 8, 11, 9)
        v.setSpacing(5)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._badge = QLabel(badge)
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setFixedHeight(18)
        self._badge.setMinimumWidth(36)
        head.addWidget(self._badge)
        head.addStretch(1)
        self._check = QLabel("")  # ✓ once the stage is reached by a run
        self._check.setStyleSheet(f"color: {_DONE}; font-weight: bold;")
        head.addWidget(self._check)
        v.addLayout(head)

        self._title = QLabel(title)
        self._title.setStyleSheet("font-weight: 600;")
        v.addWidget(self._title)

        self._pill = QLabel("—")
        self._pill.setObjectName("stagepill")
        self._pill.setAlignment(Qt.AlignCenter)
        v.addWidget(self._pill)
        v.addStretch(1)
        self._restyle()

    def set_status(self, text: str, kind: str) -> None:
        self._pill.setText(text)
        self._pill.setToolTip(text)
        self._kind = kind
        self._restyle()

    def set_selected(self, on: bool) -> None:
        self._selected = on
        self._restyle()

    def set_reached(self, on: bool) -> None:
        self._reached = on
        self._check.setText("✓" if on else "")
        self._restyle()

    def _restyle(self) -> None:
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        base_border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        if self._selected:
            border, width = accent, 2
        elif self._reached:
            border, width = _DONE, 1
        else:
            border, width = base_border, 1
        self.setStyleSheet(
            f"QFrame#stagecard {{ border: {width}px solid {border}; "
            f"border-radius: 12px; background: white; }}"
            + theme.manager.tooltip_qss())
        self._badge.setStyleSheet(
            f"background: {_DONE if self._reached else accent}; color: white; "
            f"border-radius: 9px; font-size: 10px; font-weight: bold; padding: 0 6px;")
        bg, fg = _PILL.get(self._kind, _PILL["todo"])
        self._pill.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 8px; "
            f"padding: 2px 6px; font-size: 11px;")

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PipelineRail(QWidget):
    """A horizontal rail of :class:`StageCard`s joined by chevrons.

    ``stages`` is a list of ``(badge, title)`` pairs. Emits ``selected(index)``
    when a card is clicked; the host drives selection highlight
    (:meth:`set_selected`), per-stage status (:meth:`card`), and how far the last
    run reached (:meth:`set_reached_through`).
    """

    selected = Signal(int)

    def __init__(self, stages: list[tuple[str, str]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(0)
        self._cards: list[StageCard] = []
        for i, (badge, title) in enumerate(stages):
            if i > 0:
                chev = QLabel("›")
                chev.setAlignment(Qt.AlignCenter)
                chev.setFixedWidth(22)
                chev.setStyleSheet(f"color: #B7BEC4; font-size: 22px;")
                h.addWidget(chev)
            card = StageCard(badge, title)
            card.clicked.connect(lambda idx=i: self.selected.emit(idx))
            h.addWidget(card)
            self._cards.append(card)
        h.addStretch(1)

    def card(self, i: int) -> StageCard:
        return self._cards[i]

    def set_selected(self, idx: int) -> None:
        for i, c in enumerate(self._cards):
            c.set_selected(i == idx)

    def set_reached_through(self, idx: int) -> None:
        """Light cards 0..idx as reached (idx < 0 clears all)."""
        for i, c in enumerate(self._cards):
            c.set_reached(i <= idx)
