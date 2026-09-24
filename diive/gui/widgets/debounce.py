"""
GUI.WIDGETS.DEBOUNCE: RUN A SLOW SLOT ONCE AN INPUT HAS SETTLED
===============================================================

`Debouncer` sits between a control's change signal and a slow slot (a
recompute or re-render), so holding a spinbox arrow, typing a number or
dragging a slider runs the slot once, when the value stops changing, instead of
on every intermediate value.

Connect the control to ``trigger`` (it accepts and ignores the signal's
arguments). The slot is held as a bound method, which PySide6 keeps only
weakly, so the debouncer does not keep its owner alive (see
``widgets/weak_slot.py``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QTimer

#: Default quiet time before the slot runs.
DEBOUNCE_MS = 250


class Debouncer(QObject):
    """Run `slot` once, `ms` milliseconds after the last `trigger()`."""

    def __init__(self, parent: QObject, slot: Callable[[], None],
                 ms: int = DEBOUNCE_MS) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(ms)
        self._timer.timeout.connect(slot)

    def trigger(self, *_args) -> None:
        """(Re)start the wait; the signal's arguments are ignored."""
        self._timer.start()

    def pending(self) -> bool:
        """True while a triggered run has not happened yet."""
        return self._timer.isActive()

    def cancel(self) -> None:
        """Drop a pending run."""
        self._timer.stop()

    def flush(self) -> None:
        """Run a pending slot now (a no-op when nothing is pending)."""
        if self._timer.isActive():
            self._timer.stop()
            self._timer.timeout.emit()
