"""
GUI.WIDGETS.PROGRESS_BAR: SHARED PROGRESS BAR
=============================================

One progress-bar design, used the same way everywhere. A thin slip of a bar
(16 px, text on top), hidden until work starts, that can run either *busy*
(indeterminate marquee, for a single long call that can't report progress) or
*determinate* (0-1000 permille, for staged work with a ``progress_callback``).

The colours come from the app-wide ``QProgressBar`` stylesheet (see
``theme.build_qss``), so this widget only owns the geometry + the busy/
determinate/finish state transitions.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QProgressBar, QWidget


class ProgressBar(QProgressBar):
    """App-wide progress bar with the shared geometry and busy/determinate helpers.

    Start with :meth:`start_busy` (unknown duration) or :meth:`set_progress`
    (known fraction), and call :meth:`finish` when done (hides it again).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextVisible(True)
        self.setFixedHeight(16)
        self.setVisible(False)

    def start_busy(self, text: str = "Working…") -> None:
        """Show an indeterminate (marquee) bar with *text* — for a single long
        call that cannot report incremental progress."""
        self.setRange(0, 0)
        self.setFormat(text)
        self.setVisible(True)

    def set_progress(self, permille: int, text: str | None = None) -> None:
        """Show a determinate bar at *permille* (0-1000); optionally set *text*."""
        self.setRange(0, 1000)
        self.setValue(max(0, min(1000, permille)))
        if text is not None:
            self.setFormat(text)
        self.setVisible(True)

    def finish(self) -> None:
        """Hide the bar (work finished or failed)."""
        self.setVisible(False)
