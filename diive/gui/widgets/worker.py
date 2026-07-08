"""
GUI.WIDGETS.WORKER: BACKGROUND-THREAD RUNNER
============================================

A small reusable runner that takes the duplicated "spawn a daemon thread, run a
function, marshal the result back to the GUI" idiom out of every tab. Before
this, each result-producing tab hand-rolled its own ``_Signals(QObject)`` with
``done``/``failed`` signals, a ``_running`` flag, a ``threading.Thread(...)``
launch, and a try/except inside the worker. :class:`WorkerRunner` owns all of
that once.

Usage:

    self._runner = WorkerRunner()
    self._runner.done.connect(self._on_done)
    self._runner.failed.connect(self._on_failed)
    ...
    self._runner.run(self._compute_payload, series, kwargs)

The work function runs off the GUI thread; its return value is delivered via
``done`` and any exception via ``failed`` (as ``str(err)``). ``run`` is a no-op
returning ``False`` while a previous job is still in flight (``is_running``).

The connection/threading semantics mirror the per-tab ``_Signals`` objects this
replaces, so behaviour is unchanged — only the boilerplate is centralised.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal

from diive.core.utils.console import error


class WorkerRunner(QObject):
    """Runs a callable on a daemon thread and emits its result back to the GUI."""

    #: Emitted with the work function's return value on success.
    done = Signal(object)
    #: Emitted with ``str(exception)`` if the work function raised.
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._running = False

    @property
    def is_running(self) -> bool:
        """True while a job is in flight (between ``run`` and its done/failed)."""
        return self._running

    def run(self, fn, *args, **kwargs) -> bool:
        """Start ``fn(*args, **kwargs)`` on a daemon thread.

        Returns ``False`` (and does nothing) if a job is already running, so the
        caller can use it directly as a re-entry guard.
        """
        if self._running:
            return False
        self._running = True
        threading.Thread(target=self._work, args=(fn, args, kwargs),
                         daemon=True).start()
        return True

    def _work(self, fn, args, kwargs) -> None:
        try:
            result = fn(*args, **kwargs)
        except Exception as err:  # surface the library error to the GUI
            self._running = False
            # The short message goes to the per-tab status label via `failed`;
            # the full type+traceback goes to the diive console so it lands in
            # the Log tab (its sink marshals to the GUI thread via a Qt signal).
            error(f"Background task failed:\n{traceback.format_exc()}")
            self.failed.emit(str(err))
            return
        self._running = False
        self.done.emit(result)
