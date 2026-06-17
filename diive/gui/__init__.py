"""
GUI: DESKTOP APPLICATION
========================

PySide6 (Qt) desktop GUI for diive. Provides a multi-tab window for
interactive plotting today, with a registry-based tab system designed so
later additions (e.g. the flux processing chain) slot in as new tabs without
touching the main window.

PySide6 is an OPTIONAL dependency. Install the GUI extra::

    uv sync --extra gui        # or: pip install 'diive[gui]'

then launch with the console script::

    diive-gui

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def _require_pyside6() -> None:
    """Raise a friendly, actionable error if PySide6 is not installed.

    Called by the app bootstrap before any Qt import so headless users who
    never installed the ``gui`` extra get guidance instead of a bare
    ``ModuleNotFoundError``.
    """
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "The diive GUI requires PySide6, which is not installed. "
            "Install the optional GUI dependencies with "
            "`uv sync --extra gui` (or `pip install 'diive[gui]'`)."
        ) from err


def launch() -> int:
    """Start the diive desktop GUI. Returns the Qt exit code."""
    _require_pyside6()
    from diive.gui.app import run
    return run()
