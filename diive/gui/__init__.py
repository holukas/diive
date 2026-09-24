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


def _create_application():
    """The ``QApplication`` with diive's app-wide settings.

    Reuses a running instance. Kept out of `diive.gui.app` so `launch` can
    show the splash before importing the main window (the import is most of
    the startup time).
    """
    import sys

    # Windows groups a Python process under python.exe in the taskbar (and uses
    # its icon) unless the app declares its own AppUserModelID first.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("diive.gui")
        except Exception:
            pass

    from PySide6.QtWidgets import QApplication

    from diive.gui.splash import app_icon
    from diive.gui.widgets.combo import install_combo_popup_fix

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("diive")
    app.setApplicationName("diive-gui")
    # Fusion style honours stylesheet item-selection colours consistently
    # (the native Windows style ignores them in combo-box popups).
    app.setStyle("Fusion")
    # Strip the native frame/shadow from every combo-box popup (black bars on the
    # frameless translucent window) — one app-wide filter covers all dropdowns.
    install_combo_popup_fix(app)
    app.setWindowIcon(app_icon())  # taskbar / window icon (splash motif)
    return app


def launch() -> int:
    """Start the diive desktop GUI. Returns the Qt exit code."""
    # In a frozen app, a worker process (see `widgets/worker.py`) is this
    # executable started again: become that worker and exit instead of opening
    # a second window. A no-op when not frozen. The packaged entry script
    # calls it too, earlier; this covers other frozen entry points.
    import multiprocessing
    multiprocessing.freeze_support()
    _require_pyside6()
    from diive.gui.splash import create_splash, show_message

    # Paint the splash first, then import the main window behind it.
    app = _create_application()
    splash = create_splash(app)
    splash.show()
    show_message(splash, "Starting…")
    app.processEvents()

    from diive.gui.app import run
    return run(app, splash)
