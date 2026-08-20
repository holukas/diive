"""
CONFTEST: shared test-session setup
===================================

Redirects the desktop GUI's persisted preferences to a throwaway file for the
duration of a test session.

Three GUI tests call ``win.close()``, and ``MainWindow.closeEvent`` writes the
real user config -- theme, window geometry, last opened project, per-dataset
variable metadata -- to the ``QStandardPaths`` app-config location. So running
the suite silently overwrites the developer's actual diive-gui preferences with
whatever state a test happened to leave behind. ``save_config`` also writes with
a plain non-atomic ``write_text``, so two pytest processes closing a window at
the same time (pytest-xdist workers, or just two terminals) race on one file.

Pointing ``config_file`` at a tmp path fixes both. It changes nothing any test
asserts: the only test that cares about the config path monkeypatches
``config_file`` itself, and a function-scoped monkeypatch still wins over this.

Also pins matplotlib to the non-interactive Agg backend for the whole session.
PySide6 is installed, so matplotlib's default here is the interactive ``qtagg``,
and only some test modules call ``matplotlib.use("Agg")`` themselves --
``test_analyses``, ``test_heatmap_xyz``, ``test_hexbin_plot`` and
``test_selfheating`` render through whatever an alphabetically earlier module
happened to set. That works by accident in a single serial process and stops
working the moment tests are distributed, because a worker can reach a plotting
module without ever importing the one that set the backend. Setting it here, at
conftest import time (before any test module is imported), makes the intent
explicit instead of order-dependent.

Part of the diive library: https://github.com/holukas/diive
"""
import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True, scope="session")
def _isolate_gui_config(tmp_path_factory):
    """Point the GUI's settings file at a throwaway path for this session."""
    try:
        from diive.gui import config
    except ImportError:
        # The 'gui' extra is not installed, so there is no config to isolate.
        return
    target = tmp_path_factory.mktemp("gui_config") / "gui_settings.json"
    # load_config/save_config resolve config_file through the module global, so
    # rebinding it here reaches both.
    config.config_file = lambda: target
