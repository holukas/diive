"""
Frozen-app entry point for the diive desktop GUI.

PyInstaller bundles this as the executable's startup script (see diive_gui.spec).
It simply calls the same launch path as the ``diive-gui`` console script.

Some tabs compute in a worker process. In the frozen app that process is the
executable itself, started again with multiprocessing's arguments;
``freeze_support()`` turns such a start into the worker and exits, before a
second GUI could open. It must run first, ahead of the diive imports.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from diive.gui import launch
    sys.exit(launch())
