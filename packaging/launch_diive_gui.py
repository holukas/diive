"""
Frozen-app entry point for the diive desktop GUI.

PyInstaller bundles this as the executable's startup script (see diive_gui.spec).
It simply calls the same launch path as the ``diive-gui`` console script.
"""
import sys

from diive.gui import launch

if __name__ == "__main__":
    sys.exit(launch())
