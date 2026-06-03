"""
FLUX_DETECT_REMOVE_TUI_DEMO: preview the detect+remove Textual TUI
==================================================================

Launches the ``diive-tlag-pwb-detect-remove`` terminal UI in **demo mode**, so
you can see the interface without any input data. The demo runs a synthetic
two-phase pipeline (detect -> remove) that animates the progress bar and
streams Rich-styled per-chunk lines into the log panel — no files are read or
written.

This is an INTERACTIVE example (it takes over the terminal); it is therefore
NOT part of ``examples/run_all_examples.py``. Run it directly::

    uv run python examples/flux/hires/flux_detect_remove_tui_demo.py

For a real run on actual data, launch without --demo and fill in the form::

    uv run diive-tlag-pwb-detect-remove-tui

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.hires.detect_and_remove_tlag_tui import DetectRemoveTUI

if __name__ == '__main__':
    DetectRemoveTUI(demo=True).run()
