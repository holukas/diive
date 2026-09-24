"""
GUI._CLI: CONSOLE-SCRIPT ENTRYPOINT
===================================

Backs the ``diive-gui`` console script declared in pyproject.toml.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import sys


def _gui_main() -> None:
    import argparse
    import multiprocessing

    # Before argparse: in a frozen build a worker process arrives here with
    # multiprocessing's own arguments, which the parser would reject.
    multiprocessing.freeze_support()

    p = argparse.ArgumentParser(
        prog="diive-gui",
        description="Launch the diive desktop GUI (requires the 'gui' extra).",
    )
    p.parse_args()

    from diive.gui import launch
    sys.exit(launch())


if __name__ == "__main__":
    _gui_main()
