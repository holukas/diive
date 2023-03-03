import os
import pathlib
from pathlib import Path


def get_filetypes() -> dict:
    """Search files in path and store in dictionary as filename/filepath pairs"""
    filetypes = {}
    path = pathlib.Path(__file__).parent.resolve()  # Search in this file's folder
    for file in os.listdir(path):
        filepath = path / file
        if os.path.isfile(filepath):
            filetypes[filepath.stem] = Path(filepath)
    return filetypes
