"""
I/O: INPUT/OUTPUT UTILITIES
============================

File I/O for parquet, EddyPro, binary extraction, and FLUXNET format conversion.

Part of the diive library: https://github.com/holukas/diive
"""

from importlib import import_module
from typing import TYPE_CHECKING

# Imported on first attribute access (PEP 562), mirroring diive/__init__.py.
# formats.fluxnet imports ManualRemoval, which triggers the whole preprocessing
# tree (outlier detection -> plotting -> bokeh) — ~0.8 s that a caller reaching
# only for diive.io.binary should not pay.
_LAZY_SUBMODULES = frozenset({'binary', 'formats'})

if TYPE_CHECKING:
    from diive.io import binary
    from diive.io import formats


def __getattr__(name: str):
    """Import a submodule on first access. See _LAZY_SUBMODULES."""
    if name in _LAZY_SUBMODULES:
        return import_module(f'diive.io.{name}')
    raise AttributeError(f"module 'diive.io' has no attribute '{name}'")


def __dir__() -> list:
    return sorted(__all__)


__all__ = [
    'binary',
    'formats',
]
