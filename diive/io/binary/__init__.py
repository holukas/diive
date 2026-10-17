"""
BINARY: BINARY FILE UTILITIES
==============================

Extract values that are encoded in a subrange of the bits of an integer.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.io.binary.extract import get_encoded_value_from_int
from diive.io.binary.extract import get_encoded_value_series

__all__ = [
    'get_encoded_value_from_int',
    'get_encoded_value_series',
]
