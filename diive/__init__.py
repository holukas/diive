"""
diive: time series processing for ecosystem research
====================================================

A Python library for eddy covariance and meteorological time series: gap-filling,
flux processing, outlier detection, corrections, QA/QC, analysis and plotting.

https://github.com/holukas/diive
"""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from diive.configs.exampledata import load_exampledata_parquet
from diive.configs.exampledata import load_exampledata_parquet_lae
from diive.core.dfun.frames import keep_records_where
from diive.core.dfun.frames import keep_vars
from diive.core.dfun.frames import transform_yearmonth_matrix_to_longform
from diive.core.dfun.stats import sstats
from diive.core.io.filereader import ReadFileType
from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet
from diive.core.io.files import load_parquet_many
from diive.core.io.files import save_parquet
from diive.core.io.files import to_diive_format
from diive.core.utils.console import get_verbosity
from diive.core.utils.console import set_verbosity
from diive.io.binary.extract import get_encoded_value_from_int
from diive.io.binary.extract import get_encoded_value_series

# The namespace submodules are imported on first attribute access (PEP 562).
# Importing them eagerly pulls in sklearn, xgboost, shap and statsmodels, which
# costs ~1.7 s of the ~2.3 s import even for a script that only reads a parquet
# file. Note for packaging: a frozen build cannot see these edges statically —
# they are pinned in packaging/diive_gui.spec via hiddenimports.
_LAZY_SUBMODULES = frozenset({
    'analysis',
    'corrections',
    'events',
    'flux',
    'gapfilling',
    'outliers',
    'plotting',
    'qaqc',
    'times',
    'variables',
})

if TYPE_CHECKING:
    # Static analysers and IDEs do not evaluate __getattr__, so give them the
    # real imports. This block never runs.
    from diive import analysis
    from diive import corrections
    from diive import events
    from diive import flux
    from diive import gapfilling
    from diive import outliers
    from diive import plotting
    from diive import qaqc
    from diive import times
    from diive import variables


def __getattr__(name: str):
    """Import a namespace submodule on first access. See _LAZY_SUBMODULES."""
    if name in _LAZY_SUBMODULES:
        # import_module binds the submodule onto this package, so subsequent
        # attribute lookups find it directly and never reach __getattr__ again.
        return import_module(f'diive.{name}')
    raise AttributeError(f"module 'diive' has no attribute '{name}'")


def __dir__() -> list:
    return sorted(__all__)


try:
    __version__ = version("diive")
except PackageNotFoundError:
    # Package not installed (e.g. running from a source tree without an install).
    __version__ = "0.91.0"

__all__ = [
    '__version__',
    # Namespace submodules
    'analysis',
    'corrections',
    'events',
    'variables',
    'flux',
    'gapfilling',
    'outliers',
    'plotting',
    'qaqc',
    'times',
    # Top-level utilities
    'load_exampledata_parquet',
    'load_exampledata_parquet_lae',
    'keep_vars',
    'keep_records_where',
    'transform_yearmonth_matrix_to_longform',
    'sstats',
    'ReadFileType',
    'search_files',
    'load_parquet',
    'load_parquet_many',
    'save_parquet',
    'to_diive_format',
    'set_verbosity',
    'get_verbosity',
    'get_encoded_value_from_int',
    'get_encoded_value_series',
]
