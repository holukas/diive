"""
diive: time series processing for ecosystem research
====================================================

A Python library for eddy covariance and meteorological time series: gap-filling,
flux processing, outlier detection, corrections, QA/QC, analysis and plotting.

https://github.com/holukas/diive
"""
from importlib import import_module
from typing import TYPE_CHECKING

# The namespace submodules are imported on first attribute access (PEP 562).
# Importing them eagerly pulls in sklearn, xgboost, shap and statsmodels, which
# costs ~1.3 s of the ~2.3 s import even for a script that only reads a parquet
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

# The top-level utilities are resolved on first access as well. Imported
# eagerly they cost ~1 s (scipy.stats via sstats, pandas via the example-data
# loaders), which the GUI paid before its splash could show, because
# `diive-gui` imports this package first. Their modules are pinned in
# packaging/diive_gui.spec's hiddenimports for the same reason as above.
_LAZY_ATTRIBUTES = {
    'load_exampledata_parquet': 'diive.configs.exampledata',
    'load_exampledata_parquet_lae': 'diive.configs.exampledata',
    'keep_records_where': 'diive.core.dfun.frames',
    'keep_vars': 'diive.core.dfun.frames',
    'transform_yearmonth_matrix_to_longform': 'diive.core.dfun.frames',
    'sstats': 'diive.core.dfun.stats',
    'ReadFileType': 'diive.core.io.filereader',
    'search_files': 'diive.core.io.filereader',
    'load_parquet': 'diive.core.io.files',
    'load_parquet_many': 'diive.core.io.files',
    'save_parquet': 'diive.core.io.files',
    'to_diive_format': 'diive.core.io.files',
    'get_verbosity': 'diive.core.utils.console',
    'set_verbosity': 'diive.core.utils.console',
    'get_encoded_value_from_int': 'diive.io.binary.extract',
    'get_encoded_value_series': 'diive.io.binary.extract',
}

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

    __version__: str


def _get_version() -> str:
    # importlib.metadata costs ~0.1 s to import, so it is deferred as well.
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("diive")
    except PackageNotFoundError:
        # Package not installed (e.g. running from a source tree without an install).
        # Keep this exact assignment form: .github/workflows/publish.yml reads it.
        __version__ = "0.93.0"
        return __version__


def __getattr__(name: str):
    """Import a namespace submodule or utility on first access.

    See _LAZY_SUBMODULES and _LAZY_ATTRIBUTES.
    """
    if name in _LAZY_SUBMODULES:
        # import_module binds the submodule onto this package, so subsequent
        # attribute lookups find it directly and never reach __getattr__ again.
        return import_module(f'diive.{name}')
    if name in _LAZY_ATTRIBUTES:
        value = getattr(import_module(_LAZY_ATTRIBUTES[name]), name)
    elif name == '__version__':
        value = _get_version()
    else:
        raise AttributeError(f"module 'diive' has no attribute '{name}'")
    # Cache on the module so later lookups skip __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(__all__)


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
