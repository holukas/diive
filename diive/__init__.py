from importlib.metadata import PackageNotFoundError, version

from diive import analysis
from diive import corrections
from diive import variables
from diive import flux
from diive import gapfilling
from diive import outliers
from diive import plotting
from diive import qaqc
from diive import times

from diive.configs.exampledata import load_exampledata_parquet
from diive.configs.exampledata import load_exampledata_parquet_lae
from diive.core.dfun.frames import transform_yearmonth_matrix_to_longform
from diive.core.dfun.stats import sstats
from diive.core.io.filereader import ReadFileType
from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet
from diive.core.io.files import load_parquet_many
from diive.core.io.files import save_parquet
from diive.core.io.files import to_diive_format
from diive.io.binary.extract import get_encoded_value_from_int
from diive.io.binary.extract import get_encoded_value_series

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
    'transform_yearmonth_matrix_to_longform',
    'sstats',
    'ReadFileType',
    'search_files',
    'load_parquet',
    'load_parquet_many',
    'save_parquet',
    'to_diive_format',
    'get_encoded_value_from_int',
    'get_encoded_value_series',
]
