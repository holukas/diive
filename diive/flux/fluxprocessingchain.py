"""
FLUXPROCESSINGCHAIN: DEPRECATED ALIAS OF POSTPROCESSINGCHAIN
============================================================

Old import path of :mod:`diive.flux.postprocessingchain`, kept so existing scripts keep working.

Part of the diive library: https://github.com/holukas/diive
"""

import importlib
import pkgutil
import sys
import warnings

from diive.flux import postprocessingchain as _new

warnings.warn(
    "diive.flux.fluxprocessingchain was renamed to diive.flux.postprocessingchain. "
    "The old name still works but will be removed in a future release.",
    FutureWarning, stacklevel=2)

# Register every submodule under the old name as well. Without this, an import
# such as `diive.flux.fluxprocessingchain.levels` would load the file a second
# time as a separate module, and its classes would fail isinstance checks
# against the ones reached through the new name.
for _info in pkgutil.walk_packages(_new.__path__, _new.__name__ + '.'):
    sys.modules[__name__ + _info.name[len(_new.__name__):]] = importlib.import_module(_info.name)

sys.modules[__name__] = _new
