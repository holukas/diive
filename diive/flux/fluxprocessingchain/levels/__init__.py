"""
LEVELS: COMPOSABLE FLUX PROCESSING LEVEL CALLABLES
====================================================

Standalone pure functions, one per level.  Each ``run_level*`` function
accepts a ``FluxLevelData`` container and returns a new one — no shared
mutable state.

Exception — factories: ``make_level32_detector(data)`` returns a tuple
``(data, sod)`` because it may pre-emptively cascade-reset the container
in the L3.2 re-run case; the caller must rebind ``data`` to the returned
value. ``make_level41_engineer(data, features, **kw)`` returns a
``FeatureEngineer`` and does not modify ``data``.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.fluxprocessingchain.levels._init import init_flux_data
from diive.flux.fluxprocessingchain.levels.level2 import (
    VM97_SUBTESTS,
    level2_test_inputs,
    run_level2,
)
from diive.flux.fluxprocessingchain.levels.level31 import (
    level31_storage_col,
    run_level31,
)
from diive.flux.fluxprocessingchain.levels.level32 import make_level32_detector, run_level32
from diive.flux.fluxprocessingchain.levels.level33 import (
    run_level33_constant_ustar,
    run_level33_ustar_detection,
    run_level33_variable_ustar,
)
from diive.flux.fluxprocessingchain.levels.level41 import (
    make_level41_engineer,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)
from diive.flux.fluxprocessingchain.levels.level42 import (
    run_level42_daytime_oneflux,
    run_level42_daytime_reddyproc,
    run_level42_nighttime_oneflux,
    run_level42_nighttime_reddyproc,
)

__all__ = [
    'init_flux_data',
    'run_level2',
    'VM97_SUBTESTS',
    'level2_test_inputs',
    'run_level31',
    'level31_storage_col',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level33_ustar_detection',
    'run_level33_variable_ustar',
    'make_level41_engineer',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
    'run_level42_nighttime_oneflux',
    'run_level42_nighttime_reddyproc',
    'run_level42_daytime_reddyproc',
    'run_level42_daytime_oneflux',
]
