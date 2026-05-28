"""
LEVELS: COMPOSABLE FLUX PROCESSING LEVEL CALLABLES
====================================================

Standalone pure functions, one per level.  Each function accepts a
``FluxLevelData`` container and returns a new one — no shared mutable state.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.fluxprocessingchain.levels._init import init_flux_data
from diive.flux.fluxprocessingchain.levels.level2 import run_level2
from diive.flux.fluxprocessingchain.levels.level31 import run_level31
from diive.flux.fluxprocessingchain.levels.level32 import make_level32_detector, run_level32
from diive.flux.fluxprocessingchain.levels.level33 import (
    run_level33_constant_ustar,
    run_level33_ustar_detection,
)
from diive.flux.fluxprocessingchain.levels.level41 import (
    make_level41_engineer,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)

__all__ = [
    'init_flux_data',
    'run_level2',
    'run_level31',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level33_ustar_detection',
    'make_level41_engineer',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
]
