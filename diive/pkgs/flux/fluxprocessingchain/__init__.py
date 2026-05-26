"""
FLUX PROCESSING CHAIN: MULTI-LEVEL WORKFLOW
=============================================

Swiss FluxNet multi-level post-processing: quality flag expansion (L2), storage correction (L3.1),
outlier removal (L3.2), USTAR filtering (L3.3), gap-filling (L4.1).

Two usage patterns:

1. **Composable functions** — one callable per level, each returning a typed
   ``FluxLevelData`` container::

       from diive.pkgs.flux.fluxprocessingchain import (
           init_flux_data, run_level2, run_level31,
           make_level32_detector, run_level32,
           run_level33_constant_ustar,
           run_level41_mds, run_level41_rf, run_level41_xgb,
       )

2. **Orchestrating class** — all levels in sequence via familiar method calls::

       from diive.pkgs.flux.fluxprocessingchain import FluxProcessingChain

Part of the diive library: https://github.com/holukas/diive
"""

from diive.pkgs.flux.fluxprocessingchain.container import (
    FluxConfig,
    FluxLevelData,
    FluxMeta,
    LevelResults,
)
from diive.pkgs.flux.fluxprocessingchain.fluxprocessingchain import FluxProcessingChain
from diive.pkgs.flux.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.flux.fluxprocessingchain.level31_storagecorrection import FluxStorageCorrectionSinglePointEddyPro
from diive.pkgs.flux.fluxprocessingchain.levels import (
    init_flux_data,
    make_level32_detector,
    run_flux_chain,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level41_mds,
    run_level41_rf,
    run_level41_xgb,
)

__all__ = [
    # Containers / config
    'FluxConfig',
    'FluxLevelData',
    'FluxMeta',
    'LevelResults',
    # Composable callables — single flux
    'init_flux_data',
    'run_level2',
    'run_level31',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
    # Multi-flux helper
    'run_flux_chain',
    # Orchestrating class
    'FluxProcessingChain',
    # Level classes (for type-checking downstream)
    'FluxQualityFlagsEddyPro',
    'FluxStorageCorrectionSinglePointEddyPro',
]
