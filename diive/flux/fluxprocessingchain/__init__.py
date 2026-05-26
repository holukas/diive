"""
FLUX PROCESSING CHAIN: MULTI-LEVEL WORKFLOW
=============================================

Swiss FluxNet multi-level post-processing: quality flag expansion (L2), storage correction (L3.1),
outlier removal (L3.2), USTAR filtering (L3.3), gap-filling (L4.1).

Composable functions — one callable per level, each returning a typed ``FluxLevelData`` container::

    from diive.flux.fluxprocessingchain import (
        init_flux_data, run_level2, run_level31,
        make_level32_detector, run_level32,
        run_level33_constant_ustar,
        run_level41_mds, run_level41_rf, run_level41_xgb,
    )

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.fluxprocessingchain.container import (
    FluxConfig,
    FluxLevelData,
    FluxMeta,
    LevelResults,
)
from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
from diive.flux.lowres.storage_correction import FluxStorageCorrectionSinglePointEddyPro
from diive.flux.fluxprocessingchain.levels import (
    init_flux_data,
    make_level32_detector,
    run_level2,
    run_level31,
    run_level32,
    run_level33_constant_ustar,
    run_level33_ustar_detection,
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
    # Composable callables
    'init_flux_data',
    'run_level2',
    'run_level31',
    'make_level32_detector',
    'run_level32',
    'run_level33_constant_ustar',
    'run_level33_ustar_detection',
    'run_level41_mds',
    'run_level41_rf',
    'run_level41_xgb',
    # Level classes (for type-checking downstream)
    'FluxQualityFlagsEddyPro',
    'FluxStorageCorrectionSinglePointEddyPro',
]
