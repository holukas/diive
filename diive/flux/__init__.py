"""
FLUX: EDDY COVARIANCE PROCESSING
=================================

Low-resolution flux analysis, USTAR filtering, NEE partitioning and uncertainty.
High-frequency raw-data tooling moved to the dyco package: https://github.com/holukas/dyco
Swiss FluxNet post-processing chain for levels L2-L4.2.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux import lowres
from diive.flux import postprocessingchain
from diive.flux import partitioning
from diive.flux.partitioning import NighttimePartitioningOneFlux
from diive.flux.partitioning import partition_nee_nighttime_oneflux
from diive.flux.partitioning import NighttimePartitioningReddyProc
from diive.flux.partitioning import partition_nee_nighttime_reddyproc
from diive.flux.partitioning import DaytimePartitioningReddyProc
from diive.flux.partitioning import partition_nee_daytime_reddyproc
from diive.flux.partitioning import DaytimePartitioningOneFlux
from diive.flux.partitioning import partition_nee_daytime_oneflux
from diive.flux.partitioning import partition_nee_daytime_oneflux_scenarios
from diive.flux.partitioning import PartitioningNEE
from diive.flux.partitioning import partition_nee
from diive.flux.postprocessingchain import (
    FluxConfig,
    FluxLevelData,
    add_driver,
    init_flux_data,
    run_chain,
)
from diive.flux.lowres.timelag_analysis import TimeLagAnalysis
from diive.flux.lowres.uncertainty import JointUncertaintyPAS20
from diive.flux.lowres.uncertainty import RandomUncertaintyPAS20
from diive.flux.lowres.uncertainty import joint_uncertainty_pas20
from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
from diive.flux.lowres.ustar_cp_detection import UstarChangePointDetection
from diive.flux.lowres.ustar_detection import USTAR_MODES
from diive.flux.lowres.ustar_detection import UstarThresholdDetection
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
from diive.flux.lowres.ustar_reference import ONEFLUX_REF_PERCENTILES
from diive.flux.lowres.ustar_reference import PartitioningReferenceResult
from diive.flux.lowres.ustar_reference import UstarReferenceResult
from diive.flux.lowres.ustar_reference import oneflux_ure_cleanup
from diive.flux.lowres.ustar_reference import select_partitioning_reference
from diive.flux.lowres.ustar_reference import select_ustar_reference
from diive.flux.lowres.ustar_reddyproc_detection import UstarReddyProcDetection
from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
from diive.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds
from diive.flux.lowres.ustarthreshold import FlagMultipleVariableUstarThresholds
from diive.flux.lowres.ustarthreshold import FlagSingleConstantUstarThreshold
from diive.flux.lowres.ustarthreshold import UstarThresholdConstantScenarios

__all__ = [
    'lowres',
    'postprocessingchain',
    'partitioning',
    'NighttimePartitioningOneFlux',
    'partition_nee_nighttime_oneflux',
    'NighttimePartitioningReddyProc',
    'partition_nee_nighttime_reddyproc',
    'DaytimePartitioningReddyProc',
    'partition_nee_daytime_reddyproc',
    'DaytimePartitioningOneFlux',
    'partition_nee_daytime_oneflux',
    'partition_nee_daytime_oneflux_scenarios',
    'PartitioningNEE',
    'partition_nee',
    'FluxConfig',
    'FluxLevelData',
    'add_driver',
    'init_flux_data',
    'run_chain',
    'TimeLagAnalysis',
    'RandomUncertaintyPAS20',
    'JointUncertaintyPAS20',
    'joint_uncertainty_pas20',
    'UstarBootstrapThresholds',
    'UstarThresholdDetection',
    'USTAR_MODES',
    'UstarChangePointDetection',
    'UstarReddyProcDetection',
    'UstarMovingPointDetection',
    'ONEFLUX_REF_PERCENTILES',
    'UstarReferenceResult',
    'select_ustar_reference',
    'PartitioningReferenceResult',
    'select_partitioning_reference',
    'oneflux_ure_cleanup',
    'UstarVekuriThresholdDetection',
    'FlagMultipleConstantUstarThresholds',
    'FlagMultipleVariableUstarThresholds',
    'FlagSingleConstantUstarThreshold',
    'UstarThresholdConstantScenarios',
]


def __getattr__(name):
    # Old name of postprocessingchain; importing the alias module emits the FutureWarning.
    if name == 'fluxprocessingchain':
        import importlib
        return importlib.import_module('diive.flux.fluxprocessingchain')
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
