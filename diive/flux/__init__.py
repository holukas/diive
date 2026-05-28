"""
FLUX: EDDY COVARIANCE PROCESSING
=================================

High-resolution and low-resolution flux analysis, time lag detection, wind rotation, USTAR filtering.
Complete Swiss FluxNet processing chain for L2-L4.1 levels.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux import hires
from diive.flux import lowres
from diive.flux import fluxprocessingchain
from diive.flux.fluxprocessingchain import (
    FluxConfig,
    FluxLevelData,
    add_driver,
    init_flux_data,
    run_chain,
)
from diive.flux.hires.fluxdetectionlimit import FluxDetectionLimit
from diive.flux.hires.lag import MaxCovariance
from diive.flux.hires.lag_pwb import PreWhiteningBootstrap
from diive.flux.hires.lag_pwb import PwbBatchDetection
from diive.flux.hires.lag_pwb import PwboptLagPlot
from diive.flux.hires.windrotation import WindDoubleRotation
from diive.flux.hires.windrotation import reynolds_decomposition
from diive.flux.lowres.timelag_analysis import TimeLagAnalysis
from diive.flux.lowres.uncertainty import RandomUncertaintyPAS20
from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
from diive.flux.lowres.ustar_mp_detection import UstarMovingPointDetection
from diive.flux.lowres.ustar_vekuri_detection import UstarVekuriThresholdDetection
from diive.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds
from diive.flux.lowres.ustarthreshold import FlagSingleConstantUstarThreshold
from diive.flux.lowres.ustarthreshold import UstarDetectionMPT
from diive.flux.lowres.ustarthreshold import UstarThresholdConstantScenarios

__all__ = [
    'hires',
    'lowres',
    'fluxprocessingchain',
    'FluxConfig',
    'FluxLevelData',
    'add_driver',
    'init_flux_data',
    'run_chain',
    'FluxDetectionLimit',
    'MaxCovariance',
    'PreWhiteningBootstrap',
    'PwbBatchDetection',
    'PwboptLagPlot',
    'WindDoubleRotation',
    'reynolds_decomposition',
    'TimeLagAnalysis',
    'RandomUncertaintyPAS20',
    'UstarBootstrapThresholds',
    'UstarMovingPointDetection',
    'UstarVekuriThresholdDetection',
    'FlagMultipleConstantUstarThresholds',
    'FlagSingleConstantUstarThreshold',
    'UstarDetectionMPT',
    'UstarThresholdConstantScenarios',
]
