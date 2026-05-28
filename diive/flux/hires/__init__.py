"""
FLUX - HIGH-RESOLUTION ANALYSIS
================================

20 Hz analysis: detection limit, time lag, wind rotation for eddy covariance flux.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.flux.hires.fluxdetectionlimit import FluxDetectionLimit
from diive.flux.hires.lag import MaxCovariance
from diive.flux.hires.lag_pwb import PreWhiteningBootstrap, PwboptLagPlot
from diive.flux.hires.windrotation import WindDoubleRotation, reynolds_decomposition

__all__ = [
    'FluxDetectionLimit',
    'MaxCovariance',
    'PreWhiteningBootstrap',
    'PwboptLagPlot',
    'WindDoubleRotation',
    'reynolds_decomposition',
]
