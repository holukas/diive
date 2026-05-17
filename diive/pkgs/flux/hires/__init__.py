"""
FLUX - HIGH-RESOLUTION ANALYSIS
================================

20 Hz analysis: detection limit, time lag, wind rotation for eddy covariance flux.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.pkgs.flux.hires.fluxdetectionlimit import FluxDetectionLimit
from diive.pkgs.flux.hires.lag import MaxCovariance
from diive.pkgs.flux.hires.lag_pwb import PreWhiteningBootstrap, PwboptLagPlot
from diive.pkgs.flux.hires.windrotation import WindRotation2D

__all__ = [
    'FluxDetectionLimit',
    'MaxCovariance',
    'PreWhiteningBootstrap',
    'PwboptLagPlot',
    'WindRotation2D',
]
