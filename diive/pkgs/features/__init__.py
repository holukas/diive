"""
FEATURES: FEATURE ENGINEERING
=============================

Calculate derived variables: VPD, unit conversions, day/night flags, lag features, potential radiation.
Composable 8-stage pipeline for time series feature creation.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.pkgs.features import variables

__all__ = [
    'variables',
]
