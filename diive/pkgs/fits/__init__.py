"""
FITS: CURVE FITTING AND REGRESSION
===================================

Polynomial and custom function fitting for binned data.
Includes binning aggregation and uncertainty estimation.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.pkgs.fits.fitter import BinFitterCP

__all__ = [
    'BinFitterCP',
]
