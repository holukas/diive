"""
QAQC: QUALITY CONTROL
=====================

Public namespace ``dv.qaqc``: overall quality flag, meteo screening and the measurement and correction registry.

Part of the diive library: https://github.com/holukas/diive
"""
from diive.preprocessing.qaqc import FlagQCF
from diive.preprocessing.qaqc import StepwiseMeteoScreeningDb
from diive.preprocessing.qaqc import (
    Measurement,
    MEASUREMENTS,
    CorrectionSpec,
    CORRECTIONS,
    corrections_for_measurement,
    correction_spec,
    detect_measurement,
    measurement_label,
)

__all__ = [
    'FlagQCF',
    'StepwiseMeteoScreeningDb',
    'Measurement',
    'MEASUREMENTS',
    'CorrectionSpec',
    'CORRECTIONS',
    'corrections_for_measurement',
    'correction_spec',
    'detect_measurement',
    'measurement_label',
]
