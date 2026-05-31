"""
DRIVERANALYSIS  (EXPERIMENTAL)
==============================

Evidence-triangulation driver attribution for flux time series, organized by
epistemic level (association → temporal prediction → causation).

EXPERIMENTAL: this module is provisional. Its API (class/method names, the
convergence-table schema, verdict thresholds) may change in a future release
without a deprecation cycle. It is intentionally NOT part of the stable
``dv.analysis`` namespace — reach it via ``dv.analysis.experimental`` (or import
``diive.analysis.driveranalysis`` directly). Instantiating ``DriverAnalysis``
emits a one-time ``ExperimentalWarning``.

Part of the diive library: https://github.com/holukas/diive
"""


class ExperimentalWarning(UserWarning):
    """Raised once when a provisional (pre-release) diive feature is first used.

    Subclasses ``UserWarning`` so it shows by default but can be silenced with
    ``warnings.filterwarnings('ignore', category=ExperimentalWarning)``.
    """


from diive.analysis.driveranalysis.ale import (
    AleCurve,
    Ale2DResult,
    accumulated_local_effects,
    accumulated_local_effects_2d,
)
from diive.analysis.driveranalysis.driveranalysis import (
    DriverAnalysis,
    DriverAnalysisResult,
)

__all__ = [
    'DriverAnalysis',
    'DriverAnalysisResult',
    'AleCurve',
    'Ale2DResult',
    'accumulated_local_effects',
    'accumulated_local_effects_2d',
    'ExperimentalWarning',
]
