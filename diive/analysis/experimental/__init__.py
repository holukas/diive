"""
ANALYSIS: EXPERIMENTAL
======================

Provisional analysis tools that are part of the repository but NOT yet part of
the stable public API. Anything exposed here may change (or be removed) in a
future release without a deprecation cycle. Use in production with that caveat.

Currently provides:
    - DriverAnalysis / DriverAnalysisResult — evidence-triangulation driver
      attribution organized by epistemic level (association -> temporal -> causal).
    - AleCurve / Ale2DResult + accumulated_local_effects[_2d] — dependency-free
      Accumulated Local Effects.

Access via the ``experimental`` subnamespace::

    import diive as dv
    da = dv.analysis.experimental.DriverAnalysis(target=..., drivers=...)

The implementation lives at ``diive/analysis/driveranalysis/``; this module only
re-exports it under the experimental namespace.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.analysis.driveranalysis import (
    DriverAnalysis,
    DriverAnalysisResult,
    AleCurve,
    Ale2DResult,
    accumulated_local_effects,
    accumulated_local_effects_2d,
    ExperimentalWarning,
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
