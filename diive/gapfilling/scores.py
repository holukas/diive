"""
GAP-FILLING: MODEL SCORING (COMPATIBILITY RE-EXPORT)
=====================================================

``prediction_scores`` now lives in :mod:`diive.core.ml.scores`. It is used by
``core.ml.common`` and ``core.ml.optimization``, which sit below this package —
importing it from here made ``core.ml`` depend on ``diive.gapfilling``, a cycle
that only resolved because ``diive/__init__`` happened to import gapfilling
first. This module is kept so existing ``diive.gapfilling.scores`` imports work.

Part of the diive library: https://github.com/holukas/diive
"""

from diive.core.ml.scores import prediction_scores

__all__ = [
    'prediction_scores',
]
