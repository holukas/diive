"""
GUI.TABS.STEPWISE: STEPWISE OUTLIER SCREENING TAB
=================================================

Chain several outlier tests on one variable of the working dataset as a list of
editable **method cards**, apply corrections, and inspect what each step removes
plus the overall **QCF**. This is the plain (no-resampling) variant of the shared
screening experience — all the machinery lives in
:class:`~diive.gui.tabs._screening_base.ScreeningTabBase`; the database variant
(:mod:`diive.gui.tabs.meteo_screening`) adds resampling.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from diive.gui.tabs._screening_base import ScreeningTabBase


class StepwiseScreeningTab(ScreeningTabBase):
    """Chain outlier tests + corrections + QCF on a working-dataset variable.

    Uses the base behaviour unchanged: the variable list is the working dataset's
    columns, and 'Add' emits the per-test flags, the QCF flag, the QCF-filtered
    series, and (if any) the corrected series.
    """

    title = "Stepwise screening"
