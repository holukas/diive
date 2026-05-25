"""
CONTAINER: STANDARDIZED CONTAINER FOR COMPOSABLE FLUX PROCESSING
=================================================================

Typed data containers passed between the standalone level callables.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from diive.pkgs.flux.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
    from diive.pkgs.flux.fluxprocessingchain.level31_storagecorrection import (
        FluxStorageCorrectionSinglePointEddyPro,
    )
    from diive.pkgs.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds
    from diive.pkgs.preprocessing.outlier_detection import StepwiseOutlierDetection
    from diive.pkgs.preprocessing.qaqc import FlagQCF


@dataclass(frozen=True)
class FluxMeta:
    """Frozen site and processing metadata, shared by all levels."""
    fluxcol: str
    fluxbasevar: str
    ustarcol: str
    swinpot_col: str
    site_lat: float
    site_lon: float
    utc_offset: int
    nighttime_threshold: float
    daytime_accept_qcf_below: int
    nighttimetime_accept_qcf_below: int
    outname: str


@dataclass
class LevelResults:
    """
    Typed bag of per-level outputs accumulated as the chain progresses.

    All fields default to ``None`` / empty so a partial pipeline (e.g. L2 only)
    leaves later fields unset.  Dict-valued fields are keyed by USTAR scenario
    label (e.g. ``'CUT_16'``).
    """

    # Level-2
    level2: FluxQualityFlagsEddyPro | None = None
    level2_qcf: FlagQCF | None = None
    filteredseries_level2_qcf: pd.Series | None = None
    filteredseries_hq: pd.Series | None = None

    # Level-3.1
    level31: FluxStorageCorrectionSinglePointEddyPro | None = None
    flux_corrected_col: str | None = None
    filteredseries_level31_qcf: pd.Series | None = None

    # Level-3.2
    level32: StepwiseOutlierDetection | None = None
    level32_qcf: FlagQCF | None = None
    filteredseries_level32_qcf: pd.Series | None = None

    # Level-3.3
    level33: FlagMultipleConstantUstarThresholds | None = None
    level33_qcf: dict[str, FlagQCF] = field(default_factory=dict)
    filteredseries_level33_qcf: dict[str, pd.Series] = field(default_factory=dict)

    # Level-4.1 — one dict per gap-filling method, keyed by USTAR scenario
    level41_mds: dict[str, Any] = field(default_factory=dict)
    level41_rf: dict[str, Any] = field(default_factory=dict)
    level41_xgb: dict[str, Any] = field(default_factory=dict)

    def has_level41(self) -> bool:
        """True if any L4.1 method has produced results."""
        return bool(self.level41_mds or self.level41_rf or self.level41_xgb)

    def level41_methods(self) -> dict[str, dict[str, Any]]:
        """Return all L4.1 method dicts that have results, keyed by method name."""
        out: dict[str, dict[str, Any]] = {}
        if self.level41_mds:
            out['mds'] = self.level41_mds
        if self.level41_rf:
            out['long_term_random_forest'] = self.level41_rf
        if self.level41_xgb:
            out['long_term_xgboost'] = self.level41_xgb
        return out


@dataclass
class FluxLevelData:
    """
    Container passed between composable level callables.

    Each level returns a new ``FluxLevelData`` with updated fields; the input
    is never mutated.  ``fpc_df`` grows as each level appends its output
    columns; ``filteredseries`` is updated to the QCF-filtered flux from the
    most-recently completed level.
    """

    fpc_df: pd.DataFrame
    full_df: pd.DataFrame
    """The original input DataFrame (with day/night flags + potential
    radiation added).  Required by Level-2, Level-3.1, and Level-4.1 to
    pull meteorological features and apply EddyPro tests."""

    filteredseries: pd.Series | None
    meta: FluxMeta
    levels: LevelResults = field(default_factory=LevelResults)
    level_ids: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        rows, cols = self.fpc_df.shape
        fs = self.filteredseries
        fs_str = f"{fs.name!r} (n={fs.dropna().count()})" if fs is not None else "None"
        return (f"FluxLevelData(flux={self.meta.fluxcol!r}, "
                f"levels={self.level_ids}, "
                f"fpc_df=({rows}x{cols}), filteredseries={fs_str})")
