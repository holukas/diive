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


@dataclass
class FluxConfig:
    """
    Per-flux configuration for :func:`run_flux_chain`.

    Captures every setting that differs between fluxes (CO2, H, LE, N2O, CH4, …)
    so that a multi-flux loop can call ``run_flux_chain(df, config, **site)`` for
    each variable without repeating site-level parameters.

    **Required fields** (no defaults) force you to make explicit choices for every
    flux rather than silently inheriting wrong values from another variable.

    Example — NEE, H, and N2O configs::

        from diive.pkgs.flux.fluxprocessingchain import FluxConfig

        fc_cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.30],
            ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5,
            outlier_sigma_nighttime=5.5,
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
            level2_tests={
                'ssitc': {'apply': True, 'setflag_timeperiod': None},
                'gas_completeness': {'apply': True},
                'spectral_correction_factor': {'apply': True},
            },
            mds_swin='SW_IN_1_1_1',
            mds_ta='TA_1_1_1',
            mds_vpd='VPD_kPa_1_1_1',
        )

        h_cfg = FluxConfig(
            fluxcol='H',
            ustar_thresholds=[0.0],   # energy flux — no USTAR filtering
            ustar_labels=['CUT_NONE'],
            outlier_sigma_daytime=5.5,
            outlier_sigma_nighttime=5.5,
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1'],
            level2_tests={
                'ssitc': {'apply': True, 'setflag_timeperiod': None},
            },
            mds_swin='SW_IN_1_1_1',
            mds_ta='TA_1_1_1',
            mds_vpd='VPD_kPa_1_1_1',
        )

        n2o_cfg = FluxConfig(
            fluxcol='N2O',
            ustar_thresholds=[0.30],
            ustar_labels=['CUT_50'],
            outlier_sigma_daytime=4.0,    # chosen by inspecting the N2O record
            outlier_sigma_nighttime=3.5,  # trace gases typically need lower sigma
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1'],
            level2_tests={
                'ssitc': {'apply': True, 'setflag_timeperiod': None},
                'gas_completeness': {'apply': True},
            },
            gapfill_mds=False,  # MDS not appropriate for N2O without dedicated drivers
        )
    """

    # ------------------------------------------------------------------ required
    fluxcol: str
    """Target flux column (e.g. ``'FC'``, ``'H'``, ``'N2O'``)."""

    ustar_thresholds: list
    """USTAR threshold(s) in m s-1 — one per scenario.
    Use ``[0.0]`` for energy fluxes (H, LE) to keep all records."""

    ustar_labels: list
    """Short label per threshold (e.g. ``['CUT_50']``; ``['CUT_NONE']`` for H/LE)."""

    outlier_sigma_daytime: float
    """Hampel filter sigma for daytime outlier detection.
    Must be chosen by inspecting the flux record — no universal default applies.
    Typical range: 5–6 for CO2/H/LE; 3–5 for trace gases (N2O, CH4)."""

    outlier_sigma_nighttime: float
    """Hampel filter sigma for nighttime outlier detection.
    Must be chosen by inspecting the flux record — no universal default applies."""

    gapfilling_features: list
    """Predictor column names for ML gap-filling (RF, XGBoost).
    Must exist in ``data.full_df``.  Ignored when both ``gapfill_rf`` and
    ``gapfill_xgb`` are ``False``."""

    level2_tests: dict
    """Keyword arguments forwarded verbatim to :func:`run_level2`
    (``ssitc``, ``gas_completeness``, ``spectral_correction_factor``,
    ``signal_strength``, ``raw_data_screening_vm97``,
    ``angle_of_attack``, ``steadiness_of_horizontal_wind``)."""

    # ------------------------------------------------------------------ optional
    set_storage_to_zero: bool = False
    """Set to ``True`` when no storage measurement is available for this flux.
    The storage correction (L3.1) still runs; it just adds zero."""

    outlier_window_length: int = 48 * 13
    """Hampel filter rolling window in half-hours (default = 13 days = 624 records)."""

    gapfill_rf: bool = True
    """Run Random Forest gap-filling (L4.1)."""

    gapfill_xgb: bool = False
    """Run XGBoost gap-filling (L4.1).  Off by default — slower than RF."""

    gapfill_mds: bool = True
    """Run MDS gap-filling (L4.1)."""

    mds_swin: str | None = None
    """Shortwave incoming radiation column for MDS (W m-2; must be in ``data.full_df``)."""

    mds_ta: str | None = None
    """Air temperature column for MDS (deg C; must be in ``data.full_df``)."""

    mds_vpd: str | None = None
    """VPD column for MDS (**kPa**; must be in ``data.full_df``).
    EddyPro outputs VPD in hPa — divide by 10 before assigning here."""


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
    nighttime_accept_qcf_below: int
    outname: str


@dataclass
class LevelResults:
    """
    Typed bag of per-level outputs accumulated as the chain progresses.

    All fields default to ``None`` / empty so a partial pipeline (e.g. L2 only)
    leaves later fields unset.

    **High-quality vs. accepted-quality series:**

    - ``filteredseries_hq`` — flux with QCF=0 *only* (strictest filter, no
      soft warnings tolerated).  Used as the reference series for gap-filling
      model training.  Updated by each level that runs a QCF.
    - ``filteredseries_level*_qcf`` — flux accepted at QCF < threshold (set by
      ``daytime_accept_qcf_below`` / ``nighttime_accept_qcf_below`` in
      ``init_flux_data``).  The threshold is usually 1 or 2 depending on the
      level and site protocol.

    **USTAR scenario dicts:** ``level33_qcf``, ``filteredseries_level33_qcf``,
    ``filteredseries_level33_hq``, ``level41_*`` are keyed by the scenario labels
    supplied to ``run_level33_constant_ustar()`` (e.g. ``'CUT_16'``, ``'CUT_50'``,
    ``'CUT_84'`` for the 16th, 50th, and 84th percentiles of a bootstrap USTAR
    threshold distribution).

    ``filteredseries_level33_hq`` holds the QCF=0-only (strictest quality) series
    per USTAR scenario — the analogue of ``filteredseries_hq`` at the L3.3 level.
    """

    # Level-2
    level2: FluxQualityFlagsEddyPro | None = None
    level2_qcf: FlagQCF | None = None
    filteredseries_level2_qcf: pd.Series | None = None
    filteredseries_hq: pd.Series | None = None  # QCF=0 only; updated by each level

    # Level-3.1
    level31: FluxStorageCorrectionSinglePointEddyPro | None = None
    flux_corrected_col: str | None = None
    # Note: Level-3.1 has no quality test of its own.  filteredseries_level31_qcf
    # is the Level-2 QCF re-applied to the storage-corrected flux — not a new
    # L3.1-specific filter.  Record counts therefore match Level-2 unless some
    # storage-corrected values happened to become NaN.
    filteredseries_level31_qcf: pd.Series | None = None

    # Level-3.2
    level32: StepwiseOutlierDetection | None = None
    level32_qcf: FlagQCF | None = None
    filteredseries_level32_qcf: pd.Series | None = None

    # Level-3.3
    level33: FlagMultipleConstantUstarThresholds | None = None
    level33_qcf: dict[str, FlagQCF] = field(default_factory=dict)
    filteredseries_level33_qcf: dict[str, pd.Series] = field(default_factory=dict)
    filteredseries_level33_hq: dict[str, pd.Series] = field(default_factory=dict)

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

    Each level callable returns a *new* ``FluxLevelData``; the input is never
    mutated.

    **Two DataFrames — which one to use?**

    - ``fpc_df`` — the *processing* dataframe.  Starts with just the flux and
      USTAR columns, then grows as each level appends its flag, QCF, and
      gap-filled columns.  Use this for inspecting chain outputs, exporting
      results, or plotting quality-controlled fluxes.
    - ``full_df`` — the *full input* dataframe (original EddyPro columns plus
      potential radiation and day/night flags added by ``init_flux_data``).
      Level-2 reads EddyPro diagnostic columns from here; Level-4.1 reads
      meteorological driver columns (e.g. ``TA_1_1_1``, ``SW_IN``) from here
      as gap-filling features.  Do not modify this dataframe — level callables
      use it read-only.
    """

    fpc_df: pd.DataFrame
    full_df: pd.DataFrame
    filteredseries: pd.Series | None
    """QCF-filtered flux from the most recently completed level (accepted-quality
    threshold, not QCF=0-only).  ``None`` before any level has run."""
    meta: FluxMeta
    levels: LevelResults = field(default_factory=LevelResults)
    level_ids: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        rows, cols = self.fpc_df.shape
        fs = self.filteredseries
        if fs is not None:
            n_valid = int(fs.dropna().count())
            n_total = len(fs)
            fs_str = f"{fs.name!r} ({n_valid}/{n_total} valid)"
        else:
            fs_str = "None (use data.levels.filteredseries_level33_qcf after L3.3)"
        all_cols = self.fpc_df.columns.tolist()
        _max_shown = 12
        if len(all_cols) <= _max_shown:
            col_str = ", ".join(all_cols)
        else:
            col_str = ", ".join(all_cols[:_max_shown]) + f", ... (+{len(all_cols) - _max_shown} more)"
        return (
            f"FluxLevelData(\n"
            f"  flux        = {self.meta.fluxcol!r}\n"
            f"  levels run  = {self.level_ids}\n"
            f"  fpc_df      = {rows} rows x {cols} cols\n"
            f"  columns     = [{col_str}]\n"
            f"  filtered    = {fs_str}\n"
            f")"
        )

    def summary(self) -> str:
        """
        Return a concise data-availability summary across all completed levels.

        Shows the number of valid (non-NaN) records per QCF-filtered series,
        split by daytime and nighttime where available.  Useful for quickly
        assessing how much data each level removes.

        Returns:
            Multi-line string ready for ``print()``.
        """
        m = self.meta
        lines = [
            f"Data availability summary  —  flux: {m.fluxcol!r}",
            f"Site: lat={m.site_lat}, lon={m.site_lon}, UTC+{m.utc_offset}",
            f"QCF thresholds: daytime accept QCF < {m.daytime_accept_qcf_below}  |  "
            f"nighttime accept QCF < {m.nighttime_accept_qcf_below}",
            f"  (QCF=0: all tests pass; QCF=1: soft warnings; QCF=2: hard failure)",
            f"Total records: {len(self.fpc_df)}",
        ]

        # Try to pull daytime / nighttime masks from fpc_df
        daytime_col = 'DAYTIME'
        nighttime_col = 'NIGHTTIME'
        has_dn = daytime_col in self.fpc_df.columns and nighttime_col in self.fpc_df.columns
        if has_dn:
            day_mask = self.fpc_df[daytime_col] == 1
            night_mask = self.fpc_df[nighttime_col] == 1
            lines.append(f"  Daytime records:    {int(day_mask.sum())}")
            lines.append(f"  Nighttime records:  {int(night_mask.sum())}")
        lines.append("")

        def _row(label: str, s: pd.Series | None) -> str:
            if s is None:
                return f"  {label:<32s}  not yet run"
            n = int(s.dropna().count())
            pct = 100 * n / max(len(s), 1)
            if has_dn:
                nd = int(s[day_mask].dropna().count())
                nn = int(s[night_mask].dropna().count())
                return (f"  {label:<32s}  {n:5d} valid ({pct:5.1f}%)"
                        f"  |  day: {nd:5d}  night: {nn:5d}")
            return f"  {label:<32s}  {n:5d} valid ({pct:5.1f}%)"

        lvl = self.levels
        lines.append(_row("L2  (after QC flags)", lvl.filteredseries_level2_qcf))
        lines.append(_row("L2  (QCF=0 only, high-quality)", lvl.filteredseries_hq))
        lines.append(_row("L3.1 (storage-corrected)", lvl.filteredseries_level31_qcf))
        lines.append(_row("L3.2 (outlier-cleaned)", lvl.filteredseries_level32_qcf))
        for scen, s in lvl.filteredseries_level33_qcf.items():
            lines.append(_row(f"L3.3 USTAR={scen}", s))
            if scen in lvl.filteredseries_level33_hq:
                lines.append(_row(f"L3.3 USTAR={scen} (QCF=0 only)", lvl.filteredseries_level33_hq[scen]))

        # L4.1 gap-filling stats
        if lvl.has_level41():
            lines.append("")
            lines.append("Gap-filling (L4.1)  [measured / gap-filled / fallback]:")
            for method, col in self.gapfilled_cols().items():
                for scen, gf_col in col.items():
                    flag_col = f"FLAG_{gf_col}_ISFILLED"
                    if flag_col in self.fpc_df.columns:
                        flags = self.fpc_df[flag_col]
                        n_measured = int((flags == 0).sum())
                        n_filled = int((flags == 1).sum())
                        n_fallback = int((flags == 2).sum())
                        n_total = n_measured + n_filled + n_fallback
                        pct = 100 * n_filled / max(n_total, 1)
                        lines.append(
                            f"  {method} {scen:<10s}  total: {n_total:5d}  |  "
                            f"measured: {n_measured:5d}  |  gap-filled: {n_filled:5d} ({pct:.1f}%)"
                            + (f"  |  fallback: {n_fallback}" if n_fallback else "")
                        )

        lines.append("")
        lines.append(f"fpc_df columns ({len(self.fpc_df.columns)}): "
                     + ", ".join(self.fpc_df.columns.tolist()))
        return "\n".join(lines)

    def gapfilled_cols(self) -> dict[str, dict[str, str]]:
        """
        Return the gap-filled output column names per L4.1 method and USTAR scenario.

        Saves the user from digging into the level instances to find which column
        in ``fpc_df`` holds the gap-filled flux.

        Returns:
            Nested dict ``{method: {ustar_scenario: column_name}}``.
            Keys present only when that method has been run.

        Example::

            cols = data.gapfilled_cols()
            # {'rf': {'CUT_50': 'NEE_L3.3_CUT_50_QCF_f'},
            #  'mds': {'CUT_50': 'NEE_L3.3_CUT_50_QCF_MDS_f'}}

            gapfilled = data.fpc_df[cols['rf']['CUT_50']]
        """
        out: dict[str, dict[str, str]] = {}
        lvl = self.levels
        if lvl.level41_rf:
            out['rf'] = {scen: inst.gapfilled_.name
                         for scen, inst in lvl.level41_rf.items()}
        if lvl.level41_xgb:
            out['xgb'] = {scen: inst.gapfilled_.name
                          for scen, inst in lvl.level41_xgb.items()}
        if lvl.level41_mds:
            out['mds'] = {scen: inst.get_gapfilled_target().name
                          for scen, inst in lvl.level41_mds.items()}
        return out
