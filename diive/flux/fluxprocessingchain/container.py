"""
CONTAINER: STANDARDIZED CONTAINER FOR COMPOSABLE FLUX PROCESSING
=================================================================

Typed data containers passed between the standalone level callables.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
    from diive.flux.lowres.storage_correction import (
        FluxStorageCorrectionSinglePointEddyPro,
    )
    from diive.flux.lowres.ustar_bootstrap import UstarBootstrapThresholds
    from diive.flux.lowres.ustarthreshold import FlagMultipleConstantUstarThresholds
    from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
    from diive.preprocessing.qaqc import FlagQCF


@dataclass
class FluxConfig:
    """
    Per-flux configuration for multi-flux processing loops.

    Captures every setting that differs between fluxes (CO2, H, LE, N2O, CH4, …)
    so that a composable-function loop can process each variable without repeating
    site-level parameters.

    **Consumed by** :func:`~diive.flux.fluxprocessingchain.run_chain`.  The
    per-level ``run_level*`` functions take their own specific arguments — they
    do **not** accept a ``FluxConfig``.  Use ``run_chain(data, config)`` for the
    standard pipeline, or drop down to the composable per-level API for custom
    L3.2 outlier logic or custom feature engineering.

    **Scope.** ``FluxConfig`` captures the *high-level* decisions a typical
    user makes per flux — which L2 tests, which USTAR-detection mode, which
    gap-filling methods, which MDS / ML driver columns. It does **not** cover
    every knob each level exposes: Hampel sub-options (``use_differencing``,
    ``repeat``, ``k``), non-Hampel L3.2 detectors, MDS tolerances, ML
    hyperparameters, custom ``FeatureEngineer`` configurations, and L3.3
    diagnostic plotting are all reachable only via the composable per-level
    API. See :func:`run_chain` for the full list of what is and is not
    exposed. The composable per-level callables are the path to full control.

    Only ``fluxcol`` and ``ustar_thresholds`` are unconditionally required.
    All other fields default to ``None`` / sensible booleans and are validated
    *contextually* by ``run_chain`` — e.g. ``gapfilling_features`` is required
    only when ``gapfill_rf`` or ``gapfill_xgb`` is ``True``;
    ``outlier_sigma_*`` is unconditionally required by ``run_chain`` (L3.2
    always runs because L3.3 depends on it);
    ``mds_swin`` / ``mds_ta`` / ``mds_vpd`` are required only when
    ``gapfill_mds=True``. This way each new flux declaration only needs the
    fields that match the features it actually enables.

    Example — NEE, H, and N2O configs::

        from diive.flux.fluxprocessingchain import FluxConfig

        fc_cfg = FluxConfig(
            fluxcol='FC',
            ustar_thresholds=[0.30],
            ustar_labels=['CUT_50'],
            outlier_sigma_daytime=5.5,
            outlier_sigma_nighttime=5.5,
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
            level2_test_settings={
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
            level2_test_settings={
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
            level2_test_settings={
                'ssitc': {'apply': True, 'setflag_timeperiod': None},
                'gas_completeness': {'apply': True},
            },
            gapfill_mds=False,  # MDS not appropriate for N2O without dedicated drivers
        )

    Example — detect USTAR thresholds via bootstrap (FLUXNET standard)::

        nee_boot_cfg = FluxConfig(
            fluxcol='FC',
            ustar_detection_mode='bootstrap',
            ustar_bootstrap_ta_col='TA_1_1_1',
            ustar_bootstrap_swin_col='SW_IN_1_1_1',
            ustar_bootstrap_percentiles=(16, 50, 84),  # produces CUT_16/50/84
            outlier_sigma_daytime=5.5,
            outlier_sigma_nighttime=5.5,
            gapfilling_features=['TA_1_1_1', 'SW_IN_1_1_1', 'VPD_1_1_1'],
            level2_test_settings={'ssitc': {'apply': True, 'setflag_timeperiod': None}},
            mds_swin='SW_IN_1_1_1', mds_ta='TA_1_1_1', mds_vpd='VPD_kPa_1_1_1',
        )
        # The fitted bootstrap instance is available on data.levels.ustar_detection
        # after run_chain() for inspection of annual/per-year statistics.
    """

    # ------------------------------------------------------------------ required
    fluxcol: str
    """Target flux column (e.g. ``'FC'``, ``'H'``, ``'N2O'``)."""

    # ------------------------------------------------------------------ optional
    # Each field below is only consulted when the corresponding feature is
    # enabled. ``run_chain`` validates these contextually — e.g. it raises if
    # ``gapfill_rf=True`` but ``gapfilling_features`` is empty, but accepts
    # ``gapfilling_features=None`` when both ML methods are off.

    # ----- USTAR filtering (Level-3.3) -----
    ustar_detection_mode: str = 'constant'
    """How USTAR thresholds are obtained for Level-3.3.

    - ``'constant'`` (default) — use the values you provide in
      ``ustar_thresholds`` directly (e.g. previously computed by REddyProc).
      Fastest; no per-call computation. Required field: ``ustar_thresholds``.
    - ``'bootstrap'`` — detect thresholds from the data itself via
      multi-year bootstrap (ONEFlux moving-point method by default,
      Papale et al. 2006). Slower but fully reproducible within the
      pipeline. Required fields: ``ustar_bootstrap_ta_col`` and
      ``ustar_bootstrap_swin_col``. ``ustar_thresholds`` /
      ``ustar_labels`` are ignored in this mode — thresholds and
      ``CUT_<p>`` labels are produced from
      ``ustar_bootstrap_percentiles``."""

    ustar_thresholds: list | None = None
    """USTAR threshold(s) in m s-1 — one per scenario. Required when
    ``ustar_detection_mode='constant'``. Use ``[0.0]`` for energy fluxes
    (H, LE) to keep all records."""

    ustar_labels: list | None = None
    """Short label per threshold (e.g. ``['CUT_50']``; ``['CUT_NONE']`` for H/LE).

    - With a single threshold, ``None`` is allowed: ``run_chain`` lets the
      underlying function auto-generate the label as ``['CUT_0']``.
    - With **multiple** thresholds, ``run_chain`` **requires** explicit labels
      to avoid silently emitting non-percentile names (``CUT_0``, ``CUT_1``,
      ...) for what are typically percentile-based thresholds. Pass e.g.
      ``['CUT_16', 'CUT_50', 'CUT_84']`` to match the percentile semantics
      used everywhere else in the chain.

    Ignored when ``ustar_detection_mode='bootstrap'`` — labels are derived
    from ``ustar_bootstrap_percentiles`` (e.g. ``CUT_16`` / ``CUT_50`` /
    ``CUT_84``)."""

    ustar_bootstrap_ta_col: str | None = None
    """Air temperature column (deg C) in ``data.full_df`` used by the bootstrap
    detector to stratify nighttime records into temperature classes. Required
    when ``ustar_detection_mode='bootstrap'``; ignored otherwise."""

    ustar_bootstrap_swin_col: str | None = None
    """Shortwave incoming radiation column (W m-2) in ``data.full_df`` used by
    the bootstrap detector to identify nighttime periods. Required when
    ``ustar_detection_mode='bootstrap'``; ignored otherwise."""

    ustar_bootstrap_n_iter: int = 100
    """Bootstrap iterations per year window. Larger values give tighter
    percentile estimates at proportional runtime cost. Used only when
    ``ustar_detection_mode='bootstrap'``."""

    ustar_bootstrap_n_jobs: int = 1
    """Parallel workers for the bootstrap (1 = sequential, -1 = all CPUs).
    Used only when ``ustar_detection_mode='bootstrap'``."""

    ustar_bootstrap_percentiles: tuple = (16, 50, 84)
    """Bootstrap percentiles to compute and use as separate USTAR scenarios.
    Each value ``p`` produces one scenario labelled ``CUT_<p>``. Used only
    when ``ustar_detection_mode='bootstrap'``."""

    level2_test_settings: dict | None = None
    """Which Level-2 quality tests to run, and the settings for each.

    Shape is a **dict-of-dicts**: each top-level key is a test name, each
    value is that test's settings dict (always including ``'apply': True``
    to enable it). Example::

        level2_test_settings = {
            'ssitc': {'apply': True, 'setflag_timeperiod': None},
            'gas_completeness': {'apply': True},
            'spectral_correction_factor': {'apply': True},
        }

    Recognised top-level keys (each name selects a Level-2 test;
    omit a key to skip that test): ``ssitc``, ``gas_completeness``,
    ``spectral_correction_factor``, ``signal_strength``,
    ``raw_data_screening_vm97``, ``angle_of_attack``,
    ``steadiness_of_horizontal_wind``. See :func:`run_level2` for the
    settings each test accepts.

    ``None`` (the default) runs L2 with only the always-on missing-values
    test — no other QC test is applied."""

    outlier_sigma_daytime: float | None = None
    """Hampel filter sigma for daytime outlier detection. **Required by
    `run_chain`** — L3.2 always runs because L3.3 USTAR filtering depends on
    outlier-screened data. No universal default applies; choose by inspecting
    the flux record. Typical range: 5–6 for CO2/H/LE; 3–5 for trace gases
    (N2O, CH4). If you have screened outliers upstream and want to skip L3.2
    entirely, use the composable per-level API instead of ``run_chain``."""

    outlier_sigma_nighttime: float | None = None
    """Hampel filter sigma for nighttime outlier detection. **Required by
    ``run_chain``** for the same reason as ``outlier_sigma_daytime``."""

    gapfilling_features: list | None = None
    """Predictor column names for ML gap-filling (RF, XGBoost). Must exist in
    ``data.full_df``. Required when ``gapfill_rf=True`` or ``gapfill_xgb=True``;
    ignored otherwise."""

    set_storage_to_zero: bool = False
    """Set to ``True`` when no storage measurement is available for this flux.
    The storage correction (L3.1) still runs; it just adds zero."""

    gapfill_storage_term: bool = True
    """Whether L3.1 should gap-fill missing storage values with a rolling
    median before adding the storage term to the flux. Default ``True``
    matches ``run_level31``'s own default. Set to ``False`` to add only the
    raw (non-gap-filled) storage term — any nighttime record with missing
    storage then becomes NaN. Ignored when ``set_storage_to_zero=True``."""

    outlier_window_length: int = 48 * 13
    """Hampel filter rolling window length, **expressed as a record count** (not a
    duration). Default ``48 * 13 = 624`` records, which equals **13 days at the
    half-hourly (30-min) sampling rate** assumed throughout the flux processing
    chain.

    The chain is designed for half-hourly EddyPro output. If your data has a
    different sampling rate, scale this value yourself — e.g. for hourly data
    use ``24 * 13`` to keep the same 13-day window. The value is forwarded
    verbatim to :meth:`StepwiseOutlierDetection.flag_outliers_hampel_test` and
    is interpreted in records, never in time units."""

    gapfill_rf: bool = True
    """Run Random Forest gap-filling (L4.1)."""

    gapfill_xgb: bool = False
    """Run XGBoost gap-filling (L4.1).  Off by default — slower than RF."""

    gapfill_mds: bool = True
    """Run MDS gap-filling (L4.1)."""

    gapfill_reduce_features: bool = True
    """SHAP-based feature reduction for the ML gap-fillers (RF / XGBoost).
    Default ``True`` — the chain's L4.1 ML methods are designed around SHAP
    importance ranking; with reduction disabled the model trains on every
    engineered feature regardless of contribution and is typically weaker.
    Set to ``False`` only when you want the raw (unreduced) feature set,
    e.g. for diagnostic comparison. Ignored when both ``gapfill_rf`` and
    ``gapfill_xgb`` are ``False``."""

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

    **Treat as immutable.** Unlike :class:`FluxMeta`, this dataclass is not
    declared ``frozen=True`` — freezing would block field reassignment but
    leave the nested dicts (``level41_mds`` / ``_rf`` / ``_xgb``) and Series
    fields mutable, so it would only be half a guarantee. The chain instead
    enforces immutability by convention: every ``run_level*`` function rebuilds
    a fresh ``LevelResults`` via ``dataclasses.replace(data.levels, …)``.  Do
    not set fields, append to ``level41_*`` dicts, or modify ``filteredseries_*``
    in place — use ``replace`` if you need to extend the container.

    **High-quality vs. accepted-quality series:**

    - ``filteredseries_hq`` — flux with QCF=0 *only* (strictest filter, no
      soft warnings tolerated).  Used as the reference series for gap-filling
      model training.  Written by L2 (where the QCF=0 mask is computed) and
      by L3.1 (which reapplies that same L2 QCF=0 mask to the
      storage-corrected flux — L3.1 itself has no new QC test).  L3.2 stores
      its QCF=0 series only in ``filteredseries_level32_qcf``-adjacent
      attributes (it does not overwrite this field), and L3.3 keeps the
      per-USTAR-scenario HQ series in ``filteredseries_level33_hq`` instead.
    - ``filteredseries_level*_qcf`` — flux accepted at QCF < threshold (set by
      ``daytime_accept_qcf_below`` / ``nighttime_accept_qcf_below`` in
      ``init_flux_data``).  The threshold is usually 1 or 2 depending on the
      level and site protocol.

    The corresponding columns in ``data.fpc_df`` carry the same distinction in
    their suffixes: ``..._QCF`` for the user-accepted series and ``..._QCF0``
    for the strictly-QCF=0 series.  When ``accept_qcf_below=1`` the two
    contain identical values, but the names stay distinct so that the
    *intent* of each column is preserved across re-runs with different
    thresholds.

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
    filteredseries_hq: pd.Series | None = None
    """Flux filtered at strictly QCF=0 (highest-quality reference). Written by
    Level-2 (initial QCF=0 mask on the raw flux) and by Level-3.1 (same mask
    reapplied to the storage-corrected flux). The series name uses the
    ``_QCF0`` suffix to mark the strict-zero filter (contrast with ``_QCF``
    for the user-accepted series).

    **Lifecycle vs. L3.2 / L3.3.** L3.2 does run a QCF but does not overwrite
    this field — its QCF=0 series can be read from ``levels.level32_qcf``
    directly if needed. L3.3 does not touch this field at all and stores the
    per-USTAR-scenario HQ series in ``filteredseries_level33_hq``. Therefore
    once L3.2 or L3.3 has run, ``filteredseries_hq`` reflects only the
    *L3.1-stage* HQ series — it is **stale** for downstream use. Reach for
    ``levels.filteredseries_level33_hq[<scen>]`` post-L3.3."""

    # Level-3.1
    level31: FluxStorageCorrectionSinglePointEddyPro | None = None
    level31_qcf: FlagQCF | None = None
    """``FlagQCF`` re-aggregating the L2-inherited flags on the
    storage-corrected target. **L3.1 introduces no new quality test** —
    storage availability is provenance, not quality, and the
    ``FLAG_..._ISFILLED`` column the storage correction emits is
    deliberately not picked up by ``FlagQCF`` (no ``_TEST`` suffix).
    Provides the overall ``FLAG_L3.1_<outname>_QCF`` column and the
    per-record ``SUM`` aggregates the other levels also expose."""
    flux_corrected_col: str | None = None
    filteredseries_level31_qcf: pd.Series | None = None
    """User-accepted QCF-filtered storage-corrected flux. Equals
    ``level31_qcf.filteredseries`` — kept as a separate field for symmetry
    with the other levels."""

    # Level-3.2
    level32: StepwiseOutlierDetection | None = None
    level32_qcf: FlagQCF | None = None
    filteredseries_level32_qcf: pd.Series | None = None

    # Level-3.3
    level33: FlagMultipleConstantUstarThresholds | None = None
    level33_qcf: dict[str, FlagQCF] = field(default_factory=dict)
    filteredseries_level33_qcf: dict[str, pd.Series] = field(default_factory=dict)
    filteredseries_level33_hq: dict[str, pd.Series] = field(default_factory=dict)
    ustar_detection: 'UstarBootstrapThresholds | None' = None
    """``UstarBootstrapThresholds`` instance when ``run_level33_ustar_detection``
    was used; ``None`` otherwise."""

    # Level-4.1 — one dict per gap-filling method, keyed by USTAR scenario
    level41_mds: dict[str, Any] = field(default_factory=dict)
    level41_rf: dict[str, Any] = field(default_factory=dict)
    level41_xgb: dict[str, Any] = field(default_factory=dict)

    def has_level41(self) -> bool:
        """True if any L4.1 method has produced results."""
        return bool(self.level41_mds or self.level41_rf or self.level41_xgb)

    def level41_methods(self) -> dict[str, dict[str, Any]]:
        """Return all L4.1 method dicts that have results, keyed by short method name.

        Keys are ``'mds'``, ``'rf'``, ``'xgb'`` — matching the suffixes of the
        underlying ``level41_*`` attributes and the keys used by
        ``gapfilled_cols()`` and the plot helpers.
        """
        out: dict[str, dict[str, Any]] = {}
        if self.level41_mds:
            out['mds'] = self.level41_mds
        if self.level41_rf:
            out['rf'] = self.level41_rf
        if self.level41_xgb:
            out['xgb'] = self.level41_xgb
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

    Need a new driver mid-pipeline (e.g. a computed VPD column for MDS)?
    Use ``add_driver(data, series, name=...)`` — it adds the column to
    ``full_df`` where L4.1 will actually look for it, not to ``fpc_df``.
    """

    fpc_df: pd.DataFrame
    full_df: pd.DataFrame
    filteredseries: pd.Series | None
    """QCF-filtered flux from the most recently completed level (accepted-quality
    threshold, not QCF=0-only).  ``None`` before any level has run."""
    meta: FluxMeta
    levels: LevelResults = field(default_factory=LevelResults)
    level_ids: list[str] = field(default_factory=list)
    """Ordered list of level idstrs that have run (``'L2'``, ``'L3.1'``,
    ``'L3.2'``, ``'L3.3'``, ``'L4.1'``). Records the *cascade-aware* level
    boundaries only — for L4.1 the single entry ``'L4.1'`` is appended
    whichever subset of MDS / RF / XGBoost ran; it does not tell you which
    methods produced output. To discover which gap-filling methods are
    available, use :meth:`gapfilled_cols` or
    :meth:`LevelResults.level41_methods`. Cleared by the re-run cascade
    (see ``levels/_rerun.py``)."""
    added_columns: dict[str, list[str]] = field(default_factory=dict)
    """Columns each level added to ``fpc_df``, keyed by level idstr (``'L2'``,
    ``'L3.1'``, ``'L3.2'``, ``'L3.3'``) or per-method L4.1 key
    (``'L4.1_mds'``, ``'L4.1_rf'``, ``'L4.1_xgb'``). Used to clean up stale
    columns when a level is re-run."""

    def __repr__(self) -> str:
        rows, cols = self.fpc_df.shape
        fs = self.filteredseries
        if fs is not None:
            n_valid = int(fs.dropna().count())
            n_total = len(fs)
            fs_str = f"{fs.name!r} ({n_valid}/{n_total} valid)"
        elif 'L3.3' in self.level_ids:
            # filteredseries is deliberately cleared by L3.3 because there is
            # no single unambiguous filtered series across USTAR scenarios.
            fs_str = "None (L3.3 ran; use data.levels.filteredseries_level33_qcf[<ustar_scenario>])"
        else:
            # No level has run yet — filteredseries simply isn't populated.
            fs_str = "None (no level has run yet)"
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

    def gap_stats(self,
                 level: str = 'L3.3',
                 long_gap_records: int = 48) -> dict[str, 'GapStats']:
        """Return gap statistics for the QCF-filtered series at the given level.

        Creates a :class:`~diive.analysis.GapStats` instance for each series
        available at the requested level.  For levels with USTAR scenarios
        (L3.3) one entry is returned per scenario.

        Args:
            level: Level whose QCF-filtered series to analyse. Use the
                dotted idstrs that match the rest of the chain:
                ``'L2'``, ``'L3.1'``, ``'L3.2'``, ``'L3.3'``.
                Defaults to ``'L3.3'`` — the series that goes into gap-filling.
            long_gap_records: Gaps >= this many consecutive records are
                flagged as *long gaps* in the report and figure.  Default 48
                equals one day at 30-min resolution.

        Returns:
            ``{label: GapStats}`` — a flat mapping whose keys are whichever
            label uniquely identifies each result at that level:

            - L2 / L3.1 / L3.2: one entry keyed by the level name itself
              (``{'L2': gs}`` / ``{'L3.1': gs}`` / ``{'L3.2': gs}``).
            - L3.3: one entry per USTAR scenario keyed by the scenario label
              (``{'CUT_16': gs, 'CUT_50': gs, 'CUT_84': gs}``).

            The label vocabulary differs by level on purpose: at L2/L3.1/L3.2
            there is only one filtered series so the level name is the
            natural key, whereas at L3.3 the scenario label is the
            distinguishing axis. A generic loop
            ``for label, gs in data.gap_stats(level).items(): ...``
            therefore works at every level — the ``label`` you get back is
            the right caption to render alongside each ``GapStats``.

        Raises:
            ValueError: If the requested level has not been run yet, or if
                an unknown level idstr is passed.

        Example::

            # Analyse gaps just before gap-filling
            for scen, gs in data.gap_stats('L3.3').items():
                print(scen)
                gs.report()
                gs.showfig(title=f"Gap stats -- {scen}")

            # Single-level access
            gs = data.gap_stats('L2')['L2']
            gs.report()
        """
        from diive.analysis.gapfinder import GapStats

        # Accept the dotted form as the canonical input. The pre-harmonization
        # dotless variants ('L31', 'L32', 'L33') are silently normalised to
        # their dotted equivalents so existing code that still uses them keeps
        # working — but they are not the recommended form.
        _alias = {'L31': 'L3.1', 'L32': 'L3.2', 'L33': 'L3.3'}
        level = _alias.get(level, level)

        lvl = self.levels
        _single = {
            'L2':   lvl.filteredseries_level2_qcf,
            'L3.1': lvl.filteredseries_level31_qcf,
            'L3.2': lvl.filteredseries_level32_qcf,
        }

        if level in _single:
            series = _single[level]
            if series is None:
                raise ValueError(
                    f"Level {level!r} has not been run yet — "
                    f"call the corresponding run_level*() function first."
                )
            return {level: GapStats(series, long_gap_records=long_gap_records)}

        if level == 'L3.3':
            if not lvl.filteredseries_level33_qcf:
                raise ValueError(
                    "Level 'L3.3' has not been run yet — "
                    "call run_level33_constant_ustar() first."
                )
            return {
                scen: GapStats(s, long_gap_records=long_gap_records)
                for scen, s in lvl.filteredseries_level33_qcf.items()
            }

        raise ValueError(
            f"Unknown level {level!r}. "
            f"Valid options: 'L2', 'L3.1', 'L3.2', 'L3.3'."
        )

    def plot_cumulative_comparison(
            self,
            ustar_scenario: str | None = None,
            conv_factor: float = 1.0,
            units: str = '',
            show_measured: bool = True,
            title: str | None = None,
            saveplot: bool = False,
            path: str | None = None,
            showplot: bool = True,
    ) -> None:
        """Overlay cumulative sums of all gap-filled methods for direct comparison.

        Plots one line per gap-filling method (RF, XGBoost, MDS — whichever have
        been run) for the requested USTAR scenario on the same axes.  Optionally
        adds a dashed reference line for the measured-only series (gaps contribute
        zero, leaving flat segments) so the impact of gap-filling is visible.

        Call this after at least one ``run_level41_*`` function has completed.

        Args:
            ustar_scenario: USTAR scenario label to compare across methods (e.g.
                ``'CUT_50'``).  Defaults to the first available scenario.
            conv_factor: Multiply every flux value by this factor before
                accumulating.  For 30-min NEE in µmol CO₂ m⁻² s⁻¹ → gC m⁻²,
                use ``12.011 * 1e-6 * 1800``.  Default 1.0 (no conversion).
            units: Unit string shown in the y-axis label and legend, e.g.
                ``'gC m-2'``.
            show_measured: Add a dashed grey reference line for the L3.3
                QCF-filtered series before gap-filling (gaps counted as zero).
                Defaults to ``True``.
            title: Figure title.  Defaults to an auto-generated title.
            saveplot: Save the figure to disk.  Defaults to ``False``.
            path: Output directory when ``saveplot=True``.
            showplot: Call ``plt.show()`` after rendering.  Set to ``False``
                for headless / batch use or when embedding the figure into
                a larger composite.  Defaults to ``True``.

        Raises:
            RuntimeError: If no L4.1 method has been run yet.
            ValueError: If the requested USTAR scenario is not found.

        Example::

            UMOL_TO_GC = 12.011 * 1e-6 * 1800
            data.plot_cumulative_comparison(
                ustar_scenario='CUT_50',
                conv_factor=UMOL_TO_GC,
                units='gC m-2',
            )
        """
        import matplotlib.pyplot as plt

        cols = self.gapfilled_cols()
        if not cols:
            raise RuntimeError(
                "No gap-filled columns found. "
                "Run at least one run_level41_*() function first."
            )

        # Resolve USTAR scenario
        all_scenarios = sorted({s for mc in cols.values() for s in mc})
        if ustar_scenario is None:
            ustar_scenario = all_scenarios[0]
        elif ustar_scenario not in all_scenarios:
            raise ValueError(
                f"USTAR scenario {ustar_scenario!r} not found. "
                f"Available: {all_scenarios}"
            )

        _COLORS = {
            'rf':  '#2196F3',  # blue
            'xgb': '#FF9800',  # orange
            'mds': '#4CAF50',  # green
        }
        _LABELS = {
            'rf':  'Random Forest',
            'xgb': 'XGBoost',
            'mds': 'MDS',
        }

        fig, ax = plt.subplots(figsize=(14, 5))
        unit_str = f' ({units})' if units else ''

        # Measured-only reference (L3.3 QCF series; gaps = 0 contribution)
        if show_measured:
            meas = self.levels.filteredseries_level33_qcf.get(ustar_scenario)
            if meas is not None:
                meas_cumul = (meas * conv_factor).fillna(0).cumsum()
                final = meas_cumul.iloc[-1]
                legend_label = (f"Measured only  ({final:+.1f}{unit_str})"
                                if units else f"Measured only  ({final:+.1f})")
                ax.plot(meas_cumul.index, meas_cumul.values,
                        color='#9E9E9E', linewidth=1.2, linestyle='--',
                        alpha=0.8, label=legend_label, zorder=2)

        # One line per gap-filling method
        import warnings
        for method_key, scen_cols in cols.items():
            if ustar_scenario not in scen_cols:
                continue
            gf_col = scen_cols[ustar_scenario]
            s = self.fpc_df[gf_col]
            # Gap-filling is expected to leave no NaN — but in edge cases an
            # ML model may fail to predict a stretch (e.g. an entire year
            # missing all features). The cumulative line below treats any
            # residual NaN as zero, which can mislead the reader into thinking
            # nothing was missing. Warn so the user knows the line is not a
            # complete record.
            n_missing = int(s.isna().sum())
            if n_missing:
                warnings.warn(
                    f"Gap-filled series {gf_col!r} still has {n_missing} NaN "
                    f"record(s) for USTAR scenario {ustar_scenario!r}; the "
                    f"cumulative line treats these as zero contribution. "
                    f"Inspect data.levels.level41_{method_key}[{ustar_scenario!r}] "
                    f"to see why the gap-filler could not produce a value there.",
                    UserWarning,
                    stacklevel=2,
                )
            cumul = (s * conv_factor).fillna(0).cumsum()
            final = cumul.iloc[-1]
            base_label = _LABELS.get(method_key, method_key.upper())
            legend_label = (f"{base_label}  ({final:+.1f}{unit_str})"
                            if units else f"{base_label}  ({final:+.1f})")
            color = _COLORS.get(method_key, '#455A64')
            ax.plot(cumul.index, cumul.values,
                    color=color, linewidth=1.8, label=legend_label, zorder=3)

        ax.axhline(0, color='#455A64', linewidth=0.8, linestyle=':', alpha=0.6)
        y_label = f'Cumulative flux{unit_str}'
        ax.set_ylabel(y_label)
        ax.legend(loc='best', fontsize=9, framealpha=0.92)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4)

        auto_title = (f"Cumulative gap-filled flux  --  "
                      f"{self.meta.fluxcol}  |  USTAR scenario: {ustar_scenario}")
        ax.set_title(title or auto_title, fontsize=11)

        fig.tight_layout()

        if saveplot:
            from diive.core.plotting.plotfuncs import save_fig
            save_fig(fig=fig, title=title or auto_title, path=path)

        if showplot:
            plt.show()

    def plot_gapfilled_heatmaps(
            self,
            ustar_scenario: str | None = None,
            vmin: float | None = None,
            vmax: float | None = None,
            cmap: str = 'RdYlBu_r',
            units: str = '',
            title: str | None = None,
            saveplot: bool = False,
            path: str | None = None,
            showplot: bool = True,
    ) -> None:
        """Multi-panel heatmap: measured flux + one panel per gap-filling method.

        Shows the L3.3 QCF-filtered (pre-gap-filling) series in the top panel,
        followed by one heatmap per gap-filling method that has been run (RF,
        XGBoost, MDS).  All panels share the same colour scale so differences
        between methods are visually comparable.

        Call this after at least one ``run_level41_*`` function has completed.

        Args:
            ustar_scenario: USTAR scenario label (e.g. ``'CUT_50'``).  Defaults
                to the first available scenario.
            vmin: Minimum colour value.  Auto-computed (2nd percentile of all
                series) when ``None``.
            vmax: Maximum colour value.  Auto-computed (98th percentile) when
                ``None``.
            cmap: Matplotlib colourmap.  Defaults to ``'RdYlBu_r'``.
            units: Unit string shown on each colourbar (e.g. ``'umol m-2 s-1'``).
            title: Figure suptitle.  Auto-generated when ``None``.
            saveplot: Save the figure to disk.  Defaults to ``False``.
            path: Output directory when ``saveplot=True``.
            showplot: Call ``plt.show()`` after rendering.  Set to ``False``
                for headless / batch use or when embedding the figure into
                a larger composite.  Defaults to ``True``.

        Raises:
            RuntimeError: If no L4.1 method has been run yet.
            ValueError: If the requested USTAR scenario is not found.

        Example::

            data.plot_gapfilled_heatmaps(
                ustar_scenario='CUT_50',
                units='umol m-2 s-1',
            )
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime

        cols = self.gapfilled_cols()
        if not cols:
            raise RuntimeError(
                "No gap-filled columns found. "
                "Run at least one run_level41_*() function first."
            )

        all_scenarios = sorted({s for mc in cols.values() for s in mc})
        if ustar_scenario is not None:
            if ustar_scenario not in all_scenarios:
                raise ValueError(
                    f"USTAR scenario {ustar_scenario!r} not found. "
                    f"Available: {all_scenarios}"
                )
            scenarios_to_plot = [ustar_scenario]
        else:
            scenarios_to_plot = all_scenarios

        for scen in scenarios_to_plot:
            self._plot_heatmaps_one_scenario(
                scen, cols=cols, vmin=vmin, vmax=vmax,
                cmap=cmap, units=units, title=title,
                saveplot=saveplot, path=path, showplot=showplot,
            )

    def _plot_heatmaps_one_scenario(
            self, ustar_scenario, *, cols, vmin, vmax,
            cmap, units, title, saveplot, path, showplot,
    ) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from diive.core.plotting.heatmap_datetime import HeatmapDateTime

        _LABELS = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'mds': 'MDS'}

        # Build panel list: (series, subtitle)
        panels = []
        meas = self.levels.filteredseries_level33_qcf.get(ustar_scenario)
        if meas is not None:
            panels.append((meas, f"Before gap-filling  ({ustar_scenario})"))
        for method_key, scen_cols in cols.items():
            if ustar_scenario not in scen_cols:
                continue
            gf_col = scen_cols[ustar_scenario]
            label = _LABELS.get(method_key, method_key.upper())
            panels.append((self.fpc_df[gf_col], f"{label}  ({ustar_scenario})"))

        if not panels:
            return

        # Shared colour scale across all panels
        _vmin, _vmax = vmin, vmax
        if _vmin is None or _vmax is None:
            combined = pd.concat([s for s, _ in panels]).dropna()
            if _vmin is None:
                _vmin = float(np.nanpercentile(combined, 2))
            if _vmax is None:
                _vmax = float(np.nanpercentile(combined, 98))

        n = len(panels)
        zlabel = units or self.meta.fluxcol
        fig = plt.figure(figsize=(n * 5.5, 5), constrained_layout=True)
        gs = gridspec.GridSpec(1, n, figure=fig)

        for i, (series, subtitle) in enumerate(panels):
            ax = fig.add_subplot(gs[0, i])
            hm = HeatmapDateTime(series=series, verbose=False)
            hm.plot(ax=ax, fig=fig, title=subtitle,
                    vmin=_vmin, vmax=_vmax, cmap=cmap, zlabel=zlabel)

        auto_title = (f"Gap-filled flux heatmaps  --  "
                      f"{self.meta.fluxcol}  |  USTAR scenario: {ustar_scenario}")
        fig.suptitle(title or auto_title, fontsize=12, fontweight='bold')

        if saveplot:
            from diive.core.plotting.plotfuncs import save_fig
            save_fig(fig=fig, title=title or auto_title, path=path)

        if showplot:
            plt.show()

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
        # Insert in the canonical method order ('mds', 'rf', 'xgb') so the
        # iteration order is identical to LevelResults.level41_methods() and
        # the plot helpers — consumers can iterate either dict and get the
        # same sequence.
        #
        # Defensive lookup: the column name reported here is derived from
        # the underlying model instance (``.get_gapfilled_target().name``
        # for MDS, ``.gapfilled_.name`` for RF/XGB). That name must agree
        # with the column that was actually merged into ``fpc_df``; if a
        # future change to the underlying class ever returned a renamed
        # copy, callers would silently get a column name that doesn't
        # exist. Verify each reported name is in ``fpc_df.columns`` and
        # raise a clear error otherwise.
        lvl = self.levels
        _missing: list[str] = []

        def _check(method: str, scen: str, name: str) -> str:
            if name not in self.fpc_df.columns:
                _missing.append(f"{method}/{scen}: {name!r}")
            return name

        if lvl.level41_mds:
            out['mds'] = {scen: _check('mds', scen, inst.get_gapfilled_target().name)
                          for scen, inst in lvl.level41_mds.items()}
        if lvl.level41_rf:
            out['rf'] = {scen: _check('rf', scen, inst.gapfilled_.name)
                         for scen, inst in lvl.level41_rf.items()}
        if lvl.level41_xgb:
            out['xgb'] = {scen: _check('xgb', scen, inst.gapfilled_.name)
                          for scen, inst in lvl.level41_xgb.items()}

        if _missing:
            raise RuntimeError(
                "gapfilled_cols(): model instance reported a column name "
                "that is not present in data.fpc_df:\n"
                + "\n".join(f"  - {m}" for m in _missing)
                + "\nThis indicates the gap-filled column was renamed "
                  "between L4.1 emission and now (e.g. the underlying class "
                  "returned a renamed copy). Re-run the affected L4.1 method."
            )

        return out


def add_driver(
        data: FluxLevelData,
        series: pd.Series,
        name: str | None = None,
) -> FluxLevelData:
    """Add a Series as a driver column to ``data.full_df``.

    L4.1 gap-filling reads its driver / feature columns from ``data.full_df``,
    not ``data.fpc_df`` — a footgun, because the working dataframe (``fpc_df``)
    is the one that grows level-by-level and feels like the natural place to
    add a new variable. Use this helper to put the column where L4.1 will
    actually look for it.

    Args:
        data: Current FluxLevelData.
        series: Driver series. Its index must match ``data.full_df.index``.
        name: Column name to register the series under. Defaults to
            ``series.name``; required if ``series.name`` is ``None``.

    Returns:
        New ``FluxLevelData`` with the column added to ``full_df``.  The input
        is not mutated.

    Raises:
        ValueError: If ``series`` has no name and ``name`` is not supplied,
            its index does not match ``data.full_df.index``, or a column
            with the resolved name already exists in ``full_df``.

    Example::

        from diive.flux.fluxprocessingchain import add_driver
        from diive.variables import calc_vpd_from_ta_rh

        vpd = calc_vpd_from_ta_rh(ta=data.full_df['TA'], rh=data.full_df['RH'])
        data = add_driver(data, vpd, name='VPD_kPa')
    """
    col_name = name if name is not None else (
        str(series.name) if series.name is not None else None
    )
    if col_name is None:
        raise ValueError(
            "series has no name and no name= argument was supplied; cannot "
            "decide which column to register the driver under."
        )

    if not series.index.equals(data.full_df.index):
        # Strict equality is intentional: a silent reindex would mask
        # legitimate caller mistakes (wrong frequency, off-by-one start, a
        # subset that was meant to be full coverage). Surface the mismatch
        # and tell the user how to opt in to a reindex if that is what they
        # actually wanted.
        only_in_series = series.index.difference(data.full_df.index)
        only_in_full = data.full_df.index.difference(series.index)
        raise ValueError(
            f"series index does not match data.full_df.index "
            f"(series: {len(series)} rows, full_df: {len(data.full_df)} rows; "
            f"{len(only_in_series)} timestamps only in series, "
            f"{len(only_in_full)} only in full_df). "
            f"If the series is a subset (e.g. daytime-only) and the missing "
            f"timestamps should become NaN, reindex it first: "
            f"add_driver(data, series.reindex(data.full_df.index), name=...). "
            f"Otherwise check the frequency / start / end of the series."
        )

    if col_name in data.full_df.columns:
        raise ValueError(
            f"Column {col_name!r} already exists in data.full_df. Pick a "
            f"different name= or drop the existing column before adding."
        )

    new_full_df = data.full_df.copy()
    # Assign the Series itself rather than ``series.values``: indexes are
    # already equal so alignment is a no-op, and this preserves extension
    # dtypes (Int64, boolean, nullable string) that ``.values`` would
    # silently demote to object / float.
    new_full_df[col_name] = series
    return dataclasses.replace(data, full_df=new_full_df)
