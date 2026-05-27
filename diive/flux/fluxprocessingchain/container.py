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
    from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro
    from diive.flux.lowres.storage_correction import (
        FluxStorageCorrectionSinglePointEddyPro,
    )
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

    **Required fields** (no defaults) force you to make explicit choices for every
    flux rather than silently inheriting wrong values from another variable.

    Example — NEE, H, and N2O configs::

        from diive.flux.fluxprocessingchain import FluxConfig

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
    ustar_detection: Any = None  # UstarBootstrapThresholds instance when detection was run

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

    def gap_stats(self,
                 level: str = 'L33',
                 long_gap_records: int = 48) -> dict[str, 'GapStats']:
        """Return gap statistics for the QCF-filtered series at the given level.

        Creates a :class:`~diive.analysis.GapStats` instance for each series
        available at the requested level.  For levels with USTAR scenarios
        (L33) one entry is returned per scenario.

        Args:
            level: Level whose QCF-filtered series to analyse.
                One of ``'L2'``, ``'L31'``, ``'L32'``, ``'L33'``.
                Defaults to ``'L33'`` — the series that goes into gap-filling.
            long_gap_records: Gaps >= this many consecutive records are
                flagged as *long gaps* in the report and figure.  Default 48
                equals one day at 30-min resolution.

        Returns:
            ``{label: GapStats}`` — single entry for L2/L31/L32, one entry
            per USTAR scenario for L33.

        Raises:
            ValueError: If the requested level has not been run yet.

        Example::

            # Analyse gaps just before gap-filling
            for scen, gs in data.gap_stats('L33').items():
                print(scen)
                gs.report()
                gs.showfig(title=f"Gap stats -- {scen}")

            # Single-level access
            gs = data.gap_stats('L2')['L2']
            gs.report()
        """
        from diive.analysis.gapfinder import GapStats

        lvl = self.levels
        _single = {
            'L2':  lvl.filteredseries_level2_qcf,
            'L31': lvl.filteredseries_level31_qcf,
            'L32': lvl.filteredseries_level32_qcf,
        }

        if level in _single:
            series = _single[level]
            if series is None:
                raise ValueError(
                    f"Level {level!r} has not been run yet — "
                    f"call the corresponding run_level*() function first."
                )
            return {level: GapStats(series, long_gap_records=long_gap_records)}

        if level == 'L33':
            if not lvl.filteredseries_level33_qcf:
                raise ValueError(
                    "Level 'L33' has not been run yet — "
                    "call run_level33_constant_ustar() first."
                )
            return {
                scen: GapStats(s, long_gap_records=long_gap_records)
                for scen, s in lvl.filteredseries_level33_qcf.items()
            }

        raise ValueError(
            f"Unknown level {level!r}. "
            f"Valid options: 'L2', 'L31', 'L32', 'L33'."
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
        for method_key, scen_cols in cols.items():
            if ustar_scenario not in scen_cols:
                continue
            gf_col = scen_cols[ustar_scenario]
            s = self.fpc_df[gf_col]
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
        plt.show()

        if saveplot:
            from diive.core.plotting.plotfuncs import save_fig
            save_fig(fig=fig, title=title or auto_title, path=path)

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
                saveplot=saveplot, path=path,
            )

    def _plot_heatmaps_one_scenario(
            self, ustar_scenario, *, cols, vmin, vmax,
            cmap, units, title, saveplot, path,
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
        plt.show()

        if saveplot:
            from diive.core.plotting.plotfuncs import save_fig
            save_fig(fig=fig, title=title or auto_title, path=path)

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
