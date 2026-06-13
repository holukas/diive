"""
LEVEL 2: QUALITY FLAG EXPANSION
================================

Composable callable that expands flux quality flags from EddyPro output
and computes the overall QCF.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

from diive.core.utils.console import rule
from diive.flux.fluxprocessingchain.container import FluxLevelData
from diive.flux.fluxprocessingchain.levels._qcf import finalize_level
from diive.flux.fluxprocessingchain.levels._rerun import (
    cascade_reset,
    record_added_columns,
)
from diive.flux.lowres.quality_flags import FluxQualityFlagsEddyPro

#: The eight VM97 (Vickers & Mahrt 1997) raw-data screening sub-tests EddyPro
#: encodes in the single ``{fluxbasevar}_VM97_TEST`` integer, as
#: ``(kwarg_key, label, kind)``. ``kind`` is ``'hard'`` (failure -> flag 2) or
#: ``'soft'`` (failure -> flag 1, a marginal warning). The kwarg keys match the
#: ``raw_data_screening_vm97`` sub-keys of :func:`run_level2`.
VM97_SUBTESTS: list[tuple[str, str, str]] = [
    ("spikes", "Spike detection", "hard"),
    ("amplitude", "Amplitude resolution", "hard"),
    ("dropout", "Dropout detection", "hard"),
    ("abslim", "Absolute limits", "hard"),
    ("skewkurt_hf", "Skewness / kurtosis (hard)", "hard"),
    ("skewkurt_sf", "Skewness / kurtosis (soft)", "soft"),
    ("discont_hf", "Discontinuities (hard)", "hard"),
    ("discont_sf", "Discontinuities (soft)", "soft"),
]


def level2_test_inputs(fluxcol: str, fluxbasevar: str) -> dict[str, dict]:
    """Report the EddyPro-FLUXNET input column(s) each Level-2 test reads.

    Lets a caller (e.g. the GUI flux-chain tab) tell the user which variables a
    quality test depends on, and check upfront whether the loaded dataset
    actually provides them. Column names are templated on the flux column and
    its base variable (see :func:`diive.flux.lowres.common.detect_fluxbasevar`),
    mirroring the column names the underlying ``flag_*_eddypro_test`` functions
    read.

    The signal-strength test reads a *user-chosen* column rather than a fixed
    one, so its entry carries ``user_col=True`` and an empty ``inputs`` list —
    the caller supplies the column.

    Args:
        fluxcol: Flux column name (e.g. ``'FC'``).
        fluxbasevar: Base variable the flux was computed from (e.g. ``'CO2'``).

    Returns:
        ``{test_key: {"label": str, "inputs": [col, ...], "user_col": bool}}``,
        keyed by the ``run_level2`` test kwarg. The always-on missing-values
        test (which reads only ``fluxcol``) is omitted — it is not user-toggled.
    """
    return {
        "ssitc": {"label": "SSITC (steady-state / turbulence)",
                  "inputs": [f"{fluxcol}_SSITC_TEST"], "user_col": False},
        "gas_completeness": {"label": "Gas completeness",
                             "inputs": ["EXPECT_NR", f"{fluxbasevar}_NR"],
                             "user_col": False},
        "spectral_correction_factor": {"label": "Spectral correction factor",
                                        "inputs": [f"{fluxcol}_SCF"],
                                        "user_col": False},
        "signal_strength": {"label": "Signal strength (IRGA AGC)",
                            "inputs": [], "user_col": True},
        "raw_data_screening_vm97": {"label": "Raw-data screening (VM97)",
                                    "inputs": [f"{fluxbasevar}_VM97_TEST"],
                                    "user_col": False},
        "angle_of_attack": {"label": "Angle of attack",
                            "inputs": ["VM97_AOA_HF"], "user_col": False},
        "steadiness_of_horizontal_wind": {"label": "Steadiness of horizontal wind",
                                          "inputs": ["VM97_NSHW_HF"], "user_col": False},
    }


def run_level2(
        data: FluxLevelData,
        *,
        signal_strength: dict | None = None,
        raw_data_screening_vm97: dict | None = None,
        ssitc: dict | None = None,
        gas_completeness: dict | None = None,
        spectral_correction_factor: dict | None = None,
        angle_of_attack: dict | None = None,
        steadiness_of_horizontal_wind: dict | None = None,
) -> FluxLevelData:
    """
    Level-2: expand flux quality flags from EddyPro output.

    Each test is enabled by passing a config dict containing at least
    ``{'apply': True, ...}``.  Pass ``None`` (or omit) to skip a test.

    Every test reads a fixed EddyPro-FLUXNET input column derived from the flux
    column / its base variable (see :func:`level2_test_inputs`).  To point a test
    at a differently-named column, add a ``'col'`` key to its config dict
    (``'expect_nr_col'`` / ``'basevar_nr_col'`` for the two-column completeness
    test).  Absent or ``None`` keeps the standard templated name.

    Args:
        data: FluxLevelData from ``init_flux_data()``.
        signal_strength: Config dict with keys:

            - ``'apply'``: ``True`` to enable the test.
            - ``'signal_strength_col'``: column name of the signal strength /
              AGC (Automatic Gain Control) diagnostic from EddyPro output
              (e.g. ``'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN'``).
            - ``'method'``: ``'discard above'`` to flag records where signal
              strength *exceeds* the threshold (high AGC = dirty optics on
              some analyzers), or ``'discard below'`` to flag records where
              it *falls below* the threshold (low signal = instrument issue).
              Check your analyzer's manual to know which direction applies.
            - ``'threshold'``: numeric cutoff value (instrument-specific).

            Example for a LI-7200 where low signal strength indicates a problem::

                signal_strength={
                    'apply': True,
                    'signal_strength_col': 'CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
                    'method': 'discard below',
                    'threshold': 60,
                }
        raw_data_screening_vm97: Config dict with ``'apply': True`` and eight
            boolean sub-keys (``True`` to enable, ``False`` to skip each test):

            - ``'spikes'``: spike detection
            - ``'amplitude'``: amplitude resolution check
            - ``'dropout'``: dropout detection
            - ``'abslim'``: absolute limits check
            - ``'skewkurt_hf'``: skewness/kurtosis on high-frequency data
            - ``'skewkurt_sf'``: skewness/kurtosis on slow-response data
            - ``'discont_hf'``: discontinuities in high-frequency data
            - ``'discont_sf'``: discontinuities in slow-response data

            All eight keys must be present even if set to ``False``.
            (Vickers & Mahrt 1997)
        ssitc: Config dict; optional ``setflag_timeperiod``.
        gas_completeness: Config dict (just ``{'apply': True}``).
        spectral_correction_factor: Config dict (just ``{'apply': True}``).
        angle_of_attack: Config dict; requires ``application_dates``.
        steadiness_of_horizontal_wind: Config dict (just ``{'apply': True}``).

    Returns:
        Updated FluxLevelData with ``levels.level2``, ``levels.level2_qcf``,
        ``levels.filteredseries_level2_qcf``, and ``levels.filteredseries_hq``
        populated.
    """
    # Validate required sub-keys upfront so users get a clear error at the
    # call site, not a confusing KeyError buried inside the level class.
    if signal_strength and signal_strength.get('apply'):
        _required = ('signal_strength_col', 'method', 'threshold')
        _missing = [k for k in _required if k not in signal_strength]
        if _missing:
            raise KeyError(
                f"signal_strength config is missing required key(s): {_missing}. "
                f"Required: {{'apply': True, 'signal_strength_col': '...', "
                f"'method': 'discard above' | 'discard below', 'threshold': <int>}}"
            )

    if raw_data_screening_vm97 and raw_data_screening_vm97.get('apply'):
        _required_vm97 = ('spikes', 'amplitude', 'dropout', 'abslim',
                          'skewkurt_hf', 'skewkurt_sf', 'discont_hf', 'discont_sf')
        _missing_vm97 = [k for k in _required_vm97 if k not in raw_data_screening_vm97]
        if _missing_vm97:
            raise KeyError(
                f"raw_data_screening_vm97 config is missing required key(s): {_missing_vm97}. "
                f"All eight boolean sub-keys must be present (set to True or False): "
                f"{list(_required_vm97)}"
            )

    idstr = 'L2'
    meta = data.meta

    # If this level (or any downstream level) has already run, clear that
    # state first so a re-run produces clean output instead of duplicated /
    # stale columns. See levels/_rerun.py for the cascade semantics.
    if idstr in data.level_ids:
        data = cascade_reset(data, idstr)
    pre_columns = list(data.fpc_df.columns)

    rule("Level 2: Quality Flag Expansion")

    level2 = FluxQualityFlagsEddyPro(
        fluxcol=meta.fluxcol,
        dfin=data.full_df,
        idstr=idstr,
        fluxbasevar=meta.fluxbasevar,
    )
    level2.missing_vals_test()

    # Each test optionally overrides the EddyPro-FLUXNET input column it reads
    # via a ``'col'`` key (``'expect_nr_col'`` / ``'basevar_nr_col'`` for the
    # two-column completeness test). Absent / None -> the standard templated name.
    if ssitc and ssitc.get('apply'):
        level2.ssitc_test(setflag_timeperiod=ssitc.get('setflag_timeperiod'),
                          flagcol=ssitc.get('col'))
    if gas_completeness and gas_completeness.get('apply'):
        level2.gas_completeness_test(expect_nr_col=gas_completeness.get('expect_nr_col'),
                                     basevar_nr_col=gas_completeness.get('basevar_nr_col'))
    if spectral_correction_factor and spectral_correction_factor.get('apply'):
        level2.spectral_correction_factor_test(scfcol=spectral_correction_factor.get('col'))
    if signal_strength and signal_strength.get('apply'):
        level2.signal_strength_test(
            signal_strength_col=signal_strength['signal_strength_col'],
            method=signal_strength['method'],
            threshold=signal_strength['threshold'],
        )
    if raw_data_screening_vm97 and raw_data_screening_vm97.get('apply'):
        level2.raw_data_screening_vm97_tests(
            spikes=raw_data_screening_vm97['spikes'],
            amplitude=raw_data_screening_vm97['amplitude'],
            dropout=raw_data_screening_vm97['dropout'],
            abslim=raw_data_screening_vm97['abslim'],
            skewkurt_hf=raw_data_screening_vm97['skewkurt_hf'],
            skewkurt_sf=raw_data_screening_vm97['skewkurt_sf'],
            discont_hf=raw_data_screening_vm97['discont_hf'],
            discont_sf=raw_data_screening_vm97['discont_sf'],
            vm97col=raw_data_screening_vm97.get('col'),
        )
    if angle_of_attack and angle_of_attack.get('apply'):
        level2.angle_of_attack_test(application_dates=angle_of_attack['application_dates'],
                                    aoacol=angle_of_attack.get('col'))
    if steadiness_of_horizontal_wind and steadiness_of_horizontal_wind.get('apply'):
        level2.steadiness_of_horizontal_wind(nshwcol=steadiness_of_horizontal_wind.get('col'))

    updated, qcf = finalize_level(
        data,
        run_qcf_on_col=meta.fluxcol,
        idstr=idstr,
        level_df=level2.results,
    )

    new_levels = replace(
        updated.levels,
        level2=level2,
        level2_qcf=qcf,
        filteredseries_level2_qcf=updated.filteredseries.copy(),
        filteredseries_hq=qcf.filteredseries_hq.copy(),
    )
    level_ids = list(updated.level_ids)
    if idstr not in level_ids:
        level_ids.append(idstr)

    final = replace(updated, levels=new_levels, level_ids=level_ids)
    return replace(final, added_columns=record_added_columns(final, idstr, pre_columns))
