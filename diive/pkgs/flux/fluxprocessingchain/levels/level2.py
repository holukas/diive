"""
LEVEL 2: QUALITY FLAG EXPANSION
================================

Composable callable that expands flux quality flags from EddyPro output
and computes the overall QCF.

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

from dataclasses import replace

from diive.pkgs.flux.fluxprocessingchain.container import FluxLevelData
from diive.pkgs.flux.fluxprocessingchain.level2_qualityflags import FluxQualityFlagsEddyPro
from diive.pkgs.flux.fluxprocessingchain.levels._qcf import finalize_level


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

    level2 = FluxQualityFlagsEddyPro(
        fluxcol=meta.fluxcol,
        dfin=data.full_df,
        idstr=idstr,
        fluxbasevar=meta.fluxbasevar,
    )
    level2.missing_vals_test()

    if ssitc and ssitc.get('apply'):
        level2.ssitc_test(setflag_timeperiod=ssitc.get('setflag_timeperiod'))
    if gas_completeness and gas_completeness.get('apply'):
        level2.gas_completeness_test()
    if spectral_correction_factor and spectral_correction_factor.get('apply'):
        level2.spectral_correction_factor_test()
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
        )
    if angle_of_attack and angle_of_attack.get('apply'):
        level2.angle_of_attack_test(application_dates=angle_of_attack['application_dates'])
    if steadiness_of_horizontal_wind and steadiness_of_horizontal_wind.get('apply'):
        level2.steadiness_of_horizontal_wind()

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

    return replace(updated, levels=new_levels, level_ids=level_ids)
