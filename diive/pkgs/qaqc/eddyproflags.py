"""
Quality flags extracted from and calculated based on EddyPro output files.

DIIVE uses a standard quality flag format across all functions:
    - 0 = good quality (passes test)
    - 1 = soft warning (marginal, may indicate issues)
    - 2 = bad quality / hard fail (fails test)

EddyPro output files use different flag formats depending on the test type and file format:
    - Some flags are simple integers (0=pass, 1=fail) that need conversion
    - Some flags are multi-digit codes encoding multiple tests (e.g., VM97 8-digit codes)
    - Some data (e.g., signal strength) are continuous values requiring threshold comparison

This module provides functions to:
    1. Extract test flags from EddyPro FluxNet and full output files
    2. Convert EddyPro flag formats to DIIVE standard format
    3. Calculate new quality flags by applying thresholds to raw data

All functions return flags in DIIVE standard format (0=good, 1=soft warning, 2=bad)
to ensure consistency across the library.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.funcs.funcs import validate_id_string
from diive.pkgs.qaqc.flags import restrict_application


def flag_signal_strength_eddypro_test(df: DataFrame,
                                      signal_strength_col: str,
                                      var_col: str,
                                      method: str,
                                      threshold: int,
                                      idstr: str = None) -> Series:
    """Extract signal strength data and create a quality flag based on threshold comparison.

    Creates a quality flag by comparing signal strength values from EddyPro output against
    a user-defined threshold.

    Args:
        df: A dataframe that contains <signal_strength_col> and <var_col>.
        signal_strength_col: Name of signal strength column from EddyPro output.
        var_col: Name of the flux variable being evaluated. <var_col> is only used
            for naming the extracted flag.
        method: Threshold comparison method: 'discard below' or 'discard above'.
        threshold: Threshold value for quality assessment.
        idstr: An optional identifier string to append to the flag name.

    Returns:
        A series containing the quality flag, where 0=good values, 2=bad values.

    See Also:
        See examples/qaqc/eddyproflags.py for a complete working example.
    """
    idstr = validate_id_string(idstr=idstr)
    flagname_out = f'FLAG{idstr}_{var_col}_SIGNAL_STRENGTH_TEST'

    if signal_strength_col not in df.columns:
        raise f"The column {signal_strength_col} is not in data, please check."

    # Get original signal strength values and then
    # replace values with flag
    ss = df[signal_strength_col].copy()
    ss_flag = pd.Series(index=df.index, data=np.nan, name=flagname_out)
    if method == 'discard below':
        ss_flag[ss >= threshold] = 0
        ss_flag[ss < threshold] = 2
        signs_str = ['>=', '<']
    elif method == 'discard above':
        ss_flag[ss <= threshold] = 0
        ss_flag[ss > threshold] = 2
        signs_str = ['<=', '>']
    else:
        raise Exception(f"Error in {flag_signal_strength_eddypro_test.__name__}, "
                        f"the method {method} is unknown.")

    print(f"SIGNAL STRENGTH TEST: Generating new flag variable {flagname_out}, "
          f"newly calculated from output variable {signal_strength_col}, with "
          f"flag 0 (good values) where {signal_strength_col} {signs_str[0]} {threshold}, "
          f"flag 2 (bad values) where {signal_strength_col} {signs_str[1]} {threshold} ...")

    return ss_flag


def flag_steadiness_horizontal_wind_eddypro_test(df: DataFrame,
                                                 flux: str,
                                                 idstr: str = None) -> Series:
    """Extract wind steadiness flag from EddyPro output and convert to DIIVE format.

    Extracts the wind steadiness test flag from EddyPro FluxNet output and converts
    it to DIIVE standard format (0=good, 2=bad).

    From the EddyPro description:
        "This test assesses whether the along-wind and crosswind components of the wind vector undergo
        a systematic reduction (or increase) throughout the file. If the quadratic combination of such
        systematic variations is beyond the user-selected limit, the flux averaging period is hard-flagged
        for instationary horizontal wind (Vickers and Mahrt, 1997, Par. 6g)."

    Args:
        df: A dataframe containing EddyPro FluxNet output data with VM97_NSHW_HF column.
        flux: Name of the flux variable (used only for naming the output flag column).
        idstr: An optional identifier string to append to the flag name.

    Returns:
        A series containing the quality flag in DIIVE format, where 0=good values, 2=bad values.

    See Also:
        See examples/qaqc/eddyproflags.py for a complete working example.
    """
    idstr = validate_id_string(idstr=idstr)
    flagname_out = f"FLAG{idstr}_{flux}_VM97_NSHW_HF_TEST"
    nshw_flag = df['VM97_NSHW_HF'].copy()  # Name of the flag in EddyPro output file
    nshw_flag = nshw_flag.apply(pd.to_numeric, errors='coerce').astype(float)
    nshw_flag = nshw_flag.fillna(89)  # 9 = missing flag
    nshw_flag = nshw_flag.astype(str)
    nshw_flag = nshw_flag.str[int(1)]
    nshw_flag = nshw_flag.astype(float)
    nshw_flag = nshw_flag.replace(9, np.nan)
    nshw_flag = nshw_flag.replace(1, 2)  # Hard flag 1 corresponds to bad value
    nshw_flag.name = flagname_out

    print(f"STEADINESS OF HORIZONTAL WIND TEST: Generated new flag variable {flagname_out}, "
          f"values taken from output variable {nshw_flag.name}, with "
          f"flag 0 (good values) where test passed, "
          f"flag 2 (bad values) where test failed ...")

    return nshw_flag


def flag_angle_of_attack_eddypro_test(df: DataFrame,
                                      flux: str,
                                      idstr: str = None,
                                      application_dates: list or None = None) -> Series:
    """Extract angle of attack flag from EddyPro output and convert to DIIVE format.

    Extracts the angle of attack test flag from EddyPro FluxNet output and converts
    it to DIIVE standard format (0=good, 2=bad). The angle of attack test evaluates
    whether the wind vector relative to the sonic anemometer orientation is within
    acceptable limits.

    The EddyPro flag is stored as a 2-digit integer (e.g., 81), where the second
    digit contains the test result. Flag = 1 means the angle was too large (bad).

    Args:
        df: A dataframe containing EddyPro FluxNet output data with VM97_AOA_HF column.
        flux: Name of the flux variable (used only for naming the output flag column).
        idstr: An optional identifier string to append to the flag name.
        application_dates: Optional list of date ranges to restrict flag application.
            Format: [['2022-01-01', '2022-12-31'], ...] for selective time periods.

    Returns:
        A series containing the quality flag in DIIVE format, where 0=good values, 2=bad values.

    See Also:
        See examples/qaqc/eddyproflags.py for a complete working example.
    """
    flagname_out = f"FLAG{idstr}_{flux}_VM97_AOA_HF_TEST"
    aoa_flag = df['VM97_AOA_HF'].copy()  # Name of the flag in EddyPro output file
    aoa_flag = aoa_flag.apply(pd.to_numeric, errors='coerce').astype(float)
    aoa_flag = aoa_flag.fillna(89)  # 9 = missing flag
    aoa_flag = aoa_flag.astype(str)
    aoa_flag = aoa_flag.str[int(1)]
    aoa_flag = aoa_flag.astype(float)
    aoa_flag = aoa_flag.replace(9, np.nan)
    aoa_flag = aoa_flag.replace(1, 2)  # Hard flag 1 corresponds to bad value

    # Apply flag only during certain time periods
    if application_dates:
        aoa_flag = restrict_application(flag=aoa_flag,
                                        flagname="ANGLE OF ATTACK TEST",
                                        application_dates=application_dates,
                                        verbose=True,
                                        fill_value=np.nan)

    print(f"ANGLE OF ATTACK TEST: Generated new flag variable {flagname_out}, "
          f"values taken from output variable {aoa_flag.name}, with "
          f"flag 0 (good values) where test passed, "
          f"flag 2 (bad values) where test failed ...")

    aoa_flag.name = flagname_out
    return aoa_flag


def flags_vm97_eddypro_fluxnetfile_tests(
        df: DataFrame,
        flux: str,
        fluxbasevar: str,
        idstr: str = None,
        spikes: bool = True,
        amplitude: bool = False,
        dropout: bool = True,
        abslim: bool = False,
        skewkurt_hf: bool = False,
        skewkurt_sf: bool = False,
        discont_hf: bool = False,
        discont_sf: bool = False) -> DataFrame:
    """Extract VM97 (Vickers & Mahrt 1997) raw data quality test flags from EddyPro output.

    EddyPro performs statistical quality tests on the raw high-frequency eddy covariance
    data. These VM97 tests evaluate the quality and reliability of the raw measurements
    before flux calculation. EddyPro FluxNet files store multiple raw data tests in a
    single multi-digit integer (e.g., 80100010). This function extracts individual test
    results from each digit position and converts them to DIIVE standard format.

    The VM97 integer encodes 8 different quality tests in an 8-digit code:
    - Position 0: Always 8 (constant, no meaning)
    - Position 1: Spike detection (hard flag)
    - Position 2: Amplitude resolution (hard flag)
    - Position 3: Dropout detection (hard flag)
    - Position 4: Absolute limits (hard flag)
    - Position 5: Skewness/Kurtosis (hard flag)
    - Position 6: Skewness/Kurtosis (soft flag)
    - Position 7: Discontinuities (hard flag)
    - Position 8: Discontinuities (soft flag)

    Hard flags (_HF_) are converted from EddyPro format (1=bad) to DIIVE format (2=bad).
    Soft flags (_SF_) retain value 1 to indicate marginal/warning conditions.

    Args:
        df: A dataframe containing EddyPro FluxNet output with {fluxbasevar}_VM97_TEST column.
        flux: The flux variable being evaluated (e.g., 'FC' for carbon dioxide flux).
        fluxbasevar: The base variable used to calculate the flux (e.g., 'CO2' for FC flux).
        idstr: An optional identifier string to append to flag names.
        spikes: Extract spike detection test (position 1).
        amplitude: Extract amplitude resolution test (position 2).
        dropout: Extract dropout detection test (position 3).
        abslim: Extract absolute limits test (position 4).
        skewkurt_hf: Extract skewness/kurtosis hard flag test (position 5).
        skewkurt_sf: Extract skewness/kurtosis soft flag test (position 6).
        discont_hf: Extract discontinuities hard flag test (position 7).
        discont_sf: Extract discontinuities soft flag test (position 8).

    Returns:
        A dataframe containing selected quality flag columns in DIIVE format
        (0=good, 1=soft warning, 2=bad).

    See Also:
        See examples/qaqc/eddyproflags.py for a complete working example.

    References:
        https://www.licor.com/env/support/EddyPro/topics/despiking-raw-statistical-screening.html
    """
    idstr = validate_id_string(idstr=idstr)

    vm97 = df[f"{fluxbasevar}_VM97_TEST"].copy()
    vm97 = vm97.apply(pd.to_numeric, errors='coerce').astype(float)
    vm97 = vm97.fillna(899999999)  # 9 = flag is missing

    flagnames_out = {
        # '0': XXX,  # Index 0 is always the number `8`
        '1': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_SPIKE_HF_TEST",  # Spike detection, hard flag
        '2': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_AMPLITUDE_RESOLUTION_HF_TEST",  # Amplitude resolution, hard flag
        '3': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_DROPOUT_TEST",  # Drop-out, hard flag
        '4': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_ABSOLUTE_LIMITS_HF_TEST",  # Absolute limits, hard flag
        '5': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_SKEWKURT_HF_TEST",  # Skewness/kurtosis, hard flag
        '6': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_SKEWKURT_SF_TEST",  # Skewness/kurtosis, soft flag
        '7': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_DISCONTINUITIES_HF_TEST",  # Discontinuities, hard flag
        '8': f"FLAG{idstr}_{flux}_{fluxbasevar}_VM97_DISCONTINUITIES_SF_TEST"  # Discontinuities, soft flag
    }

    # Extract 8 individual flags from VM97 multi-flag integer
    flags_df = pd.DataFrame(index=df.index, data=vm97)
    for i, c in flagnames_out.items():
        _singleflag = vm97.astype(str)
        _singleflag = _singleflag.str[int(i)]
        _singleflag = _singleflag.astype(float)
        _singleflag = _singleflag.replace(9, np.nan)
        if '_HF_' in c:
            # Hard flag 1 corresponds to bad value, set to 2
            _singleflag = _singleflag.replace(1, 2)
        flags_df[c] = _singleflag

    # Select flags that are selected
    selected = []
    if spikes:
        selected.append('1')
    if amplitude:
        selected.append('2')
    if dropout:
        selected.append('3')
    if abslim:
        selected.append('4')
    if skewkurt_hf:
        selected.append('5')
    if skewkurt_sf:
        selected.append('6')
    if discont_hf:
        selected.append('7')
    if discont_sf:
        selected.append('8')

    # Make new dict that contains flags that we use later
    flagcols_used = {x: flagnames_out[x] for x in flagnames_out if x in selected}

    # Collect all required flags
    usedflags_df = pd.DataFrame(index=flags_df.index)
    for i, c in flagcols_used.items():
        usedflags_df[c] = flags_df[c].copy()

        print(f"RAW DATA TEST: Generated new flag variable {c}, "
              f"values taken from output variable {vm97.name} from position {i}, "
              f"based on {fluxbasevar}, with "
              f"flag 0 (good values) where test passed, "
              f"flag 2 (bad values) where test failed (for hard flags) or "
              f"flag 1 (ok values) where test failed (for soft flags) ...")

    return usedflags_df


def flag_fluxbasevar_completeness_eddypro_test(df: DataFrame, flux: str,
                                               fluxbasevar: str,
                                               thres_good: float = 0.99,
                                               thres_ok: float = 0.97,
                                               idstr: str = None) -> Series:
    """Check completeness of the variable that was used to calculate the respective flux.

    Default threshold values from Sabbatini et al. (2018).

    Example:
        `CO2` is the base variable that was used to calculate flux `FC`, the test is therefore
         run on `CO2`.

    Checks number of records of the relevant base variable available for each averaging interval
    and calculates completeness flag as follows (default):
    - `0` for files where >= 99% of base variable are available
    - `1` for files where >= 97% and < 99% of base variable are available
    - `2` for files where < 97% of base variable are available

    List of flux base variables and the corresponding fluxes:
    - `CO2`: used to calculate `FC`
    - `H2O`: used to calculate `FH2O`
    - `H2O`: used to calculate `LE`
    - `H2O`: used to calculate `ET`
    - `T_SONIC`: used to calculate `H`
    - `N2O`: used to calculate `FN2O`
    - `CH4`: used to calculate `FCH4`

    Args:
        df: A DataFrame containing EddyPro data from the _fluxnet_ file.
        flux: The name of the flux variable for which the completeness info is available in *df*.
        fluxbasevar: The name of the variable that was used to calculate *flux* in EddyPro.
        thres_good: The threshold for a good flag (default: 0.99, corresponds to 99%, meaning that
            99% of potential records of *gas* were available to calculate *flux*).
        thres_ok: The threshold for an ok flag (default: 0.97, corresponds to 97%).
        idstr: An optional identifier string to append to the flag name.

    Returns:
        A pandas Series containing the completeness flag variable.
    """

    idstr = validate_id_string(idstr=idstr)
    flagname_out = f'FLAG{idstr}_{flux}_COMPLETENESS_TEST'
    expected_n_records = df['EXPECT_NR'].copy()
    fluxbasevar_n_records = df[f'{fluxbasevar}_NR'].copy()
    fluxbasevar_n_records_perc = fluxbasevar_n_records.divide(expected_n_records)

    completeness_flag = Series(index=df.index, data=np.nan, name=flagname_out)
    completeness_flag[fluxbasevar_n_records_perc >= thres_good] = 0
    completeness_flag[(fluxbasevar_n_records_perc >= thres_ok) & (fluxbasevar_n_records_perc < thres_good)] = 1
    completeness_flag[fluxbasevar_n_records_perc < thres_ok] = 2

    print(
        f"FLUX BASE VARIABLE COMPLETENESS TEST: Generated new flag variable {flagname_out}, "
        f"newly calculated from variable {fluxbasevar}, with "
        f"flag 0 (good values) where available number of records for {fluxbasevar} >= {thres_good}, "
        f"flag 1 (ok values) >= {thres_ok} and < {thres_good}, "
        f"flag 2 (bad values) < {thres_ok}..."
    )

    return completeness_flag


def flag_spectral_correction_factor_eddypro_test(
        df: DataFrame,
        flux: str,
        thres_good: int = 2,
        thres_ok: int = 4,
        idstr: str = None):
    """
    Generates a spectral correction factor test flag based on results from EddyPro,
    categorizing data quality into good, ok, and bad. The flag is created as a new Series
    and values are based on the provided thresholds.

    Args:
        df (DataFrame): Input DataFrame containing columns for spectral correction factors.
        flux (str): Name of the flux variable whose spectral correction factor is tested.
        thres_good (int, optional): Threshold value below which the data quality is considered good.
            Default is 2.
        thres_ok (int, optional): Threshold value below which the data quality is considered ok,
            but above the `thres_good`. Default is 4.
        idstr (str, optional): Identifier string to customize the generated flag's variable name.
            If None, default naming will be applied.

    Returns:
        Series: A new pandas Series containing the flag values:
            - 0 for good values (below `thres_good`),
            - 1 for ok values (between `thres_good` and `thres_ok`),
            - 2 for bad values (equal to or above `thres_ok`).

    """
    idstr = validate_id_string(idstr=idstr)
    flagname_out = f'FLAG{idstr}_{flux}_SCF_TEST'
    scf = df[f'{flux}_SCF'].copy()
    scf_flag = Series(index=df.index, data=np.nan, name=flagname_out)
    scf_flag[scf < thres_good] = 0
    scf_flag[(scf >= thres_good) & (scf < thres_ok)] = 1
    scf_flag[scf >= thres_ok] = 2

    print(f"SPECTRAL CORRECTION FACTOR TEST: Generating new flag variable {scf_flag.name}, "
          f"newly calculated from output variable {scf.name}, with"
          f"flag 0 (good values) where {scf.name} < {thres_good}, "
          f"flag 1 (ok values) where {scf.name} >= {thres_good} and < {thres_ok}, "
          f"flag 2 (bad values) where {scf.name} >= {thres_ok}...")

    return scf_flag


def flag_ssitc_eddypro_test(df: DataFrame, flux: str, setflag_timeperiod: dict = None,
                            idstr: str = None) -> Series:
    """Create series based on the SSITC test flag variable from an EddyPro output file.

    SSITC = Steady State and Integral Turbulence Characteristics test.

    This method calls the external `flag_ssitc_eddypro_test` function, which
    evaluates the flux data based on steady state and integral turbulence
    criteria (Mauder & Foken, 2004). The resulting flag is added to the
    internal `_results` dataframe.

    Args:
        df: A DataFrame containing EddyPro data from the _fluxnet_ or _full_output_ file.
        flux: The name of the flux variable for which the SSITC test was performed. The name of the
            SSITC test variable will be detected based on this variable.
        setflag_timeperiod: Specifies time periods when the flag is set to given value.
            Example:
                Set flag 1 to value 2 between '2022-05-01' and '2023-09-30',
                and between '2024-04-02' and '2024-04-19' (dates inclusive):
                    {2: [
                            [1, '2022-05-01', '2023-09-30'],
                            [1, '2024-04-02', '2024-04-19']
                        ]
                    }
        idstr: An optional identifier string to append to the flag name.

    Returns:
        A pandas Series containing the new flag variable.
    """
    idstr = validate_id_string(idstr=idstr)
    flagname_out = f'FLAG{idstr}_{flux}_SSITC_TEST'
    flagname = f'{flux}_SSITC_TEST'
    ssitc_flag = Series(index=df.index, data=df[flagname], name=flagname_out)

    print(f"SSITC TEST: Generated new flag variable {flagname_out}, "
          f"values taken from output variable {flagname} ...")

    if setflag_timeperiod:
        for flagval, dates in setflag_timeperiod.items():
            for fromto in dates:
                _target = fromto[0]
                _from = fromto[1]
                _to = fromto[2]
                _locs = (ssitc_flag.index >= _from) & (ssitc_flag.index <= _to) & (ssitc_flag == _target)
                ssitc_flag[_locs] = flagval
                print(f"    Flag {flagname_out} with value {_target} was set to {flagval} "
                      f"between {_from} and {_to}")

    return ssitc_flag
