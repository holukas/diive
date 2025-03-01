"""
Quality flags that depend on EddyPro output files.
"""
import numpy as np
import pandas as pd
from diive.core.funcs.funcs import validate_id_string
from diive.pkgs.qaqc.flags import restrict_application
from pandas import DataFrame, Series


def flag_signal_strength_eddypro_test(df: DataFrame,
                                      signal_strength_col: str,
                                      var_col: str,
                                      method: str,
                                      threshold: int,
                                      idstr: str = None) -> Series:
    """Flag flux values where signal strength / AGC is not sufficient (too high or too low).

    Args:
        df: A dataframe that contains <signal_strength_col> and <var>.
        signal_strength_col: Name of signal strength or AGC variable.
        var_col: Name of the variable for which the flag is created.
        method: Can be 'discard above' or 'discard below' the <threshold>.
        threshold: Threshold to remove data points.
        idstr: An optional identifier string to append to the flag name.

    Returns:
        A series containing the test flag, where 0=good values, 2=bad values.
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
    """Create flag for steadiness of horizontal wind u from the sonic anemometer.

    From the EddyPro description:
        "This test assesses whether the along-wind and crosswind components of the wind vector undergo
        a systematic reduction (or increase) throughout the file. If the quadratic combination of such
        systematic variations is beyond the user-selected limit, the flux averaging period is hard-flagged
        for instationary horizontal wind (Vickers and Mahrt, 1997, Par. 6g)."

    - The flag looks the same in the _fluxnet_ and _full_output_ files, but has
    different names.
    - Flag = 1 means that the wind was not stationary.
    - This is a hard flag, meaning that in EddyPro results flag 1 = bad values.

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
    """Flag from EddyPro output files is an integer and looks like this, e.g.: 81.
    The integer contains angle-of-attack test results for the sonic anemometer.

    Flag = 1 means that the angle was too large.

    The flag looks the same in the _fluxnet_ and _full_output_ files, but have
    different names.

    -- 1 digit:
    attack_angle_hf	            8aa	                            80

    This is a hard flag:
    _HF_ = hard flag (flag 1 = bad values)

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


def flags_vm97_eddypro_fulloutputfile_tests(
        df: DataFrame,
        units: dict,
        flux: str,
        gas: str,
        idstr: str = None,
        spikes: bool = True,
        amplitude: bool = False,
        dropout: bool = True,
        abslim: bool = False,
        skewkurt_hf: bool = False,
        skewkurt_sf: bool = False,
        discont_hf: bool = False,
        discont_sf: bool = False) -> DataFrame:
    """Flags from EddyPro full_output files that contain results from quality tests
     on raw data, based on Vickers and Mahrt (1997).

    The flags are stored in an integer and looks like this, e.g.: 800011199.
    One integer contains *one test* for *multiple* gases. Each number except the
    first one corresponds to the test result of the respective flag for the
    variable given in the units.

    EddyPro outputs the raw data flags as 0 or 1, whereby 1 can correspond to
    bad data (if the selected flag is a hard flag, _hf) or to OK data (if the selected
    flag is a soft flag, _sf). If the flag is 9 then the test result is missing
    or not relevant, e.g. when no CH4 flux was calculated.

    This function considers all 8-digit VM97 flags:
    Flag name                   Units                           Flag results
    spikes_hf                   8u/v/w/ts/co2/h2o/ch4/none	    800000099
    amplitude_resolution_hf     8u/v/w/ts/co2/h2o/ch4/none	    800000099
    drop_out_hf	                8u/v/w/ts/co2/h2o/ch4/none	    800000099
    absolute_limits_hf	        8u/v/w/ts/co2/h2o/ch4/none	    800000199
    skewness_kurtosis_hf	    8u/v/w/ts/co2/h2o/ch4/none	    800000099
    skewness_kurtosis_sf	    8u/v/w/ts/co2/h2o/ch4/none	    800011199
    discontinuities_hf	        8u/v/w/ts/co2/h2o/ch4/none	    800000000
    discontinuities_sf	        8u/v/w/ts/co2/h2o/ch4/none	    800000000

    The last digit can be various gases. For example, if N2O flux was calculated
    then the last flag (none) will be n2o. The first number in the flag results
    is always 8.

    -- 4 digits:
    timelag_hf	                8co2/h2o/ch4/none	            81000
    timelag_sf	                8co2/h2o/ch4/none	            81100

    -- 1 digit:
    attack_angle_hf	            8aa	                            80
    non_steady_wind_hf	        8U	                            80


    """
    idstr = validate_id_string(idstr=idstr)

    used_flags = []
    if spikes:
        # Spike detection, hard flag
        used_flags.append('spikes_hf')
    if amplitude:
        # Amplitude resolution, hard flag
        used_flags.append('amplitude_resolution_hf')
    if dropout:
        # Drop-out, hard flag
        used_flags.append('drop_out_hf')
    if abslim:
        # Absolute limits, hard flag
        used_flags.append('absolute_limits_hf')
    if skewkurt_hf:
        # Skewness/kurtosis, hard flag
        used_flags.append('skewness_kurtosis_hf')
    if skewkurt_sf:
        # Skewness/kurtosis, soft flag
        used_flags.append('skewness_kurtosis_sf')
    if discont_hf:
        # Discontinuities, hard flag
        used_flags.append('discontinuities_hf')
    if discont_sf:
        # Discontinuities, soft flag
        used_flags.append('discontinuities_sf')

    allflags_df = df[used_flags].copy()
    allflags_df = allflags_df.fillna(899999999)  # Fill missing values, showing that all flags are missing (9)

    usedflags_df = pd.DataFrame(index=df.index)
    for _flag in allflags_df:
        this_flag = allflags_df[_flag].astype(str)  # Complete flag
        _units = units[_flag]  # Units string
        _units = _units.replace('8', '')  # Remove number 8 from units string (not needed, has no flag meaning)
        _units = _units.split('/')  # Divide units string
        gas_idx = _units.index(gas)  # Get index of var
        this_flag = this_flag.str.get(gas_idx)  # Extract element at the passed position, for all records
        this_flag = this_flag.astype(int)
        this_flag.loc[this_flag == 9] = np.nan
        if _flag.endswith("_hf"):
            this_flag.loc[this_flag == 1] = 2  # 2 = bad quality value
        flagname_out = f"FLAG{idstr}_{flux}_{gas}_VM97_{_flag}_TEST"
        usedflags_df[flagname_out] = this_flag

        print(f"RAW DATA TEST: Generated new flag variable {flagname_out}, "
              f"values taken from output variable {_flag} from position {gas_idx}, "
              f"based on {gas}, with "
              f"flag 0 (good values) where test passed, "
              f"flag 2 (bad values) where test failed (for hard flags) or "
              f"flag 1 (ok values) where test failed (for soft flags) ...")

    return usedflags_df


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
    """Flag from EddyPro fluxnet files is an integer and looks like this, e.g.: 801000100
    One integer contains *multiple tests* for *one* gas.

    There is one flag for each gas, which is different from the flag output in the
    EddyPro full output file (there, one integer describes *one test* and then contains
    flags for *multiple gases*).

    _HF_ = hard flag (flag 0 = good values, flag 1 = bad values) --> will be converted to 2 = bad values
    _SF_ = soft flag (flag 0 = good values, flag 1 = ok values) --> 1 remains 1 = ok values

    See also the official EddyPro documentation:
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


def _exception_full_output_scf(flux: str, gas: str):
    """Find name of scf variable.
    EddyPro is inconsistent in the _full_output_ file because
    it uses the base variable (e.g. co2) as part of the scf name
    for some fluxes, and the flux name (e.g. H) for others.
    """
    string = None
    if any(n in flux for n in ['H', 'Tau', 'LE']):
        string = flux
    elif any(n in flux for n in ['co2_flux', 'n2o_flux', 'ch4_flux', 'h2o_flux']):
        string = gas
    scfname = f'{string}_scf'
    return scfname


def flag_ssitc_eddypro_test(df: DataFrame, flux: str, setflag_timeperiod: dict = None,
                            idstr: str = None) -> Series:
    """Create series based on the SSITC test flag variable from an EddyPro output file.

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
