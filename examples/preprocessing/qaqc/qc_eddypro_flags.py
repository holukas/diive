"""
===================================
EddyPro Quality Flags
===================================

Extract and convert EddyPro quality flags to DIIVE standard format.
Understand signal quality, test failures, and data completeness.
"""

# %%
# Load EddyPro FluxNet data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Read example EddyPro output file containing quality test flags.

from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN

df, metadata = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()

print("EddyPro FluxNet data loaded:")
print(f"  Records: {len(df)}")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Columns: {df.shape[1]}")

# %%
# Signal strength test
# ^^^^^^^^^^^^^^^^^^^^^
#
# Identify measurements with poor sensor signal quality.
# Low signal indicates dust, condensation, or optical drift.

from diive.pkgs.preprocessing.qaqc import flag_signal_strength_eddypro_test

flag_signal = flag_signal_strength_eddypro_test(
    df=df,
    signal_strength_col='CUSTOM_SIGNAL_STRENGTH_IRGA72_MEAN',
    var_col='FC',
    method='discard below',
    threshold=99,
    idstr='_L41'
)

n_retained = (flag_signal == 0).sum()
n_discarded = (flag_signal == 2).sum()

print("\nSignal Strength Quality Test:")
print(f"  Retained: {n_retained} records ({100*n_retained/(n_retained+n_discarded):.1f}%)")
print(f"  Discarded: {n_discarded} records ({100*n_discarded/(n_retained+n_discarded):.1f}%)")

# %%
# Wind steadiness test
# ^^^^^^^^^^^^^^^^^^^^
#
# Evaluate horizontal wind stability throughout measurement period.
# Non-stationary wind indicates poor measurement conditions.

from diive.pkgs.preprocessing.qaqc import flag_steadiness_horizontal_wind_eddypro_test

flag_wind = flag_steadiness_horizontal_wind_eddypro_test(
    df=df,
    flux='FC',
    idstr='_L41'
)

print("\nWind Steadiness Quality Test:")
print(f"  Steady: {(flag_wind == 0).sum()}")
print(f"  Non-steady: {(flag_wind == 2).sum()}")

# %%
# Angle of attack test
# ^^^^^^^^^^^^^^^^^^^^
#
# Check if wind approaches anemometer at acceptable angle.
# Large angles degrade measurement accuracy.

from diive.pkgs.preprocessing.qaqc import flag_angle_of_attack_eddypro_test

flag_aoa = flag_angle_of_attack_eddypro_test(
    df=df,
    flux='FC',
    idstr='_L41'
)

print("\nAngle of Attack Quality Test:")
print(f"  Acceptable: {(flag_aoa == 0).sum()}")
print(f"  Poor angle: {(flag_aoa == 2).sum()}")

# %%
# Vickers & Mahrt (1997) raw data tests
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Extract 8 individual quality tests from EddyPro VM97 codes.
# Tests evaluate stationarity, amplitude, dropout, limits, etc.

from diive.pkgs.preprocessing.qaqc import flags_vm97_eddypro_fluxnetfile_tests

flags_vm97 = flags_vm97_eddypro_fluxnetfile_tests(
    df=df,
    flux='FC',
    fluxbasevar='CO2',
    idstr='_L41',
    spikes=True,
    amplitude=True,
    dropout=True,
    abslim=True,
    skewkurt_hf=True,
    skewkurt_sf=True,
    discont_hf=True,
    discont_sf=True
)

print("\nVM97 Raw Data Quality Tests (8 individual tests):")
print("-" * 60)

test_info = [
    ('Spike detection', 'SPIKE_HF'),
    ('Amplitude resolution', 'AMPLITUDE_RESOLUTION_HF'),
    ('Dropout detection', 'DROPOUT_TEST'),
    ('Absolute limits', 'ABSOLUTE_LIMITS_HF'),
    ('Skewness/Kurtosis (hard)', 'SKEWKURT_HF'),
    ('Skewness/Kurtosis (soft)', 'SKEWKURT_SF'),
    ('Discontinuities (hard)', 'DISCONTINUITIES_HF'),
    ('Discontinuities (soft)', 'DISCONTINUITIES_SF'),
]

for test_name, test_suffix in test_info:
    flag_col = [col for col in flags_vm97.columns if test_suffix in col][0]
    flag_series = flags_vm97[flag_col]
    n_pass = (flag_series == 0).sum()
    n_fail = (flag_series == 2).sum()
    n_warn = (flag_series == 1).sum()
    print(f"{test_name:30s}  Pass: {n_pass:5d}  Fail: {n_fail:5d}  Warn: {n_warn:5d}")

# %%
# Base variable completeness
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Evaluate data availability for flux calculation.
# CO2 completeness determines CO2 flux (FC) reliability.

from diive.pkgs.preprocessing.qaqc import flag_fluxbasevar_completeness_eddypro_test

flag_complete = flag_fluxbasevar_completeness_eddypro_test(
    df=df,
    flux='FC',
    fluxbasevar='CO2',
    thres_good=0.99,
    thres_ok=0.97,
    idstr='_L41'
)

n_good = (flag_complete == 0).sum()
n_ok = (flag_complete == 1).sum()
n_bad = (flag_complete == 2).sum()

print("\nBase Variable (CO2) Completeness:")
print(f"  Good (≥99% complete): {n_good}")
print(f"  Ok (97-99% complete): {n_ok}")
print(f"  Bad (<97% complete): {n_bad}")

# %%
# Summary
# ^^^^^^^
#
# EddyPro quality flags help identify measurement reliability issues.
# Use these flags to establish data quality thresholds for your analysis.

print("\n" + "="*60)
print("Key Takeaways")
print("="*60)
print("""
- Signal strength: Optical/sensor quality indicator
- Steadiness: Measurement stationarity check
- Angle of attack: Wind approach angle
- VM97 tests: Statistical tests on raw data (8 types)
- Completeness: Data availability for flux calculation

Combine these into overall quality flag for robust data filtering.
""")
