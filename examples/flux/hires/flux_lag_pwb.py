"""
=================================================
Time Lag Detection (PreWhiteningBootstrap / PWB)
=================================================

Detect time lag between a scalar and vertical wind using the pre-whitening
with block-bootstrap (PWB) cross-correlation procedure.

Unlike covariance maximisation (MaxCovariance), PWB is robust for low-magnitude
fluxes such as CH4 and N2O, where the cross-covariance function often lacks a
distinct peak. Pre-whitening removes serial correlation that inflates spurious
CCF peaks. Block-bootstrapping quantifies detection uncertainty so unreliable
lags can be identified and replaced.

Reference: Vitale D et al. (2024), Environmental and Ecological Statistics 31:219-244.
doi:10.1007/s10651-024-00615-9
"""

# %%
# Create synthetic high-flux data with known lag
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Simulate 30 minutes of 20 Hz data (36000 records) where the scalar has a
# known lag of 1.5 s relative to the vertical wind component. High correlation
# between wind and scalar mimics a moderate/high flux situation (e.g. CO2).

import numpy as np
import pandas as pd

import diive as dv

np.random.seed(42)

hz = 20  # acquisition frequency (Hz)
n_records = hz * 60 * 3  # 3 minutes (10% of a 30-min averaging period)
lag_true_s = 1.5  # known lag in seconds
lag_true_records = int(lag_true_s * hz)  # 30 records

# AR(1) turbulent wind: phi = 0.8 gives realistic autocorrelation
phi = 0.8
w = np.zeros(n_records)
for t in range(1, n_records):
    w[t] = phi * w[t - 1] + np.random.normal(0, 0.3)

# Scalar: strongly correlated with lagged wind (high flux, easy case)
s_noise = np.random.normal(0, 0.5, n_records)
s = np.roll(w, lag_true_records) * 5 + s_noise
s[:lag_true_records] = s[lag_true_records]  # fill initial wrap-around

df_highflux = pd.DataFrame({'W': w, 'CO2': s})

print("=" * 70)
print("Synthetic data: HIGH-FLUX case")
print("=" * 70)
print(f"  Acquisition frequency : {hz} Hz")
print(f"  Records               : {n_records}")
print(f"  Known lag             : {lag_true_s} s ({lag_true_records} records)")
print(f"  W correlation with CO2: {np.corrcoef(w, np.roll(s, -lag_true_records))[0, 1]:.3f}")

# %%
# Run PWB on high-flux data
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With a strong, distinct cross-correlation peak, the mode over bootstrap
# samples should agree closely with the true lag and the HDI range should
# be narrow (< 0.5 s), indicating a reliable result.

pwb_high = dv.PreWhiteningBootstrap(
    df=df_highflux,
    var_w='W',
    var_scalar='CO2',
    hz=hz,
    lag_max_s=10.0,  # search window: +/- 10 s
    n_bootstrap=9,  # N_B bootstrap samples (99 in production)
    block_length_s=20.0,  # block length L = 20 s (paper default)
    segment_name='high-flux'
)

pwb_high.run()
res_high = pwb_high.results

print("\n" + "=" * 70)
print("PWB results: HIGH-FLUX case")
print("=" * 70)
print(f"  AR order selected     : {res_high['ar_order']}")
print(f"  Detected lag          : {res_high['tlag_s']:.3f} s  (true: {lag_true_s} s)")
print(f"  95% HDI               : [{res_high['hdi_lo_s']:.3f}, {res_high['hdi_hi_s']:.3f}] s")
print(f"  HDI range             : {res_high['hdi_range_s']:.3f} s")
print(f"  Reliable (HDI < 0.5s) : {res_high['is_reliable']}")

# %%
# Create synthetic low-flux data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Simulate near-zero flux: scalar is dominated by noise, lag signal is very
# weak (low correlation). This is the typical situation for N2O or CH4 at
# ecosystem equilibrium, where CM-based methods often produce mirroring errors.

np.random.seed(7)
s_lowflux = np.roll(w, lag_true_records) * 0.05 + np.random.normal(0, 1.0, n_records)
s_lowflux[:lag_true_records] = s_lowflux[lag_true_records]

df_lowflux = pd.DataFrame({'W': w, 'N2O': s_lowflux})

print("\n" + "=" * 70)
print("Synthetic data: LOW-FLUX case")
print("=" * 70)
print(f"  Known lag             : {lag_true_s} s ({lag_true_records} records)")
print(f"  W correlation with N2O: {np.corrcoef(w, np.roll(s_lowflux, -lag_true_records))[0, 1]:.4f}")

# %%
# Run PWB on low-flux data
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# With a weak cross-correlation signal, the HDI range should be wide (> 0.5 s),
# flagging the detected lag as unreliable. This triggers the PWB^OPT strategy:
# replace the unreliable lag with the closest preceding reliable optimal lag.

pwb_low = dv.PreWhiteningBootstrap(
    df=df_lowflux,
    var_w='W',
    var_scalar='N2O',
    hz=hz,
    lag_max_s=10.0,
    n_bootstrap=9,  # N_B bootstrap samples (99 in production)
    block_length_s=20.0,
    segment_name='low-flux'
)

pwb_low.run()
res_low = pwb_low.results

print("\n" + "=" * 70)
print("PWB results: LOW-FLUX case")
print("=" * 70)
print(f"  AR order selected     : {res_low['ar_order']}")
print(f"  Detected lag          : {res_low['tlag_s']:.3f} s  (true: {lag_true_s} s)")
print(f"  95% HDI               : [{res_low['hdi_lo_s']:.3f}, {res_low['hdi_hi_s']:.3f}] s")
print(f"  HDI range             : {res_low['hdi_range_s']:.3f} s")
print(f"  Reliable (HDI < 0.5s) : {res_low['is_reliable']}")

# %%
# Visualize PWB results
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The left panel shows the bootstrap lag distribution with the mode (detected lag)
# and the 95% HDI shaded. The right panel shows the mean smoothed CCF across
# bootstrap samples with conventional significance lines (+-1.96/sqrt(N_B)).

fig_high = pwb_high.plot(
    title='PWB time lag detection — high-flux (CO2)',
    showplot=True
)

fig_low = pwb_low.plot(
    title='PWB time lag detection — low-flux (N2O)',
    showplot=True
)

print("\n[OK] PWB time lag detection complete.")
