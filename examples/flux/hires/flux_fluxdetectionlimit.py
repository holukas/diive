"""
====================================
Flux Detection Limit Calculation
====================================

Calculate the minimum detectable flux based on measurement noise.

The flux detection limit (FDL) represents the minimum detectable flux for a given
time period based on the noise in the eddy covariance measurements. This is useful
for understanding measurement sensitivity and data quality.

Best for: Assessing measurement sensitivity and data quality in eddy covariance studies.
"""

# %%
# Create synthetic high-resolution eddy covariance data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generate synthetic wind components and scalar data with realistic turbulent structures.

import pandas as pd
import numpy as np
import diive as dv

# Create synthetic high-resolution data (10 Hz, 30 minutes = 18000 records)
n_records = 18000
np.random.seed(42)

# Wind components (m/s) - turbulent data with autocorrelation
# Create autocorrelated turbulence by filtering white noise
window_size = 20  # Smooth over ~2 seconds at 10 Hz
u_noise = np.random.normal(2.0, 0.5, n_records)
v_noise = np.random.normal(0.1, 0.3, n_records)
w_noise = np.random.normal(0.0, 0.2, n_records)

# Apply moving average to create smooth, continuous turbulent structures
u = np.convolve(u_noise, np.ones(window_size)/window_size, mode='same')
v = np.convolve(v_noise, np.ones(window_size)/window_size, mode='same')
w = np.convolve(w_noise, np.ones(window_size)/window_size, mode='same')

# Scalar (N2O in nmol mol-1) - strongly correlated with lagged w
# This creates a clear covariance peak at ~1 second lag (10 records at 10 Hz)
lag_records = 10  # 1 second lag at 10 Hz
w_lagged = np.roll(w, lag_records)
w_lagged[:lag_records] = w_lagged[lag_records]  # Fill initial lag

# Add smooth scalar field with strong correlation to lagged w
n2o_noise = np.random.normal(0, 1.5, n_records)
n2o_smooth = np.convolve(n2o_noise, np.ones(window_size)/window_size, mode='same')
n2o = 300 + n2o_smooth + 2.5 * w_lagged

# Sonic temperature (K) - with diurnal variation
ts = 290 + 5 * np.sin(np.linspace(0, 2*np.pi, n_records))
ts += np.random.normal(0, 0.1, n_records)

# Water vapor (mol/mol) - typical values
h2o = 0.01 + 0.002 * np.random.normal(0, 1, n_records)
h2o = np.clip(h2o, 0.005, 0.02)

# Pressure (Pa) - constant at sea level
pressure = np.full(n_records, 101325.0)

# Create DataFrame
df = pd.DataFrame({
    'u': u,
    'v': v,
    'w': w,
    'N2O': n2o,
    'Ts': ts,
    'H2O': h2o,
    'Pressure': pressure
})

print("=" * 80)
print("Synthetic High-Resolution Data")
print("=" * 80)
print(f"Data shape: {df.shape}")
print(f"\nWind statistics:")
print(f"  u (m/s): {u.mean():.2f} ± {u.std():.2f}")
print(f"  v (m/s): {v.mean():.2f} ± {v.std():.2f}")
print(f"  w (m/s): {w.mean():.2f} ± {w.std():.2f}")

# %%
# Calculate flux detection limit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Run the detection limit calculation based on noise in the covariance function.

fdl = dv.flux.FluxDetectionLimit(
    df=df,
    u_col='u',
    v_col='v',
    w_col='w',
    c_col='N2O',
    ts_col='Ts',
    h2o_col='H2O',
    press_col='Pressure',
    default_lag=1.0,  # seconds
    noise_range=20,  # seconds
    lag_range=[-180, 180],  # seconds (standard range for FDL calculation)
    lag_stepsize=10,  # records
    sampling_rate=10,  # Hz
    create_covariance_plot=True,
    title_covariance_plot='Flux Detection Limit Analysis'
)

fdl.run()
results = fdl.get_detection_limit()

# %%
# Examine flux detection limit results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Interpret the calculated detection limit and signal-to-noise ratio.

print(f"\n" + "=" * 80)
print("Flux Detection Limit Results")
print("=" * 80)

print(f"\nNoise and Detection Limit:")
print(f"  Noise RMSE: {results['flux_noise_rmse']:.6f} nmol m-2 s-1")
print(f"  Detection Limit (3×RMSE): {results['flux_detection_limit']:.6f} nmol m-2 s-1")

print(f"\nSignal Quality:")
print(f"  Signal at default lag: {results['flux_signal_at_default_lag']:.6f} nmol m-2 s-1")
print(f"  Signal at max covariance: {results['flux_signal_at_cov_max_lag']:.6f} nmol m-2 s-1")
print(f"  Max covariance shift: {results['cov_max_shift']} records")

print(f"\nSignal-to-Noise Metrics:")
print(f"  Signal-to-noise ratio: {results['signal_to_noise']:.2f}")
print(f"  Signal-to-detection-limit: {results['signal_to_detection_limit']:.2f}")

# %%
# Assess measurement quality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Quality assessment based on signal-to-noise ratio.

snr = results['signal_to_noise']
if snr > 10:
    quality = "Excellent - Signal is well above noise"
elif snr > 3:
    quality = "Good - Signal clearly detectable"
elif snr > 1:
    quality = "Marginal - Signal near detection limit"
else:
    quality = "Poor - Signal obscured by noise"

print(f"\n" + "=" * 80)
print("Data Quality Assessment")
print("=" * 80)
print(f"\nQuality: {quality}")
print(f"SNR: {snr:.2f}")

# Display covariance plot
try:
    fig_cov = fdl.get_fig_cov()
    fig_cov.show()
    print("[OK] Covariance plot generated")
except Exception as e:
    print(f"Plot display info: {type(e).__name__}")

print("\n[OK] Flux detection limit analysis complete.")
