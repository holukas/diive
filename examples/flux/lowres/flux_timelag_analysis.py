"""
============================
Time Lag Analysis for EC Data
============================

Analyze and visualize time lags in eddy covariance flux measurements.

This example demonstrates how to detect optimal time lags for gas concentration
measurements (CO2, H2O) relative to wind measurements. Time lag detection is
critical for accurate flux computation in eddy covariance systems.

The workflow includes:
- Histogram-based lag distribution analysis
- Gradient-based peak range detection
- EddyPro-compatible range adjustment (0.05s discrete steps)
- Multi-panel visualization showing overview and zoomed perspectives
"""

# %%
# Load data and initialize analyzer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Load your eddy covariance data containing time lag columns (e.g., CO2_TLAG_ACTUAL).
# Initialize TimeLagAnalysis with configuration parameters.

# Load data
from diive.configs.exampledata import load_exampledata_parquet_tlag_vars_level0
from diive.flux.lowres.timelag_analysis import TimeLagAnalysis

# Expected columns: CO2_TLAG_ACTUAL, H2O_TLAG_ACTUAL, etc.
# Data should have datetime index

df = load_exampledata_parquet_tlag_vars_level0()
# [print(c) for c in df.columns if "TLAG" in c]

# Initialize analysis with configuration
# Adjust these parameters based on your site and instrument setup
analysis = TimeLagAnalysis(
    df=df,
    ignore_fringe_bins=[5, 10],  # Exclude edge bins with accumulated non-physical lags
    lag_window_min=0.05,  # Lower bound of acceptable lag range (seconds)
    lag_window_max=1.00,  # Upper bound of acceptable lag range (seconds)
    histogram_startbin=0,  # First histogram bin to display
    histogram_endbin=10,  # Last histogram bin to display
    gradient_threshold=0.15,  # Edge detection sensitivity (lower = stricter)
    zoom_margin=[0.5, 0.8]  # Zoom offsets: [before_peak, after_peak] seconds
)

# %%
# Analyze individual gas
# ^^^^^^^^^^^^^^^^^^^^^
#
# Perform time lag analysis for a specific gas species.

# Analyze CO2 time lags
co2_results = analysis.analyze_gas(gas='CO2')

print(f"CO2 Analysis Results:")
print(f"  Peak lag: {co2_results['peak']:.3f}s")
print(f"  Detected range: {co2_results['peak_min']:.3f}–{co2_results['peak_max']:.3f}s")
print(f"  EddyPro input: {co2_results['eddypro_min']:.3f}–{co2_results['eddypro_max']:.3f}s")
print(f"  Data span: {co2_results['first_date']} to {co2_results['last_date']}")

# %%
# Create visualization
# ^^^^^^^^^^^^^^^^^^^^
#
# Generate 4-panel figure showing overview and zoomed perspectives.

fig = analysis.plot_gas(
    gas='CO2',
    outdir=None,                                # Do not save figure to disk
    figsize=(18, 9),                            # Figure dimensions
    show=True                                   # Display figure
)

