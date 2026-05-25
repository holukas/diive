"""
================================================
Flux Processing Chain - Quick Exploratory Mode
================================================

Simplified flux processing workflow with default settings for rapid data checking.
Processes raw EddyPro files through quality control and outlier detection to produce
data at processing levels L2-L3.3 (USTAR filtering for NEE).

Best for: Quick exploratory analysis, rapid data checking, initial quality assessment.
See: fluxprocessingchain.py for the comprehensive multi-level workflow example.
"""

# %%
# Quick flux processing with default settings
# =============================================
#
# The QuickFluxProcessingChain class simplifies flux processing by using
# extensive defaults, requiring minimal user configuration. This is ideal for
# exploratory data checking and rapid quality assessment. For detailed
# multi-level processing with full parameter control, see fluxprocessingchain.py.

import diive as dv
from diive.pkgs.flux.fluxprocessingchain import QuickFluxProcessingChain

# %%
# Configure site and processing parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Set basic site information (coordinates, UTC offset) and quality thresholds.
# The QuickFluxProcessingChain uses sensible defaults for all other parameters.

# Flux variables to process
FLUXVARS = ['FC', 'LE', 'H']  # CO2, latent heat, sensible heat

# Directory containing EddyPro output files
SOURCEDIRS = [r"diive\configs\exampledata\EDDYPRO-FLUXNET-CSV-30MIN_multiple"]

# Site metadata
SITE_LAT = 47.41887   # Latitude (CH-DAV example site)
SITE_LON = 8.491318   # Longitude (CH-DAV example site)
UTC_OFFSET = 1        # UTC+01:00 (Central European Time)

# Quality control thresholds
NIGHTTIME_THRESHOLD = 20        # W m-2, below this is nighttime
DAYTIME_ACCEPT_QCF_BELOW = 2    # Accept data quality flag < 2 during day
NIGHTTIME_ACCEPT_QCF_BELOW = 2  # Accept data quality flag < 2 at night

# Signal strength testing (optional)
TEST_SIGNAL_STRENGTH = True
TEST_SIGNAL_STRENGTH_COL = 'CUSTOM_AGC_MEAN'
TEST_SIGNAL_STRENGTH_METHOD = 'discard above'
TEST_SIGNAL_STRENGTH_THRESHOLD = 90

# %%
# Initialize and run quick processing chain
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create QuickFluxProcessingChain instance with the configured parameters.
# The processing chain automatically handles:
# - Level 2 (L2): Quality flag expansion
# - Level 3.1 (L3.1): Storage correction
# - Level 3.2 (L3.2): Outlier detection
# - Level 3.3 (L3.3): USTAR filtering for CO2/CH4 fluxes

qfpc = QuickFluxProcessingChain(
    fluxvars=FLUXVARS,
    sourcedirs=SOURCEDIRS,
    site_lat=SITE_LAT,
    site_lon=SITE_LON,
    utc_offset=UTC_OFFSET,
    nighttime_threshold=NIGHTTIME_THRESHOLD,
    daytime_accept_qcf_below=DAYTIME_ACCEPT_QCF_BELOW,
    nighttime_accept_qcf_below=NIGHTTIME_ACCEPT_QCF_BELOW,
    test_signal_strength=TEST_SIGNAL_STRENGTH,
    test_signal_strength_col=TEST_SIGNAL_STRENGTH_COL,
    test_signal_strength_method=TEST_SIGNAL_STRENGTH_METHOD,
    test_signal_strength_threshold=TEST_SIGNAL_STRENGTH_THRESHOLD
)

# %%
# Inspect processed results
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The processing chain returns a DataFrame with 38+ columns containing:
# - Raw input flux values
# - Quality flags at each processing level
# - Processed flux data at L3.3
# - USTAR friction velocity
# - Potential radiation and day/night classification

results_df = qfpc.fpc.fpc_df

print(f"Processing complete: {len(results_df)} records")
print(f"\nOutput columns: {results_df.shape[1]}")
print(f"Time period: {results_df.index.min()} to {results_df.index.max()}")

# Show sample of processed data
print("\nFirst few records of processed fluxes:")
print(results_df[[col for col in results_df.columns if 'L3.3' in col]].head())

print("\n✓ Quick flux processing complete.")
