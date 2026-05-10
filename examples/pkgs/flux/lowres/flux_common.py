"""
============================
Flux Variable Base Detection
============================

Detect which base variable (CO2, H2O, etc.) was used to calculate a given flux variable.

Demonstrates flux variable detection by showing which base variables
(measured quantities) are used to calculate different flux variables.
This is useful when working with eddy covariance data files to understand
the relationships between measured scalars and calculated fluxes.

Best for: Understanding FLUXNET variable nomenclature and relationships.
"""

# %%
# Detect base variables from flux variable names
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Demonstrates flux variable detection by querying which base variables
# (measured quantities) are used to calculate different flux variables.

from diive.pkgs.flux.lowres.common import detect_fluxbasevar, fluxbasevars_fluxnetfile

print("=" * 80)
print("Available flux variables in FLUXNET format:")
print("=" * 80)
print(f"\n{list(fluxbasevars_fluxnetfile.keys())}")

# %%
# Base variable detection for common flux variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Test various flux variables to determine which base variables
# (measured scalars) are used in their calculation.

print(f"\n" + "=" * 80)
print("Base Variable Detection")
print("=" * 80)

# Test various flux variables
test_fluxes = ['FC', 'FH2O', 'LE', 'H', 'FN2O', 'FCH4']

print(f"\nDetecting base variables for each flux:")
for flux_var in test_fluxes:
    if flux_var in fluxbasevars_fluxnetfile:
        base_var = detect_fluxbasevar(flux_var)
        print(f"  {flux_var:<8} -> {base_var}")
    else:
        print(f"  {flux_var:<8} -> [not found]")

# %%
# Use case: Measurement documentation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When processing eddy covariance data, understanding the relationships
# between flux variables and their base variables is essential.

print(f"\n" + "=" * 80)
print("Measurement Relationships")
print("=" * 80)

print(f"\nWhen processing eddy covariance data:")
print(f"  - FC is CO2 flux, calculated from CO2 measurements")
print(f"  - FH2O is H2O flux, calculated from H2O measurements")
print(f"  - LE is latent energy flux, also from H2O measurements")
print(f"  - H is sensible heat flux, calculated from sonic temperature")
print(f"  - FN2O is N2O flux, calculated from N2O measurements")
print(f"  - FCH4 is CH4 flux, calculated from CH4 measurements")

print(f"\n[OK] Flux variable detection complete.")
