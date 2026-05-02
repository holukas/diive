"""
Examples for flux variable utilities and base variable detection.

Detects which base variable (e.g., CO2, H2O) was used to calculate
a given flux variable (e.g., FC, FH2O, LE). Useful for understanding
measurement nomenclature in eddy covariance data files.

Run this script to see flux variable examples:
    python examples/flux/common.py

See Also
--------
diive.pkgs.flux.common : Flux variable definitions and detection functions.
"""
import diive as dv


def example_detect_fluxbasevar():
    """Detect base variables from flux variable names.

    Demonstrates flux variable detection by showing which base variables
    (measured quantities) are used to calculate different flux variables.
    This is useful when working with eddy covariance data files to understand
    the relationships between measured scalars and calculated fluxes.
    """
    from diive.pkgs.flux.common import detect_fluxbasevar, fluxbasevars_fluxnetfile

    print("=" * 80)
    print("Example: Flux Variable Base Detection")
    print("=" * 80)

    print(f"\nAvailable flux variables in FLUXNET format:")
    print(f"{list(fluxbasevars_fluxnetfile.keys())}")

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

    print(f"\n" + "=" * 80)
    print("Use Case: Measurement Documentation")
    print("=" * 80)

    print(f"\nWhen processing eddy covariance data:")
    print(f"  - FC is CO2 flux, calculated from CO2 measurements")
    print(f"  - FH2O is H2O flux, calculated from H2O measurements")
    print(f"  - LE is latent energy flux, also from H2O measurements")
    print(f"  - H is sensible heat flux, calculated from sonic temperature")
    print(f"  - FN2O is N2O flux, calculated from N2O measurements")
    print(f"  - FCH4 is CH4 flux, calculated from CH4 measurements")


if __name__ == '__main__':
    print("=" * 80)
    print("Flux Variable Examples: Base Variable Detection")
    print("=" * 80)
    print()

    example_detect_fluxbasevar()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
