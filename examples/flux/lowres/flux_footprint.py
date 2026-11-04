"""
===================================================
Flux Footprint (FFP): Single Record and Climatology
===================================================

Calculate the flux footprint after Kljun et al. (2015) with ``ffp_footprint`` (one
record) and ``ffp_climatology`` (many records). Both give the same results as the
authors' Python code.

The footprint tells where the measured flux comes from. The source-area contour
lines enclose 10 %, 20 %, ... 80 % of the footprint, starting at its peak. The
example shows the inputs of the model, how stability moves the footprint, and the
footprint climatology of one month of measurements.

Best for: Checking which area a flux tower sees, and how far that area reaches
under different conditions.

Reference: Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple
two-dimensional parameterisation for Flux Footprint Prediction (FFP). Geoscientific
Model Development, 8(11), 3695-3713. https://doi.org/10.5194/gmd-8-3695-2015
"""

# %%
# Footprint of a single record
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The model needs, per record:
#
# - ``zm``: measurement height above the displacement height, z - d (m)
# - ``h``: boundary-layer height (m)
# - ``ol``: Obukhov length (m)
# - ``sigmav``: standard deviation of the lateral wind velocity (m s-1)
# - ``ustar``: friction velocity (m s-1)
# - ``z0`` (roughness length, m) or ``umean`` (mean wind speed at ``zm``, m s-1)
#
# ``wind_dir`` rotates the footprint into the wind direction, so x points east and
# y north. ``rs`` selects the source areas of the contour lines. The result holds
# the crosswind-integrated footprint (``x_ci``, ``f_ci``), the 2-D footprint
# (``x_2d``, ``y_2d``, ``f_2d``) and the contour lines (``xr``, ``yr``).

import matplotlib.pyplot as plt
import numpy as np

import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN


def reach(result):
    """Farthest distance (m) of each contour line from the tower."""
    return {f"{r:.0%}": (None if xr is None else round(float(np.hypot(xr, yr).max()), 1))
            for r, xr, yr in zip(result.rs, result.xr, result.yr)}


single = dv.flux.ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4,
                               z0=0.15, wind_dir=220.)

print(f"Peak of the footprint: {single.x_ci_max:.0f} m upwind of the tower")
print("Farthest distance of each contour line (m):")
print(reach(single))

dv.plotting.FootprintPlot(single).plot()
plt.show(block=False)

# %%
# Stability moves the footprint
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same record under different stability. Under unstable conditions (``ol`` < 0)
# strong mixing brings the flux to the sensor from close by. Under stable
# conditions (``ol`` > 0) the footprint peaks farther away and has a long tail.
# The crosswind-integrated footprint ``f_ci`` shows this along the wind.

colors = ['#F44336', '#FF9800', '#455A64', '#2196F3', '#3F51B5']
fig, ax = plt.subplots(figsize=(8, 4.5))
for ol, color in zip([-10., -100., -10000., 200., 50.], colors):
    fp = dv.flux.ffp_footprint(zm=20., h=1500., ol=ol, sigmav=0.6, ustar=0.4, z0=0.15,
                               rs=0.8)
    name = 'near neutral' if abs(ol) > 5000 else ('unstable' if ol < 0 else 'stable')
    ax.plot(fp.x_ci, fp.f_ci, color=color, lw=2,
            label=f"L = {ol:g} m ({name}): peak {fp.x_ci_max:.0f} m, 80 % reaches {reach(fp)['80%']:.0f} m")
ax.set_xlim(0, 1500)
ax.set_xlabel("Distance upwind of the tower (m)")
ax.set_ylabel("Crosswind-integrated footprint (m$^{-1}$)")
ax.legend(frameon=False)
fig.tight_layout()
plt.show(block=False)

# %%
# Footprint climatology of one month
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Half-hourly EddyPro output of one month (July) from a grassland site. It holds all
# turbulence inputs: ``USTAR``, ``MO_LENGTH``, ``V_SIGMA``, ``WD`` and ``WS``. The
# sonic anemometer is 1.3 m above ground and the displacement height is 0.27 m, so
# ``zm`` is 1.03 m. No bundled dataset of a full year has all inputs.
#
# The boundary-layer height is not measured here and is set to 1000 m for every
# record. It matters mainly under unstable conditions, so use measured or
# modelled values for real studies. The mean wind speed replaces the roughness
# length.
#
# The domain must hold the largest contour line. A contour line that reaches the
# edge of the domain is returned as None. ``n_jobs`` sets the number of worker
# processes (default: all cores), and the result does not depend on it.

df, _ = load_exampledata_EDDYPRO_FLUXNET_CSV_30MIN()
zm = 1.3 - 0.27

clim = dv.flux.ffp_climatology(zm=zm, h=np.full(len(df), 1000.), ol=df['MO_LENGTH'],
                               sigmav=df['V_SIGMA'], ustar=df['USTAR'], wind_dir=df['WD'],
                               umean=df['WS'], domain=[-60., 60., -60., 60.], dx=0.25)

# Records with missing input or input outside the range of the model are skipped,
# here mostly records with u* below 0.1 m s-1.
print(f"{clim.n} of {len(df)} records in the climatology")
print(f"Records with u* <= 0.1 m s-1: {(df['USTAR'] <= 0.1).sum()}")
print("Farthest distance of each contour line (m):")
print(reach(clim))

dv.plotting.FootprintPlot(clim).plot()
plt.show()

# sphinx_gallery_thumbnail_number = 3
