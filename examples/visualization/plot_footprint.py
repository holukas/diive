"""
======================================
Flux Footprint: Single and Climatology
======================================

Map of the flux footprint after Kljun et al. (2015). ``FootprintPlot`` takes the
result of ``ffp_footprint`` (one record) or ``ffp_climatology`` (many records)
and draws the footprint field, the source-area contour lines, the tower, the
footprint peak, range rings and a north arrow. The subtitle gives the key
numbers: the number of footprints, the peak distance and how far the 80 %
source area reaches.

The example first plots a single footprint, then the footprint climatology of
ten days of measurements at a grassland site, and finally compares unstable and
stable conditions side by side.

Reference: Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple
two-dimensional parameterisation for Flux Footprint Prediction (FFP). Geoscientific
Model Development, 8(11), 3695-3713. https://doi.org/10.5194/gmd-8-3695-2015
"""

# %%
# Single footprint
# ^^^^^^^^^^^^^^^^
#
# One half-hour over a forest: measurement height above the displacement height
# ``zm`` 20 m, boundary-layer height ``h`` 1500 m, slightly unstable (Obukhov
# length ``ol`` -50 m) and wind from the southwest (220 degrees). With a wind
# direction the footprint is rotated into geographic coordinates, so x points
# east and y north.
#
# Without ``ax``, ``plot()`` adds a side panel for a single footprint. The upper
# part shows the crosswind-integrated footprint with its peak, the lower part its
# cumulative sum along the wind. The two 80 % numbers measure different things.
# The map gives the farthest point of the 80 % contour line. The side panel gives
# the distance upwind within which 80 % of the crosswind-integrated footprint lies.

import matplotlib.pyplot as plt
import numpy as np

import diive as dv
from diive.configs.exampledata import load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN
from diive.flux.lowres.footprint import ffp_climatology, ffp_footprint

single = ffp_footprint(zm=20., h=1500., ol=-50., sigmav=0.6, ustar=0.4, z0=0.15,
                       wind_dir=220.)

fp_single = dv.plotting.FootprintPlot(single)
fp_single.plot()
plt.show(block=False)

print(f"Peak: {fp_single.peak_distance:.0f} m, "
      f"80 % contour reaches {fp_single.source_area_distance:.0f} m, "
      f"cumulative 80 % at {fp_single.cumulative_distance:.0f} m")

# %%
# Footprint climatology from measured data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The EddyPro full output example holds ten days of half-hourly data from a
# grassland site. It contains all turbulence inputs of the model: friction
# velocity ``u*``, Obukhov length ``L``, the variance of the crosswind component
# ``v_var`` (the model needs its square root), wind direction and wind speed.
# The measurement height above the displacement height follows from the
# stability parameter ``(z-d)/L`` times ``L``.
#
# The boundary-layer height is not measured at the site. Here it is set to
# 1000 m for every record. It mainly affects the footprint under unstable
# conditions, so use a measured or modelled value for real studies.
#
# The mean wind speed (``umean``) replaces the roughness length. Records with
# input outside the range of the model, for example ``u*`` below 0.1 m s-1,
# are skipped, and the subtitle reports how many footprints remain.

df, meta = load_exampledata_EDDYPRO_FULL_OUTPUT_CSV_30MIN()

zm = float((df['(z-d)/L'] * df['L']).median())
print(f"Measurement height above displacement height: {zm:.2f} m")

inputs = df[['L', 'v_var', 'u*', 'wind_dir', 'wind_speed']].copy()
inputs['sigmav'] = np.sqrt(inputs['v_var'])
inputs['h'] = 1000.


def climatology(data):
    """Footprint climatology on a 500 m x 500 m grid with 1 m cells."""
    return ffp_climatology(zm=zm, h=data['h'], ol=data['L'], sigmav=data['sigmav'],
                           ustar=data['u*'], wind_dir=data['wind_dir'],
                           umean=data['wind_speed'], domain=[-250, 250, -250, 250],
                           dx=1., verbose=0)


clim = climatology(inputs)

fp_clim = dv.plotting.FootprintPlot(clim)
fp_clim.plot()
plt.show(block=False)

print(fp_clim.info_text())

# %%
# Unstable and stable conditions side by side
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With ``ax``, ``plot()`` draws only the map, and the colour bar takes its space
# from that axes. A ``FormatStyle`` sets the title. Under stable conditions
# (``L > 0``) the turbulence is weaker and the footprint reaches farther from the
# tower than under unstable conditions (``L < 0``).
#
# ``source_area`` selects the contour line that the subtitle reports and the map
# highlights, here the 50 % source area.

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), layout='constrained')
for ax, (title, rows) in zip(axes, [('Unstable (L < 0)', inputs['L'] < 0),
                                    ('Stable (L > 0)', inputs['L'] > 0)]):
    fp = dv.plotting.FootprintPlot(climatology(inputs[rows]), source_area=0.5)
    fp.plot(ax=ax, format_style=dv.plotting.FormatStyle(title=title, show_grid=False))
    print(f"{title}: {fp.info_text()}")
plt.show(block=False)

# sphinx_gallery_thumbnail_number = 2
