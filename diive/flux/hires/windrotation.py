"""
WIND ROTATION: SONIC ANEMOMETER TILT CORRECTION
================================================

Rotate wind vectors to account for sonic anemometer tilt relative to true vertical.

Part of the diive library: https://github.com/holukas/diive

References:
    (WIL01) Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001). Sonic Anemometer Tilt
        Correction Algorithms. Boundary-Layer Meteorology, 99(1), 127–150. https://doi.org/10.1023/A:1018966204465

    - Tilt correction in EddyPro: https://www.licor.com/env/support/EddyPro/topics/anemometer-tilt-correction.html

"""

import math

from pandas import Series


def reynolds_decomposition(x: Series) -> Series:
    """Decompose a time series into its turbulent fluctuation about the mean.

    Reynolds decomposition splits x into a mean and a turbulent fluctuation:
        x = mean(x) + x'
    Returns x' = x - mean(x).

    In eddy covariance processing this is applied after double rotation to obtain
    turbulent fluctuations of wind components and scalars, which are then used
    to compute covariances and fluxes.

    Parameters
    ----------
    x : Series
        Time series of any wind component or scalar (e.g. w, CO2, H2O).

    Returns
    -------
    Series
        Turbulent fluctuation x' = x - mean(x), same index and name as input.

    See Also
    --------
    WindDoubleRotation : Coordinate rotation and tilt correction for eddy covariance measurements.
    """
    return x - x.mean()


class WindDoubleRotation:
    """Coordinate rotation and tilt correction for eddy covariance measurements.

    Performs double rotation of the coordinate system to align with mean wind
    direction. After rotation, mean(v2) ~ 0 and mean(w2) ~ 0, so the rotated
    frame correctly separates mean transport from turbulent fluctuations.

    Turbulent fluctuations of the rotated wind components and any scalar are
    computed separately with `reynolds_decomposition` after rotation:

        wr = WindDoubleRotation(u=u, v=v, w=w)
        w_prime = reynolds_decomposition(wr.w2)
        c_prime = reynolds_decomposition(c)

    Attributes
    ----------
    theta : float
        First rotation angle in radians (yaw: aligns mean wind to x-axis).
    phi : float
        Second rotation angle in radians (pitch: sets mean vertical wind to zero).
    u2 : Series
        Rotated u component after double rotation (m s-1).
    v2 : Series
        Rotated v component after double rotation, mean ~ 0 (m s-1).
    w2 : Series
        Rotated w component after double rotation, mean ~ 0 (m s-1).

    See Also
    --------
    reynolds_decomposition : Compute turbulent fluctuation x' = x - mean(x).
    MaxCovariance : Detect time lag between wind and scalars using covariance.
    FluxDetectionLimit : Calculate minimum detectable flux based on measurement noise.

    Example
    -------
    See `examples/flux/hires/flux_windrotation.py` for complete examples demonstrating
    wind rotation and tilt correction with synthetic eddy covariance data.
    """

    def __init__(self, u: Series, v: Series, w: Series):
        """Coordinate rotation and tilt correction.

        Args:
            u: Horizontal wind component in x direction (m s-1)
            v: Horizontal wind component in y direction (m s-1)
            w: Vertical wind component in z direction (m s-1)
        """
        self.u = u
        self.v = v
        self.w = w

        self._run()

    def _run(self):
        self.theta, self.phi = self._rot_angles_from_mean_wind()
        self.u2, self.v2, self.w2 = self._rotate_wind()

    def _rot_angles_from_mean_wind(self):
        """
        Calculate rotation angles for double rotation from mean wind.

        The rotation angles are calculated from mean wind, but are later
        applied sample-wise to the full high-resolution data (typically 20Hz
        for wind data).

        Note that rotation angles are given in radians.

        First rotation angle:
            theta = atan2(v_mean, u_mean)

        Second rotation angle:
            phi = atan2(w1, u1)

        """
        u_mean = self.u.mean()
        v_mean = self.v.mean()
        w_mean = self.w.mean()

        # First rotation angle Theta, in radians
        # (WIL01)   0 = tan-1 * ( mean(v) / mean(u) )
        theta = math.atan2(v_mean, u_mean)

        # Perform first rotation of coordinate system for mean wind
        # Make v component of mean wind zero --> v_temp becomes zero
        u1 = u_mean * math.cos(theta) + v_mean * math.sin(theta)
        # v1 = -u_mean * math.sin(theta) + v_mean * math.cos(theta)  -> 0 by definition
        w1 = w_mean

        # Second rotation angle Phi, in radians
        # (WIL01)    Q = tan-1 * ( mean(w1) / mean(u1) )
        phi = math.atan2(w1, u1)

        return theta, phi

    def _rotate_wind(self):
        """Use rotation angles from mean wind to perform double rotation
        on high-resolution wind data.
        """

        # Perform first rotation of coordinate system
        # Make v component zero --> mean of high-res v becomes zero (or very close to)
        # From Wilczak et al. (2001):
        # *Note that 0 is used as symbol for Theta (rotation angle)*
        #   0 = tan-1 * ( mean(v) / mean(u) )
        #   u1 = u * cos 0 + v * sin 0
        #   v1 = -u * sin 0 + v * cos 0
        #   w1 = w
        u1 = self.u * math.cos(self.theta) + self.v * math.sin(self.theta)
        v1 = -self.u * math.sin(self.theta) + self.v * math.cos(self.theta)
        w1 = self.w

        # Perform second rotation of coordinate system
        # Make w component zero --> mean of high-res w becomes zero (or very close to)
        # From Wilczak et al. (2001):
        # *Note that Q is used as symbol for Phi (rotation angle)*
        #   Q = tan-1 * ( mean(w1) / mean(u1) )
        #   u2 = u1 * cos Q + w1 * sin Q
        #   v2 = v1
        #   w2 = -u1 * sin Q + w1 * cos Q
        u2 = u1 * math.cos(self.phi) + w1 * math.sin(self.phi)
        v2 = v1
        w2 = -u1 * math.sin(self.phi) + w1 * math.cos(self.phi)

        return u2, v2, w2
