"""
For more info on wind rotation / tilt correction see:

References:
    References:

    (WIL01) Wilczak, J. M., Oncley, S. P., & Stage, S. A. (2001). Sonic Anemometer Tilt
        Correction Algorithms. Boundary-Layer Meteorology, 99(1), 127–150. https://doi.org/10.1023/A:1018966204465

    - Tilt correction in EddyPro_ https://www.licor.com/env/support/EddyPro/topics/anemometer-tilt-correction.html

"""

import math

import pandas as pd
from pandas import Series


class WindRotation2D:
    """Coordinate rotation and tilt correction for eddy covariance measurements.

    Performs double rotation of the coordinate system to align with mean wind
    direction, enabling proper calculation of turbulent fluctuations. This is
    essential for eddy covariance flux calculations.

    Examples
    --------
    See `examples/echires/windrotation.py` for complete examples demonstrating
    wind rotation with synthetic and real eddy covariance data.
    """

    def __init__(self, u: Series, v: Series, w: Series, c: Series):
        """Coordinate rotation and calculation of turbulent fluctuations.

        Args:
            u: Horizontal wind component in x direction (m s-1)
            v: Horizontal wind component in y direction (m s-1)
            w: Vertical wind component in z direction (m s-1)
            c: Scalar for which turbulent fluctuation is calculated

        """
        self.u = u
        self.v = v
        self.w = w
        self.c = c

        self._primes_df = pd.DataFrame

        self._run()

    @property
    def primes_df(self) -> pd.DataFrame:
        """Overall flag, calculated from individual flags from multiple iterations."""
        if not isinstance(self._primes_df, pd.DataFrame):
            raise Exception('No overall flag available.')
        return self._primes_df

    def get_primes(self) -> pd.DataFrame:
        """Return turbulent fluctuations of all wind components and scalar."""
        return self.primes_df

    def _run(self):
        self.theta, self.phi = self.rot_angles_from_mean_wind()
        self.u2, self.v2, self.w2 = self.rotate_wind()
        self.u_prime, self.v_prime, self.w_prime, self.c_prime = self.turbulent_fluctuations()
        self._assign_names()
        self._collect_primes()

    def _collect_primes(self):
        frame = {
            self.u_prime.name: self.u_prime,
            self.v_prime.name: self.v_prime,
            self.w_prime.name: self.w_prime,
            self.c_prime.name: self.c_prime,
        }
        self._primes_df = pd.DataFrame.from_dict(frame)

    def _assign_names(self):
        self.u_prime.name = f'{self.u.name}_TURB'
        self.v_prime.name = f'{self.v.name}_TURB'
        self.w_prime.name = f'{self.w.name}_TURB'
        self.c_prime.name = f'{self.c.name}_TURB'

    def turbulent_fluctuations(self):
        """Reynold's decomposition """
        u_rot_mean = self.u2.mean()
        u_prime = self.u2 - u_rot_mean
        v_rot_mean = self.v2.mean()
        v_prime = self.v2 - v_rot_mean
        w_rot_mean = self.w2.mean()
        w_prime = self.w2 - w_rot_mean
        c_mean = self.c.mean()
        c_prime = self.c - c_mean
        return u_prime, v_prime, w_prime, c_prime

    def rot_angles_from_mean_wind(self):
        """
        Calculate rotation angles for double rotation from mean wind

        The rotation angles are calculated from mean wind, but are later
        applied sample-wise to the full high-resolution data (typically 20Hz
        for wind data).

        Note that rotation angles are given in radians.

        First rotation angle:
            theta = tan-1 (v_mean / u_mean)

        Second rotation angle:
            phi = tan-1 (w_temp / u_temp)

        """

        u_mean = self.u.mean()
        v_mean = self.v.mean()
        w_mean = self.w.mean()

        # First rotation angle Theta, in radians
        # (WIL01)   0 = tan-1 * ( mean(v) / mean(u) )
        theta = math.atan(v_mean / u_mean)

        # Perform first rotation of coordinate system for mean wind
        # Make v component of mean wind zero --> v_temp becomes zero
        u1 = u_mean * math.cos(theta) + v_mean * math.sin(theta)
        # v_temp = -u_mean * math.sin(angle_r1) + v_mean * math.cos(angle_r1)
        w1 = w_mean

        # Second rotation angle Phi, in radians
        # (WIL01)    Q = tan-1 * ( mean(w1) / mean(u1) )
        phi = math.atan(w1 / u1)

        return theta, phi

    def rotate_wind(self):
        """Use rotation angles from mean wind to perform double rotation
        on high-resolution wind data.
        """

        # Perform first rotation of coordinate system
        # Make v component zero --> mean of high-res v_temp_col becomes zero (or very close to)
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
        # Make w component zero --> mean of high-res w_rot_col becomes zero (or very close to)
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
