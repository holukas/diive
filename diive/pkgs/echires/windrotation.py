import math

import pandas as pd
from pandas import Series


class WindRotation2D:

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
            raise Exception(f'No overall flag available.')
        return self._primes_df

    def get_primes(self) -> pd.DataFrame:
        """Return turbulent fluctuations of all wind components and scalar."""
        return self.primes_df

    def _run(self):
        self.angle_r1, self.angle_r2 = self.rot_angles_from_mean_wind()
        self.u_rot, self.v_rot, self.w_rot = self.rotate_wind()
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
        u_rot_mean = self.u_rot.mean()
        u_prime = self.u_rot - u_rot_mean
        v_rot_mean = self.v_rot.mean()
        v_prime = self.v_rot - v_rot_mean
        w_rot_mean = self.w_rot.mean()
        w_prime = self.w_rot - w_rot_mean
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
            thita = tan-1 (v_mean / u_mean)

        Second rotation angle:
            phi = tan-1 (w_temp / u_temp)

        """

        u_mean = self.u.mean()
        v_mean = self.v.mean()
        w_mean = self.w.mean()

        # First rotation angle, in radians
        angle_r1 = math.atan(v_mean / u_mean)

        # Perform first rotation of coordinate system for mean wind
        # Make v component of mean wind zero --> v_temp becomes zero
        u_temp = u_mean * math.cos(angle_r1) + v_mean * math.sin(angle_r1)
        v_temp = -u_mean * math.sin(angle_r1) + v_mean * math.cos(angle_r1)
        w_temp = w_mean

        # Second rotation angle, in radians
        angle_r2 = math.atan(w_temp / u_temp)

        # For calculating the rotation angles, it is not necessary to perform the second
        # rotation of the coordinate system for mean wind
        # Make v component zero, vm = 0
        # u_rot = u_temp * math.degrees(math.cos(angle_r2)) + w_temp * math.degrees(math.sin(angle_r2))
        # v_rot = v_temp
        # w_rot = -u_temp * math.degrees(math.sin(angle_r2)) + w_temp * math.degrees(math.cos(angle_r2))

        return angle_r1, angle_r2

    def rotate_wind(self):
        """
        Use rotation angles from mean wind to perform double rotation
        on high-resolution wind data
        """

        # Perform first rotation of coordinate system
        # Make v component zero --> mean of high-res v_temp_col becomes zero (or very close to)
        u_temp = self.u * math.cos(self.angle_r1) + self.v * math.sin(self.angle_r1)
        v_temp = -self.u * math.sin(self.angle_r1) + self.v * math.cos(self.angle_r1)
        w_temp = self.w

        # Perform second rotation of coordinate system
        # Make w component zero --> mean of high-res w_rot_col becomes zero (or very close to)
        u_rot = u_temp * math.cos(self.angle_r2) + w_temp * math.sin(self.angle_r2)
        v_rot = v_temp
        w_rot = -u_temp * math.sin(self.angle_r2) + w_temp * math.cos(self.angle_r2)

        return u_rot, v_rot, w_rot


def example():
    # from diive.pkgs.echires.windrotation import WindRotation2D
    from diive.core.io.filereader import ReadFileType

    from diive.core.io.filereader import search_files

    OUTDIR = r'F:\TMP\das_filesplitter'

    # SEARCHDIRS = [r'L:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\_testdata']
    # PATTERN = 'CH-DAS_*.csv.gz'
    # FILEDATEFORMAT = 'CH-DAS_%Y%m%d%H%M.csv.gz'
    # FILE_GENERATION_RES = '6h'
    # DATA_NOMINAL_RES = 0.05
    # FILES_HOW_MANY = 1
    # FILETYPE = 'ETH-SONICREAD-BICO-CSVGZ-20HZ'
    # DATA_SPLIT_DURATION = '30min'
    # DATA_SPLIT_OUTFILE_PREFIX = 'CH-DAS_'
    # DATA_SPLIT_OUTFILE_SUFFIX = '_30MIN-SPLIT'
    #
    # from diive.core.io.filesplitter import FileSplitterMulti
    # fsm = FileSplitterMulti(
    #     outdir=OUTDIR,
    #     searchdirs=SEARCHDIRS,
    #     pattern=PATTERN,
    #     file_date_format=FILEDATEFORMAT,
    #     file_generation_res=FILE_GENERATION_RES,
    #     data_res=DATA_NOMINAL_RES,
    #     files_how_many=FILES_HOW_MANY,
    #     filetype=FILETYPE,
    #     data_split_duration=DATA_SPLIT_DURATION,
    #     data_split_outfile_prefix=DATA_SPLIT_OUTFILE_PREFIX,
    #     data_split_outfile_suffix=DATA_SPLIT_OUTFILE_SUFFIX
    # )
    # fsm.run()

    filelist = search_files(searchdirs=r'F:\TMP\das_filesplitter\splits', pattern='CH-DAS_*.csv')

    # Settings
    U = 'U_[R350-B]'
    V = 'V_[R350-B]'
    W = 'W_[R350-B]'
    C = 'CH4_DRY_[QCL-C2]'

    for filepath in filelist:
        # Read file
        df, meta = ReadFileType(filepath=filepath,
                                filetype='GENERIC-CSV-HEADER-1ROW-TS-MIDDLE-FULL-NS-30MIN',
                                data_nrows=None,
                                output_middle_timestamp=True).get_filedata()

        # Wind rotation for turbulent fluctuations
        u = df[U].copy()
        v = df[V].copy()
        w = df[W].copy()
        c = df[C].copy()
        wr = WindRotation2D(u=u, v=v, w=w, c=c)
        w_prime, c_prime = wr.get_primes()

        newdf = df.copy()
        newdf[w_prime.name] = w_prime.copy()
        newdf[c_prime.name] = c_prime.copy()
        # subset = pd.concat([w_prime, c_prime], ignore_index=False, axis=1)


if __name__ == '__main__':
    example()
