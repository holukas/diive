import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame

import diive.core.io.filedetector as fd
from diive.core.io.filereader import ReadFileType
from diive.core.io.filereader import search_files
from diive.core.times.times import create_timestamp
from diive.pkgs.echires.windrotation import WindRotation2D


class FileSplitter:

    def __init__(
            self,
            filepath: str or Path,
            filename_pattern: str,
            filename_date_format: str,
            filetype: str,
            data_nominal_res: float,
            data_split_duration: str,
            expected_duration: int,
            file_name: str,
            file_start: str,
            outdir: str,
            data_split_outfile_prefix: str = "",
            data_split_outfile_suffix: str = "",
            compress_splits: bool = False,
            rotation: bool = False,
            u_var: str = None,
            v_var: str = None,
            w_var: str = None,
            c_var: str = None,
            outfile_limit_n_rows: int = None
    ):
        """Split file into multiple smaller parts and export them as multiple CSV files.

        Args:
            filepath: Path to the file to split.
            outdir: Output directory for splitted files.
            filename_pattern: File names to search for in *searchdirs*.
            filename_date_format: Full parsing string for getting the datetime info from the file name.
            data_nominal_res: Nominal time resolution of the data in found files in seconds.
                For example 0.05 for 20 Hz data, 1 for 1 Hz data, 10 for data that were
                recorded every 10 minutes, etc.
            filetype: The diive internal filetype of found files as defined in diive/configs/filetypes.
            data_split_duration: Defines the duration of one split. Accepts pandas frequency strings.
                For example files with *file_generation_freq='1h'* and *data_split_duration='30min'* would
                split the hourly files into two splits, with each split comprising data over 30 minutes.
            expected_duration: Expected duration of the file in *filepath* in total seconds.
                For example 3600 if the file is one hour long.
            data_split_outfile_prefix: Prefix for split file names.
            data_split_outfile_suffix: Suffix for split file names.
            compress_splits: If *True*, compress splits using 'gzip'. If *False*, splits
                are saved as CSV files.
            rotation: If *True*, the wind components *u*, *v*, *w* and the scalar *c* are rotated, using
                2-D wind rotation, i.e., the turbulent departures of wind and scalar are calcualted
                using Reynold's averaging.
            u_var: Name of the horizontal wind component in x direction (m s-1)
            v_var: Name of the horizontal wind component in y direction (m s-1)
            w_var: Name of the vertical wind component in z direction (m s-1)
            c_var: Name of the scalar for which turbulent fluctuation is calculated
            outfile_limit_n_rows: Limit splits to this number of rows. Mainly implemented
                for faster testing.
        """
        self.filepath = Path(filepath) if isinstance(filepath, str) else filepath
        self.filename_pattern = filename_pattern
        self.filename_date_format = filename_date_format
        self.filetype = filetype
        self.data_nominal_res = data_nominal_res
        self.data_split_duration = data_split_duration
        self.outdir = Path(outdir)
        self.expected_duration = expected_duration
        self.file_name = file_name
        self.file_start = file_start
        self.data_split_outfile_prefix = data_split_outfile_prefix
        self.data_split_outfile_suffix = data_split_outfile_suffix
        self.compress_splits = compress_splits
        self.rotation = rotation
        self.u_col = u_var
        self.v_col = v_var
        self.w_col = w_var
        self.c_col = c_var
        self.outfile_limit_n_rows = outfile_limit_n_rows

        # Init new vars
        self._filestats_df = DataFrame()
        self._splitstats_df = DataFrame()

    @property
    def filestats_df(self) -> DataFrame:
        """Return stats about files."""
        if not isinstance(self._filestats_df, DataFrame):
            raise Exception(f'No filestats available.')
        return self._filestats_df

    @property
    def splitstats_df(self) -> DataFrame:
        """Return stats about splits."""
        if not isinstance(self._splitstats_df, DataFrame):
            raise Exception(f'No splitstats available.')
        return self._splitstats_df

    def get_stats(self) -> tuple[DataFrame, DataFrame]:
        return self.filestats_df, self.splitstats_df

    def run(self):
        print(f"\n\nWorking on file '{self.filepath.name}'")
        print(f"    Path to file: {self.filepath}")

        # Read file
        file_df, meta = ReadFileType(filepath=self.filepath,
                                     filetype=self.filetype,
                                     data_nrows=None,
                                     output_middle_timestamp=False).get_filedata()

        # Add timestamp to each record
        file_df, true_resolution = create_timestamp(df=file_df,
                                                    file_start=self.file_start,
                                                    data_nominal_res=self.data_nominal_res,
                                                    expected_duration=self.expected_duration)

        # Collect file data stats
        self._filestats_df = fd.add_data_stats(df=file_df,
                                               true_resolution=true_resolution,
                                               filename=self.file_name,
                                               found_records=len(file_df))

        self._splitstats_df = self._loop_splits(file_df=file_df,
                                                outdir_splits=self.outdir)

    def _rotate_split(self, split_df: pd.DataFrame):
        wr = WindRotation2D(u=split_df[self.u_col],
                            v=split_df[self.v_col],
                            w=split_df[self.w_col],
                            c=split_df[self.c_col])
        primes_df = wr.get_primes()
        split_df = pd.concat([split_df, primes_df], axis=1)
        return split_df

    def _loop_splits(self, file_df, outdir_splits):
        counter_splits = -1
        file_df['index'] = pd.to_datetime(file_df.index)

        if self.rotation:
            self.data_split_outfile_suffix = f"{self.data_split_outfile_suffix}_ROT"

        if self.compress_splits:
            file_extension = '.csv.gz'
        else:
            file_extension = '.csv'

        # Loop segments
        splits_overview_df = pd.DataFrame()
        split_grouped = file_df.groupby(pd.Grouper(key='index', freq=self.data_split_duration))
        for split_key, split_df in split_grouped:
            counter_splits += 1
            split_start = split_df.index[0]
            split_end = split_df.index[-1]

            if self.rotation:
                split_df = self._rotate_split(split_df=split_df)

            split_df = split_df.fillna(-9999, inplace=False)

            # Name for split file
            split_name = (f"{self.data_split_outfile_prefix}"
                          f"{split_start.strftime('%Y%m%d%H%M%S')}"
                          f"{self.data_split_outfile_suffix}{file_extension}")

            # Export
            print(f"    Saving split {split_name} | n_records: {len(split_df.index)} "
                  f"| n_columns: {len(split_df.columns)} "
                  f"| start: {split_start} | end: {split_end} | wind_rotation: {self.rotation}")
            split_filepath = outdir_splits / f"{split_name}"
            compression = 'gzip' if self.compress_splits else None

            if self.outfile_limit_n_rows:
                split_df = split_df.iloc[
                           0:self.outfile_limit_n_rows]  # Limit number of exported records, useful for testing

            split_df.to_csv(split_filepath, compression=compression)

            splits_overview_df.loc[split_name, 'start'] = split_start
            splits_overview_df.loc[split_name, 'end'] = split_end
            splits_overview_df.loc[split_name, 'source_file'] = self.filepath.name
            splits_overview_df.loc[split_name, 'source_path'] = self.filepath
            splits_overview_df.loc[split_name, 'n_records'] = len(split_df.index)
            splits_overview_df.loc[split_name, 'n_columns'] = len(split_df.columns)
            splits_overview_df.loc[split_name, 'split_filepath'] = split_filepath
            splits_overview_df.loc[split_name, 'wind_rotation_1=yes'] = int(self.rotation)

        return splits_overview_df

        # # Stats todo maybe later
        # for col in split_df.columns:
        #     self.splits_overview_df.loc[split_name, f'numvals_{col}'] = split_df[col].dropna().size
        #     try:
        #         self.splits_overview_df.loc[split_name, f'median_{col}'] = split_df[col].median()
        #     except TypeError:
        #         self.splits_overview_df.loc[split_name, f'median_{col}'] = np.nan


def setup_output_dirs(outdir: str, del_previous_results=False):
    """Make output directories."""
    new_dirs = ['stats', 'splits']
    outdirs = {}

    # Store keys and full paths in dict
    for nd in new_dirs:
        # outdirs[nd] = Path(outdir)
        outdirs[nd] = outdir / Path(nd)

    # Make dirs
    for key, path in outdirs.items():
        if not Path.is_dir(path):
            print(f"Creating folder {path} ...")
            os.makedirs(path)
        else:
            if del_previous_results:
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    try:
                        if os.path.isfile(filepath) or os.path.islink(filepath):
                            print(f"Deleting file {filepath} ...")
                            os.unlink(filepath)
                        # elif os.path.isdir(filepath):
                        #     shutil.rmtree(filepath)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (filepath, e))

    return outdirs


class FileSplitterMulti:

    def __init__(
            self,
            outdir: str,
            searchdirs: str or list,
            filename_pattern: str,
            filename_date_format: str,
            file_generation_freq: str,
            data_nominal_res: float,
            filetype: str,
            data_split_duration: str,
            files_split_how_many: int = None,
            data_split_outfile_prefix: str = "",
            data_split_outfile_suffix: str = "",
            compress_splits: bool = False,
            rotation: bool = False,
            u_var: str = None,
            v_var: str = None,
            w_var: str = None,
            c_var: str = None,
            outfile_limit_n_rows: int = None
    ):
        """Split multiple files into multiple smaller parts 
        and save them as CSV or compressed CSV.
        
        For example, this allows to read multiple files comprising data over
        one hour and split each of them into two 30-minutes files.   
        
        Args:
            outdir: Output directory for splitted files.
            searchdirs: List of directories to search for files
            filename_pattern: File names to search for in *searchdirs*.
            filename_date_format: Full parsing string for getting the datetime info from the file name.
            file_generation_freq: File generation frequency. Accepts pandas frequency strings.
                For example '1h' if one file was generated every hour, '30min' if one file was 
                generated every 30 minutes, etc. 
            data_nominal_res: Nominal time resolution of the data in found files in seconds.
                For example 0.05 for 20 Hz data, 1 for 1 Hz data, 10 for data that were
                recorded every 10 minutes, etc.  
            filetype: The diive internal filetype of found files as defined in diive/configs/filetypes.
            data_split_duration: Defines the duration of one split. Accepts pandas frequency strings.
                For example files with *file_generation_freq='1h'* and *data_split_duration='30min'* would
                split the hourly files into two splits, with each split comprising data over 30 minutes.   
            files_split_how_many: How many found files are split. 
            data_split_outfile_prefix: Prefix for split file names.
            data_split_outfile_suffix: Suffix for split file names.
            compress_splits: If *True*, compress splits using 'gzip'. If *False*, splits
                are saved as CSV files.
            rotation: If *True*, the wind components *u*, *v*, *w* and the scalar *c* are rotated, using 
                2-D wind rotation, i.e., the turbulent departures of wind and scalar are calcualted
                using Reynold's averaging.
            u_var: Name of the horizontal wind component in x direction (m s-1)
            v_var: Name of the horizontal wind component in y direction (m s-1)
            w_var: Name of the vertical wind component in z direction (m s-1)
            c_var: Name of the scalar for which turbulent fluctuation is calculated
            outfile_limit_n_rows: Limit splits to this number of rows. Mainly implemented
                for faster testing. 
        """

        self.outdir = outdir
        self.searchdirs = searchdirs
        self.filename_pattern = filename_pattern
        self.filename_date_format = filename_date_format
        self.file_generation_freq = file_generation_freq
        self.data_nominal_res = data_nominal_res
        self.files_split_how_many = files_split_how_many
        self.filetype = filetype
        self.data_split_duration = data_split_duration
        self.data_split_outfile_prefix = data_split_outfile_prefix
        self.data_split_outfile_suffix = data_split_outfile_suffix
        self.compress_splits = compress_splits
        self.rotation = rotation
        self.outfile_limit_n_rows = outfile_limit_n_rows

        if rotation:
            self.u_var = u_var
            self.v_var = v_var
            self.w_var = w_var
            self.c_var = c_var
        else:
            self.u_var = None
            self.v_var = None
            self.w_var = None
            self.c_var = None

    def run(self):
        outdirs = self._setup_output_dirs()
        filelist = self._search_files()
        files_overview_df = self._detect_files(filelist=filelist)
        self._split_files(files_overview_df=files_overview_df, outdirs=outdirs)

    def _split_files(self, files_overview_df: DataFrame, outdirs: dict):
        filecounter = 0
        coll_filestats_df = DataFrame()
        coll_splitstats_df = DataFrame()

        # Split files into smaller files
        for file_idx, file_info_row in files_overview_df.iterrows():

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            filecounter += 1

            fs = FileSplitter(
                filepath=file_info_row['filepath'],
                filename_pattern=self.filename_pattern,  # Accepts regex
                filename_date_format=self.filename_date_format,  # Date format in filename
                filetype=self.filetype,
                data_nominal_res=self.data_nominal_res,  # Measurement every 0.05s
                data_split_duration=self.data_split_duration,  # Split into 30min segments
                data_split_outfile_prefix=self.data_split_outfile_prefix,
                data_split_outfile_suffix=self.data_split_outfile_suffix,
                outdir=outdirs['splits'],
                expected_duration=file_info_row['expected_duration'],
                file_name=file_info_row['filename'],
                file_start=file_info_row['start'],
                rotation=self.rotation,
                u_var=self.u_var,
                v_var=self.v_var,
                w_var=self.w_var,
                c_var=self.c_var,
                compress_splits=self.compress_splits,
                outfile_limit_n_rows=self.outfile_limit_n_rows
            )
            fs.run()
            filestats_df, splitstats_df = fs.get_stats()

            if filecounter == 1:
                coll_filestats_df = filestats_df.copy()
                coll_splitstats_df = splitstats_df.copy()
            else:
                coll_filestats_df = pd.concat([coll_filestats_df, filestats_df], axis=0, ignore_index=False)
                coll_splitstats_df = pd.concat([coll_splitstats_df, splitstats_df], axis=0, ignore_index=False)

            # if filecounter == 3:
            #     break

        # Export
        files_overview_df.to_csv(outdirs['stats'] / '0_files_overview.csv')
        coll_filestats_df.to_csv(outdirs['stats'] / '1_filestats.csv')
        coll_splitstats_df.to_csv(outdirs['stats'] / '2_splitstats.csv')

    def _detect_files(self, filelist: list) -> DataFrame:
        # Detect expected and unexpected files from filelist
        print("\nDetecting expected and unexpected files from filelist ...")
        fide = fd.FileDetector(filelist=filelist,
                               file_date_format=self.filename_date_format,
                               file_generation_res=self.file_generation_freq,
                               data_res=self.data_nominal_res,
                               files_how_many=self.files_split_how_many)
        fide.run()
        files_overview_df = fide.get_results()
        print(files_overview_df)
        return files_overview_df

    def _setup_output_dirs(self) -> dict:
        # Create output dirs
        print(f"\nCreating output dirs in folder {self.outdir} ...")
        outdirs = setup_output_dirs(outdir=self.outdir, del_previous_results=True)
        for folder, path in outdirs.items():
            print(f"    --> Created folder {folder} in {path}.")
        return outdirs

    def _search_files(self) -> list:
        # Search files with PATTERN
        print(f"\nSearching files with pattern {self.filename_pattern} in dir {self.searchdirs} ...")
        filelist = search_files(searchdirs=self.searchdirs, pattern=self.filename_pattern)
        for filepath in filelist:
            print(f"    --> Found file: {filepath.name} in {filepath}.")
        return filelist


def example():
    OUTDIR = r'F:\TMP\testdata_dyco\1-splits'
    SEARCHDIRS = [r'F:\TMP\testdata_dyco\0-raw_data_ascii']
    PATTERN = 'CH-DAV_*.csv.gz'
    FILEDATEFORMAT = 'CH-DAV_%Y%m%d%H%M.csv.gz'
    FILE_GENERATION_RES = '6h'
    DATA_NOMINAL_RES = 0.05
    FILES_HOW_MANY = None
    FILETYPE = 'ETH-SONICREAD-BICO-CSVGZ-20HZ'
    DATA_SPLIT_DURATION = '30min'
    DATA_SPLIT_OUTFILE_PREFIX = 'CH-DAV_'
    DATA_SPLIT_OUTFILE_SUFFIX = '_30MIN-SPLIT'
    COMPRESS_SPLITS = False
    ROTATION = True
    U = 'U_[HS50-A]'
    V = 'V_[HS50-A]'
    W = 'W_[HS50-A]'
    # C = 'CH4_DRY_[QCL-C2]'
    C = 'CO2_DRY_[IRGA72-A]'
    OUTFILE_LIMIT_N_ROWS = None  # int or None, for testing

    fsm = FileSplitterMulti(
        outdir=OUTDIR,
        searchdirs=SEARCHDIRS,
        filename_pattern=PATTERN,
        filename_date_format=FILEDATEFORMAT,
        file_generation_freq=FILE_GENERATION_RES,
        data_nominal_res=DATA_NOMINAL_RES,
        files_split_how_many=FILES_HOW_MANY,
        filetype=FILETYPE,
        data_split_duration=DATA_SPLIT_DURATION,
        data_split_outfile_prefix=DATA_SPLIT_OUTFILE_PREFIX,
        data_split_outfile_suffix=DATA_SPLIT_OUTFILE_SUFFIX,
        rotation=ROTATION,
        u_var=U,
        v_var=V,
        w_var=W,
        c_var=C,
        compress_splits=COMPRESS_SPLITS,
        outfile_limit_n_rows=OUTFILE_LIMIT_N_ROWS
    )
    fsm.run()


if __name__ == "__main__":
    example()
