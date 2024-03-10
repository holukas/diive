import os
from pathlib import Path

import pandas as pd
from pandas import DataFrame

import diive.core.io.filedetector as fd
from diive.core.io.filereader import ReadFileType
from diive.core.io.filereader import search_files
from diive.core.times.times import create_timestamp


class FileSplitter:

    def __init__(
            self,
            filepath: str or Path,
            file_pattern: str,
            file_date_format: str,
            file_type: str,
            data_nominal_res: float,
            data_split_duration: str,
            expected_duration: int,
            file_name: str,
            file_start: str,
            outdir: str,
            data_split_outfile_prefix: str = "",
            data_split_outfile_suffix: str = ""
    ):
        """Split file into multiple smaller files and export them to csv file.


        Args:
            filepath:
            file_pattern:
            file_date_format:
            file_type:
            data_nominal_res:
            data_split_duration:
            outdir:
        """
        self.filepath = Path(filepath) if isinstance(filepath, str) else filepath
        self.file_pattern = file_pattern
        self.file_date_format = file_date_format
        self.filetype = file_type
        self.data_nominal_res = data_nominal_res
        self.data_split_duration = data_split_duration
        self.outdir = Path(outdir)
        self.expected_duration = expected_duration
        self.file_name = file_name
        self.file_start = file_start
        self.data_split_outfile_prefix = data_split_outfile_prefix
        self.data_split_outfile_suffix = data_split_outfile_suffix

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
        df, true_resolution = create_timestamp(df=file_df,
                                               file_start=self.file_start,
                                               data_nominal_res=self.data_nominal_res,
                                               expected_duration=self.expected_duration)

        # Collect file data stats
        self._filestats_df = fd.add_data_stats(df=file_df,
                                               true_resolution=true_resolution,
                                               filename=self.file_name,
                                               found_records=len(file_df))

        self._splitstats_df = self.loop_splits(file_df=file_df,
                                               outdir_splits=self.outdir)

    def loop_splits(self, file_df, outdir_splits):
        counter_splits = -1
        file_df['index'] = pd.to_datetime(file_df.index)

        # Loop segments
        splits_overview_df = pd.DataFrame()
        split_grouped = file_df.groupby(pd.Grouper(key='index', freq=self.data_split_duration))
        for split_key, split_df in split_grouped:
            counter_splits += 1
            split_start = split_df.index[0]
            split_end = split_df.index[-1]

            # Name for split file
            split_name = (f"{self.data_split_outfile_prefix}"
                          f"_{split_start.strftime('%Y%m%d%H%M%S')}"
                          f"{self.data_split_outfile_suffix}.csv")

            split_df = split_df.fillna(-9999, inplace=False)

            # Export
            print(f"    Saving split {split_name} | n_records: {len(split_df.index)} "
                  f"| n_columns: {len(split_df.columns)} "
                  f"| start: {split_start} | end: {split_end}")
            split_df.to_csv(outdir_splits / f"{split_name}")

            splits_overview_df.loc[split_name, 'start'] = split_start
            splits_overview_df.loc[split_name, 'end'] = split_end
            splits_overview_df.loc[split_name, 'source_file'] = self.filepath.name
            splits_overview_df.loc[split_name, 'source_path'] = self.filepath
            splits_overview_df.loc[split_name, 'n_records'] = len(split_df.index)
            splits_overview_df.loc[split_name, 'n_columns'] = len(split_df.columns)

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
            pattern: str,
            file_date_format: str,
            file_generation_res: str,
            data_res: float,
            filetype: str,
            data_split_duration: str,
            files_how_many: int = None,
            data_split_outfile_prefix: str = "",
            data_split_outfile_suffix: str = ""
    ):
        self.outdir = outdir
        self.searchdirs = searchdirs
        self.file_pattern = pattern
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.data_nominal_res = data_res
        self.files_how_many = files_how_many
        self.filetype = filetype
        self.data_split_duration = data_split_duration
        self.data_split_outfile_prefix = data_split_outfile_prefix
        self.data_split_outfile_suffix = data_split_outfile_suffix

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
                file_pattern=self.file_pattern,  # Accepts regex
                file_date_format=self.file_date_format,  # Date format in filename
                file_type=self.filetype,
                data_nominal_res=self.data_nominal_res,  # Measurement every 0.05s
                data_split_duration=self.data_split_duration,  # Split into 30min segments
                data_split_outfile_prefix=self.data_split_outfile_prefix,
                data_split_outfile_suffix=self.data_split_outfile_suffix,
                outdir=outdirs['splits'],
                expected_duration=file_info_row['expected_duration'],
                file_name=file_info_row['filename'],
                file_start=file_info_row['start']
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
        coll_filestats_df.to_csv(outdirs['stats'] / '1_filestats.csv')
        coll_splitstats_df.to_csv(outdirs['stats'] / '2_splitstats.csv')

    def _detect_files(self, filelist: list) -> DataFrame:
        # Detect expected and unexpected files from filelist
        print("\nDetecting expected and unexpected files from filelist ...")
        fide = fd.FileDetector(filelist=filelist,
                               file_date_format=self.file_date_format,
                               file_generation_res=self.file_generation_res,
                               data_res=self.data_nominal_res,
                               files_how_many=self.files_how_many)
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
        print(f"\nSearching files with pattern {self.file_pattern} in dir {self.searchdirs} ...")
        filelist = search_files(searchdirs=self.searchdirs, pattern=self.file_pattern)
        for filepath in filelist:
            print(f"    --> Found file: {filepath.name} in {filepath}.")
        return filelist


def example():
    OUTDIR = r'F:\TMP\das_filesplitter'
    SEARCHDIRS = [r'L:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\_testdata']
    PATTERN = 'CH-DAS_*.csv.gz'
    FILEDATEFORMAT = 'CH-DAS_%Y%m%d%H%M.csv.gz'
    FILE_GENERATION_RES = '6h'
    DATA_NOMINAL_RES = 0.05
    FILES_HOW_MANY = 1
    FILETYPE = 'ETH-SONICREAD-BICO-CSVGZ-20HZ'
    DATA_SPLIT_DURATION = '30min'
    DATA_SPLIT_OUTFILE_PREFIX = 'CH-DAS_'
    DATA_SPLIT_OUTFILE_SUFFIX = '_30MIN-SPLIT'

    fsm = FileSplitterMulti(
        outdir=OUTDIR,
        searchdirs=SEARCHDIRS,
        pattern=PATTERN,
        file_date_format=FILEDATEFORMAT,
        file_generation_res=FILE_GENERATION_RES,
        data_res=DATA_NOMINAL_RES,
        files_how_many=FILES_HOW_MANY,
        filetype=FILETYPE,
        data_split_duration=DATA_SPLIT_DURATION,
        data_split_outfile_prefix=DATA_SPLIT_OUTFILE_PREFIX,
        data_split_outfile_suffix=DATA_SPLIT_OUTFILE_SUFFIX
    )
    fsm.run()


if __name__ == "__main__":
    example()
