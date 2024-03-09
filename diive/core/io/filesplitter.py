from pathlib import Path

import pandas as pd

import files

import diive.core.io.filesdetector as fd

class FileSplitter:
    splits_overview_df = pd.DataFrame()

    def __init__(self,
                 file_pattern, file_date_format, file_generation_res, files_how_many,
                 data_nominal_res, data_split_duration, indir, outdir):

        self.file_pattern = file_pattern
        self.file_date_format = file_date_format
        self.file_generation_res = file_generation_res
        self.files_how_many = files_how_many

        self.data_nominal_res = data_nominal_res
        self.data_split_duration = data_split_duration

        self.indir = Path(indir)
        self.outdir = Path(outdir)

        self.run()



    def run(self):

        outdirs = fd.setup_output_dirs(outdir=self.outdir,
                                          del_previous_results=True)

        files_overview_df = fd.FilesDetector(indir=self.indir,
                                                outdir=outdirs['stats'],
                                                file_pattern=self.file_pattern,
                                                file_date_format=self.file_date_format,
                                                file_generation_res=self.file_generation_res,
                                                data_res=self.data_nominal_res,
                                                files_how_many=self.files_how_many).get()

        # Loop through files and their splits
        self.loop_files(files_overview_df=files_overview_df,
                        outdirs=outdirs)

    def loop_files(self, files_overview_df, outdirs):

        for file_idx, file_info_row in files_overview_df.iterrows():

            # Check file availability
            if file_info_row['file_available'] == 0:
                continue

            print(f"\n\nWorking on file '{file_info_row['filepath'].name}'")
            print(f"    Path to file: {file_info_row['filepath']}")

            # Read and prepare data file
            file_df = fd.read_raw_data(filepath=file_info_row['filepath'],
                                          nrows=None)

            # Add timestamp to each record
            file_df, true_resolution = fd.insert_datetime_index(df=file_df,
                                                                   file_info_row=file_info_row,
                                                                   data_nominal_res=self.data_nominal_res)

            # Add stats to overview
            files_overview_df = fd.add_data_stats(df=file_df,
                                                     true_resolution=true_resolution,
                                                     filename=file_info_row['filename'],
                                                     files_overview_df=files_overview_df,
                                                     found_records=len(file_df))

            self.loop_splits(file_df=file_df,
                             file_info_row=file_info_row,
                             outdir_splits=outdirs['splits'])

        files_overview_df.to_csv(outdirs['stats'] / '1_files_stats.csv')
        self.splits_overview_df.to_csv(outdirs['stats'] / '2_splits_stats.csv')

    def loop_splits(self, file_df, file_info_row, outdir_splits):
        counter_splits = -1
        file_df['index'] = pd.to_datetime(file_df.index)

        # Loop segments
        split_grouped = file_df.groupby(pd.Grouper(key='index', freq=self.data_split_duration))
        for split_key, split_df in split_grouped:
            counter_splits += 1
            split_start = split_df.index[0]
            split_end = split_df.index[-1]
            split_name = f"{split_start.strftime('%Y%m%d%H%M%S')}.csv"

            # # Calculate turbulence (wind rotation) and add to data
            # turb_split_df = WindRotation(source_df=split_df,
            #                              u_col=self.wind_u_col, v_col=self.wind_v_col,
            #                              w_col=self.wind_w_col, scalar_col=self.scalar_col).get()
            # turb_split_df.drop(['index'], axis=1, inplace=True)  # 'index' col auto-created during grouping
            # turb_split_df.fillna(-9999, inplace=True)

            print(f"    Saving split {split_name}")
            turb_split_df.to_csv(outdir_splits / f"{split_name}")

            self.splits_overview_df.loc[split_name, 'start'] = split_start
            self.splits_overview_df.loc[split_name, 'end'] = split_end
            self.splits_overview_df.loc[split_name, 'source_file'] = file_info_row['filepath'].name
            self.splits_overview_df.loc[split_name, 'source_path'] = file_info_row['filepath']

            # # Stats todo maybe later
            # for col in split_df.columns:
            #     self.splits_overview_df.loc[split_name, f'numvals_{col}'] = split_df[col].dropna().size
            #     try:
            #         self.splits_overview_df.loc[split_name, f'median_{col}'] = split_df[col].median()
            #     except TypeError:
            #         self.splits_overview_df.loc[split_name, f'median_{col}'] = np.nan


if __name__ == "__main__":
    FileSplitter(file_pattern='CH-DAS_*.csv.gz',  # Accepts regex
                 file_date_format='%Y%m%d%H%M',  # Date format in filename
                 file_generation_res='6H',  # One file expected every x hour(s)
                 data_nominal_res=0.05,  # Measurement every 0.05s
                 files_how_many=False,
                 data_split_duration='30min',  # 30min segments
                 indir=r'F:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\_testdata',
                 outdir=r'F:\TMP\das_filesplitter')
