import datetime as dt
import zipfile as zf
from pathlib import Path

import numpy as np
import pandas as pd
# from PyQt5 import QtWidgets as qw
from common.io.dirs import verify_dir
from diive.common.utils.config import validate_filetype_config
from diive.common.dfun.frames import add_second_header_row
from diive.common.dfun.times import make_timestamp_microsec_suffix


def parse_events_file(filepath, settings_dict):
    """
    Read file into df.
    """

    parse_dates = [0]  # Column is not renamed for Events

    # Read data file
    # --------------
    parse = lambda x: dt.datetime.strptime(x, settings_dict['TIMESTAMP']['DATETIME_FORMAT'])
    data_df = pd.read_csv(filepath,
                          skiprows=settings_dict['DATA']['SKIP_ROWS'],
                          header=settings_dict['DATA']['HEADER_ROWS'],
                          na_values=settings_dict['DATA']['NA_VALUES'],
                          encoding='utf-8',
                          delimiter=settings_dict['DATA']['DELIMITER'],
                          mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=parse_dates,
                          date_parser=parse,
                          index_col=None,
                          dtype=None)

    return data_df


def compare_len_header_vs_data(filepath, skip_rows_list, header_rows_list):
    """
    Check whether there are more data columns than given in the header.

    If not checked, this would results in an error when reading the csv file
    with .read_csv, because the method expects an equal number of header and
    data columns. If this check is True, then the difference between the length
    of the first data row and the length of the header row(s) can be used to
    automatically generate names for the missing header columns.
    """
    # Check number of columns of the first data row after the header part
    skip_num_lines = len(header_rows_list) + len(skip_rows_list)
    first_data_row_df = pd.read_csv(filepath, skiprows=skip_num_lines,
                                    header=None, nrows=1)
    len_data_cols = first_data_row_df.columns.size

    # Check number of columns of the header part
    header_cols_df = pd.read_csv(filepath, skiprows=skip_rows_list,
                                 header=header_rows_list, nrows=0)
    len_header_cols = header_cols_df.columns.size

    # Check if there are more data columns than header columns
    if len_data_cols > len_header_cols:
        more_data_cols_than_header_cols = True
        num_missing_header_cols = len_data_cols - len_header_cols
    else:
        more_data_cols_than_header_cols = False
        num_missing_header_cols = 0

    # Generate missing header columns if necessary
    header_cols_list = header_cols_df.columns.to_list()
    generated_missing_header_cols_list = []
    sfx = make_timestamp_microsec_suffix()
    if more_data_cols_than_header_cols:
        for m in list(range(1, num_missing_header_cols + 1)):
            missing_col = (f'unknown_{m}-{sfx}', '[-unknown-]')
            generated_missing_header_cols_list.append(missing_col)
            header_cols_list.append(missing_col)

    return more_data_cols_than_header_cols, num_missing_header_cols, header_cols_list, generated_missing_header_cols_list


def parse_csv_file(filepath, settings_dict):
    """ Read file into df, using parsing info from settings_dict """
    print(filepath)

    # Check data
    # ----------
    more_data_cols_than_header_cols, num_missing_header_cols, header_cols_list, generated_missing_header_cols_list = \
        compare_len_header_vs_data(filepath=filepath,
                                   skip_rows_list=settings_dict['DATA']['SKIP_ROWS'],
                                   header_rows_list=settings_dict['DATA']['HEADER_ROWS'])

    # Read data file
    # --------------
    # Column settings for parsing dates / times correctly
    parsed_index_col = ('index', '[parsed]')
    parse_dates = settings_dict['TIMESTAMP']['INDEX_COLUMN']
    parse_dates = {parsed_index_col: parse_dates}
    # date_parser = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')
    date_parser = lambda x: dt.datetime.strptime(x, settings_dict['TIMESTAMP']['DATETIME_FORMAT'])

    data_df = pd.read_csv(filepath,
                          # skiprows=[0],
                          skiprows=settings_dict['DATA']['HEADER_SECTION_ROWS'],
                          header=None,
                          names=header_cols_list,
                          na_values=settings_dict['DATA']['NA_VALUES'],
                          encoding='utf-8',
                          delimiter=settings_dict['DATA']['DELIMITER'],
                          mangle_dupe_cols=True,
                          keep_date_col=False,
                          parse_dates=parse_dates,
                          date_parser=date_parser,
                          index_col=None,
                          dtype=None,
                          skip_blank_lines=True,
                          engine='python')

    # for col in data_df.columns:
    #     boolean_findings = data_df[col].astype(str).str.contains('inf')
    #     if boolean_findings.sum() > 0:
    #         print(data_df[col][boolean_findings])
    #         print(f"{col}  {data_df[col].dtype}")

    # There exist certain instances where the float64 data column can contain
    # non-numeric values that are interpreted as a float64 inf, which is basically
    # a NaN value. To harmonize missing values inf is also set to NaN.
    data_df = data_df.replace(float('inf'), np.nan)
    data_df = data_df.replace(float('-inf'), np.nan)

    data_df = standardize_index(df=data_df, settings_dict=settings_dict)

    # Check if there is only one header row, if yes, then add second row (standardization)
    if len(settings_dict['DATA']['HEADER_ROWS']) == 1:
        data_df = add_second_header_row(df=data_df)

    return data_df, generated_missing_header_cols_list


def read_selected_events_file(index_data_df, filepath, settings_dict):
    """
    Read the selected events file.

    To read this file, data has already to be loaded. The index if the available
    data is used to convert the info from the Events file into a usable format
    that can be merged with the available data.
    """

    # Default Event columns
    event_start_col = ('EVENT_START', '[yyyy-mm-dd]')
    event_end_col = ('EVENT_END', '[yyyy-mm-dd]')
    event_type_col = ('EVENT', '[type]')

    # INFO FROM EVENTS FILE
    # ---------------------
    # Initial df of Events, directly read from the file
    _events_df = parse_events_file(filepath=filepath, settings_dict=settings_dict)
    _events_df[event_start_col] = pd.to_datetime(_events_df[event_start_col], format='%Y-%m-%d')
    _events_df[event_end_col] = pd.to_datetime(_events_df[event_end_col], format='%Y-%m-%d')

    listed_event_types = []
    for ix, row in _events_df.iterrows():
        listed_event_types.append(row[event_type_col])

    # COLLECTED EVENTS
    # ----------------
    # Collect Events data in df, each event type is in a separate column
    events_df = pd.DataFrame(index=index_data_df)
    event_type_cols = []
    for col in listed_event_types:
        event_type_tuple_col = (col, '[event]')
        events_df[event_type_tuple_col] = 0  # Init column w/ zeros (0=event did not take place)
        event_type_cols.append(event_type_tuple_col)

    # Get start and end time for events
    for ix, row in _events_df.iterrows():
        start = _events_df.loc[ix, event_start_col]
        end = _events_df.loc[ix, event_end_col]
        filter = (events_df.index.date >= start) & (events_df.index.date <= end)

        event_type_tuple_col = (row[event_type_col], '[event]')
        events_df.loc[filter, event_type_tuple_col] = 1

    # SANITIZE
    # --------
    # Sanitize time series, numeric data is needed
    # After this conversion, all columns are of float64 type, strings will be substituted
    # by NaN.
    # todo For now, columns that contain only NaNs are still in the df.
    events_df = events_df.apply(pd.to_numeric, errors='coerce')

    return events_df


def read_selected_data_file(filepath: Path, filetype_config: dict):
    """
    Read the file selected in the dialog window.
    *** todo this description needs update

    Reads the header section of the data file, which consists of 3 lbl_OptionRefinements_Header rows
    in EddyPro (EP) full_output files.

    In the EP file, row #0 contains the categories, which we do not need, therefore
    this row can be skipped when we read the lbl_OptionRefinements_Header section --> skiprows=[0]

    Row #1 contains the variable names, row #2 the respective units. Therefore only
    the first two lines of the file are read.

    Rows #1 and #2 are read as objects to avoid the problem caused by mangle_dupe_cols=True,
    which auto adds the suffix '.1' etc to duplicate columns. This suffix generates problems
    when later addressing the cols by name, because '.1' cols are not stored as tuples
    (which can be easily addressed), but as strings in the MultiIndex.

    The argument lbl_OptionRefinements_Header=None makes sure that no defined lbl_OptionRefinements_Header is built when reading
    the data from the file. The MultiIndex lbl_OptionRefinements_Header is built manually from data_df_colnames
    and data_df_units and then added to the used dataframe.
    """

    # Compression
    if filetype_config['FILE']['COMPRESSION'] == 'none':
        dir_temp_unzipped = '-not-needed-'
    else:
        filepath, dir_temp_unzipped = unzip_file(filepath=filepath)

    datafilereader = io.DataFileReader(
        filepath=filepath,
        data_skiprows=filetype_config['DATA']['SKIP_ROWS'],
        data_headerrows=filetype_config['DATA']['HEADER_ROWS'],
        data_headersection_rows=filetype_config['DATA']['HEADER_SECTION_ROWS'],
        data_na_vals=filetype_config['DATA']['NA_VALUES'],
        data_delimiter=filetype_config['DATA']['DELIMITER'],
        data_freq=filetype_config['DATA']['FREQUENCY'],
        timestamp_idx_col=filetype_config['TIMESTAMP']['INDEX_COLUMN'],
        timestamp_datetime_format=filetype_config['TIMESTAMP']['DATETIME_FORMAT'],
        timestamp_start_middle_end=filetype_config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD']
    )
    data_df = datafilereader._read()
    generated_missing_header_cols_list = datafilereader.generated_missing_header_cols_list

    # data_df, generated_missing_header_cols_list = parse_csv_file(filepath=filepath, settings_dict=filetype_config)

    # # SANITIZE
    # # --------
    # # Sanitize time series, numeric data is needed
    # # After this conversion, all columns are of float64 type, strings will be substituted
    # # by NaN. This means columns that contain only strings, e.g. the columns 'date' or
    # # 'filename' in the EddyPro full_output file, contain only NaNs after this step.
    # # Not too problematic in case of 'date', b/c the index contains the datetime info.
    # # todo For now, columns that contain only NaNs are still in the df.
    # # todo at some point, the string columns should also be considered
    # data_df = data_df.apply(pd.to_numeric, errors='coerce')

    # # FREQUENCY
    # # ---------
    # # Downsample, if needed
    # data_df, filetype_config['DATA']['FREQUENCY'] = downsample_data(df=data_df,
    #                                                                 freq=filetype_config['DATA']['FREQUENCY'],
    #                                                                 max_freq='1S')

    # # Set freq, at the same time this makes index CONTINUOUS w/o date gaps
    # data_df = data_df.asfreq(filetype_config['DATA']['FREQUENCY'])

    # # METADATA todo
    # # --------
    # # Collect variables metadata
    # data_meta_df = pd.DataFrame()
    # data_meta_df['_tuples'] = data_df.columns.to_list()
    # data_meta_df['var_index'] = data_meta_df.index
    # data_meta_df[['variable', 'units']] = pd.DataFrame(data_meta_df['_tuples'].tolist(), index=data_meta_df.index)
    # data_meta_df['modification_time'] = dt.datetime.now()
    # data_meta_df['tags'] = np.nan
    # data_meta_df['source_file'] = filepath
    #
    # for c in all_data_df.columns:
    #     data_meta_df['dtype'] = all_data_df[c].dtype
    #     data_meta_df['chksum'] = all_data_df[c].checdtype

    # # Detect freq for files from their timestamps
    # freq = infer_freq(df_index=data_df.index)  ## try to infer the frequency of the data

    # UNZIP
    # ----- todo reactivate at some point
    # if settings['zipped'] == 'Yes':
    #     for del_file in os.listdir(dir_temp_unzipped):
    #         del_filepath = dir_temp_unzipped / del_file
    #         os.remove(del_filepath)
    #     shutil.rmtree(dir_temp_unzipped)

    return data_df, generated_missing_header_cols_list


def standardize_index(df, settings_dict):
    # Index name is now the same for all filetypes w/ timestamp in data
    df.set_index([('index', '[parsed]')], inplace=True)
    df.index.name = ('TIMESTAMP', '[yyyy-mm-dd HH:MM:SS]')
    # Make sure the index is datetime
    df.index = pd.to_datetime(df.index)

    # Shift timestamp by half-frequency, if needed
    if settings_dict['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] == 'middle':
        pass
    else:
        timedelta = pd.to_timedelta(settings_dict['DATA']['FREQUENCY']) / 2
        if settings_dict['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] == 'end':
            df.index = df.index - pd.Timedelta(timedelta)
        elif settings_dict['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] == 'start':
            df.index = df.index + pd.Timedelta(timedelta)

    return df


def unzip_file(filepath):
    """ Unzips zipped file in filepath

        Unzips the zipped file to a temporary directory, which is later, after
        data have been read in, deleted.

        Returns the filepath to the unzipped file and the directory to which
        the zipped file has been extracted to.
    """
    with zf.ZipFile(filepath, 'r') as zip_ref:
        dir_temp_unzipped = '{}{}'.format(filepath, ".temp")  # dir as string, w/ .amp.temp at the end of dir name
        zip_ref.extractall(dir_temp_unzipped)

    ext = '.csv'
    dir_temp_unzipped = Path(dir_temp_unzipped)  # give dir as Path
    filename_unzipped = Path(dir_temp_unzipped).stem  # remove .temp for filename, only .amp, stem makes str
    filename_unzipped = Path(filename_unzipped).with_suffix(ext)  # replace .amp with .csv
    filepath = dir_temp_unzipped / filename_unzipped
    return filepath, dir_temp_unzipped


def select_source_files(start_dir, text="Open File"):
    """
    Load data from file(s), multiple files can be selected,
    using .getOpenFileNames instead of .getOpenFileName (note the plural s)
    """
    options = qw.QFileDialog.Options()
    filename_list, _ = qw.QFileDialog.getOpenFileNames(None, text, str(start_dir), '*.*',
                                                       options=options)  # open dialog
    return filename_list


def select_single_source_file(start_dir, text="Open File"):
    """
    Load data from one single file, only one file can be selected,
    using .getOpenFileName instead of .getOpenFileNames (note the missing plural s)
    """
    options = qw.QFileDialog.Options()
    filename, _ = qw.QFileDialog.getOpenFileName(None, text, str(start_dir), '*.*',
                                                 options=options)  # open dialog
    return filename


def select_output_dir(start_dir, rundir_name):
    """ Select output directory for this project, all exported files will be saved in this folder. """
    selected_dir = qw.QFileDialog.getExistingDirectory(None, 'Select Output Folder for Project',
                                                       str(start_dir))  # open dialog
    # hint: file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    selected_dir = Path(selected_dir)
    outdir = selected_dir / rundir_name
    verify_dir(dir=outdir)
    return outdir


# def search_files(src_dir, ext: str):
#     """ Search files and store their filename and the path to the file in dictionary. """
#     found_files_dict = {}
#     for root, dirs, files in os.walk(src_dir):
#         for idx, settings_file_name in enumerate(files):
#             if fnmatch.fnmatch(settings_file_name, ext):
#                 filepath = Path(root) / settings_file_name
#                 found_files_dict[settings_file_name] = filepath
#     return found_files_dict


def parse_settingsfile_todict(filepath):
    """Open and read settings file and store contents in dict"""
    settings_dict = {}
    with open(filepath) as input_file:
        for line in input_file:  # cycle through all lines in settings settings_file_name
            if line.startswith('#'):
                pass
            else:
                if '=' in line:  # identify lines that contain setting
                    line_id, line_setting = line.split('=')
                    line_id = line_id.strip()
                    line_setting = line_setting.strip()
                    settings_dict[line_id] = line_setting  # store setting from settings_file_name in dict

    settings_dict = validate_filetype_config(filetype_config=settings_dict)

    return settings_dict


