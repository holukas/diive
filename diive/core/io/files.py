import datetime as dt
import zipfile as zf
from pathlib import Path
import time
import pandas as pd
import pickle


def save_as_pickle(outpath:str, filename: str, data) -> str:
    """Save data as pickle"""
    outpath = Path(outpath)
    filepath = Path(outpath) / f"{filename}.pickle"
    tic = time.time()
    pickle_out = open(filepath, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    toc = time.time() - tic
    print(f"Saved pickle {filepath} ({toc:.3f} seconds).")
    return str(filepath)


def load_pickle(filepath: str):
    """Import data from pickle"""
    tic = time.time()
    pickle_in = open(filepath, "rb")
    data = pickle.load(pickle_in)
    toc = time.time() - tic
    print(f"Loaded pickle {filepath} ({toc:.3f} seconds).")
    return data


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
