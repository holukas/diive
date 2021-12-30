def validate_filetype_config(filetype_config: dict):
    """Convert to required types"""

    # GENERAL
    filetype_config['GENERAL']['NAME'] = str(filetype_config['GENERAL']['NAME'])
    filetype_config['GENERAL']['DESCRIPTION'] = str(filetype_config['GENERAL']['DESCRIPTION'])
    filetype_config['GENERAL']['TAGS'] = list(filetype_config['GENERAL']['TAGS'])

    # FILE
    filetype_config['FILE']['EXTENSION'] = str(filetype_config['FILE']['EXTENSION'])
    filetype_config['FILE']['COMPRESSION'] = str(filetype_config['FILE']['COMPRESSION'])

    # TIMESTAMP
    filetype_config['TIMESTAMP']['DESCRIPTION'] = str(filetype_config['TIMESTAMP']['DESCRIPTION'])

    filetype_config['TIMESTAMP']['INDEX_COLUMN'] = list(filetype_config['TIMESTAMP']['INDEX_COLUMN'])
    filetype_config['TIMESTAMP']['INDEX_COLUMN'] = \
        _convert_timestamp_idx_col(var=filetype_config['TIMESTAMP']['INDEX_COLUMN'])

    filetype_config['TIMESTAMP']['DATETIME_FORMAT'] = str(filetype_config['TIMESTAMP']['DATETIME_FORMAT'])
    filetype_config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'] = str(
        filetype_config['TIMESTAMP']['SHOWS_START_MIDDLE_OR_END_OF_RECORD'])

    # DATA
    filetype_config['DATA']['HEADER_SECTION_ROWS'] = list(filetype_config['DATA']['HEADER_SECTION_ROWS'])
    filetype_config['DATA']['SKIP_ROWS'] = list(filetype_config['DATA']['SKIP_ROWS'])
    filetype_config['DATA']['HEADER_ROWS'] = list(filetype_config['DATA']['HEADER_ROWS'])
    filetype_config['DATA']['NA_VALUES'] = list(filetype_config['DATA']['NA_VALUES'])
    filetype_config['DATA']['FREQUENCY'] = str(filetype_config['DATA']['FREQUENCY'])
    filetype_config['DATA']['DELIMITER'] = str(filetype_config['DATA']['DELIMITER'])

    return filetype_config


def _convert_timestamp_idx_col(var: int or list):
    """Convert to list of tuples if needed

    Since YAML is not good at processing list of tuples,
    they are given as list of lists,
        e.g. [ [ "date", "[yyyy-mm-dd]" ], [ "time", "[HH:MM]" ] ].
    In this case, convert to list of tuples,
        e.g.  [ ( "date", "[yyyy-mm-dd]" ), ( "time", "[HH:MM]" ) ].
    """
    new = var
    if isinstance(var[0], int):
        pass
    elif isinstance(var[0], list):
        for idx, c in enumerate(var):
            new[idx] = (c[0], c[1])
    return new
