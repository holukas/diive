from diive.core.utils.config import validate_filetype_config
from diive.core.io.filereader import ConfigFileReader


def load(file, ctx):
    """ Load example files from resources. """

    if file == 'EDDYPRO_FULL_OUTPUT_30MIN':
        configfilepath = ctx.filetype_EDDYPRO_FULL_OUTPUT_30MIN
        examplefilepath = ctx.file_EDDYPRO_FULL_OUTPUT_30MIN
    elif file == 'DIIVE_CSV_30MIN':
        configfilepath = ctx.filetype_DIIVE_CSV_30MIN
        examplefilepath = ctx.file_DIIVE_CSV_30MIN
    elif file == 'REDDYPROC_30MIN':
        configfilepath = ctx.filetype_REDDYPROC_30MIN
        examplefilepath = ctx.file_REDDYPROC_30MIN
    else:
        configfilepath = ctx.filetype_EDDYPRO_FULL_OUTPUT_30MIN
        examplefilepath = ctx.file_EDDYPRO_FULL_OUTPUT_30MIN

    filetype_config = ConfigFileReader(configfilepath=configfilepath).read()
    filetype_config = validate_filetype_config(filetype_config=filetype_config)


    # elif file == 'DIIVE_CSV_1H':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_DIIVE_CSV_1H)
    #     initial_data = ctx.file_DIIVE_CSV_1H
    # elif file == 'DIIVE_CSV_1MIN':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_DIIVE_CSV_1MIN)
    #     initial_data = ctx.file_DIIVE_CSV_1MIN
    # elif file == 'SMEAR_II_30MIN':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_SMEAR_II_30MIN)
    #     initial_data = ctx.file_SMEAR_II_30MIN
    # # elif file == 'EDDYPRO_FULL_OUTPUT_30MIN':
    # #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_EDDYPRO_FULL_OUTPUT_30MIN)
    # #     initial_data = ctx.file_EDDYPRO_FULL_OUTPUT_30MIN
    # # elif file == 'EDDYPRO_FULL_OUTPUT_30MIN':
    # #     settings_dict = ConfigFileReader(configfilepath=ctx.filetype_EDDYPRO_FULL_OUTPUT_30MIN).read()
    # #     # settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_EDDYPRO_FULL_OUTPUT_30MIN)
    # #     initial_data = ctx.file_EDDYPRO_FULL_OUTPUT_30MIN
    # elif file == 'FLUXNET_FULLSET_30MIN':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_FLUXNET_FULLSET_30MIN)
    #     initial_data = ctx.file_FLUXNET_FULLSET_30MIN
    # elif file == 'ICOS_10S':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_ICOS_10S)
    #     initial_data = ctx.file_ICOS_10S
    # elif file == 'Events':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_Events)
    #     initial_data = ctx.file_Events
    # elif file == 'ETH_METEOSCREENING_30MIN_FORMAT-A':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_ETH_METEOSCREENING_30MIN_FORMAT_A)
    #     initial_data = ctx.file_ETH_METEOSCREENING_30MIN_FORMAT_A
    # elif file == 'ETH_METEOSCREENING_30MIN_FORMAT-B':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_ETH_METEOSCREENING_30MIN_FORMAT_B)
    #     initial_data = ctx.file_ETH_METEOSCREENING_30MIN_FORMAT_B
    # elif file == 'REDDYPROC_30MIN':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_REDDYPROC_30MIN)
    #     initial_data = ctx.file_REDDYPROC_30MIN
    # elif file == 'TOA5_1MIN':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_TOA5_1MIN)
    #     initial_data = ctx.file_TOA5_1MIN
    # elif file == 'TOA5_10S':
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_TOA5_10S)
    #     initial_data = ctx.file_TOA5_10S
    #
    # else:
    #     settings_dict = parse_settingsfile_todict(filepath=ctx.filetype_DIIVE_CSV_30MIN)
    #     initial_data = ctx.file_DIIVE_CSV_30MIN  # from resources



    return examplefilepath, filetype_config
