"""
CORE.IO.DB.INFLUX.CONFIG: READ INFLUXDB CONFIG DIRECTORY
========================================================

Reading of the YAML configuration files that describe an InfluxDB connection and
the local file/unit conventions.

The configuration directory passed to :class:`~diive.core.io.db.influx.influxio.InfluxIO`
(``dirconf``) is expected to have the following layout::

    <dirconf>/
        dirs.yaml             # directory settings (passthrough)
        units.yaml            # raw-unit -> standardized-unit mapping
        filegroups/
            *.yaml            # one file per filetype definition
    <dirconf>_secret/         # sibling directory, NOT inside <dirconf>
        dbconf.yaml           # InfluxDB connection: url, token, org

The database connection file lives in a *sibling* directory named
``<dirconf>_secret`` so that secrets can be kept out of the (often
version-controlled) main config directory.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import yaml


def read_configfile(config_file) -> dict:
    """Load configuration from a single YAML file.

    kudos: https://stackoverflow.com/questions/57687058/yaml-safe-load-special-character-from-file

    Args:
        config_file: path to a YAML file.

    Returns:
        Parsed YAML as a dict.
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def get_conf_filetypes(folder: Path, ext: str = 'yaml') -> dict:
    """Read all filetype config files with extension *ext* found under *folder*.

    Each file is expected to contain a single top-level key (the filetype name).

    Args:
        folder: directory to search recursively for ``*.{ext}`` files.
        ext: file extension to match (without the dot).

    Returns:
        Mapping of filetype name -> filetype settings.
    """
    folder = str(folder)  # Required as string for os.walk
    conf_filetypes = {}
    for root, dirs, files in os.walk(folder):
        for f in files:
            if fnmatch.fnmatch(f, f'*.{ext}'):
                _filepath = Path(root) / f
                _dict = read_configfile(config_file=_filepath)
                _key = list(_dict.keys())[0]
                _vals = _dict[_key]
                conf_filetypes[_key] = _vals
    return conf_filetypes
