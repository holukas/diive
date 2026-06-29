"""
CORE.IO.DB.INFLUX.CONNECTION: INFLUXDB CLIENT FACTORY
=====================================================

Thin factory helpers around ``influxdb-client``. The client library is imported
*lazily* inside the functions so that importing this module (and the rest of
``diive.core.io.db``) never requires the optional ``db`` dependency group; only
actually opening a connection does.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def get_client(conf_db: dict):
    """Build an :class:`influxdb_client.InfluxDBClient` from a connection dict.

    *conf_db* must contain ``url``, ``token`` and ``org`` (read from
    ``<dirconf>_secret/dbconf.yaml``).
    """
    from influxdb_client import InfluxDBClient
    return InfluxDBClient(url=conf_db['url'], token=conf_db['token'], org=conf_db['org'],
                          timeout=999_000, enable_gzip=True)


def get_query_api(client):
    return client.query_api()


def get_delete_api(client):
    return client.delete_api()
