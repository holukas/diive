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

    Requests are retried up to 5 times on connection errors and on HTTP 429 and
    5xx, with exponential back-off. POST is retried too: InfluxDB writes,
    deletes and queries are all POST, and repeating them is safe (a point
    written twice is stored once). The synchronous writes of ``upload_singlevar``
    rely on these retries.
    """
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write.retry import WritesRetry
    from urllib3.util.retry import Retry
    return InfluxDBClient(url=conf_db['url'], token=conf_db['token'], org=conf_db['org'],
                          timeout=999_000, enable_gzip=True,
                          retries=WritesRetry(total=5, retry_interval=5, exponential_base=2,
                                              max_retry_delay=30,
                                              allowed_methods=Retry.DEFAULT_ALLOWED_METHODS | {'POST'}))


def get_query_api(client):
    return client.query_api()


def get_delete_api(client):
    return client.delete_api()
