"""
CORE.IO.DB.INFLUX.CONNECTION: INFLUXDB CLIENT FACTORY
=====================================================

Thin factory helpers around ``influxdb-client``. The client library is imported
*lazily* inside the functions so that importing this module (and the rest of
``diive.core.io.db``) never requires the optional ``db`` extra; only
actually opening a connection does.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import cache

# HTTP read timeout of one request, and the retry settings of get_client.
TIMEOUT_S = 999
RETRY_TOTAL = 5
RETRY_MAX_DELAY_S = 30
# Time budget of one request, counted from its first failure: long enough for
# all retries of a request that runs into the read timeout every time, so the
# number of retries is the limit, not the time.
RETRY_MAX_TIME_S = RETRY_TOTAL * (TIMEOUT_S + RETRY_MAX_DELAY_S)


@cache
def _per_request_retry_class():
    """Return a WritesRetry subclass that gives each request its own time budget.

    WritesRetry sets its deadline (``retry_timeout``) once, in its constructor,
    and ``new()`` copies it. urllib3 starts every request from the same retry
    object, so the deadline would count from the creation of the client and
    stop all retries after ``max_retry_time``, however long the client is used.
    Here the deadline starts at the first failure of each request instead.
    Built on first use because ``influxdb_client`` is an optional dependency.
    """
    from influxdb_client.client.write.retry import WritesRetry

    class PerRequestWritesRetry(WritesRetry):
        def increment(self, method=None, url=None, response=None, error=None,
                      _pool=None, _stacktrace=None):
            if not self.history:
                # First failure of this request. Work on a copy: self is shared
                # by all requests of the connection pool.
                fresh = self.new()
                fresh.retry_timeout = datetime.now() + timedelta(seconds=self.max_retry_time)
                return WritesRetry.increment(fresh, method, url, response, error, _pool, _stacktrace)
            return super().increment(method, url, response, error, _pool, _stacktrace)

    return PerRequestWritesRetry


def get_client(conf_db: dict):
    """Build an :class:`influxdb_client.InfluxDBClient` from a connection dict.

    *conf_db* must contain ``url``, ``token`` and ``org`` (read from
    ``<dirconf>_secret/dbconf.yaml``).

    Each request is retried up to 5 times on connection errors, read timeouts
    (999 s) and HTTP status 429 or higher, with exponential back-off (5 s to
    30 s). POST is retried too: InfluxDB writes, deletes and queries are all
    POST, and repeating them is safe (a point written twice is stored once).
    The retries of a request do not depend on how long the client has been
    open, so a long upload through one client is retried as well. The
    synchronous writes of ``upload_singlevar`` rely on these retries.
    """
    from influxdb_client import InfluxDBClient
    from urllib3.util.retry import Retry
    retry_cls = _per_request_retry_class()
    return InfluxDBClient(url=conf_db['url'], token=conf_db['token'], org=conf_db['org'],
                          timeout=TIMEOUT_S * 1000, enable_gzip=True,
                          retries=retry_cls(total=RETRY_TOTAL, retry_interval=5, exponential_base=2,
                                            max_retry_delay=RETRY_MAX_DELAY_S,
                                            max_retry_time=RETRY_MAX_TIME_S,
                                            allowed_methods=Retry.DEFAULT_ALLOWED_METHODS | {'POST'}))


def get_query_api(client):
    return client.query_api()


def get_delete_api(client):
    return client.delete_api()
