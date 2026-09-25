"""
QAQC.CODEGEN: RENDER A DATABASE METEO SCREENING AS A RUNNABLE SCRIPT
====================================================================

Turn the choices made for a :class:`~diive.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb`
run (e.g. in the GUI's Meteo screening tab) into a runnable script: download,
outlier tests, QCF, corrections and resampling.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import inspect

_DIRCONF_PLACEHOLDER = "PATH/TO/DIRCONF"


def _method_defaults(method: str) -> dict:
    """Default kwargs of a ``StepwiseMeteoScreeningDb.flag_*`` method, so the
    rendered call omits values left at their default."""
    from diive.preprocessing.qaqc.meteoscreening import StepwiseMeteoScreeningDb
    fn = getattr(StepwiseMeteoScreeningDb, method)
    return {p.name: p.default
            for p in inspect.signature(fn).parameters.values()
            if p.default is not inspect.Parameter.empty}


def _download_lines(download: dict | None, field: str, utc_offset: float) -> list[str]:
    """The block that sets ``data_detailed``: an ``InfluxIO.download`` call, or
    a marked placeholder when *download* is None."""
    if download is None:
        return ["# PLACEHOLDER: set data_detailed to your data, a dict {field: DataFrame}",
                "# as InfluxIO.download() returns it: TIMESTAMP_END index, the field",
                "# column and the database tag columns.",
                "data_detailed = ..."]
    dirconf = download.get("dirconf")
    dirconf_line = (f"dbc = InfluxIO(dirconf={dirconf!r})" if dirconf else
                    f"dbc = InfluxIO(dirconf={_DIRCONF_PLACEHOLDER!r})"
                    "  # PLACEHOLDER: folder with the database configs")
    lines = [f"# Download; start/stop and the timestamps are in UTC{utc_offset:+g}, "
             "stop is excluded",
             dirconf_line,
             "data_simple, data_detailed, assigned_measurements = dbc.download(",
             f"    bucket={download['bucket']!r},"]
    if download.get("measurement"):
        lines.append(f"    measurements=[{download['measurement']!r}],")
    lines += [f"    fields=[{field!r}],",
              f"    start={download['start']!r},",
              f"    stop={download['stop']!r},",
              f"    timezone_offset_to_utc_hours={utc_offset!r},"]
    if download.get("data_version"):
        lines.append(f"    data_version={download['data_version']!r},")
    lines.append(")")
    return lines


def meteoscreening_to_code(steps: list[dict], *, field: str, site: str,
                           site_lat: float, site_lon: float, utc_offset: float,
                           download: dict | None = None,
                           corrections: list[dict] | None = None,
                           to_freqstr: str = "30min", agg: str = "mean",
                           mincounts_perc: float = .25) -> str:
    """Render a ``StepwiseMeteoScreeningDb`` screening of one field as a runnable script.

    The script gets ``data_detailed`` (from ``InfluxIO.download`` or a marked
    placeholder), builds the class, runs each step followed by ``addflag()``,
    calls ``finalize_outlier_detection()``, applies the corrections with
    ``set_corrections()`` and resamples. Step kwargs left at the method's default
    are omitted.

    Args:
        steps: Ordered outlier tests, ``{"method": str, "kwargs": dict}``, where
            ``method`` is a ``flag_*`` method of ``StepwiseMeteoScreeningDb``.
            Steps with ``"enabled": False`` are skipped.
        field: The screened field (variable name in the database).
        site: Site name passed to the class.
        site_lat, site_lon: Site coordinates for day/night and potential radiation.
        utc_offset: Offset of the data timestamps to UTC in hours. It is used for
            the download and for the class, which must agree.
        download: ``{"bucket", "start", "stop"}`` and optionally ``"measurement"``,
            ``"data_version"`` and ``"dirconf"`` (config folder; a placeholder when
            missing) to render an ``InfluxIO.download`` call. ``start`` and ``stop``
            are date strings in the ``utc_offset`` timezone, ``stop`` is excluded.
            None renders a placeholder for ``data_detailed`` instead.
        corrections: Ordered ``{"key", "kwargs"}`` corrections for
            ``set_corrections()``; none are rendered when empty.
        to_freqstr, agg, mincounts_perc: Arguments of ``resample()``.

    Returns:
        The script as a string, ending in a newline.

    Example:
        >>> from diive.preprocessing.qaqc.codegen import meteoscreening_to_code
        >>> code = meteoscreening_to_code(
        ...     [{'method': 'flag_outliers_abslim_test', 'kwargs': {'minval': -30, 'maxval': 50}}],
        ...     field='TA_T1_2_1', site='ch-xyz', site_lat=47.29, site_lon=7.73, utc_offset=1,
        ...     download={'bucket': 'ch-xyz_raw', 'measurement': 'TA',
        ...               'start': '2024-07-01 00:10:00', 'stop': '2024-07-05 00:10:00',
        ...               'data_version': 'raw'},
        ...     corrections=[{'key': 'setto_max', 'kwargs': {'threshold': 40}}])
        >>> print(code)
        import diive as dv
        from diive.core.io.db.influx import InfluxIO  # needs: uv sync --group db
        <BLANKLINE>
        # Download; start/stop and the timestamps are in UTC+1, stop is excluded
        dbc = InfluxIO(dirconf='PATH/TO/DIRCONF')  # PLACEHOLDER: folder with the database configs
        data_simple, data_detailed, assigned_measurements = dbc.download(
            bucket='ch-xyz_raw',
            measurements=['TA'],
            fields=['TA_T1_2_1'],
            start='2024-07-01 00:10:00',
            stop='2024-07-05 00:10:00',
            timezone_offset_to_utc_hours=1,
            data_version='raw',
        )
        <BLANKLINE>
        mscr = dv.qaqc.StepwiseMeteoScreeningDb(
            data_detailed=data_detailed,
            fields='TA_T1_2_1',
            site='ch-xyz',
            site_lat=47.29,
            site_lon=7.73,
            utc_offset=1,
        )
        <BLANKLINE>
        # Outlier tests, each followed by addflag(), then the overall flag (QCF)
        mscr.start_outlier_detection()
        mscr.flag_outliers_abslim_test(
            minval=-30,
            maxval=50,
        )
        mscr.addflag()
        mscr.finalize_outlier_detection()
        <BLANKLINE>
        # Corrections
        mscr.set_corrections([
            {'key': 'setto_max', 'kwargs': {'threshold': 40}},
        ])
        <BLANKLINE>
        # Resample; the result has TIMESTAMP_END and the database tags
        mscr.resample(to_freqstr='30min', agg='mean', mincounts_perc=0.25)
        resampled = mscr.resampled_detailed['TA_T1_2_1']
        <BLANKLINE>
    """
    lines = ["import diive as dv"]
    if download is not None:
        lines.append("from diive.core.io.db.influx import InfluxIO  # needs: uv sync --group db")
    lines.append("")
    lines += _download_lines(download, field, utc_offset)
    lines += ["",
              "mscr = dv.qaqc.StepwiseMeteoScreeningDb(",
              "    data_detailed=data_detailed,",
              f"    fields={field!r},",
              f"    site={site!r},",
              f"    site_lat={site_lat!r},",
              f"    site_lon={site_lon!r},",
              f"    utc_offset={utc_offset!r},",
              ")",
              "",
              "# Outlier tests, each followed by addflag(), then the overall flag (QCF)",
              "mscr.start_outlier_detection()"]
    for step in steps:
        if not step.get("enabled", True):
            continue
        method = step["method"]
        defaults = _method_defaults(method)
        kwarg_lines = [f"    {k}={v!r}," for k, v in step.get("kwargs", {}).items()
                       if not (k in defaults and v == defaults[k])]
        lines += [f"mscr.{method}(", *kwarg_lines, ")"] if kwarg_lines else [f"mscr.{method}()"]
        lines.append("mscr.addflag()")
    lines.append("mscr.finalize_outlier_detection()")
    if corrections:
        lines += ["", "# Corrections", "mscr.set_corrections(["]
        lines += [f"    {{'key': {c['key']!r}, 'kwargs': {c.get('kwargs', {})!r}}},"
                  for c in corrections]
        lines.append("])")
    lines += ["",
              "# Resample; the result has TIMESTAMP_END and the database tags",
              f"mscr.resample(to_freqstr={to_freqstr!r}, agg={agg!r}, "
              f"mincounts_perc={mincounts_perc!r})",
              f"resampled = mscr.resampled_detailed[{field!r}]"]
    return "\n".join(lines) + "\n"
