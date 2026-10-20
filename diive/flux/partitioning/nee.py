"""
PARTITIONING.NEE: ONE ENTRY POINT FOR THE FOUR NEE PARTITIONING PORTS
=====================================================================

Select the NEE partitioning with two switches, ``method`` (nighttime or
daytime) and ``mode`` (ONEFlux or REddyProc), the way ``FluxMDS(mode=...)``
selects the MDS gap-filling rules. ONEFlux, the approach used by ICOS and
FLUXNET2015, is the default.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from pandas import DataFrame, Series

from diive.flux.partitioning.daytime_oneflux import DaytimePartitioningOneFlux
from diive.flux.partitioning.daytime_reddyproc import DaytimePartitioningReddyProc
from diive.flux.partitioning.nighttime_oneflux import NighttimePartitioningOneFlux
from diive.flux.partitioning.nighttime_reddyproc import NighttimePartitioningReddyProc

PARTITION_METHODS = ('nighttime', 'daytime')
PARTITION_MODES = ('oneflux', 'reddyproc')

# Per (method, mode): the backend class, which PartitioningNEE input feeds each
# backend argument ({backend_arg: input_name}), the inputs it takes only when
# given (``optional``, same shape), whether the backend takes vpd_in_kpa, and
# the backend-specific options accepted via **backend_kwargs. Daytime
# REddyProc uses the gap-filled drivers throughout, so its ``ta`` and
# ``sw_in`` arguments get ta_f and sw_in_f.
_BACKENDS = {
    ('nighttime', 'oneflux'): dict(
        cls=NighttimePartitioningOneFlux,
        inputs={'ta': 'ta', 'sw_in': 'sw_in', 'nee_f': 'nee_f', 'ta_f': 'ta_f',
                'lat': 'lat'},
        vpd_unit=False, options=()),
    ('nighttime', 'reddyproc'): dict(
        cls=NighttimePartitioningReddyProc,
        inputs={'ta': 'ta', 'sw_in': 'sw_in', 'nee_f': 'nee_f', 'ta_f': 'ta_f',
                'lat': 'lat', 'lon': 'lon', 'utc_offset': 'utc_offset'},
        vpd_unit=False, options=()),
    ('daytime', 'oneflux'): dict(
        cls=DaytimePartitioningOneFlux,
        inputs={'ta': 'ta', 'sw_in': 'sw_in', 'ta_f': 'ta_f', 'sw_in_f': 'sw_in_f',
                'vpd': 'vpd'},
        optional={'nee_f': 'nee_f'},
        vpd_unit=True, options=('reject_alpha_at_start',)),
    ('daytime', 'reddyproc'): dict(
        cls=DaytimePartitioningReddyProc,
        inputs={'ta': 'ta_f', 'sw_in': 'sw_in_f', 'vpd': 'vpd', 'lat': 'lat',
                'lon': 'lon', 'utc_offset': 'utc_offset'},
        vpd_unit=True, options=('nee_sd',)),
}


def partitioning_inputs(method: str, mode: str = 'oneflux') -> tuple[str, ...]:
    """Names of the :class:`PartitioningNEE` inputs (besides ``nee``) a method/mode needs.

    Example: ``partitioning_inputs('daytime', 'reddyproc')`` returns
    ``('ta_f', 'sw_in_f', 'vpd', 'lat', 'lon', 'utc_offset')``.
    """
    _check_method_mode(method, mode)
    return tuple(_BACKENDS[(method, mode)]['inputs'].values())


def partitioning_optional_inputs(method: str, mode: str = 'oneflux') -> tuple[str, ...]:
    """Names of the inputs a method/mode takes when given, but does not need.

    ``('nee_f',)`` for daytime / oneflux (see :class:`PartitioningNEE`),
    ``('nee_sd',)`` for daytime / reddyproc (a backend option, passed as a
    keyword), ``()`` for the nighttime method.
    """
    _check_method_mode(method, mode)
    spec = _BACKENDS[(method, mode)]
    named = tuple(spec.get('optional', {}).values())
    series_options = tuple(o for o in spec['options'] if o == 'nee_sd')
    return named + series_options


def _check_method_mode(method: str, mode: str) -> None:
    if method not in PARTITION_METHODS:
        raise ValueError(f"method must be one of {PARTITION_METHODS}, got {method!r}.")
    if mode not in PARTITION_MODES:
        raise ValueError(f"mode must be one of {PARTITION_MODES}, got {mode!r}.")


class PartitioningNEE:
    """Partition NEE into GPP and RECO, with the method and mode as switches.

    One entry point for the four partitioning ports. ``method`` picks the
    algorithm: ``'nighttime'`` (Reichstein et al. 2005, temperature response of
    nighttime NEE) or ``'daytime'`` (Lasslop et al. 2010, light-response curve
    of daytime NEE). ``mode`` picks the reference implementation:
    ``'oneflux'`` (default, the ICOS and FLUXNET2015 approach) or
    ``'reddyproc'``. The work is done by the matching class, which is
    available as :attr:`backend`. Its columns carry its suffix:

    - nighttime / oneflux: :class:`NighttimePartitioningOneFlux`, ``*_NT_OF``
    - nighttime / reddyproc: :class:`NighttimePartitioningReddyProc`, ``*_NT_RP``
    - daytime / oneflux: :class:`DaytimePartitioningOneFlux`, ``*_DT_OF``
    - daytime / reddyproc: :class:`DaytimePartitioningReddyProc`, ``*_DT_RP``

    The results are those of the backend, unchanged. Their column names keep
    the backend suffix, so results of different modes can sit side by side in
    one DataFrame. :attr:`reco` and :attr:`gpp` return RECO and GPP for any
    method and mode.

    Inputs each method/mode needs (besides ``nee``):

    - nighttime / oneflux: ``ta``, ``sw_in``, ``nee_f``, ``ta_f``, ``lat``
    - nighttime / reddyproc: ``ta``, ``sw_in``, ``nee_f``, ``ta_f``, ``lat``,
      ``lon``, ``utc_offset``
    - daytime / oneflux: ``ta``, ``sw_in``, ``ta_f``, ``sw_in_f``, ``vpd``
    - daytime / reddyproc: ``ta_f``, ``sw_in_f``, ``vpd``, ``lat``, ``lon``,
      ``utc_offset``

    Daytime / oneflux also takes ``nee_f`` when given: ONEFlux reads the
    gap-filled NEE for windows without measured nighttime NEE (see
    :class:`DaytimePartitioningOneFlux`). Without it those windows are not
    fitted and a warning lists them. :func:`partitioning_optional_inputs`
    lists such inputs.

    Inputs the chosen method/mode does not use are ignored, so the same call
    can switch between modes. A missing input raises a ``ValueError`` that
    names it.

    Example:
        >>> import diive as dv
        >>> df = dv.load_exampledata_parquet()
        >>> part = dv.flux.PartitioningNEE(
        ...     nee=df['NEE_CUT_REF_orig'], method='daytime', mode='oneflux',
        ...     ta=df['Tair_orig'], sw_in=df['Rg_orig'], ta_f=df['Tair_f'],
        ...     sw_in_f=df['Rg_f'], vpd=df['VPD_f'], vpd_in_kpa=False).run()  # VPD_f in hPa
        >>> part.gpp  # GPP_DT_OF, and part.results holds all backend columns
    """

    def __init__(self,
                 nee: Series,
                 *,
                 method: str,
                 mode: str = 'oneflux',
                 ta: Series | None = None,
                 sw_in: Series | None = None,
                 ta_f: Series | None = None,
                 sw_in_f: Series | None = None,
                 nee_f: Series | None = None,
                 vpd: Series | None = None,
                 lat: float | None = None,
                 lon: float | None = None,
                 utc_offset: float | None = None,
                 vpd_in_kpa: bool = True,
                 verbose: int = 2,
                 **backend_kwargs):
        """
        Args:
            nee: Measured NEE (umol m-2 s-1), gaps as NaN.
            method: ``'nighttime'`` (Reichstein et al. 2005) or ``'daytime'``
                (Lasslop et al. 2010).
            mode: ``'oneflux'`` (default) or ``'reddyproc'``.
            ta: Measured air temperature (degC), gaps as NaN.
            sw_in: Measured incoming shortwave radiation (W m-2), gaps as NaN.
            ta_f: Gap-filled air temperature (degC).
            sw_in_f: Gap-filled incoming shortwave radiation (W m-2).
            nee_f: Gap-filled NEE (umol m-2 s-1), for the GPP residual of the
                nighttime method. Optional for daytime / oneflux, which reads
                it for windows without measured nighttime NEE, as ONEFlux
                does.
            vpd: Gap-filled vapour pressure deficit, for the daytime method.
                In kPa by default, see ``vpd_in_kpa``.
            lat: Site latitude in decimal degrees.
            lon: Site longitude in decimal degrees.
            utc_offset: Time zone offset of the timestamps from UTC in hours.
            vpd_in_kpa: If True (default), ``vpd`` is in kPa. Set False if it is
                in hPa. Used by the daytime method only.
            verbose: Console verbosity level (0 silent, 1 warnings, 2 progress
                + report, 3 debug). Default 2.
            **backend_kwargs: Options only one backend has, passed on to it:
                ``reject_alpha_at_start`` (daytime / oneflux, see
                :class:`DaytimePartitioningOneFlux`) and ``nee_sd`` (daytime /
                reddyproc, see :class:`DaytimePartitioningReddyProc`). Any
                other keyword raises a ``TypeError``.
        """
        _check_method_mode(method, mode)
        spec = _BACKENDS[(method, mode)]
        given = {'ta': ta, 'sw_in': sw_in, 'ta_f': ta_f, 'sw_in_f': sw_in_f,
                 'nee_f': nee_f, 'vpd': vpd, 'lat': lat, 'lon': lon,
                 'utc_offset': utc_offset}

        unknown = [k for k in backend_kwargs if k not in spec['options']]
        if unknown:
            accepted = ', '.join(spec['options']) or 'none'
            raise TypeError(
                f"PartitioningNEE(method={method!r}, mode={mode!r}) got unexpected "
                f"keyword argument(s): {', '.join(unknown)}. Options this backend "
                f"accepts: {accepted}.")

        needed = list(spec['inputs'].values())
        missing = [name for name in needed if given[name] is None]
        if nee is None:
            missing.insert(0, 'nee')
        if missing:
            raise ValueError(
                f"PartitioningNEE(method={method!r}, mode={mode!r}) needs "
                f"nee, {', '.join(needed)}. Missing: {', '.join(missing)}.")

        kwargs = {arg: given[name] for arg, name in spec['inputs'].items()}
        kwargs.update({arg: given[name] for arg, name in spec.get('optional', {}).items()
                       if given[name] is not None})
        if spec['vpd_unit']:
            kwargs['vpd_in_kpa'] = vpd_in_kpa
        kwargs.update(backend_kwargs)

        self.method = method
        self.mode = mode
        self.verbose = verbose
        self.backend = spec['cls'](nee=nee, verbose=verbose, **kwargs)

    def run(self) -> "PartitioningNEE":
        """Run the backend partitioning and populate :attr:`results`."""
        self.backend.run()
        return self

    @property
    def results(self) -> DataFrame:
        """The backend results DataFrame, columns with their backend suffix."""
        return self.backend.results

    @property
    def reco(self) -> Series:
        """Ecosystem respiration (umol m-2 s-1) of the chosen method/mode."""
        return self.backend.reco

    @property
    def gpp(self) -> Series:
        """Gross primary production (umol m-2 s-1) of the chosen method/mode."""
        return self.backend.gpp

    def report(self) -> None:
        """Print the backend's per-year summary of the partitioning result."""
        self.backend.report()


def partition_nee(nee: Series,
                  *,
                  method: str,
                  mode: str = 'oneflux',
                  ta: Series | None = None,
                  sw_in: Series | None = None,
                  ta_f: Series | None = None,
                  sw_in_f: Series | None = None,
                  nee_f: Series | None = None,
                  vpd: Series | None = None,
                  lat: float | None = None,
                  lon: float | None = None,
                  utc_offset: float | None = None,
                  vpd_in_kpa: bool = True,
                  verbose: int = 2,
                  **backend_kwargs) -> DataFrame:
    """Functional wrapper around :class:`PartitioningNEE`.

    See :class:`PartitioningNEE` for argument semantics.

    Returns:
        The backend results DataFrame (e.g. RECO_NT_OF, GPP_NT_OF, ...).
    """
    return PartitioningNEE(
        nee, method=method, mode=mode, ta=ta, sw_in=sw_in, ta_f=ta_f,
        sw_in_f=sw_in_f, nee_f=nee_f, vpd=vpd, lat=lat, lon=lon,
        utc_offset=utc_offset, vpd_in_kpa=vpd_in_kpa, verbose=verbose,
        **backend_kwargs).run().results
