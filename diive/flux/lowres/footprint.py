"""
FOOTPRINT: FLUX FOOTPRINT PREDICTION (FFP) OF KLJUN ET AL. (2015)
=================================================================

Two-dimensional flux footprint of a single record and footprint climatology of many records.

Part of the diive library: https://github.com/holukas/diive

The model is the simple parameterisation of Kljun et al. (2015). Both functions reproduce
the Python implementation of the authors (FFP version 1.42) bitwise on valid input:

- ``ffp_footprint`` follows ``FFP``: crosswind-integrated footprint on the scaled grid,
  the 2-D footprint in the along-wind frame, optionally rotated into the wind direction.
- ``ffp_climatology`` follows ``FFP_climatology``: footprints on a fixed physical grid,
  rotated per record, averaged and smoothed twice with a 3x3 kernel.

Deliberate deviations, none of them changes the result of valid input:

- Invalid input raises ``ValueError`` in ``ffp_footprint`` (the reference prints a
  message and continues with the invalid values). ``ffp_climatology`` skips invalid
  records as the reference does, and also records with a missing (NaN) input or with
  ``log(zm / z0) - psi_f <= 0``, where the reference stops with an exception. Records with
  ``ol = 0`` or ``umean <= 0`` are invalid (the reference stops or returns a meaningless
  footprint).
- ``rs`` values above 90 % are dropped with a warning in both functions (the reference
  ``FFP`` stops instead).
- Cropping keeps the orientation of the uncropped output and keeps the index ranges
  inside the grid. The reference ``FFP`` returns cropped arrays transposed, and both
  reference functions stop when the cropped range reaches the edge of the grid.
- Contour lines come from contourpy with the settings of ``matplotlib.pyplot.contour``
  (same vertices), without creating a pyplot figure.

Reference:
    Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple two-dimensional
    parameterisation for Flux Footprint Prediction (FFP). Geoscientific Model Development,
    8(11), 3695-3713. https://doi.org/10.5194/gmd-8-3695-2015
"""

import numbers
from dataclasses import dataclass, field

import numpy as np
from numpy import ma

from diive.core.utils.console import Progress, warn

# Model parameters (Kljun et al. 2015, Eqs. 14, 17, 19)
_A = 1.4524
_B = -1.9914
_C = 1.4622
_D = 0.1359
_AC = 2.17
_BC = 1.66
_CC = 20.0
_XSTAR_END = 30
_OLN = 5000  # Limit of |L| for neutral scaling
_K = 0.4  # von Karman constant

DEFAULT_RS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class FFPFootprint:
    """Footprint of a single record (``ffp_footprint``).

    Attributes:
        x_ci_max: Distance of the footprint peak from the tower (m).
        x_ci: Distances of the crosswind-integrated footprint (m).
        f_ci: Crosswind-integrated footprint (m-1).
        x_2d: x grid of the 2-D footprint (m), rotated if a wind direction was given.
            Axis 0 runs along the wind, axis 1 across it.
        y_2d: y grid of the 2-D footprint (m), rotated if a wind direction was given.
        f_2d: 2-D footprint (m-2).
        rs: Source-area fractions of the contour lines, ascending.
        fr: Footprint value of each contour line, None where the line reaches the
            edge of the grid.
        xr: x vertices of each contour line (m), None where ``fr`` is None.
        yr: y vertices of each contour line (m), None where ``fr`` is None.
    """
    x_ci_max: float
    x_ci: np.ndarray = field(repr=False)
    f_ci: np.ndarray = field(repr=False)
    x_2d: np.ndarray = field(repr=False)
    y_2d: np.ndarray = field(repr=False)
    f_2d: np.ndarray = field(repr=False)
    rs: list | None = None
    fr: list | None = None
    xr: list | None = field(default=None, repr=False)
    yr: list | None = field(default=None, repr=False)


@dataclass(frozen=True)
class FFPClimatology:
    """Footprint climatology of many records (``ffp_climatology``).

    Attributes:
        x_2d: x grid (m, positive towards east), from ``numpy.meshgrid``.
        y_2d: y grid (m, positive towards north).
        fclim_2d: Mean footprint of all valid records (m-2), smoothed if requested.
        n: Number of footprints in the climatology.
        valid: Per record, True if its footprint is in the climatology.
        rs: Source-area fractions of the contour lines, ascending.
        fr: Footprint value of each contour line, None where the line reaches the
            edge of the domain.
        xr: x vertices of each contour line (m), None where ``fr`` is None.
        yr: y vertices of each contour line (m), None where ``fr`` is None.
    """
    x_2d: np.ndarray = field(repr=False)
    y_2d: np.ndarray = field(repr=False)
    fclim_2d: np.ndarray = field(repr=False)
    n: int
    valid: np.ndarray = field(repr=False)
    rs: list | None = None
    fr: list | None = None
    xr: list | None = field(default=None, repr=False)
    yr: list | None = field(default=None, repr=False)


def ffp_footprint(zm: float, h: float, ol: float, sigmav: float, ustar: float, *,
                  z0: float = None, umean: float = None, wind_dir: float = None,
                  rs=DEFAULT_RS, rslayer: bool = False, nx: int = 1000,
                  crop: bool = False, verbose: int | bool | None = None) -> FFPFootprint:
    """Flux footprint of a single record (Kljun et al. 2015).

    Use it also to find a domain large enough for ``ffp_climatology``.

    Args:
        zm: Measurement height above displacement height, z - d (m).
        h: Boundary-layer height (m), above 10 m.
        ol: Obukhov length (m).
        sigmav: Standard deviation of the lateral wind velocity (m s-1).
        ustar: Friction velocity (m s-1), above 0.1.
        z0: Roughness length (m). Either ``z0`` or ``umean`` is needed. If both are
            given, ``z0`` is used.
        umean: Mean wind speed at ``zm`` (m s-1).
        wind_dir: Wind direction (degrees, 0-360) to rotate the footprint into.
            None keeps the footprint along the positive x axis.
        rs: Source-area fraction(s) for contour lines, as fractions (0.8) or percent
            (80), at most 90 %. A single value above 90 % becomes 90 %. None gives no
            contour lines.
        rslayer: If True, also calculate the footprint when ``zm`` lies in the
            roughness sublayer (``zm <= 12.5 * z0``), where the model gives only a
            rough estimate.
        nx: Number of grid elements of the scaled footprint, at least 600.
        crop: Crop the grid to the largest contour line in ``rs`` (80 % if ``rs`` is
            None).
        verbose: Verbosity of warnings.

    Returns:
        FFPFootprint

    Raises:
        ValueError: If an input is missing or outside the range of the model.
    """
    if umean is not None and z0 is not None:
        umean = None
    _check_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbose,
                  wind_dir_required=False)
    if nx < 600:
        raise ValueError("nx must be at least 600.")
    rs = _normalize_rs(rs, verbose)

    # Scaled X* and crosswind-integrated scaled F*
    xstar_ci_param = np.linspace(_D, _XSTAR_END, nx + 2)
    xstar_ci_param = xstar_ci_param[1:]
    fstar_ci_param = _A * (xstar_ci_param - _D) ** _B * np.exp(-_C / (xstar_ci_param - _D))
    ind_notnan = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]

    # Scaled sig_y*
    sigystar_param = _AC * np.sqrt(_BC * xstar_ci_param ** 2 / (1 + _CC * xstar_ci_param))

    # Real-scale x and f_ci
    if z0 is not None:
        psi_f = _psi_f(zm, ol)
        if not np.log(zm / z0) - psi_f > 0:
            raise ValueError("log(zm / z0) - psi_f must be positive.")
        x = xstar_ci_param * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
        x_ci = x
        f_ci = fstar_ci_param / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f)
    else:
        x = xstar_ci_param * zm / (1. - zm / h) * (umean / ustar * _K)
        x_ci = x
        f_ci = fstar_ci_param / zm * (1. - zm / h) / (umean / ustar * _K)

    # Peak location
    xstarmax = -_C / _B + _D
    if z0 is not None:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
    else:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (umean / ustar * _K)

    # Real-scale sig_y
    sigy = sigystar_param / _scale_const(zm, ol) * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan

    # Real-scale f(x, y). Row by row: the vectorized form differs in the last bit.
    dx = x_ci[2] - x_ci[1]
    y_pos = np.arange(0, (len(x_ci) / 2.) * dx * 1.5, dx)
    f_pos = np.empty((len(f_ci), len(y_pos)))
    for ix in range(len(f_ci)):
        f_pos[ix, :] = f_ci[ix] * 1 / (np.sqrt(2 * np.pi) * sigy[ix]) * np.exp(-y_pos ** 2 / (2 * sigy[ix] ** 2))

    # Negative y by symmetry
    y = np.concatenate((-y_pos[::-1][:-1], y_pos))
    f_2d = np.concatenate((f_pos[:, ::-1][:, :-1], f_pos), axis=1)
    x_2d = np.tile(x[:, None], (1, len(y)))
    y_2d = np.tile(y, (len(x), 1))

    # Contour lines of the R% source areas, starting at the peak
    dy = dx
    frs = xrs = yrs = None
    if rs is not None:
        frs, xrs, yrs = _contours(x_2d, y_2d, f_2d, dx, dy, rs)

    if crop:
        crop_x, crop_y = _crop_contour(x_2d, y_2d, f_2d, dx, dy, rs, xrs, yrs)
        if crop_x is None:
            warn("No contour line inside the grid, the footprint is not cropped.", verbose=verbose)
        else:
            irange = _crop_range(x_2d[:, 0], crop_x, len(x_2d[:, 0]))
            jrange = _crop_range(y_2d[0], crop_y, len(y_2d[0]))
            ix = np.ix_(irange, jrange)
            x_2d, y_2d, f_2d = x_2d[ix], y_2d[ix], f_2d[ix]

    # Rotate into the wind direction
    if wind_dir is not None:
        wind_dir = wind_dir * np.pi / 180.
        x_2d, y_2d = _rotate(x_2d, y_2d, wind_dir)
        if rs is not None:
            for i in range(len(rs)):
                if xrs[i] is None:
                    continue
                xrs[i], yrs[i] = _rotate(xrs[i], yrs[i], wind_dir)

    return FFPFootprint(x_ci_max=x_ci_max, x_ci=x_ci, f_ci=f_ci, x_2d=x_2d, y_2d=y_2d,
                        f_2d=f_2d, rs=rs, fr=frs, xr=xrs, yr=yrs)


def ffp_climatology(zm, h, ol, sigmav, ustar, wind_dir, *, z0=None, umean=None,
                    domain=None, dx: float = None, dy: float = None, nx: int = None,
                    ny: int = None, rs=DEFAULT_RS, rslayer: bool = False,
                    smooth_data: bool = True, crop: bool = False,
                    verbose: int | bool | None = None) -> FFPClimatology:
    """Footprint climatology of many records (Kljun et al. 2015).

    Calculates the footprint of each record on a fixed grid around the tower at
    (0, 0), rotated into its wind direction, and averages all valid footprints.
    Records with a missing or invalid input are skipped.

    All per-record inputs take a scalar or a sequence (list, array, Series) of one
    value per record. ``zm``, ``z0`` and ``umean`` may also be a single value for
    all records.

    Args:
        zm: Measurement height above displacement height, z - d (m).
        h: Boundary-layer height (m), above 10 m.
        ol: Obukhov length (m).
        sigmav: Standard deviation of the lateral wind velocity (m s-1).
        ustar: Friction velocity (m s-1), above 0.1.
        wind_dir: Wind direction (degrees, 0-360).
        z0: Roughness length (m). Either ``z0`` or ``umean`` is needed. If ``z0`` is
            given, ``umean`` is ignored.
        umean: Mean wind speed at ``zm`` (m s-1).
        domain: Domain ``[xmin, xmax, ymin, ymax]`` (m). Takes precedence over
            ``nx``/``ny``.
        dx: Cell size in x (m). ``dy`` defaults to ``dx``. Takes precedence over
            ``nx``/``ny`` when ``domain`` is given.
        dy: Cell size in y (m).
        nx: Number of cells in x. ``ny`` defaults to ``nx``.
        ny: Number of cells in y.
        rs: Source-area fraction(s) for contour lines, as in ``ffp_footprint``.
        rslayer: If True, keep records with ``zm`` in the roughness sublayer.
        smooth_data: Smooth the climatology twice with a 3x3 kernel.
        crop: Crop the grid to the largest contour line in ``rs`` (80 % if ``rs`` is
            None).
        verbose: Verbosity of progress lines and warnings.

    Without ``domain``, ``dx`` and ``nx`` the grid is 2 km x 2 km with 2 m cells.
    Without ``domain`` but with ``dx`` and ``nx`` it is centred on the tower. With
    only one of them the domain is 2 km x 2 km.

    Returns:
        FFPClimatology

    Raises:
        ValueError: If no record gives a valid footprint, or the inputs have
            different lengths.
    """
    zms, hs, ols, sigmavs, ustars, wind_dirs = [np.asarray(v, dtype=float).ravel()
                                                for v in (zm, h, ol, sigmav, ustar, wind_dir)]
    ts_len = len(ustars)
    if any(len(v) != ts_len for v in (sigmavs, wind_dirs, hs, ols)):
        raise ValueError("h, ol, sigmav, ustar and wind_dir must have the same length.")
    zms = _per_record(zms, ts_len, 'zm')
    if z0 is not None:
        z0s = _per_record(np.asarray(z0, dtype=float).ravel(), ts_len, 'z0')
        umeans = None
    elif umean is not None:
        umeans = _per_record(np.asarray(umean, dtype=float).ravel(), ts_len, 'umean')
        z0s = None
    else:
        raise ValueError("Either z0 or umean is needed.")

    rs = _normalize_rs(rs, verbose)
    xmin, xmax, ymin, ymax, dx, dy, nx, ny = _resolve_domain(domain, dx, dy, nx, ny)

    # Cartesian and polar grid, north up and angles clockwise
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)
    rho = np.sqrt(x_2d ** 2 + y_2d ** 2)
    theta = np.arctan2(x_2d, y_2d)
    fclim_2d = np.zeros(x_2d.shape)

    valid = np.zeros(ts_len, dtype=bool)
    progress = Progress("Footprint climatology", ts_len, verbose=verbose)
    for i in range(ts_len):
        progress.step()
        # Python floats, as the reference receives them from lists
        ustar_i, sigmav_i, h_i, ol_i = float(ustars[i]), float(sigmavs[i]), float(hs[i]), float(ols[i])
        wind_dir_i, zm_i = float(wind_dirs[i]), float(zms[i])
        z0_i = None if z0s is None else float(z0s[i])
        umean_i = None if umeans is None else float(umeans[i])
        try:
            _check_inputs(ustar_i, sigmav_i, h_i, ol_i, wind_dir_i, zm_i, z0_i, umean_i,
                          rslayer, verbose=0, wind_dir_required=True)
        except ValueError:
            continue

        rotated_theta = theta - wind_dir_i * np.pi / 180.

        # Real-scale crosswind-integrated footprint
        if z0_i is not None:
            psi_f = _psi_f(zm_i, ol_i)
            if not (np.log(zm_i / z0_i) - psi_f) > 0:
                continue
            xstar = (rho * np.cos(rotated_theta) / zm_i * (1. - (zm_i / h_i))
                     / (np.log(zm_i / z0_i) - psi_f))
            px = xstar > _D
            xs = xstar[px]
            fstar = _A * (xs - _D) ** _B * np.exp(-_C / (xs - _D))
            f_ci = fstar / zm_i * (1. - (zm_i / h_i)) / (np.log(zm_i / z0_i) - psi_f)
        else:
            xstar = (rho * np.cos(rotated_theta) / zm_i * (1. - (zm_i / h_i))
                     / (umean_i / ustar_i * _K))
            px = xstar > _D
            xs = xstar[px]
            fstar = _A * (xs - _D) ** _B * np.exp(-_C / (xs - _D))
            f_ci = fstar / zm_i * (1. - (zm_i / h_i)) / (umean_i / ustar_i * _K)

        # Scaled sig_y* and real-scale sig_y
        sigystar = _AC * np.sqrt(_BC * np.abs(xs) ** 2 / (1 + _CC * np.abs(xs)))
        sigy = sigystar / _scale_const(zm_i, ol_i) * zm_i * sigmav_i / ustar_i
        sigy[sigy < 0] = np.nan

        # Real-scale f(x, y). Cells outside px add zero, as in the reference.
        f_2d = (f_ci / (np.sqrt(2 * np.pi) * sigy)
                * np.exp(-(rho[px] * np.sin(rotated_theta[px])) ** 2 / (2. * sigy ** 2)))
        fclim_2d[px] = fclim_2d[px] + f_2d
        valid[i] = True

    n = int(valid.sum())
    progress.done(f"{n} valid footprints")
    if n == 0:
        raise ValueError("No valid footprint was calculated.")
    if n < ts_len:
        warn(f"{ts_len - n} of {ts_len} records skipped (missing or invalid input).",
             verbose=verbose)

    fclim_2d = fclim_2d / n
    if smooth_data:
        from scipy.signal import convolve2d
        skernel = np.array([[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]])
        fclim_2d = convolve2d(fclim_2d, skernel, mode='same')
        fclim_2d = convolve2d(fclim_2d, skernel, mode='same')

    frs = xrs = yrs = None
    if rs is not None:
        frs, xrs, yrs = _contours(x_2d, y_2d, fclim_2d, dx, dy, rs)
        if any(fr is None for fr in frs):
            warn("Not all contour lines lie inside the domain.", verbose=verbose)

    if crop:
        crop_x, crop_y = _crop_contour(x_2d, y_2d, fclim_2d, dx, dy, rs, xrs, yrs)
        if crop_x is None:
            warn("No contour line inside the domain, the climatology is not cropped.",
                 verbose=verbose)
        else:
            jrange = _crop_range(y_2d[:, 0], crop_y, y_2d.shape[0])
            irange = _crop_range(x_2d[0, :], crop_x, x_2d.shape[1])
            ix = np.ix_(jrange, irange)
            x_2d, y_2d, fclim_2d = x_2d[ix], y_2d[ix], fclim_2d[ix]

    return FFPClimatology(x_2d=x_2d, y_2d=y_2d, fclim_2d=fclim_2d, n=n, valid=valid,
                          rs=rs, fr=frs, xr=xrs, yr=yrs)


def _check_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbose,
                  wind_dir_required):
    """Raise ValueError for missing input or input outside the model's range."""
    required = [zm, h, ol, sigmav, ustar] + ([wind_dir] if wind_dir_required else [])
    if any(v is None or np.isnan(v) for v in required):
        raise ValueError("zm, h, ol, sigmav and ustar are required.")
    if wind_dir is not None and np.isnan(wind_dir):
        raise ValueError("wind_dir is missing.")
    if (z0 is None or np.isnan(z0)) and (umean is None or np.isnan(umean)):
        raise ValueError("Either z0 or umean is needed.")
    if zm <= 0.:
        raise ValueError("zm (measurement height) must be larger than zero.")
    if z0 is not None and z0 <= 0.:
        raise ValueError("z0 (roughness length) must be larger than zero.")
    if umean is not None and umean <= 0.:
        raise ValueError("umean (mean wind speed) must be larger than zero.")
    if h <= 10.:
        raise ValueError("h (boundary-layer height) must be larger than 10 m.")
    if zm > h:
        raise ValueError("zm (measurement height) must be smaller than h (boundary-layer height).")
    if z0 is not None and zm <= 12.5 * z0:
        if not rslayer:
            raise ValueError("zm (measurement height) must be above the roughness sublayer "
                             "(12.5 * z0), or set rslayer=True.")
        warn("zm (measurement height) lies in the roughness sublayer (12.5 * z0), "
             "the footprint is only a rough estimate.", verbose=verbose)
    if ol == 0:
        raise ValueError("ol (Obukhov length) must not be zero.")
    if float(zm) / ol <= -15.5:
        raise ValueError("zm / ol must be larger than -15.5.")
    if sigmav <= 0:
        raise ValueError("sigmav must be larger than zero.")
    if ustar <= 0.1:
        raise ValueError("ustar must be larger than 0.1 m s-1.")
    if wind_dir is not None and (wind_dir > 360 or wind_dir < 0):
        raise ValueError("wind_dir must be between 0 and 360.")


def _psi_f(zm, ol):
    """Stability correction of the wind profile (unstable/neutral or stable)."""
    if ol <= 0 or ol >= _OLN:
        xx = (1 - 19.0 * zm / ol) ** 0.25
        return np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi / 2
    return -5.3 * zm / ol


def _scale_const(zm, ol):
    """Scaling constant of sig_y (convective or stable)."""
    if abs(ol) > _OLN:
        ol = -1E6
    if ol <= 0:
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
    else:
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.55
    return min(scale_const, 1.0)


def _normalize_rs(rs, verbose):
    """Source-area fractions as an ascending list, at most 0.9."""
    if rs is None:
        return None
    if isinstance(rs, numbers.Number):
        if 0.9 < rs <= 1 or 90 < rs <= 100:
            rs = 0.9
        rs = [rs]
    rs = list(rs)
    if np.max(rs) >= 1:
        rs = [r / 100. for r in rs]
    if np.max(rs) > 0.9:
        warn("rs values above 90 % were dropped.", verbose=verbose)
        rs = [r for r in rs if r <= 0.9]
    return list(np.sort(rs))


def _per_record(values, ts_len, name):
    """One value per record, from a single value or a sequence."""
    if len(values) == 1:
        return np.full(ts_len, values[0])
    if len(values) != ts_len:
        raise ValueError(f"{name} must be a single value or have one value per record.")
    return values


def _resolve_domain(domain, dx, dy, nx, ny):
    """Domain bounds, cell sizes and cell counts, with the reference's precedence."""
    if isinstance(dx, numbers.Number) and dy is None:
        dy = dx
    if isinstance(dy, numbers.Number) and dx is None:
        dx = dy
    if not all(isinstance(v, numbers.Number) for v in (dx, dy)):
        dx = dy = None
    if isinstance(nx, numbers.Integral) and ny is None:
        ny = nx
    if isinstance(ny, numbers.Integral) and nx is None:
        nx = ny
    if not all(isinstance(v, numbers.Integral) for v in (nx, ny)):
        nx = ny = None
    if domain is not None:
        domain = list(domain)
        if len(domain) != 4:
            raise ValueError("domain must be [xmin, xmax, ymin, ymax].")

    if dx is None and nx is None and domain is None:
        domain = [-1000., 1000., -1000., 1000.]
        dx = dy = 2.
        nx = ny = 1000
    elif domain is not None:
        if dx is not None:
            nx = int((domain[1] - domain[0]) / dx)
            ny = int((domain[3] - domain[2]) / dy)
        else:
            if nx is None:
                nx = ny = 1000
            dx = (domain[1] - domain[0]) / float(nx)
            dy = (domain[3] - domain[2]) / float(ny)
    elif dx is not None and nx is not None:
        domain = [-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2]
    elif dx is not None:
        domain = [-1000, 1000, -1000, 1000]
        nx = int((domain[1] - domain[0]) / dx)
        ny = int((domain[3] - domain[2]) / dy)
    else:
        domain = [-1000, 1000, -1000, 1000]
        dx = (domain[1] - domain[0]) / float(nx)
        dy = (domain[3] - domain[2]) / float(nx)
    return (*domain, dx, dy, nx, ny)


def _contour_levels(f, dx, dy, rs):
    """Footprint value at which the cumulative footprint, from the peak down, reaches r."""
    sf = np.sort(f, axis=None)[::-1]
    msf = ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))
    csf = msf.cumsum().filled(np.nan) * dx * dy
    return [sf[np.nanargmin(np.abs(csf - r))] for r in rs]


def _contour_vertices(x, y, f, lev):
    """First contour line of f at lev, None if it reaches the edge of the grid."""
    import contourpy
    # The settings of matplotlib.pyplot.contour (default rcParams)
    gen = contourpy.contour_generator(x, y, ma.masked_invalid(f, copy=False), name='mpl2014',
                                      corner_mask=True, line_type=contourpy.LineType.SeparateCode)
    lines, _ = gen.lines(lev)
    if not lines:
        return None, None
    segs = lines[0]
    if x.min() >= segs[:, 0].min() or segs[:, 0].max() >= x.max() or \
            y.min() >= segs[:, 1].min() or segs[:, 1].max() >= y.max():
        return None, None
    return segs[:, 0], segs[:, 1]


def _contours(x_2d, y_2d, f_2d, dx, dy, rs):
    """Contour values and vertices of the source areas rs."""
    frs = _contour_levels(f_2d, dx, dy, rs)
    xrs, yrs = [], []
    for i, fr in enumerate(frs):
        xr, yr = _contour_vertices(x_2d, y_2d, f_2d, fr)
        if xr is None:
            frs[i] = None
        xrs.append(xr)
        yrs.append(yr)
    return frs, xrs, yrs


def _crop_contour(x_2d, y_2d, f_2d, dx, dy, rs, xrs, yrs):
    """Vertices of the contour to crop to: the largest r inside the grid, else None."""
    if rs is None:
        return _contour_vertices(x_2d, y_2d, f_2d, _contour_levels(f_2d, dx, dy, [0.8])[0])
    inside = [i for i, xr in enumerate(xrs) if xr is not None]
    if not inside:
        return None, None
    return xrs[inside[-1]], yrs[inside[-1]]


def _crop_range(coords, vertices, n):
    """Indices of coords within the floored/ceiled vertex range, plus one on each side."""
    lo, hi = np.floor(np.min(vertices)), np.ceil(np.max(vertices))
    idx = np.where((coords >= lo) & (coords <= hi))[0]
    idx = np.concatenate(([idx[0] - 1], idx, [idx[-1] + 1]))
    return idx[(idx >= 0) & (idx <= n - 1)]


def _rotate(x, y, wind_dir_rad):
    """Rotate along-wind coordinates into the wind direction (radians, clockwise from north)."""
    dist = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    return dist * np.sin(wind_dir_rad - angle), dist * np.cos(wind_dir_rad - angle)
