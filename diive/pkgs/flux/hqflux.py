import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.pkgs.outlierdetection.hampel import HampelDaytimeNighttime
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorAllData


def _deprecated_analyze_highest_quality_flux(flux: Series, nighttime_flag: Series, showplot: bool = True):
    """
    Analyzes and filters highest-quality flux data by separating out outliers and calculating
    rolling median and standard deviation. The function operates on both daytime and nighttime
    flux data, applies a Local Outlier Factor (LOF) to identify outliers, and provides detailed
    output of both outliers and non-outliers within specified conditions. Optionally, it generates
    plots to visualize the results.

    Args:
        flux (Series): A pandas Series containing the flux data to be analyzed and filtered.
        nighttime_flag (Series): A pandas Series serving as a flag for nighttime (1) and daytime (0).
        showplot (bool): Indicates whether to display plots for the processed data. Defaults to True.

    Raises:
        KeyError: Raised when expected data columns in the input are unavailable.
        ValueError: Raised when the input flux or nighttime_flag Series are invalid or incompatible.
    """
    hqdf_filtered = pd.DataFrame(index=flux.index)
    for d in range(0, 2):
        timeofday = 'NIGHTTIME' if d == 1 else 'DAYTIME'

        hq = flux.loc[nighttime_flag == d].copy()
        # flux = self.flags.loc[self.nighttime == d, self.filteredseriescol_hq].copy()
        n_neighbors = int(hq.dropna().count() / 200)
        n_neighbors = n_neighbors if n_neighbors > 0 else 10
        contamination = 'auto'
        repeat = False
        print(f"\n>>> Removing outliers from highest-quality {timeofday} fluxes ({hq.name})")
        print(f">>> Outlier removal method: Local outlier factor across all data (n_neighbors={n_neighbors}, "
              f"contamination={contamination}, repeat={repeat})")
        lof = LocalOutlierFactorAllData(series=hq, n_neighbors=n_neighbors, contamination=contamination,
                                        showplot=showplot, verbose=True, n_jobs=-1)
        lof.calc(repeat=repeat)

        flag = lof.get_flag()
        _flags = pd.concat([hq, nighttime_flag, flag], axis=1)

        non_outlier_locs = (_flags[nighttime_flag.name] == d) & (_flags[flag.name] == 0)
        non_outlier_s = _flags.loc[non_outlier_locs, hq.name].copy()

        outlier_locs = (_flags[nighttime_flag.name] == d) & (_flags[flag.name] == 2)
        outlier_s = _flags.loc[outlier_locs, hq.name].copy()

        s_filtered = lof.filteredseries
        winsize = int(s_filtered.count() / 10)
        rmedian_filtered = s_filtered.rolling(window=winsize, center=True, min_periods=1).median()
        sd_filtered = s_filtered.std()

        hqdf_filtered[f'FLUX_{timeofday}'] = s_filtered.copy()
        hqdf_filtered[f'ROLLING_MEDIAN_{timeofday}'] = rmedian_filtered
        hqdf_filtered[f'SD_{timeofday}'] = sd_filtered
        hqdf_filtered[f'WINSIZE_{timeofday}'] = winsize

        non_outliers_s_above_zero = non_outlier_s[non_outlier_s >= 0].copy()
        print(f">>> Largest non-outlier flux >= 0 {timeofday}:   {non_outliers_s_above_zero.max()}")
        print(f">>> Smallest non-outlier flux >= 0 {timeofday}:  {non_outliers_s_above_zero.min()}")

        non_outliers_s_below_zero = non_outlier_s[non_outlier_s < 0].copy()
        print(f">>> Largest non-outlier flux < 0 {timeofday}:    {non_outliers_s_below_zero.max()}")
        print(f">>> Smallest non-outlier flux < 0 {timeofday}:   {non_outliers_s_below_zero.min()}")

        outliers_s_above_zero = outlier_s[outlier_s >= 0].copy()
        print(f">>> Largest outlier flux >= 0 {timeofday}:   {outliers_s_above_zero.max()}")
        print(f">>> Smallest outlier flux >= 0 {timeofday}:  {outliers_s_above_zero.min()}")

        outliers_s_below_zero = outlier_s[outlier_s < 0].copy()
        print(f">>> Largest outlier flux < 0 {timeofday}:    {outliers_s_below_zero.max()}")
        print(f">>> Smallest outlier flux < 0 {timeofday}:   {outliers_s_below_zero.min()}")

    if showplot:
        fig = plt.figure(facecolor='white', figsize=(16, 7))
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        # gs.update(wspace=0.3, hspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_nt = fig.add_subplot(gs[1, 0])

        for t in ['DAYTIME', 'NIGHTTIME']:
            t_ax = ax if t == 'DAYTIME' else ax_nt
            fluxcol = f'FLUX_{t}'
            rmediancol = f'ROLLING_MEDIAN_{t}'
            sdcol = f'SD_{t}'
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[fluxcol],
                      label=f"{t} flux", color="#607D8B", linestyle='none', markeredgewidth=1,
                      marker='o', alpha=.5, markersize=6, markeredgecolor="#607D8B", fillstyle='none')
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol],
                      label=f"rolling median", color="#FF6F00", linestyle='solid',
                      marker='none', alpha=.5, linewidth=3)
            style_sd = dict(linestyle='dashed', marker='none', alpha=.5, linewidth=3)
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol].add(hqdf_filtered[sdcol] * 3),
                      label=f"rolling median + 3 SD", color="#F44336", **style_sd)
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol].sub(hqdf_filtered[sdcol] * 3),
                      label=f"rolling median - 3 SD", color="#00BCD4", **style_sd)
            t_ax.axhline(hqdf_filtered[fluxcol].quantile(.99), linestyle='dotted', label="99th percentile",
                         color="#2196F3")
            t_ax.axhline(hqdf_filtered[fluxcol].quantile(.01), linestyle='dotted', label="1st percentile",
                         color="#9C27B0")
            pf.default_legend(ax=t_ax, labelspacing=0.2, ncol=3)
            # ax.set_ylim(hq.quantile(0.005), hq.quantile(0.995))

        fig.suptitle(f"Highest-quality fluxes {flux.name} after preliminary outlier removal", fontsize=16)
        fig.tight_layout()
        fig.show()


def analyze_highest_quality_flux(flux: Series,
                                 lat: float,
                                 lon: float,
                                 utc_offset: int,
                                 window_length: int = None,
                                 n_sigma_dt: float = 5.5,
                                 n_sigma_nt: float = 5.5,
                                 use_differencing: bool = True,
                                 showplot: bool = True,
                                 figsize: tuple = (16, 7),
                                 show_percentiles: bool = True,
                                 return_summary: bool = False):
    """
    Analyzes and filters highest-quality flux data using Hampel filter (robust Median Absolute Deviation).
    Separates daytime and nighttime automatically via solar geometry, applies Hampel outlier detection,
    and calculates rolling statistics.

    Hampel filter:
    - Uses Median Absolute Deviation (MAD), which is robust to extreme outliers
    - Optionally applies double-differencing (Papale et al. 2006) to remove biological trends
    - Automatically adapts thresholds for day/night turbulence regimes
    - Avoids user-defined parameters like n_neighbors and contamination

    Args:
        flux (Series): A pandas Series containing the flux data to be analyzed and filtered.
        lat (float): Latitude of measurement site in degrees (e.g., 47.286). Range: -90 to 90.
        lon (float): Longitude of measurement site in degrees (e.g., 7.734). Range: -180 to 180.
        utc_offset (int): UTC offset in hours (e.g., 1 for CET, -8 for PST).
        window_length (int): Rolling window size for Hampel filter.
            Default: None (auto = data_count / 100, minimum 13 for ~6 hours at 30-min frequency).
        n_sigma_dt (float): Hampel threshold for DAYTIME records (stricter for high turbulence).
            Default is 5.5 (robust). Use 4.0 for stricter filtering.
        n_sigma_nt (float): Hampel threshold for NIGHTTIME records (lenient for stable conditions).
            Default is 5.5. Use 2.5 for stricter filtering.
        use_differencing (bool): If True, applies double-differencing (Papale method) to remove
            trends and isolate spikes. Default True. Set False to detect outliers in raw values.
        showplot (bool): If True, displays outlier detection summary plots. Defaults to True.
        figsize (tuple): Figure size for plots (width, height). Default: (16, 7).
        show_percentiles (bool): If True, shows 1st and 99th percentiles on plot. Default: True.
        return_summary (bool): If True, returns tuple (dataframe, summary_dict). Default: False.

    Returns:
        DataFrame with columns:
        - FLUX_DAYTIME / FLUX_NIGHTTIME: Filtered flux after outlier removal
        - ROLLING_MEDIAN_DAYTIME / ROLLING_MEDIAN_NIGHTTIME: 10% window rolling median
        - SD_DAYTIME / SD_NIGHTTIME: Standard deviation of filtered data
        - WINSIZE_DAYTIME / WINSIZE_NIGHTTIME: Rolling window size used

        If return_summary=True, returns:
        - tuple: (DataFrame, dict) where dict contains outlier statistics

    Raises:
        TypeError: If flux is not a pandas Series
        ValueError: If coordinates or thresholds are invalid
        RuntimeError: If Hampel filter calculation fails

    Notes:
        Uses Hampel filter (Hampel, 1974) with Median Absolute Deviation for robust
        outlier detection. Daytime/nighttime separation uses solar elevation angle
        calculated from site coordinates and UTC offset.

        Rolling window = 10% of valid data points (robust to sparse periods).

    Example:
        >>> import diive as dv
        >>> from diive.pkgs.flux.hqflux import analyze_highest_quality_flux
        >>> df = dv.load_exampledata_parquet()
        >>> flux_hq = df['NEE_CUT_REF'].copy()
        >>> results = analyze_highest_quality_flux(
        ...     flux=flux_hq,
        ...     lat=47.286,
        ...     lon=7.734,
        ...     utc_offset=1,
        ...     window_length=48*7,  # ~7 days at 30-min frequency
        ...     n_sigma_dt=4.0,      # Stricter daytime
        ...     n_sigma_nt=2.5,      # Lenient nighttime
        ...     use_differencing=True,
        ...     showplot=True,
        ...     return_summary=True
        ... )
    """
    # Input validation
    if not isinstance(flux, Series):
        raise TypeError(f"flux must be pd.Series, got {type(flux)}")
    if flux.empty:
        raise ValueError("flux Series cannot be empty")
    if not flux.index.dtype.name.startswith('datetime'):
        raise TypeError(f"flux index must be datetime, got {flux.index.dtype}")
    if not -90 <= lat <= 90:
        raise ValueError(f"lat must be -90 to 90, got {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"lon must be -180 to 180, got {lon}")
    if window_length is not None and window_length < 1:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if n_sigma_dt <= 0 or n_sigma_nt <= 0:
        raise ValueError(f"n_sigma_dt and n_sigma_nt must be positive, got {n_sigma_dt}, {n_sigma_nt}")

    hqdf_filtered = pd.DataFrame(index=flux.index)
    summary = {}

    # Auto-configure window length if not provided
    if window_length is None:
        window_length = max(int(flux.dropna().count() / 100), 13)

    # Apply Hampel filter to entire series with built-in day/night separation
    print(f"\n>>> Removing outliers from highest-quality fluxes ({flux.name})")
    print(f">>> Outlier removal method: Hampel filter (MAD-based)")
    print(f">>> Parameters: window_length={window_length}, "
          f"n_sigma_dt={n_sigma_dt}, n_sigma_nt={n_sigma_nt}, "
          f"use_differencing={use_differencing}")

    try:
        hampel = HampelDaytimeNighttime(
            series=flux,
            lat=lat,
            lon=lon,
            utc_offset=utc_offset,
            window_length=window_length,
            n_sigma_dt=n_sigma_dt,
            n_sigma_nt=n_sigma_nt,
            use_differencing=use_differencing,
            separate_day_night=True,
            showplot=showplot,
            verbose=True
        )
        hampel.calc(repeat=False)
    except Exception as e:
        raise RuntimeError(f"Hampel filter calculation failed: {e}") from e

    # Extract outlier flag
    flag = hampel.get_flag()
    s_filtered = hampel.filteredseries

    # Calculate summary statistics
    n_total = len(flux)
    n_valid = (flag == 0).sum()
    n_outliers = (flag == 2).sum()
    pct_outliers = (n_outliers / n_total * 100) if n_total > 0 else 0

    summary['total_records'] = n_total
    summary['valid_records'] = n_valid
    summary['outliers_found'] = n_outliers
    summary['outlier_pct'] = pct_outliers
    summary['window_length'] = window_length
    summary['n_sigma_dt'] = n_sigma_dt
    summary['n_sigma_nt'] = n_sigma_nt

    print(f"\n>>> Outlier Detection Summary:")
    print(f">>> Total records:     {n_total}")
    print(f">>> Valid records:     {n_valid} ({100-pct_outliers:.1f}%)")
    print(f">>> Outliers detected: {n_outliers} ({pct_outliers:.1f}%)")

    # Process daytime and nighttime separately for statistics
    for d, timeofday in enumerate(['DAYTIME', 'NIGHTTIME']):
        # Determine period mask
        if d == 0:
            is_period = hampel.is_daytime
        else:
            is_period = hampel.is_nighttime

        # Filter by time of day and quality flag
        non_outlier_locs = is_period & (flag == 0)
        outlier_locs = is_period & (flag == 2)

        non_outlier_s = flux.loc[non_outlier_locs].copy()
        outlier_s = flux.loc[outlier_locs].copy()

        # Filtered series for this period
        s_filtered_period = s_filtered.loc[is_period].copy()

        # Calculate rolling statistics on filtered data
        winsize = max(int(s_filtered_period.count() / 10), 1)
        rmedian_filtered = s_filtered_period.rolling(window=winsize, center=True, min_periods=1).median()
        sd_filtered = s_filtered_period.std()

        # Store results
        hqdf_filtered[f'FLUX_{timeofday}'] = s_filtered_period.copy()
        hqdf_filtered[f'ROLLING_MEDIAN_{timeofday}'] = rmedian_filtered
        hqdf_filtered[f'SD_{timeofday}'] = sd_filtered
        hqdf_filtered[f'WINSIZE_{timeofday}'] = winsize

        # Calculate and print statistics
        non_outliers_s_above_zero = non_outlier_s[non_outlier_s >= 0].copy()
        if len(non_outliers_s_above_zero) > 0:
            print(f">>> Largest non-outlier flux >= 0 {timeofday}:   {non_outliers_s_above_zero.max():.6f}")
            print(f">>> Smallest non-outlier flux >= 0 {timeofday}:  {non_outliers_s_above_zero.min():.6f}")
        else:
            print(f">>> No non-outlier flux >= 0 {timeofday}")

        non_outliers_s_below_zero = non_outlier_s[non_outlier_s < 0].copy()
        if len(non_outliers_s_below_zero) > 0:
            print(f">>> Largest non-outlier flux < 0 {timeofday}:    {non_outliers_s_below_zero.max():.6f}")
            print(f">>> Smallest non-outlier flux < 0 {timeofday}:   {non_outliers_s_below_zero.min():.6f}")
        else:
            print(f">>> No non-outlier flux < 0 {timeofday}")

        outliers_s_above_zero = outlier_s[outlier_s >= 0].copy()
        if len(outliers_s_above_zero) > 0:
            print(f">>> Largest outlier flux >= 0 {timeofday}:   {outliers_s_above_zero.max():.6f}")
            print(f">>> Smallest outlier flux >= 0 {timeofday}:  {outliers_s_above_zero.min():.6f}")

        outliers_s_below_zero = outlier_s[outlier_s < 0].copy()
        if len(outliers_s_below_zero) > 0:
            print(f">>> Largest outlier flux < 0 {timeofday}:    {outliers_s_below_zero.max():.6f}")
            print(f">>> Smallest outlier flux < 0 {timeofday}:   {outliers_s_below_zero.min():.6f}")

    if showplot:
        fig = plt.figure(facecolor='white', figsize=figsize)
        gs = gridspec.GridSpec(2, 1)  # rows, cols
        ax = fig.add_subplot(gs[0, 0])
        ax_nt = fig.add_subplot(gs[1, 0])

        for t in ['DAYTIME', 'NIGHTTIME']:
            t_ax = ax if t == 'DAYTIME' else ax_nt
            fluxcol = f'FLUX_{t}'
            rmediancol = f'ROLLING_MEDIAN_{t}'
            sdcol = f'SD_{t}'

            # Plot flux points
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[fluxcol],
                      label=f"{t} flux", color="#607D8B", linestyle='none', markeredgewidth=1,
                      marker='o', alpha=.5, markersize=6, markeredgecolor="#607D8B", fillstyle='none')

            # Plot rolling median
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol],
                      label=f"rolling median", color="#FF6F00", linestyle='solid',
                      marker='none', alpha=.5, linewidth=3)

            # Plot SD bands
            style_sd = dict(linestyle='dashed', marker='none', alpha=.5, linewidth=3)
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol].add(hqdf_filtered[sdcol] * 3),
                      label=f"rolling median + 3 SD", color="#F44336", **style_sd)
            t_ax.plot(hqdf_filtered.index, hqdf_filtered[rmediancol].sub(hqdf_filtered[sdcol] * 3),
                      label=f"rolling median - 3 SD", color="#00BCD4", **style_sd)

            # Optionally plot percentiles
            if show_percentiles:
                t_ax.axhline(hqdf_filtered[fluxcol].quantile(.99), linestyle='dotted', label="99th percentile",
                             color="#2196F3")
                t_ax.axhline(hqdf_filtered[fluxcol].quantile(.01), linestyle='dotted', label="1st percentile",
                             color="#9C27B0")

            pf.default_legend(ax=t_ax, labelspacing=0.2, ncol=3)

        fig.suptitle(f"Highest-quality fluxes {flux.name} after Hampel outlier removal", fontsize=16)
        fig.tight_layout()
        fig.show()

    if return_summary:
        return hqdf_filtered, summary
    else:
        return hqdf_filtered
