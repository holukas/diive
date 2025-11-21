from dbc_influxdb import dbcInflux  # Needed for communicating with the database
import calendar

import matplotlib.gridspec as grid_spec
import numpy as np
import seaborn as sns
from matplotlib.pyplot import cm
import pandas as pd
import diive as dv
from diive.pkgs.createvar.potentialradiation import potrad, potrad_eot

# filepath = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\20_MERGE_DATA\21.4_FLUXES_L1_noSHC_IRGA75+METEO7.parquet"
# filepath = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\20_MERGE_DATA\22.4_FLUXES_L1_IRGA72+METEO7_2016-2024.parquet"
# df = dv.load_parquet(filepath=filepath)
# df = df.loc[df.index.year < 2010].copy()
# # years = [2015]
# years = sorted(list(set(df.index.year)))
# months = list(set(df.index.month))
# colors = cm.Spectral_r(np.linspace(0, 1, len(years)))  # Create one color for each year

SITE_LAT = 47.478333  # CH-LAE
SITE_LON = 8.364389  # CH-LAE
UTC_OFFSET = 1
NIGHTTIME_THRESHOLD = 20
# [print(c) for c in df.columns if "PPFD" in c]
# varcol = 'PPFD_IN_T1_47_1_gfXG'
# varcol = 'VPD_T1_47_1_gfXG'
# varcol = 'TA_T1_47_1_gfXG'
# varcol = 'SW_IN_T1_47_1_gfXG'
swinpotcol = 'SW_IN_POT'

# Test on raw data
filepath = r"F:\TMP\CH-LAE_iDL_T1_47_1_TBL1_20210701-0000.dat"
df = pd.read_csv(filepath, sep=',', skiprows=[0,2,3], index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)
varcol = 'SW_IN_T1_47_1_Avg'
df = df[[varcol]].copy()

# BUCKET_PROCESSING = 'ch-lae_raw'
# START = '2021-07-01 00:00:01'
# STOP = '2021-08-01 00:00:01'
# DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'
# dbc = dbcInflux(dirconf=DIRCONF)
# df, _, _ = dbc.download(
#     bucket=BUCKET_PROCESSING,
#     measurements=['SW'],
#     fields=['SW_IN_T1_47_1'],
#     start=START,
#     stop=STOP,
#     timezone_offset_to_utc_hours=1,
#     data_version=['raw'])
# varcol = 'SW_IN_T1_47_1'

# Testing resampling with middle timestamp
from diive.core.times.resampling import resample_series_to_30MIN
series = df[varcol].copy()
from diive.core.times.times import DetectFrequency
freq = DetectFrequency(index=series.index, verbose=True).get()
series = series.asfreq(freq)
series.index.name = 'TIMESTAMP_END'
resampled = resample_series_to_30MIN(series=series)
df = pd.DataFrame(resampled)
from diive.core.times.times import insert_timestamp
df = insert_timestamp(data=df, convention='middle', set_as_index=True)
# df.loc["2022-08-01 09:15"]

# Overwrite potential radiation
df[swinpotcol] = potrad_eot(timestamp_index=df.index, lat=SITE_LAT, lon=SITE_LON, utc_offset=UTC_OFFSET)
# df[swinpotcol] = potrad(timestamp_index=df.index, lat=SITE_LAT, lon=SITE_LON, utc_offset=UTC_OFFSET)
print(df)

# dc = DielCycle(series=series)
# title = f'{var}'
# units = 'units'
# dc.plot(ax=None, title=title, txt_ylabel_units=units,
#         each_month=True, legend_n_col=2)

def execute_phase_shift_fft():
    # ==========================================
    # EXECUTION
    # ==========================================

    # Run the FFT detection
    df_phase = calculate_phase_shift_fft(df,
                                         col_meas=varcol,
                                         col_pot=swinpotcol,
                                         min_clearness=0.6)

    df_phase.index = pd.to_datetime(df_phase.index)

    # Filter for valid days (amplitude check removes pure noise days)
    valid_phase = df_phase[df_phase['amplitude_meas'] > 1000]  # Threshold depends on your units (W/m2 sum)

    # ==========================================
    # PLOTTING
    # ==========================================

    fig = plt.figure(figsize=(14, 10))
    gs = grid_spec.GridSpec(3, 2)

    # 1. Scatter Plot: Shift vs Date
    ax1 = fig.add_subplot(gs[0, :])  # Span top row
    ax1.scatter(valid_phase.index, valid_phase['shift_minutes'],
                color='teal', alpha=0.6, s=15, label='Daily Phase Shift')

    # Add Rolling Median (Smoothed trend)
    rolling = valid_phase['shift_minutes'].rolling(window=15, center=True).median()
    ax1.plot(valid_phase.index, rolling, color='red', linewidth=2, label='15-Day Rolling Median')

    ax1.set_ylabel("Time Shift (Minutes)")
    ax1.set_title(f"Phase Shift Detection (FFT Method)\nPositive = Measured data is LATE (Lagging)")
    ax1.axhline(0, color='k', linewidth=1)
    ax1.axhline(60, color='gray', linestyle=':', alpha=0.5)  # DST
    ax1.axhline(-60, color='gray', linestyle=':', alpha=0.5)  # DST
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Histogram: Distribution of Shifts
    ax2 = fig.add_subplot(gs[1, 0])
    try:
        sns.histplot(valid_phase['shift_minutes'], bins=50, kde=True, ax=ax2, color='teal')
    except:
        ax2.hist(valid_phase['shift_minutes'], bins=50, color='teal', alpha=0.7)

    ax2.set_xlabel("Shift (Minutes)")
    ax2.set_title("Distribution of Time Shifts")
    ax2.axvline(0, color='k')
    median_val = valid_phase['shift_minutes'].median()
    ax2.axvline(median_val, color='red', linestyle='--')
    ax2.text(median_val, ax2.get_ylim()[1] * 0.9, f" Median: {median_val:.2f} min", color='red')

    # 3. Polar Plot: Phase Angle Visualization (The "Clock")
    # This visualizes where the "average" peak of the day is happening relative to noon
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')

    # Convert minutes back to radians for polar plot
    # 0 degrees = Noon in this context if we treat potential as reference
    # We plot the DIFFERENCE
    rads = (valid_phase['shift_minutes'] / 1440) * 2 * np.pi

    ax3.scatter(rads, valid_phase['amplitude_meas'], c='teal', alpha=0.3, s=10)
    ax3.set_theta_zero_location("N")  # Top is 0 shift
    ax3.set_theta_direction(-1)  # Clockwise
    ax3.set_title("Phase Shift Polar Plot (0 = Perfect Sync)")
    # Limit the view to the top sector (assuming shifts are within +/- 4 hours)
    ax3.set_thetamin(-45)
    ax3.set_thetamax(45)

    # 4. Monthly Boxplot (To see seasonal drift)
    ax4 = fig.add_subplot(gs[2, :])  # Span bottom row
    valid_phase['Month'] = valid_phase.index.month
    valid_phase.boxplot(column='shift_minutes', by='Month', ax=ax4, grid=False,
                        patch_artist=True, boxprops=dict(facecolor="teal", alpha=0.5))
    ax4.set_title("Monthly variability of Time Shift")
    ax4.set_ylabel("Shift (Minutes)")
    ax4.axhline(0, color='k', linewidth=1)
    fig.suptitle("")  # Remove pandas auto-title

    plt.tight_layout()
    plt.show()


def calculate_phase_shift_fft(df, col_meas, col_pot, min_clearness=0.5):
    """
    Calculates time shift by comparing the Phase Angle of the fundamental
    24-hour frequency component of Measured vs Potential radiation.

    Math:
    Delta_t (min) = (Delta_phi (rad) / 2*pi) * 1440 min
    """

    results = {}

    # Determine sampling interval in minutes
    freq = pd.infer_freq(df.index)
    freq = '1min' if freq == 'min' else freq
    if freq:
        dt_min = pd.to_timedelta(freq).total_seconds() / 60
    else:
        dt_min = (df.index[1] - df.index[0]).total_seconds() / 60

    points_per_day = int(1440 / dt_min)

    # Group by day
    grouped = df.groupby(df.index.date)
    print(f"Analyzing {len(grouped)} days using FFT Phase Shift...")

    for date, group in grouped:
        # 1. Data Check
        if len(group) < (points_per_day * 0.9):  # Skip incomplete days
            results[date] = {'shift_minutes': np.nan, 'amplitude_meas': 0}
            continue

        # Fill NaNs with 0 (night/missing) for FFT safety
        y_meas = group[col_meas].fillna(0).values
        y_pot = group[col_pot].fillna(0).values

        # Basic Clearness Check (Filter out heavy overcast days)
        if np.sum(y_pot) > 0:
            clearness = np.sum(y_meas) / np.sum(y_pot)
            if clearness < min_clearness:
                results[date] = {'shift_minutes': np.nan, 'amplitude_meas': 0}
                continue
        else:
            continue

        # 2. The Math: Discrete Fourier Transform for k=1 (Fundamental 24h cycle)
        # We don't need a full FFT; we just need the complex number for the 1-day period.
        # X_k = sum(x_n * exp(-i * 2*pi * k * n / N))

        N = len(y_meas)
        n = np.arange(N)

        # The "Basis Vector" for exactly 1 cycle per day
        # If your data is not exactly 24h chunks, this needs adjustment,
        # but groupby(date) usually ensures this.
        k = 1
        basis = np.exp(-1j * 2 * np.pi * k * n / N)

        # Project data onto the basis (Dot product)
        # These are Complex numbers representing the 24h wave
        X_meas = np.sum(y_meas * basis)
        X_pot = np.sum(y_pot * basis)

        # 3. Get Phase Angles (in Radians)
        phi_meas = np.angle(X_meas)
        phi_pot = np.angle(X_pot)

        # 4. Calculate Difference
        delta_phi = phi_meas - phi_pot

        # Handle Wrap-around (e.g., if diff is > pi or < -pi)
        # This ensures the shift is finding the shortest path around the circle
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi

        # 5. Convert Radians to Minutes
        # 2*pi radians = 1440 minutes (24 hours)
        shift_minutes = (delta_phi / (2 * np.pi)) * 1440

        # Note on Sign:
        # If phi_meas is larger (later phase), the wave is shifted to the RIGHT (Delayed/Late)
        # However, in time series, a "Late" signal usually means we need to subtract time.
        # Let's standardize: Positive result = Measured Peak is LATER than Potential Peak.

        results[date] = {
            'shift_minutes': shift_minutes,
            'amplitude_meas': np.abs(X_meas)  # Strength of the signal
        }

    return pd.DataFrame.from_dict(results, orient='index')


def plot_monthly_dielcycles(df, varcol, swinpotcol):
    fig = plt.figure(facecolor='white', figsize=(16, 7))
    gs = grid_spec.GridSpec(3, 4)  # rows, cols
    # gs.update(wspace=0, hspace=0, left=0.09, right=0.97, top=0.95, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])
    ax9 = fig.add_subplot(gs[2, 0])
    ax10 = fig.add_subplot(gs[2, 1])
    ax11 = fig.add_subplot(gs[2, 2])
    ax12 = fig.add_subplot(gs[2, 3])
    axmap = {1: ax1, 2: ax2, 3: ax3, 4: ax4, 5: ax5, 6: ax6, 7: ax7, 8: ax8, 9: ax9, 10: ax10, 11: ax11, 12: ax12}
    final_handles = []
    final_labels = []

    for month in months:
        # Get all data for month
        subset = df.loc[df.index.month == month].copy()
        series = subset[varcol].copy()
        swinseries = subset[swinpotcol].copy()

        ax = axmap[month]
        ax2 = ax.twinx()
        swinpot_dc = diel_cycle(series=swinseries, mean=True, std=True, each_month=False)
        means_swinpot = swinpot_dc['mean'].copy()
        means_swinpot = means_swinpot.droplevel(level=0)
        time_strings = means_swinpot.index.astype(str)
        time_delta = pd.to_timedelta(time_strings)
        new_index_strings = (pd.to_datetime('today').normalize() + time_delta).strftime('%H:%M')
        means_swinpot.index = new_index_strings
        means_swinpot.plot(ax=ax2, label=f'SW_IN_POT', color="black", zorder=99, lw=2, ls='--')
        if month == 8:
            ax2.set_ylabel(f"{swinpotcol}")

        # Build diel cycle for each year
        for yix, year in enumerate(years):
            color = colors[yix]
            series_year = series.loc[series.index.year == year].copy()
            series_dc = diel_cycle(series=series_year, mean=True, std=True, each_month=False)

            means_series = series_dc['mean'].copy()
            means_series = means_series.droplevel(level=0)
            time_strings = means_series.index.astype(str)
            time_delta = pd.to_timedelta(time_strings)
            new_index_strings = (pd.to_datetime('today').normalize() + time_delta).strftime('%H:%M')
            means_series.index = new_index_strings

            means_series.plot(ax=ax, label=f'{year}', color=color, zorder=99, lw=2, alpha=0.6)
            if month == 5:
                ax.set_ylabel(f"{varcol}")
            if month in [9, 10, 11, 12]:
                ax.set_xlabel('Time of Day')
            monthstr = calendar.month_abbr[month]
            ax.set_title(monthstr)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            # Capture legend handles and labels
            if month == 1:
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                final_handles = lines + lines2
                final_labels = labels + labels2

            combined = pd.DataFrame()
            combined[varcol] = means_series
            combined[swinpotcol] = means_swinpot
            larger = combined.loc[combined[varcol] > combined[swinpotcol]]
            if not larger.empty:
                print(larger)

            # print(means_series.loc[means_series == means_series.max()].values[0])
            # print(means_series.loc[means_series == means_series.max()].index)

    # 1. Create the Global Legend
    # loc='upper center' aligns the legend's anchor point
    # bbox_to_anchor=(0.5, 0.93) places it at X=50% (center), Y=93% (near top)
    fig.legend(final_handles, final_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.92),
               ncol=np.ceil(len(years) / 2),  # Or specific number like 6
               frameon=False)  # Optional: removes box border

    # 2. Add Title
    fig.suptitle(f"TITLE", fontsize=16, y=0.98)

    # 3. Adjust Layout
    # tight_layout organizes the grids, rect=[...] leaves space at the top
    # rect format: [left, bottom, right, top]
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    fig.show()


from diive.core.times.resampling import diel_cycle


# ---


def plot_radiation_fingerprint(df, var_col, year):
    # Filter for one year
    df_year = df.loc[df.index.year == year].copy()
    # Pivot: Index=Date, Columns=Time, Values=Radiation
    df_year['Date'] = df_year.index.date
    df_year['Time'] = df_year.index.time
    pivot = df_year.pivot(index='Date', columns='Time', values=var_col)
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(pivot, aspect='auto', cmap='inferno', vmin=900, vmax=901, origin='lower')
    ax.set_title(f"Radiation Fingerprint - {year}")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Day of Year")
    plt.colorbar(im, label="W/m2")
    plt.tight_layout()
    plt.show()


# ---
def detect_noon_shift(df, sw_col, pot_col):
    # Resample to daily to find max time
    # Note: We need the exact timestamp of the max value, not just the max value

    daily_stats = pd.DataFrame()

    # 1. Calculate Daily Totals to identify clear days
    daily_sums = df[[sw_col, pot_col]].resample('D').sum()
    daily_stats['clearness_index'] = daily_sums[sw_col] / daily_sums[pot_col]

    # 2. Find timestamp of max value per day
    # This gives us the index (timestamp) where the max occurred
    idx_max_sw = df[sw_col].groupby(df.index.date).idxmax()
    idx_max_pot = df[pot_col].groupby(df.index.date).idxmax()

    # 3. Calculate time difference in minutes
    # We align them by date
    diffs = []
    for date in idx_max_sw.index:
        if pd.isna(idx_max_sw[date]) or pd.isna(idx_max_pot[date]):
            diffs.append(np.nan)
            continue

        t_sw = idx_max_sw[date]
        t_pot = idx_max_pot[date]

        # Calculate minute difference
        delta = (t_sw - t_pot).total_seconds() / 60
        diffs.append(delta)

    daily_stats['time_shift_minutes'] = diffs

    # Filter for clear days (e.g., > 70% of potential) to remove cloudy noise
    clear_days = daily_stats[daily_stats['clearness_index'] > 0.7]

    clear_days['time_shift_minutes'].plot(style='.', title="Time Shift of Measured Data (Minutes)")
    plt.show()

    return clear_days['time_shift_minutes']


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_timeshift_highres(df, col_meas, col_pot,
                                max_shift_min=120,
                                upsample_freq='1min',
                                min_clearness_index=0.5):
    """
    Detects time shifts with high precision by upsampling data (interpolation)
    before cross-correlating.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with DateTimeIndex.
    col_meas : str
        Measured radiation column name.
    col_pot : str
        Potential radiation column name.
    max_shift_min : int
        Max shift to search for in minutes (e.g., 120 = +/- 2 hours).
    upsample_freq : str
        Resolution to interpolate to (e.g., '1min').
    min_clearness_index : float
        0.0 to 1.0. Only analyze days where Measure/Potential > this value.
        Helps filter out heavily overcast days that produce noise.

    Returns:
    --------
    pd.DataFrame with 'shift_minutes' and 'max_corr'.
    """

    results = {}

    # Group by date to handle drift over time
    grouped = df.groupby(df.index.date)
    print(f"Analyzing {len(grouped)} days at {upsample_freq} resolution...")

    # Pre-calculate steps for the progress log
    count = 0
    total = len(grouped)

    for date, group in grouped:
        count += 1

        # 1. Preliminary Filters (Skip useless days)
        pot_sum = group[col_pot].sum()
        meas_sum = group[col_meas].sum()

        # Skip if potential is near zero (winter errors/polar night)
        if pot_sum < 100:
            results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
            continue

        # Skip if day is too cloudy (Clearness Index)
        # Cloudy days have flat shapes that cross-correlate poorly
        if (meas_sum / pot_sum) < min_clearness_index:
            results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
            continue

        try:
            # 2. Upsampling (The Magic Step)
            # Create a high-res time index for this specific day
            start_time = group.index[0]
            end_time = group.index[-1]
            highres_idx = pd.date_range(start_time, end_time, freq=upsample_freq)

            # Interpolate POTENTIAL using Cubic Spline (Physics is smooth)
            # We limit area to daytime to avoid interpolation weirdness at edges
            daytime_mask = group[col_pot] > 0
            if daytime_mask.sum() < 5: continue

            # Reindex and Interpolate
            # We use 'pchip' or 'cubic' for potential because the sun moves smoothly
            ts_pot_hr = group[col_pot].reindex(highres_idx).interpolate(method='pchip').fillna(0)

            # Interpolate MEASURED using Linear (Clouds are sharp/jagged)
            # Cubic interpolation on measured data can cause "ringing" artifacts
            ts_meas_hr = group[col_meas].reindex(highres_idx).interpolate(method='linear').fillna(0)

            # 3. Cross-Correlation
            # We only look at the "sun up" portion to save computation
            # and avoid correlating night noise (0 vs 0)
            sun_up = ts_pot_hr > 10
            ts_pot_hr = ts_pot_hr[sun_up]
            ts_meas_hr = ts_meas_hr[sun_up]

            if len(ts_pot_hr) == 0: continue

            lags = range(-max_shift_min, max_shift_min + 1)
            best_corr = -1
            best_lag = 0

            # Iterate through lags (in minutes)
            for lag in lags:
                # Shift measured data
                shifted = ts_meas_hr.shift(lag)

                # Correlation
                corr = ts_pot_hr.corr(shifted)

                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            # Store result (Lag is in minutes because freq='1min')
            results[date] = {
                'shift_minutes': best_lag,
                'max_corr': best_corr
            }

        except Exception as e:
            # Fallback for empty days or interpolation errors
            results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}

    print("Analysis complete.")
    return pd.DataFrame.from_dict(results, orient='index')


def execute_crosscorr():
    # ==========================================
    # EXECUTION
    # ==========================================

    # Run the high-res detection
    # This might take 10-20 seconds depending on dataset size
    shift_df = calculate_timeshift_highres(df,
                                           col_meas=varcol,
                                           col_pot=swinpotcol,
                                           max_shift_min=120,
                                           min_clearness_index=0.6)  # Only look at reasonably clear days

    shift_df.index = pd.to_datetime(shift_df.index)

    # ==========================================
    # VISUALIZATION
    # ==========================================

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})

    # Filter for very high confidence (sunny days)
    confident = shift_df[shift_df['max_corr'] > 0.97]

    # 1. Scatter Plot of Drift over Time
    sc = ax1.scatter(confident.index, confident['shift_minutes'],
                     c=confident['max_corr'], cmap='viridis', s=15, alpha=0.8)
    ax1.set_ylabel("Time Shift (Minutes)")
    ax1.set_title(
        f"High-Resolution Time Shift Detection (1-min precision)\nPositive = Measured is Early | Negative = Measured is Late")
    ax1.axhline(0, c='k', lw=1)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax1, label="Correlation Strength")

    # Add bands for common errors
    ax1.axhline(60, c='r', ls=':', alpha=0.4)
    ax1.text(shift_df.index[0], 62, "DST (+60m)", color='red', fontsize=8)
    ax1.axhline(-60, c='r', ls=':', alpha=0.4)

    # 2. Histogram of Shifts (To find the systematic error)
    # This tells you: "Most days are shifted by exactly X minutes"
    ax2.hist(confident['shift_minutes'], bins=range(-120, 120, 2), color='tab:blue', alpha=0.7, edgecolor='k')
    ax2.set_xlabel("Shift Minutes")
    ax2.set_ylabel("Frequency (Days)")
    ax2.set_title("Distribution of Detected Shifts")
    ax2.axvline(0, c='k', lw=2)
    try:
        median_shift = confident['shift_minutes'].median()
        ax2.axvline(median_shift, c='r', ls='--', lw=2)
        ax2.text(median_shift + 2, ax2.get_ylim()[1] * 0.9, f"Median: {median_shift:.1f} min", color='r',
                 fontweight='bold')
    except:
        pass

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_monthly_dielcycles(df, varcol, swinpotcol)
    # detect_noon_shift(df, varcol, swinpotcol)
    # plot_radiation_fingerprint(df, varcol, 2006)
    # plot_radiation_fingerprint(df, varcol, 2007)
    # plot_radiation_fingerprint(df, varcol, 2008)
    # plot_radiation_fingerprint(df, varcol, 2009)
    # plot_radiation_fingerprint(df, varcol, 2010)
    # plot_radiation_fingerprint(df, varcol, 2011)
    # execute_crosscorr()
    execute_phase_shift_fft()