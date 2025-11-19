import calendar

import matplotlib.gridspec as grid_spec
import numpy as np
from matplotlib.pyplot import cm

import diive as dv
from diive.pkgs.createvar.potentialradiation import potrad

filepath = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_ch-lae_flux_product\dataset_ch-lae_flux_product\notebooks\20_MERGE_DATA\21.4_FLUXES_L1_noSHC_IRGA75+METEO7.parquet"
df = dv.load_parquet(filepath=filepath)

# years = [2015]
years = sorted(list(set(df.index.year)))
months = list(set(df.index.month))

SITE_LAT = 47.478333  # CH-LAE
SITE_LON = 8.364389  # CH-LAE
UTC_OFFSET = 1
NIGHTTIME_THRESHOLD = 20
# [print(c) for c in df.columns if "PPFD" in c]
# varcol = 'PPFD_IN_T1_47_1_gfXG'
# varcol = 'VPD_T1_47_1_gfXG'
# varcol = 'TA_T1_47_1_gfXG'
varcol = 'SW_IN_T1_47_1_gfXG'
swinpotcol = 'SW_IN_POT'

# Create one color for each year
colors = cm.Spectral_r(np.linspace(0, 1, len(years)))

# Overwrite potential radiation
df[swinpotcol] = potrad(timestamp_index=df.index, lat=47.286417, lon=7.733750, utc_offset=1,
                        use_atmospheric_transmission=False)


# dc = DielCycle(series=series)
# title = f'{var}'
# units = 'units'
# dc.plot(ax=None, title=title, txt_ylabel_units=units,
#         each_month=True, legend_n_col=2)

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


def calculate_timeshift_crosscorr(df, col_meas, col_pot, max_shift_hours=2, scan_range_min=None):
    """
    Detects time shifts by cross-correlating measured vs potential radiation per day.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with DateTimeIndex.
    col_meas : str
        Name of measured radiation column.
    col_pot : str
        Name of potential radiation column.
    max_shift_hours : int
        Maximum expected shift to look for (limits search space).
    scan_range_min : int (optional)
        If provided, overrides the automatic resolution detection.
        Step size for the lag search in minutes.

    Returns:
    --------
    pd.DataFrame containing 'shift_minutes' and 'correlation' indexed by Day.
    """

    # 1. Determine data resolution (time step) in minutes
    if scan_range_min is None:
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Fallback: calculate from first two rows
            diff = (df.index[1] - df.index[0]).total_seconds() / 60
            step_size = int(diff)
        else:
            step_size = int(pd.to_timedelta(freq).total_seconds() / 60)
    else:
        step_size = scan_range_min

    print(f"Detected time step: {step_size} minutes")

    # Define the lags we want to test (in integer steps)
    # e.g., if resolution is 10min and max_shift is 2h, we test -12 to +12 steps
    steps_range = int((max_shift_hours * 60) / step_size)
    lags = range(-steps_range, steps_range + 1)

    results = {}

    # Group by day
    # We iterate over days to handle clock drift (which changes over time)
    # or sudden jumps (DST).
    grouped = df.groupby(df.index.date)

    print(f"Analyzing {len(grouped)} days...")

    for date, group in grouped:
        # 2. Filter: Only analyze if we have significant data
        # Skip days where potential radiation sum is too low (very short winter days or errors)
        if group[col_pot].sum() < 100:
            results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
            continue

        # Focus only on daylight hours to avoid high correlation of 0=0 at night
        daylight = group[group[col_pot] > 10]

        if len(daylight) < 10:  # Not enough daylight data points
            results[date] = {'shift_minutes': np.nan, 'max_corr': np.nan}
            continue

        ts_meas = daylight[col_meas]
        ts_pot = daylight[col_pot]

        best_corr = -1
        best_lag_steps = 0
        corrs = []

        # 3. The Cross-Correlation Loop
        # We shift the MEASURED data.
        # If shifting it by +1 step makes it match POTENTIAL, the data was 1 step EARLY (lag is negative).
        for lag in lags:
            # shift() moves data down (positive lag) or up (negative lag)
            shifted_meas = ts_meas.shift(lag)

            # Calculate correlation (ignoring NaNs created by shift)
            corr = ts_pot.corr(shifted_meas)

            corrs.append(corr)

            if corr > best_corr:
                best_corr = corr
                best_lag_steps = lag

        # Convert lag steps back to minutes
        # Note: If we shifted measured by +1 to match, measured was early.
        # Usually we want to know: "Add X minutes to timestamp to fix it".
        # If lag is +1 (shifted down), the timestamp needs to increase.
        shift_minutes = best_lag_steps * step_size

        results[date] = {
            'shift_minutes': shift_minutes,  # Negative means sensor is late, Positive means sensor is early
            'max_corr': best_corr,
            'avg_corr': np.mean(corrs)
        }

    return pd.DataFrame.from_dict(results, orient='index')


def execute_crosscorr():
    # ==========================================
    # EXECUTION
    # ==========================================

    # Run the detection
    # Assuming 'df' is your dataframe from your snippet
    shift_df = calculate_timeshift_crosscorr(df,
                                             col_meas=varcol,
                                             col_pot=swinpotcol,
                                             max_shift_hours=6)

    # Convert index to datetime for plotting
    shift_df.index = pd.to_datetime(shift_df.index)

    # ==========================================
    # PLOTTING RESULTS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: The detected time shift
    # We filter for days with high correlation (confident matches)
    # confident = shift_df[shift_df['avg_corr'] > 0.6]
    confident = shift_df[shift_df['max_corr'] > 0.98]

    ax1.scatter(confident.index, confident['shift_minutes'], alpha=0.6, s=14, c='tab:blue')
    ax1.set_ylabel("Estimated Time Shift (min)")
    ax1.set_title(f"Detected Time Shift: {varcol} vs {swinpotcol}")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=1)

    # Add horizontal bands for common errors
    ax1.axhline(60, color='red', linestyle=':', alpha=0.5, label='DST (+1h)')
    ax1.axhline(-60, color='red', linestyle=':', alpha=0.5, label='DST (-1h)')
    ax1.legend()

    # Plot 2: Correlation Strength (Quality Check)
    # Low correlation implies cloudy days where shift detection is unreliable
    ax2.plot(shift_df.index, shift_df['max_corr'], color='gray', alpha=0.5)
    ax2.set_ylabel("Correlation Coefficient")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0.8, 1.0)  # Focus on the top range
    ax2.grid(True)
    ax2.text(shift_df.index[0], 0.81, "Low correlation = Cloudy/Overcast (Ignore these shifts)", color='gray')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_monthly_dielcycles(df, varcol, swinpotcol)
    # detect_noon_shift(df, varcol, swinpotcol)
    plot_radiation_fingerprint(df, varcol, 2006)
    plot_radiation_fingerprint(df, varcol, 2007)
    plot_radiation_fingerprint(df, varcol, 2008)
    plot_radiation_fingerprint(df, varcol, 2009)
    plot_radiation_fingerprint(df, varcol, 2010)
    plot_radiation_fingerprint(df, varcol, 2011)
    # execute_crosscorr()
