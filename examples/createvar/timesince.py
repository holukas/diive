"""
Examples for time-since calculations using TimeSince class.

TimeSince counts consecutive records since the last occurrence of a condition
by tracking when values fall outside a specified limit range. Useful for:
- Dry period detection (time since last precipitation > 0)
- Frost period detection (time since freezing temperature <= 0°C)
- Warm spell analysis
- Event-based time tracking

Run this script to see all examples:
    python examples/createvar/timesince.py

See Also
--------
diive.TimeSince : Class documentation with parameter details and algorithm explanation.
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import diive as dv


def example_timesince_precipitation():
    """Calculate time since last precipitation event (dry period detection).

    Demonstrates counting records since the last occurrence of precipitation > 0 mm.
    This is useful for identifying dry periods, drought conditions, and drought severity.

    Parameters used:
    - lower_lim=0: Only values > 0 are considered "in range" (precipitation events)
    - include_lim=False: Use exclusive boundary (<, >) so 0 mm is not counted as precipitation
    """
    # Load example data - July 2022 precipitation
    df = dv.load_exampledata_parquet()
    series_prec = df.loc[(df.index.year == 2022) & (df.index.month == 7),
                         "PREC_TOT_T1_25+20_1"].copy()

    # Create TimeSince counter: count records since last precipitation > 0 mm
    ts_prec = dv.TimeSince(series_prec, lower_lim=0, include_lim=False)
    ts_prec.calc()

    print("Time since last precipitation (July 2022):")
    print(f"Maximum dry period: {ts_prec.get_timesince().max()} records (~{ts_prec.get_timesince().max() * 0.5:.1f} hours)")
    print(f"Mean dry period: {ts_prec.get_timesince().mean():.1f} records (~{ts_prec.get_timesince().mean() * 0.5:.1f} hours)")
    print(f"\nFirst 10 rows:")
    print(ts_prec.get_full_results().head(10))

    # Visualize 3-panel time series
    ts_prec_df = ts_prec.get_full_results()

    fig = plt.figure(facecolor='white', figsize=(16, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    dv.plot_time_series(ax=ax1, series=ts_prec_df['PREC_TOT_T1_25+20_1']).plot(color='#1565C0')
    dv.plot_time_series(ax=ax2, series=ts_prec_df['TIMESINCE_PREC_TOT_T1_25+20_1']).plot(color='#D32F2F')
    dv.plot_time_series(ax=ax3, series=ts_prec_df['FLAG_IS_OUTSIDE_RANGE']).plot(color='#00BCD4')

    ax1.set_title("Measured Precipitation (mm)", fontsize=12, fontweight='bold')
    ax2.set_title("Time Since Last Precipitation (records)", fontsize=12, fontweight='bold')
    ax3.set_title("Flag (0=precipitation observed, 1=no precipitation)", fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)

    fig.show()


def example_timesince_temperature():
    """Calculate time since last freezing temperature event (frost period detection).

    Demonstrates counting records since the last occurrence of temperature <= 0°C.
    This is useful for identifying frost periods, freeze events, and cold spells.

    Parameters used:
    - upper_lim=0: Only values <= 0°C are considered "in range" (freezing temperatures)
    - include_lim=True: Use inclusive boundary (<=, >=) so exactly 0°C is included
    """
    # Load example data - March 2022 temperature
    df = dv.load_exampledata_parquet()
    series_ta = df.loc[(df.index.year == 2022) & (df.index.month == 3),
                       "Tair_f"].copy()

    # Create TimeSince counter: count records since last freezing temperature <= 0°C
    ts_ta = dv.TimeSince(series_ta, upper_lim=0, include_lim=True)
    ts_ta.calc()

    print("Time since last freezing temperature (March 2022):")
    print(f"Maximum warm period: {ts_ta.get_timesince().max()} records (~{ts_ta.get_timesince().max() * 0.5:.1f} hours)")
    print(f"Mean warm period: {ts_ta.get_timesince().mean():.1f} records (~{ts_ta.get_timesince().mean() * 0.5:.1f} hours)")
    print(f"\nTemperature range: {series_ta.min():.1f}°C to {series_ta.max():.1f}°C")

    # Visualize 3-panel time series
    ts_ta_df = ts_ta.get_full_results()

    fig = plt.figure(facecolor='white', figsize=(16, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    dv.plot_time_series(ax=ax1, series=ts_ta_df['Tair_f']).plot(color='#1565C0')
    dv.plot_time_series(ax=ax2, series=ts_ta_df['TIMESINCE_Tair_f']).plot(color='#D32F2F')
    dv.plot_time_series(ax=ax3, series=ts_ta_df['FLAG_IS_OUTSIDE_RANGE']).plot(color='#00BCD4')

    ax1.set_title("Measured Air Temperature (°C)", fontsize=12, fontweight='bold')
    ax2.set_title("Time Since Last Freezing Temperature (records)", fontsize=12, fontweight='bold')
    ax3.set_title("Flag (0=temperature <= 0°C, 1=temperature > 0°C)", fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)

    fig.show()


def example_timesince_heatmap_comparison():
    """Visualize time-since patterns using heatmap (daily and hourly breakdown).

    Shows daily and hourly patterns of precipitation occurrence and dry periods.
    Heatmaps reveal which hours of the day and which calendar days have the
    longest dry periods, useful for understanding precipitation patterns.

    Parameters used (same as example 1):
    - lower_lim=0, include_lim=False: Count records since last precipitation > 0 mm
    """
    # Load example data - July 2022 precipitation
    df = dv.load_exampledata_parquet()
    series_prec = df.loc[(df.index.year == 2022) & (df.index.month == 7),
                         "PREC_TOT_T1_25+20_1"].copy()

    # Create TimeSince counter
    ts_prec = dv.TimeSince(series_prec, lower_lim=0, include_lim=False)
    ts_prec.calc()
    ts_prec_df = ts_prec.get_full_results()

    # Create 3-panel heatmap comparison
    fig = plt.figure(facecolor='white', figsize=(18, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    dv.plot_heatmap_datetime(ax=ax1, series=ts_prec_df['PREC_TOT_T1_25+20_1']).plot()
    dv.plot_heatmap_datetime(ax=ax2, series=ts_prec_df['TIMESINCE_PREC_TOT_T1_25+20_1']).plot()
    dv.plot_heatmap_datetime(ax=ax3, series=ts_prec_df['FLAG_IS_OUTSIDE_RANGE']).plot()

    ax1.set_title("Precipitation (mm)", fontsize=12, fontweight='bold')
    ax2.set_title("Time Since Last Precipitation (records)", fontsize=12, fontweight='bold')
    ax3.set_title("Flag (0=precipitation, 1=dry)", fontsize=12, fontweight='bold')

    # Remove left y-axis labels for middle and right plots
    ax2.tick_params(left=True, labelleft=False)
    ax3.tick_params(left=True, labelleft=False)

    fig.show()


if __name__ == '__main__':
    print("=" * 80)
    print("TimeSince Examples: Counting records since condition occurrence")
    print("=" * 80)
    print()

    print("Example 1: Dry period detection (time since last precipitation > 0 mm)")
    print("-" * 80)
    example_timesince_precipitation()

    print("\n" + "=" * 80)
    print("Example 2: Frost period detection (time since last temperature <= 0°C)")
    print("-" * 80)
    example_timesince_temperature()

    print("\n" + "=" * 80)
    print("Example 3: Heatmap visualization (daily/hourly patterns)")
    print("-" * 80)
    example_timesince_heatmap_comparison()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
