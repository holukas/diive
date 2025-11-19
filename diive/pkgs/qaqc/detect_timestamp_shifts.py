import calendar

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import cm

import diive as dv
from diive.core.times.resampling import diel_cycle
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
# [print(c) for c in df.columns if "TA" in c]
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
