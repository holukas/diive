import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
from diive.pkgs.outlierdetection.lof import LocalOutlierFactorAllData


def analyze_highest_quality_flux(flux: Series, nighttime_flag: Series, showplot: bool = True):
    hqdf = pd.DataFrame(index=flux.index)
    for d in range(0, 2):
        timeofday = 'NIGHTTIME' if d == 1 else 'DAYTIME'

        hq = flux.loc[nighttime_flag == d].copy()
        # flux = self.flags.loc[self.nighttime == d, self.filteredseriescol_hq].copy()
        n_neighbors = int(hq.dropna().count() / 100)
        contamination = 'auto'
        repeat = False
        print(f">>> Removing outliers from highest-quality {timeofday} fluxes ({hq.name})")
        print(f">>> Outlier removal method: Local outlier factor across all data (n_neighbors={n_neighbors}, "
              f"contamination={contamination}, repeat={repeat})")
        lof = LocalOutlierFactorAllData(series=hq, n_neighbors=n_neighbors, contamination=contamination,
                                        showplot=showplot, verbose=True, n_jobs=-1)
        lof.calc(repeat=repeat)

        flag = lof.get_flag()
        _flags = pd.concat([hq, nighttime_flag, flag], axis=1)
        non_outlier_locs = (_flags[nighttime_flag.name] == d) & (_flags[flag.name] == 0)
        non_outlier_s = _flags.loc[non_outlier_locs, hq.name].copy()

        s = lof.filteredseries
        winsize = int(s.count() / 10)
        rmedian = s.rolling(window=winsize, center=True, min_periods=1).median()
        sd = s.std()

        hqdf[f'FLUX_{timeofday}'] = s.copy()
        hqdf[f'ROLLING_MEDIAN_{timeofday}'] = rmedian
        hqdf[f'SD_{timeofday}'] = sd
        hqdf[f'WINSIZE_{timeofday}'] = winsize

        outliers_s_above_zero = non_outlier_s[non_outlier_s >= 0].copy()
        print(f">>> Largest non-outlier flux >= 0 {timeofday}:   {outliers_s_above_zero.max()}")
        print(f">>> Smallest non-outlier flux >= 0 {timeofday}:  {outliers_s_above_zero.min()}")

        outliers_s_below_zero = non_outlier_s[non_outlier_s < 0].copy()
        print(f">>> Largest non-outlier flux < 0 {timeofday}:    {outliers_s_below_zero.max()}")
        print(f">>> Smallest non-outlier flux < 0 {timeofday}:   {outliers_s_below_zero.min()}")

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
            t_ax.plot(hqdf.index, hqdf[fluxcol],
                      label=f"{t} flux", color="#607D8B", linestyle='none', markeredgewidth=1,
                      marker='o', alpha=.5, markersize=6, markeredgecolor="#607D8B", fillstyle='none')
            t_ax.plot(hqdf.index, hqdf[rmediancol],
                      label=f"rolling median", color="#FF6F00", linestyle='solid',
                      marker='none', alpha=.5, linewidth=3)
            style_sd = dict(linestyle='dashed', marker='none', alpha=.5, linewidth=3)
            t_ax.plot(hqdf.index, hqdf[rmediancol].add(hqdf[sdcol] * 3),
                      label=f"rolling median + 3 SD", color="#F44336", **style_sd)
            t_ax.plot(hqdf.index, hqdf[rmediancol].sub(hqdf[sdcol] * 3),
                      label=f"rolling median - 3 SD", color="#00BCD4", **style_sd)
            t_ax.axhline(hqdf[fluxcol].quantile(.99), linestyle='dotted', label="99th percentile", color="#2196F3")
            t_ax.axhline(hqdf[fluxcol].quantile(.01), linestyle='dotted', label="1st percentile", color="#9C27B0")
            pf.default_legend(ax=t_ax, labelspacing=0.2, ncol=3)
            # ax.set_ylim(hq.quantile(0.005), hq.quantile(0.995))

        fig.suptitle(f"Highest-quality fluxes {flux.name} after outlier removal", fontsize=16)
        fig.tight_layout()
        fig.show()
