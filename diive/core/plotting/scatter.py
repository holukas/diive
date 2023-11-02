import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf


class ScatterXY:
    def __init__(
            self,
            x: Series,
            y: Series,
            xunits: str = None,
            yunits: str = None,
            title: str = None,
            ax: plt.Axes = None,
            nbins: int = 0,
            xlim: list = None,
            ylim: list = None
    ):
        """

        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

        """
        self.xname = x.name
        self.yname = y.name
        self.xunits = xunits
        self.yunits = yunits
        self.ax = ax
        self.nbins = nbins
        self.xlim = xlim
        self.ylim = ylim

        self.xy_df = pd.concat([x, y], axis=1)
        self.xy_df = self.xy_df.dropna()

        self.title = title if title else f"{self.yname} vs. {self.xname}"

        self.fig = None

        if self.nbins > 0:
            self._databinning()

    def _databinning(self):
        group, bins = pd.qcut(self.xy_df[self.xname], q=self.nbins, retbins=True, duplicates='drop')
        groupcol = f'GROUP_{self.xname}'
        self.xy_df[groupcol] = group
        self.xy_df_binned = self.xy_df.groupby(groupcol).agg({'mean', 'std', 'count'})

    def plot(self):
        """Generate plot"""
        if not self.ax:
            # Create ax if none is given
            self.fig, self.ax = pf.create_ax(figsize=(8, 8))
            self._plot()
            plt.tight_layout()
            self.fig.show()
        else:
            # Otherwise plot to given ax
            self._plot()

    def _plot(self, nbins: int = 10):
        """Generate plot on axis"""
        nbins += 1  # To include zero
        label = self.yname
        self.ax.scatter(x=self.xy_df[self.xname],
                        y=self.xy_df[self.yname],
                        c='none',
                        s=40,
                        marker='o',
                        edgecolors='#607D8B',
                        label=label)

        if self.nbins > 0:
            _min = self.xy_df_binned[self.yname]['count'].min()
            _max = self.xy_df_binned[self.yname]['count'].max()
            self.ax.scatter(x=self.xy_df_binned[self.xname]['mean'],
                            y=self.xy_df_binned[self.yname]['mean'],
                            c='none',
                            s=80,
                            marker='o',
                            edgecolors='r',
                            lw=2,
                            label=f"binned data, mean±SD "
                                  f"({_min}-{_max} values per bin)")
            self.ax.errorbar(x=self.xy_df_binned[self.xname]['mean'],
                             y=self.xy_df_binned[self.yname]['mean'],
                             xerr=self.xy_df_binned[self.xname]['std'],
                             yerr=self.xy_df_binned[self.yname]['std'],
                             elinewidth=3, ecolor='red', alpha=.6, lw=0)

        self._apply_format()
        self.ax.locator_params(axis='x', nbins=nbins)
        self.ax.locator_params(axis='y', nbins=nbins)

    def _apply_format(self):

        if self.xlim:
            xmin = self.xlim[0]
            xmax = self.xlim[1]
            self.ax.set_xlim(xmin, xmax)

        if self.ylim:
            ymin = self.ylim[0]
            ymax = self.ylim[1]
            self.ax.set_ylim(ymin, ymax)

        pf.add_zeroline_y(ax=self.ax, data=self.xy_df[self.yname])

        pf.default_format(ax=self.ax,
                          ax_xlabel_txt=self.xname,
                          ax_ylabel_txt=self.yname,
                          # txt_ylabel_units=self.yunits,
                          txt_ylabel_units=self.yunits)

        pf.default_legend(ax=self.ax,
                          labelspacing=0.2,
                          ncol=1)

        # pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        # if self.showplot:
        #     self.fig.suptitle(f"{self.title}", fontsize=theme.FIGHEADER_FONTSIZE)
        #     self.fig.tight_layout()


def example():
    from pathlib import Path
    FOLDER = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\notebooks\Workbench\FLUXNET_CH4-N2O_Committee_WP2\data"

    # from diive.core.io.filereader import search_files, MultiDataFileReader
    # filepaths = search_files(FOLDER, "*.csv")
    # filepaths = [fp for fp in filepaths if "_fluxnet_" in fp.stem and fp.stem.endswith("_adv")]
    # print(filepaths)
    # fr = MultiDataFileReader(filetype='EDDYPRO_FLUXNET_30MIN', filepaths=filepaths)
    # df = fr.data_df
    # from diive.core.io.files import save_parquet
    # save_parquet(outpath=FOLDER, filename="data", data=df)

    from diive.core.io.files import load_parquet
    filepath = Path(FOLDER) / 'data.parquet'
    df = load_parquet(filepath=filepath)

    _filter = df['SW_IN_POT'] > 50
    df = df[_filter].copy()

    # fluxcol = 'FCH4'
    xcol = 'Rg_1_1_1'
    ycol = 'Ta_1_1_1'

    x = df[xcol].copy()
    y = df[ycol].copy()

    # # Plot to given ax
    # fig, ax = pf.create_ax()
    # Scatter(x=x, y=y, ax=ax).plot()
    # fig.tight_layout()
    # fig.show()

    # Plot without given ax
    ScatterXY(x=x, y=y, nbins=10).plot()
    # Scatter(x=x, y=y, nbins=10, ylim=[0, 2]).plot()

    # series_units = r'($\mathrm{gC\ m^{-2}}$)'


if __name__ == '__main__':
    example()
