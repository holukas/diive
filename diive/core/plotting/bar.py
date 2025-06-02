import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme


class LongtermAnomaliesYear:

    def __init__(self,
                 series: Series,
                 reference_start_year: int,
                 reference_end_year: int,
                 series_label: str = None,
                 series_units: str = None):
        """Calculate and plot long-term anomaly for a variable, per year, compared to a reference period.

        Args:
            series: Time series for anomalies with one value per year.
            reference_start_year: First year of the reference period.
            reference_end_year: Last year of the reference period.
            series_label: Label for *series* on the y-axis of the plot.
            series_units: Units for *series* on the y-axis of the plot.

        - Example notebook available in:
            notebooks/Plotting/LongTermAnomalies.ipynb
        """
        self.series = series
        self.series_units = series_units
        self.series_label = series_label
        self.reference_start_year = reference_start_year
        self.reference_end_year = reference_end_year

        self.series.sort_index(ascending=True)
        self.data_first_year = self.series.index.min()
        self.data_last_year = self.series.index.max()

        # Create axis
        self.fig, self.ax = pf.create_ax()

        self.anomalies_df = self._calc_reference()

    def _apply_format(self):
        title = f"{self.series_label} anomaly per year ({self.data_first_year}-{self.data_last_year})"
        self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)

        ref_mean = self.anomalies_df['reference_mean'].iloc[-1]
        ref_sd = self.anomalies_df['reference_sd'].iloc[-1]
        ref_n_years = (self.reference_end_year - self.reference_start_year) + 1
        last10 = self.anomalies_df[self.series.name].tail(10)
        last10_mean = last10.mean()
        last10_std = last10.std()
        self.ax.text(0.98, 0.02, f"reference period mean: {ref_mean:.2f}±{ref_sd:.2f}sd "
                                 f"({self.reference_start_year}-{self.reference_end_year}, "
                                 f"{ref_n_years} years)\n"
                                 f"last 10 years mean: {last10_mean:.2f}±{last10_std:.2f}sd "
                                 f"({last10.index[0]}-{last10.index[-1]})",
                     size=theme.AX_LABELS_FONTSIZE, color='black', backgroundcolor='none', transform=self.ax.transAxes,
                     alpha=0.8, horizontalalignment='right', verticalalignment='bottom')
        nbins = 50 if len(self.series) > 50 else len(self.series)
        self.ax.locator_params(axis='x', nbins=nbins)
        pf.default_format(ax=self.ax,
                          ax_xlabel_txt='Year',
                          ax_ylabel_txt=f"{self.series_label} anomaly",
                          txt_ylabel_units=self.series_units,
                          showgrid=False)
        self.ax.axhline(0, lw=1, color='black')
        self.ax.set_xlim(-1, len(self.series))
        # pf.nice_date_ticks(ax=self.ax, which='x', locator='year')
        self.fig.tight_layout()

    def _calc_reference(self):
        anomalies_df = pd.DataFrame(self.series)

        ref_subset = self.series.loc[(self.series.index >= self.reference_start_year)
                                     & (self.series.index <= self.reference_end_year)]
        # ref_subset = self.series.between(self.reference_start_ix, self.reference_end_ix)
        anomalies_df['reference_mean'] = ref_subset.mean()
        anomalies_df['reference_sd'] = ref_subset.std()
        anomalies_df['anomaly'] = anomalies_df[self.series.name].sub(anomalies_df['reference_mean'])
        anomalies_df['anomaly_above'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] >= 0]
        anomalies_df['anomaly_below'] = anomalies_df['anomaly'].loc[anomalies_df['anomaly'] < 0]
        return anomalies_df

    def get(self):
        """Return axis"""
        return self.ax

    def plot(self, showplot: bool = True):
        # ax1.plot(ta_longterm.index.values, ta_longterm['diff'].values)
        self.anomalies_df['anomaly_above'].plot.bar(color='#EF5350', ax=self.ax, legend=False, width=.7)
        # self.anomalies_df['anomaly_above'].plot.bar(color='#F44336', ax=self.ax, legend=False)
        self.anomalies_df['anomaly_below'].plot.bar(color='#42A5F5', ax=self.ax, legend=False, width=.7)
        # self.anomalies_df['anomaly_below'].plot.bar(color='#2196F3', ax=self.ax, legend=False)
        # ta_longterm_anomalies.plot.bar(x='year', y='Temperature', color='#2196F3', ax=ax1)
        # ta_longterm_anomalies_above.plot.bar(x='year', y='Temperature', color='red', ax=ax1)
        # ta_longterm_anomalies_below.plot.bar(x='year', y='Temperature', color='blue', ax=ax1)

        self._apply_format()

        if showplot:
            self.fig.show()


def example():
    # ## Long-term TA
    # ## space-separated data
    # data_longterm_TA = r"L:\Sync\luhk_work\80 - SITES\CH-DAV\Data\Datasets\MeteoSwiss\CH-DAV_1864-2023_TA-YEARLY_Meteoswiss_order_120443_data.txt"
    # df = pd.read_csv(data_longterm_TA, header=0, encoding='utf-8', delimiter=';',
    #                  keep_date_col=False, index_col='time', dtype=None,
    #                  engine='python')
    # series = df['tre200y0'].copy()
    # series_label = "Air temperature"
    # LongtermAnomaliesYear(series=series,
    #                       series_label=series_label,
    #                       series_units='(°C)',
    #                       reference_start_year=1961,
    #                       reference_end_year=1990).plot()

    from diive.core.io.files import load_parquet
    df = load_parquet(
        filepath=r"F:\Sync\luhk_work\40 - DATA\DATASETS\2025_FORESTS\2-parquet_merged\CH-Dav_ENF_ICOS+FXN_1997-2024.parquet",
        output_middle_timestamp=True,
        sanitize_timestamp=False
    )

    series = df['TA_F'].copy()
    series = series.resample('YE').mean()
    series.index = series.index.year
    series_label = "CH-DAV: Air temperature"
    LongtermAnomaliesYear(series=series,
                          series_label=series_label,
                          series_units='(°C)',
                          reference_start_year=1997,
                          reference_end_year=2024).plot()


if __name__ == '__main__':
    example()
