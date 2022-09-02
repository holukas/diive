import pandas as pd
from pandas import Series

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme


class LongtermAnomaliesYear:

    def __init__(self,
                 series: Series,
                 series_units: str,
                 reference_start_year: int,
                 reference_end_year: int):
        self.series = series
        self.series_units = series_units
        self.reference_start_year = reference_start_year
        self.reference_end_year = reference_end_year

        self.series.sort_index(ascending=True)
        self.data_first_year = self.series.index.min()
        self.data_last_year = self.series.index.max()

        # Create axis
        self.fig, self.ax = pf.create_ax()

        self.anomalies_df = self._calc_reference()

    def _apply_format(self):
        title = f"Anomaly per year ({self.data_first_year}-{self.data_last_year})"
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
                     size=theme.AXLABELS_FONTSIZE, color='black', backgroundcolor='none', transform=self.ax.transAxes,
                     alpha=0.8, horizontalalignment='right', verticalalignment='bottom')
        nbins = 50 if len(self.series) > 50 else len(self.series)
        self.ax.locator_params(axis='x', nbins=nbins)
        pf.default_format(ax=self.ax,
                          txt_xlabel='Year',
                          txt_ylabel=self.series.name,
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
        self.anomalies_df['anomaly_above'].plot.bar(color='#F44336', ax=self.ax, legend=False)
        self.anomalies_df['anomaly_below'].plot.bar(color='#2196F3', ax=self.ax, legend=False)
        # ta_longterm_anomalies.plot.bar(x='year', y='Temperature', color='#2196F3', ax=ax1)
        # ta_longterm_anomalies_above.plot.bar(x='year', y='Temperature', color='red', ax=ax1)
        # ta_longterm_anomalies_below.plot.bar(x='year', y='Temperature', color='blue', ax=ax1)

        self._apply_format()

        if showplot:
            self.fig.show()


def example():
    ## Long-term TA
    ## space-separated data
    data_longterm_TA = r"L:\Dropbox\luhk_work\_current\CH-DAV_1864-2021_TA-YEARLY_Meteoswiss_order_105469_data.txt"
    ta_longterm = pd.read_csv(data_longterm_TA, header=0, encoding='utf-8', delimiter=';',
                              keep_date_col=False, index_col='time', dtype=None,
                              engine='python')
    # ta_longterm = ta_longterm['tre200y0'].copy()
    ta_longterm = ta_longterm['tre200y0'].copy()
    # ta_longterm = ta_longterm.set_index('time')
    # ta_longterm.index = pd.to_datetime(ta_longterm.index, format='%Y')
    LongtermAnomaliesYear(series=ta_longterm,
                          series_units='(°C)',
                          reference_start_year=1864,
                          reference_end_year=1913).plot()

    # reference = ta_longterm.iloc[0:50]
    # reference_mean = reference['tre200y0'].mean()
    # ta_longterm['diff'] = ta_longterm['tre200y0'].sub(reference_mean)


if __name__ == '__main__':
    example()
