import pandas as pd
from dbc_influxdb import dbcInflux  # Needed for communicating with the database

pd.options.display.width = 9999
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 3000)


def example():
    import matplotlib.pyplot as plt

    DIRCONF = r'L:\Sync\luhk_work\20 - CODING\22 - POET\configs'
    FIELDS = [
        'PREC_TOT_GF1_1_1',
        'SWC_GF1_0.05_1',
        # 'SWC_GF1_0.05_2',
        # 'SWC_GF1_0.15_1',
        # 'SWC_GF1_0.25_1',
        # 'SWC_GF1_0.2_2',
        # 'SWC_GF1_0.1_2'
    ]
    BUCKET_PROCESSING = 'ch-fru_processing'
    START = '2023-06-01 00:00:01'
    STOP = '2023-07-01 00:00:01'

    dbc = dbcInflux(dirconf=DIRCONF)
    data_simple, data_detailed, assigned_measurements = dbc.download(
        bucket=BUCKET_PROCESSING,
        measurements=['SWC', 'PREC'],
        fields=FIELDS,
        start=START,
        stop=STOP,
        timezone_offset_to_utc_hours=1,
        data_version='meteoscreening')

    x = data_simple['PREC_TOT_GF1_1_1'].copy()
    y = data_simple['SWC_GF1_0.05_1'].copy()
    # plt.scatter(x, y,)
    # plt.show()

    # # Covariance between PREC and SWC
    # coll = pd.Series()
    # for s in range(-5, 5):
    #     _y = y.shift(s)
    #     # print(f"{s} {x.cov(_y)}")
    #     # coll = coll.append(x.cov(_y), ignore_index=True)
    #     # coll = pd.concat([coll, x.cov(_y)], ignore_index=True)
    #     covariance = x.cov(_y)
    #     coll.loc[s] = covariance

    # Calculate difference of y to previous y
    # diff > 0 means y increased
    y_shifted = y.shift(1)
    diff = y - y_shifted

    # Create flag that shows if diff increased
    flag_increased = diff.copy()
    flag_increased.loc[flag_increased >= 0] = 1  # 1=yes, diff increased, i.e. y increased
    flag_increased.loc[flag_increased < 0] = 0

    # fantastic: https://stackoverflow.com/questions/27626542/counting-consecutive-positive-value-in-python-array
    # y = self.class_df[self.flag_col]
    y = flag_increased.copy()
    yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
    # yy = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

    negative_endpoints = yy - yy.shift(1)
    negative_endpoints = negative_endpoints[negative_endpoints < 0]

    startpoints = yy[yy == 1]

    cc = pd.concat([startpoints, negative_endpoints], axis=0, ignore_index=False)
    cc = cc.sort_index()

    n_events = len(cc) / 2

    

    yy.plot()
    plt.show()

    yy.to_csv(r"F:\TMP\temp.csv")

    yy.sort_values(ascending=False)

    yy['idx'] = yy.groupby(yy).cumcount().add(1)





    flag_increased_filtered = flag_increased[flag_increased == 1]






    # Create flag that shows if diff decreased
    flag_decreased = diff.copy()
    flag_decreased.loc[flag_decreased >= 0] = 0
    flag_decreased.loc[flag_decreased < 0] = 1

    filter_increased = flag_increased == 1
    flag_increased_filtered = flag_increased[filter_increased]


    # # # Narrow down the df to contain only the gap data
    # filter_isgap = self.gapfinder_fullres_df[self.isgap_col] == 1
    # self.gapfinder_fullres_df = self.gapfinder_fullres_df[filter_isgap]
    # datetime_col = self.gapfinder_fullres_df.index.name
    # self.gapfinder_fullres_df[datetime_col] = self.gapfinder_fullres_df.index  # needed for aggregate

    # Detect where the flag changed by comparing flag to previous flag
    # change = 1 if sign changed
    flag_increased_shifted = flag_increased.shift(1)
    change = flag_increased != flag_increased_shifted
    change = change.astype(int)  # Convert bool to int, 1=True, 0=False

    # Detect where flag changed to increasing,
    # which is the case for locations where flag=1 and change=1
    changetoincreasing = flag_increased + change
    locs = changetoincreasing[changetoincreasing == 2]
    locs.plot()
    plt.show()

    increasing_cumsum = flag_increased.cumsum()

    changetoincreasing.plot()

    cols = {
        'y': y,
        'y_shifted': y_shifted,
        'diff': diff,
        'flag_increased': flag_increased,
        'flag_increased_shifted': flag_increased_shifted,
        'change': change,
        'changetoincreasing': changetoincreasing,
        'increasing_cumsum': increasing_cumsum
    }
    xdf = pd.DataFrame.from_dict(cols)
    xdf.plot()
    plt.show()
    xdf.to_csv(r"f:\TMP\temp.csv")

    xdf[xdf.index.name] = xdf.index
    test_df = \
        xdf.groupby('increasing_cumsum').aggregate({
            xdf.index.name: ['min', 'max'],
            'increasing_cumsum': ['count'],
            'increasing': ['sum']
        })

    test_df.sort_values(by=('increasing', 'sum'), ascending=False)
    xdf.to_csv(r"f:\TMP\temp.csv")

    # from scipy.stats import pearsonr
    # pearson_coef, _ = pearsonr(x, y)
    # print("Pearson correlation coefficient:", pearson_coef)
    # spearman_coef, _ = spearmanr(x, y)
    # print("Spearman correlation coefficient:", spearman_coef)
    #
    # kendall_coef, _ = kendalltau(x, y)
    # print("Kendall correlation coefficient:", kendall_coef)

    # TESTING RANDOM FOREST
    # TESTING RANDOM FOREST
    # TESTING RANDOM FOREST
    # # RF with increase!
    # data_simple_increase = data_simple.copy()
    # for c in data_simple_increase.columns:
    #     data_simple_increase[c] = data_simple_increase[c] - data_simple_increase[c].shift(1)
    #
    # from diive.pkgs.gapfilling.randomforest_ts import RandomForestTS
    # rfts = RandomForestTS(
    #     input_df=data_simple_increase,
    #     target_col='PREC_TOT_GF1_1_1',
    #     verbose=1,
    #     # features_lag=None,
    #     features_lag=[-10, 10],
    #     # include_timestamp_as_features=False,
    #     include_timestamp_as_features=True,
    #     # add_continuous_record_number=False,
    #     add_continuous_record_number=True,
    #     sanitize_timestamp=True,
    #     perm_n_repeats=9,
    #     n_estimators=9,
    #     random_state=42,
    #     min_samples_split=4,
    #     min_samples_leaf=2,
    #     n_jobs=-1
    # )
    # rfts.reduce_features()
    # rfts.report_feature_reduction()
    #
    # rfts.trainmodel(showplot_scores=True, showplot_importance=True)
    # rfts.report_traintest()
    #
    # rfts.fillgaps(showplot_scores=True, showplot_importance=True)
    # rfts.report_gapfilling()
    #
    # rfts.gapfilling_df_
    #
    # observed = data_simple['PREC_TOT_GF1_1_1']
    # gapfilled = rfts.get_gapfilled_target()
    #
    # # Plot
    # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
    # HeatmapDateTime(series=observed).show()
    # HeatmapDateTime(series=gapfilled).show()


if __name__ == '__main__':
    example()
