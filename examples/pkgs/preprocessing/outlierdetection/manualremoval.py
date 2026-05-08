"""
Manual Outlier Removal examples.

This module demonstrates the ManualRemoval class for explicitly removing
data points or date ranges from a time series.
"""


def example_manual_removal_single_dates():
    """Remove data points for specific single dates.

    Demonstrates ManualRemoval with list of individual datetime strings
    to remove. Useful for flagging known equipment failures or measurement errors.
    """
    import diive as dv

    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()

    remove_dates = [
        '2018-07-05 10:30:00',
        '2018-07-12 14:15:00',
        '2018-07-20 09:00:00'
    ]

    mr = dv.ManualRemoval(
        series=s,
        remove_dates=remove_dates,
        showplot=True,
        verbose=True
    )

    mr.calc()


def example_manual_removal_date_ranges():
    """Remove data points for entire date ranges.

    Demonstrates ManualRemoval with lists specifying start and end dates
    for removal. Useful for removing periods with known issues (maintenance,
    sensor drift, power outages).
    """
    import diive as dv

    df = dv.load_exampledata_parquet()
    s = df['Tair_f'].copy()
    s = s.loc[s.index.year == 2018].copy()
    s = s.loc[s.index.month == 7].copy()

    remove_dates = [
        ['2018-07-02', '2018-07-04'],      # First period (date only)
        ['2018-07-15 08:00:00', '2018-07-15 16:00:00'],  # Daytime period
        ['2018-07-25', '2018-07-27']       # Multi-day period
    ]

    mr = dv.ManualRemoval(
        series=s,
        remove_dates=remove_dates,
        showplot=True,
        verbose=True
    )

    mr.calc()


if __name__ == '__main__':
    example_manual_removal_single_dates()
    example_manual_removal_date_ranges()
