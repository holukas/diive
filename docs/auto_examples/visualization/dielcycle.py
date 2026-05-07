"""
Examples for DielCycle visualization.

DielCycle: Visualize diurnal (daily) cycles separated by month or other periods.

Run this script to display example plots:
    python examples/visualization/dielcycle.py
"""

import diive as dv


def example_dielcycle_monthly():
    """DielCycle separated by month.

    Displays mean CO2 flux for each hour of the day, separated by month,
    showing seasonal changes in diurnal patterns.
    """
    df = dv.load_exampledata_parquet()
    series = df['NEE_CUT_REF_f'].copy()

    dc = dv.plot_diel_cycle(series=series)
    title = r'$\mathrm{Mean\ CO_2\ flux\ (2013-2024)}$'
    units = r'($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)'
    dc.plot(ax=None, title=title, txt_ylabel_units=units,
            each_month=True, legend_n_col=2)


if __name__ == '__main__':
    print("Running DielCycle examples...")

    print("\n1. DielCycle separated by month...")
    example_dielcycle_monthly()

    print("\nAll examples completed!")
