"""
Examples for ScatterXY visualization.

ScatterXY: Scatter plot of two variables with optional third variable as color.

Run this script to display example plots:
    python examples/visualization/scatter_xy.py
"""

import diive as dv


def example_scatter_temperature_vpd():
    """ScatterXY: Temperature vs VPD.

    Shows relationship between air temperature and vapor pressure deficit.
    """
    df = dv.load_exampledata_parquet()

    scatter = dv.plot_scatter_xy(
        x=df['Tair_f'],
        y=df['VPD_f']
    )
    scatter.plot()


def example_scatter_nee_vpd():
    """ScatterXY: NEE vs VPD colored by radiation.

    Shows CO2 flux response to vapor pressure deficit,
    with solar radiation shown as point color (3-variable scatter).
    """
    df = dv.load_exampledata_parquet()

    scatter = dv.plot_scatter_xy(
        x=df['VPD_f'],
        y=df['NEE_CUT_REF_f'],
        z=df['Rg_f'],
        title='CO2 flux response to VPD'
    )
    scatter.plot(
        xlabel='VPD (hPa)',
        ylabel=r'NEE flux ($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
        zlabel=r'Radiation ($\mathrm{W\ m^{-2}}$)',
        cmap='plasma',
        show_colorbar=True
    )


def example_scatter_nee_vpd_binned():
    """ScatterXY: NEE vs VPD with bin aggregation.

    Shows aggregated CO2 flux response to vapor pressure deficit
    with median and interquartile range overlay (nbins=10).
    """
    df = dv.load_exampledata_parquet()

    scatter = dv.plot_scatter_xy(
        x=df['VPD_f'],
        y=df['NEE_CUT_REF_f'],
        z=df['Rg_f'],
        title='CO2 flux response to VPD (binned aggregation)',
        nbins=10,
        binagg='median'
    )
    scatter.plot(
        xlabel='VPD (hPa)',
        ylabel=r'NEE flux ($\mathrm{µmol\ CO_2\ m^{-2}\ s^{-1}}$)',
        zlabel=r'Radiation ($\mathrm{W\ m^{-2}}$)',
        cmap='plasma',
        show_colorbar=True
    )


if __name__ == '__main__':
    print("Running ScatterXY examples...")

    print("\n1. Temperature vs VPD scatter plot...")
    example_scatter_temperature_vpd()

    print("\n2. NEE vs VPD colored by radiation...")
    example_scatter_nee_vpd()

    print("\n3. NEE vs VPD binned aggregation with median and IQR...")
    example_scatter_nee_vpd_binned()

    print("\nAll examples completed!")
