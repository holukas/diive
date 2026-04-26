"""
Examples for stratified binning analysis.

Run this script to see example plots:
    python examples/analyses/decoupling.py
"""
import diive as dv


def example_stratified_analysis_photosynthetic_decoupling():
    """Stratified binning analysis: GPP response to VPD across temperature classes.

    Demonstrates stratified analysis of photosynthetic decoupling: how does net
    ecosystem productivity (NEE) respond to vapor pressure deficit (VPD) across
    different temperature classes?
    """
    # Load example data
    data_df = dv.load_exampledata_parquet()

    # Use summer months only
    data_df = data_df.loc[(data_df.index.month >= 6) & (data_df.index.month <= 9)].copy()

    # Select variables
    vpd_col = 'VPD_f'
    nee_col = 'NEE_CUT_REF_f'
    ta_col = 'Tair_f'
    gpp_col = 'GPP_DT_CUT_REF'
    swin_col = 'Rg_f'

    df = data_df[[nee_col, gpp_col, ta_col, vpd_col, swin_col]].copy()

    # Filter to daytime only
    daytime_locs = (df[swin_col] > 50) & (df[ta_col] > 5)
    df = df[daytime_locs].copy()

    # Rename for clarity
    rename_dict = {
        ta_col: 'air_temperature',
        vpd_col: 'vapor_pressure_deficit',
        nee_col: 'net_ecosystem_productivity'
    }
    df = df.rename(columns=rename_dict, inplace=False)

    # Prepare data
    ta_col = 'air_temperature'
    vpd_col = 'vapor_pressure_deficit'
    nee_col = 'net_ecosystem_productivity'
    df = df[[ta_col, vpd_col, nee_col]].copy()
    df[nee_col] = df[nee_col].multiply(-1)  # CO2 uptake (positive)

    # Analyze with StratifiedAnalysis
    analysis = dv.stratified_analysis(
        df=df,
        zvar=ta_col,
        xvar=vpd_col,
        yvar=nee_col,
        n_bins_z=10,
        n_bins_x=5,
        conversion=False
    )

    # Calculate bins and display results
    analysis.calcbins()
    analysis.showplot_decoupling_sbm(marker='o', emphasize_lines=True)

    # Access results as dictionary
    binaggs = analysis.get_binaggs()
    first = next(iter(binaggs))
    print("First z bin sample:")
    print(binaggs[first])

    # Get all results as a single dataframe
    print("\nAll results as dataframe:")
    print(analysis.results)


if __name__ == '__main__':
    example_stratified_analysis_photosynthetic_decoupling()
