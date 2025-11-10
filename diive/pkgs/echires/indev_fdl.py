from pathlib import Path
from diive.pkgs.echires.windrotation import WindRotation2D
import pandas as pd
from diive.core.io.filereader import search_files

# Dirs
INDIR = [r'F:\Sync\luhk_work\CURRENT\flux_detection_limit\raw']
OUTDIR = r'F:\Sync\luhk_work\CURRENT\flux_detection_limit\OUT'
filepaths = search_files(INDIR, "*.txt")

# Column names
u_col = 'x'  # m s-1
v_col = 'y'  # m s-1
w_col = 'z'  # m s-1
n2o_col = 'N2Od'  # dry umol mol-1 (ppm), but must be dry mole fraction in nmol mol-1 (ppb)
h2o_col = 'H2O'  # umol mol-1 (ppm), but must be mol mol-1
press_col = 'CellP'  # Torr, but will be converted to Pa; cell pressure?
ta_col = 'Ts'  # °C

file_results_df = pd.DataFrame(columns=['file', 'cov_flux'])

for ix, fp in enumerate(filepaths):
    # if ix > 0:
    #     break
    print(f"File: {fp}")
    try:
        df = pd.read_csv(fp)
    except:
        continue
    keepcols = [u_col, v_col, w_col, n2o_col, h2o_col, press_col, ta_col]
    df = df[keepcols].copy()

    # Conversions
    df[n2o_col] = df[n2o_col].multiply(10 ** 3)  # Convert from umol mol-1 to nmol mol-1
    df[h2o_col] = df[h2o_col].div(10 ** 6)  # Convert from mmol mol-1 to mol mol-1
    df[press_col] = 100000  # todo example Pa ambient
    # df[press_col] = df[press_col].multiply(133.322)  # From Torr to Pa
    df[ta_col] = df[ta_col].add(273.15)  # From degC to K
    e_col = 'e'
    pd_col = 'pd'

    # e = partial pressure of water vapor (Pa) = H2O mole fraction (mol mol-1) * air pressure (Pa)
    # pd = dry air partial pressure (Pa)
    df[e_col] = df[h2o_col] * df[press_col]
    df[pd_col] = df[press_col] - df['e']

    # 2D wind rotation
    r = WindRotation2D(u=df[u_col],
                       v=df[v_col],
                       w=df[w_col],
                       c=df[n2o_col])
    primes_df = r.get_primes()

    df = pd.concat([df, primes_df], axis=1)

    df['R'] = 8.31446261815324  # Universal gas constant, m3 Pa K-1 mol-1
    df['ta_mean'] = df[ta_col].mean()
    df['pd_mean'] = df[pd_col].mean()
    df['flux_conversion_factor'] = 1 / ((df['R'] * df['ta_mean']) / df['pd_mean'])
    df['cov'] = df['z_TURB'].cov(df['N2Od_TURB'].shift(-30))
    df['cov_flux'] = df['cov'] * df['flux_conversion_factor']


    new_results = [
        fp.name,
        df['cov_flux'].iloc[-1],
        # results['flux_detection_limit'],
        # results['flux_noise_rmse'],
        # results['cov_max_shift'],
        # results['flux_signal_at_cov_max_shift'],
        # results['signal_to_noise'],
        # results['signal_to_detection_limit']
    ]
    file_results_df.loc[len(file_results_df)] = new_results

    # print(file_results_df)

    # Save after each file
    outfile = Path(OUTDIR) / 'results_N2O.csv'
    # outfile = Path(outdir) / 'results_CH4.csv'
    file_results_df.to_csv(outfile)

    # print(df)