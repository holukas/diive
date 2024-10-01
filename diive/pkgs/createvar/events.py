from datetime import datetime
from pathlib import Path

from diive.core.io.files import load_parquet

SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\dataset_cha_fp2024_2005-2023\40_FLUXES_L1_IRGA+QCL+LGR_mergeData"
FILENAME = r"41.1_CH-CHA_IRGA_LGR+QCL_Level-1_eddypro_fluxnet_2005-2023_meteo7.parquet"
FILEPATH = Path(SOURCEDIR) / FILENAME
maindf = load_parquet(filepath=FILEPATH)
locs = (maindf.index.year >= 2023) & (maindf.index.year <= 2023)
df = maindf.loc[locs, 'FC'].copy()
df['EVENT'] = 0  # 0 = no event

e_date = ['2023-06-14', '2023-06-19']

start = datetime.strptime(e_date[0], '%Y-%m-%d').date()
end = datetime.strptime(e_date[1], '%Y-%m-%d').date()

locs = (df.index.date >= start) & (df.index.date <= end)
df = df.loc[locs]
print(df)
