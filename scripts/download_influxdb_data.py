import matplotlib.pyplot as plt
from dbc_influxdb import dbcInflux

from diive.core.io.files import save_parquet

SITE = 'ch-lae'
MEASUREMENTS = ['RH']
FIELDS = ['RH_T1_47_1']
DIRCONF = r'F:\Sync\luhk_work\20 - CODING\22 - POET\configs'
BUCKET_RAW = f'{SITE}_processed'  # The 'bucket' where data are stored in the database, e.g., 'ch-lae_raw' contains all raw data for CH-LAE
dbc = dbcInflux(dirconf=DIRCONF)
data_simple, data_detailed, assigned_measurements = dbc.download(
    bucket=BUCKET_RAW,
    measurements=MEASUREMENTS,
    fields=FIELDS,
    start='2015-08-01 00:00:01',
    stop='2015-09-01 00:00:01',
    timezone_offset_to_utc_hours=1,
    data_version=['meteoscreening_mst']
)
print(data_simple)
print(data_detailed)
print(assigned_measurements)

# data_simple['SW_IN_T1_47_1'].plot(x_compat=True)
# plt.show()

# Export data to parquet for fast testing
FILENAME = "downloaded_data"
OUTPATH = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\diive\configs\exampledata"
filepath = save_parquet(filename=FILENAME, data=data_simple,
                        outpath=OUTPATH)
