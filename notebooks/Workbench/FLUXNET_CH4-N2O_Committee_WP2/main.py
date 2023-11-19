import importlib.metadata
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet
from diive.core.plotting.scatter import ScatterXY

version_diive = importlib.metadata.version("diive")
print(f"diive version: v{version_diive}")


# Variable names in data files
USTAR1 = 'USTAR'
USTAR2 = 'USTAR_1_1_1'  # Alternative name in some files
FLUX1 = 'FCH4'
FLUX2 = 'FCH4_1_1_1'  # Alternative name in some files

# Source folder with data files in parquet format
SOURCEDIR = r"F:\Sync\luhk_work\20 - CODING\21 - DIIVE\diive\notebooks\Workbench\FLUXNET_CH4-N2O_Committee_WP2\data\CH4\FLUXNET-CH4 Community Product Special Version preUSTAR\PARQUET"

foundfiles = search_files(searchdirs=[SOURCEDIR], pattern='*.parquet')
[print(f"{ix}: {f}") for ix, f in enumerate(foundfiles)]

n_files = len(foundfiles)
plots_per_row = 5
n_rows = int(np.ceil(n_files / plots_per_row))
print(f"The figure will have {n_files} subplots, arranged into {n_rows} rows.")

fig = plt.figure(facecolor='white', figsize=(plots_per_row * 10, n_rows * 10), dpi=72)
gs = gridspec.GridSpec(n_rows, plots_per_row)  # rows, cols
# gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)

axes_dict = {}
current_row = 0
col = 0
for ix, ff in enumerate(foundfiles):
    if col > plots_per_row-1:
        col = 0
        current_row += 1
    ax = fig.add_subplot(gs[current_row, col])
    df = load_parquet(ff)
    site = ff.name.replace('AMF_', '').split('_')[0]
    xcol = USTAR1 if USTAR1 in df.columns else USTAR2
    ycol = FLUX1 if FLUX1 in df.columns else FLUX2
    sca = ScatterXY(x=df[xcol], y=df[ycol], nbins=20, title=site, binagg='median', ax=ax, ylim='auto').plot()
    col += 1

fig.tight_layout()
fig.show()





# foundfiles = search_files(searchdirs=[SOURCEDIR], pattern='*.parquet')
# # foundfiles = foundfiles[10:20]
# bdf = pd.DataFrame()
#
#
#
# n_files = len(foundfiles)
# plots_per_row = 5
# rows = int(np.ceil(n_files / plots_per_row))
#
# fig = plt.figure(facecolor='white', figsize=(plots_per_row*10, rows*10), dpi=72)
# gs = gridspec.GridSpec(rows, plots_per_row)  # rows, cols
# # gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
#
# axes_dict = {}
# current_row = 0
# col = 0
#
# # Prepare figure
# for ix, ff in enumerate(foundfiles):
#     if col > plots_per_row-1:
#         col = 0
#         current_row += 1
#     axes_dict[ix] = fig.add_subplot(gs[current_row, col])
#     col += 1
#
# for ix, ff in enumerate(foundfiles):
#     ax = axes_dict[ix]
#     df = load_parquet(ff)
#     site = ff.name.replace('AMF_', '').split('_')[0]
#     xcol = x1col if x1col in df.columns else x2col
#     ycol = y1col if y1col in df.columns else y2col
#     # sca = ScatterXY(x=df[xcol], y=df[ycol], nbins=50, title=site, binagg='mean')
#     sca = ScatterXY(x=df[xcol], y=df[ycol], nbins=20, title=site, binagg='median', ax=ax, ylim='auto')
#     # sca = ScatterXY(x=df[xcol], y=df[ycol], nbins=50, title=site, binagg='median')
#     sca.plot()
#
#     bx = sca.xy_df_binned[xcol]['median']
#     by = sca.xy_df_binned[ycol]['median']
#     bxmax = bx.max()
#     bymax = by.max()
#     bx = bx.divide(bxmax)
#     by = by.divide(bymax)
#
#     if bymax > 0:
#         frame = {'USTAR_NORM': bx,
#                  'FCH4_NORM': by}
#         incoming = pd.DataFrame.from_dict(frame)
#
#         if ix == 0:
#             bdf = incoming.copy()
#         else:
#             bdf = pd.concat([bdf, incoming], axis=0)
#     else:
#         print(bymax, by.max())
#
# # fig.tight_layout()
# # fig.show()
#
# bdf = bdf.sort_values(by='USTAR_NORM', ascending=True)
# # sca = ScatterXY(x=bdf['USTAR_NORM'], y=bdf['FCH4_NORM'], nbins=20)
# sca = ScatterXY(x=bdf['USTAR_NORM'], y=bdf['FCH4_NORM'], xlim=[0, 1], ylim=[-1.4, 1.4], nbins=20)
# # sca = ScatterXY(x=bdf['USTAR_NORM'], y=bdf['FCH4_NORM'], xlim=[0, 1], ylim=[0, 1], nbins=20)
# sca.plot()
#
# # # Convert to PARQUET files
# # didnotwork = []
# # foundfiles = search_files(searchdirs=[CSVDIR], pattern='*.csv')
# # for ff in foundfiles:
# #     try:
# #         d = ReadFileType(filepath=ff, filetype='FLUXNET-CH4-HH-CSV-30MIN', output_middle_timestamp=True)
# #         df, meta = d.get_filedata()
# #         save_parquet(filename=ff.name, data=df, outpath=PARQUETDIR)
# #     except:
# #         didnotwork.append(ff.name)
# # print(didnotwork)
#
#
# # [print(f) for f in foundfiles]
#
# # Read first file to get original order of columns
#
# # for ff in foundfiles:
# # d = ReadFileType(filepath=ff, filetype='FLUXNET-CH4-HH-CSV-30MIN', output_middle_timestamp=True)
# # df, meta = d.get_filedata()
# #
# # # # Keep daytime data
# # # _filter = _df['Rg_f'] > 50
# # # df = _df[_filter].copy()
# #
# # # Get variable data
# # x = df[xcol].copy()
# # y = df[ycol].copy()
# # ScatterXY(x=x, y=y, nbins=20, xlim=[0, 1], ylim=[-100, 100], title="XXX").plot()
