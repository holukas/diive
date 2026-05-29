"""
==========================================
Interactive Time Series with a Range Tool
==========================================

Navigate long time series with a Bokeh RangeTool: a detail panel on top and a
full-series overview below with a draggable selection that pans and zooms the
detail view.

Best for: Exploring long records while keeping the overall context in view

Reference: https://docs.bokeh.org/en/latest/docs/examples/interaction/tools/range_tool.html
"""

import diive as dv

# %%
# Load example data
# ^^^^^^^^^^^^^^^^^

df = dv.load_exampledata_parquet()
# Use a full year so there is plenty to scroll through
df = df.loc[df.index.year == 2022].copy()
print(f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")

# %%
# Range-tool plot
# ^^^^^^^^^^^^^^^
#
# The top panel shows a slice of the series (here the first 10%). Drag or resize
# the shaded box in the bottom overview panel to pan and zoom the detail view.

series = df['Tair_f'].copy()

ts = dv.plotting.TimeSeries(series=series)
ts.plot_rangetool(
    height=300,           # detail panel height (px)
    width=900,            # plot width (px)
    overview_height=130,  # overview panel height (px)
    init_range=0.1        # start zoomed to the first 10% of the record
)

print("\nCreated range-tool time series plot")
print("Drag the shaded box in the lower overview to navigate the detail panel above")
