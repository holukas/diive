# Visualization & Plotting

Learn how to create publication-quality visualizations.

## Key Modules

- `diive.core.plotting.HeatmapDateTime` — Time-based heatmaps
- `diive.core.plotting.HeatmapXYZ` — XYZ binned heatmaps
- `diive.core.plotting.HexbinPlot` — Hexagonal binning
- `diive.core.plotting.TimeSeries` — Time series line plots
- `diive.core.plotting.ScatterXY` — Scatter plots
- `diive.core.plotting.HistogramPlot` — Histograms
- `diive.core.plotting.DielCycle` — Diel cycle patterns
- `diive.core.plotting.RidgeLinePlot` — Ridge line plots
- `diive.core.plotting.Cumulative` — Cumulative plots

## Example Notebooks

- [Cumulative plots](../../notebooks/plotting/Cumulative.ipynb)
- [Cumulative plots per year](../../notebooks/plotting/CumulativesPerYear.ipynb)
- [Diel cycle](../../notebooks/plotting/DielCycle.ipynb)
- [Heatmap by date/time](../../notebooks/plotting/HeatmapDateTime.ipynb)
- [Heatmap XYZ](../../notebooks/plotting/HeatmapXYZ.ipynb)
- [Heatmap year-month](../../notebooks/plotting/HeatmapYearMonth.ipynb)
- [Hexbin plot](../../notebooks/plotting/HexbinPlot.ipynb)
- [Histogram](../../notebooks/plotting/Histogram.ipynb)
- [Long-term anomalies](../../notebooks/plotting/LongTermAnomalies.ipynb)
- [Ridge line plot](../../notebooks/plotting/RidgeLine.ipynb)
- [Scatter plot](../../notebooks/plotting/ScatterXY.ipynb)
- [Time series](../../notebooks/plotting/TimeSeries.ipynb)

## Quick Example

```python
import diive as dv
import matplotlib.pyplot as plt

# Load data
df = dv.load_exampledata_parquet()

# Create hexbin plot
hm = dv.hexbinplot(
    x=df['Tair_f'],
    y=df['VPD_f'],
    z=df['NEE_CUT_REF_f'],
    gridsize=12,
    figsize=(10, 8),
    dpi=300
)
hm.plot()
plt.savefig('flux_vs_drivers.png', dpi=300, bbox_inches='tight')
```

## See Also

- [FAQ: Visualization](../faq.md#visualization)
- [Common Workflows: Creating publication-quality figures](../guide/workflows.md#visualization-workflow)
