# Frequently Asked Questions

## Installation & Setup

### How do I install diive?

Use pip, conda, or poetry:

```bash
pip install diive
# or
conda install -c conda-forge diive
```

### Which Python versions are supported?

diive requires Python 3.10 or later. Tested with Python 3.10 and 3.11.

## Data Loading & I/O

### How do I load EddyPro output files?

Use `ReadFileType` to automatically detect the file format:

```python
import diive as dv

reader = dv.ReadFileType(filepath)
df = reader.read()
```

See [Loading Data](examples/io.md) for more examples.

### What file formats does diive support?

- EddyPro flux files (.txt)
- EddyPro biomet files
- TOA5 (Campbell Scientific)
- Parquet (.parquet)
- CSV (.csv)
- Custom formats (via DataFileReader)

### How do I save processed data?

```python
import diive as dv

dv.save_parquet(df, filepath)
```

## Processing & Analysis

### What's the difference between HeatmapXYZ and HexbinPlot?

- **HeatmapXYZ**: Creates a 2D grid of rectangular cells from X, Y, Z data. Good when X/Y values are discrete or you want a regular grid.
- **HexbinPlot**: Creates hexagonal bins. Better for continuous data with uneven distributions. More visually appealing for large datasets.

### How do I handle missing values (NaN)?

diive respects pandas NaN handling. Options:

- Use quality control flags to mask suspicious data
- Use gap-filling algorithms to estimate missing values
- Interpolate using `fillna()` or linear interpolation

### How do I aggregate data to a different time resolution?

See [Time Series Processing](examples/timeseries.md) for resampling examples.

## Gap-Filling

### Which gap-filling method should I use?

It depends on your data and requirements:

- **MDS (Marginal Distribution Sampling)**: Good for smaller gaps, fast, interpretable
- **Random Forest**: Good for medium-sized gaps, captures nonlinear relationships
- **XGBoost**: More powerful, better for complex patterns, requires tuning
- **Linear interpolation**: Simple, only for very small gaps

See [Gap-Filling Comparison](guide/workflows.md#gap-filling-comparison) for a detailed comparison.

### Can I compare multiple gap-filling methods?

Yes! Run each method on the same data and compare statistics:

```python
from diive.pkgs.gapfilling import FluxMDS, RandomForestTS, XGBoostTS

# Create fillers
mds = FluxMDS(...)
rf = RandomForestTS(...)
xgb = XGBoostTS(...)

# Fill gaps
mds_filled = mds.df_filled
rf_filled = rf.df_filled
xgb_filled = xgb.df_filled

# Compare
import pandas as pd
comparison = pd.DataFrame({
    'MDS': mds_filled,
    'RF': rf_filled,
    'XGBoost': xgb_filled
})
```

## Quality Control & Outliers

### How do I detect outliers?

diive provides multiple outlier detection methods:

```python
from diive.pkgs.outlierdetection import zScore, HampelDaytimeNighttime

# Simple z-score
detector = zScore(data=df['NEE_CUT_REF_f'], threshold=3)
detector.detect()

# More advanced: Hampel filter for day/night separately
detector = HampelDaytimeNighttime(data=df['NEE_CUT_REF_f'])
detector.detect()
```

See [Quality Control Examples](examples/qc.md) for more methods.

### What's the difference between flagging and removing data?

- **Flagging**: Mark suspicious data with a flag value; keep the original data for reference
- **Removing**: Delete or set to NaN; data is lost

diive recommends flagging first, then deciding what to do with flagged data based on your workflow.

## Visualization

### How do I create publication-quality figures?

Use matplotlib styling with diive's plotting classes:

```python
import matplotlib.pyplot as plt
import diive as dv

# Set style before creating plots
plt.style.use('seaborn-v0_8-darkgrid')

# Create plot
hm = dv.heatmapdatetime(data=df['NEE_CUT_REF_f'])
hm.plot()
```

See [Plotting Examples](examples/plotting.md) for detailed examples.

### Can I customize colors and fonts?

Yes, all plotting classes accept matplotlib parameters:

```python
hm = dv.hexbinplot(
    x=df['Tair_f'],
    y=df['VPD_f'],
    z=df['NEE_CUT_REF_f'],
    cmap='RdYlBu_r',  # colormap
    figsize=(10, 6),
    dpi=300  # for publication
)
```

## Data Uploads & Standards

### How do I format data for FLUXNET upload?

Use the formatting utilities:

```python
from diive.pkgs.formats import FormatEddyProFluxnetFileForUpload

formatter = FormatEddyProFluxnetFileForUpload(
    df_data=df,
    site_code='CH-Oe2',
    ...
)
formatter.format()
```

See [Flux Processing Examples](examples/flux.md) for complete examples.

## Performance & Scaling

### How do I speed up processing for large datasets?

- Use `polars` instead of `pandas` for I/O (faster for large files)
- Subset data by year/month before processing
- Parallelize gap-filling across sites using Python's `multiprocessing`
- Pin numpy/pandas versions for consistency

### What's the typical memory requirement?

For typical flux tower data (30-minute resolution, 1-2 years):
- ~100 MB for DataFrame
- Gap-filling algorithms: add ~50-200 MB depending on method
- Plotting: minimal overhead

## Contributing & Development

### How do I contribute to diive?

1. Fork the repository on GitHub
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See the project README for more details.

### How do I report a bug?

Open an issue on [GitHub](https://github.com/username/diive) with:
- Python version
- diive version (`import diive; print(diive.__version__)`)
- Minimal code to reproduce
- Error traceback

---

Didn't find your answer? [Open an issue](https://github.com/username/diive) or start a discussion.
