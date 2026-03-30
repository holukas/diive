# Getting Started with diive

**diive** is a Python library for processing and analyzing time series data from ecosystem observations, particularly eddy covariance flux measurements. It provides tools for data quality control, gap-filling, statistical analysis, and visualization.

Originally developed by the ETH Grassland Sciences group for [Swiss FluxNet](https://www.swissfluxnet.ethz.ch/), diive is designed for scientists and data engineers working with environmental monitoring data.

## What can you do with diive?

- **Load & Read**: Parse EddyPro flux data, meteorological files, and custom formats
- **Quality Control**: Automated outlier detection, flag generation, and data screening
- **Gap-Filling**: Multiple algorithms (MDS, Random Forest, XGBoost) for missing data
- **Analysis**: Binning, aggregation, correlation, and statistical summaries
- **Visualization**: Publication-quality heatmaps, time series, scatter plots, and histograms
- **Flux Processing**: Complete workflows from raw measurements to FLUXNET-ready data

## Installation

Install diive using pip, conda, or poetry:

```bash
# pip
pip install diive

# conda
conda install -c conda-forge diive

# poetry (from project directory)
poetry install
```

## Quick Start

### 1. Load Example Data

```python
import diive as dv
import pandas as pd

# Load built-in example data (parquet format)
df = dv.load_exampledata_parquet()
print(df.head())
```

### 2. Process Data

```python
# Example: Detect and remove outliers
from diive.pkgs.outlierdetection import zScore

outlier_detector = zScore(data=df['NEE_CUT_REF_f'])
outlier_detector.detect()
clean_data = df.loc[~outlier_detector.flag, 'NEE_CUT_REF_f']
```

### 3. Create Visualization

```python
# Create a hexagonal bin plot
hm = dv.hexbinplot(
    x=df['Tair_f'],
    y=df['VPD_f'],
    z=df['NEE_CUT_REF_f'],
    gridsize=12
)
hm.show()
```

## Next Steps

Choose your path based on what you want to do:

### 📥 **Reading Data**
Start with [Loading & Reading Data](examples/io.md) to learn how to:
- Read EddyPro output files
- Load meteorological data
- Work with parquet and CSV formats

### 🔧 **Processing Time Series**
Check out [Time Series Processing](examples/timeseries.md) for:
- Detecting time resolution
- Resampling and aggregation
- Timestamp handling

### 📊 **Quality Control & Outlier Detection**
Explore [Quality Control](examples/qc.md) to:
- Flag suspicious measurements
- Detect outliers using multiple methods
- Screen meteorological data

### 📈 **Gap-Filling**
Learn about gap-filling strategies in [Gap-Filling](examples/gapfilling.md):
- MDS method for large gaps
- Machine learning approaches (Random Forest, XGBoost)
- Method comparison and parameter optimization

### 💨 **Flux Processing**
Follow [Flux Processing Workflows](examples/flux.md) for:
- Complete Level-2 to Level-4 processing pipelines
- USTAR filtering and self-heating correction
- Uncertainty estimation

### 📉 **Visualization**
Create publication-quality figures with [Plotting Examples](examples/plotting.md):
- Heatmaps (time-based, XYZ binned, hexagonal)
- Time series and scatter plots
- Diel cycles and ridge line plots

## Project Structure

**diive** is organized into two main areas:

- **Core modules** (`diive.core.*`) — Foundational utilities shared across the library
  - File I/O, timestamp handling, statistics, plotting, machine learning

- **Package modules** (`diive.pkgs.*`) — Domain-specific algorithms
  - Quality control, gap-filling, flux processing, variable creation, corrections

See [System Architecture](guide/architecture.md) for a detailed overview.

## Common Workflows

- [Flux processing from raw EddyPro output → publication-ready data](guide/workflows.md#flux-processing-pipeline)
- [Gap-filling: comparison of methods](guide/workflows.md#gap-filling-comparison)
- [Quality control: automated outlier detection](guide/workflows.md#quality-control-workflow)
- [Creating publication-quality figures](guide/workflows.md#visualization-workflow)

## API Design Patterns

Understanding diive's conventions will help you write cleaner code:

- Series naming requirements (`.name` attribute)
- Column naming patterns (e.g., `BIN_Tair_f` for binned variables)
- Choosing between similar classes
- Common parameter patterns

See [API Design & Patterns](guide/api_design.md) for details.

## Need Help?

- **Common questions?** Check the [FAQ](faq.md)
- **API reference?** Browse the [API Documentation](../api)
- **Examples?** Explore the [Example Notebooks](examples)
- **Still stuck?** Open an issue on [GitHub](https://github.com/username/diive)

## Citation

If you use diive in your research, please cite:

```bibtex
@software{holukas2024diive,
  title = {diive: Time series processing for ecosystem observations},
  author = {Hörtnagl, Lukas},
  year = {2024},
  url = {https://github.com/username/diive}
}
```

---

Ready to dive in? Start with the [examples](examples) or explore [system architecture](guide/architecture.md).
