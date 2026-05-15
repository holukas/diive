# Example Dataset Documentation

## Overview

The example dataset is a **10-year record of eddy covariance measurements** from the Davos site (CH-DAV), a mixed Alpine forest ecosystem in Switzerland. This dataset is included with DIIVE for demonstration purposes and is used throughout the examples.

## Dataset Characteristics

| Property | Value |
|----------|-------|
| **Site** | Davos, Switzerland (CH-DAV) |
| **Ecosystem** | Mixed Alpine forest |
| **Time Period** | 2013-01-01 to 2022-12-31 |
| **Duration** | 10 years (3,651 days) |
| **Temporal Resolution** | 30 minutes |
| **Total Records** | 175,296 half-hour measurements |
| **Spatial Coverage** | Single-point eddy covariance tower |
| **Data Source** | FLUXNET / Swiss FluxNet |

## Loading the Dataset

```python
import diive as dv

# Load example data
df = dv.load_exampledata_parquet()

# Basic info
print(df.shape)          # (175296, 37)
print(df.index.min())    # 2013-01-01 00:15:00
print(df.index.max())    # 2022-12-31 23:45:00
print(df.columns)        # List of 37 variables
```

## Variables (37 columns)

### **Flux Variables**

| Variable | Units | Description | Availability |
|----------|-------|-------------|---------------|
| `NEE_CUT_REF_f` | µmol m⁻² s⁻¹ | Net Ecosystem Exchange (reference USTAR) — **gap-filled** | Complete (0% missing) |
| `NEE_CUT_16_f` | µmol m⁻² s⁻¹ | NEE with conservative USTAR threshold (16th percentile) — gap-filled | Complete |
| `NEE_CUT_84_f` | µmol m⁻² s⁻¹ | NEE with lenient USTAR threshold (84th percentile) — gap-filled | Complete |
| `NEE_CUT_REF_orig` | µmol m⁻² s⁻¹ | Original measured NEE (reference) | ~67% (66.8% missing) |
| `NEE_CUT_16_orig` | µmol m⁻² s⁻¹ | Original NEE (conservative USTAR) | ~40% (60.0% missing) |
| `NEE_CUT_84_orig` | µmol m⁻² s⁻¹ | Original NEE (lenient USTAR) | ~23% (77.2% missing) |
| `GPP_CUT_REF_f` | µmol m⁻² s⁻¹ | Gross Primary Productivity (gap-filled) | Complete |
| `GPP_CUT_16_f` | µmol m⁻² s⁻¹ | GPP (conservative USTAR, gap-filled) | Complete |
| `GPP_CUT_84_f` | µmol m⁻² s⁻¹ | GPP (lenient USTAR, gap-filled) | Complete |
| `GPP_DT_CUT_REF` | µmol m⁻² s⁻¹ | GPP from daytime NEE (reference) | Complete |
| `GPP_DT_CUT_16` | µmol m⁻² s⁻¹ | GPP from daytime NEE (conservative) | Complete |
| `GPP_DT_CUT_84` | µmol m⁻² s⁻¹ | GPP from daytime NEE (lenient) | Complete |
| `Reco_CUT_REF` | µmol m⁻² s⁻¹ | Ecosystem respiration (reference) | Complete |
| `Reco_CUT_16` | µmol m⁻² s⁻¹ | Respiration (conservative USTAR) | Complete |
| `Reco_CUT_84` | µmol m⁻² s⁻¹ | Respiration (lenient USTAR) | Complete |
| `Reco_DT_CUT_REF` | µmol m⁻² s⁻¹ | Respiration from nighttime NEE | Complete |
| `Reco_DT_CUT_16` | µmol m⁻² s⁻¹ | Respiration (conservative) | Complete |
| `Reco_DT_CUT_84` | µmol m⁻² s⁻¹ | Respiration (lenient) | Complete |
| `Reco_DT_CUT_REF_SD` | µmol m⁻² s⁻¹ | Standard deviation of respiration | Complete |
| `LE_f` | W m⁻² | Latent heat flux (gap-filled) | Complete |
| `LE_orig` | W m⁻² | Latent heat flux (measured) | ~71% (29.2% missing) |
| `ET_f` | mm day⁻¹ | Evapotranspiration (converted from LE) | Complete |

### **Meteorological Variables**

| Variable | Units | Description | Availability |
|----------|-------|-------------|---------------|
| `Tair_f` | °C | Air temperature (gap-filled) | Complete |
| `Tair_orig` | °C | Air temperature (measured) | ~100% (0% missing) |
| `RH` | % | Relative humidity | ~99.6% (0.4% missing) |
| `VPD_f` | hPa | Vapor Pressure Deficit (gap-filled) | Complete |
| `VPD_orig` | hPa | VPD (measured) | ~100% (0% missing) |
| `PA` | hPa | Atmospheric pressure | ~96.9% (3.1% missing) |

### **Radiation Variables**

| Variable | Units | Description | Availability |
|----------|-------|-------------|---------------|
| `Rg_f` | W m⁻² | Global shortwave radiation (gap-filled) | Complete |
| `Rg_orig` | W m⁻² | Global shortwave radiation (measured) | ~100% (0% missing) |
| `PPFD` | µmol m⁻² s⁻¹ | Photosynthetic photon flux density | ~99.9% (0.1% missing) |
| `LW_IN` | W m⁻² | Incoming longwave radiation | ~88.1% (11.9% missing) |

### **Soil/Water Variables**

| Variable | Units | Description | Availability |
|----------|-------|-------------|---------------|
| `SWC_FF0_0.15_1` | % | Soil water content at 0.15 m depth | ~93.9% (6.1% missing) |

### **Precipitation & Other**

| Variable | Units | Description | Availability |
|----------|-------|-------------|---------------|
| `PREC_TOT_T1_25+20_1` | mm | Total precipitation | ~99.7% (0.3% missing) |

### **Quality Control & Processing**

| Variable | Type | Description | Availability |
|----------|------|-------------|---------------|
| `QCF_NEE` | 0/1/2 | NEE Quality Control Flag (0=good, 1=marginal, 2=bad) | ~82.5% (17.5% missing) |
| `QCF_LE` | 0/1/2 | LE Quality Control Flag | ~78.2% (21.8% missing) |
| `Ustar_CUT_REF_Thres` | m/s | Friction velocity (USTAR) threshold | Complete |

## Data Quality Notes

### **Gap-Filling Status**
- **Gap-filled variables** (suffix `_f`): NEE, GPP, LE, temperature, VPD, radiation
  - Ready to use for analysis without additional gap-filling
  - Used in flux processing pipeline (Levels 2-4.1)
- **Original measured variables** (suffix `_orig`): Raw measurements with gaps
  - Useful for demonstrating gap-filling methods
  - Contains natural measurement gaps during instrument downtime, maintenance, precipitation, etc.

### **Missing Data Patterns**
- **High-gap fluxes**: NEE measurements with USTAR filtering have 60-77% gaps (conservative vs. lenient thresholds)
- **Meteorological data**: Generally complete (>99%)
- **Quality flags**: ~18% missing (no QC applied during those periods)

### **USTAR Filtering Scenarios**
The dataset includes three USTAR threshold scenarios:
- **REF (Reference)**: Standard threshold
- **CUT_16**: Conservative (16th percentile) — more stringent filtering
- **CUT_84**: Lenient (84th percentile) — less stringent filtering

This allows demonstration of uncertainty quantification in flux processing.

## Recommended Uses

### **Good For:**
- ✓ Demonstrating gap-filling methods (RandomForest, XGBoost, MDS)
- ✓ Quality control and outlier detection workflows
- ✓ Visualization and time series analysis
- ✓ Feature engineering and variable creation
- ✓ Flux processing pipeline (Levels 2-4.1)
- ✓ Uncertainty quantification
- ✓ Comparison of USTAR filtering scenarios

### **Not Suitable For:**
- ✗ Establishing ecosystem-specific conclusions (only one site, one ecosystem)
- ✗ High-frequency eddy covariance analysis (30-min resolution is too coarse for spectral analysis)
- ✗ Sub-hourly process studies

## Data Availability & Citation

This example dataset is derived from the **Swiss FluxNet** and **FLUXNET2015** archives. The Davos site (CH-DAV) is a long-term measurement station operated by Swiss Federal Institute for Forest, Snow and Landscape Research (WSL).

**For proper citation**, refer to FLUXNET2015 data policy and the site-specific data access agreements.

## Example Code

Load and inspect:
```python
import diive as dv

df = dv.load_exampledata_parquet()

# Select NEE (measured vs gap-filled)
nee_measured = df['NEE_CUT_REF_orig']  # Has gaps
nee_filled = df['NEE_CUT_REF_f']       # Complete, gap-filled

# Select meteorological drivers
meteo = df[['Tair_f', 'RH', 'VPD_f', 'PPFD', 'Rg_f']]

# Check quality
qcf_nee = df['QCF_NEE']  # Quality flag

# Filter by USTAR scenario
nee_conservative = df['NEE_CUT_16_f']  # More gaps removed
nee_lenient = df['NEE_CUT_84_f']       # Fewer gaps removed
```

## File Location

```
diive/configs/exampledata/
└── exampledata_PARQUET_CH-DAV_FP2022.5_2013-2022_ID20230206154316_30MIN.parquet
```

Loaded automatically via:
```python
df = diive.load_exampledata_parquet()
```
