# Package Modules API Reference

Package modules provide domain-specific algorithms and workflows for ecosystem data processing.

## Analyses

Data analysis and aggregation utilities.

- `diive.pkgs.analyses.GapFinder` - Find gaps in time series data
- `diive.pkgs.analyses.GridAggregator` - Aggregate data into 2D grids

## Binary

Binary-encoded value handling.

- `diive.pkgs.binary` - Binary encoding and decoding utilities

## Corrections

Measurement corrections and offset handling.

- `diive.pkgs.corrections.OffsetCorrection` - Apply offset corrections
- `diive.pkgs.corrections.WindDirectionOffset` - Wind direction offset corrections
- `diive.pkgs.corrections.SetToThreshold` - Set values to threshold
- `diive.pkgs.corrections.SetToMissing` - Set invalid values to missing

## Variable Creation

Derive new variables from measurements.

- `diive.pkgs.createvar.DaytimeNighttimeFlag` - Flag daytime/nighttime periods
- `diive.pkgs.createvar.TimeSince` - Calculate time since event

## Eddy Covariance (20 Hz)

High-frequency eddy covariance processing.

- `diive.pkgs.echires.FluxDetectionLimit` - Calculate flux detection limits
- `diive.pkgs.echires.WindRotation2D` - 2D wind rotation
- `diive.pkgs.echires.MaxCovariance` - Maximum covariance analysis

## Fits

Fitting utilities for binned data.

- `diive.pkgs.fits.BinFitterCP` - Fitting for change points in bins

## Flux Processing

Flux quality control and processing chains.

- `diive.pkgs.flux.FluxProcessingChain` - Complete flux processing workflow

## Formats

Format conversion for data standards.

- `diive.pkgs.formats.FormatEddyProFluxnetFileForUpload` - Format for FLUXNET upload
- `diive.pkgs.formats.FormatMeteoForEddyProFluxProcessing` - Format for EddyPro processing

## Gap-Filling

Methods for filling missing data.

- `diive.pkgs.gapfilling.XGBoostTS` - XGBoost gap-filling
- `diive.pkgs.gapfilling.RandomForestTS` - Random Forest gap-filling
- `diive.pkgs.gapfilling.QuickFillRFTS` - Fast Random Forest gap-filling
- `diive.pkgs.gapfilling.LongTermGapFillingBase` - Base class for long-term gap-filling
- `diive.pkgs.gapfilling.FluxMDS` - Marginal Distribution Sampling for flux data

## Outlier Detection

Automated outlier and anomaly detection.

- `diive.pkgs.outlierdetection.HampelDaytimeNighttime` - Hampel filter for day/night
- `diive.pkgs.outlierdetection.zScore` - Z-score outlier detection
- `diive.pkgs.outlierdetection.zScoreDaytimeNighttime` - Z-score for day/night
- `diive.pkgs.outlierdetection.LocalOutlierFactorAllData` - Local Outlier Factor
- `diive.pkgs.outlierdetection.AbsoluteLimits` - Absolute limit checking
- `diive.pkgs.outlierdetection.LocalSD` - Local standard deviation method
- `diive.pkgs.outlierdetection.ManualRemoval` - Manual outlier removal
- `diive.pkgs.outlierdetection.MissingValues` - Identify missing values
- `diive.pkgs.outlierdetection.TrimLow` - Trim low values

## Quality Assurance & Quality Control

FLUXNET QC and data screening.

- `diive.pkgs.qaqc.FlagQCF` - FLUXNET QC flag handling
- `diive.pkgs.qaqc.StepwiseMeteoScreeningDb` - Meteorological screening
