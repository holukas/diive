# Package Modules API Reference

Package modules provide domain-specific algorithms and workflows for ecosystem data processing.

## Analyses

Data analysis and aggregation utilities.

```{autosummary}

diive.pkgs.analyses.GapFinder
diive.pkgs.analyses.GridAggregator
```

## Binary

Binary-encoded value handling.

```{autosummary}

diive.pkgs.binary
```

## Corrections

Measurement corrections and offset handling.

```{autosummary}

diive.pkgs.corrections.OffsetCorrection
diive.pkgs.corrections.WindDirectionOffset
diive.pkgs.corrections.SetToThreshold
diive.pkgs.corrections.SetToMissing
```

## Variable Creation

Derive new variables from measurements.

```{autosummary}

diive.pkgs.createvar.DaytimeNighttimeFlag
diive.pkgs.createvar.TimeSince
```

## Eddy Covariance (20 Hz)

High-frequency eddy covariance processing.

```{autosummary}

diive.pkgs.echires.FluxDetectionLimit
diive.pkgs.echires.WindRotation2D
diive.pkgs.echires.MaxCovariance
```

## Fits

Fitting utilities for binned data.

```{autosummary}

diive.pkgs.fits.BinFitterCP
```

## Flux Processing

Flux quality control and processing chains.

```{autosummary}

diive.pkgs.flux.FluxProcessingChain
```

## Formats

Format conversion for data standards.

```{autosummary}

diive.pkgs.formats.FormatEddyProFluxnetFileForUpload
diive.pkgs.formats.FormatMeteoForEddyProFluxProcessing
```

## Gap-Filling

Methods for filling missing data.

```{autosummary}

diive.pkgs.gapfilling.XGBoostTS
diive.pkgs.gapfilling.RandomForestTS
diive.pkgs.gapfilling.QuickFillRFTS
diive.pkgs.gapfilling.LongTermGapFillingBase
diive.pkgs.gapfilling.FluxMDS
```

## Outlier Detection

Automated outlier and anomaly detection.

```{autosummary}

diive.pkgs.outlierdetection.HampelDaytimeNighttime
diive.pkgs.outlierdetection.zScore
diive.pkgs.outlierdetection.zScoreDaytimeNighttime
diive.pkgs.outlierdetection.LocalOutlierFactorAllData
diive.pkgs.outlierdetection.AbsoluteLimits
diive.pkgs.outlierdetection.LocalSD
diive.pkgs.outlierdetection.ManualRemoval
diive.pkgs.outlierdetection.MissingValues
diive.pkgs.outlierdetection.TrimLow
```

## Quality Assurance & Quality Control

FLUXNET QC and data screening.

```{autosummary}

diive.pkgs.qaqc.FlagQCF
diive.pkgs.qaqc.StepwiseMeteoScreeningDb
```
