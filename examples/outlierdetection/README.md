# Outlier Detection Examples

Quality control and outlier detection methods for time series data.

## Overview

10+ robust outlier detection algorithms:

- **Absolute Limits** — Min/max threshold filtering with separate day/night thresholds
- **Hampel Filter** — Spike detection using Median Absolute Deviation (MAD)
- **Local Standard Deviation** — Adaptive rolling window thresholds
- **Z-Score** — Global, rolling, and day/night-stratified z-score detection
- **Z-Score Increments** — Detect abrupt changes in values
- **Local Outlier Factor (LOF)** — Density-based anomaly detection
- **Manual Removal** — Explicit outlier flagging and removal
- **Trim Filter** — Symmetric removal of extreme values
- **Step-Wise Orchestration** — Chain multiple methods for progressive filtering

## Use Cases

- Quality assurance for eddy covariance flux measurements
- Identifying instrument malfunctions and data artifacts
- Progressive multi-stage outlier filtering
- Robust data cleaning before gap-filling
