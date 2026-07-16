# Data Corrections Examples

Examples demonstrating data corrections including sensor offset detection, bias removal, and value clipping.

**7 examples covering physical corrections, calibration drift detection, and data validation.**

## Contents by Correction Type

### Sensor Offset & Bias Corrections
- **correction_relativehumidity_offset.py** — Fix RH saturation issues (measurements >100%)
- **correction_radiation_offset.py** — Correct radiation nighttime offset (non-zero readings at night)
- **correction_measurement_offset_replicate.py** — Detect constant bias between two instruments
- **correction_winddir_offset.py** — Correct wind direction calibration drift

### Value Replacement & Clipping
- **correction_set_exact_values_to_missing.py** — Replace exact values with NaN (error codes, sentinel values)
- **correction_setto_value.py** — Replace values in specific periods (e.g., known malfunction times)
- **correction_setto_threshold.py** — Clip values to physically realistic min/max bounds

## Use Cases

**Fix humidity oversaturation:**
```python
import diive as dv

# RH sensor drifts >100% due to aging/contamination.
# The daily mean excess is removed as offset, values are capped at 100%.
corrected = dv.corrections.remove_relativehumidity_offset(series=df['RH'], showplot=True)
```

**Remove nighttime radiation offset:**
```python
import diive as dv

# Radiation sensor reads non-zero at night (thermal offset).
# Nighttime is derived from solar geometry, so the site location is needed.
corrected = dv.corrections.remove_nighttime_zero_offset(
    series=df['SW_IN'],
    lat=47.478333,
    lon=8.364389,
    utc_offset=1,
    showplot=True
)
```

**Detect instrument offset:**
```python
import diive as dv

# Two sensors show constant bias, offset found by brute-force search
offset_corrector = dv.corrections.MeasurementOffsetFromReplicate(
    measurement=df['TA_primary'],
    replicate=df['TA_reference'],
    offset_start=-10,
    offset_end=10,
    offset_stepsize=0.1
)
offset = offset_corrector.get_offset()  # Detected offset, added to correct
corrected = offset_corrector.get_corrected_measurement()
```

**Mask known problems:**
```python
import numpy as np
import diive as dv

# Instrument malfunction 2024-01-15 to 2024-01-17.
# Each entry in *dates* is either a single timestamp or a [start, end] range.
corrected = dv.corrections.setto_value(
    series=df['CO2'],
    dates=[['2024-01-15 00:00:00', '2024-01-17 00:00:00']],
    value=np.nan,
    verbose=1
)
```

## Related Documentation

See `dv.corrections` for available corrections:
- `remove_relativehumidity_offset` — Relative humidity saturation correction
- `remove_nighttime_zero_offset` — Nighttime radiation offset removal
- `nighttime_zero_offset_diagnostics` — Diagnostics for the nighttime radiation offset
- `NighttimeZeroOffsetResult` — Result container returned by the diagnostics
- `MeasurementOffsetFromReplicate` — Instrument bias detection against a replicate
- `WindDirOffset` — Wind direction calibration
- `set_exact_values_to_missing` — Replace sentinel values
- `setto_value` — Replace given timestamps or periods with a constant
- `setto_threshold` — Clip to a min or max bound
- `apply_corrections` — Apply a list of corrections to one series

## Running Examples

```bash
# Sensor bias corrections
uv run python examples/preprocessing/corrections/correction_relativehumidity_offset.py
uv run python examples/preprocessing/corrections/correction_radiation_offset.py
uv run python examples/preprocessing/corrections/correction_measurement_offset_replicate.py
uv run python examples/preprocessing/corrections/correction_winddir_offset.py

# Value replacement & clipping
uv run python examples/preprocessing/corrections/correction_set_exact_values_to_missing.py
uv run python examples/preprocessing/corrections/correction_setto_value.py
uv run python examples/preprocessing/corrections/correction_setto_threshold.py

# Run all correction examples
uv run python examples/run_all_examples.py
```
