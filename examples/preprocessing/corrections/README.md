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
from diive.preprocessing.corrections import OffsetRH

# RH sensor drifts >100% due to aging/contamination
corrector = OffsetRH(series=df['RH'], max_saturation=100.0)
corrected = corrector.get_corrected()
```

**Remove nighttime radiation offset:**
```python
from diive.preprocessing.corrections import OffsetRadiation

# Radiation sensor reads non-zero at night (thermal offset)
corrector = OffsetRadiation(
    series=df['SW_IN'],
    swinpot_col=df['SW_IN_POT'],
    site_lat=47.5, site_lon=8.4
)
corrected = corrector.get_corrected()
```

**Detect instrument offset:**
```python
from diive.preprocessing.corrections import OffsetMeasurementReplicates

# Two sensors show constant bias
offset = OffsetMeasurementReplicates(
    series1=df['TA_primary'],
    series2=df['TA_reference']
)
bias = offset.get_offset()  # Constant difference
```

**Mask known problems:**
```python
from diive.preprocessing.corrections import SetToValue

# Instrument malfunction 2024-01-15 to 2024-01-17
corrector = SetToValue(
    series=df['CO2'],
    start_date='2024-01-15',
    end_date='2024-01-17',
    value=np.nan
)
corrected = corrector.get_corrected()
```

## Related Documentation

See `diive.pkgs.preprocessing.corrections` for available correction classes:
- `OffsetRH` — Relative humidity saturation correction
- `OffsetRadiation` — Nighttime radiation offset removal
- `OffsetMeasurementReplicates` — Instrument bias detection
- `OffsetWindDir` — Wind direction calibration
- `SetExactValuesToMissing` — Replace sentinel values
- `SetToValue` — Replace period with constant
- `SetToThreshold` — Clip to min/max bounds

## Running Examples

```bash
# Sensor bias corrections
uv run python examples/pkgs/preprocessing/corrections/correction_relativehumidity_offset.py
uv run python examples/pkgs/preprocessing/corrections/correction_radiation_offset.py
uv run python examples/pkgs/preprocessing/corrections/correction_measurement_offset_replicate.py
uv run python examples/pkgs/preprocessing/corrections/correction_winddir_offset.py

# Value replacement & clipping
uv run python examples/pkgs/preprocessing/corrections/correction_set_exact_values_to_missing.py
uv run python examples/pkgs/preprocessing/corrections/correction_setto_value.py
uv run python examples/pkgs/preprocessing/corrections/correction_setto_threshold.py

# Run all correction examples
uv run python examples/run_all_examples.py
```
