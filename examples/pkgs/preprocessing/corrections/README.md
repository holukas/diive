# Data Corrections Examples

Examples demonstrating correction methods for time series data including offset and bias corrections.

## Contents

**Offset Corrections (Sphinx Gallery format):**
- **correction_relativehumidity_offset.py** — Fix RH measurements exceeding 100%
- **correction_radiation_offset.py** — Correct radiation nighttime offset
- **correction_measurement_offset_replicate.py** — Detect offset vs. reference replicate
- **correction_winddir_offset.py** — Correct wind direction calibration drift

**Value Replacement & Clipping:**
- **correction_set_exact_values_to_missing.py** — Set exact values to NaN (remove error codes/sentinel values)
- **correction_setto_value.py** — Replace values in specific time periods with a constant
- **correction_setto_threshold.py** — Clip values to minimum/maximum thresholds

## Related Documentation

See `diive.pkgs.preprocessing.corrections` for available correction classes and functions.

## Usage

```bash
uv run python examples/pkgs/preprocessing/corrections/correction_relativehumidity_offset.py
uv run python examples/pkgs/preprocessing/corrections/correction_radiation_offset.py
uv run python examples/pkgs/preprocessing/corrections/correction_measurement_offset_replicate.py
uv run python examples/pkgs/preprocessing/corrections/correction_winddir_offset.py
uv run python examples/pkgs/preprocessing/corrections/correction_set_exact_values_to_missing.py
uv run python examples/pkgs/preprocessing/corrections/correction_setto_value.py
uv run python examples/pkgs/preprocessing/corrections/correction_setto_threshold.py
```
