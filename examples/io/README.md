# Data I/O Examples

Read, write, and manipulate data in various formats.

## Contents

### Parquet File I/O
- **io_load_save_parquet.py** — Save and reload DataFrames as Parquet files with automatic timestamp sanitization

### EddyPro CSV File Reading
- **io_read_single_file_with_datafilereader.py** — Read single file with manual parameter specification
- **io_read_multiple_files_with_multidatafilereader.py** — Load and merge multiple files with pre-defined filetype
- **io_read_single_file_with_readfiletype.py** — Read single file with pre-defined filetype configuration

### Binary Data Extraction
- **io_extract.py** — Extract individual bits from integer values with optional gain factors

## Related Documentation

See `diive.io.binary` for:
- `get_encoded_value_from_int()` — Extract bits from a single integer
- `get_encoded_value_series()` — Extract bits from a series of integers

## Use Cases

**Decode binary-encoded measurements:**
```python
from diive.io.binary import get_encoded_value_from_int, get_encoded_value_series

# Extract bits 5-7 from diagnostic codes
diagnostic_code = 156  # Example 8-bit integer
value = get_encoded_value_from_int(
    value=diagnostic_code,
    position=5,  # Start at bit 5
    n_bits=3,   # Extract 3 bits
    gain=1.0
)

# Apply to series
flags = get_encoded_value_series(
    series=df['DIAG_BYTE'],
    position=5,
    n_bits=3,
    gain=1.0
)
```

**Extract quality indicators:**
```python
from diive.io.binary import get_encoded_value_series

# AGC mean (often encoded in lower bits)
agc = get_encoded_value_series(
    series=df['AGC_ENCODED'],
    position=0,    # Start at bit 0
    n_bits=8,      # 8-bit value
    gain=0.1       # Scale: each unit = 0.1%
)
# Result: AGC in percent (0-100)
```

## Running Examples

```bash
# Binary value extraction
uv run python examples/pkgs/io/io_extract.py

# Run all examples
uv run python examples/run_all_examples.py
```

## Common Applications

- **Diagnostic byte decoding** — Extract individual test flags from multi-bit diagnostic codes
- **Quality indicator extraction** — AGC mean, signal strength from encoded bytes
- **Instrumental metadata** — Sensor state, mode, configuration bits
- **EddyPro file processing** — Decoding binary-encoded QC flags
- **Data validation** — Checking measurement quality from encoded bits
