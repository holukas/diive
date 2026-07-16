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

Available at the top level as `dv.`, implemented in `diive.io.binary.extract`:
- `get_encoded_value_from_int()` — Extract bits from a single integer
- `get_encoded_value_series()` — Extract bits from a series of integers

The bit subrange is given as `bit_start` (inclusive) and `bit_end` (exclusive), counted from
the left of the zero-padded binary representation. `n_bits` is the total word width the value
is padded to (default 8), not the number of bits extracted.

## Use Cases

**Decode binary-encoded measurements:**
```python
import diive as dv

# Extract bits 5-7 from diagnostic codes
diagnostic_code = 156  # Example 8-bit integer, binary 10011100
value = dv.get_encoded_value_from_int(
    integer=diagnostic_code,
    bit_start=5,  # First bit of the subrange (inclusive)
    bit_end=8,    # End of the subrange (exclusive), so bits 5-7
    gain=1,
    base=2,
    n_bits=8      # Word width the integer is padded to
)
# Result: bits '100' -> 4

# Apply to series
flags = dv.get_encoded_value_series(
    int_series=df['DIAG_BYTE'],
    bit_start=5,
    bit_end=8,
    gain=1,
    base=2,
    n_bits=8
)
```

**Extract quality indicators:**
```python
import diive as dv

# AGC mean, encoded in bits 4-7 of the gas analyzer diagnostic value
agc = dv.get_encoded_value_series(
    int_series=df['GA_DIAG_VALUE'],
    bit_start=4,
    bit_end=8,
    gain=6.25,  # Scale: each unit = 6.25%
    base=2,
    n_bits=8
)
# Result: AGC in percent (0-100), e.g. 250 -> bits '1010' (10) -> 62.5
```

## Running Examples

```bash
# Binary value extraction
uv run python examples/io/io_extract.py

# Run all examples
uv run python examples/run_all_examples.py
```

## Common Applications

- **Diagnostic byte decoding** — Extract individual test flags from multi-bit diagnostic codes
- **Quality indicator extraction** — AGC mean, signal strength from encoded bytes
- **Instrumental metadata** — Sensor state, mode, configuration bits
- **EddyPro file processing** — Decoding binary-encoded QC flags
- **Data validation** — Checking measurement quality from encoded bits
