# Binary Data I/O Examples

Extract and manipulate information stored in binary-encoded values and bit fields.

## Contents

### Binary Value Extraction
- **io_extract.py** — Extract individual bits from integer values with optional gain factors

## Related Documentation

See `diive.pkgs.io.binary` for:
- `get_encoded_value_from_int()` — Extract bits from a single integer
- `get_encoded_value_series()` — Extract bits from a series of integers

## Running Examples

```bash
# Binary value extraction
uv run python examples/pkgs/io/io_extract.py
```

Or run all examples:

```bash
uv run python examples/run_all_examples.py
```

## Use Cases

- Decoding diagnostic flags from instrument output
- Working with binary-encoded measurement quality indicators
- Extracting AGC and other encoded values from data providers
- Bit-level data manipulation and value extraction
