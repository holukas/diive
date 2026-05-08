# Code Review: TimestampSanitizer Class
**Status**: Comprehensive review for solid foundation  
**Date**: 2026-05-08

---

## EXECUTIVE SUMMARY

The `TimestampSanitizer` class is **well-architected** and handles a complex problem elegantly. However, there are **critical issues** and **improvements** needed for production robustness:

- ❌ **Critical**: Type hints use deprecated `Series or DataFrame` syntax
- ❌ **Critical**: Exception handling is too broad (bare `except:`)
- ⚠️ **High**: No error recovery/logging mechanism
- ⚠️ **High**: Docstring lacks best practices section
- ✅ **Good**: Clear separation of concerns with helper functions
- ✅ **Good**: Verbose mode useful for debugging

---

## 1. CRITICAL ISSUES

### 1.1 Type Hints: Use Union Instead of `or`

**Problem**: Lines 54, 116, 341, etc. use deprecated syntax:
```python
data: Series or DataFrame  # ❌ WRONG - runtime type hinting syntax
```

**Impact**: 
- Not type-checkable by mypy/pyright
- Confusing to IDE type inference
- Non-PEP 604 compliant

**Fix**:
```python
from typing import Union

data: Union[Series, DataFrame]  # ✅ PEP 484 compliant
# or if Python 3.10+:
data: Series | DataFrame  # ✅ PEP 604 modern
```

**Files affected**:
- Line 54, 116, 341 (and throughout entire `times.py`)

---

### 1.2 Bare Exception Handling

**Problem**: Line 362
```python
except:
    raise Exception("Conversion of timestamp to datetime format failed.")
```

**Issues**:
- Catches system exceptions (KeyboardInterrupt, SystemExit)
- Hides actual error for debugging
- No traceback preserved
- Violates best practices

**Fix**:
```python
except (ValueError, TypeError) as e:
    raise ValueError(
        f"Failed to convert timestamp to datetime format. "
        f"Original error: {e}"
    ) from e
```

**Similar issues**: Line 362

---

### 1.3 No Data Validation at Entry Point

**Problem**: `__init__` doesn't validate input:
```python
def __init__(self, data: Union[Series, DataFrame], ...):
    self.data = data.copy()  # ❌ What if data is None? Empty? No index?
```

**Impact**: Silent failures later in pipeline, hard to debug

**Fix**:
```python
def __init__(self, data: Union[Series, DataFrame], ...):
    self._validate_input(data)
    self.data = data.copy()

def _validate_input(self, data: Union[Series, DataFrame]) -> None:
    """Validate input before processing"""
    if data is None:
        raise TypeError("data cannot be None")
    if data.empty:
        raise ValueError("data cannot be empty")
    if data.index is None or len(data.index) == 0:
        raise ValueError("data must have a valid index")
    if not isinstance(data.index, pd.DatetimeIndex) and not isinstance(data.index, pd.Index):
        # Will catch issues before conversion step
        pass
```

---

## 2. HIGH PRIORITY ISSUES

### 2.1 State Management Issue: `inferred_freq` Double Assignment

**Problem**: Lines 110 & 144-145
```python
try:
    self.inferred_freq = None if not data.index.freq else data.index.freq  # Line 110
except AttributeError:
    self.inferred_freq = None

# ... later ...
if not self.inferred_freq:
    self.inferred_freq = DetectFrequency(...).get()  # Line 145
```

**Issue**: If `data.index.freq` exists, frequency detection is skipped. This creates **silent failure path**:
- User might have irregular frequency that pandas didn't detect
- Pre-existing wrong frequency isn't corrected

**Fix**: Always run detection, compare with existing:
```python
def _run(self):
    # Always detect frequency
    detected_freq = DetectFrequency(index=self.data.index, verbose=self.verbose).get()
    
    # If data had freq, validate consistency
    if self.data.index.freq and detected_freq != str(self.data.index.freq):
        if self.verbose:
            print(f"⚠️ Frequency mismatch: data.index.freq={self.data.index.freq}, "
                  f"detected={detected_freq}")
    
    self.inferred_freq = detected_freq
```

---

### 2.2 Silent Data Loss in Pipeline

**Problem**: The pipeline modifies `self.data` at each step. If any step fails mid-pipeline:
```python
self.data = convert_timestamp_to_datetime(...)  # Step 2 - fails
self.data = remove_rows_nat(...)  # Step 3 - never reaches
# User gets partially transformed data, doesn't know what happened
```

**Better approach**: Use try-except wrapper that reports what failed:
```python
def _run(self):
    steps = [
        ("timestamp naming validation", self._step_validate_naming),
        ("datetime conversion", self._step_convert_datetime),
        ("NaT removal", self._step_remove_nat),
        # ...
    ]
    
    for step_name, step_func in steps:
        try:
            step_func()
        except Exception as e:
            raise RuntimeError(
                f"Failed at step '{step_name}': {e}. "
                f"Data may be partially transformed."
            ) from e
```

---

### 2.3 Inefficient: `.copy()` on Every Call

**Problem**: Line 98
```python
self.data = data.copy()  # Line 98 - copies entire dataset
```

**Issue**: For large datasets, this is expensive. If user only wants to read (not modify), copy is unnecessary.

**Better**: 
```python
def __init__(self, data, make_copy=True, ...):
    self.data = data.copy() if make_copy else data
    # Document: "Set make_copy=False only if you won't modify data elsewhere"
```

---

## 3. MEDIUM PRIORITY: CODE QUALITY

### 3.1 Return Type Annotation Missing on Key Method

**Problem**: Line 116-117
```python
def get(self) -> Series or DataFrame:
    return self.data
```

Should be:
```python
def get(self) -> Union[Series, DataFrame]:
    return self.data
```

---

### 3.2 Unused Import

**Line 4**: 
```python
from ast import Index  # ❌ Never used
```

Remove it.

---

### 3.3 Magic Number in `DetectFrequency`

**Line 890**:
```python
checkrange = 1000
for ndr in range(checkrange, 3, -1):  # What does 1000 mean? Why 3?
```

Should be:
```python
MAX_CHECK_RANGE = 1000
MIN_CHECK_RANGE = 3

for ndr in range(MAX_CHECK_RANGE, MIN_CHECK_RANGE, -1):
```

---

## 4. DOCSTRING IMPROVEMENTS

### 4.1 Class Docstring: Missing Details

**Current**: Lines 24-51
```python
"""
Validate and prepare timestamps...
The processing pipeline (in order):
1. Validate timestamp naming...
"""
```

**Should add**:

```python
"""
Validate and prepare timestamps for time series data processing.

Performs comprehensive validation and sanitization of datetime indices through
a sequence of checks and transformations. Acts as a wrapper combining various
timestamp processing functions into a single, configurable interface.

The processing pipeline (in order):
1. Validate timestamp naming convention
2. Convert timestamp index to datetime format
...

Notes
-----
- All steps are reversible up to step 6 (frequency validation)
- Step 8 (regularize) fills gaps with NaN data rows
- Default settings assume END-of-period timestamps as input
- Does NOT preserve original data if validation fails

Raises
------
ValueError
    If nominal_freq doesn't match detected frequency
TypeError
    If data is None or empty
ValueError
    If timestamp index is not properly named

Examples
--------
>>> import pandas as pd
>>> import diive as dv
>>> df = pd.DataFrame({'value': [1, 2, 3]}, 
...                     index=pd.date_range('2023-01-01', periods=3, freq='h'))
>>> df.index.name = 'TIMESTAMP_END'
>>> sanitizer = dv.TimestampSanitizer(df, verbose=False)
>>> clean_df = sanitizer.get()

See Also
--------
examples/timeseries/timestamp_sanitizer.py — Timestamp validation with multiple examples
"""
```

---

### 4.2 __init__ Docstring: Missing Raises & Examples

**Current**: Lines 64-97 (incomplete)

**Add**:
```python
Parameters
----------
data : Union[Series, DataFrame]
    Data with timestamp index to be validated and processed.
    Index name must be one of: 'TIMESTAMP_END', 'TIMESTAMP_START', 'TIMESTAMP_MIDDLE'
    Index must be a DatetimeIndex (or convertible to it).
output_middle_timestamp : bool, optional
    If True, converts all timestamps to middle-of-period format.
    If False, keeps original timestamp convention. Default is True.
validate_naming : bool, optional
    Check if timestamp is correctly named. Allowed names are 'TIMESTAMP_END',
    'TIMESTAMP_START', and 'TIMESTAMP_MIDDLE'. Default is True.
    Set to False ONLY if you've already validated naming yourself.
convert_to_datetime : bool, optional
    Convert timestamp index to datetime format. Default is True.
    Set to False ONLY if index is already DatetimeIndex.
remove_index_nat : bool, optional
    Remove rows without timestamp (NaT values). Default is True.
    Set to False to keep NaT rows (will likely cause errors later).
sort_ascending : bool, optional
    Sort timestamp in ascending order. Default is True.
    Set to False ONLY if data is already sorted.
remove_duplicates : bool, optional
    Remove duplicate timestamps (keep last occurrence). Default is True.
    Set to False to keep duplicates (use with caution - can cause issues).
regularize : bool, optional
    Generate continuous timestamp without date gaps. Default is True.
    If True, gaps are filled with NaN rows.
    Set to False to keep data gaps.
nominal_freq : str, optional
    Expected time resolution of data timestamp index. If provided, detected
    frequency is validated against this. Raises ValueError if they don't match.
    Examples: '10s', '5s', 's', '30min', '5min', 'min', '1h', '3h', 'h'.
    Default is None (no frequency validation).
verbose : bool, optional
    Print detailed status messages during processing. Default is False.

Raises
------
TypeError
    If data is None or not a Series/DataFrame
ValueError
    If data is empty
ValueError
    If timestamp index name is not valid
ValueError
    If nominal_freq doesn't match detected frequency
RuntimeError
    If any processing step fails

Examples
--------
Basic usage with default settings:

>>> df = dv.load_exampledata_parquet()
>>> sanitizer = dv.TimestampSanitizer(data=df['NEE_CUT_REF_f'], verbose=False)
>>> clean_series = sanitizer.get()

With frequency validation:

>>> sanitizer = dv.TimestampSanitizer(
...     data=df['NEE_CUT_REF_f'],
...     nominal_freq='30min',  # Expect 30-minute data
...     verbose=True
... )

Selective processing (skip some steps):

>>> sanitizer = dv.TimestampSanitizer(
...     data=df,
...     regularize=False,        # Keep gaps
...     output_middle_timestamp=False  # Keep end-of-period format
... )
"""
```

---

## 5. RECOMMENDED IMPROVEMENTS

### 5.1 Add Logging Instead of Print Statements

**Current**: Mixes `print()` and manual formatting
**Better**: Use standard `logging` module

```python
import logging

logger = logging.getLogger(__name__)

class TimestampSanitizer:
    def __init__(self, ..., verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
    
    def _run(self):
        if self.verbose:
            logger.debug("Sanitizing timestamp ...")
```

---

### 5.2 Add Progress/Status Tracking

```python
class TimestampSanitizer:
    def __init__(self, ...):
        self.status = {}  # Track what happened
        self.warnings = []
        self.errors = []
    
    def get_status(self) -> dict:
        """Return summary of what was done"""
        return {
            'original_shape': ...,
            'final_shape': ...,
            'rows_removed': ...,
            'duplicates_removed': ...,
            'inferred_frequency': ...,
            'warnings': self.warnings,
        }
```

---

### 5.3 Add Pre-flight Checks

```python
def _preflight_checks(self):
    """Verify data is suitable for sanitization"""
    checks = {
        'has_data': len(self.data) > 0,
        'has_index': self.data.index is not None,
        'index_not_empty': len(self.data.index) > 0,
        'index_named': self.data.index.name is not None,
    }
    
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise ValueError(f"Preflight checks failed: {failed}")
```

---

## 6. BEST PRACTICES FOR DOCSTRINGS

Add this section to class docstring:

```python
Notes
-----
**Best Practices:**

1. **Input Validation**: Always check your data BEFORE passing to TimestampSanitizer:
   
   >>> assert not df.empty, "Empty data"
   >>> assert df.index is not None, "No index"

2. **Frequency Specification**: If you know expected frequency, specify it to catch errors:
   
   >>> sanitizer = TimestampSanitizer(data, nominal_freq='30min')

3. **Verbose Mode**: Use verbose=True during development, False for production:
   
   >>> sanitizer = TimestampSanitizer(data, verbose=True)

4. **Error Handling**: Always wrap in try-except for robustness:
   
   >>> try:
   ...     sanitizer = TimestampSanitizer(data)
   ... except ValueError as e:
   ...     print(f"Timestamp validation failed: {e}")

5. **Check Inferred Frequency**: After processing, verify detected frequency:
   
   >>> if sanitizer.inferred_freq != expected_freq:
   ...     print(f"⚠️ Unexpected frequency: {sanitizer.inferred_freq}")

6. **Pipeline Reversibility**: Once regularize=True is applied, gaps are filled.
   Keep original data if you need it later.
```

---

## 7. SUMMARY OF CHANGES

| Priority | Issue | Fix |
|----------|-------|-----|
| 🔴 CRITICAL | Type hints `Series or DataFrame` | Use `Union[Series, DataFrame]` |
| 🔴 CRITICAL | Bare `except:` clause | Catch specific exceptions |
| 🔴 CRITICAL | No input validation | Add `_validate_input()` method |
| 🟠 HIGH | `inferred_freq` logic flaw | Always detect, compare with existing |
| 🟠 HIGH | Silent data loss in pipeline | Add try-except per step |
| 🟠 HIGH | Docstring missing Raises section | Add comprehensive error docs |
| 🟡 MEDIUM | Unused import | Remove `from ast import Index` |
| 🟡 MEDIUM | Magic numbers | Define constants |
| 🟢 LOW | Logging | Use `logging` module |
| 🟢 LOW | Status tracking | Add `get_status()` method |

---

## 8. ESTIMATED EFFORT

- **Type hints**: 1 hour (affects whole file)
- **Error handling**: 2 hours (test each case)
- **Docstrings**: 2 hours (write comprehensive examples)
- **Input validation**: 1 hour
- **Logging/status**: 1 hour
- **Testing**: 2-3 hours

**Total**: 9-12 hours for production-ready

---

## CONCLUSION

**The foundation is solid.** TimestampSanitizer solves a genuine problem with clear logic. With these fixes, it becomes **truly production-ready**.

**Priority fixes before shipping**:
1. Type hints (non-negotiable for IDE support)
2. Exception handling (debugging nightmare without it)
3. Input validation (fail fast, not later)
4. Docstring improvements (best practice)

**After that**: logging, status tracking, and comprehensive error messages.

