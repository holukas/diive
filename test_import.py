#!/usr/bin/env python
import traceback

try:
    from diive.core.times.times import TimestampSanitizer
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
