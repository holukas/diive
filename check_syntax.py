import py_compile
import sys

try:
    py_compile.compile('diive/core/times/times.py', doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)
