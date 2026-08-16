"""
TEST_DOCSTRING_REFS: DOCSTRING CROSS-REFERENCES RESOLVE
=======================================================

Two cheap checks on what library docstrings point at. Nothing here executes a
docstring example — that is a separate, larger job (finding L85). These catch the
faults that actually occurred:

- ``examples/...`` pointers that no longer exist. A folder rename left 30 of 87
  distinct pointers dangling, so "just reference the script instead of inlining a
  sample" swaps one rotting thing for another unless the reference is checked.
- ``dv.<attr>`` names inside ``>>>`` samples that are not on the public API. Finding
  L35 was exactly this (``dv.UstarBootstrapThresholds`` for
  ``dv.flux.UstarBootstrapThresholds``), and it sat broken through two rounds of
  review because nothing executes or resolves these lines.

Docstrings are collected with ``ast``, not by importing, so ``diive.gui`` is covered
without needing the optional PySide6 dependency.

Part of the diive library: https://github.com/holukas/diive
"""

import ast
import re
import unittest
from functools import reduce
from pathlib import Path

import diive as dv

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'diive'

# 'examples/flux/lowres/flux_common.py' anywhere in prose.
EXAMPLE_PATH = re.compile(r'examples/[\w./-]+\.py')
# 'dv.flux.UstarMovingPointDetection(' -> 'flux.UstarMovingPointDetection'.
# Trailing dots/parens are stripped by the character class, so a sentence-ending
# 'dv.plotting.' does not become an attribute named ''.
DV_ATTR = re.compile(r'\bdv\.([A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_])')


def _docstrings():
    """Yield (path, lineno, docstring) for every module, class and function."""
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError) as err:  # pragma: no cover
            raise AssertionError(f"{path.relative_to(REPO_ROOT)} does not parse: {err}")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if doc:
                yield path, getattr(node, 'lineno', 1), doc


class TestDocstringExamplePointers(unittest.TestCase):
    """Every ``examples/...`` path named in a docstring must exist."""

    def test_every_example_pointer_resolves(self):
        missing = []
        seen = set()
        for path, lineno, doc in _docstrings():
            for ref in EXAMPLE_PATH.findall(doc):
                if not (REPO_ROOT / ref).is_file():
                    missing.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {ref}")
                seen.add(ref)
        self.assertGreater(len(seen), 50,
                           "expected many example pointers; the regex or the tree is wrong")
        self.assertEqual(missing, [], "docstrings point at example scripts that do not exist:\n  "
                         + "\n  ".join(missing))


class TestDocstringPublicApiNames(unittest.TestCase):
    """Every ``dv.<attr>`` used in a ``>>>`` sample must resolve on the real API.

    Only ``>>>`` lines are checked. Prose mentions a namespace loosely
    (``dv.plotting`` classes), whereas a sample is code a reader will copy.
    """

    @staticmethod
    def _resolves(dotted):
        try:
            reduce(getattr, dotted.split('.'), dv)
        except AttributeError:
            return False
        return True

    def test_every_dv_reference_in_a_sample_resolves(self):
        unresolved = []
        checked = 0
        for path, lineno, doc in _docstrings():
            for line in doc.splitlines():
                stripped = line.strip()
                if not stripped.startswith('>>>'):
                    continue
                for dotted in DV_ATTR.findall(stripped):
                    checked += 1
                    if not self._resolves(dotted):
                        unresolved.append(
                            f"{path.relative_to(REPO_ROOT)}:{lineno} -> dv.{dotted}")
        self.assertGreater(checked, 10,
                           "expected several dv.* references in samples; the regex is wrong")
        self.assertEqual(unresolved, [], "docstring samples use names that are not on the "
                         "public API:\n  " + "\n  ".join(unresolved))


if __name__ == '__main__':
    unittest.main()
