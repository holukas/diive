"""
TEST_IMPORTS: the public namespace surface
==========================================

`import diive as dv` exposes ten domain namespaces, each re-exporting its public
symbols through an `__all__`. Nothing verified those lists: a symbol dropped from
a re-export would vanish from the public API with the whole suite still green.

Driven off `diive.__init__._LAZY_SUBMODULES` rather than a hard-coded list, so a
new namespace is covered the moment it is registered -- and the registration
itself is checked, since a namespace has to be added in four places at once
(`_LAZY_SUBMODULES`, the `TYPE_CHECKING` block, `__all__`, and the PyInstaller
spec's `hiddenimports`).

Run: pytest tests/test_imports.py -v
"""
import ast
import pathlib
import unittest
from importlib import import_module

import diive as dv
from diive import _LAZY_SUBMODULES

REPO = pathlib.Path(__file__).resolve().parent.parent


def _init_tree() -> ast.Module:
    return ast.parse((REPO / "diive" / "__init__.py").read_text(encoding="utf-8"))


class TestImports(unittest.TestCase):

    def test_imports(self):
        import diive as a
        import diive.configs as b
        import diive.core as c
        import diive.core.plotting as d
        import diive.analysis as e
        print(a, b, c, d, e)


class TestNamespaceExports(unittest.TestCase):
    """Every namespace's `__all__` must be honest: each name importable, and the
    same object whether reached through the module or through `dv`."""

    def test_every_namespace_has_an_all(self):
        for name in sorted(_LAZY_SUBMODULES):
            with self.subTest(namespace=name):
                module = import_module(f"diive.{name}")
                self.assertTrue(hasattr(module, "__all__"),
                                f"diive.{name} defines no __all__")
                self.assertTrue(module.__all__, f"diive.{name}.__all__ is empty")

    def test_every_exported_name_resolves(self):
        for name in sorted(_LAZY_SUBMODULES):
            module = import_module(f"diive.{name}")
            for symbol in module.__all__:
                with self.subTest(namespace=name, symbol=symbol):
                    self.assertTrue(
                        hasattr(module, symbol),
                        f"'{symbol}' is in diive.{name}.__all__ but not importable")

    def test_exports_are_reachable_through_dv(self):
        # The attribute-style access the docs and the GUI use (dv.outliers.Hampel).
        for name in sorted(_LAZY_SUBMODULES):
            module = import_module(f"diive.{name}")
            namespace = getattr(dv, name)
            self.assertIs(namespace, module)
            for symbol in module.__all__:
                with self.subTest(namespace=name, symbol=symbol):
                    self.assertIs(getattr(namespace, symbol), getattr(module, symbol))

    def test_no_duplicate_exports_within_a_namespace(self):
        for name in sorted(_LAZY_SUBMODULES):
            with self.subTest(namespace=name):
                names = import_module(f"diive.{name}").__all__
                duplicates = {n for n in names if names.count(n) > 1}
                self.assertEqual(set(), duplicates)

    def test_exported_names_are_public(self):
        for name in sorted(_LAZY_SUBMODULES):
            module = import_module(f"diive.{name}")
            for symbol in module.__all__:
                with self.subTest(namespace=name, symbol=symbol):
                    self.assertFalse(symbol.startswith("_"),
                                     f"'{symbol}' is private but exported")


class TestNamespaceRegistration(unittest.TestCase):
    """A namespace has to be registered in four places. Missing one fails either
    silently (a stale IDE/type-checker view) or only in the frozen GUI build,
    which is the worst place to find out.
    """

    def test_lazy_submodules_match_the_namespace_packages_on_disk(self):
        # A namespace package is a diive/<name>/__init__.py that is not one of
        # the internal support packages.
        internal = {"core", "configs", "io", "gui", "preprocessing", "fits", "main"}
        on_disk = {
            path.parent.name
            for path in (REPO / "diive").glob("*/__init__.py")
            if path.parent.name not in internal
        }
        self.assertEqual(on_disk, set(_LAZY_SUBMODULES),
                         "a namespace package exists that is not registered in "
                         "_LAZY_SUBMODULES (or vice versa)")

    def test_every_namespace_is_in_the_top_level_all(self):
        self.assertTrue(set(_LAZY_SUBMODULES) <= set(dv.__all__),
                        "a namespace is missing from diive.__all__")

    def test_every_namespace_is_in_the_type_checking_block(self):
        # Static analysers do not evaluate __getattr__, so the TYPE_CHECKING
        # block carries the real imports. It never runs, so only a parse sees it.
        imported = set()
        for node in ast.walk(_init_tree()):
            if not (isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "TYPE_CHECKING"):
                continue
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "diive":
                    imported.update(alias.name for alias in stmt.names)
        missing = set(_LAZY_SUBMODULES) - imported
        self.assertEqual(set(), missing,
                         "namespace(s) missing from the TYPE_CHECKING block in "
                         "diive/__init__.py")

    def test_every_namespace_is_in_the_pyinstaller_hiddenimports(self):
        # PyInstaller cannot follow a PEP 562 __getattr__, so an unlisted
        # namespace is simply absent from the frozen GUI.
        spec = REPO / "packaging" / "diive_gui.spec"
        if not spec.exists():
            self.skipTest("packaging/diive_gui.spec not present")
        text = spec.read_text(encoding="utf-8")
        missing = [name for name in sorted(_LAZY_SUBMODULES)
                   if f'"diive.{name}"' not in text]
        self.assertEqual([], missing,
                         "namespace(s) missing from hiddenimports in "
                         "packaging/diive_gui.spec")


if __name__ == '__main__':
    unittest.main()
