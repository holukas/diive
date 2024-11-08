import unittest


class TestImports(unittest.TestCase):

    def test_imports(self):
        import diive as a
        import diive.configs as b
        import diive.core as c
        import diive.core.plotting as d
        import diive.pkgs as e
        print(a, b, c, d, e)
