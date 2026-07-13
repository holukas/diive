"""
Tests for the shared Rich console (diive/core/utils/console.py).

Focus: the console must be built environment-aware so it renders correctly in
Jupyter (wider width, legible rule) without changing terminal behaviour.
"""
import builtins
import contextlib
import unittest

import diive.core.utils.console as con


@contextlib.contextmanager
def _fake_jupyter():
    """Make Rich's `_is_jupyter()` report a notebook kernel for the block.

    Rich checks a bare `get_ipython` whose shell class is `ZMQInteractiveShell`
    (a real notebook / qtconsole), so inject one into builtins.
    """
    class ZMQInteractiveShell:
        pass

    had = hasattr(builtins, "get_ipython")
    prev = getattr(builtins, "get_ipython", None)
    builtins.get_ipython = lambda: ZMQInteractiveShell()
    try:
        yield
    finally:
        if had:
            builtins.get_ipython = prev
        else:
            del builtins.get_ipython


class TestConsoleEnvironment(unittest.TestCase):

    def test_terminal_build_is_not_jupyter(self):
        c = con._build_console()
        self.assertFalse(c.is_jupyter)

    def test_terminal_rule_style_is_default(self):
        # In a terminal the rule keeps Rich's default line style (None here).
        with _fake_jupyter():
            pass  # ensure the fixture leaves no residue
        self.assertIsNone(con._rule_line_style)

    def test_jupyter_build_pins_renderer_and_width(self):
        with _fake_jupyter():
            c = con._build_console()
        self.assertTrue(c.is_jupyter)
        self.assertEqual(c.width, con._JUPYTER_CONSOLE_WIDTH)

    def test_jupyter_width_is_wider_than_default_80(self):
        # The bug: Jupyter otherwise falls back to 80 columns and wraps tables.
        self.assertGreater(con._JUPYTER_CONSOLE_WIDTH, 80)


class TestRefreshConsole(unittest.TestCase):

    def tearDown(self):
        # Always restore a terminal console for the rest of the suite.
        con.refresh_console()

    def test_refresh_switches_rule_style_for_jupyter(self):
        with _fake_jupyter():
            con.refresh_console()
            self.assertTrue(con.console.is_jupyter)
            self.assertIsNotNone(con._rule_line_style)
        con.refresh_console()
        self.assertFalse(con.console.is_jupyter)
        self.assertIsNone(con._rule_line_style)

    def test_refresh_preserves_registered_mirrors(self):
        from rich.console import Console
        mirror = Console()
        con.add_console_sink(mirror)
        con.refresh_console()
        self.assertIn(mirror, con.console._mirrors)
        con.remove_console_sink(mirror)


class TestJupyterRuleIsLegible(unittest.TestCase):
    """Regression guard for the reported bug: the rule line rendered as bright
    green (#00ff00), illegible on a white notebook background."""

    def test_rule_line_is_not_bright_green_in_jupyter(self):
        import rich.jupyter as rj

        grabbed = []
        orig_init = rj.JupyterRenderable.__init__

        def capture_init(self, html, text):
            grabbed.append(html)
            orig_init(self, html, text)

        with _fake_jupyter():
            con.refresh_console()
            rj.JupyterRenderable.__init__ = capture_init
            try:
                con.rule("GAP-FILLING")
            finally:
                rj.JupyterRenderable.__init__ = orig_init
        con.refresh_console()

        self.assertTrue(grabbed)
        rule_html = grabbed[0]
        self.assertNotIn("#00ff00", rule_html.lower())


if __name__ == "__main__":
    unittest.main()
