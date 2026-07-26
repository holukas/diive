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


class TestConsoleStringsAreCp1252Safe(unittest.TestCase):
    """Printed strings must survive a Windows cp1252 stdout.

    Python falls back to the locale encoding (cp1252 on a default Windows
    install) whenever stdout is a pipe or a redirect, so a printed character
    outside that range raises `UnicodeEncodeError` and kills the run. This
    happened for real: `FlagQCF.report_qcf_flags()` printed U+2550 box-drawing
    rules and crashed under `python ... | head` or `> log.txt`, while passing in
    a terminal and under pytest (both UTF-8).

    Scope and blind spots -- this check is a floor, not a guarantee:

    * It inspects string *literals* passed to the console helpers, to
      `_console.print/log/rule`, to builtin `print`, and to `raise`. A string
      assembled into a variable first and printed later is NOT seen.
    * `diive/gui/` is excluded: Qt renders Unicode natively and never touches
      stdout.
    * The Textual TUI (`detect_and_remove_tlag_tui.py`) is excluded: it paints
      its own screen buffer rather than writing to stdout.
    * Docstrings and comments are ignored on purpose -- they are never printed
      by the library itself.
    """

    #: Console helpers from diive.core.utils.console, plus builtin print and the
    #: `out()` wrapper in the hires CLI (which forwards to console.print).
    EMITTER_NAMES = frozenset({
        'print', 'info', 'detail', 'warn', 'error', 'success', 'rule', 'vspace',
        'out',
    })
    #: Methods on a Rich console object.
    EMITTER_METHODS = frozenset({'print', 'log', 'rule'})
    #: Files whose output never reaches a plain stdout stream.
    EXCLUDED = ('gui', 'detect_and_remove_tlag_tui.py')

    @staticmethod
    def _offending_chars(text):
        bad = []
        for char in text:
            try:
                char.encode('cp1252')
            except UnicodeEncodeError:
                bad.append(char)
        return bad

    @classmethod
    def _is_emitter(cls, call):
        import ast
        func = call.func
        if isinstance(func, ast.Name):
            return func.id in cls.EMITTER_NAMES
        if isinstance(func, ast.Attribute):
            return func.attr in cls.EMITTER_METHODS
        return False

    def _scan(self):
        """Yield (path, lineno, literal, bad_chars) for every offending literal."""
        import ast
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        for path in sorted((repo / 'diive').rglob('*.py')):
            rel = path.relative_to(repo).as_posix()
            if any(part in rel for part in self.EXCLUDED):
                continue
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                emitting = (isinstance(node, ast.Call) and self._is_emitter(node))
                raising = isinstance(node, ast.Raise)
                if not (emitting or raising):
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        bad = self._offending_chars(sub.value)
                        if bad:
                            yield rel, sub.lineno, sub.value, bad

    def test_no_printed_literal_breaks_cp1252(self):
        offenders = list(self._scan())
        if offenders:
            report = '\n'.join(
                f"  {rel}:{line}  {''.join(sorted(set(bad)))}  "
                f"(U+{'/U+'.join(f'{ord(c):04X}' for c in sorted(set(bad)))})"
                for rel, line, _text, bad in offenders)
            self.fail(
                f"{len(offenders)} printed string literal(s) contain characters "
                f"cp1252 cannot encode, so they crash on a redirected Windows "
                f"stdout. Use ASCII equivalents (= - | -> <= ~):\n{report}")

    def test_the_scanner_would_notice_a_bad_character(self):
        # Guard against the check silently passing because the scan is broken.
        self.assertEqual(self._offending_chars('plain ascii'), [])
        self.assertEqual(self._offending_chars('rule ═'), ['═'])
        self.assertEqual(self._offending_chars('arrow →'), ['→'])
        # Characters cp1252 *does* cover must not be flagged (e.g. the degree
        # sign, which the hires CLI prints legitimately).
        self.assertEqual(self._offending_chars('angle 12.5°'), [])


if __name__ == "__main__":
    unittest.main()
