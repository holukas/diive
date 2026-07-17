"""
CONSOLE
=======

Shared Rich console and structured output helpers for the diive library.

Single module-level Console instance shared across all modules.  Import the
helpers directly::

    from diive.core.utils.console import rule, info, success, warn, detail

Verbosity constants (pass as ``verbose=`` to any helper):

    VERBOSE_SILENT   = 0  — no output
    VERBOSE_ERROR    = 1  — errors and warnings only
    VERBOSE_PROGRESS = 2  — section headers and key results  (default)
    VERBOSE_DEBUG    = 3  — all detail lines
"""

from rich.console import Console

try:  # reuse Rich's own check so diive renders the way Rich decides to
    from rich.console import _is_jupyter
except ImportError:  # pragma: no cover - private API renamed/removed
    def _is_jupyter() -> bool:
        return False

VERBOSE_SILENT = 0
VERBOSE_ERROR = 1
VERBOSE_PROGRESS = 2
VERBOSE_DEBUG = 3


class _TeeConsole(Console):
    """Rich Console that also forwards output to registered mirror consoles.

    Lets an external consumer (e.g. the desktop GUI) receive a copy of all
    library console output without the library depending on it: register a
    mirror with :func:`add_console_sink` and it gets every ``print`` / ``log``
    / ``rule`` call. Mirror errors are swallowed so a failing sink never breaks
    library output.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Create the tee console with an empty list of mirror sinks."""
        super().__init__(*args, **kwargs)
        self._mirrors: list = []

    def add_mirror(self, mirror) -> None:
        """Register a mirror sink to receive copies of console output."""
        if mirror not in self._mirrors:
            self._mirrors.append(mirror)

    def remove_mirror(self, mirror) -> None:
        """Unregister a previously added mirror sink (no-op if absent)."""
        if mirror in self._mirrors:
            self._mirrors.remove(mirror)

    def _forward(self, method: str, args, kwargs) -> None:
        for mirror in list(self._mirrors):
            try:
                getattr(mirror, method)(*args, **kwargs)
            except Exception:
                pass  # a broken sink must not break library output

    def print(self, *args, **kwargs) -> None:
        """Print and forward the call to every registered mirror."""
        super().print(*args, **kwargs)
        self._forward("print", args, kwargs)

    def log(self, *args, **kwargs) -> None:
        """Log and forward the call to every registered mirror."""
        # `log` writes directly (it does not route through `self.print`), so it
        # must forward to mirrors itself.
        super().log(*args, **kwargs)
        self._forward("log", args, kwargs)

    # NOTE: no `rule` override. Rich's `Console.rule()` renders via `self.print`,
    # which is already overridden to forward — overriding `rule` too would
    # forward each rule to mirrors twice (once as the Rule renderable via print,
    # once again here).


#: Width the shared console renders at inside Jupyter. Terminals auto-detect
#: their width; Jupyter otherwise falls back to a fixed 80 columns, which wraps
#: the wider report tables.
_JUPYTER_CONSOLE_WIDTH = 100


def _build_console() -> _TeeConsole:
    """Build the shared console configured for the current environment.

    Rich freezes ``is_jupyter`` at construction, so this is decided once for the
    session (in a notebook, diive is imported in-kernel, so detection is
    correct). In Jupyter, pin the Jupyter renderer and a wider width so the
    report tables do not wrap at 80 columns; in a terminal, let Rich
    auto-detect.
    """
    if _is_jupyter():
        return _TeeConsole(highlight=False, force_jupyter=True,
                           width=_JUPYTER_CONSOLE_WIDTH)
    return _TeeConsole(highlight=False)


console = _build_console()

#: Rule line style. Rich's default rule renders bright green (#00ff00), which is
#: illegible on a white notebook background; use a neutral grey in Jupyter and
#: keep the terminal default (None) elsewhere.
_rule_line_style: str | None = "grey50" if _is_jupyter() else None


def refresh_console() -> None:
    """Rebuild the shared console for the current environment, keeping mirrors.

    ``is_jupyter`` is fixed when a Rich ``Console`` is built, so a diive import
    that ran before the interactive frontend was ready would keep terminal
    settings. Call this to re-detect. The structured helpers below always use
    the current console; modules that imported ``console`` by name keep the
    previous object.
    """
    global console, _rule_line_style
    mirrors = list(getattr(console, "_mirrors", []))
    console = _build_console()
    _rule_line_style = "grey50" if _is_jupyter() else None
    for mirror in mirrors:
        console.add_mirror(mirror)


def add_console_sink(mirror) -> None:
    """Register a mirror console to receive a copy of all library output.

    Args:
        mirror: Any object with ``print`` / ``log`` / ``rule`` methods (e.g. a
            Rich ``Console`` writing to a GUI panel).
    """
    console.add_mirror(mirror)


def remove_console_sink(mirror) -> None:
    """Stop forwarding library output to a previously registered mirror."""
    console.remove_mirror(mirror)


def _vlevel(verbose: int | bool) -> int:
    """Normalize bool or int to an int verbosity level."""
    if isinstance(verbose, bool):
        return VERBOSE_PROGRESS if verbose else VERBOSE_SILENT
    return int(verbose)


def rule(title: str = '', *, verbose: int | bool = VERBOSE_PROGRESS,
         min_level: int = VERBOSE_PROGRESS) -> None:
    """Print a horizontal rule with an optional centred title."""
    if _vlevel(verbose) >= min_level:
        styled = f"[bold blue]{title}[/bold blue]" if title else ""
        if _rule_line_style is not None:
            console.rule(styled, style=_rule_line_style)
        else:
            console.rule(styled)


def info(msg: str, *, verbose: int | bool = VERBOSE_PROGRESS,
         min_level: int = VERBOSE_PROGRESS) -> None:
    """Print an informational line (cyan bullet)."""
    if _vlevel(verbose) >= min_level:
        console.print(f"  [cyan]>[/cyan] {msg}")


def success(msg: str, *, verbose: int | bool = VERBOSE_PROGRESS,
            min_level: int = VERBOSE_PROGRESS) -> None:
    """Print a success line (green check)."""
    if _vlevel(verbose) >= min_level:
        console.print(f"  [green]v[/green] {msg}")


def warn(msg: str, *, verbose: int | bool = VERBOSE_PROGRESS,
         min_level: int = VERBOSE_ERROR) -> None:
    """Print a warning line (yellow exclamation)."""
    if _vlevel(verbose) >= min_level:
        console.print(f"  [yellow]![/yellow] {msg}")


def error(msg: str, *, verbose: int | bool = VERBOSE_PROGRESS,
          min_level: int = VERBOSE_ERROR) -> None:
    """Print an error line (bold red cross)."""
    if _vlevel(verbose) >= min_level:
        console.print(f"  [bold red]x[/bold red] {msg}")


def detail(msg: str, *, verbose: int | bool = VERBOSE_PROGRESS,
           min_level: int = VERBOSE_DEBUG) -> None:
    """Print a dim detail line (only at VERBOSE_DEBUG level by default)."""
    if _vlevel(verbose) >= min_level:
        console.print(f"  [dim]{msg}[/dim]")


def vspace(text: str = "", *, verbose: int | bool = VERBOSE_PROGRESS,
           min_level: int = VERBOSE_PROGRESS) -> None:
    """Print a blank separator line in a terminal / GUI log, nothing in Jupyter.

    A blank line cleanly separates report phases in a terminal, but in Jupyter
    every ``console.print`` is a separate display block with its own margin, so
    an empty print becomes a full empty block of dead vertical space. Suppress
    it there to keep notebook output dense. ``text`` lets a caller reproduce a
    wider terminal gap (e.g. ``"\\n"``); it is still dropped in Jupyter.
    """
    if _vlevel(verbose) >= min_level and not _is_jupyter():
        console.print(text)
