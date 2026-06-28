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


console = _TeeConsole(highlight=False)


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
