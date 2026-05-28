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

console = Console(highlight=False)


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
