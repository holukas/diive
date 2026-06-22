"""
PRINTS
======

Legacy console output helpers.  Delegates to the shared Rich console in
``diive.core.utils.console``.
"""
from functools import wraps

from diive.core.utils.console import console, info


class ConsoleOutputDecorator:
    """Wrap a function with a start/done console message."""

    def __init__(self, spacing: bool = True):
        """Configure the decorator (*spacing* adds blank lines around the messages)."""
        self.spacing = spacing

    def __call__(self, func):
        """Wrap *func* so it prints a start message, runs, then a done message."""
        @wraps(func)
        def my_logic(*args, **kwargs):
            printid = PrintID(id=func.__name__, spacing=self.spacing)
            results = func(*args, **kwargs)
            printid.done()
            return results

        return my_logic


class PrintID:
    """Print a labelled start message for an operation and let callers print follow-ups."""

    def __init__(self, id: str, spacing: bool = True):
        """Store the label *id* and print the start message."""
        self.id = id
        self.spacing = spacing
        self.section()

    def section(self):
        """Print the 'running <id> ...' start message."""
        info(f"running {self.id} ...")

    def str(self, txt: str):
        """Print *txt* prefixed with the operation label."""
        console.print(f"[{self.id}]  {txt}")

    def done(self):
        """Mark the operation as done (currently a no-op)."""
        pass
