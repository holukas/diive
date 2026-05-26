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
        self.spacing = spacing

    def __call__(self, func):
        @wraps(func)
        def my_logic(*args, **kwargs):
            printid = PrintID(id=func.__name__, spacing=self.spacing)
            results = func(*args, **kwargs)
            printid.done()
            return results

        return my_logic


class PrintID:

    def __init__(self, id: str, spacing: bool = True):
        self.id = id
        self.spacing = spacing
        self.section()

    def section(self):
        info(f"running {self.id} ...")

    def str(self, txt: str):
        console.print(f"[{self.id}]  {txt}")

    def done(self):
        pass
