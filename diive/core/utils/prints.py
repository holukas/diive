"""
PRINTS
======

Functions for printing info.

"""
from functools import wraps


class ConsoleOutputDecorator:
    """
    Wrapper for printing info before and after execution of *func*

    Kudos:
    (1) https://docs.python.org/2/library/functools.html#functools.wraps
    (2) https://stackoverflow.com/questions/10176226/how-do-i-pass-extra-arguments-to-a-python-decorator
    """

    def __init__(self, spacing: bool = True):
        """
        Parameters for how to execute the decorator

        Args:
            spacing: If *True*, insert two empty rows before output
        """
        self.spacing = spacing

    def __call__(self, func):
        """
        Call to decorated function *func*

        Args:
            func: Function that is executed

        Returns:
            Function results
        """

        @wraps(func)
        # @wraps creates yet another wrapper around a decorated function that restores its type as a function
        # while preserving the docstring.
        # https://stackoverflow.com/questions/72492374/how-to-make-python-help-function-work-well-with-decorators
        def my_logic(*args, **kwargs):
            # Whatever logic the decorator is supposed to implement goes in here
            id = func.__name__
            printid = PrintID(id=id, spacing=self.spacing)
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
        if self.spacing:
            print("")
            print("")
        # self.str(txt=f"{'=' * 40}")
        self.str(txt=f"{self.id}")
        # self.str(txt=f"{'=' * 40}")

    def str(self, txt: str):
        # print(f"{txt}")
        print(f"[{self.id}]  {txt}")

    def done(self):
        pass
        # print(f"[{self.id}]  Done.")
        # print(f"[{self.id}]  {'_' * 40}")
