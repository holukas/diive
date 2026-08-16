"""
GUI.WIDGETS.WEAK_SLOT: SIGNAL TARGET THAT DOES NOT OWN ITS OBJECT
=================================================================

PySide6 keeps a **bound method** connected to a signal alive only weakly, so
``btn.clicked.connect(self._run)`` cannot keep ``self`` from being collected. A
**lambda** is different: the connection owns it, the lambda owns its closure, and
the closure owns ``self``. Since the connection lives on the C++ side, Python's
collector cannot see that cycle and the object leaks -- together with every
widget it holds (a `DiiveTab` is ~500 of them).

``weak_slot`` covers the case a bare bound method cannot: a slot that needs
arguments the signal does not carry (``lambda: self._run_level(idx)``). It binds
those arguments and keeps the method's object weakly, so it is safe to hand to
``connect``. Once the object is gone the slot is a no-op.

Signal arguments are truncated to what the target still accepts, the same way Qt
truncates for a plain bound method, so ``weak_slot(self._fail, "buckets")``
against a ``failed(str)`` signal calls ``self._fail("buckets", err)`` while
``weak_slot(self._run_level, 2)`` against ``clicked(bool)`` calls
``self._run_level(2)``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import inspect
import weakref
from typing import Callable

# A *args target takes whatever the signal offers.
_UNLIMITED = 1 << 16


def _positional_capacity(method: Callable) -> int:
    """How many positional arguments `method` accepts (self already bound)."""
    n = 0
    for p in inspect.signature(method).parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            return _UNLIMITED
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
    return n


def weak_slot(method: Callable, *args) -> Callable:
    """Return a connectable slot calling ``method(*args, …)`` without owning it.

    `method` must be a bound method; only a weak reference to its object is
    kept. Extra signal arguments are appended up to the target's arity.
    """
    ref = weakref.WeakMethod(method)
    n_from_signal = max(0, _positional_capacity(method) - len(args))

    def _slot(*signal_args):
        target = ref()
        if target is None:
            return None
        return target(*args, *signal_args[:n_from_signal])

    return _slot
