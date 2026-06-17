"""
CORE.UTILS.DOCSTRINGS: EXTRACT PER-PARAMETER HELP FROM DOCSTRINGS
================================================================

Pull a ``{parameter_name: description}`` mapping out of an object's
documentation, so callers (e.g. the GUI) can show per-parameter help (tooltips)
without duplicating text that would go stale.

Handles the two styles diive uses:

- **Google-style ``Args:`` sections** on functions/methods (``parse_google_args``).
- **Attribute docstrings** on classes / dataclasses — the string literal written
  directly under a field (``attribute_docstrings``), as ``FluxConfig`` uses.

``param_docs(obj)`` dispatches on whichever applies.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

_SECTION_NAMES = {"args", "arguments", "parameters"}
_PARAM_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _clean(text: str) -> str:
    """Collapse whitespace/newlines to a single-spaced string."""
    return " ".join(text.split())


def parse_google_args(doc: str | None) -> dict[str, str]:
    """Parse a Google-style ``Args:`` section into ``{param: description}``."""
    if not doc:
        return {}
    lines = doc.splitlines()
    start = header_indent = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1].lower() in _SECTION_NAMES:
            start, header_indent = idx + 1, len(line) - len(line.lstrip())
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    current = None
    param_indent = None
    for line in lines[start:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            break  # section ended
        if param_indent is None:
            param_indent = indent
        match = _PARAM_RE.match(line.strip())
        if match and indent == param_indent:
            current = match.group(1)
            out[current] = match.group(2)
        elif current is not None:
            out[current] += " " + line.strip()
    return {k: _clean(v) for k, v in out.items() if v.strip()}


def attribute_docstrings(cls: type) -> dict[str, str]:
    """Extract attribute docstrings (string literal under each field) from a class."""
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    classdef = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if classdef is None:
        return {}

    out: dict[str, str] = {}
    body = classdef.body
    for i, node in enumerate(body[:-1]):
        name = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
        if name is None:
            continue
        nxt = body[i + 1]
        if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)):
            out[name] = _clean(nxt.value.value)
    return out


def param_docs(obj) -> dict[str, str]:
    """Best-effort ``{param: description}`` for a class or callable.

    Classes: attribute docstrings, plus any ``Args:`` entries in the class
    docstring (attribute docstrings win). Callables: the ``Args:`` section.
    """
    if inspect.isclass(obj):
        docs = parse_google_args(inspect.getdoc(obj))
        docs.update(attribute_docstrings(obj))  # attribute docstrings take priority
        return docs
    return parse_google_args(inspect.getdoc(obj))
