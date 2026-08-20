"""
GUI.WIDGETS.STATE_UTILS: GENERIC WIDGET STATE (SAVE / RESTORE)
=============================================================

Tiny helpers so each tab's :meth:`DiiveTab.save_state` / ``restore_state`` can
serialize its standard Qt controls without bespoke per-widget code. A tab maps a
stable key to each control it wants persisted; these read/write the value by
widget type.

GUI-only presentation glue.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
)


def widget_value(w):
    """Serializable value of a supported control, or None if unsupported."""
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QLineEdit):
        return w.text()
    if isinstance(w, QPlainTextEdit):
        return w.toPlainText()
    return None


def set_widget_value(w, value) -> bool:
    """Apply a value produced by :func:`widget_value` back onto a control.

    Returns ``False`` when the value could not be applied: a non-editable
    :class:`QComboBox` whose saved entry is no longer among its items keeps a
    *different* selection, which the caller must be able to report. Every other
    case (including ``value is None``) returns ``True``.
    """
    if value is None:
        return True
    if isinstance(w, QCheckBox):
        w.setChecked(bool(value))
    elif isinstance(w, QSpinBox):
        w.setValue(int(value))
    elif isinstance(w, QDoubleSpinBox):
        w.setValue(float(value))
    elif isinstance(w, QComboBox):
        if w.isEditable():
            w.setCurrentText(str(value))  # preserve typed values not in the list
        else:
            i = w.findText(str(value))
            if i < 0:
                return False
            w.setCurrentIndex(i)
    elif isinstance(w, QLineEdit):
        w.setText(str(value))
    elif isinstance(w, QPlainTextEdit):
        w.setPlainText(str(value))
    return True


def save_controls(controls: dict) -> dict:
    """``{key: serializable value}`` for a ``{key: widget}`` mapping."""
    return {k: widget_value(w) for k, w in controls.items()}


def restore_controls(controls: dict, values: dict) -> list[str]:
    """Apply a saved ``{key: value}`` dict onto a ``{key: widget}`` mapping.

    Returns the keys whose saved value could not be applied (see
    :func:`set_widget_value`) — those controls now hold something *other* than
    what was saved, e.g. a combo whose column was renamed since. Callers that
    don't surface it can ignore the return value; :func:`unrestored_message`
    turns it into a status line.
    """
    unrestored = []
    for key, w in controls.items():
        if key in (values or {}) and not set_widget_value(w, values[key]):
            unrestored.append(key)
    return unrestored


def unrestored_message(keys: list, labels: dict | None = None) -> str:
    """Status-line wording for the keys :func:`restore_controls` could not apply."""
    names = ", ".join(str((labels or {}).get(k, k)) for k in keys)
    return (f"Saved selection no longer available for: {names}. Those controls kept "
            "their current value — check them before running.")
