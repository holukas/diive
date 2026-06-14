"""
GUI.WIDGETS.CORRECTIONS_PANEL: METEO CORRECTIONS PICKER
=======================================================

A checkable list of high-resolution corrections, filtered to the ones that make
sense for the selected *measurement* (e.g. the radiation zero-offset correction
only appears for shortwave radiation / PPFD). Each enabled correction with its
parameters is returned as a ``{"key", "kwargs"}`` dict — the exact shape the
library's :func:`diive.preprocessing.corrections.apply.apply_corrections`
consumes.

GUI-only: which corrections apply to which measurement is library domain
knowledge (:func:`diive.qaqc.corrections_for_measurement`); this widget only
renders the rows, collects parameters, and parses the small text inputs (date
ranges, value lists) into the kwargs the library expects.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from diive.qaqc import (
    CORRECTIONS,
    correction_spec,
    corrections_for_measurement,
)
from diive.preprocessing.qaqc.measurements import (
    CORR_RADIATION_ZERO_OFFSET,
    CORR_RELATIVEHUMIDITY_OFFSET,
    CORR_SETTO_MAX,
    CORR_SETTO_MIN,
    CORR_SETTO_VALUE,
    CORR_SET_EXACT_TO_MISSING,
)


def _parse_dates(text: str) -> list:
    """Parse a date-range text field into the ``dates`` list ``setto_value``
    expects. Entries are separated by ``;``; a range uses ``..`` (e.g.
    ``2022-04-01..2022-04-05``), a single timestamp is given on its own
    (``2022-04-01``). Whitespace and empty entries are ignored."""
    out: list = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            out.append([a.strip(), b.strip()])
        else:
            out.append(part)
    return out


def _parse_values(text: str) -> list:
    """Parse a comma-separated list of numbers (e.g. ``0, -9999``). Tokens that
    are not numbers are skipped."""
    out: list = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


class _CorrectionRow(QWidget):
    """One correction: enable checkbox + label + inline parameter widgets."""

    changed = Signal()

    def __init__(self, key: str, coords_available: bool,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        spec = correction_spec(key)
        h = QHBoxLayout(self)
        h.setContentsMargins(2, 2, 2, 2)

        self.enable = QCheckBox(spec.label if spec else key)
        self.enable.setToolTip(spec.description if spec else "")
        self.enable.toggled.connect(lambda *_: self.changed.emit())
        h.addWidget(self.enable)

        # Coordinate-dependent corrections are disabled (with a hint) when the
        # site is not configured — running them with default 0/0 coords would
        # produce a meaningless day/night split.
        if spec is not None and spec.needs_coords and not coords_available:
            self.enable.setEnabled(False)
            self.enable.setToolTip(
                spec.description + "\n\nConfigure the site (Project settings) "
                "to enable this correction.")

        self._param_widgets: list[QWidget] = []
        self._build_params(key, h)

        h.addStretch(1)

    def _build_params(self, key: str, h: QHBoxLayout) -> None:
        if key in (CORR_SETTO_MAX, CORR_SETTO_MIN):
            h.addWidget(QLabel("threshold:"))
            self.threshold = QDoubleSpinBox()
            self.threshold.setRange(-1e9, 1e9)
            self.threshold.setDecimals(3)
            self.threshold.setValue(30.0 if key == CORR_SETTO_MAX else -5.0)
            self.threshold.valueChanged.connect(lambda *_: self.changed.emit())
            h.addWidget(self.threshold)
            self._param_widgets.append(self.threshold)
        elif key == CORR_SETTO_VALUE:
            h.addWidget(QLabel("dates:"))
            self.dates = QLineEdit()
            self.dates.setPlaceholderText("2022-04-01..2022-04-05; 2022-09-05..2022-09-07")
            self.dates.setMinimumWidth(260)
            self.dates.setToolTip(
                "Date ranges separated by ';'. A range uses '..' "
                "(start..end, both inclusive); a single timestamp stands alone.")
            self.dates.textChanged.connect(lambda *_: self.changed.emit())
            h.addWidget(self.dates)
            h.addWidget(QLabel("value:"))
            self.value = QDoubleSpinBox()
            self.value.setRange(-1e9, 1e9)
            self.value.setDecimals(3)
            self.value.valueChanged.connect(lambda *_: self.changed.emit())
            h.addWidget(self.value)
            self._param_widgets += [self.dates, self.value]
        elif key == CORR_SET_EXACT_TO_MISSING:
            h.addWidget(QLabel("values:"))
            self.values = QLineEdit()
            self.values.setPlaceholderText("0, -9999")
            self.values.setMinimumWidth(160)
            self.values.setToolTip("Comma-separated values set to missing (NaN).")
            self.values.textChanged.connect(lambda *_: self.changed.emit())
            h.addWidget(self.values)
            self._param_widgets.append(self.values)

    def is_enabled(self) -> bool:
        return self.enable.isChecked() and self.enable.isEnabled()

    def kwargs(self) -> dict:
        if self.key in (CORR_SETTO_MAX, CORR_SETTO_MIN):
            return {"threshold": self.threshold.value()}
        if self.key == CORR_SETTO_VALUE:
            return {"dates": _parse_dates(self.dates.text()),
                    "value": self.value.value()}
        if self.key == CORR_SET_EXACT_TO_MISSING:
            return {"values": _parse_values(self.values.text())}
        return {}

    def has_valid_params(self) -> bool:
        """Whether the enabled correction has the inputs it needs to do anything."""
        kw = self.kwargs()
        if self.key == CORR_SETTO_VALUE:
            return bool(kw.get("dates"))
        if self.key == CORR_SET_EXACT_TO_MISSING:
            return bool(kw.get("values"))
        return True

    # --- state (raw control values, for project save/restore) ---
    def get_state(self) -> dict:
        st: dict = {"enabled": self.enable.isChecked()}
        if self.key in (CORR_SETTO_MAX, CORR_SETTO_MIN):
            st["threshold"] = self.threshold.value()
        elif self.key == CORR_SETTO_VALUE:
            st["dates"] = self.dates.text()
            st["value"] = self.value.value()
        elif self.key == CORR_SET_EXACT_TO_MISSING:
            st["values"] = self.values.text()
        return st

    def set_state(self, st: dict) -> None:
        self.enable.setChecked(bool(st.get("enabled", False)))
        if self.key in (CORR_SETTO_MAX, CORR_SETTO_MIN) and "threshold" in st:
            self.threshold.setValue(float(st["threshold"]))
        elif self.key == CORR_SETTO_VALUE:
            self.dates.setText(str(st.get("dates", "")))
            if "value" in st:
                self.value.setValue(float(st["value"]))
        elif self.key == CORR_SET_EXACT_TO_MISSING:
            self.values.setText(str(st.get("values", "")))


class CorrectionsPanel(QWidget):
    """Checkable correction rows, filtered to the current measurement.

    Emits ``changed`` whenever a row is toggled or a parameter edited. The host
    reads :meth:`corrections` to get the enabled ``{"key", "kwargs"}`` chain in
    display order.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._measurement: str | None = None
        self._coords_available = False
        self._rows: dict[str, _CorrectionRow] = {}
        #: Last-known per-key control state, so toggling the measurement (which
        #: rebuilds the rows) does not lose a user's earlier edits.
        self._saved: dict[str, dict] = {}
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(2)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #6B7780;")
        self._v.addWidget(self._hint)
        self._rebuild()

    def set_coords_available(self, available: bool) -> None:
        if available != self._coords_available:
            self._coords_available = available
            self._rebuild()

    def set_measurement(self, code: str | None) -> None:
        if code != self._measurement:
            self._measurement = code
            self._rebuild()

    def measurement(self) -> str | None:
        return self._measurement

    def _rebuild(self) -> None:
        # Snapshot current edits before tearing the rows down.
        for key, row in self._rows.items():
            self._saved[key] = row.get_state()
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows = {}

        keys = corrections_for_measurement(self._measurement)
        if self._measurement is None:
            self._hint.setText(
                "Generic corrections (any measurement). Pick a measurement to "
                "surface measurement-specific corrections.")
        else:
            self._hint.setText(
                "Corrections applicable to this measurement, applied in order "
                "to the QCF-filtered series.")
        for key in keys:
            row = _CorrectionRow(key, self._coords_available, self)
            if key in self._saved:
                row.set_state(self._saved[key])
            row.changed.connect(self.changed.emit)
            self._v.addWidget(row)
            self._rows[key] = row
        self.changed.emit()

    def corrections(self) -> list[dict]:
        """Enabled corrections (with parameters) in display order. Enabled rows
        with empty required inputs are skipped (they would be no-ops)."""
        out = []
        for key in corrections_for_measurement(self._measurement):
            row = self._rows.get(key)
            if row is not None and row.is_enabled() and row.has_valid_params():
                out.append({"key": key, "kwargs": row.kwargs()})
        return out

    # --- state ---
    def get_state(self) -> dict:
        # Merge live rows over the saved snapshot so off-measurement edits persist.
        state = dict(self._saved)
        for key, row in self._rows.items():
            state[key] = row.get_state()
        return {"measurement": self._measurement, "rows": state}

    def set_state(self, st: dict) -> None:
        self._saved = dict(st.get("rows") or {})
        self.set_measurement(st.get("measurement"))
        # set_measurement rebuilds and applies _saved; if the measurement was
        # unchanged it won't rebuild, so apply directly.
        for key, row in self._rows.items():
            if key in self._saved:
                row.set_state(self._saved[key])
