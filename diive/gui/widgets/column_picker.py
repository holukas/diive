"""
GUI.WIDGETS.COLUMN_PICKER: SPEC-DRIVEN VARIABLE COMBO GROUP
==========================================================

A group of variable-selection combo boxes, one per declared input, each with a
green ✓ / red ✗ availability marker and auto-seeded from the dataset's column
names. This is the shared version of the "combo + marker, auto-picked by needle"
block that the partitioning and uncertainty tabs each hand-rolled.

Each input is a spec dict:

  * ``key``      — identifier returned by :meth:`picks`,
  * ``label``    — form-row label,
  * ``needle``   — substring to match (``str`` or a list of alternatives tried
    in order), passed to :func:`diive.variables.auto_pick_column`,
  * ``prefer``   — optional substring ranked first when present,
  * ``avoid``    — optional substring that disqualifies a column,
  * ``optional`` — when true the combo gains a leading ``"(none)"`` and stays on
    it unless the distinctive ``prefer`` token actually appears,
  * ``tip``      — optional combo tooltip.

Presentation + name-based seeding only; the name matching itself is library
domain logic (``auto_pick_column``).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from diive.variables import auto_pick_column

#: Sentinel shown for an unselected optional input.
NONE_ITEM = "(none)"


class ColumnPicker(QGroupBox):
    """A group box of per-input variable combos with availability markers.

    Build with the input ``specs`` (see the module docstring), call
    :meth:`seed` whenever the dataset changes (fills the combos + auto-picks),
    and read selections back with :meth:`picks`. :meth:`combos` exposes the raw
    combo boxes for project save/restore. ``changed`` fires on any edit.
    """

    changed = Signal()

    def __init__(self, specs: list[dict], title: str = "Input columns",
                 parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._specs = specs
        self._combos: dict[str, QComboBox] = {}
        self._avail: dict[str, QLabel] = {}
        self._cols: set[str] = set()
        form = QFormLayout(self)
        for spec in specs:
            combo = QComboBox()
            if spec.get("tip"):
                combo.setToolTip(spec["tip"])
            avail = QLabel("")
            avail.setStyleSheet("font-size: 11px;")
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(combo, stretch=1)
            rh.addWidget(avail)
            form.addRow(spec["label"], row)
            self._combos[spec["key"]] = combo
            self._avail[spec["key"]] = avail
            combo.currentTextChanged.connect(self._on_changed)

    def _on_changed(self, *_) -> None:
        self.refresh_availability()
        self.changed.emit()

    def seed(self, columns) -> None:
        """Repopulate every combo from ``columns`` and auto-pick a default.

        Keeps the current selection if it's still present; otherwise guesses via
        :func:`auto_pick_column` (trying each ``needle`` alternative in order).
        """
        cols = [str(c) for c in columns]
        self._cols = set(cols)
        for spec in self._specs:
            combo = self._combos[spec["key"]]
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if spec.get("optional"):
                combo.addItem(NONE_ITEM)
            combo.addItems(cols)
            if cur and cur in cols:
                combo.setCurrentText(cur)
            else:
                needles = spec["needle"]
                if isinstance(needles, str):
                    needles = [needles]
                prefer = spec.get("prefer")
                for needle in needles:  # try each naming alternative in order
                    guess = auto_pick_column(cols, needle, prefer=prefer,
                                             avoid=spec.get("avoid"))
                    # An optional input stays "(none)" unless its distinctive token
                    # (prefer) actually appears — don't guess a wrong column for it.
                    if guess and spec.get("optional") and prefer \
                            and prefer not in guess.upper():
                        guess = ""
                    if guess:
                        combo.setCurrentText(guess)
                        break
            combo.blockSignals(False)
        self.refresh_availability()

    def refresh_availability(self, *_) -> None:
        """Update each combo's ✓/✗ marker against the last-seeded column set."""
        for key, combo in self._combos.items():
            txt = combo.currentText()
            if txt in ("", NONE_ITEM):
                self._avail[key].setText("")
            elif txt in self._cols:
                self._avail[key].setText("✓")
                self._avail[key].setStyleSheet("color: #2E7D32; font-size: 11px;")
            else:
                self._avail[key].setText("✗")
                self._avail[key].setStyleSheet("color: #C62828; font-size: 11px;")

    def picks(self) -> dict[str, str]:
        """Current selection per input key (``"(none)"`` for an empty optional)."""
        return {k: c.currentText() for k, c in self._combos.items()}

    def combos(self) -> dict[str, QComboBox]:
        """The raw combo boxes, keyed by input key (for project save/restore)."""
        return dict(self._combos)
