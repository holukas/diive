"""
GUI.WIDGETS.STEPWISE_CARDS: METHOD CARDS FOR THE STEPWISE OUTLIER CHAIN
======================================================================

The stepwise screening tab builds its outlier chain as a row of draggable,
editable **method cards** instead of a flat list. Each card shows one step
(method + a compact parameter summary + how many points it removed once run) and
carries inline controls: reorder (◀ ▶), edit (re-open the param form seeded with
the step's current kwargs), and delete. A trailing dashed *＋ Add step* ghost
card opens the same editor with no preset.

GUI-only: these are display widgets and a small editor dialog around the shared
``stepwise_method_params`` registry. The step shape (``{"method", "kwargs"}``)
and all detection are the library's.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from diive.gui import icons, theme
from diive.gui.widgets.stepwise_method_params import (
    STEP_METHOD_BY_KEY,
    method_labels,
)
from PySide6.QtWidgets import QComboBox

_CARD_W = 210
_C_MUTED = "#6B7780"


def kwargs_summary(kwargs: dict, max_items: int | None = None) -> str:
    """Compact ``k=v, …`` one-liner for a step's kwargs (empty -> 'defaults').

    ``max_items`` caps how many entries are shown, appending an ellipsis when
    the rest are dropped — keeps the card summary short; the full string is used
    for the card's tooltip.
    """
    if not kwargs:
        return "defaults"
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if not v:
                continue  # drop the noisy False flags
            parts.append(k)
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    if not parts:
        return "defaults"
    if max_items is not None and len(parts) > max_items:
        return ", ".join(parts[:max_items]) + ", …"
    return ", ".join(parts)


class StepEditorDialog(QDialog):
    """Pick a method and set its parameters; returns a ``{"method", "kwargs"}`` step.

    Used both to add a new step (no preset) and to edit an existing one (the
    method combo and the param widget are seeded from the step).
    """

    def __init__(self, parent: QWidget | None = None, step: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit step" if step else "Add step")
        self.setModal(True)
        self.setMinimumWidth(340)
        v = QVBoxLayout(self)

        self.method = QComboBox()
        for key, label in method_labels():
            self.method.addItem(label, key)
        v.addWidget(self.method)

        self._param_box = QVBoxLayout()
        self._param_widget = None
        v.addLayout(self._param_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

        # Seed the method index *before* wiring the change handler so the param
        # widget is built exactly once (a connected setCurrentIndex would build
        # it again and leave a ghost form behind).
        if step:
            idx = self.method.findData(step["method"])
            if idx >= 0:
                self.method.setCurrentIndex(idx)
        self._on_method_changed()
        if step:
            self._param_widget.load(step.get("kwargs", {}))
        self.method.currentIndexChanged.connect(self._on_method_changed)

    def _on_method_changed(self, *_) -> None:
        key = self.method.currentData()
        if self._param_widget is not None:
            self._param_box.removeWidget(self._param_widget)
            # Detach immediately: deleteLater alone keeps the old form painted on
            # top until the event loop runs, which a modal exec() defers.
            self._param_widget.setParent(None)
            self._param_widget.deleteLater()
        self._param_widget = STEP_METHOD_BY_KEY[key]()
        self._param_box.addWidget(self._param_widget)

    def step(self) -> dict:
        return self._param_widget.step()

    @classmethod
    def get_step(cls, parent: QWidget | None, step: dict | None = None) -> dict | None:
        """Open the editor; return the resulting step, or ``None`` if cancelled."""
        dlg = cls(parent, step)
        if dlg.exec() == QDialog.Accepted:
            return dlg.step()
        return None


class _CardSignals(QObject):
    clicked = Signal()
    edit = Signal()
    delete = Signal()
    move_left = Signal()
    move_right = Signal()
    toggle = Signal(bool)


class StepCard(QFrame):
    """One step in the chain: method + param summary + removed badge + controls.

    A dumb display widget — the tab owns the ``steps`` list and rebuilds cards
    from it; the card only reports user intent (click/edit/delete/move) via
    signals.
    """

    def __init__(self, index: int, step: dict, removed: int | None = None,
                 selected: bool = False, enabled: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sig = _CardSignals()
        self.clicked = self._sig.clicked
        self.edit = self._sig.edit
        self.delete = self._sig.delete
        self.move_left = self._sig.move_left
        self.move_right = self._sig.move_right
        self.toggle = self._sig.toggle
        self._selected = selected
        self._enabled = enabled

        self.setFixedSize(_CARD_W, 128)
        self.setObjectName("stepcard")
        self.setCursor(Qt.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)

        label = STEP_METHOD_BY_KEY[step["method"]].label
        head = QHBoxLayout()
        self._num = QLabel(str(index + 1))
        self._num.setAlignment(Qt.AlignCenter)
        self._num.setFixedSize(18, 18)
        head.addWidget(self._num)
        self._title = QLabel(label)
        self._title.setWordWrap(True)
        head.addWidget(self._title, stretch=1)
        # Active toggle: unchecking skips this step in the chain.
        self._active = QCheckBox()
        self._active.setChecked(enabled)
        self._active.setToolTip("Active — uncheck to skip this step")
        self._active.toggled.connect(self.toggle)
        head.addWidget(self._active)
        v.addLayout(head)

        kw = step.get("kwargs", {})
        self._summary = QLabel(kwargs_summary(kw, max_items=3))
        self._summary.setWordWrap(True)
        self._summary.setToolTip(kwargs_summary(kw))  # full kwargs on hover
        v.addWidget(self._summary)
        v.addStretch(1)  # pin the controls to the bottom of the fixed-height card

        bottom = QHBoxLayout()
        bottom.setSpacing(2)
        if removed is not None:
            badge = QLabel(f"− {removed}")
            badge.setToolTip(f"{removed} points removed by this step")
            badge.setStyleSheet("color: #E53935; font-weight: bold; font-size: 11px;")
            bottom.addWidget(badge)
        bottom.addStretch(1)
        for glyph, tip, sig in (("‹", "Move left", self.move_left),
                                ("›", "Move right", self.move_right),
                                ("✎", "Edit step", self.edit)):
            b = QToolButton()
            b.setText(glyph)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(sig)
            bottom.addWidget(b)
        trash = QToolButton()
        trash.setIcon(icons.trash_icon())
        trash.setToolTip("Delete step")
        trash.setAutoRaise(True)
        trash.setCursor(Qt.PointingHandCursor)
        trash.clicked.connect(self.delete)
        bottom.addWidget(trash)
        v.addLayout(bottom)

        self._restyle()

    def setSelected(self, on: bool) -> None:
        self._selected = on
        self._restyle()

    def _restyle(self) -> None:
        accent = theme.manager.tokens.get("ACCENT", "#3A4D5C")
        border = accent if self._selected \
            else theme.manager.tokens.get("BORDER", "#E6E6E3")
        width = 2 if self._selected else 1
        style = "dashed" if not self._enabled else "solid"
        bg = "#F5F5F4" if not self._enabled else "white"
        self.setStyleSheet(
            f"QFrame#stepcard {{ border: {width}px {style} {border}; border-radius: 10px; "
            f"background: {bg}; }}")
        # Grey out the content when the step is inactive.
        num_bg = "#B0B7BD" if not self._enabled else accent
        self._num.setStyleSheet(
            f"background: {num_bg}; color: white; border-radius: 9px; "
            f"font-size: 10px; font-weight: bold;")
        title_color = "#9AA1A7" if not self._enabled else "inherit"
        self._title.setStyleSheet(f"font-weight: bold; color: {title_color};")
        summary_color = "#B0B7BD" if not self._enabled else _C_MUTED
        self._summary.setStyleSheet(f"color: {summary_color}; font-size: 11px;")

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class AddStepCard(QFrame):
    """Trailing dashed ghost card that requests a new step on click."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sig = _CardSignals()
        self.clicked = self._sig.clicked
        self.setFixedSize(_CARD_W, 128)
        self.setObjectName("addcard")
        self.setCursor(Qt.PointingHandCursor)
        v = QVBoxLayout(self)
        btn = QPushButton("＋ Add step")
        btn.setFlat(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self.clicked)
        v.addStretch(1)
        v.addWidget(btn)
        v.addStretch(1)
        border = theme.manager.tokens.get("BORDER", "#E6E6E3")
        self.setStyleSheet(
            f"QFrame#addcard {{ border: 1px dashed {border}; border-radius: 10px; "
            f"background: transparent; }}")

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
