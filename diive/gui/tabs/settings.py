"""
GUI.TABS.SETTINGS: APPEARANCE SETTINGS (live preview)
=====================================================

Edit the GUI appearance (colours from `diive.gui.theme.manager`) with a live
preview: every change re-applies the theme app-wide and repaints, so the whole
GUI updates as you pick colours. A sample variable list on the right previews
the pill tags and selection highlights with the current settings.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.variable_delegate import (
    CREATED_ROLE,
    NAME_ROLE,
    PANEL_ROLE,
    VariableDelegate,
)

#: (name, panel-order, created) sample rows for the pill/highlight preview.
_PREVIEW_ROWS = [
    ("Tair_f (primary panel)", 1, False),
    ("VPD_f (extra panel)", 2, False),
    ("NEE_CUT_REF_f", 0, False),
    ("GPP_CUT_REF_f", 0, False),
    ("Reco_CUT_REF", 0, False),
    ("LE_f", 0, False),
    ("ET_f", 0, False),
    ("Rg_f", 0, False),
    ("PPFD_IN", 0, False),
    (".my_new_feature", 0, True),
]


class _ColorSwatch(QPushButton):
    """A clickable colour swatch bound to a getter/setter of a hex string."""

    def __init__(self, get_hex, set_hex) -> None:
        super().__init__()
        self._get, self._set = get_hex, set_hex
        self.setFixedSize(46, 22)
        self.clicked.connect(self._pick)
        self.refresh()

    def refresh(self) -> None:
        self.setStyleSheet(
            f"background: {self._get()}; border: 1px solid #607D8B; border-radius: 4px;")

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._get()), self, "Pick colour")
        if chosen.isValid():
            self._set(chosen.name())


class SettingsTab(DiiveTab):
    """Live-editable appearance settings with a pill preview."""

    title = "Appearance"

    def build(self) -> QWidget:
        self._swatches: list[_ColorSwatch] = []

        root = QWidget()
        layout = QHBoxLayout(root)

        # Left: scrollable settings.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        col = QVBoxLayout(inner)

        col.addWidget(self._preset_group())
        col.addWidget(self._pill_group())
        col.addWidget(self._ui_group())
        col.addWidget(self._timeseries_group())
        col.addWidget(self._layout_group())

        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(lambda: theme.manager.reset(silent=False))
        col.addWidget(reset)
        col.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        # Right: live preview.
        right = QVBoxLayout()
        right.addWidget(QLabel("Preview"))
        self.preview = QListWidget()
        self.preview.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.preview.setItemDelegate(VariableDelegate(self.preview))
        self.preview.setFixedWidth(280)
        for name, order, created in _PREVIEW_ROWS:
            item = QListWidgetItem(name)
            item.setData(NAME_ROLE, name)
            item.setData(PANEL_ROLE, order)
            item.setData(CREATED_ROLE, created)
            self.preview.addItem(item)
        right.addWidget(self.preview, stretch=1)
        layout.addLayout(right)

        theme.manager.changed.connect(self._on_theme_changed)
        return root

    # --- setting groups ---
    def _preset_group(self) -> QGroupBox:
        box = QGroupBox("Preset")
        v = QVBoxLayout(box)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(theme.PRESETS.keys()))
        self.preset_combo.setCurrentText(theme.manager.preset_name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        v.addWidget(self.preset_combo)
        self.restart_hint = QLabel("Window style changes apply after restarting diive.")
        self.restart_hint.setWordWrap(True)
        self.restart_hint.setVisible(False)
        v.addWidget(self.restart_hint)
        self._refresh_restart_hint()
        return box

    @staticmethod
    def _on_preset_changed(name: str) -> None:
        theme.manager.set_preset(name)  # live-applies palette/typography/icons

    def _refresh_restart_hint(self) -> None:
        """Show the relaunch note when the picked preset's chrome isn't live yet."""
        built = theme.manager.built_chrome
        target = theme.PRESETS.get(theme.manager.preset_name, {}).get("chrome")
        self.restart_hint.setStyleSheet(f"color: {theme.manager.tokens['ACCENT']};")
        self.restart_hint.setVisible(built is not None and target != built)

    def _pill_group(self) -> QGroupBox:
        box = QGroupBox("Pill colours")
        form = QFormLayout(box)
        for kind in theme.manager.pills:
            sw = self._swatch(
                lambda k=kind: theme.manager.pills[k][1],
                lambda h, k=kind: self._set_pill(k, h))
            label = theme.manager.pills[kind][0]
            form.addRow(label, sw)
        # NEW pill
        sw_new = self._swatch(
            lambda: theme.manager.new_pill[1],
            lambda h: self._set_new_pill(h))
        form.addRow(theme.manager.new_pill[0], sw_new)
        return box

    def _ui_group(self) -> QGroupBox:
        box = QGroupBox("Interface colours")
        form = QFormLayout(box)
        for token, label in [
            ("HOVER_BG", "Hover"), ("ACCENT", "Accent"), ("LIST_BG", "List background"),
            ("INPUT_BG", "Input fields"),
            ("BORDER", "Border"), ("PRIMARY_BG", "Selected (primary)"),
            ("EXTRA_BG", "Selected (extra)"),
        ]:
            sw = self._swatch(
                lambda t=token: theme.manager.tokens[t],
                lambda h, t=token: self._set_token(t, h))
            form.addRow(label, sw)
        return box

    def _timeseries_group(self) -> QGroupBox:
        box = QGroupBox("Time-series line colours")
        row = QHBoxLayout(box)
        for i in range(len(theme.manager.ts_colors)):
            sw = self._swatch(
                lambda i=i: theme.manager.ts_colors[i],
                lambda h, i=i: self._set_ts(i, h))
            row.addWidget(sw)
        row.addStretch(1)
        return box

    def _layout_group(self) -> QGroupBox:
        box = QGroupBox("Layout")
        form = QFormLayout(box)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(140, 500)
        self.width_spin.setSingleStep(10)
        self.width_spin.setSuffix(" px")
        self.width_spin.setValue(theme.manager.list_width)
        self.width_spin.valueChanged.connect(self._set_list_width)
        form.addRow("Variable list width", self.width_spin)
        return box

    @staticmethod
    def _set_list_width(value: int) -> None:
        theme.manager.list_width = value
        theme.manager.apply()

    def _swatch(self, get_hex, set_hex) -> _ColorSwatch:
        sw = _ColorSwatch(get_hex, set_hex)
        self._swatches.append(sw)
        return sw

    # --- setters (mutate live theme + apply -> live preview) ---
    @staticmethod
    def _set_pill(kind: str, hexcolor: str) -> None:
        theme.manager.pills[kind][1] = hexcolor
        theme.manager.apply()

    @staticmethod
    def _set_new_pill(hexcolor: str) -> None:
        theme.manager.new_pill[1] = hexcolor
        theme.manager.apply()

    @staticmethod
    def _set_token(token: str, hexcolor: str) -> None:
        theme.manager.tokens[token] = hexcolor
        theme.manager.apply()

    @staticmethod
    def _set_ts(index: int, hexcolor: str) -> None:
        theme.manager.ts_colors[index] = hexcolor
        theme.manager.apply()

    def _on_theme_changed(self) -> None:
        for sw in self._swatches:
            sw.refresh()
        self.width_spin.blockSignals(True)
        self.width_spin.setValue(theme.manager.list_width)
        self.width_spin.blockSignals(False)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(theme.manager.preset_name)
        self.preset_combo.blockSignals(False)
        self._refresh_restart_hint()
        self.preview.viewport().update()
