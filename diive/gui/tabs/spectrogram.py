"""
GUI.TABS.SPECTROGRAM: TIME-FREQUENCY (SPECTROGRAM) EXPLORER
==========================================================

"When is each cyclic pattern strong?" Pick a variable and see a spectrogram —
a short-time Fourier transform that shows how the strength of cycles (e.g. the
24-hour photosynthesis rhythm) changes over the record.

The transform is the library's `dv.analysis.spectrogram`; this tab only collects
the window options, maps the result onto calendar time / cycles-per-day axes,
renders it, and shows a plain-language explanation of what a spectrogram is. No
signal processing of its own (strict GUI<->library separation).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import diive as dv
from diive.gui import theme
from diive.gui.tabs.base import DiiveTab
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.variable_panel import VariablePanel, lock_panel_handle

#: A variable with a strong daily cycle makes the spectrogram instantly legible.
_DEFAULT_VAR = "NEE_CUT_REF_f"

_EXPLANATION = (
    "<b>What a spectrogram shows.</b> It reveals how the strength of repeating "
    "(cyclic) patterns in a variable changes over time. <b>Time</b> runs along "
    "the x-axis, <b>frequency</b> (cycles per day) up the y-axis, and <b>colour</b> "
    "is power — brighter means a stronger cycle at that moment and frequency. "
    "A bright horizontal band at <b>1 cycle/day</b> is the daily (diel) rhythm — "
    "for fluxes it usually strengthens in the growing season and fades in winter. "
    "Bands at 2, 3, … cycles/day are <b>overtones</b> that describe the shape of "
    "the daily cycle. A <b>wider analysis window</b> gives finer frequency detail "
    "but blurs timing; a <b>narrower window</b> does the reverse."
)


class SpectrogramTab(DiiveTab):
    """Spectrogram (time-frequency) view of the selected variable."""

    title = "Spectrogram"

    def build(self) -> QWidget:
        self._df = None
        self._target = None
        self._spec = None           # dict from dv.analysis.spectrogram
        self._valid_index = None    # timestamps of the non-NaN samples
        self._rec_per_day = 1.0
        self._error = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.varpanel = VariablePanel()
        self.varpanel.selected.connect(self._on_select)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_explanation())
        rl.addWidget(self._build_controls())
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)

        splitter.addWidget(self.varpanel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lock_panel_handle(splitter)  # fixed-width list → no misleading ↔ cursor
        outer.addWidget(splitter)
        return root

    # --- sub-widgets ---------------------------------------------------
    def _build_explanation(self) -> QWidget:
        self.explanation = QLabel(_EXPLANATION)
        self.explanation.setWordWrap(True)
        self.explanation.setTextFormat(Qt.TextFormat.RichText)
        list_bg = theme.manager.tokens["LIST_BG"]
        border = theme.manager.tokens["BORDER"]
        self.explanation.setStyleSheet(
            f"QLabel {{ background: {list_bg}; border-bottom: 1px solid {border};"
            f" padding: 8px 12px; color: #37474F; }}")
        return self.explanation

    def _build_controls(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)

        lay.addWidget(QLabel("Window (records)"))
        self.nperseg = QSpinBox()
        self.nperseg.setRange(16, 8192)
        self.nperseg.setSingleStep(64)
        self.nperseg.setValue(512)
        self.nperseg.setToolTip("Samples per analysis window. Larger = finer "
                                "frequency detail, coarser timing.")
        lay.addWidget(self.nperseg)

        lay.addWidget(QLabel("Overlap %"))
        self.overlap = QSpinBox()
        self.overlap.setRange(0, 95)
        self.overlap.setValue(50)
        self.overlap.setToolTip("Overlap between successive windows (smoother in time).")
        lay.addWidget(self.overlap)

        lay.addWidget(QLabel("Window"))
        self.window = QComboBox()
        self.window.addItems(["hann", "hamming", "blackman"])
        lay.addWidget(self.window)

        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self._update_current)
        lay.addWidget(self.update_btn)

        lay.addSpacing(14)
        lay.addWidget(QLabel("Max cycles/day"))
        self.max_freq = QDoubleSpinBox()
        self.max_freq.setRange(0.5, 48.0)
        self.max_freq.setSingleStep(0.5)
        self.max_freq.setValue(4.0)
        self.max_freq.setToolTip("Upper limit of the frequency (y) axis.")
        self.max_freq.valueChanged.connect(self._on_view_changed)
        lay.addWidget(self.max_freq)

        lay.addWidget(QLabel("Colormap"))
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "magma", "plasma", "inferno", "turbo", "cividis"])
        self.cmap.currentTextChanged.connect(self._on_view_changed)
        lay.addWidget(self.cmap)
        lay.addStretch(1)
        return bar

    # --- data flow -----------------------------------------------------
    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"target": self._target,
                "controls": save_controls(
                    {"nperseg": self.nperseg, "overlap": self.overlap,
                     "window": self.window, "max_freq": self.max_freq,
                     "cmap": self.cmap})}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        # `state.get("controls") or state` tolerates older projects that saved
        # the control values flat at the top level.
        restore_controls({"nperseg": self.nperseg, "overlap": self.overlap,
                          "window": self.window, "max_freq": self.max_freq,
                          "cmap": self.cmap},
                         state.get("controls") or state)
        t = state.get("target")
        if t and self._df is not None and t in self._df.columns:
            self._on_select(t)

    def on_data_loaded(self, df, created: set | None = None) -> None:
        self._df = df
        self.varpanel.set_variables(df.columns, created)
        cols = [str(c) for c in df.columns]
        numeric = df.select_dtypes(include="number").columns.tolist()
        if _DEFAULT_VAR in cols and _DEFAULT_VAR in numeric:
            default = _DEFAULT_VAR
        elif numeric:
            default = str(numeric[0])
        else:
            return
        self._on_select(default)

    def _on_select(self, name: str, _additive: bool = False) -> None:
        if not name or self._df is None:
            return
        self._target = name
        self.varpanel.set_panels([name])
        self.varpanel.run_with_loading(name, self._compute)

    def _update_current(self) -> None:
        if self._target is not None and self._df is not None:
            self.varpanel.run_with_loading(self._target, self._compute)

    def _on_view_changed(self, _v=None) -> None:
        # Frequency limit / colormap are cheap -> re-render without recomputing.
        if self._spec is not None:
            self._render()

    def _compute(self) -> None:
        series = self._df[self._target]
        valid = series.dropna()
        self._valid_index = valid.index
        # Records per day from the index spacing -> frequency in cycles/day.
        delta = pd.Series(self._df.index).diff().median()
        self._rec_per_day = (
            pd.Timedelta("1D") / delta
            if pd.notna(delta) and delta > pd.Timedelta(0) else 1.0)
        nperseg = self.nperseg.value()
        noverlap = int(nperseg * self.overlap.value() / 100)
        self._spec = None
        self._error = None
        try:
            # The transform itself is the library's; the tab only reads it back.
            self._spec = dv.analysis.spectrogram(
                series, nperseg=nperseg, noverlap=noverlap,
                window=self.window.currentText())
        except Exception as err:
            self._error = str(err)
        self._render()

    # --- rendering -----------------------------------------------------
    def _render(self) -> None:
        ax = self.canvas.new_axes(1)[0]
        if self._spec is None:
            ax.text(0.5, 0.5, self._error or "No spectrogram",
                    ha="center", va="center", wrap=True, transform=ax.transAxes)
            self.canvas.draw()
            return
        try:
            spec = self._spec
            cycles_per_day = spec["frequencies"] * self._rec_per_day
            # Map each segment centre (in valid-sample positions) to its real
            # timestamp, so the x-axis is calendar time even across gaps.
            n = len(self._valid_index)
            pos = np.clip(np.round(spec["times"]).astype(int), 0, n - 1)
            x = mdates.date2num(self._valid_index[pos].to_pydatetime())
            mesh = ax.pcolormesh(x, cycles_per_day, spec["power_db"],
                                 shading="gouraud", cmap=self.cmap.currentText())
            ax.set_ylim(0, self.max_freq.value())
            ax.axhline(1.0, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency (cycles per day)")
            ax.set_title(f"Spectrogram — {self._target}")
            cb = self.canvas.fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01)
            cb.set_label("Power (dB)")
        except Exception as err:
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot plot:\n{err}", ha="center", va="center",
                    wrap=True, transform=ax.transAxes)
        self.canvas.draw()
