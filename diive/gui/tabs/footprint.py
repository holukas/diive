"""
GUI.TABS.FOOTPRINT: FLUX FOOTPRINT TAB
======================================

Flux ▸ Flux footprint. The footprint of one set of conditions (Kljun et al.
2015): enter measurement height, roughness, boundary-layer height, Obukhov
length, sigma_v, u* and optionally a wind direction, and the map with its
source-area contours and the crosswind-integrated profile update as the values
change. The tab needs no loaded data.

All computation is library work (``dv.flux.ffp_footprint``) and all drawing is
``dv.plotting.FootprintPlot``. The tab collects the values, runs the model on a
worker thread and lays out the panels (strict GUI<->library separation).

Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple
two-dimensional parameterisation for Flux Footprint Prediction (FFP).
Geoscientific Model Development, 8(11), 3695-3713.
https://doi.org/10.5194/gmd-8-3695-2015

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from diive.flux.lowres.codegen import ffp_footprint_to_code
from diive.gui.tabs.base import DiiveTab
from diive.gui.tabs.overview import HeroBand
from diive.gui.widgets.copy_button import CopyPythonButton
from diive.gui.widgets.debounce import Debouncer
from diive.gui.widgets.mpl_canvas import MplCanvas
from diive.gui.widgets.tab_chrome import build_titlebar
from diive.gui.widgets.worker import LatestRunner

_C_MUTED = "#6B7780"

#: Roughness combo label -> which input it enables.
_ROUGHNESS = {"Roughness length z0": "z0", "Mean wind speed": "umean"}

_CITATION = ("Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015). A simple "
             "two-dimensional parameterisation for Flux Footprint Prediction (FFP). "
             "Geoscientific Model Development, 8(11), 3695-3713. "
             "https://doi.org/10.5194/gmd-8-3695-2015")


def _spin(lo: float, hi: float, value: float, decimals: int, step: float,
          suffix: str, tip: str) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setSuffix(suffix)
    w.setToolTip(tip)
    w.setKeyboardTracking(False)  # one update per entered value, not per digit
    return w


class FootprintTab(DiiveTab):
    """Flux footprint of a single set of conditions (Kljun et al. 2015)."""

    title = "Flux footprint"
    intro = ("Footprint of one set of conditions after Kljun et al. (2015). The map "
             "and the profile update when a value changes.")

    # --- build ---------------------------------------------------------
    def build(self) -> QWidget:
        self._plot = None  # FootprintPlot of the last successful run

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.copy_btn = CopyPythonButton(self._python_code)
        outer.addLayout(build_titlebar(self.title, self.copy_btn))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        self.status = QLabel(self.intro)
        self.status.setStyleSheet("padding: 6px 10px; color: #444;")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        self.hero = HeroBand("FFP", "#EDE7F6", "#4527A0")
        rl.addWidget(self.hero)
        self.canvas = MplCanvas()
        rl.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

        self._runner = LatestRunner()
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)
        self._debounce = Debouncer(root, self._request)
        for w in self._state_controls().values():
            for name in ("valueChanged", "toggled", "currentTextChanged", "editingFinished"):
                sig = getattr(w, name, None)
                if sig is not None:
                    sig.connect(self._debounce.trigger)
                    break
        self._apply_roughness_mode()
        self._request()
        return root

    def _build_controls(self) -> QWidget:
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(10, 6, 10, 6)

        intro = QLabel(self.intro)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(intro)

        site = QGroupBox("Measurement and surface")
        sf = QFormLayout(site)
        self.zm = _spin(0.1, 1000, 20, 2, 0.5, " m",
                        "Measurement height above displacement height, z - d.")
        self.rough_combo = QComboBox()
        self.rough_combo.addItems(list(_ROUGHNESS))
        self.rough_combo.setToolTip("Give the roughness length or the mean wind speed. "
                                    "The model needs one of the two.")
        self.rough_combo.currentTextChanged.connect(self._apply_roughness_mode)
        self.z0 = _spin(0.0001, 10, 0.1, 4, 0.01, " m", "Roughness length.")
        self.umean = _spin(0.01, 50, 3, 2, 0.1, " m s-1", "Mean wind speed at zm.")
        self.rslayer = QCheckBox("Allow roughness sublayer")
        self.rslayer.setToolTip("Also calculate the footprint when zm lies in the roughness "
                                "sublayer (zm <= 12.5 z0). The model then gives only a "
                                "rough estimate.")
        sf.addRow("Height zm", self.zm)
        sf.addRow("Roughness from", self.rough_combo)
        sf.addRow("z0", self.z0)
        sf.addRow("Wind speed", self.umean)
        sf.addRow(self.rslayer)
        self._site_form = sf
        v.addWidget(site)

        atm = QGroupBox("Atmosphere")
        af = QFormLayout(atm)
        self.h = _spin(10.1, 10000, 1500, 0, 50, " m", "Boundary-layer height, above 10 m.")
        self.ol = _spin(-100000, 100000, -50, 1, 10, " m",
                        "Obukhov length. Negative is unstable, positive stable, a large "
                        "absolute value near-neutral. zm / L must be above -15.5.")
        self.sigmav = _spin(0.01, 10, 0.6, 2, 0.05, " m s-1",
                            "Standard deviation of the lateral wind velocity.")
        self.ustar = _spin(0.11, 5, 0.4, 2, 0.05, " m s-1", "Friction velocity, above 0.1.")
        af.addRow("Boundary layer h", self.h)
        af.addRow("Obukhov length L", self.ol)
        af.addRow("sigma_v", self.sigmav)
        af.addRow("u*", self.ustar)
        v.addWidget(atm)

        wind = QGroupBox("Wind direction")
        wf = QFormLayout(wind)
        self.use_wd = QCheckBox("Rotate into the wind direction")
        self.use_wd.setChecked(True)
        self.use_wd.setToolTip("Unticked, the map shows the footprint along the wind "
                               "(upwind distance and crosswind distance).")
        self.wind_dir = _spin(0, 360, 230, 0, 10, " deg", "Wind direction (from), 0-360.")
        self.use_wd.toggled.connect(self.wind_dir.setEnabled)
        wf.addRow(self.use_wd)
        wf.addRow("Direction", self.wind_dir)
        v.addWidget(wind)

        cont = QGroupBox("Source areas and grid")
        cf = QFormLayout(cont)
        self.rs = QLineEdit("10, 20, 30, 40, 50, 60, 70, 80")
        self.rs.setToolTip("Source areas of the contour lines in %, comma separated, "
                           "at most 90.")
        self.source_area = QSpinBox()
        self.source_area.setRange(10, 90)
        self.source_area.setSingleStep(10)
        self.source_area.setValue(80)
        self.source_area.setSuffix(" %")
        self.source_area.setKeyboardTracking(False)
        self.source_area.setToolTip("Source area highlighted on the map and reported above "
                                    "it. Use one of the contour lines.")
        self.nx = QSpinBox()
        self.nx.setRange(600, 5000)
        self.nx.setSingleStep(100)
        self.nx.setValue(1000)
        self.nx.setKeyboardTracking(False)
        self.nx.setToolTip("Number of grid elements of the footprint, at least 600. More "
                           "is finer and slower.")
        self.crop = QCheckBox("Crop grid to the largest contour")
        cf.addRow("Contours", self.rs)
        cf.addRow("Highlight", self.source_area)
        cf.addRow("Grid size", self.nx)
        cf.addRow(self.crop)
        v.addWidget(cont)

        disp = QGroupBox("Display")
        df = QFormLayout(disp)
        self.show_profile = QCheckBox("Profile and cumulative")
        self.show_profile.setChecked(True)
        self.show_rings = QCheckBox("Range rings")
        self.show_rings.setChecked(True)
        self.show_labels = QCheckBox("Contour labels")
        self.show_labels.setChecked(True)
        for w in (self.show_profile, self.show_rings, self.show_labels):
            df.addRow(w)
        v.addWidget(disp)

        cite = QLabel(f"Please cite: {_CITATION}")
        cite.setWordWrap(True)
        cite.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cite.setStyleSheet(f"color: {_C_MUTED}; font-size: 11px; padding-top: 6px;")
        v.addWidget(cite)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(host)
        scroll.setFixedWidth(330)
        return scroll

    def _apply_roughness_mode(self, *_) -> None:
        use_z0 = _ROUGHNESS[self.rough_combo.currentText()] == "z0"
        self._site_form.setRowVisible(self.z0, use_z0)
        self._site_form.setRowVisible(self.umean, not use_z0)
        self.rslayer.setEnabled(use_z0)

    # --- state ---------------------------------------------------------
    def _state_controls(self) -> dict:
        return {"zm": self.zm, "roughness": self.rough_combo, "z0": self.z0,
                "umean": self.umean, "rslayer": self.rslayer, "h": self.h, "ol": self.ol,
                "sigmav": self.sigmav, "ustar": self.ustar, "use_wd": self.use_wd,
                "wind_dir": self.wind_dir, "rs": self.rs, "source_area": self.source_area,
                "nx": self.nx, "crop": self.crop, "show_profile": self.show_profile,
                "show_rings": self.show_rings, "show_labels": self.show_labels}

    def save_state(self) -> dict:
        from diive.gui.widgets.state_utils import save_controls
        return {"controls": save_controls(self._state_controls())}

    def restore_state(self, state: dict) -> None:
        from diive.gui.widgets.state_utils import restore_controls
        restore_controls(self._state_controls(), state.get("controls"))
        self.wind_dir.setEnabled(self.use_wd.isChecked())
        self._apply_roughness_mode()
        self._request()

    # --- run -----------------------------------------------------------
    def _params(self) -> dict:
        """Model inputs as ``ffp_footprint`` keyword arguments.

        Raises:
            ValueError: If the contour list is not a list of numbers.
        """
        use_z0 = _ROUGHNESS[self.rough_combo.currentText()] == "z0"
        text = self.rs.text().replace(";", ",")
        try:
            rs = [float(t) for t in text.split(",") if t.strip()]
        except ValueError:
            raise ValueError("Contours must be numbers in %, comma separated.") from None
        return dict(zm=self.zm.value(), h=self.h.value(), ol=self.ol.value(),
                    sigmav=self.sigmav.value(), ustar=self.ustar.value(),
                    z0=self.z0.value() if use_z0 else None,
                    umean=None if use_z0 else self.umean.value(),
                    wind_dir=self.wind_dir.value() if self.use_wd.isChecked() else None,
                    rs=[r / 100. for r in rs] or None,
                    rslayer=self.rslayer.isChecked() and use_z0,
                    nx=self.nx.value(), crop=self.crop.isChecked())

    def _request(self) -> None:
        try:
            params = self._params()
        except ValueError as exc:
            self._on_failed(str(exc))
            return
        self.status.setText("Calculating the footprint ...")
        self._runner.submit(self._compute_payload, params, self.source_area.value())

    @staticmethod
    def _compute_payload(params: dict, source_area: int):
        """Footprint, its plot object and the inputs (pure, runs on the worker thread)."""
        from diive.core.plotting.footprint import FootprintPlot
        from diive.flux.lowres.footprint import ffp_footprint
        try:
            fp = ffp_footprint(**params, verbose=0)
        except ValueError as exc:  # input outside the model's range: a message, not a crash
            return str(exc), params, source_area
        return FootprintPlot(fp, source_area=source_area / 100.), params, source_area

    def _on_failed(self, message: str) -> None:
        self._plot = None
        self.hero.clear()
        self.status.setText(f"No footprint: {message}")
        self.canvas.show_message("No footprint for these values.\n" + message)

    def _on_done(self, payload) -> None:
        plot, params, area = payload
        if isinstance(plot, str):
            self._on_failed(plot)
            return
        self._plot, self._plot_params, self._plot_area = plot, params, area
        self._render()

    def _render(self) -> None:
        plot, p = self._plot, self._plot_params
        metrics = [("PEAK", f"{plot.peak_distance:.0f} m",
                    "Distance of the footprint peak from the tower")]
        if plot.source_area_distance is not None:
            metrics.append((f"{plot.source_area_r * 100:g} % CONTOUR",
                            f"{plot.source_area_distance:.0f} m",
                            "Largest distance of this source-area contour from the tower"))
        if plot.cumulative_distance is not None:
            metrics.append((f"{self._plot_area} % ALONG WIND",
                            f"{plot.cumulative_distance:.0f} m",
                            "Upwind distance within which the crosswind-integrated "
                            "footprint reaches this source area"))
        metrics.append(("zm / L", f"{p['zm'] / p['ol']:.3g}",
                        "Stability parameter: negative unstable, positive stable"))
        self.hero.set_metrics(metrics)
        missing = [r for r, fr in zip(plot.result.rs or [], plot.result.fr or []) if fr is None]
        note = ""
        if missing:
            note = (" Contours " + ", ".join(f"{r * 100:g} %" for r in missing)
                    + " reach the edge of the grid and are not drawn.")
        self.status.setText(plot.info_text().replace("   ·   ", ", ") + "." + note)

        fig = self.canvas.fig
        self.canvas.reset_layout()
        with_profile = self.show_profile.isChecked()
        if with_profile:
            gs = fig.add_gridspec(2, 2, width_ratios=(1.3, 1))
            ax = fig.add_subplot(gs[:, 0])
            ax_prof = fig.add_subplot(gs[0, 1])
            ax_cum = fig.add_subplot(gs[1, 1], sharex=ax_prof)
        else:
            ax = fig.add_subplot()
        plot.plot(ax=ax, show_rings=self.show_rings.isChecked(),
                  show_contour_labels=self.show_labels.isChecked())
        if with_profile:
            plot.plot_profile(ax_profile=ax_prof, ax_cumulative=ax_cum)
        self.canvas.draw()
        self.canvas.reset_history()

    # --- codegen -------------------------------------------------------
    def _python_code(self) -> str | None:
        try:
            p = self._params()
        except ValueError:
            return None
        return ffp_footprint_to_code(**p, source_area=self.source_area.value() / 100.)
