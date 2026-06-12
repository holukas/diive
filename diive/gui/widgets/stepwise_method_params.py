"""
GUI.WIDGETS.STEPWISE_METHOD_PARAMS: PARAM WIDGETS FOR THE STEPWISE OUTLIER CHAIN
===============================================================================

One small widget per ``StepwiseOutlierDetection.flag_*`` method, collecting that
method's parameters and producing a ``{"method": str, "kwargs": dict}`` *step* —
the shape the library's ``level32_to_code`` consumes and the shape a stepwise
chain (flux L3.2 tab, standalone stepwise tab) feeds to the detector.

GUI-only: these are widgets, labels, and tooltips. The method *names* and their
parameter *meanings* are the library's (`StepwiseOutlierDetection`); detection
itself runs in the library. The day/night split needs no coordinates here — the
detector is built with the site coordinates, so a step only carries
``separate_daytime_nighttime`` (plus per-period thresholds where the method has
them).

Add a method = add a ``_StepParams`` subclass and list it in ``STEP_METHODS``.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QWidget,
)


def _fspin(default: float, lo: float, hi: float, step: float = 0.5,
           decimals: int = 2) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setDecimals(decimals)
    sp.setValue(default)
    return sp


def _ispin(default: int, lo: int, hi: int) -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setValue(default)
    return sp


class _StepParams(QWidget):
    """Base for a method's parameter form. Subclasses set ``method``/``label``,
    build their rows in ``_build``, and map widgets to kwargs in ``kwargs``."""

    #: Name of the ``StepwiseOutlierDetection`` method this step calls.
    method = ""
    #: Human label for the method picker.
    label = ""
    #: Whether the method offers a daytime/nighttime split.
    supports_daynight = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self._build(form)

    # --- subclass hooks ---
    def _build(self, form: QFormLayout) -> None:
        raise NotImplementedError

    def kwargs(self) -> dict:
        raise NotImplementedError

    def load(self, kwargs: dict) -> None:
        """Seed the form widgets from an existing step's kwargs (for editing).

        Subclasses with parameters override this; the default is a no-op so
        parameter-less methods (e.g. missing-values) need not implement it.
        """
        return

    # --- result ---
    def step(self) -> dict:
        """The ``{"method", "kwargs"}`` step for codegen / the detector chain."""
        return {"method": self.method, "kwargs": self.kwargs()}

    # --- shared rows ---
    def _add_repeat(self, form: QFormLayout, default: bool = True) -> None:
        self.repeat_cb = QCheckBox("Repeat until no more outliers")
        self.repeat_cb.setChecked(default)
        self.repeat_cb.setToolTip(
            "Re-run the test until a pass finds no new outliers (removed points are "
            "excluded from the next pass). Off = a single pass.")
        form.addRow(self.repeat_cb)

    def _repeat_kwargs(self) -> dict:
        return {"repeat": self.repeat_cb.isChecked()}


class HampelParams(_StepParams):
    method = "flag_outliers_hampel_test"
    label = "Hampel filter"
    supports_daynight = True

    def _build(self, form: QFormLayout) -> None:
        self.window = _ispin(48 * 13, 3, 1_000_000)
        self.window.setToolTip("Sliding-window size in record count (624 = 13 days at "
                               "30-min sampling, Papale 2006).")
        form.addRow("Window (records)", self.window)
        self.n_sigma = _fspin(5.5, 0.1, 50.0)
        self.n_sigma.setToolTip("Threshold width in MADs; lower = stricter. Used for all "
                                "records unless day/night separation is on.")
        form.addRow("n sigma (global)", self.n_sigma)
        self.diff_cb = QCheckBox("Use double-differencing (Papale 2006)")
        self.diff_cb.setChecked(True)
        form.addRow(self.diff_cb)
        self.dn_cb = QCheckBox("Separate daytime / nighttime")
        form.addRow(self.dn_cb)
        self.n_sigma_dt = _fspin(5.5, 0.1, 50.0)
        self.n_sigma_dt.setToolTip("Threshold (n sigma) for daytime records.")
        self.n_sigma_nt = _fspin(5.5, 0.1, 50.0)
        self.n_sigma_nt.setToolTip("Threshold (n sigma) for nighttime records.")
        form.addRow("Daytime n sigma", self.n_sigma_dt)
        form.addRow("Nighttime n sigma", self.n_sigma_nt)
        self.n_sigma_dt.setEnabled(False)
        self.n_sigma_nt.setEnabled(False)
        self.dn_cb.toggled.connect(self._toggle_dn)
        self._add_repeat(form)

    def _toggle_dn(self, on: bool) -> None:
        self.n_sigma_dt.setEnabled(on)
        self.n_sigma_nt.setEnabled(on)
        if on:  # seed per-period from the global value (convention)
            self.n_sigma_dt.setValue(self.n_sigma.value())
            self.n_sigma_nt.setValue(self.n_sigma.value())

    def kwargs(self) -> dict:
        kw = dict(window_length=self.window.value(), n_sigma=self.n_sigma.value(),
                  use_differencing=self.diff_cb.isChecked(),
                  separate_daytime_nighttime=self.dn_cb.isChecked(),
                  **self._repeat_kwargs())
        if self.dn_cb.isChecked():
            kw.update(n_sigma_daytime=self.n_sigma_dt.value(),
                      n_sigma_nighttime=self.n_sigma_nt.value())
        return kw

    def load(self, kwargs: dict) -> None:
        self.window.setValue(int(kwargs.get("window_length", self.window.value())))
        self.n_sigma.setValue(float(kwargs.get("n_sigma", self.n_sigma.value())))
        self.diff_cb.setChecked(bool(kwargs.get("use_differencing", True)))
        # Set the toggle first (it seeds the per-period spins from the global
        # value), then overwrite with the saved per-period thresholds.
        self.dn_cb.setChecked(bool(kwargs.get("separate_daytime_nighttime", False)))
        if kwargs.get("n_sigma_daytime") is not None:
            self.n_sigma_dt.setValue(float(kwargs["n_sigma_daytime"]))
        if kwargs.get("n_sigma_nighttime") is not None:
            self.n_sigma_nt.setValue(float(kwargs["n_sigma_nighttime"]))
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class LocalSDParams(_StepParams):
    method = "flag_outliers_localsd_test"
    label = "Local SD"
    supports_daynight = True

    def _build(self, form: QFormLayout) -> None:
        self.n_sd = _fspin(7.0, 0.1, 50.0)
        self.n_sd.setToolTip("Number of standard deviations from the rolling median; "
                             "lower = stricter.")
        form.addRow("n SD (global)", self.n_sd)
        self.winsize = _ispin(0, 0, 1_000_000)
        self.winsize.setToolTip("Rolling-window size in records (0 = auto).")
        form.addRow("Window (records, 0=auto)", self.winsize)
        self.constant_cb = QCheckBox("Constant SD across the record")
        form.addRow(self.constant_cb)
        self.dn_cb = QCheckBox("Separate daytime / nighttime")
        form.addRow(self.dn_cb)
        self.n_sd_dt = _fspin(7.0, 0.1, 50.0)
        self.n_sd_nt = _fspin(7.0, 0.1, 50.0)
        form.addRow("Daytime n SD", self.n_sd_dt)
        form.addRow("Nighttime n SD", self.n_sd_nt)
        self.n_sd_dt.setEnabled(False)
        self.n_sd_nt.setEnabled(False)
        self.dn_cb.toggled.connect(self._toggle_dn)
        self._add_repeat(form)

    def _toggle_dn(self, on: bool) -> None:
        self.n_sd_dt.setEnabled(on)
        self.n_sd_nt.setEnabled(on)
        if on:
            self.n_sd_dt.setValue(self.n_sd.value())
            self.n_sd_nt.setValue(self.n_sd.value())

    def kwargs(self) -> dict:
        win = self.winsize.value() or None
        if self.dn_cb.isChecked():
            # LocalSD takes list [daytime, nighttime] for the per-period split.
            n_sd = [self.n_sd_dt.value(), self.n_sd_nt.value()]
            winsize = [win, win] if win is not None else None
        else:
            n_sd = self.n_sd.value()
            winsize = win
        return dict(n_sd=n_sd, winsize=winsize, constant_sd=self.constant_cb.isChecked(),
                    separate_daytime_nighttime=self.dn_cb.isChecked(),
                    **self._repeat_kwargs())

    def load(self, kwargs: dict) -> None:
        self.constant_cb.setChecked(bool(kwargs.get("constant_sd", False)))
        dn = bool(kwargs.get("separate_daytime_nighttime", False))
        self.dn_cb.setChecked(dn)
        n_sd = kwargs.get("n_sd", self.n_sd.value())
        win = kwargs.get("winsize", 0)
        if dn and isinstance(n_sd, (list, tuple)):
            self.n_sd_dt.setValue(float(n_sd[0]))
            self.n_sd_nt.setValue(float(n_sd[1]))
            win0 = win[0] if isinstance(win, (list, tuple)) and win else win
            self.winsize.setValue(int(win0) if win0 else 0)
        else:
            val = n_sd[0] if isinstance(n_sd, (list, tuple)) else n_sd
            self.n_sd.setValue(float(val))
            self.winsize.setValue(int(win) if isinstance(win, int) and win else 0)
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class ZScoreParams(_StepParams):
    method = "flag_outliers_zscore_test"
    label = "z-score"
    supports_daynight = True

    def _build(self, form: QFormLayout) -> None:
        self.thres = _fspin(4.0, 0.1, 50.0)
        self.thres.setToolTip("z-score threshold; typical 2.5–5, lower = stricter.")
        form.addRow("z threshold", self.thres)
        self.dn_cb = QCheckBox("Separate daytime / nighttime")
        self.dn_cb.setToolTip("Compute the z-score within daytime and nighttime "
                              "records separately.")
        form.addRow(self.dn_cb)
        self._add_repeat(form)

    def kwargs(self) -> dict:
        return dict(thres_zscore=self.thres.value(),
                    separate_daytime_nighttime=self.dn_cb.isChecked(),
                    **self._repeat_kwargs())

    def load(self, kwargs: dict) -> None:
        self.thres.setValue(float(kwargs.get("thres_zscore", self.thres.value())))
        self.dn_cb.setChecked(bool(kwargs.get("separate_daytime_nighttime", False)))
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class ZScoreRollingParams(_StepParams):
    method = "flag_outliers_zscore_rolling_test"
    label = "z-score (rolling)"

    def _build(self, form: QFormLayout) -> None:
        self.thres = _fspin(4.0, 0.1, 50.0)
        self.thres.setToolTip("Rolling z-score threshold; lower = stricter.")
        form.addRow("z threshold", self.thres)
        self.winsize = _ispin(0, 0, 1_000_000)
        self.winsize.setToolTip("Rolling-window size in records (0 = auto).")
        form.addRow("Window (records, 0=auto)", self.winsize)
        self._add_repeat(form)

    def kwargs(self) -> dict:
        return dict(thres_zscore=self.thres.value(),
                    winsize=self.winsize.value() or None, **self._repeat_kwargs())

    def load(self, kwargs: dict) -> None:
        self.thres.setValue(float(kwargs.get("thres_zscore", self.thres.value())))
        win = kwargs.get("winsize")
        self.winsize.setValue(int(win) if win else 0)
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class IncrementsParams(_StepParams):
    method = "flag_outliers_increments_zcore_test"
    label = "z-score (increments)"

    def _build(self, form: QFormLayout) -> None:
        self.thres = _fspin(30.0, 0.1, 200.0)
        self.thres.setToolTip("z-score threshold on record-to-record increments; "
                              "flags abrupt jumps.")
        form.addRow("z threshold", self.thres)
        self._add_repeat(form)

    def kwargs(self) -> dict:
        return dict(thres_zscore=self.thres.value(), **self._repeat_kwargs())

    def load(self, kwargs: dict) -> None:
        self.thres.setValue(float(kwargs.get("thres_zscore", self.thres.value())))
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class LOFParams(_StepParams):
    method = "flag_outliers_lof_test"
    label = "Local outlier factor"
    supports_daynight = True

    def _build(self, form: QFormLayout) -> None:
        self.n_neighbors = _ispin(0, 0, 1_000_000)
        self.n_neighbors.setToolTip("Number of neighbours (0 = auto, ~1/200 of records).")
        form.addRow("Neighbours (0=auto)", self.n_neighbors)
        self.contamination = _fspin(0.0, 0.0, 0.5, step=0.01, decimals=3)
        self.contamination.setToolTip("Expected outlier fraction (0 = auto).")
        form.addRow("Contamination (0=auto)", self.contamination)
        self.dn_cb = QCheckBox("Separate daytime / nighttime")
        form.addRow(self.dn_cb)
        self._add_repeat(form)

    def kwargs(self) -> dict:
        return dict(
            n_neighbors=self.n_neighbors.value() or None,
            contamination=(self.contamination.value() if self.contamination.value() > 0
                           else None),
            separate_daytime_nighttime=self.dn_cb.isChecked(),
            **self._repeat_kwargs())

    def load(self, kwargs: dict) -> None:
        nn = kwargs.get("n_neighbors")
        self.n_neighbors.setValue(int(nn) if nn else 0)
        cont = kwargs.get("contamination")
        self.contamination.setValue(float(cont) if cont else 0.0)
        self.dn_cb.setChecked(bool(kwargs.get("separate_daytime_nighttime", False)))
        self.repeat_cb.setChecked(bool(kwargs.get("repeat", True)))


class MissingValsParams(_StepParams):
    method = "flag_missingvals_test"
    label = "Missing values"

    def _build(self, form: QFormLayout) -> None:
        note = QLabel("Flags records that are already missing. No parameters.")
        note.setWordWrap(True)
        form.addRow(note)

    def kwargs(self) -> dict:
        return {}


#: Ordered registry of the outlier methods offered by the stepwise picker.
STEP_METHODS: list[type[_StepParams]] = [
    HampelParams,
    LocalSDParams,
    ZScoreParams,
    ZScoreRollingParams,
    IncrementsParams,
    LOFParams,
    MissingValsParams,
]

#: Method name -> params-widget class.
STEP_METHOD_BY_KEY: dict[str, type[_StepParams]] = {c.method: c for c in STEP_METHODS}


def method_labels() -> list[tuple[str, str]]:
    """``[(method_name, label), ...]`` in registry order — for a picker combo."""
    return [(c.method, c.label) for c in STEP_METHODS]
