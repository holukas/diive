"""
GUI.TABS.CORRECTIONS_NIGHTTIME_OFFSET: REMOVE NIGHTTIME ZERO OFFSET TAB
======================================================================

Remove a nighttime zero-offset from a variable that should read zero at night:
the daily nighttime mean is subtracted as the offset and nighttime values are
forced to zero (`dv.corrections.remove_nighttime_zero_offset`). Needs site
coordinates for the day/night split.

All the preview/threading/plotting machinery lives in :class:`BaseCorrectionTab`;
this tab only declares the correction key, the hero chip, and that it needs
coordinates. Intended for variables that are zero at night — shortwave radiation
(SW_IN / SW_OUT) and PPFD (PPFD_IN / PPFD_OUT). Available for any variable
(suggestion, not a lock).

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout

from diive.gui.tabs._correction_base import (
    BaseCorrectionTab,
    _C_CORRECTED,
    _C_RAW,
)
from diive.preprocessing.corrections.offsetcorrection import (
    nighttime_zero_offset_diagnostics,
)
from diive.preprocessing.qaqc.measurements import CORR_RADIATION_ZERO_OFFSET

_C_OFFSET = "#FB8C00"   # orange 600 — the daily offset
_C_STEP = "#1E88E5"     # blue 600   — series after subtracting the offset
_C_ZERO = "#CFD8DC"     # blue-grey 100 — zero reference line
_C_OK = "#2E7D32"       # green 800   — "no negatives remain" confirmation
_C_BAD = "#C62828"      # red 800     — negatives present


class NighttimeZeroOffsetTab(BaseCorrectionTab):
    """Remove a nighttime zero-offset from a variable that should be zero at night.

    Overrides the base preview/hero: the library exposes every intermediate
    series (`nighttime_zero_offset_diagnostics`) so the tab can show the full
    pipeline — original, daily offset, series minus offset, final corrected — and
    a stats hero of below-zero counts before/after (the night-after count
    confirming nighttime no longer dips below zero)."""

    title = "Remove nighttime zero offset"
    intro = ("For variables that should read zero at night (e.g. shortwave "
             "radiation, PPFD). For each day, the mean of that day's nighttime "
             "values is the offset, subtracted from all of the day's records "
             "(days with no nighttime data use the median of all daily offsets). "
             "Nighttime values are then forced to zero and, optionally, any "
             "remaining negatives clamped to zero.")
    method_suffix = "NIGHTOFFSET"
    corr_key = CORR_RADIATION_ZERO_OFFSET
    method_chip_label = "NIGHTTIME OFFSET"
    method_chip_bg = "#FFF3E0"
    method_chip_fg = "#E65100"
    needs_coords = True
    suited_for = ("Suited for variables that read zero at night — shortwave "
                  "radiation (SW_IN, SW_OUT) and PPFD (PPFD_IN, PPFD_OUT).")

    def _add_method_rows(self, form: QFormLayout) -> None:
        self.clamp_cb = QCheckBox("Clamp negative values to zero")
        self.clamp_cb.setChecked(True)
        self.clamp_cb.setToolTip(
            "After removing the offset and zeroing the night, set any remaining "
            "negative values (daytime included) to zero — enforces the variable's "
            "physical floor of zero. Uncheck to keep offset-corrected negatives "
            "(e.g. to inspect them).")
        form.addRow(self.clamp_cb)

    def _current_kwargs(self) -> dict:
        return {"clamp_negatives": self.clamp_cb.isChecked()}

    def _method_controls(self) -> dict:
        return {"clamp_negatives": self.clamp_cb}

    def _apply(self, series, kwargs: dict, coords: tuple):
        lat, lon, utc = coords
        res = nighttime_zero_offset_diagnostics(
            series, lat=lat, lon=lon, utc_offset=utc,
            clamp_negatives=kwargs.get("clamp_negatives", True))
        extra = {
            "offset": res.offset,
            "corrected_by_offset": res.corrected_by_offset,
            "n_below_zero_before": res.n_below_zero_before,
            "n_below_zero_before_night": res.n_below_zero_before_night,
            "n_below_zero_after": res.n_below_zero_after,
            "n_below_zero_after_night": res.n_below_zero_after_night,
            "n_night": res.n_night,
        }
        return res.corrected, extra

    @staticmethod
    def _ok_value(n: int) -> str:
        """Green '0 ✓' when no negatives remain, else a red count."""
        if n == 0:
            return f"<span style='color:{_C_OK}'>0 ✓</span>"
        return f"<span style='color:{_C_BAD}'>{n:,}</span>"

    def _hero_metrics(self, payload: dict) -> list:
        e = payload["extra"]
        return [
            ("BELOW ZERO BEFORE", f"{e['n_below_zero_before']:,}",
             "Records with negative radiation before correction"),
            ("NIGHT BELOW ZERO BEFORE", f"{e['n_below_zero_before_night']:,}",
             "Nighttime records with negative radiation before correction "
             "(the offset being removed)"),
            ("BELOW ZERO AFTER", self._ok_value(e["n_below_zero_after"]),
             "Records with negative radiation after correction (should be 0)"),
            ("NIGHT BELOW ZERO AFTER", self._ok_value(e["n_below_zero_after_night"]),
             "Nighttime records with negative radiation after correction "
             "(should be 0 — confirms the night no longer dips below zero)"),
        ]

    def _status_text(self, payload: dict) -> str:
        e = payload["extra"]
        ok = e["n_below_zero_after"] == 0 and e["n_below_zero_after_night"] == 0
        confirm = ("No records remain below zero after correction."
                   if ok else
                   f"{e['n_below_zero_after']:,} records still below zero.")
        return (f"{e['n_below_zero_before']:,} records were below zero before "
                f"({e['n_below_zero_before_night']:,} at night). {confirm} "
                f"'Add' keeps {payload['corrected'].name}.")

    def _render_result(self, payload: dict) -> None:
        """Four stacked panels sharing the time axis: original, daily offset,
        series minus offset, and the final corrected series. Zero lines make the
        below-zero records (and their removal) visible."""
        series = self._df[payload["var"]]
        e = payload["extra"]
        offset = e["offset"]
        step = e["corrected_by_offset"]
        final = payload["corrected"]

        self.canvas.reset_layout()
        fig = self.canvas.fig
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
        ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

        ax1.plot(series.index, series.to_numpy(), color=_C_RAW, lw=0.6)
        ax1.axhline(0, color=_C_ZERO, lw=0.8, zorder=0)
        ax1.set_title(f"{series.name} — original", fontsize=9)

        ax2.plot(offset.index, offset.to_numpy(), color=_C_OFFSET, lw=0.9)
        ax2.set_title("daily nighttime-mean offset", fontsize=9)

        ax3.plot(step.index, step.to_numpy(), color=_C_STEP, lw=0.6)
        ax3.axhline(0, color=_C_ZERO, lw=0.8, zorder=0)
        ax3.set_title("series − offset (before night set-to-zero / clamp)",
                      fontsize=9)

        ax4.plot(final.index, final.to_numpy(), color=_C_CORRECTED, lw=0.6)
        ax4.axhline(0, color=_C_ZERO, lw=0.8, zorder=0)
        ax4.set_title("corrected (night forced to zero, negatives clamped)",
                      fontsize=9)
        self.canvas.draw()
