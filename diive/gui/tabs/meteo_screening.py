"""
GUI.TABS.METEO_SCREENING: SCREEN & RESAMPLE HIGH-RES DATABASE METEO
==================================================================

The full screening experience (:class:`~diive.gui.tabs._screening_base.ScreeningTabBase`)
applied to a high-resolution meteo field handed over from the Database explorer,
**plus resampling**. It is feature-identical to the Stepwise screening tab —
editable outlier-test cards, corrections, QCF, the live preview, Copy Python —
and adds a **Resample** page; on Add it emits the screened (and corrected) series
resampled to the chosen resolution (default 30MIN), ready to merge into the
working dataset.

How it differs from the plain screening tab, all via the base's seams:
- **Data source**: a synthetic ``self._df`` built from the staged ``data_detailed``
  (the field's data column on its TIMESTAMP_END index), so all the base machinery
  works unchanged. The database tags are kept aside and re-attached on emit.
- **Extra inspector page**: ``Resample`` (target resolution + aggregation + min counts).
- **Emit**: the screened series resampled to the target resolution, converted
  END -> MIDDLE (so it aligns onto diive's TIMESTAMP_MIDDLE index), with a
  collision rename, an overlap guard, and the InfluxDB origin + all db tags in
  the variable's history.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from diive.core.io.db.influx.common import TAGS
from diive.core.metadata import ATTRS_KEY, DERIVED, provenance_attr
from diive.core.times.resampling import resample_series_to_freq
from diive.core.times.times import (
    DetectFrequency,
    TimestampSanitizer,
    convert_series_timestamp_to_middle,
)
from diive.gui import site
from diive.gui.tabs._screening_base import ScreeningTabBase
from diive.qaqc import detect_measurement

_C_MUTED = "#6B7780"
#: Common target resolutions offered in the resample picker (editable).
_FREQS = ["30min", "10min", "15min", "1h"]


class MeteoScreeningTab(ScreeningTabBase):
    """Full screening + resampling for a high-res field from the database."""

    title = "Meteo screening (database)"
    add_button_label = "Add resampled to dataset"
    provenance_op = "Imported from InfluxDB & screened"

    def build(self) -> QWidget:
        # DB-specific state (the base is otherwise self._df / self._var-centric).
        self._tags: dict = {}          # {field: {tag: value}} from the download
        self._db_meta: dict = {}        # bucket / measurement / data_version / utc_offset
        self._dataset_index = None      # working dataset index (overlap check)
        self._dataset_columns: set = set()  # working dataset columns (collision check)
        self._target_freq: str | None = None  # main-df resolution = resample target
        self._source_freq: str | None = None  # downloaded data resolution
        self._utc_offset = None         # timezone the staged data are in (offset to UTC)
        return super().build()

    # --- extra inspector page: Resample ---
    def _inspector_pages(self) -> list:
        pages = super()._inspector_pages()
        pages.append(("Resample", self._build_resample_page, self._INSPECTOR_W))
        return pages

    def _build_resample_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        box = QGroupBox("Resample (after screening)")
        f = QFormLayout(box)
        self._freq = QComboBox()
        self._freq.setEditable(True)
        self._freq.addItems(_FREQS)
        self._freq.setToolTip("Target resolution, e.g. 30min (default), 10min, 1h.")
        self._agg = QComboBox()
        self._agg.addItems(["mean", "sum"])
        self._agg.setToolTip("mean for most meteo, sum for precipitation.")
        self._mincounts = QDoubleSpinBox()
        self._mincounts.setRange(0.0, 1.0)
        self._mincounts.setSingleStep(0.05)
        self._mincounts.setValue(0.9)
        self._mincounts.setToolTip(
            "Minimum fraction of an interval that must be present for the "
            "aggregate to be computed (else the resampled value is missing). "
            "0.9 (default) requires 90% coverage; lower values accept "
            "sparsely-populated intervals.")
        f.addRow("Resolution", self._freq)
        f.addRow("Aggregation", self._agg)
        f.addRow("Min. records (frac)", self._mincounts)
        for w in (self._freq, self._agg):
            w.currentTextChanged.connect(self._on_resample_changed)
        self._mincounts.valueChanged.connect(self._on_resample_changed)
        v.addWidget(box)
        hint = QLabel(
            "The screened (and corrected) series is resampled to this resolution, "
            "then added on TIMESTAMP_MIDDLE so it aligns with the working dataset. "
            "The default target is the working dataset's resolution; if the "
            "downloaded data already match it, no resampling is done.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_C_MUTED};")
        v.addWidget(hint)
        self._resample_info = QLabel("")
        self._resample_info.setWordWrap(True)
        v.addWidget(self._resample_info)
        v.addStretch(1)
        return page

    @staticmethod
    def _freq_str(index) -> str | None:
        """Normalised resolution of *index* (e.g. '30min', '1h'), or None."""
        if index is None or len(index) < 2:
            return None
        try:
            detected = DetectFrequency(index).freq
        except Exception:
            return None
        if not detected:
            return None
        off = to_offset(detected)
        return f"{off.n}{off.name}"

    @staticmethod
    def _same_freq(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        try:
            return to_offset(a) == to_offset(b)
        except Exception:
            return a == b

    def _agg_hint(self) -> str:
        """Flag a plain arithmetic mean on a circular/bounded variable (non-blocking)."""
        if self._agg.currentText() != "mean":
            return ""
        meas = detect_measurement(self._var) if self._var else None
        if meas == "WD":
            return ("  NOTE: wind direction is circular; a plain mean is misleading "
                    "across the 0/360 deg wrap.")
        if meas == "RH":
            return ("  NOTE: relative humidity is bounded (0-100%); a plain mean can "
                    "be misleading.")
        return ""

    def _update_resample_info(self) -> None:
        """Show source vs target resolution and whether resampling will happen."""
        src = self._source_freq or "?"
        tgt = self._target_freq or "?"
        selected = self._freq.currentText().strip()
        if self._same_freq(self._source_freq, selected):
            verdict = "-> already at target resolution, no resampling needed."
        else:
            verdict = f"-> resample to {selected} ({self._agg.currentText()})."
        self._resample_info.setText(
            f"Source: {src}  ·  dataset target: {tgt}  {verdict}{self._agg_hint()}")

    def _on_resample_changed(self, *_) -> None:
        """Resample settings changed: refresh the info + rebuild the column."""
        self._update_resample_info()
        self._build_result()
        self.add_btn.setEnabled(self._result_df is not None and not self._result_df.empty)

    # --- staged data intake (from the Database explorer) ---
    def load_staged(self, payload: dict) -> None:
        data_detailed = payload.get("data_detailed") or {}
        field = payload.get("field")
        frame = data_detailed.get(field)
        if frame is None or field is None or field not in frame.columns:
            self.status.setText("No data received from the Database explorer.")
            return

        # Regular, gap-aware END-timestamp series so both screening and resampling
        # have a clean, regular index to work on.
        clean = TimestampSanitizer(
            data=frame[field], output_middle_timestamp=False, verbose=False).get()
        self._df = pd.DataFrame({field: clean})

        # Keep the database tags + origin aside; re-attached on emit.
        self._tags = {field: {t: str(frame[t].iloc[0]) for t in TAGS
                              if t in frame.columns and len(frame[t])}}
        self._db_meta = {k: payload.get(k) for k in
                         ("bucket", "measurement", "data_version", "utc_offset")}
        self._utc_offset = payload.get("utc_offset")

        # Detect the source resolution and default the resample target to the
        # working dataset's resolution (so matching data won't be resampled).
        self._source_freq = self._freq_str(self._df.index)
        if self._target_freq:
            self._freq.setCurrentText(self._target_freq)

        self.varpanel.set_variables(self._df.columns, None)
        meas = detect_measurement(field)
        self._agg.setCurrentText("sum" if meas == "PREC" else "mean")
        self._select(field)  # base: show raw, rebuild cards, detect measurement
        self._update_resample_info()
        same = self._same_freq(self._source_freq, self._freq.currentText().strip())
        note = ("already at the dataset resolution, so no resampling is needed"
                if same else f"will be resampled to {self._freq.currentText().strip()}")
        self.status.setText(
            f"Loaded {field} from the database ({len(clean)} records, "
            f"{self._source_freq or 'unknown'} resolution, {self._tz_label()}); {note}. "
            f"Add screening steps, then add to the dataset.{self._tz_warning()}")

    def _tz_label(self) -> str:
        """The timezone the staged data are in, e.g. 'UTC+01:00'."""
        if self._utc_offset is None:
            return "timezone unknown"
        total = int(round(float(self._utc_offset) * 60))
        sign = "+" if total >= 0 else "-"
        h, m = divmod(abs(total), 60)
        return f"UTC{sign}{h:02d}:{m:02d}"

    def _tz_warning(self) -> str:
        """Flag a download-vs-project timezone mismatch (the DB is UTC; the
        download offset must match the dataset's timezone or the merge is shifted)."""
        if (self._utc_offset is not None and site.manager.configured
                and float(self._utc_offset) != float(site.manager.utc_offset)):
            return (f"  WARNING: downloaded in {self._tz_label()} but the project "
                    f"timezone is UTC{'+' if site.manager.utc_offset >= 0 else ''}"
                    f"{site.manager.utc_offset} — re-download with the matching offset "
                    f"in the Database explorer, or the merged data will be time-shifted.")
        return ""

    def on_data_loaded(self, df, created=None) -> None:
        # The screening source is the staged DB field, NOT the working dataset;
        # keep the dataset's index/columns (overlap + collision checks) and its
        # resolution (the resample target the downloaded data is compared to).
        self._dataset_index = df.index if df is not None else None
        self._dataset_columns = set(map(str, df.columns)) if df is not None else set()
        self._target_freq = self._freq_str(df.index) if df is not None else None
        self.corrections_panel.set_coords_available(site.manager.configured)

    # --- emit: resample the screened series, END -> MIDDLE, with tags ---
    def _emit_frame(self):
        if self._df is None or self._var is None:
            return None
        field = self._var
        final = self._corrected if self._corrected is not None else self._base_series()
        if final is None or final.dropna().empty:
            return None

        freq = self._freq.currentText().strip() or "30min"
        series_end = final.copy()
        series_end.index.name = "TIMESTAMP_END"
        try:
            resampled_end = resample_series_to_freq(
                series_end, to_freqstr=freq, agg=self._agg.currentText(),
                mincounts_perc=self._mincounts.value())
        except Exception as err:
            self.status.setText(f"Resampling failed: {err}")
            return None
        if resampled_end.dropna().empty:
            return None
        resampled_end.index.name = "TIMESTAMP_END"
        # END -> MIDDLE so the column aligns onto diive's TIMESTAMP_MIDDLE index.
        series_mid = convert_series_timestamp_to_middle(resampled_end)

        name = self._unique_name(field)
        series_mid.name = name

        db_tags = self._tags.get(field, {})
        units = db_tags.get("units", "")
        data_version = db_tags.get("data_version", "")
        history = {
            "source": "InfluxDB",
            "bucket": self._db_meta.get("bucket"),
            "measurement": self._db_meta.get("measurement"),
            "db_field": field,
            "utc_offset": self._utc_offset,
            "timezone": self._tz_label(),
            "resample_freq": freq,
            "resample_agg": self._agg.currentText(),
            **db_tags,
        }
        pills = [t for t in ["from-database", "meteo-screened", units, data_version] if t]
        out = pd.DataFrame({name: series_mid})
        out.attrs[ATTRS_KEY] = {
            name: provenance_attr(
                origin=DERIVED, parent=None, operation=self.provenance_op,
                params=history, tags=pills),
        }
        return out

    def _unique_name(self, field: str) -> str:
        """A column name not already in the working dataset (numeric suffix)."""
        if field not in self._dataset_columns:
            return field
        i = 1
        while f"{field}_{i}" in self._dataset_columns:
            i += 1
        return f"{field}_{i}"

    # --- add: overlap guard + meaningful status ---
    def _add_to_dataset(self) -> None:
        if self._result_df is None or self._result_df.empty:
            return
        name = str(self._result_df.columns[0])
        series = self._result_df[name]
        n_total = int(series.count())

        overlap = None
        if self._dataset_index is not None and len(self._dataset_index):
            overlap = int(series.index.isin(self._dataset_index).sum())
            if overlap == 0:
                ds = self._dataset_index
                self.status.setText(
                    f"Not added: the resampled data "
                    f"({series.index.min():%Y-%m-%d %H:%M} to "
                    f"{series.index.max():%Y-%m-%d %H:%M}, {self._freq.currentText().strip()}) "
                    f"does not overlap the working dataset "
                    f"({ds.min():%Y-%m-%d %H:%M} to {ds.max():%Y-%m-%d %H:%M}). "
                    f"Download the meteo for the same period as your data.")
                return

        self.featuresCreated.emit(self._result_df)
        msg = f"Added {name} ({n_total} records) to the dataset"
        if name != self._var:
            msg = f"Added {name} (renamed from existing {self._var}, {n_total} records)"
        if overlap is not None and overlap < n_total:
            msg += (f"; {overlap} align with the dataset's timestamps, "
                    f"the rest fall outside its range (NaN)")
        self.status.setText(msg + ".")
        self.add_btn.setEnabled(False)
