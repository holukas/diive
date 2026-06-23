"""
TEST_GUI: smoke tests for the PySide6 desktop GUI
=================================================

Headless (offscreen) tests that the GUI builds and its core behaviours work:
tab structure, the shared variable list (filter + pills), multi-instance menu
tabs, live theme edits, the Overview stats, and diive-format parquet saving.

Run: pytest tests/test_gui.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import diive as dv


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def example_year():
    # One year of the example data: 10x fewer rows than the full record, so the
    # heatmap renders in the GUI tests are ~10x cheaper. Loaded/sliced once.
    df = dv.load_exampledata_parquet()
    return dv.times.keep_daterange(df, "2021-01-01", "2021-12-31 23:30")


@pytest.fixture
def window(app, monkeypatch, example_year):
    # Make the GUI use only one year of example data, including the constructor's
    # auto-load -- patch the loader the main window calls. A fresh copy per call
    # keeps tests isolated (feature engineering mutates the loaded frame).
    import diive
    monkeypatch.setattr(diive, "load_exampledata_parquet", lambda: example_year.copy())
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    yield win
    # Teardown: stop this window reacting to the app-wide event store. The store
    # is a process-wide singleton, so without this every window the suite creates
    # stays subscribed and they all re-render on the next events edit -- dozens of
    # accumulated matplotlib renders that can segfault. Also reset the store so
    # event state can't leak between tests.
    from diive.gui import events as _events
    try:
        _events.manager.changed.disconnect(win._on_events_changed)
    except (RuntimeError, TypeError):
        pass
    try:
        _events.manager.categories_changed.disconnect(
            win._on_event_categories_changed)
    except (RuntimeError, TypeError):
        pass
    try:
        _events.manager.focus_requested.disconnect(win._focus_event_on_overview)
    except (RuntimeError, TypeError):
        pass
    _events.manager.events.clear()
    _events.manager.visible = True
    _events.manager.categories = dict(_events.DEFAULT_CATEGORIES)


def _tabs(win):
    tw = win._tabwidget
    return [tw.tabText(i) for i in range(tw.count())]


def test_default_tabs(window):
    assert _tabs(window) == ["Overview", "Log"]


def test_example_data_autoloaded(window):
    assert window._data is not None
    assert window._data.shape[1] > 0


def test_variable_panel_shared_and_filter(window):
    overview, log = window._tabs[0], window._tabs[1]
    # Overview has a shared VariablePanel populated with all columns.
    from diive.gui.widgets.variable_panel import VariablePanel
    assert isinstance(overview.varpanel, VariablePanel)
    n = overview.varpanel.list.count()
    assert n == window._data.shape[1]
    # Separator-insensitive subsequence filter: 'gpp16' -> GPP_CUT_16_f etc.
    overview.varpanel.search.setText("gpp16")
    window.show()
    QApplication.processEvents()
    visible = [overview.varpanel.list.item(i)
               for i in range(n) if not overview.varpanel.list.item(i).isHidden()]
    assert visible and all("GPP" in it.data(Qt.ItemDataRole.UserRole) for it in visible)


def test_pill_classification():
    from diive.gui.widgets.variable_delegate import _pill_for
    assert _pill_for("GPP_CUT_REF_f")[0] == "GPP"
    assert _pill_for("NEE_CUT_REF_f")[0] == "NEE"
    assert _pill_for("PPFD")[0] == "PPFD"           # bare PPFD now tags
    assert _pill_for("TA_f")[0] == "TA"             # air temperature
    assert _pill_for("Tair_f")[0] == "TA"
    assert _pill_for("VPD_f")[0] == "VPD"
    assert _pill_for("SWC_FF0_0.15_1")[0] == "SWC"
    # Methane / nitrous oxide / water-vapour fluxes (bare and suffixed).
    assert _pill_for("FCH4")[0] == "FCH4"
    assert _pill_for("FCH4_orig")[0] == "FCH4"
    assert _pill_for("FN2O_f")[0] == "FN2O"
    assert _pill_for("FH2O")[0] == "FH2O"
    assert _pill_for("FC")[0] == "FC"               # FC is not caught by FCH4
    assert _pill_for("RH_f") is None                # unrecognised -> no pill


def test_multi_instance_plot_tabs(window):
    window._open_menu_tab("Heatmap date/time")
    window._open_menu_tab("Heatmap date/time")
    window._open_menu_tab("Time series")
    assert "Heatmap date/time 1" in _tabs(window)
    assert "Heatmap date/time 2" in _tabs(window)
    assert "Time series 1" in _tabs(window)


def test_close_tab_focuses_previous_not_log(window):
    tw = window._tabwidget
    # Layout: Overview(0), Log(1), then menu tabs appended after.
    window._open_menu_tab("Heatmap date/time")  # index 2
    window._open_menu_tab("Time series")        # index 3
    assert _tabs(window) == ["Overview", "Log", "Heatmap date/time 1", "Time series 1"]

    # Closing the last tab falls back to the tab on its left (the heatmap).
    tw.setCurrentIndex(3)
    window._on_tab_close(3)
    assert tw.tabText(tw.currentIndex()) == "Heatmap date/time 1"

    # Closing the only remaining menu tab would land on Log -> redirect to Overview.
    window._on_tab_close(2)
    assert tw.tabText(tw.currentIndex()) == "Overview"


def test_tabs_movable_renamable_and_close_buttons(window):
    tw = window._tabwidget
    bar = tw.tabBar()
    from PySide6.QtWidgets import QTabBar
    right = QTabBar.ButtonPosition.RightSide
    # Drag-to-reorder is enabled.
    assert tw.isMovable()
    # Always-on Overview/Log have no close button (they stay open).
    for i in range(tw.count()):
        assert bar.tabButton(i, right) is None

    # A menu tab gets a (custom, visible) close button.
    window._open_menu_tab("Time series")
    idx = tw.currentIndex()
    assert bar.tabButton(idx, right) is not None

    # Rename changes the display label only.
    tw.setTabText(idx, "My series")  # what _rename_tab does after the dialog
    assert tw.tabText(idx) == "My series"

    # The custom button closes its tab regardless of current order.
    n_before = tw.count()
    bar.tabButton(idx, right).click()
    assert tw.count() == n_before - 1
    assert "Overview" in _tabs(window) and "Log" in _tabs(window)


def test_plot_settings_live_render(window):
    from diive.gui.widgets.plot_settings import HEATMAP, PlotSettingsPanel
    window._open_menu_tab("Heatmap date/time")
    tab = window._menu_tab_list[-1]
    # Settings panel is present and exposes the heatmap parameter set.
    assert isinstance(tab.settings, PlotSettingsPanel)
    vals = tab.settings.values()
    assert {"cmap", "vmin", "vmax", "ax_orientation"} <= set(vals)
    assert tab._panels  # default variable rendered

    def _fallback(tab):
        # _draw_one writes a "Cannot plot ..." text into the axes on failure.
        return [t for ax in tab.canvas.fig.axes for t in ax.texts
                if "Cannot plot" in t.get_text()]

    # Toggle a spread of non-default settings; params apply only on "Update
    # plot", after which the heatmap must still draw (no error-fallback text).
    tab.settings.cmap.setCurrentText("viridis")
    tab.settings.orientation.setCurrentText("horizontal")
    tab.settings.vmin.setText("-5")
    tab.settings.vmax.setText("5")
    tab.settings.show_values.setChecked(True)
    tab.settings.cb_extend.setCurrentText("both")
    tab.settings.fmt_axlabel_fs.setValue(8)
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab.settings.values()["cmap"] == "viridis"
    assert not _fallback(tab)

    # Reverse-colormap toggle (heatmap): appends/strips the _r suffix.
    tab.settings.cmap.setCurrentText("viridis")
    tab.settings.reverse_cmap.setChecked(True)
    assert tab.settings.values()["cmap"] == "viridis_r"
    tab.settings.reverse_cmap.setChecked(False)
    assert tab.settings.values()["cmap"] == "viridis"

    window._open_menu_tab("Time series")
    ts = window._menu_tab_list[-1]
    ts_vals = ts.settings.values()
    assert {"linewidth", "alpha", "marker", "drop_gaps", "markersize", "_format",
            "_axes"} <= set(ts_vals)
    assert "title" in ts_vals["_format"]  # chrome now in the shared Format group
    ts.settings.marker.setChecked(True)
    ts.settings.drop_gaps.setChecked(True)
    ts.settings.linewidth.setValue(4.0)
    ts.settings.markersize.setValue(6.0)
    ts.settings.fmt_yunits.setText("umol")
    ts.settings.fmt_title.setText("Custom title")
    ts.update_btn.click()  # params apply on the button, not on edit
    QApplication.processEvents()
    assert not _fallback(ts)
    # Explicit title overrides the variable-name default.
    assert any(a.get_title() == "Custom title" for a in ts.canvas.fig.axes)

    # GUI-only Axes pass: log Y + a Y limit take effect on the data axis.
    ts.settings.ax_logy.setChecked(True)
    ts.settings.ax_ymin.setText("1")
    ts.settings.ax_ymax.setText("100")
    ts.update_btn.click()
    QApplication.processEvents()
    tax = ts.canvas.fig.axes[0]
    assert tax.get_yscale() == "log"
    assert {round(tax.get_ylim()[0]), round(tax.get_ylim()[1])} == {1, 100}
    assert not _fallback(ts)

    # Year/month heatmap: distinct settings (aggregation + ranks) and renders.
    window._open_menu_tab("Heatmap year/month")
    ym = window._menu_tab_list[-1]
    # Controls carry their library plot() docstring as a tooltip.
    assert "Colormap" in tab.settings.cmap.toolTip()

    vals = ym.settings.values()
    assert {"agg", "ranks", "cmap"} <= set(vals)
    assert vals["cmap"] is None  # default "auto"
    assert ym._panels and not _fallback(ym)
    ym.settings.agg.setCurrentText("sum")
    ym.settings.ranks.setChecked(True)
    ym.settings.orientation.setCurrentText("horizontal")
    ym.update_btn.click()
    QApplication.processEvents()
    assert not _fallback(ym)


def test_windrose_tab(app):
    # The wind rose is role-picked (value + wind direction required, colour
    # optional) and polar. The bundled CH-DAV example has no wind direction, so
    # build a synthetic frame carrying one.
    import numpy as np
    import pandas as pd
    from diive.gui.tabs.plotting import PlottingTab
    from diive.gui.widgets.plot_settings import WINDROSE

    idx = pd.date_range("2021-01-01", periods=2000, freq="30min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "co2_flux": rng.normal(-2, 5, len(idx)),
        "wind_dir": rng.uniform(0, 360, len(idx)),
        "air_temperature": rng.normal(285, 4, len(idx)),
    }, index=idx)

    def _fallback(tab):
        return [t for ax in tab.canvas.fig.axes for t in ax.texts
                if "Cannot plot" in t.get_text()]

    tab = PlottingTab(WINDROSE, "Wind rose")
    tab.widget()
    tab.on_data_loaded(df, set())
    QApplication.processEvents()

    # Settings expose the wind-rose parameter set; defaults are seeded.
    vals = tab.settings.values()
    assert {"agg", "n_sectors", "z_agg", "cmap", "show_colorbar"} <= set(vals)
    # Value + direction auto-seeded -> a polar axes + colorbar, no error fallback.
    assert tab._xyz == ["co2_flux", "wind_dir"]
    assert any(ax.name == "polar" for ax in tab.canvas.fig.axes)
    assert not _fallback(tab)

    # Add the optional colour variable and change the aggregation.
    tab._on_selected("air_temperature", True)
    tab.settings.wr_agg.setCurrentText("median")
    tab.settings.wr_nsectors.setValue(16)
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab._xyz == ["co2_flux", "wind_dir", "air_temperature"]
    assert not _fallback(tab)

    # Hiding the colorbar drops the extra axes (only the polar axes remains).
    tab.settings.wr_show_colorbar.setChecked(False)
    tab.update_btn.click()
    QApplication.processEvents()
    assert len(tab.canvas.fig.axes) == 1
    assert not _fallback(tab)


def test_params_apply_only_on_update_button(window):
    # Editing a parameter must NOT re-render; only the "Update plot" button
    # applies pending changes. (Variable selection still renders live — covered
    # elsewhere.) Use a spinbox, which DOES emit `changed`, to prove the tab no
    # longer renders on that signal.
    import numpy as np
    window._open_menu_tab("Time series")
    tab = window._menu_tab_list[-1]
    QApplication.processEvents()

    def _main_line(tab):
        return max((l for l in tab.canvas.fig.axes[0].get_lines()),
                   key=lambda l: np.asarray(l.get_xdata()).size)

    before = _main_line(tab).get_linewidth()
    tab.settings.linewidth.setValue(before + 4.0)  # emits `changed`
    QApplication.processEvents()
    assert _main_line(tab).get_linewidth() == before  # not applied yet
    tab.update_btn.click()
    QApplication.processEvents()
    assert _main_line(tab).get_linewidth() != before  # button applied it


def test_keep_daterange_library():
    df = dv.load_exampledata_parquet()
    sub = dv.times.keep_daterange(df, "2021-06-01", "2021-06-30 23:30")
    assert sub.index.min() >= pd.Timestamp("2021-06-01")
    assert sub.index.max() <= pd.Timestamp("2021-06-30 23:30")
    assert len(sub) < len(df)
    assert df.shape[0] == 175296  # original untouched (non-destructive)
    # Open bounds and invalid order.
    assert dv.times.keep_daterange(df, start="2022-01-01").index.min() >= pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError):
        dv.times.keep_daterange(df, "2022-01-01", "2021-01-01")


def test_gui_daterange_subselection(window):
    full_shape = window._full_data.shape
    assert window._data is window._full_data  # full range initially
    assert not window._reset_range_act.isEnabled()
    # Narrow to one month (non-destructive: full record kept in the background).
    window._range = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30 23:30"))
    window._apply_range()
    QApplication.processEvents()
    assert window._data.shape[0] < full_shape[0]
    assert window._full_data.shape == full_shape  # original intact
    assert window._reset_range_act.isEnabled()
    # Tabs see the narrowed data.
    assert window._tabs[0].varpanel.list.count() == window._data.shape[1]
    # Reset restores the full range.
    window._reset_range()
    QApplication.processEvents()
    assert window._data.shape == full_shape
    assert not window._reset_range_act.isEnabled()


def test_daterange_dialog_clamps_and_orders(app):
    from diive.gui.widgets.daterange_dialog import DateRangeDialog
    start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31 23:30")
    dlg = DateRangeDialog(start, end)
    # Picks default to the full span and stay within bounds.
    got_start, got_end = dlg.selected_range()
    assert got_start >= start and got_end <= end


def test_overview_layout_stable_on_zoom(window):
    # The canvas freezes constrained layout after rendering so zoom/pan don't
    # continuously reflow the panels. The heatmap colorbar can trigger a one-time
    # settle over the first couple of zooms (full-range vs zoomed tick-label
    # density), after which the layout must be STABLE — verify repeated zooms no
    # longer move any panel.
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    QApplication.processEvents()
    fig = overview.canvas.fig
    ts_ax = fig.axes[0]

    def zoom_step(lo, hi):
        x0, x1 = ts_ax.get_xlim()
        ts_ax.set_xlim(x0 + (x1 - x0) * lo, x0 + (x1 - x0) * hi)
        overview.canvas.draw()
        QApplication.processEvents()

    # A couple of settling zooms, then assert the layout no longer shifts.
    zoom_step(0.3, 0.7)
    zoom_step(0.3, 0.7)
    before = [tuple(a.get_position().bounds) for a in fig.axes]
    zoom_step(0.25, 0.75)
    after = [tuple(a.get_position().bounds) for a in fig.axes]
    assert before == after  # stable after the initial settle (no perpetual jitter)

    # A resize must re-solve the frozen layout to the new size (otherwise a layout
    # computed at the tiny pre-show size stays collapsed). After a resize the
    # bottom panels should have a sensible (non-collapsed) width.
    fig.set_size_inches(20, 11)
    overview.canvas._on_resize(None)
    QApplication.processEvents()
    bottom_widths = [a.get_position().width for a in fig.axes
                     if a.get_position().y0 < 0.45]
    assert sum(w > 0.1 for w in bottom_widths) >= 3


def test_hover_value_lookup(app, example_year):
    import types
    import numpy as np
    from diive.gui.widgets.hover import HoverAnnotator
    from diive.gui.widgets.mpl_canvas import MplCanvas

    series = example_year["Tair_f"]  # one year -> cheaper heatmap mesh

    # Line panel: snaps to the nearest sample and reports its value.
    canvas = MplCanvas()
    assert isinstance(canvas.hover, HoverAnnotator)
    # Toolbar coordinate readout is hidden; a "Hover values" toggle controls it.
    assert canvas._toolbar.coordinates is False
    assert canvas._hover_toggle.isChecked() and canvas.hover._enabled
    canvas._hover_toggle.setChecked(False)
    assert canvas.hover._enabled is False
    canvas._hover_toggle.setChecked(True)
    ax = canvas.new_axes(1)[0]
    dv.plotting.TimeSeries(series).plot(ax=ax)
    canvas.draw()
    line = next(l for l in ax.get_lines() if np.asarray(l.get_xdata()).size > 2)
    x = np.asarray(line.get_xdata(orig=False), float)
    y = np.asarray(line.get_ydata(orig=False), float)
    i = 5000
    px, py = ax.transData.transform((x[i], y[i]))
    ev = types.SimpleNamespace(inaxes=ax, xdata=x[i], ydata=y[i], x=px, y=py)
    hx, hy, text, marker = canvas.hover._value_at(ax, ev)
    assert marker is True
    assert abs(hy - y[i]) < 1e-9
    assert f"{y[i]:.4g}" in text

    # Heatmap panel: reads the cell under the cursor.
    canvas2 = MplCanvas()
    ax2 = canvas2.new_axes(1)[0]
    dv.plotting.HeatmapDateTime(series).plot(
        ax=ax2, fig=canvas2.fig, cb_digits_after_comma="auto")
    canvas2.draw()
    qm = next(c for c in ax2.collections if c.__class__.__name__ == "QuadMesh")
    xb, yb, vals = canvas2.hover._mesh_grid(qm)
    xc = 0.5 * (xb[10] + xb[11])
    yc = 0.5 * (yb[5] + yb[6])
    ev2 = types.SimpleNamespace(inaxes=ax2, xdata=xc, ydata=yc, x=0, y=0)
    _, _, text2, marker2 = canvas2.hover._heatmap_value(ax2, qm, ev2)
    assert marker2 is False
    assert f"{float(vals[5, 10]):.4g}" in text2


def test_save_dpi_spinbox(app):
    # The canvas exposes a Save-DPI spinbox (default 150) whose value the
    # Save action passes through to savefig.
    import matplotlib as mpl
    from diive.gui.widgets.mpl_canvas import MplCanvas
    canvas = MplCanvas()
    assert canvas.save_dpi() == 150
    canvas._dpi_spin.setValue(300)
    assert canvas.save_dpi() == 300
    # The toolbar's save_figure sets savefig.dpi from the spinbox while saving.
    captured = {}

    def fake_save(*args, **kwargs):
        captured["dpi"] = mpl.rcParams["savefig.dpi"]

    canvas._canvas.figure.savefig = fake_save
    # Drive the toolbar override directly (no file dialog): patch the base save.
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    orig = NavigationToolbar2QT.save_figure
    NavigationToolbar2QT.save_figure = lambda self, *a: mpl.rcParams["savefig.dpi"]
    try:
        result = canvas._toolbar.save_figure()
    finally:
        NavigationToolbar2QT.save_figure = orig
    assert result == 300  # the elevated DPI was active during the base save
    assert mpl.rcParams["savefig.dpi"] != 300  # restored afterwards


def test_feature_engineer_index_only(window):
    import time
    # Timestamp + record-number features need no selected variables (they derive
    # from the index); created features are listed explicitly.
    window._open_menu_tab("Feature engineering")
    tab = window._menu_tab_list[-1]
    assert tab.selected.count() == 0  # nothing selected
    tab.ts_cb.setChecked(True)
    tab.rec_cb.setChecked(True)
    tab._run()
    for _ in range(200):
        QApplication.processEvents()
        time.sleep(0.02)
        if tab._created_df is not None:
            break
    assert tab._created_df is not None
    n = len(tab._created_df.columns)
    assert n > 0
    assert tab.created_list.count() == n  # explicit "newly created" list populated
    assert tab.add_btn.isEnabled()


def test_flux_chain_tab_level31(app):
    # The flux chain needs real EddyPro-FLUXNET input (FC, USTAR, *_TEST cols),
    # so load the dedicated EC example dataset (one month for speed).
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()  # build
    tab.on_data_loaded(df)
    assert tab.fluxcol.currentText() == "FC"  # default flux column detected
    # Params carry their library docstring as a tooltip.
    assert "latitude" in tab.site_lat.toolTip().lower()

    # Copy-Python emits runnable composable code matching what Run does (through L3.1).
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "init_flux_data(" in code and "run_level2(" in code and "run_level31(" in code

    # Run Level 2 -> Level 3.1 (synchronous core) and render into the canvas.
    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(), tab._level31_kwargs())
    assert data.filteredseries is not None
    assert data.filteredseries.dropna().count() > 0
    # The storage-corrected L3.1 column exists in the working dataframe.
    assert any("L3.1" in str(c) for c in data.fpc_df.columns)
    tab._on_done(data)
    QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]


def test_flux_chain_tab_level32(app):
    # L3.2 chain: add outlier steps via the registry, run through make_level32_detector
    # + run_level32, and surface the overall QCF separately.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams, ZScoreParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)
    # Build a two-step L3.2 chain (as the picker would).
    tab._steps = [HampelParams().step(), ZScoreParams().step()]
    tab._update_run_label()
    assert "3.2" in tab.run_btn.text()

    # Copy-Python now renders the L3.2 composable chain.
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "make_level32_detector" in code and "run_level32(" in code
    assert code.count("sod.addflag()") == 2

    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps)
    assert getattr(data.levels, "level32_qcf", None) is not None
    assert any("L3.2" in str(c) for c in data.fpc_df.columns)
    tab._on_done(data)
    QApplication.processEvents()
    assert "QCF" in tab.summary.toPlainText()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]

    # Steps round-trip through save/restore (project persistence).
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2._steps == tab._steps
    assert tab2.l32_steps_list.count() == 2


def test_stepwise_screening_tab(app):
    # Chain outlier tests on one variable: per-step removals, a separate QCF, and
    # commit of flags + QCF + filtered series.
    import diive as dv
    from diive.gui.tabs.stepwise import StepwiseScreeningTab
    from diive.gui.widgets.stepwise_method_params import HampelParams, ZScoreParams

    df = dv.variables.generate_noisy_timeseries(
        start_date="2024-01-01", periods=48 * 20, freq="30min",
        trend_slope=0.01, seasonal_strength=9, noise_level=2, outlier_fraction=0.1)
    df.index.name = "TIMESTAMP_END"

    tab = StepwiseScreeningTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._select("observed_value")

    emitted = {}
    tab.featuresCreated.connect(lambda d: emitted.update(df=d))

    tab._steps = [HampelParams().step(), ZScoreParams().step()]
    # Drive the worker synchronously (signals deliver in-thread under the test app).
    # configured=True so SW_IN_POT is computed and the report gets day/night.
    tab._worker(df, "observed_value", tab._steps, tab._coords(), True, tab._run_id)
    QApplication.processEvents()

    # Cards mirror the chain; the run carries per-step removals + detection bounds.
    assert len(tab._step_cards) == 2
    assert len(tab._payload["removed"]) == 2
    assert len(tab._payload["bounds"]) == 2
    # The screening report is built (overall + day/night) and shown in the panel.
    report = tab._payload["report"]
    assert "STEPWISE SCREENING REPORT" in report
    assert "OVERALL" in report and "DAYTIME" in report and "NIGHTTIME" in report
    assert tab.report_text.toPlainText() == report
    assert tab.report_copy_btn.isEnabled()

    # A stale worker result (run id != current) is ignored, so out-of-order
    # thread completions can't overwrite the payload with the wrong run.
    good = tab._payload
    tab._on_done({"run_id": tab._run_id + 1})
    assert tab._payload is good
    # Per-step removal counts must sum to the overall total: addflag mutates the
    # cleaned series in place, so a missing per-step copy would alias it and make
    # later steps report 0 removed even though points were dropped.
    det = tab._payload["detector"]
    total = int(det.series_hires_orig.notna().sum() - det.series_hires_cleaned.notna().sum())
    assert sum(len(idx) for idx in tab._payload["removed"]) == total
    assert "QCF" in tab.qcf_label.text()
    assert tab.add_btn.isEnabled()
    assert tab.copy_btn.isEnabled()
    # Selecting a card re-renders (with the limit band on) without error.
    tab.limits_cb.setChecked(True)
    tab._select_step(0)
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]

    # The Copy-Python provider returns a valid, runnable script.
    code = tab._code_provider()
    compile(code, "<gen>", "exec")

    # Editing a step in place round-trips its kwargs back into the param widget.
    edited = HampelParams()
    edited.load(tab._steps[0]["kwargs"])
    assert edited.kwargs()["n_sigma"] == tab._steps[0]["kwargs"]["n_sigma"]

    # The step editor builds exactly one param form (no ghost form left behind,
    # the cause of the overlapping-labels bug) and seeds it from the step.
    from diive.gui.widgets.stepwise_cards import StepEditorDialog
    dlg = StepEditorDialog(step=tab._steps[0])
    assert dlg._param_box.count() == 1
    assert dlg.step()["method"] == tab._steps[0]["method"]
    dlg.deleteLater()

    # Commit emits the flags + a clean QCF flag + the QCF-filtered series, index-aligned.
    tab._add_to_dataset()
    cols = list(emitted["df"].columns)
    assert any(str(c).endswith("_STEPWISE_QCF") for c in cols)          # filtered series
    assert any(str(c) == "FLAG_STEPWISE_observed_value_QCF" for c in cols)  # overall flag
    assert sum(str(c).endswith("_TEST") for c in cols) == 2             # one flag per step
    assert emitted["df"].index.equals(df.index)                        # aligns on merge

    # Toggling a step off skips it in the chain: it contributes no removals, the
    # per-step lists stay aligned to the cards, and the total still adds up.
    tab._steps[0]["enabled"] = False
    tab._worker(df, "observed_value", tab._steps, tab._coords(), True, tab._run_id)
    QApplication.processEvents()
    assert len(tab._payload["removed"]) == 2
    assert len(tab._payload["removed"][0]) == 0           # disabled step removes nothing
    det = tab._payload["detector"]
    total = int(det.series_hires_orig.notna().sum() - det.series_hires_cleaned.notna().sum())
    assert sum(len(idx) for idx in tab._payload["removed"]) == total


def test_stepwise_screening_corrections(app):
    # The corrections phase: the measurement is auto-detected from the variable
    # name and gates which corrections appear (radiation zero offset only for SW/
    # PPFD); enabled corrections produce a corrected column and feed the script.
    import diive as dv
    from diive.gui.tabs.stepwise import StepwiseScreeningTab
    from diive.gui.widgets.stepwise_method_params import ZScoreParams
    from diive.gui import site

    site.manager.update(author="t", description="", name="X", latitude=47.4,
                        longitude=8.5, elevation=500, utc_offset=1)

    df = dv.variables.generate_noisy_timeseries(
        start_date="2024-06-01", periods=48 * 10, freq="30min", trend_slope=0.0,
        seasonal_strength=5, noise_level=1, outlier_fraction=0.05)
    df = df.rename(columns={"observed_value": "SW_IN_T1_2_1"})
    df.index.name = "TIMESTAMP_END"

    tab = StepwiseScreeningTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._select("SW_IN_T1_2_1")

    # SW is detected from the name; the radiation zero-offset correction (only
    # meaningful for radiation) is offered, and the generic ones too.
    assert tab.meas_combo.currentData() == "SW"
    assert tab.corrections_panel.measurement() == "SW"
    assert "radiation_zero_offset" in tab.corrections_panel._rows
    assert "setto_max" in tab.corrections_panel._rows

    emitted = {}
    tab.featuresCreated.connect(lambda d: emitted.update(df=d))

    # Corrections run standalone (no outlier steps) on the raw series. They apply
    # only on Run — editing the rows just marks the run pending.
    tab.corrections_panel._rows["radiation_zero_offset"].enable.setChecked(True)
    rmax = tab.corrections_panel._rows["setto_max"]
    rmax.enable.setChecked(True)
    rmax.threshold.setValue(800)
    assert tab._corrected is None                          # not applied until Run
    assert tab.run_corrections_btn.text().endswith("•")    # pending indicator
    tab.run_corrections_btn.click()
    QApplication.processEvents()
    assert tab._corrected is not None
    assert list(tab._result_df.columns) == ["SW_IN_T1_2_1_CORRECTED"]
    assert tab.add_btn.isEnabled()
    assert tab.run_corrections_btn.text().endswith("•") is False  # cleared after run
    # No outlier steps -> no reproducible chain script yet.
    assert tab.copy_btn.isEnabled() is False

    # With an outlier step, the QCF columns + the corrected column are all
    # emitted, and the generated script includes the corrections block.
    tab._steps = [ZScoreParams().step()]
    tab._worker(df, "SW_IN_T1_2_1", tab._steps, tab._coords(), True, tab._run_id)
    QApplication.processEvents()
    cols = list(tab._result_df.columns)
    assert any(c.endswith("_CORRECTED") for c in cols)
    assert any(c.endswith("_STEPWISE_QCF") for c in cols)
    code = tab._code_provider()
    compile(code, "<gen>", "exec")
    assert "corrected = cleaned.copy()" in code
    assert "remove_nighttime_zero_offset" in code

    # Switching to a non-radiation measurement drops the radiation correction.
    idx = tab.meas_combo.findData("TA")
    tab.meas_combo.setCurrentIndex(idx)
    QApplication.processEvents()
    assert "radiation_zero_offset" not in tab.corrections_panel._rows
    assert "setto_max" in tab.corrections_panel._rows


def test_correction_tabs(app):
    # The standalone correction tabs (RF/XGB-style shared template): each is one
    # correction on a selected variable, producing a corrected column + provenance,
    # with a reproducible script. Available for any variable (no measurement lock).
    import diive as dv
    from diive.core.metadata import ATTRS_KEY
    from diive.gui.tabs.corrections_nighttime_offset import NighttimeZeroOffsetTab
    from diive.gui.tabs.corrections_setto_threshold import SetToMaxThresholdTab
    from diive.gui.tabs.corrections_set_missing import SetExactToMissingTab
    from diive.gui import site

    site.manager.update(author="t", description="", name="X", latitude=47.4,
                        longitude=8.5, elevation=500, utc_offset=1)

    df = dv.variables.generate_noisy_timeseries(
        start_date="2024-06-01", periods=48 * 10, freq="30min", trend_slope=0.0,
        seasonal_strength=5, noise_level=1, outlier_fraction=0.0)
    df = df.rename(columns={"observed_value": "SW_IN_T1_2_1"})
    df.index.name = "TIMESTAMP_END"

    # --- Nighttime zero offset: needs coords (seeded from the site) ---
    tab = NighttimeZeroOffsetTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._select("SW_IN_T1_2_1")
    assert tab.needs_coords and tab.lat.value() == pytest.approx(47.4)

    emitted = {}
    tab.featuresCreated.connect(lambda d: emitted.update(df=d))
    tab._worker(df["SW_IN_T1_2_1"], tab._current_kwargs(),
                tab.lat.value(), tab.lon.value(), tab.utc.value())
    QApplication.processEvents()
    assert list(tab._result_df.columns) == ["SW_IN_T1_2_1_NIGHTOFFSET"]
    assert "SW_IN_T1_2_1_NIGHTOFFSET" in tab._result_df.attrs[ATTRS_KEY]
    assert tab.add_btn.isEnabled()
    # Diagnostics power the multi-panel preview + below-zero hero stats; after
    # the correction no records remain below zero (night forced to zero + clamp).
    e = tab._last_payload["extra"]
    assert {"offset", "corrected_by_offset", "n_below_zero_after",
            "n_below_zero_after_night"} <= e.keys()
    assert e["n_below_zero_after"] == 0
    assert e["n_below_zero_after_night"] == 0
    # clamp_negatives is on by default and omitted from the script; turning it off
    # surfaces in both the kwargs and the generated code.
    assert tab.clamp_cb.isChecked() and tab._current_kwargs() == {"clamp_negatives": True}
    assert "clamp_negatives" not in tab._python_code()
    tab.clamp_cb.setChecked(False)
    assert "clamp_negatives=False" in tab._python_code()
    tab.clamp_cb.setChecked(True)
    tab._add()
    QApplication.processEvents()
    assert "SW_IN_T1_2_1_NIGHTOFFSET" in emitted["df"].columns
    code = tab._python_code()
    compile(code, "<gen>", "exec")
    assert "remove_nighttime_zero_offset" in code  # general-purpose library fn

    # --- Set to max threshold: a generic, parameterized correction ---
    tmax = SetToMaxThresholdTab()
    tmax.widget()
    tmax.on_data_loaded(df)
    tmax._select("SW_IN_T1_2_1")
    tmax.threshold.setValue(5.0)
    tmax._worker(df["SW_IN_T1_2_1"], tmax._current_kwargs(), None, None, None)
    QApplication.processEvents()
    corrected = tmax._result_df["SW_IN_T1_2_1_SETMAX"]
    assert corrected.max() <= 5.0 + 1e-9            # capped
    assert "setto_threshold" in tmax._python_code()

    # --- Validation: set-exact-to-missing needs values before it runs ---
    miss = SetExactToMissingTab()
    miss.widget()
    miss.on_data_loaded(df)
    miss._select("SW_IN_T1_2_1")
    miss._run()                                      # no values entered
    assert miss._result_df is None
    assert "value" in miss.status.text().lower()


def test_flux_chain_tab_level33_detection(app):
    # L3.3 supports auto-detecting the USTAR threshold (moving point bootstrap)
    # as an alternative to constant thresholds. The detected CUT percentiles
    # become USTAR scenarios; Copy Python renders run_level33_ustar_detection.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN()
    ta = [c for c in df.columns if c.upper().startswith("TA_")][0]
    sw = [c for c in df.columns if c.upper().startswith("SW_IN") and "POT" not in c.upper()][0]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)
    # TA must not auto-pick a USTAR column ("TA" is a substring of "USTAR").
    assert "USTAR" not in tab.l33_ta.currentText().upper()

    tab.l33_enable.setChecked(True)
    tab.l33_mode.setCurrentIndex(1)  # detect
    tab.l33_ta.setCurrentText(ta)
    tab.l33_swin.setCurrentText(sw)
    tab.l33_niter.setValue(15)
    tab._steps = [HampelParams().step()]

    kw = tab._level33_kwargs()
    assert kw and kw.get("_detection") is True
    assert kw["ta_col"] == ta and kw["swin_col"] == sw

    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "run_level33_ustar_detection(" in code
    assert "run_level33_constant_ustar(" not in code

    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps, kw)
    assert "CUT_50" in data.levels.filteredseries_level33_qcf
    assert getattr(data.levels, "ustar_detection", None) is not None

    # Mode round-trips through save/restore.
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2.l33_mode.currentIndex() == 1
    assert tab2.l33_ta.currentText() == ta


def test_flux_chain_tab_level33_vut(app):
    # L3.3 detection can apply per-year VUT thresholds instead of a constant CUT.
    # CUT and VUT are mutually exclusive strategies.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN()
    ta = [c for c in df.columns if c.upper().startswith("TA_")][0]
    sw = [c for c in df.columns if c.upper().startswith("SW_IN") and "POT" not in c.upper()][0]

    tab = FluxChainTab(); tab.widget(); tab.on_data_loaded(df)
    tab.l33_enable.setChecked(True)
    tab.l33_mode.setCurrentIndex(1)   # detect
    tab.l33_apply.setCurrentIndex(1)  # VUT (per-year)
    tab.l33_ta.setCurrentText(ta)
    tab.l33_swin.setCurrentText(sw)
    tab.l33_niter.setValue(10)
    tab._steps = [HampelParams().step()]

    kw = tab._level33_kwargs()
    assert kw.get("mode") == "vut"
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "mode='vut'" in code

    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps, kw)
    scen = list(data.levels.filteredseries_level33_qcf)
    assert scen and all(s.startswith("VUT_") for s in scen)


def test_ustar_detection_tab(app):
    # Standalone USTAR detection tab: single seasonal detection + multi-year
    # bootstrap, both via the library detectors, results into table + plot.
    from diive.gui.tabs.ustar_detection import UstarDetectionTab
    from diive.configs.exampledata import load_exampledata_parquet_lae
    df = load_exampledata_parquet_lae()

    tab = UstarDetectionTab()
    tab.widget()
    tab.on_data_loaded(df)
    # Auto-pick should land on real TA / USTAR / SW_IN columns (not cross-match).
    assert tab.ta_col.currentText().upper().startswith("TA")
    assert "USTAR" in tab.ustar_col.currentText().upper()
    assert tab.swin_col.currentText().upper().startswith("SW_IN")

    # Single detection (run the worker synchronously, then drain the signal).
    kwargs = dict(nee_col=tab.nee_col.currentText(), ta_col=tab.ta_col.currentText(),
                  ustar_col=tab.ustar_col.currentText(), swin_col=tab.swin_col.currentText(),
                  ta_classes_count=7, ustar_classes_count=20, forward_mode_n=2)
    tab._worker(df, kwargs, None)
    QApplication.processEvents()
    assert tab.table.rowCount() == 5  # 4 seasons + annual
    labels = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
    assert "Annual (max)" in labels

    # Bootstrap path (few iterations) produces a per-year + CUT table.
    tab._worker(df, kwargs, dict(n_iter=10, n_jobs=1, percentiles=(16, 50, 84)))
    QApplication.processEvents()
    labels = [tab.table.item(r, 0).text() for r in range(tab.table.rowCount())]
    assert "CUT (constant)" in labels
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "failed" in t.get_text().lower()]


def test_flux_chain_tab_level33(app):
    # L3.3 USTAR filtering: requires an L3.2 step, applies constant thresholds,
    # and exposes per-scenario QCFs.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Enabling L3.3 without an L3.2 step is rejected up front (no run).
    tab.l33_enable.setChecked(True)
    tab._ustar = [(0.1, "CUT_50")]
    tab._update_run_label()
    assert "3.3" in tab.run_btn.text()
    tab._run()
    assert "requires at least one Level 3.2" in tab.summary.toPlainText()

    # With an L3.2 step, the chain runs through L3.3 and exposes scenarios.
    tab._steps = [HampelParams().step()]
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "run_level33_constant_ustar(" in code

    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps, tab._level33_kwargs())
    assert getattr(data.levels, "level33_qcf", None)  # non-empty scenario dict
    assert "CUT_50" in data.levels.filteredseries_level33_qcf
    tab._on_done(data)
    QApplication.processEvents()
    assert "USTAR" in tab.summary.toPlainText()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]

    # USTAR thresholds round-trip through save/restore.
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2.l33_enable.isChecked()
    assert tab2._ustar == [(0.1, "CUT_50")]


def test_flux_chain_tab_pipeline_rail(app):
    # The chain tab is laid out as a pipeline rail (stage cards) + a stacked
    # inspector: selecting a card swaps the stage's controls, status pills reflect
    # the live config, and a run lights the reached cards.
    from diive.gui.tabs.fluxchain import FluxChainTab, _STAGES
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # One card per pipeline stage; selecting one drives the stacked inspector.
    assert len(tab.rail._cards) == len(_STAGES)
    tab._select_stage(3)
    assert tab._pages.currentIndex() == 3

    # Status pills derive from the live controls.
    statuses = tab._stage_statuses()
    assert statuses[0][0] == "FC"                      # input: flux column
    assert statuses[3] == ("none", "off")              # L3.2: no steps yet
    assert statuses[4] == ("off", "off")               # L3.3: disabled
    assert statuses[5] == ("off", "off")               # L4.1: no method

    # Configuring stages updates their pills (kind reflects active/off/warn).
    tab._steps = [HampelParams().step()]
    tab.l33_enable.setChecked(True)
    tab._ustar = [(0.1, "CUT_50")]
    tab.l41_mds.setChecked(True)
    tab._update_run_label()
    s = tab._stage_statuses()
    assert s[3] == ("1 step", "set")
    assert s[4] == ("1 scenario", "set")
    assert s[5][1] == "set" and "mds" in s[5][0]

    # A run lights every reached card with the green ✓ (through L3.2 here).
    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps)
    tab._on_done(data)
    QApplication.processEvents()
    assert tab.rail._cards[3]._reached          # L3.2 reached
    assert not tab.rail._cards[5]._reached      # L4.1 not run


def test_flux_chain_tab_level2_details(app):
    # The L2 page shows the variables each test reads, exposes the 8 VM97 sub-tests,
    # gates tests on input availability, and the run produces a QCF report.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.flux.fluxprocessingchain import VM97_SUBTESTS
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Vars header reflects the detected flux column + base variable.
    assert "FC" in tab.l2_header.text() and "CO2" in tab.l2_header.text()
    # Each test has a column picker seeded to the standard EddyPro column, and
    # the availability marker confirms it's present.
    assert tab.l2_cols["ssitc"][0].currentText() == "FC_SSITC_TEST"
    assert tab.l2_cols["gas_completeness"][1].currentText() == "CO2_NR"
    assert tab.l2_inputs["ssitc"].text().startswith("✓")
    # All 8 VM97 sub-tests are exposed; defaults are spikes + dropout.
    assert set(tab.l2_vm97_checks) == {k for k, _, _ in VM97_SUBTESTS}
    assert tab.l2_vm97_checks["spikes"].isChecked()
    assert not tab.l2_vm97_checks["amplitude"].isChecked()

    # Editing the VM97 sub-tests flows into the L2 settings (all 8 keys present);
    # the column equals the default, so no 'col' override is added.
    tab.l2_vm97_checks["amplitude"].setChecked(True)
    tab.l2_vm97_checks["dropout"].setChecked(False)
    vm97 = tab._level2_settings()["raw_data_screening_vm97"]
    assert vm97["apply"] is True
    assert vm97["amplitude"] is True and vm97["dropout"] is False
    assert set(vm97) == {"apply"} | {k for k, _, _ in VM97_SUBTESTS}

    # Picking a different SSITC column adds a 'col' override to that test's config.
    tab.l2_cols["ssitc"][0].setCurrentText("EXPECT_NR")
    assert tab._level2_settings()["ssitc"]["col"] == "EXPECT_NR"
    tab.l2_cols["ssitc"][0].setCurrentText("FC_SSITC_TEST")  # restore default
    assert "col" not in tab._level2_settings()["ssitc"]

    # Clearing a test's column disables it (and drops it from settings).
    tab.l2_cols["spectral_correction_factor"][0].setCurrentText("")
    assert not tab.l2_checks["spectral_correction_factor"].isEnabled()
    assert "spectral_correction_factor" not in tab._level2_settings()
    tab.l2_cols["spectral_correction_factor"][0].setCurrentText("FC_SCF")  # restore

    # Signal strength has no column chosen -> no entry.
    assert "signal_strength" not in tab._level2_settings()

    # Running L2 fills the QCF report panel (per-test retained/rejected breakdown).
    # Drive the per-level workers synchronously: init, then L2 on its output.
    tab._level_worker({"idx": 0, "kind": "init", "init_kwargs": tab._init_kwargs()},
                      None, df)
    QApplication.processEvents()
    tab._level_worker({"idx": 1, "kind": "level2", "settings": tab._level2_settings()},
                      tab._data, df)
    QApplication.processEvents()
    assert tab._reached == 1
    assert tab.report.toPlainText().strip()
    assert tab.report_copy.isEnabled()
    # The report breaks rejection down per test (QCF screening report).
    assert "OVERALL" in tab.report.toPlainText()

    # VM97 sub-tests + a custom column pick round-trip through save/restore.
    tab.l2_cols["ssitc"][0].setCurrentText("EXPECT_NR")
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2.l2_vm97_checks["amplitude"].isChecked()
    assert not tab2.l2_vm97_checks["dropout"].isChecked()
    assert tab2.l2_cols["ssitc"][0].currentText() == "EXPECT_NR"


def test_flux_chain_tab_level_info(app):
    # Every level page mirrors L2: it shows the column(s) it reads with an
    # availability marker, and lets you pick them (USTAR on Input, storage on
    # L3.1, MDS drivers on L4.1).
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Input: USTAR column picker, seeded to 'USTAR', flows into init kwargs.
    assert tab.ustarcol.currentText() == "USTAR"
    assert tab._init_kwargs()["ustarcol"] == "USTAR"
    assert tab.ustar_mark.text().startswith("✓")

    # L3.1: the storage marker shows the auto-detected column for the flux.
    assert "SC_SINGLE" in tab.strg_mark.text()       # FC -> SC_SINGLE
    assert tab.strg_mark.text().startswith("✓")

    # L3.3: an info line names the USTAR column it filters on.
    assert "USTAR" in tab.l33_info.text()

    # L4.1: the MDS driver marker reflects the three drivers' availability.
    assert "SW_IN" in tab.mds_mark.text() and "VPD" in tab.mds_mark.text()

    # A non-'USTAR'-named friction-velocity column can be picked and initializes.
    df2 = df.rename(columns={"USTAR": "FRICTION_VEL"})
    tab.on_data_loaded(df2)
    tab.ustarcol.setCurrentText("FRICTION_VEL")
    assert tab._init_kwargs()["ustarcol"] == "FRICTION_VEL"
    data = tab._compute(df2, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs())
    assert data.filteredseries.dropna().count() > 0

    # USTAR-column pick round-trips through save/restore.
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df2)
    tab2.restore_state(state)
    assert tab2.ustarcol.currentText() == "FRICTION_VEL"


def test_flux_chain_tab_level41_layout(app):
    # L4.1 settings are split into per-method sections shown only when the method
    # is enabled, and features use the searchable checkable picker.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # With no method enabled, every settings section is hidden (no clutter).
    assert not tab._l41_shared.isVisibleTo(tab._l41_shared.parent())
    assert not tab._l41_rf.isVisibleTo(tab._l41_rf.parent())
    assert not tab._l41_mds.isVisibleTo(tab._l41_mds.parent())

    # Enabling MDS shows only the MDS section; the RF/XGB shared box stays hidden.
    tab.l41_mds.setChecked(True)
    assert tab._l41_mds.isVisibleTo(tab._l41_mds.parent())
    assert not tab._l41_shared.isVisibleTo(tab._l41_shared.parent())
    assert not tab._l41_rf.isVisibleTo(tab._l41_rf.parent())

    # Enabling RF reveals the shared (features/seed) box + the RF section.
    tab.l41_rf.setChecked(True)
    assert tab._l41_shared.isVisibleTo(tab._l41_shared.parent())
    assert tab._l41_rf.isVisibleTo(tab._l41_rf.parent())

    # Feature picker: filter + select-all acts on the filtered subset only.
    fp = tab.l41_features
    assert fp.count() == len(df.columns)
    fp._filter.setText("VPD")
    fp._select_all()
    picked = fp.selected()
    assert picked and all("VPD" in p for p in picked)
    # Clearing the filter and ticking one more is additive (selection is sticky).
    fp._filter.clear()
    fp.set_selected(picked + [str(df.columns[0])])
    assert set(fp.selected()) == set(picked) | {str(df.columns[0])}


def test_flux_chain_tab_per_level_run(app):
    # Each level runs separately, its output feeding the next; the per-level run
    # buttons are gated by how far the chain has reached.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Before any run: only the Input (init) level button is enabled.
    assert tab._level_run_btns[0].isEnabled()
    assert not tab._level_run_btns[1].isEnabled()
    # Running a downstream level before its predecessor is refused.
    tab._run_level(2)
    assert "Run L2 first" in tab.summary.toPlainText()

    # Run each level in turn on a synchronous worker; the output feeds forward.
    def run(idx):
        plan = tab._level_plan(idx)
        assert plan is not None
        tab._level_worker(plan, tab._data, tab._df)
        QApplication.processEvents()

    run(0)                                  # Input: init_flux_data
    assert tab._data is not None and tab._reached == 0
    assert tab._level_run_btns[1].isEnabled()   # L2 unlocked
    assert not tab._level_run_btns[2].isEnabled()

    run(1)                                  # L2 on the init output
    assert tab._reached == 1
    assert "L2 done" in tab.summary.toPlainText()

    run(2)                                  # L3.1 on the L2 output
    assert tab._reached == 2
    assert any("L3.1" in str(c) for c in tab._data.fpc_df.columns)

    # L3.2 needs steps; add one, then run.
    tab._steps = [HampelParams().step()]
    run(3)
    assert tab._reached == 3
    assert getattr(tab._data.levels, "level32_qcf", None) is not None

    # The rail lights every reached card with the green ✓ up to L3.2.
    assert tab.rail._cards[3]._reached
    assert not tab.rail._cards[4]._reached

    # Re-running an earlier level cascades the deeper state away (reach resets).
    run(2)
    assert tab._reached == 2
    assert not tab.rail._cards[3]._reached


def test_flux_chain_tab_level41(app):
    # L4.1 gap-filling: fan out one gap-fill per USTAR scenario. MDS is fast and
    # has no ML training, so drive a real run with it; rf/xgb are checked via codegen.
    from diive.gui.tabs.fluxchain import FluxChainTab
    from diive.gui.widgets.stepwise_method_params import HampelParams
    from diive.configs.exampledata import load_exampledata_parquet_lae_level1_30MIN
    df = load_exampledata_parquet_lae_level1_30MIN().loc["2024-07":"2024-07"]

    tab = FluxChainTab()
    tab.widget()
    tab.on_data_loaded(df)
    # MDS driver combos auto-pick the gap-filled drivers (FLAG_* skipped).
    assert tab.mds_vpd.currentText() == "VPD_T1_47_1_gfXG"
    assert tab.mds_swin.currentText() == "SW_IN_T1_47_1_gfXG"

    # A full chain through L3.3 is required for L4.1.
    tab._steps = [HampelParams().step()]
    tab.l33_enable.setChecked(True)
    tab._ustar = [(0.1, "CUT_50")]

    # Enabling MDS deepens the run target to 4.1.
    tab.l41_mds.setChecked(True)
    tab._update_run_label()
    assert "4.1" in tab.run_btn.text()

    # Copy-Python renders the full L2 -> L4.1 composable chain (rf+xgb+mds form).
    tab.l41_rf.setChecked(True)
    tab.l41_xgb.setChecked(True)
    feat0 = str(df.columns[0])
    tab.l41_features.set_selected([feat0])
    assert tab.l41_features.selected() == [feat0]
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "run_level41_mds(" in code
    assert "run_level41_rf(" in code and "run_level41_xgb(" in code
    assert code.count("make_level41_engineer(") == 1
    # The random seed is always pinned (this is what makes rf/xgb reproducible),
    # so it appears in both ML calls; untouched hyperparameters are omitted.
    assert code.count("random_state=42") == 2
    assert "n_estimators" not in code  # default -> omitted
    # Editing a hyperparameter + seed flows into the config and the script.
    tab.l41_seed.setValue(7)
    tab.rf_n_est.setValue(350)
    tab.l41_reduce.setChecked(True)
    cfg = tab._level41_cfg()
    assert cfg["rf_kwargs"] == {"random_state": 7, "n_estimators": 350}
    assert cfg["reduce_features"] is True
    code2 = tab._code()
    assert "random_state=7" in code2 and "n_estimators=350" in code2
    assert "reduce_features=True" in code2
    tab.l41_seed.setValue(42); tab.rf_n_est.setValue(100); tab.l41_reduce.setChecked(False)

    # Run MDS only (fast). Drive the synchronous core directly.
    tab.l41_rf.setChecked(False)
    tab.l41_xgb.setChecked(False)
    cfg = tab._level41_cfg()
    assert cfg["methods"] == ["mds"]
    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings(),
                        tab._level31_kwargs(), tab._steps, tab._level33_kwargs(), cfg)
    cols = data.gapfilled_cols()
    assert "mds" in cols and "CUT_50" in cols["mds"]
    tab._on_done(data)
    QApplication.processEvents()
    assert "Level 4.1 done" in tab.summary.toPlainText()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]

    # The heatmaps view embeds via plot_gapfilled_heatmaps(fig=...).
    tab.l41_view.setCurrentText("Gap-filled heatmaps")
    tab._on_done(data)
    QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]

    # Selecting an L4.1 method without features is rejected up front.
    tab.l41_view.setCurrentText("Cumulative comparison")
    tab.l41_rf.setChecked(True)
    tab.l41_mds.setChecked(False)
    tab.l41_features.set_selected([])
    tab._run()
    assert "feature" in tab.summary.toPlainText().lower()

    # L4.1 config round-trips through save/restore (methods + features + drivers).
    tab.l41_rf.setChecked(False)
    tab.l41_mds.setChecked(True)
    tab.l41_features.set_selected([feat0])
    state = tab.save_state()
    tab2 = FluxChainTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2.l41_mds.isChecked()
    assert tab2._level41_cfg()["methods"] == ["mds"]
    assert tab2.l41_features.selected() == tab.l41_features.selected()


def test_stepwise_method_params_run_on_detector(app):
    # Every stepwise method-params widget must produce a {method, kwargs} step that
    # StepwiseOutlierDetection actually accepts and runs — this guards the kwarg
    # mapping (which differs from the raw-detector tabs) against the library API.
    import diive as dv
    from diive.preprocessing.outlier_detection import StepwiseOutlierDetection
    from diive.flux.fluxprocessingchain import level32_to_code
    from diive.gui.widgets.stepwise_method_params import STEP_METHODS

    df = dv.variables.generate_noisy_timeseries(
        start_date="2024-01-01", periods=48 * 20, freq="30min",
        trend_slope=0.01, seasonal_strength=9, noise_level=2, outlier_fraction=0.1)
    df.index.name = "TIMESTAMP_END"
    coords = dict(site_lat=46.8, site_lon=8.6, utc_offset=1)

    steps = []
    for cls in STEP_METHODS:
        widget = cls()
        step = widget.step()
        assert hasattr(StepwiseOutlierDetection, step["method"])
        steps.append(step)
        # output_middle_timestamp=False keeps the input index (GUI alignment path).
        det = StepwiseOutlierDetection(dfin=df, col="observed_value",
                                       output_middle_timestamp=False, **coords)
        getattr(det, step["method"])(**step["kwargs"])
        det.addflag()
        assert det.flags.shape[1] == 1  # exactly one committed flag

    # A Hampel day/night step (per-period thresholds) also runs.
    from diive.gui.widgets.stepwise_method_params import HampelParams
    hp = HampelParams()
    hp.dn_cb.setChecked(True)
    det = StepwiseOutlierDetection(dfin=df, col="observed_value",
                                   output_middle_timestamp=False, **coords)
    getattr(det, hp.step()["method"])(**hp.step()["kwargs"])
    det.addflag()
    assert det.flags.shape[1] == 1

    # The collected steps render a compilable L3.2 script.
    code = level32_to_code(
        init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8, utc_offset=1),
        level2_settings={"ssitc": {"apply": True, "setflag_timeperiod": None}},
        level31_kwargs={}, level32_steps=steps)
    compile(code, "<gen>", "exec")


def test_all_menu_items_have_icons(window):
    # Every (non-separator) menu entry carries a drawn icon. Menus live as inline
    # header dropdown buttons in the Studio shell.
    from PySide6.QtWidgets import QToolButton
    count = 0
    for btn in window._header.findChildren(QToolButton, "headermenu"):
        menu = btn.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.isSeparator():
                continue
            count += 1
            assert not action.icon().isNull(), action.text()
    assert count >= 12  # File/Data/Plot/Outliers/Flux/Analyze/Settings/Help entries


def test_diel_cycle_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Diel cycle").isNull()
    window._open_menu_tab("Diel cycle")
    tab = window._menu_tab_list[-1]
    assert tab._panels  # default variable rendered
    fig = tab.canvas.fig
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    vals = tab.settings.values()
    assert {"mean", "std", "each_month"} <= set(vals)
    assert "legend_loc" in vals["_format"]  # chrome now lives in the shared Format group
    # Ctrl+click stacks a second variable (shared time-of-day x-axis).
    tab._on_selected("Tair_f", True)
    QApplication.processEvents()
    assert tab._panels == ["NEE_CUT_REF_f", "Tair_f"]
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]

    # Per-month colouring: with "one curve per month" on, the GUI must pass
    # color=None so DielCycle auto-colours each month distinctly (the bug fix).
    tab._panels = ["Tair_f"]
    tab.settings.dc_each_month.setChecked(True)
    tab.update_btn.click()  # params apply on the button, not on edit
    QApplication.processEvents()
    ax = tab.canvas.fig.axes[0]
    line_colors = {tuple(l.get_color()) if not isinstance(l.get_color(), str) else l.get_color()
                   for l in ax.get_lines()}
    assert len(line_colors) > 1  # months drawn in different colours, not all one
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # Legend position is wired through.
    tab.settings.fmt_legend_loc.setCurrentText("upper right")
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab.settings.values()["_format"]["legend_loc"] == "upper right"
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_scatter_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Scatter XY").isNull()
    window._open_menu_tab("Scatter XY")
    tab = window._menu_tab_list[-1]
    # Seeded with X, Y (2 vars) -> plain scatter renders.
    assert len(tab._xyz) >= 2
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    sc_vals = tab.settings.values()
    assert {"nbins", "binagg", "cmap", "show_colorbar", "markersize", "alpha",
            "vmin", "vmax", "_format", "_axes"} <= set(sc_vals)
    assert "title" in sc_vals["_format"]  # chrome moved into the shared Format group
    # Add a third variable (Z) -> colour scatter (extra colorbar axis).
    tab._xyz = ["Tair_f", "NEE_CUT_REF_f", "VPD_f"]
    tab._render()
    QApplication.processEvents()
    assert len(tab.canvas.fig.axes) >= 2  # scatter + colorbar
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert tab.settings.z_role.text() == "VPD_f"  # role readout updated

    # Marker size / opacity reach the scatter collection.
    tab.settings.sc_markersize.setValue(80)
    tab.settings.sc_alpha.setValue(0.3)
    tab._render()
    QApplication.processEvents()
    sax = tab.canvas.fig.axes[0]
    coll = sax.collections[0]
    assert abs(coll.get_sizes()[0] - 80) < 1e-6
    assert abs(coll.get_alpha() - 0.3) < 1e-6
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]

    # Reverse-colormap toggle flips the cmap name (GUI-only _r suffix).
    tab.settings.sc_cmap.setCurrentText("viridis")
    tab.settings.sc_reverse_cmap.setChecked(True)
    assert tab.settings.values()["cmap"] == "viridis_r"

    # GUI-only Axes pass: invert Y and a Y limit apply to the data axis.
    tab.settings.sc_reverse_cmap.setChecked(False)
    tab.settings.ax_invert_y.setChecked(True)
    tab.settings.ax_ymin.setText("0")
    tab.settings.ax_ymax.setText("5")
    tab._render()
    QApplication.processEvents()
    sax = tab.canvas.fig.axes[0]
    lo, hi = sax.get_ylim()
    assert lo > hi  # inverted
    assert {round(lo), round(hi)} == {0, 5}
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_cumulative_year_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Cumulative year").isNull()
    window._open_menu_tab("Cumulative year")
    tab = window._menu_tab_list[-1]
    assert tab._panels
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    vals = tab.settings.values()
    assert {"show_reference", "highlight_year", "digits_after_comma"} <= set(vals)
    assert vals["highlight_year"] is None  # "none" -> no highlight
    # Highlight year is a dropdown populated from the data's years.
    items = [tab.settings.cy_highlight.itemText(i)
             for i in range(tab.settings.cy_highlight.count())]
    assert items[0] == "none"
    assert "2021" in items  # the example data's year is offered
    tab.settings.cy_show_reference.setChecked(True)
    tab.settings.cy_highlight.setCurrentText("2021")
    tab.update_btn.click()  # params apply on the button, not on edit
    QApplication.processEvents()
    assert tab.settings.values()["highlight_year"] == 2021
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_ridgeline_and_plot_icons(window):
    from diive.gui.icons import plot_menu_icon
    # Every Plot-menu method gets a non-empty drawn icon.
    for label in ("Heatmap date/time", "Heatmap year/month", "Time series", "Ridgeline"):
        assert not plot_menu_icon(label).isNull()

    window._open_menu_tab("Ridgeline")
    tab = window._menu_tab_list[-1]
    # Ridgeline manages its own figure layout (not the canvas constrained one).
    assert tab.canvas.auto_layout is False
    assert tab._panels  # default variable rendered
    fig = tab.canvas.fig
    assert len(fig.axes) > 1  # one density ridge per period
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # Single-variable: ctrl+click does not add a comparison panel.
    tab._on_selected("Tair_f", True)
    QApplication.processEvents()
    assert tab._panels == ["Tair_f"]


def test_hexbin_tab(window):
    from diive.gui.icons import plot_menu_icon
    assert not plot_menu_icon("Hexbin").isNull()

    window._open_menu_tab("Hexbin")
    tab = window._menu_tab_list[-1]
    QApplication.processEvents()
    # Seeds three roles (driver/driver/flux) and renders a hexbin on open.
    assert len(tab._xyz) == 3
    fig = tab.canvas.fig
    assert fig.axes and fig.axes[0].collections  # hexbin polycollection drawn
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # Role readout reflects the assignment.
    assert tab.settings.x_role.text() == tab._xyz[0]
    assert tab.settings.z_role.text() == tab._xyz[2]

    # Click cycling: clicking an assigned variable removes it; an incomplete
    # selection shows the prompt instead of a plot.
    x0 = tab._xyz[0]
    tab._on_selected(x0, False)
    QApplication.processEvents()
    assert x0 not in tab._xyz and len(tab._xyz) == 2
    assert any("X, Y, Z" in t.get_text() for a in tab.canvas.fig.axes for t in a.texts)
    # Re-adding fills the freed slot again (back to three).
    tab._on_selected(x0, False)
    QApplication.processEvents()
    assert len(tab._xyz) == 3


def test_gap_dashboard_tab(window):
    import types
    import matplotlib.dates as mdates
    from diive.gui.icons import menu_icon
    assert not menu_icon("Gaps & coverage").isNull()

    window._open_menu_tab("Gaps & coverage")
    tab = window._menu_tab_list[-1]
    # run_with_loading defers the compute one tick; flush it.
    for _ in range(50):
        QApplication.processEvents()

    # Defaults to the gappiest variable so the dashboard is useful on open.
    assert tab._target == "NEE_CUT_84_orig"
    fig = tab.canvas.fig
    assert len(fig.axes) >= 2  # availability heatmap + gap timeline (+ colorbars)
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # Stat cards (minus trailing stretch) and a populated long-gap table.
    assert tab.stats_layout.count() - 1 == 6
    n_rows = tab.table.rowCount()
    assert n_rows > 0

    # Clickable gap map, table -> highlight overlay on the timeline.
    tab.table.selectRow(0)
    QApplication.processEvents()
    assert len(tab._highlight) == 2  # span + ring

    # Clickable gap map, timeline click -> nearest gap highlighted + row selected.
    g0 = tab._long_gaps.iloc[0]
    ev = types.SimpleNamespace(
        inaxes=tab._timeline_ax,
        xdata=mdates.date2num(g0["GAP_START"]),
        ydata=float(g0["GAP_LENGTH"]))
    tab._on_click(ev)
    QApplication.processEvents()
    assert len(tab._highlight) == 2
    sel = tab.table.selectionModel().selectedRows()
    assert sel and sel[0].row() == 0

    # Raising the long-gap threshold lists fewer gaps (library recompute).
    tab.threshold.setValue(tab.threshold.value() * 4)
    for _ in range(50):
        QApplication.processEvents()
    assert tab.table.rowCount() < n_rows

    # Single-instance: re-opening focuses the existing tab.
    window._open_menu_tab("Gaps & coverage")
    assert _tabs(window).count("Gaps & coverage") == 1


def test_driver_explorer_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Driver explorer").isNull()

    window._open_menu_tab("Driver explorer")
    tab = window._menu_tab_list[-1]
    for _ in range(60):
        QApplication.processEvents()

    # Opens on a flux target, ranks the other variables, shows the top scatter.
    assert tab._target == "NEE_CUT_REF_f"
    assert tab._ranked is not None and len(tab._ranked) > 0
    assert tab.table.rowCount() == len(tab._ranked)
    assert tab.stats_layout.count() - 1 == 6  # stat cards
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # Ranked by |corr| descending (no lag scan yet -> all lags 0).
    abs_corr = tab._ranked["ABS_CORR"].to_numpy()
    assert (abs_corr[:-1] >= abs_corr[1:]).all()
    assert (tab._ranked["BEST_LAG"] == 0).all()

    # Lag scan applies on the button and can pick non-zero lags.
    tab.max_lag.setValue(6)
    tab.rank_btn.click()
    for _ in range(60):
        QApplication.processEvents()
    assert int(tab._ranked["BEST_LAG"].abs().max()) <= 6
    assert (tab._ranked["BEST_LAG"] != 0).any()

    # Click a ranked driver -> target-vs-driver scatter renders.
    tab.table.selectRow(min(3, tab.table.rowCount() - 1))
    QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert tab.canvas.fig.axes  # a scatter axis exists

    # Single-instance: re-opening focuses the existing tab.
    window._open_menu_tab("Driver explorer")
    assert _tabs(window).count("Driver explorer") == 1


def test_seasonal_trend_tab(app):
    # Needs several years (annual STL needs >= 2 cycles), so build a standalone
    # tab with multi-year data instead of the one-year `window` fixture.
    from diive.gui.tabs.seasonaltrend import SeasonalTrendTab
    from diive.gui.icons import menu_icon
    assert not menu_icon("Seasonal trend & anomalies").isNull()

    df = dv.load_exampledata_parquet().loc["2018":"2022"]  # 5 years
    tab = SeasonalTrendTab()
    tab.widget()
    tab.on_data_loaded(df)
    for _ in range(80):
        QApplication.processEvents()

    # Decomposition view: STL ran (regression — it used to always raise) and the
    # four component panels drew.
    assert tab._target == "Tair_f"
    assert tab._decomp is not None
    assert tab._decomp["strength"] > 0.4
    fig = tab.canvas.fig
    assert len(fig.axes) == 4
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert tab.stats_layout.count() - 1 == 6  # stat cards
    assert (tab.ref_start.value(), tab.ref_end.value()) == (2018, 2022)

    # Anomaly view renders a single bar chart vs the reference period.
    tab.view.setCurrentText("Yearly anomalies")
    QApplication.processEvents()
    assert len(tab.canvas.fig.axes) == 1
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]

    # Switching method recomputes on the Update button.
    tab.view.setCurrentText("Decomposition")
    tab.method.setCurrentText("Classical")
    tab.update_btn.click()
    for _ in range(60):
        QApplication.processEvents()
    assert tab._decomp is not None
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_seasonal_trend_short_data_graceful(window):
    # The window fixture has one year -> annual decomposition can't run. The tab
    # must show a friendly message (not crash), and the anomaly view still works.
    window._open_menu_tab("Seasonal trend & anomalies")
    tab = window._menu_tab_list[-1]
    for _ in range(60):
        QApplication.processEvents()
    assert tab._decomp is None
    msgs = [t.get_text() for a in tab.canvas.fig.axes for t in a.texts]
    assert any("2 years" in m for m in msgs)
    tab.view.setCurrentText("Yearly anomalies")
    QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_spectrogram_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Spectrogram").isNull()

    window._open_menu_tab("Spectrogram")
    tab = window._menu_tab_list[-1]
    for _ in range(60):
        QApplication.processEvents()

    assert tab._target == "NEE_CUT_REF_f"
    assert tab._spec is not None
    assert round(tab._rec_per_day) == 48  # half-hourly -> 48 records/day
    fig = tab.canvas.fig
    qmesh = [c for a in fig.axes for c in a.collections
             if c.__class__.__name__ == "QuadMesh"]
    assert qmesh  # spectrogram mesh drawn
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    # The explanation text is present.
    assert "cycle" in tab.explanation.text()

    # Window length applies on Update (recompute -> different segmentation).
    before = tab._spec["power"].shape
    tab.nperseg.setValue(128)
    tab.update_btn.click()
    for _ in range(40):
        QApplication.processEvents()
    assert tab._spec["power"].shape != before

    # Max cycles/day is a live re-render (y-limit only).
    tab.max_freq.setValue(2.0)
    QApplication.processEvents()
    assert round(tab.canvas.fig.axes[0].get_ylim()[1], 1) == 2.0

    # Single-instance.
    window._open_menu_tab("Spectrogram")
    assert _tabs(window).count("Spectrogram") == 1


def test_histogram_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Histogram").isNull()

    window._open_menu_tab("Histogram")
    tab = window._menu_tab_list[-1]
    for _ in range(40):
        QApplication.processEvents()

    assert tab._panels  # default variable rendered
    fig = tab.canvas.fig
    assert not [t for a in fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert len(fig.axes) >= 2  # histogram + z-score twiny axis
    vals = tab.settings.values()
    assert {"n_bins", "highlight_peak", "show_zscores", "show_counts"} <= set(vals)

    # Single-variable (like the ridgeline): Ctrl+click replaces, never stacks.
    tab._on_selected("Tair_f", True)
    QApplication.processEvents()
    assert tab._panels == ["Tair_f"]
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]

    # Settings apply on the Update button (no z-score axis when disabled).
    tab.settings.hist_nbins.setValue(30)
    tab.settings.hist_zscores.setChecked(False)
    tab.settings.hist_info.setChecked(False)
    tab.update_btn.click()
    for _ in range(20):
        QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert len(tab.canvas.fig.axes) == 1  # twiny gone with z-scores off


def test_splash_screen(app):
    from PySide6.QtWidgets import QSplashScreen
    from diive.gui.splash import make_splash_pixmap, create_splash, AUTHOR, SUPPORTERS
    # Artwork renders to a non-empty, high-DPI-aware pixmap.
    pm = make_splash_pixmap(2.0)
    assert not pm.isNull()
    assert pm.devicePixelRatio() == 2.0 and pm.width() > 0
    # Author is baked in; supporters list exists (empty by default, extensible).
    assert AUTHOR == "Lukas Hörtnagl"
    assert isinstance(SUPPORTERS, list)
    # create_splash returns a ready QSplashScreen.
    splash = create_splash(app)
    assert isinstance(splash, QSplashScreen)
    assert not splash.pixmap().isNull()

    # Help > About reuses the same artwork as a modal dialog.
    from PySide6.QtWidgets import QLabel
    from diive.gui.splash import _AboutDialog, _WIDTH, _HEIGHT
    dlg = _AboutDialog()
    assert dlg.isModal()
    assert (dlg.width(), dlg.height()) == (_WIDTH, _HEIGHT)
    has_art = [l for l in dlg.findChildren(QLabel)
               if l.pixmap() is not None and not l.pixmap().isNull()]
    assert has_art  # the splash pixmap is shown
    dlg.accept()


def test_appearance_singleton(window):
    window._open_menu_tab("Appearance")
    window._open_menu_tab("Appearance")
    assert _tabs(window).count("Appearance") == 1


def test_project_settings_tab(window):
    from diive.gui import site
    # Lives under the Settings menu and opens a single instance.
    window._open_menu_tab("Project settings")
    window._open_menu_tab("Project settings")
    assert _tabs(window).count("Project settings") == 1
    tab = window._menu_tab_list[-1]
    # Entering + saving stores the values app-wide for later use by functions.
    tab.name.setText("CH-DAV")
    tab.author.setText("Jane Doe")
    tab.description.setPlainText("Test project notes.")
    tab.lat.setValue(46.8153)
    tab.lon.setValue(9.8559)
    tab.utc.setValue(1)
    tab._save()
    assert site.manager.configured
    assert site.manager.author == "Jane Doe"
    assert site.manager.description == "Test project notes."
    # Author + description round-trip through the persistence dict (config + project).
    assert site.manager.as_dict()["author"] == "Jane Doe"
    restored = site.SiteManager()
    restored.load_dict(site.manager.as_dict())
    assert restored.author == "Jane Doe"
    assert restored.description == "Test project notes."
    assert site.manager.latitude == 46.8153
    assert site.manager.longitude == 9.8559
    assert site.manager.utc_offset == 1
    # Persistence round-trips through the as_dict/load_dict pair.
    from diive.gui.site import SiteManager
    restored = SiteManager()
    restored.load_dict(site.manager.as_dict())
    assert restored.as_dict() == site.manager.as_dict()
    # Reset the process-wide singleton so other tests start clean.
    site.manager.update(name="", latitude=0.0, longitude=0.0, elevation=0.0,
                        utc_offset=0)
    site.manager.configured = False


def test_keep_vars_subset():
    df = dv.load_exampledata_parquet()
    wanted = [df.columns[3], df.columns[0]]  # out-of-order on purpose
    out = dv.keep_vars(df, wanted)
    assert list(out.columns) == wanted          # order preserved
    assert out.shape[0] == df.shape[0]          # rows untouched
    assert df.shape[1] > out.shape[1]           # input untouched (still full)
    import pytest as _pt
    with _pt.raises(ValueError):
        dv.keep_vars(df, ["does_not_exist"])


def test_select_variables_tab_updates_overview(window):
    tw = window._tabwidget
    window._open_menu_tab("Select variables")
    sel = window._menu_tab_list[-1]
    n_all = sel.available.list.count()
    assert n_all == window._data.shape[1]
    assert hasattr(sel, "subsetSelected")

    names = [str(c) for c in window._data.columns]
    sel._select(names[5])
    sel._select(names[1])
    # Moved out of "available" into "selected".
    assert sel.available.list.count() == n_all - 2
    assert sel.selected.names() == [names[5], names[1]]

    sel._confirm()  # what the Confirm button does
    QApplication.processEvents()
    overview = window._tabs[0]
    assert overview.varpanel.names() == [names[5], names[1]]


def test_select_variables_subset_is_app_wide(window):
    """The subset narrows `_data` for every (non-pinned) tab, not just the
    Overview; the picker keeps the full list and reset restores everything."""
    names = [str(c) for c in window._full_data.columns]
    n_all = window._full_data.shape[1]

    # A pinned tab must keep its full dataset after a subset is applied.
    window._open_menu_tab("Time series")
    pinned = window._menu_tab_list[-1]
    window._toggle_pin(pinned)
    pushed = []
    pinned.on_data_loaded = lambda df, created=None: pushed.append(df)

    window._apply_var_subset([names[5], names[1]])
    QApplication.processEvents()
    assert pushed == []  # frozen: never received the narrowed dataset
    # Every non-pinned tab now sees only the 2-column data.
    assert list(window._data.columns) == [names[5], names[1]]
    assert window._reset_subset_act.isEnabled()
    # The picker opted into the full record, so it can still re-add the others.
    window._open_menu_tab("Select variables")
    sel = window._menu_tab_list[-1]
    assert sel.available.list.count() + sel.selected.list.count() == n_all
    assert sel.selected.names() == [names[5], names[1]]  # reflects active subset

    window._reset_var_subset()
    QApplication.processEvents()
    assert window._data.shape[1] == n_all
    assert not window._reset_subset_act.isEnabled()


def test_pin_freezes_menu_tab(window):
    window._open_menu_tab("Time series")
    tab = window._menu_tab_list[-1]

    window._toggle_pin(tab)  # what the right-click "Pin" does
    assert tab in window._pinned

    # A pinned tab is skipped by data pushes; unpinning re-syncs it once.
    calls = []
    tab.on_data_loaded = lambda df, created=None: calls.append(df)
    window._apply_range()
    QApplication.processEvents()
    assert calls == []                       # frozen: no push received
    window._toggle_pin(tab)                  # unpin -> re-sync
    QApplication.processEvents()
    assert len(calls) == 1


def test_hampel_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Hampel filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    # Run the worker directly (synchronous) instead of the background thread.
    series = window._data[var]
    tab._worker(series, dict(window_length=48 * 13, n_sigma=5.5,
                             use_differencing=True, separate_day_night=False), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_HAMPEL", f"FLAG_{var}_OUTLIER_HAMPEL_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]

    # The cleaned (bottom) panel autoscales its y-axis to the outlier-free range:
    # its y-axis is independent of the spike-stretched top panel, while the time
    # x-axis stays linked for synchronized pan/zoom.
    top, bot = tab.canvas.fig.axes[0], tab.canvas.fig.axes[1]
    assert not bot.get_shared_y_axes().joined(top, bot)
    assert bot.get_shared_x_axes().joined(top, bot)

    n_before = window._data.shape[1]
    tab._add()  # what "Add cleaned + flag to dataset" does
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    # The flag is 0 (ok) / 2 (outlier); the cleaned series drops the flagged rows.
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_metadata_provenance_from_outlier_run(window):
    from diive.core.metadata import DERIVED, FAVORITE, MODIFIED, ORIGINAL
    from diive.gui import metadata_store

    store = metadata_store.manager.store
    var = "Tair_f"
    # Freshly loaded columns carry an "original" baseline.
    assert store.get(var).origin == ORIGINAL

    window._open_menu_tab("Hampel filter")
    tab = window._menu_tab_list[-1]
    tab._select(var)
    series = window._data[var]
    tab._worker(series, dict(window_length=48 * 13, n_sigma=5.5,
                             use_differencing=True, separate_day_night=False), True)
    QApplication.processEvents()
    tab._add()
    QApplication.processEvents()

    cleaned, flag = f"{var}_HAMPEL", f"FLAG_{var}_OUTLIER_HAMPEL_TEST"
    md = store.get(cleaned)
    assert md.origin == MODIFIED
    assert md.parents == [var]
    assert "hampel" in md.tags
    assert len(md.provenance) >= 1  # >=1: a shared window may run this twice
    assert store.get(flag).origin == DERIVED

    # User tags persist (favorite); provenance tags do not round-trip.
    metadata_store.manager.add_user_tag(var, FAVORITE)
    saved = store.user_tags()
    assert saved.get(var) == [FAVORITE]
    assert cleaned not in saved  # function-set "hampel" tag is not persisted

    # Favorites float to the top of the variable list (re-sorted on the tag's
    # `changed` signal), so the first row is now a favorite.
    QApplication.processEvents()
    assert metadata_store.manager.is_favorite(
        tab.varpanel.list.item(0).data(Qt.ItemDataRole.UserRole))


def test_metadata_tags_are_per_dataset(window):
    import numpy as np
    import pandas as pd

    from diive.gui import metadata_store
    store = metadata_store.manager.store

    idx = window._data.index[:50]
    var = str(window._data.columns[0])
    # Load a real (persisted) dataset — the example auto-load is intentionally
    # clean and unpersisted, so it can't serve as "dataset one" here.
    df1 = pd.DataFrame({var: np.arange(len(idx), dtype=float)}, index=idx)
    window._set_data(df1, source="dataset one")
    key_a = window._dataset_key
    assert key_a == "dataset one"
    metadata_store.manager.add_user_tag(var, "ds1tag")
    assert "ds1tag" in store.get(var).tags

    # A different dataset that happens to share the column name must NOT inherit
    # dataset 1's tag (tags are namespaced by dataset, not by variable name).
    df2 = pd.DataFrame({var: np.arange(len(idx), dtype=float)}, index=idx)
    window._set_data(df2, source="dataset two")
    QApplication.processEvents()
    assert "ds1tag" not in store.get(var).tags

    # Re-loading the first dataset (same key + column) restores its tag.
    df3 = pd.DataFrame({var: np.arange(len(idx), dtype=float)}, index=idx)
    window._set_data(df3, source=key_a)
    QApplication.processEvents()
    assert "ds1tag" in store.get(var).tags


def test_metadata_explorer_clear_all(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from diive.gui import metadata_store
    store = metadata_store.manager.store

    window._open_menu_tab("Metadata explorer")
    tab = window._menu_tab_list[-1]
    var = str(window._data.columns[0])
    tab._select(var)
    metadata_store.manager.add_user_tag(var, "favorite")
    metadata_store.manager.add_user_tag(var, "lukas")
    store.set_description(var, "a note")
    assert store.user_data()["tags"]

    # Confirm the (destructive) dialog, then clear.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    tab._clear_all()
    QApplication.processEvents()
    assert store.user_data() == {"tags": {}, "descriptions": {}}


def test_metadata_edit_navigation_from_varlist(window):
    from diive.gui import metadata_store

    var = str(window._data.columns[2])
    # What a variable-list "Edit metadata…" right-click triggers.
    metadata_store.manager.request_edit(var)
    QApplication.processEvents()
    explorers = [t for t in window._menu_tab_list
                 if getattr(t, "_menu_label", None) == "Metadata explorer"]
    assert len(explorers) == 1                 # opened the (single-instance) tab
    assert explorers[0]._current == var        # focused on the clicked variable

    # Requesting another variable focuses the same tab and reselects.
    var2 = str(window._data.columns[3])
    metadata_store.manager.request_edit(var2)
    QApplication.processEvents()
    explorers = [t for t in window._menu_tab_list
                 if getattr(t, "_menu_label", None) == "Metadata explorer"]
    assert len(explorers) == 1
    assert explorers[0]._current == var2


def test_metadata_explorer_clear_one_via_right_click(window):
    from diive.gui import metadata_store
    store = metadata_store.manager.store

    window._open_menu_tab("Metadata explorer")
    tab = window._menu_tab_list[-1]
    keep = str(window._data.columns[0])
    target = str(window._data.columns[1])
    for var in (keep, target):
        metadata_store.manager.add_user_tag(var, "favorite")
        store.set_description(var, "note")
    tab._select(target)

    # The panel's right-click "Remove all tags & note" emits clearRequested.
    tab.varpanel.clearRequested.emit(target)
    QApplication.processEvents()
    assert not store.get(target).tags - {"original"}  # only auto baseline remains
    assert store.get(target).description == ""
    assert "favorite" in store.get(keep).tags          # other variable untouched


def test_metadata_example_data_loads_clean(window):
    from diive.gui import metadata_store
    store = metadata_store.manager.store

    var = str(window._data.columns[0])
    key = "example data (CH-DAV)"
    # Pretend a stale example entry was persisted from a previous session.
    window._saved_metadata[key] = {"tags": {var: ["favorite"]}, "descriptions": {}}

    window._load_example()  # reload the bundled example
    QApplication.processEvents()

    assert "favorite" not in store.get(var).tags        # not applied — clean
    assert key not in window._saved_metadata            # stale entry purged
    assert window._dataset_key is None                  # example isn't persisted


def test_project_save_and_open(window, tmp_path, monkeypatch):
    from diive.core.io import project as projmod
    from diive.gui import app as appmod
    from diive.gui import metadata_store, site
    from PySide6.QtWidgets import QFileDialog
    store = metadata_store.manager.store

    var = str(window._data.columns[0])
    metadata_store.manager.add_user_tag(var, "favorite")
    store.set_description(var, "important variable")
    site.manager.update(name="X", latitude=46.8, longitude=9.8,
                        elevation=1000.0, utc_offset=1)

    folder = tmp_path / "Proj.diive"
    assert window._write_project(folder, "Proj")
    assert projmod.is_project(folder)

    # Change state, then open the project back — it must restore everything.
    window._load_example()  # clears tags + project + (here) the site is changed too
    site.manager.update(name="Y", latitude=0.0, longitude=0.0,
                        elevation=0.0, utc_offset=0)
    assert "favorite" not in store.get(var).tags
    assert window._project_dir is None

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(folder)))
    window._open_project()
    QApplication.processEvents()

    assert window._project_name == "Proj"
    assert window._project_dir == folder
    assert "favorite" in store.get(var).tags
    assert store.get(var).description == "important variable"
    assert site.manager.latitude == 46.8           # site restored from the project
    assert var in [str(c) for c in window._data.columns]


def test_frameless_resize_cursor_no_int_error(app):
    # Regression: PySide6 Qt.Edge flags aren't int()-able; the helper must use
    # `.value` (the eventFilter previously raised TypeError on every mouse move).
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QWidget

    from diive.gui.widgets.frameless import FramelessResizeHelper
    w = QWidget(); w.resize(400, 300)
    h = FramelessResizeHelper(w, w)
    assert h._cursor_for(h._edges(QPoint(2, 2))) == Qt.CursorShape.SizeFDiagCursor
    assert h._cursor_for(h._edges(QPoint(2, 150))) == Qt.CursorShape.SizeHorCursor
    assert h._cursor_for(h._edges(QPoint(200, 150))) == Qt.CursorShape.ArrowCursor
    assert h._edges(QPoint(200, 150)).value == 0  # interior -> no edge


def test_startup_loads_example_when_no_project(window):
    # The fixture builds MainWindow() with no saved project -> example data.
    assert window._data is not None
    assert window._project_dir is None


def test_startup_reopens_last_project(window, tmp_path, monkeypatch):
    from diive.gui import metadata_store
    from diive.gui.app import MainWindow
    store = metadata_store.manager.store

    var = str(window._data.columns[0])
    metadata_store.manager.add_user_tag(var, "favorite")
    folder = tmp_path / "Proj.diive"
    window._write_project(folder, "Proj")

    # A fresh window told this is the last project reopens it on startup
    # (and must NOT fall back to the example data).
    def _boom():
        raise AssertionError("example data should not be loaded")
    monkeypatch.setattr("diive.load_exampledata_parquet", _boom)

    win2 = MainWindow(config={"last_project": str(folder)})
    QApplication.processEvents()
    assert win2._project_name == "Proj"
    assert win2._project_dir == folder
    assert "favorite" in store.get(var).tags


def test_project_saves_and_restores_open_tabs(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    # Open a couple of menu tabs (the workspace to capture).
    window._open_menu_tab("Time series")
    window._open_menu_tab("Driver explorer")
    open_labels = {t._menu_label for t in window._menu_tab_list}
    assert {"Time series", "Driver explorer"} <= open_labels

    folder = tmp_path / "Proj.diive"
    assert window._write_project(folder, "Proj")

    # Close everything, then reopening the project must restore the tabs.
    window._close_all_menu_tabs()
    assert window._menu_tab_list == []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(folder)))
    window._open_project()
    QApplication.processEvents()

    restored = {t._menu_label for t in window._menu_tab_list}
    assert {"Time series", "Driver explorer"} <= restored
    # Always-on tabs are never duplicated by the restore.
    assert _tabs(window)[:2] == ["Overview", "Log"]


def test_project_restores_per_tab_state(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    # Driver explorer with non-default settings.
    window._open_menu_tab("Driver explorer")
    drv = window._menu_tab_list[-1]
    target = str(window._data.select_dtypes("number").columns[0])
    drv.method.setCurrentText("Spearman")
    drv.max_lag.setValue(5)
    drv._on_select(target)
    # A time-series plot showing two specific variables + a non-default setting.
    window._open_menu_tab("Time series")
    ts = window._menu_tab_list[-1]
    cols = [str(c) for c in window._data.columns][:2]
    ts._on_selected(cols[0], False)
    ts._on_selected(cols[1], True)
    ts.settings.linewidth.setValue(4.5)
    ts.settings.fmt_title.setText("My plot")
    # Overview focused on a specific variable; the Time series tab is active.
    ovar = str(window._data.columns[3])
    window._tabs[0]._on_select(ovar)
    window._tabwidget.setCurrentWidget(ts.widget())
    QApplication.processEvents()

    folder = tmp_path / "P.diive"
    window._write_project(folder, "P")
    window._close_all_menu_tabs()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(folder)))
    window._open_project()
    QApplication.processEvents()

    drv2 = next(t for t in window._menu_tab_list if t._menu_label == "Driver explorer")
    assert drv2._target == target
    assert drv2.method.currentText() == "Spearman"
    assert drv2.max_lag.value() == 5
    ts2 = next(t for t in window._menu_tab_list if t._menu_label == "Time series")
    assert ts2._panels == cols  # the two selected variables came back
    assert ts2.settings.linewidth.value() == 4.5  # settings restored too
    assert ts2.settings.fmt_title.text() == "My plot"
    assert window._tabs[0]._current == ovar  # Overview selection restored
    # The previously-active tab regains focus (not just landing on Overview).
    cur = window._tabwidget.currentIndex()
    assert window._tabwidget.tabText(cur) == "Time series 1"


def test_metadata_namespace_migrates_legacy_flat_config():
    from diive.gui.app import _namespace_metadata
    # Legacy flat {name: [tags]} migrates under the first dataset key.
    out = _namespace_metadata({"NEE": ["favorite"]}, "site_x")
    assert out == {"site_x": {"tags": {"NEE": ["favorite"]}, "descriptions": {}}}
    # An already-namespaced blob passes through unchanged.
    ns = {"site_x": {"tags": {"NEE": ["favorite"]}, "descriptions": {}}}
    assert _namespace_metadata(ns, "other") == ns


def test_metadata_explorer_note_capped_at_50_words(window):
    from diive.gui import metadata_store

    window._open_menu_tab("Metadata explorer")
    tab = window._menu_tab_list[-1]
    var = str(window._data.columns[0])
    tab._select(var)
    QApplication.processEvents()

    assert not tab._desc_save.isEnabled()      # nothing to save on a fresh view
    long_note = " ".join(f"word{i}" for i in range(70))
    tab._desc_edit.setPlainText(long_note)
    assert tab._desc_save.isEnabled()          # editing re-enables it
    tab._save_description()  # what the "Save note" button does
    QApplication.processEvents()

    stored = metadata_store.manager.store.get(var).description
    assert len(stored.split()) == 50           # capped
    assert tab._desc_edit.toPlainText() == stored  # editor reflects truncation
    # Saved: button greys out and confirms; editing it again reactivates it.
    assert not tab._desc_save.isEnabled()
    assert tab._desc_save.text() == "Saved ✓"
    tab._desc_edit.setPlainText("changed again")
    assert tab._desc_save.isEnabled()
    assert tab._desc_save.text() == "Save note"

    # Switching variables flushes the note (no data loss without an explicit save).
    other = str(window._data.columns[1])
    tab._desc_edit.setPlainText("a short pending note")
    tab._select(other)
    QApplication.processEvents()
    assert metadata_store.manager.store.get(var).description == "a short pending note"


def test_localsd_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Local SD filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    # Run the worker directly (synchronous) instead of the background thread.
    series = window._data[var]
    tab._worker(series, dict(n_sd=7, winsize=480, constant_sd=False), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_LOCALSD", f"FLAG_{var}_OUTLIER_LOCALSD_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]

    n_before = window._data.shape[1]
    tab._add()  # what "Add cleaned + flag to dataset" does
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_absolutelimits_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Absolute limits filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    # Run the worker directly (synchronous) instead of the background thread.
    series = window._data[var]
    # Limits inside the data range so a few points fall outside and get flagged.
    lo, hi = float(series.quantile(0.01)), float(series.quantile(0.99))
    tab._worker(series, dict(minval=lo, maxval=hi), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_ABSLIM", f"FLAG_{var}_OUTLIER_ABSLIM_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]
    assert (window._data is not None) and (tab._result_df[flag] == 2).any()

    n_before = window._data.shape[1]
    tab._add()  # what "Add cleaned + flag to dataset" does
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_trimlow_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Trim-low filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    # Default mode trims the whole series with NO coordinates (day/night is opt-in).
    series = window._data[var]
    # Lower limit above the data minimum so some low values are trimmed (plus an
    # equal number of the highest values, by the symmetric-trim rule).
    ll = float(series.quantile(0.05))
    tab._worker(series, dict(lower_limit=ll, trim_daytime=False,
                             trim_nighttime=False), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_TRIMLOW", f"FLAG_{var}_OUTLIER_TRIMLOW_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]
    assert (tab._result_df[flag] == 2).any()

    n_before = window._data.shape[1]
    tab._add()  # what "Add cleaned + flag to dataset" does
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_trimlow_outlier_tab_daynight_split(window):
    """Opting into a day/night split passes coordinates and screens that period."""
    window._open_menu_tab("Trim-low filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    series = window._data[var]
    ll = float(series.quantile(0.05))
    tab._worker(series, dict(lower_limit=ll, trim_daytime=True, trim_nighttime=True,
                             lat=46.8, lon=9.8, utc_offset=1), True)
    QApplication.processEvents()
    flag = f"FLAG_{var}_OUTLIER_TRIMLOW_TEST"
    assert (tab._result_df[flag] == 2).any()
    assert set(tab._result_df[flag].dropna().unique()) <= {0, 2}


def test_zscore_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Z-score filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    # Run the worker directly (synchronous) instead of the background thread.
    series = window._data[var]
    tab._worker(series, dict(thres_zscore=4.0), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_ZSCORE", f"FLAG_{var}_OUTLIER_ZSCORE_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]

    n_before = window._data.shape[1]
    tab._add()  # what "Add cleaned + flag to dataset" does
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_zscorerolling_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Z-score (rolling) filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    series = window._data[var]
    tab._worker(series, dict(thres_zscore=4.0, winsize=480), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_ZSCOREROLLING", f"FLAG_{var}_OUTLIER_ZSCOREROLLING_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]

    n_before = window._data.shape[1]
    tab._add()
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_zscoreincrements_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Z-score (increments) filter")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    series = window._data[var]
    tab._worker(series, dict(thres_zscore=4.0), True)
    QApplication.processEvents()
    cleaned, flag = f"{var}_ZSCOREINCREMENTS", f"FLAG_{var}_OUTLIER_INCRZ_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]

    n_before = window._data.shape[1]
    tab._add()
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_manualremoval_outlier_tab_keeps_original_cleaned_flag(window):
    window._open_menu_tab("Manual removal")
    tab = window._menu_tab_list[-1]
    var = "Tair_f"
    tab._select(var)
    series = window._data[var]
    # Exercise the text-box parsing: a single timestamp, a whole-day bare date,
    # and a 'start to end' range -> the library's remove_dates list.
    single = series.index[10].strftime("%Y-%m-%d %H:%M:%S")
    day = series.index[0].strftime("%Y-%m-%d")
    rng = (f"{series.index[100].strftime('%Y-%m-%d %H:%M:%S')} to "
           f"{series.index[120].strftime('%Y-%m-%d %H:%M:%S')}")
    tab.dates_edit.setPlainText(f"{single}\n{day}\n{rng}")
    kwargs = tab._current_kwargs()
    assert any(isinstance(e, list) for e in kwargs["remove_dates"])  # range parsed
    tab._worker(series, kwargs, False)
    QApplication.processEvents()
    cleaned, flag = f"{var}_MANUAL", f"FLAG_{var}_OUTLIER_MANUAL_TEST"
    assert list(tab._result_df.columns) == [cleaned, flag]
    assert (tab._result_df[flag] == 2).sum() > 0  # records were flagged

    n_before = window._data.shape[1]
    tab._add()
    QApplication.processEvents()
    cols = [str(c) for c in window._data.columns]
    assert var in cols                 # original kept, untouched
    assert cleaned in cols and flag in cols
    assert window._data.shape[1] == n_before + 2
    assert set(window._data[flag].dropna().unique()) <= {0, 2}


def test_overview_and_log_not_pinnable(window):
    overview, log = window._tabs[0], window._tabs[1]
    window._toggle_pin(overview)
    window._toggle_pin(log)
    assert overview not in window._pinned
    assert log not in window._pinned


def test_live_theme_edit(window):
    from diive.gui import theme
    from diive.gui.widgets.variable_delegate import _pill_for
    theme.manager.pills["GPP"][1] = "#000000"
    theme.manager.apply()
    assert _pill_for("GPP_CUT_REF_f")[1].name() == "#000000"
    theme.manager.reset(silent=False)
    assert _pill_for("GPP_CUT_REF_f")[1].name() != "#000000"


def test_theme_persistence_roundtrip():
    from diive.gui import theme
    theme.manager.reset(silent=True)
    theme.manager.pills["GPP"][1] = "#abcdef"
    theme.manager.list_width = 321
    data = theme.manager.as_dict()
    theme.manager.reset(silent=True)
    theme.manager.load_dict(data)
    assert theme.manager.pills["GPP"][1] == "#abcdef"
    assert theme.manager.list_width == 321
    theme.manager.reset(silent=True)


def test_studio_look_defaults(app):
    # The GUI ships a single Studio look: uppercase tracked labels and the
    # canvas/ink/radius structural tokens build_qss reads.
    from diive.gui import theme
    theme.manager.reset(silent=True)
    assert theme.manager.typography["uppercase"] is True
    for key in ("CANVAS", "INK", "RADIUS"):
        assert key in theme.manager.tokens
    assert theme.manager.label_text("Open") == "OPEN"
    # No preset machinery remains.
    assert not hasattr(theme.manager, "set_preset")
    assert not hasattr(theme.manager, "chrome")


def test_theme_override_persists_through_roundtrip():
    from diive.gui import theme
    theme.manager.reset(silent=True)
    theme.manager.tokens["ACCENT"] = "#abcdef"  # user tweak on top of Studio
    data = theme.manager.as_dict()
    theme.manager.reset(silent=True)
    theme.manager.load_dict(data)
    assert theme.manager.tokens["ACCENT"] == "#abcdef"  # override survives
    theme.manager.reset(silent=True)


def test_old_classic_config_loads_into_studio():
    from diive.gui import theme
    theme.manager.reset(silent=True)
    # A config saved by the removed Classic look (a "preset" key + classic
    # tokens) must load into Studio: the preset key is ignored and the structural
    # tokens are re-pinned to the Studio defaults, while other overrides survive.
    theme.manager.load_dict(
        {"preset": "Classic", "tokens": {"RADIUS": "6", "ACCENT": "#2196F3"},
         "list_width": 240})
    assert theme.manager.tokens["RADIUS"] == "12"      # structural -> Studio
    assert theme.manager.tokens["ACCENT"] == "#2196F3"  # non-structural kept
    theme.manager.reset(silent=True)


def test_studio_chrome_builds_frameless_with_header(app, monkeypatch, example_year):
    from diive.gui import theme
    import diive
    monkeypatch.setattr(diive, "load_exampledata_parquet", lambda: example_year.copy())
    try:
        from diive.gui.app import MainWindow
        win = MainWindow()
        win.show()
        QApplication.processEvents()
        # Studio shell: a custom header replaces the native menu bar, and the
        # window is frameless. The tab structure is unchanged.
        assert win._header is not None
        assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert _tabs(win) == ["Overview", "Log"]
        # The full menu tree lives as inline dropdown buttons in the header
        # (File/Data/Plot/Outliers/Flux/Analyze/Settings/Help), each with a populated QMenu.
        from PySide6.QtWidgets import QToolButton
        menu_btns = win._header.findChildren(QToolButton, "headermenu")
        assert len(menu_btns) == 8
        assert all(b.menu() is not None and b.menu().actions() for b in menu_btns)
        # Open and Save live inside the File menu (no separate buttons).
        file_items = [a.text() for a in menu_btns[0].menu().actions()]
        assert any("Open" in t for t in file_items)
        assert any("Save" in t for t in file_items)
        win.close()
    finally:
        theme.manager.reset(silent=True)


def test_overview_hero_stats(window):
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    QApplication.processEvents()
    # All stats live in the hero band (no bottom ribbon). The persistent slots
    # are keyed by label and populated with non-empty values.
    hero = overview.hero
    assert hero._name.text() == "NEE_CUT_REF_f"
    for label in ("STARTDATE", "PERIOD", "NOV", "COVERAGE", "GAPS",
                  "MEAN ± SD", "SUM", "MEDIAN", "P01", "P99", "MAX"):
        assert label in hero._slots
        assert hero._slots[label]._value.text()  # non-empty
    assert not hasattr(overview, "stats_layout")  # bottom ribbon removed


def test_to_diive_format_flattens_and_names():
    df = dv.load_exampledata_parquet()
    out = dv.to_diive_format(df, timestamp_name="TIMESTAMP_MIDDLE")
    assert out.columns.nlevels == 1
    assert out.index.name == "TIMESTAMP_MIDDLE"


def test_to_diive_format_requires_valid_name():
    df = dv.load_exampledata_parquet()
    df.index.name = "something_else"
    with pytest.raises(ValueError):
        dv.to_diive_format(df, timestamp_name=None)


def test_save_parquet_diive_format_roundtrip(tmp_path):
    df = dv.load_exampledata_parquet()
    path = dv.save_parquet(filename="gui_test", data=df, outpath=str(tmp_path),
                           enforce_diive_format=True, timestamp_name="TIMESTAMP_MIDDLE")
    reloaded = dv.load_parquet(filepath=path)
    assert reloaded.index.name in ("TIMESTAMP_MIDDLE", "TIMESTAMP_END")
    assert reloaded.columns.nlevels == 1
    assert reloaded.shape[1] == df.shape[1]


def test_event_creates_flag_column(window):
    from diive.gui import events
    from diive.events import Event
    events.manager.clear()
    start = window._full_data.index.min()
    events.manager.add(Event("Fert1", start + pd.Timedelta("10D"),
                             category="fertilization"))
    events.manager.add(Event("Graze", start + pd.Timedelta("20D"),
                             start + pd.Timedelta("23D"), category="grazing"))
    cols = [c for c in window._full_data.columns if str(c).startswith("EVENT_")]
    assert set(cols) == {"EVENT_Fert1", "EVENT_Graze"}
    assert window._full_data["EVENT_Fert1"].sum() == 1     # instant -> one record
    assert window._full_data["EVENT_Graze"].sum() > 1      # period -> many records
    assert {"EVENT_Fert1", "EVENT_Graze"} <= window._created
    events.manager.clear()


def test_event_removal_drops_only_owned_column(window):
    from diive.gui import events
    from diive.events import Event
    events.manager.clear()
    # A plain EVENT_-named data column not backed by an event must survive.
    window._full_data["EVENT_External"] = 0
    events.manager.add(Event("Mine", window._full_data.index.min()))
    assert "EVENT_Mine" in window._full_data.columns
    events.manager.clear()  # removes only the event-backed column
    assert "EVENT_Mine" not in window._full_data.columns
    assert "EVENT_External" in window._full_data.columns
    window._full_data.drop(columns=["EVENT_External"], inplace=True)


def test_event_visibility_toggle(window):
    from diive.gui import events
    events.manager.set_visible(False)
    assert window._show_events_act.isChecked() is False
    events.manager.set_visible(True)
    assert window._show_events_act.isChecked() is True


def test_events_tab_and_dialog_build(window):
    from diive.gui.tabs.events import EventsTab
    from diive.gui.widgets.add_event_dialog import AddEventDialog
    tab = EventsTab()
    tab.widget()
    tab.on_data_loaded(window._full_data)
    dlg = AddEventDialog(window._full_data.index.min(), window._full_data.index.max())
    ev = dlg.make_event()  # default = instant at data start
    assert not ev.is_range


def test_mds_gapfill_to_code():
    # The MDS codegen renders a runnable FluxMDS(...).run() snippet with the three
    # drivers + tolerances; no feature list and no SHAP reduction (unlike ML).
    from diive.gapfilling.codegen import mds_gapfill_to_code
    code = mds_gapfill_to_code(
        "NEE", "Rg_f", "Tair_f", "VPD_f",
        {"swin_tol": [20, 50], "ta_tol": 2.5, "vpd_tol": 0.5, "avg_min_n_vals": 5})
    assert "dv.gapfilling.FluxMDS(" in code
    assert "flux=target" in code and "swin='Rg_f'" in code
    assert "ta='Tair_f'" in code and "vpd='VPD_f'" in code
    assert "ta_tol=2.5" in code and "model.run()" in code
    assert "reduce_features" not in code  # MDS has none
    assert "_gfMDS" in code


def test_randunc_to_code():
    # The random-uncertainty codegen renders a runnable
    # RandomUncertaintyPAS20(...).run() snippet from the five inputs + VPD unit.
    from diive.flux.lowres.codegen import randunc_to_code
    code = randunc_to_code(
        fluxcol="NEE_CUT_REF_orig", fluxgapfilledcol="NEE_CUT_REF_f",
        tacol="Tair_f", vpdcol="VPD_f", swincol="Rg_f", vpd_in_kpa=False)
    assert "dv.flux.RandomUncertaintyPAS20(" in code
    assert "fluxcol=flux" in code and "fluxgapfilledcol='NEE_CUT_REF_f'" in code
    assert "tacol='Tair_f'" in code and "vpdcol='VPD_f'" in code and "swincol='Rg_f'" in code
    assert "vpd_in_kpa=False" in code and "randunc.run()" in code
    assert "_RANDUNC" in code


def test_mds_quality_breakdown_helper():
    # The library exposes the per-quality-level breakdown (level/count/pct/desc)
    # the GUI plot reads, keeping the level->description map in the library.
    from diive.gapfilling.mds import FluxMDS, mds_quality_description
    df = dv.load_exampledata_parquet()
    df = dv.times.keep_daterange(df, "2022-07-01", "2022-07-31 23:30")
    model = FluxMDS(df=df, flux="NEE_CUT_REF_orig", swin="Rg_f", ta="Tair_f",
                    vpd="VPD_f", verbose=0)
    # The progress callback fires per quality level and reaches done == total.
    progress = []
    model.run(progress_callback=lambda *a: progress.append(a))
    assert progress and progress[-1][0] == progress[-1][1]  # last done == total
    bd = model.quality_breakdown()
    assert list(bd.columns) == ["level", "count", "pct", "description"]
    assert (bd["level"] == 0).any()  # observed row present
    assert int(bd["count"].sum()) == len(df)
    assert mds_quality_description(0) == "measured"
    # Granular flag = method*1000 + time_window (days): 2014 = SWIN-only, 14 d.
    desc = mds_quality_description(2014)
    assert "SWIN only" in desc and "14" in desc

    # The colour/marker-by-quality time series embeds into a supplied axes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    model.plot_quality_timeseries(ax=ax)
    assert ax.get_lines()  # drew at least one quality series
    plt.close(fig)


def test_random_uncertainty_tab(app, example_year):
    # The random-uncertainty tab builds, auto-seeds the five inputs, runs the
    # PAS20 cascade synchronously and emits a single {flux}_RANDUNC column.
    from PySide6.QtWidgets import QApplication
    from diive.gui.tabs.uncertainty_randunc import RandomUncertaintyTab
    # One month keeps the per-record cascade fast in the test.
    df = dv.times.keep_daterange(example_year, "2021-03-01", "2021-03-31 23:30")
    tab = RandomUncertaintyTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Inputs auto-seed to present columns; the measured flux avoids the _F column.
    picks = tab._picks()
    assert all(v in df.columns for v in picks.values())
    assert picks["flux"] != picks["flux_f"]

    # Copy Python renders a runnable snippet reflecting the current picks.
    code = tab._python_code()
    assert "RandomUncertaintyPAS20(" in code and picks["flux"] in code

    # Drive the worker synchronously: _compute_payload off-thread, then _on_done.
    payload = tab._compute_payload(df[[picks["flux"], picks["flux_f"], picks["ta"],
                                       picks["vpd"], picks["swin"]]].copy(),
                                   picks, tab.vpd_in_kpa.isChecked())
    tab._on_done(payload)
    QApplication.processEvents()
    assert tab._result_df is not None
    assert list(tab._result_df.columns) == [f"{picks['flux']}_RANDUNC"]
    assert tab.add_btn.isEnabled()
    # The preview rendered without a failure message.
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Plot failed" in t.get_text()]
    # Three panels incl. the measured-flux vs uncertainty scatter.
    titles = [a.get_title() for a in tab.canvas.fig.axes]
    assert any("Uncertainty vs flux" in t for t in titles)

    # Config round-trips through save/restore (the input picks + VPD unit).
    tab.vpd_in_kpa.setChecked(False)
    state = tab.save_state()
    tab2 = RandomUncertaintyTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2._picks()["flux"] == picks["flux"]
    assert tab2.vpd_in_kpa.isChecked() is False


def test_joint_uncertainty_tab(app, example_year):
    # The joint-uncertainty tab builds, auto-seeds randunc + the 16th/84th USTAR
    # scenarios biased to the randunc flux, runs synchronously and emits a single
    # {base}_JOINTUNC column.
    from PySide6.QtWidgets import QApplication
    from diive.gui.tabs.uncertainty_jointunc import JointUncertaintyTab
    df = dv.times.keep_daterange(example_year, "2021-03-01", "2021-03-31 23:30").copy()
    # Fabricate the RANDUNC column the tab needs (the cascade itself is tested
    # in test_uncertainty.py; here only the joint plumbing matters).
    df["NEE_CUT_REF_RANDUNC"] = 1.5
    tab = JointUncertaintyTab()
    tab.widget()
    tab.on_data_loaded(df)

    # Scenario picks bias to the randunc flux base (NEE, not GPP).
    picks = tab._picks()
    assert all(v in df.columns for v in picks.values())
    assert picks["randunc"] == "NEE_CUT_REF_RANDUNC"
    assert picks["lower"] == "NEE_CUT_16_f" and picks["upper"] == "NEE_CUT_84_f"
    assert tab._divisor() == 2.0

    code = tab._python_code()
    assert "JointUncertaintyPAS20(" in code and "NEE_CUT_16_f" in code

    payload = tab._compute_payload(
        df[[picks["randunc"], picks["lower"], picks["upper"], picks["flux_f"]]].copy(),
        picks, tab._divisor())
    tab._on_done(payload)
    QApplication.processEvents()
    assert tab._result_df is not None
    assert list(tab._result_df.columns) == ["NEE_CUT_REF_JOINTUNC"]
    assert tab.add_btn.isEnabled()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Plot failed" in t.get_text()]
    titles = [a.get_title() for a in tab.canvas.fig.axes]
    assert any("decomposition" in t.lower() for t in titles)

    # Switching the percentile convention re-picks scenarios + the divisor.
    tab.divisor_combo.setCurrentIndex(1)
    assert tab._divisor() != 2.0


def test_gapfilling_mds_tab(app, example_year):
    # The MDS tab builds, auto-seeds the three drivers, runs synchronously and
    # emits the *_gfMDS + flag columns; the slimmed Results panel populates.
    from PySide6.QtWidgets import QApplication
    from diive.gui.tabs.gapfilling_mds import MdsGapFillingTab
    df = example_year
    tab = MdsGapFillingTab()
    tab.widget()
    tab.on_data_loaded(df)
    tab._set_target("NEE_CUT_REF_orig")

    # Drivers auto-seed to present columns (availability marker ✓).
    drivers = tab._driver_names()
    assert all(v in df.columns for v in drivers.values())
    assert "VPD" in drivers["vpd"].upper() and "TA" in drivers["ta"].upper()

    # Reproducible snippet reflects the current target + drivers.
    code = tab._python_code()
    assert "FluxMDS(" in code and "NEE_CUT_REF_orig" in code

    # Pin the drivers and run the worker synchronously (the runner forwards to
    # _on_done; calling _compute_payload + _on_done directly keeps it off-thread).
    swin, ta, vpd = "Rg_f", "Tair_f", "VPD_f"
    work = df[["NEE_CUT_REF_orig", swin, ta, vpd]].copy()
    payload = tab._compute_payload(work, "NEE_CUT_REF_orig", swin, ta, vpd,
                                   tab._method_kwargs())
    tab._on_done(payload)
    QApplication.processEvents()
    assert tab._result_df is not None
    assert "NEE_CUT_REF_orig_gfMDS" in tab._result_df.columns
    assert "FLAG_NEE_CUT_REF_orig_gfMDS_ISFILLED" in tab._result_df.columns
    assert tab.add_btn.isEnabled()
    # Heatmaps + results panel rendered without a failure message.
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Plot failed" in t.get_text()]

    # A wheel over a results-panel plot must scroll the dashboard: every embedded
    # canvas resolves the enclosing scroll area so it can forward wheel events.
    from diive.gui.widgets.mpl_canvas import MplCanvas
    panel_canvases = tab.results_panel.findChildren(MplCanvas)
    assert panel_canvases
    assert all(c._enclosing_scroll_area() is tab.results_panel for c in panel_canvases)

    # Config round-trips through save/restore (target + drivers + tolerances).
    tab.vpd_tol.setValue(0.8)
    state = tab.save_state()
    tab2 = MdsGapFillingTab(); tab2.widget(); tab2.on_data_loaded(df)
    tab2.restore_state(state)
    assert tab2._target == "NEE_CUT_REF_orig"
    assert tab2.vpd_tol.value() == 0.8
