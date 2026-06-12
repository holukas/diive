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
    tab.settings.axlabels_fontsize.setValue(8)
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
    assert {"linewidth", "alpha", "marker", "drop_gaps", "markersize", "title",
            "_axes"} <= set(ts.settings.values())
    ts.settings.marker.setChecked(True)
    ts.settings.drop_gaps.setChecked(True)
    ts.settings.linewidth.setValue(4.0)
    ts.settings.markersize.setValue(6.0)
    ts.settings.series_units.setText("umol")
    ts.settings.title.setText("Custom title")
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


def test_overview_layout_frozen_on_zoom(window):
    # Constrained layout reflows on every draw, making panels jump while
    # zooming; the canvas freezes it after the initial render so zoom/pan stay
    # stable. Verify a simulated zoom does not move any panel.
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    QApplication.processEvents()
    fig = overview.canvas.fig
    before = [tuple(a.get_position().bounds) for a in fig.axes]
    ts_ax = fig.axes[0]
    x0, x1 = ts_ax.get_xlim()
    ts_ax.set_xlim(x0 + (x1 - x0) * 0.3, x0 + (x1 - x0) * 0.6)  # zoom in
    overview.canvas.draw()
    QApplication.processEvents()
    after = [tuple(a.get_position().bounds) for a in fig.axes]
    assert before == after  # panels stayed put

    # ...but a resize must re-solve the frozen layout to the new size (otherwise
    # a layout computed at the tiny pre-show size stays collapsed). After a
    # resize the bottom panels should have a sensible (non-collapsed) width.
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
    assert {"mean", "std", "each_month", "legend_loc"} <= set(tab.settings.values())
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
    tab.settings.dc_legend_loc.setCurrentText("upper right")
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab.settings.values()["legend_loc"] == "upper right"
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]


def test_scatter_tab(window):
    from diive.gui.icons import menu_icon
    assert not menu_icon("Scatter XY").isNull()
    window._open_menu_tab("Scatter XY")
    tab = window._menu_tab_list[-1]
    # Seeded with X, Y (2 vars) -> plain scatter renders.
    assert len(tab._xyz) >= 2
    assert not [t for a in tab.canvas.fig.axes for t in a.texts if "Cannot plot" in t.get_text()]
    assert {"nbins", "binagg", "cmap", "show_colorbar", "markersize", "alpha",
            "vmin", "vmax", "title", "_axes"} <= set(tab.settings.values())
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
    assert tab._current == "NEE_CUT_84_orig"
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
    assert not menu_icon("Seasonal-trend & anomalies").isNull()

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
    window._open_menu_tab("Seasonal-trend & anomalies")
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
    ts.settings.title.setText("My plot")
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
    assert ts2.settings.title.text() == "My plot"
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


def test_overview_stats_cards(window):
    overview = window._tabs[0]
    overview._on_select("NEE_CUT_REF_f")
    QApplication.processEvents()
    # Cards = layout items minus the trailing stretch.
    assert overview.stats_layout.count() - 1 > 5


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
