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
    assert _pill_for("TA_f") is None


def test_multi_instance_plot_tabs(window):
    window._open_menu_tab("Heatmap date/time")
    window._open_menu_tab("Heatmap date/time")
    window._open_menu_tab("Time series")
    assert "Heatmap date/time 1" in _tabs(window)
    assert "Heatmap date/time 2" in _tabs(window)
    assert "Time series 1" in _tabs(window)


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

    # Toggle a spread of non-default settings; each edit re-renders live and the
    # heatmap must still draw (no error-fallback text).
    tab.settings.cmap.setCurrentText("viridis")
    tab.settings.orientation.setCurrentText("horizontal")
    tab.settings.vmin.setText("-5")
    tab.settings.vmax.setText("5")
    tab.settings.show_values.setChecked(True)
    tab.settings.cb_extend.setCurrentText("both")
    tab.settings.axlabels_fontsize.setValue(8)
    QApplication.processEvents()
    assert tab.settings.values()["cmap"] == "viridis"
    assert not _fallback(tab)

    window._open_menu_tab("Time series")
    ts = window._menu_tab_list[-1]
    assert {"linewidth", "alpha", "marker", "drop_gaps"} <= set(ts.settings.values())
    ts.settings.marker.setChecked(True)
    ts.settings.drop_gaps.setChecked(True)
    ts.settings.linewidth.setValue(4.0)
    ts.settings.series_units.setText("umol")
    QApplication.processEvents()
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
    QApplication.processEvents()
    assert not _fallback(ym)


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


def test_flux_chain_tab_level2(app):
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

    # Copy-Python emits runnable composable code matching what Run does.
    code = tab._code()
    compile(code, "<gen>", "exec")
    assert "run_level2(" in code and "init_flux_data(" in code

    # Run Level 2 (synchronous core) and render into the canvas.
    data = tab._compute(df, tab._init_kwargs(), tab._level2_settings())
    assert data.filteredseries is not None
    assert data.filteredseries.dropna().count() > 0
    tab._on_done(data)
    QApplication.processEvents()
    assert not [t for a in tab.canvas.fig.axes for t in a.texts
                if "Cannot plot" in t.get_text()]


def test_all_menu_items_have_icons(window):
    # Every (non-separator) menu entry carries a drawn icon.
    menubar = window.menuBar()
    count = 0
    for menu_action in menubar.actions():
        menu = menu_action.menu()
        if menu is None:
            continue
        for action in menu.actions():
            if action.isSeparator():
                continue
            count += 1
            assert not action.icon().isNull(), action.text()
    assert count >= 12  # File/Data/Plot/Tools/Settings/Help entries


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


def test_appearance_singleton(window):
    window._open_menu_tab("Appearance")
    window._open_menu_tab("Appearance")
    assert _tabs(window).count("Appearance") == 1


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
