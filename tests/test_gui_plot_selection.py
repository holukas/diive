"""
TEST_GUI_PLOT_SELECTION: plot tabs keep their setup when the data change
========================================================================

Adding a column in any tab pushes new data to the open plot tabs. The tabs
used to reset their panels, X/Y roles and per-panel settings to the defaults
on every push (L338). They now keep everything for the variables still in the
data and drop only what is gone.

Run: pytest tests/test_gui_plot_selection.py -v
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc
import sys
import threading
import traceback

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import diive as dv


@pytest.fixture(autouse=True)
def slot_exceptions():
    """Fail the test if a Qt-invoked slot or worker thread raised.

    PySide6 hands such exceptions to ``sys.excepthook`` and carries on, so a
    test would otherwise pass with a crashed handler (copied from
    tests/test_gui.py)."""
    captured = []

    def _hook(etype, value, tb):
        captured.append("".join(traceback.format_exception(etype, value, tb)))

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        captured.append("".join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback)))

    prev_hook, prev_thread_hook = sys.excepthook, threading.excepthook
    sys.excepthook, threading.excepthook = _hook, _thread_hook
    try:
        yield captured
    finally:
        sys.excepthook, threading.excepthook = prev_hook, prev_thread_hook
    if captured:
        pytest.fail(
            f"{len(captured)} exception(s) raised inside a Qt slot or worker "
            "thread and swallowed by PySide6:\n\n" + "\n".join(captured),
            pytrace=False)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def example_year():
    df = dv.load_exampledata_parquet()
    return dv.times.keep_daterange(df, "2021-01-01", "2021-12-31 23:30")


def _destroy_window(win):
    """Destroy a MainWindow completely (see tests/test_gui.py for why)."""
    QApplication.processEvents()
    win.close()
    shiboken6.delete(win)
    win._tabs.clear()
    win._menu_tab_list.clear()
    win._pinned.clear()
    gc.collect()
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def window(app, monkeypatch, example_year):
    import diive
    from diive.gui import config
    monkeypatch.setattr(config, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(diive, "load_exampledata_parquet",
                        lambda: example_year.copy())
    # No dialog may block the run: answer every question and warning at once,
    # including the "Save changes?" prompt on close. A test that needs "Yes"
    # patches `question` again.
    discard = QMessageBox.StandardButton.Discard
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: discard)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr("diive.gui.app.ask_save_changes",
                        lambda *a, **k: "discard", raising=False)
    from diive.gui.app import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()
    yield win
    _destroy_window(win)
    from diive.gui import events as _events
    _events.manager.events.clear()
    _events.manager.visible = True
    _events.manager.categories = dict(_events.DEFAULT_CATEGORIES)


def _axes_replaced(canvas, previous):
    """True when a real re-render happened since `previous` was captured
    (copied from tests/test_gui.py)."""
    return all(ax not in canvas.fig.axes for ax in previous)


def _settle(tab):
    """Run the deferred render of a list click (VariablePanel.run_with_loading)."""
    tab.varpanel.flush_pending()
    QApplication.processEvents()


def _open(win, label):
    win._open_menu_tab(label)
    tab = win._menu_tab_list[-1]
    _settle(tab)
    return tab


def _add_column(win, name="BRANDNEW"):
    """Add a column the way a feature tab does (a data push to the open tabs)."""
    new = win._full_data[["Tair_f"]].rename(columns={"Tair_f": name}) * 2
    win._add_features(new)
    QApplication.processEvents()


def _delete_column(win, monkeypatch, name):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    win._delete_variable(name)
    QApplication.processEvents()
    assert name not in win._data.columns


def _texts(canvas):
    return [t.get_text() for ax in canvas.fig.axes for t in ax.texts]


def _two_panel_timeseries(win):
    tab = _open(win, "Time series")
    tab._on_selected("GPP_DT_CUT_84", False)
    _settle(tab)
    tab._on_selected("GPP_DT_CUT_REF", True)
    _settle(tab)
    # A setting of the second (active) panel only, applied with Update plot.
    tab.settings.linewidth.setValue(5.0)
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab._panels == ["GPP_DT_CUT_84", "GPP_DT_CUT_REF"]
    return tab


def test_timeseries_keeps_two_panels_on_data_push(window):
    tab = _two_panel_timeseries(window)
    snaps = {n: list(s) for n, s in tab._panel_settings.items()}
    assert snaps["GPP_DT_CUT_84"] != snaps["GPP_DT_CUT_REF"]
    prev_axes = list(tab.canvas.fig.axes)

    _add_column(window)

    assert "BRANDNEW" in tab._df.columns
    assert "BRANDNEW" in tab.varpanel.names()
    assert tab._panels == ["GPP_DT_CUT_84", "GPP_DT_CUT_REF"]
    assert tab._panel_settings == snaps
    assert tab._active_panel == "GPP_DT_CUT_REF"
    assert tab.settings.linewidth.value() == 5.0
    assert tab.panel_pills.isVisibleTo(tab.widget())
    # Re-rendered with both panels, titled by their variables.
    assert _axes_replaced(tab.canvas, prev_axes)
    titles = [ax.get_title(loc) for ax in tab.canvas.fig.axes
              for loc in ("left", "center")]
    assert any("GPP_DT_CUT_84" in t for t in titles)
    assert any("GPP_DT_CUT_REF" in t for t in titles)
    assert not any(t.startswith("Cannot plot") for t in _texts(tab.canvas))
    assert not tab.update_btn.isEnabled()


def test_timeseries_delete_drops_only_that_panel(window, monkeypatch):
    tab = _two_panel_timeseries(window)
    first = list(tab._panel_settings["GPP_DT_CUT_84"])
    prev_axes = list(tab.canvas.fig.axes)

    _delete_column(window, monkeypatch, "GPP_DT_CUT_REF")

    assert tab._panels == ["GPP_DT_CUT_84"]
    assert list(tab._panel_settings) == ["GPP_DT_CUT_84"]
    assert tab._panel_settings["GPP_DT_CUT_84"] == first
    assert tab._active_panel == "GPP_DT_CUT_84"
    # The controls now show the kept panel's settings, not the deleted one's.
    assert tab.settings.state() == first
    assert not tab.panel_pills.isVisibleTo(tab.widget())
    assert _axes_replaced(tab.canvas, prev_axes)
    assert "GPP_DT_CUT_REF" not in tab.varpanel.names()


def test_timeseries_deleted_colorby_variable_is_dropped(window, monkeypatch):
    tab = _open(window, "Time series")
    tab.settings.color_by.setCurrentText("Tair_f")
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab.settings.values()["color_by"] == "Tair_f"

    _delete_column(window, monkeypatch, "Tair_f")

    assert tab._panels == ["NEE_CUT_REF_f"]
    assert tab.settings.values()["color_by"] is None
    assert tab.settings.color_by.findText("Tair_f") < 0


def test_timeseries_deleted_colorby_of_inactive_panel(window, monkeypatch):
    # The inactive panel's colour-by variable is deleted: that panel must fall
    # back to no colour-by, not take the active panel's pick.
    tab = _two_panel_timeseries(window)
    tab.settings.color_by.setCurrentText("VPD_f")       # panel 2 (active)
    tab._on_pill_changed(0)
    tab.settings.color_by.setCurrentText("Tair_f")      # panel 1
    tab._on_pill_changed(1)
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab._active_panel == "GPP_DT_CUT_REF"

    _delete_column(window, monkeypatch, "Tair_f")

    assert tab._panels == ["GPP_DT_CUT_84", "GPP_DT_CUT_REF"]
    assert tab.settings.values()["color_by"] == "VPD_f"
    tab._on_pill_changed(0)
    assert tab.settings.values()["color_by"] is None


def test_timeseries_falls_back_to_default_when_nothing_is_left(window, monkeypatch):
    tab = _open(window, "Time series")
    tab._on_selected("GPP_DT_CUT_84", False)
    _settle(tab)
    prev_axes = list(tab.canvas.fig.axes)

    _delete_column(window, monkeypatch, "GPP_DT_CUT_84")

    assert tab._panels == ["NEE_CUT_REF_f"]
    assert list(tab._panel_settings) == ["NEE_CUT_REF_f"]
    assert tab._active_panel == "NEE_CUT_REF_f"
    assert _axes_replaced(tab.canvas, prev_axes)


def _scatter(win, x, y, z=None):
    tab = _open(win, "Scatter XY")
    tab.settings.sc_x.setCurrentText(x)
    tab.settings.sc_y.setCurrentText(y)
    if z is not None:
        tab.settings.sc_z.setCurrentText(z)
    tab.settings.sc_markersize.setValue(120.0)
    tab.update_btn.click()
    QApplication.processEvents()
    return tab


def test_scatter_keeps_roles_on_data_push(window):
    tab = _scatter(window, "GPP_CUT_REF_f", "GPP_DT_CUT_16")
    assert tab._xyz == ["GPP_CUT_REF_f", "GPP_DT_CUT_16"]
    prev_axes = list(tab.canvas.fig.axes)

    _add_column(window)

    assert tab._xyz == ["GPP_CUT_REF_f", "GPP_DT_CUT_16"]
    assert tab.settings.xyz_values() == ["GPP_CUT_REF_f", "GPP_DT_CUT_16"]
    assert tab.settings.sc_x.findText("BRANDNEW") >= 0
    assert tab.settings.sc_markersize.value() == 120.0
    assert _axes_replaced(tab.canvas, prev_axes)
    ax = tab.canvas.fig.axes[0]
    assert "GPP_CUT_REF_f" in ax.get_xlabel()
    assert "GPP_DT_CUT_16" in ax.get_ylabel()


def test_scatter_delete_drops_only_that_role(window, monkeypatch):
    tab = _scatter(window, "GPP_CUT_REF_f", "GPP_DT_CUT_16", z="VPD_f")
    assert tab._xyz == ["GPP_CUT_REF_f", "GPP_DT_CUT_16", "VPD_f"]

    # The optional colour role is left empty.
    _delete_column(window, monkeypatch, "VPD_f")
    assert tab._xyz == ["GPP_CUT_REF_f", "GPP_DT_CUT_16"]
    assert tab.settings.xyz_values() == ["GPP_CUT_REF_f", "GPP_DT_CUT_16"]

    # A required role takes its default, the other role stays.
    prev_axes = list(tab.canvas.fig.axes)
    _delete_column(window, monkeypatch, "GPP_DT_CUT_16")
    assert tab._xyz == ["GPP_CUT_REF_f", "NEE_CUT_REF_f"]
    assert tab.settings.xyz_values() == ["GPP_CUT_REF_f", "NEE_CUT_REF_f"]
    assert _axes_replaced(tab.canvas, prev_axes)
    assert "NEE_CUT_REF_f" in tab.canvas.fig.axes[0].get_ylabel()


def test_heatmap_keeps_variable_and_settings_on_data_push(window):
    tab = _open(window, "Heatmap date/time")
    tab._on_selected("GPP_DT_CUT_84", False)
    _settle(tab)
    tab.settings.cmap.setCurrentText("viridis")
    tab.update_btn.click()
    QApplication.processEvents()
    snap = list(tab._panel_settings["GPP_DT_CUT_84"])
    prev_axes = list(tab.canvas.fig.axes)

    _add_column(window)

    assert tab._panels == ["GPP_DT_CUT_84"]
    assert tab._panel_settings == {"GPP_DT_CUT_84": snap}
    assert tab.settings.cmap.currentText() == "viridis"
    assert _axes_replaced(tab.canvas, prev_axes)
    # The heatmap is drawn as an image (as_image).
    assert tab.canvas.fig.axes[0].get_images()[0].get_cmap().name == "viridis"


@pytest.mark.parametrize("label", [
    "Heatmap year/month", "Diel cycle", "Cumulative year", "Cumulative",
    "Ridgeline", "Histogram", "Shifted distribution", "Tree ring", "Waterfall",
])
def test_single_and_multi_panel_types_keep_variable(window, label):
    tab = _open(window, label)
    tab._on_selected("Tair_f", False)
    _settle(tab)
    prev_axes = list(tab.canvas.fig.axes)

    _add_column(window)

    assert tab._panels == ["Tair_f"]
    assert list(tab._panel_settings) == ["Tair_f"]
    assert _axes_replaced(tab.canvas, prev_axes)


@pytest.mark.parametrize("label", ["Hexbin", "Heatmap x/y/z", "Wind rose"])
def test_role_types_keep_roles(window, label):
    tab = _open(window, label)
    roles = ["Tair_f", "VPD_f", "GPP_DT_CUT_84"]
    tab.settings.set_xyz(*roles)
    tab._on_xyz_changed()
    tab.update_btn.click()
    QApplication.processEvents()
    assert tab._xyz == roles
    prev_axes = list(tab.canvas.fig.axes)

    _add_column(window)

    assert tab._xyz == roles
    assert tab.settings.xyz_values() == roles
    assert _axes_replaced(tab.canvas, prev_axes)


def test_new_tab_and_restore_unchanged(window):
    # First load still shows the default, and a saved setup restores into a
    # fresh tab as before.
    tab = _two_panel_timeseries(window)
    state = tab.save_state()
    fresh = _open(window, "Time series")
    assert fresh._panels == ["NEE_CUT_REF_f"]
    fresh.restore_state(state)
    _settle(fresh)
    assert fresh._panels == ["GPP_DT_CUT_84", "GPP_DT_CUT_REF"]
    assert fresh._panel_settings == tab._panel_settings
    assert fresh._active_panel == "GPP_DT_CUT_REF"


# --- 3-D surface tabs ---------------------------------------------------------

def _stub_pyvista_canvas(monkeypatch):
    """Replace the 3-D tabs' GL canvas with a recording stub.

    VTK cannot create an OpenGL window under the offscreen Qt platform, so the
    real `Pyvista3DCanvas` can't be built in tests. Mesh construction is pure
    CPU work, so a stub plotter lets the render pipeline run (copied from
    tests/test_gui.py).
    """
    pytest.importorskip("pyvista")
    from PySide6.QtWidgets import QWidget

    import diive.gui.tabs.surface3d as surface3d

    class _StubPlotter:
        def __init__(self):
            self.meshes = []

        def clear(self):
            self.meshes.clear()

        def add_mesh(self, mesh, **_kw):
            self.meshes.append(mesh)

        def show_axes(self):
            pass

    class _StubCanvas(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.plotter = _StubPlotter()

        def on_interaction_start(self, _cb):
            pass

        def on_first_show(self, _cb):
            pass

        def clear(self):
            self.plotter.clear()

        def render(self):
            pass

        def set_view(self, *_a, **_kw):
            pass

        def apply_shadows(self, *_a, **_kw):
            pass

    monkeypatch.setattr(surface3d, "Pyvista3DCanvas", _StubCanvas)


def _surface3d(win, monkeypatch, name):
    _stub_pyvista_canvas(monkeypatch)
    tab = _open(win, "3D surface")
    tab._on_select(name)
    _settle(tab)
    tab.ybin.setValue(3)
    tab.cmap.setCurrentText("viridis")
    tab._rerender_debounce.flush()
    QApplication.processEvents()
    assert tab._target == name and tab._framed_target == name
    assert tab._grid_height is not None
    return tab


def test_surface3d_keeps_variable_and_settings_on_data_push(window, monkeypatch):
    tab = _surface3d(window, monkeypatch, "Tair_f")
    prev_height = tab._grid_height

    _add_column(window)

    assert "BRANDNEW" in tab.varpanel.names()
    assert tab._target == "Tair_f"
    assert tab._framed_target == "Tair_f"  # camera view kept
    assert tab.ybin.value() == 3
    assert tab.cmap.currentText() == "viridis"
    # Re-rendered from the pushed data.
    assert tab._grid_height is not None and tab._grid_height is not prev_height


def test_surface3d_falls_back_to_default_when_variable_is_gone(window, monkeypatch):
    tab = _surface3d(window, monkeypatch, "GPP_DT_CUT_84")

    _delete_column(window, monkeypatch, "GPP_DT_CUT_84")

    assert tab._target == tab.default_var == "NEE_CUT_REF_f"
    assert tab._framed_target == "NEE_CUT_REF_f"
    assert tab.ybin.value() == 3
    assert tab._grid_height is not None


def _surfacexyz(win, monkeypatch, x, y, z):
    _stub_pyvista_canvas(monkeypatch)
    tab = _open(win, "3D surface (X/Y/Z)")
    combos = tab.picker.combos()
    for key, name in zip(("x", "y", "z"), (x, y, z), strict=True):
        combos[key].setCurrentText(name)
    tab.nbins.setValue(12)
    tab.agg.setCurrentText("median")
    tab._rerender_debounce.flush()
    QApplication.processEvents()
    assert tab.picker.picks() == {"x": x, "y": y, "z": z}
    assert tab._framed_target == (x, y, z)
    return tab


def test_surfacexyz_keeps_roles_and_settings_on_data_push(window, monkeypatch):
    roles = ("VPD_f", "Tair_f", "GPP_DT_CUT_84")
    tab = _surfacexyz(window, monkeypatch, *roles)
    prev_height = tab._grid_height

    _add_column(window)

    assert "BRANDNEW" in tab.varpanel.names()
    assert tab.picker.combos()["x"].findText("BRANDNEW") >= 0
    assert tab.picker.picks() == dict(zip(("x", "y", "z"), roles, strict=True))
    assert tab.nbins.value() == 12
    assert tab.agg.currentText() == "median"
    assert tab._framed_target == roles  # camera view kept
    assert tab._grid_height is not None and tab._grid_height is not prev_height


def test_surfacexyz_gone_role_takes_its_default(window, monkeypatch):
    tab = _surfacexyz(window, monkeypatch, "VPD_f", "Tair_f", "GPP_DT_CUT_84")

    _delete_column(window, monkeypatch, "Tair_f")

    # The gone Y role takes the default of the first load (second numeric
    # column), X and Z stay.
    numeric = [str(c) for c in
               window._data.select_dtypes(include="number").columns]
    assert tab.picker.picks() == {"x": "VPD_f", "y": numeric[1],
                                  "z": "GPP_DT_CUT_84"}
    assert tab.nbins.value() == 12
    assert tab._grid_height is not None
