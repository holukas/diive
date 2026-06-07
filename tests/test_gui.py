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


@pytest.fixture
def window(app):
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
    window._open_menu_tab("Heatmap")
    window._open_menu_tab("Heatmap")
    window._open_menu_tab("Time series")
    assert "Heatmap 1" in _tabs(window)
    assert "Heatmap 2" in _tabs(window)
    assert "Time series 1" in _tabs(window)


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
