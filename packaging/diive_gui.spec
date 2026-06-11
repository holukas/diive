# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the diive desktop GUI (Windows, one-folder build).

Build with the helper script:  packaging/build_gui.ps1
Or directly:                   uv run pyinstaller packaging/diive_gui.spec --noconfirm

Produces  dist/diive-gui/  containing diive-gui.exe + all dependencies.
Zip that folder and hand it to users; they unzip and run diive-gui.exe.
No Python / uv / pip needed on their machine.

This is a one-FOLDER build on purpose: diive's scientific stack (shap, xgboost,
sklearn, ...) is large, and a one-file exe would unpack all of it to a temp dir
on every launch (slow + flaky). One-folder starts fast.
"""
import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
)

# Embedded EXE/taskbar icon (relative to this spec via PyInstaller's SPECPATH).
_ICON = os.path.join(SPECPATH, "diive.ico")

datas = []
binaries = []
hiddenimports = []


def _no_test_submodules(name):
    """Filter for collect_all: drop bundled test suites.

    collect_all() otherwise pulls in the full *.tests.* / *.testing trees of
    numba/sklearn/statsmodels/pyarrow (thousands of modules the app never runs),
    bloating the build by hundreds of MB. These packages scatter `.tests` under
    many subpackages, so prefix-listing them in `excludes` is unreliable; this
    filters every nesting level at collection time.
    """
    parts = name.split(".")
    return not ({"tests", "testing", "conftest"} & set(parts))

# --- diive package data --------------------------------------------------
# Runtime-required non-.py files: filetype YAMLs (ReadFileType / File > Open)
# and the bundled example dataset that auto-loads on startup.
datas += collect_data_files("diive", includes=["configs/**/*"])

# --- heavy / dynamically-imported third-party packages -------------------
# These use lazy/plugin-style imports or ship compiled binaries & data that
# PyInstaller's static analysis misses. collect_all grabs submodules + data +
# binaries for each. Add to this list when a frozen run raises ModuleNotFound
# or a missing-data error for a library.
_collect = [
    "shap",
    "numba",
    "llvmlite",
    "sklearn",
    "statsmodels",
    "category_encoders",
    "skopt",
    "pyarrow",
    "matplotlib",
]
for _pkg in _collect:
    _d, _b, _h = collect_all(_pkg, filter_submodules=_no_test_submodules)
    datas += _d
    binaries += _b
    hiddenimports += _h

# xgboost is collected manually: collect_all() walks every submodule, and
# importing xgboost.testing pulls in test-only deps (hypothesis/pytest) that
# aren't installed, which aborts the build. We only need its compiled DLL +
# data files (VERSION, lib/) and the public import path.
binaries += collect_dynamic_libs("xgboost")
datas += collect_data_files("xgboost")
hiddenimports += ["xgboost", "xgboost.core", "xgboost.sklearn"]

# --- optional 3-D plotting (gui3d extra: pyvista + pyvistaqt + VTK) -------
# Only bundled if the gui3d extra is installed in the build env; a 2-D-only
# build (gui extra alone) skips it, matching the runtime lazy-import design
# (the 3D tab shows an install notice if VTK is absent). VTK ships hundreds of
# compiled modules + data files that PyInstaller's static analysis misses, so
# collect_all is required. `vtkmodules` is the real package; `vtk` is a thin
# wrapper. qtpy/pyvistaqt route VTK's render window through PySide6, so the
# QtOpenGL(Widgets) modules must NOT be excluded below.
import importlib.util as _ilu

if _ilu.find_spec("pyvista") is not None and _ilu.find_spec("pyvistaqt") is not None:
    for _pkg in ["vtkmodules", "vtk", "pyvista", "pyvistaqt"]:
        _d, _b, _h = collect_all(_pkg, filter_submodules=_no_test_submodules)
        datas += _d
        binaries += _b
        hiddenimports += _h
    # pyvistaqt imports the interactor from vtkmodules.qt at runtime; ensure the
    # static analyzer keeps it (collect_all covers it, but pin it explicitly).
    hiddenimports += ["vtkmodules.qt.QVTKRenderWindowInteractor"]

# --- Qt modules we never use (keep the build smaller) --------------------
# matplotlib's qtagg backend only needs QtWidgets/QtGui/QtCore. Dropping
# WebEngine/Quick/Qml/3D/Multimedia/etc. removes hundreds of MB.
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtQmlModels",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtNfc",
    "PySide6.QtTextToSpeech",
    # Causal-discovery extra is intentionally NOT shipped in the GUI build.
    "tigramite",
    "econml",
    # Dev-only / unused-at-runtime heavyweights.
    "tkinter",
    "IPython",
    "jupyterlab",
    "notebook",
    "pytest",
    # Bundled test suites are filtered out at collection time instead — see
    # _no_test_submodules() above (handles every nesting level reliably).
]

a = Analysis(
    ["launch_diive_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="diive-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX often trips antivirus; not worth it here.
    console=False,        # GUI app -> no console window.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_ICON,  # generated by packaging/make_icon.py from the splash motif
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="diive-gui",
)
