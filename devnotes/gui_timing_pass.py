"""
GUI TIMING PASS: HOW LONG EACH GUI ACTION BLOCKS THE WINDOW
===========================================================

Opens the desktop GUI offscreen on the full bundled example data (10 years of
30-min records), then loads the data, clicks a few Overview variables, opens
every menu tab once, and runs a few app-wide actions (data push with all tabs
open, theme change). For each action it prints:

- **ready**: wall time until the GUI is idle again (worker jobs finished,
  queued renders and resize settle timers done);
- **max freeze**: the longest GUI-thread block, measured as the largest gap
  between ticks of a 5 ms QTimer, i.e. how long the window stopped responding.

Run from the repo root (needs the 'gui' extra; takes a few minutes):

    MPLBACKEND=Agg .venv/Scripts/python.exe devnotes/gui_timing_pass.py

Measure with nothing else running. The ``if __name__ == "__main__"`` guard is
required: the Seasonal trend tab computes in a spawned worker process, which
re-imports the main module. Results are recorded in devnotes/GUI_PERFORMANCE.md.

Part of the diive library: https://github.com/holukas/diive
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MPLBACKEND"] = "Agg"

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = None

ROWS = []


class GapMonitor:
    def __init__(self):
        self.timer = QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self._tick)
        self.last = None
        self.max_gap = 0.0

    def _tick(self):
        now = time.perf_counter()
        if self.last is not None:
            self.max_gap = max(self.max_gap, now - self.last)
        self.last = now

    def start(self):
        self.last = time.perf_counter()
        self.max_gap = 0.0
        self.timer.start()

    def stop(self):
        self._tick()
        self.timer.stop()
        return self.max_gap


MON = None


def busy(win):
    """True while any tab still has background or queued work."""
    for tab in win._tabs:
        runner = getattr(tab, "_runner", None)
        if runner is not None:
            for attr in ("is_busy", "is_running"):
                if getattr(runner, attr, False):
                    return True
        vp = getattr(tab, "varpanel", None)
        if vp is not None and getattr(vp, "_pending_fn", None) is not None:
            return True
        canvas = getattr(tab, "canvas", None)
        rt = getattr(canvas, "_relayout_timer", None)
        if rt is not None and rt.isActive():
            return True
    io = getattr(win, "_io_runner", None)
    if io is not None and getattr(io, "is_running", False):
        return True
    return False


def settle(win, timeout=180.0):
    t_end = time.perf_counter() + timeout
    idle_since = None
    while time.perf_counter() < t_end:
        app.processEvents()
        if busy(win):
            idle_since = None
        else:
            idle_since = idle_since or time.perf_counter()
            if time.perf_counter() - idle_since > 0.25:  # stay idle a moment (timers may re-arm)
                return True
        time.sleep(0.002)
    return False


def measure(label, fn, win):
    MON.start()
    t0 = time.perf_counter()
    try:
        fn()
        ok = settle(win)
        err = "" if ok else "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:60]
    wall = time.perf_counter() - t0 - (0.25 if not err else 0)  # minus the idle confirmation
    gap = MON.stop()
    ROWS.append((label, wall, gap, err))
    print(f"{wall*1000:8.0f} ms ready | {gap*1000:7.0f} ms max freeze | {label} {err}", flush=True)




def main():
    global app
    app = QApplication.instance() or QApplication([])
    global MON
    MON = GapMonitor()
    t0 = time.perf_counter()
    from diive.gui.app import MainWindow  # noqa: E402
    from diive.gui import metadata_store, theme  # noqa: E402
    from diive.gui.registry import MENU_TABS  # noqa: E402
    print(f"{(time.perf_counter()-t0)*1000:8.0f} ms import diive.gui.app", flush=True)

    win = MainWindow(None, False)
    win.resize(1600, 1000)
    win.show()
    app.processEvents()
    measure("load example + first Overview render", lambda: win._load_example(), win)
    ov = win._tabs[0]
    cols = [c for c in win._data.columns if win._data[c].dtype.kind == "f"]
    for c in ["NEE_CUT_REF_f", "Tair_f", "VPD_f", "Rg_f"]:
        if c in cols:
            measure(f"Overview select {c}", lambda c=c: ov._on_select(c), win)

    skip_menus = {"Database"}
    skip_labels = {"3D surface", "3D surface (X/Y/Z)"}
    for menu, entries in MENU_TABS.items():
        if menu in skip_menus:
            continue
        for label in entries:
            if label in skip_labels:
                continue
            measure(f"open [{menu}] {label}", lambda label=label: win._open_menu_tab(label), win)

    print("open tabs:", len(win._tabs), flush=True)
    tw = win._tabwidget
    tw.setCurrentIndex(0)
    settle(win)
    measure("data push, 60+ tabs open, Overview visible", lambda: win._apply_range(), win)
    measure("switch to a stale plot tab (catch-up)", lambda: tw.setCurrentIndex(6), win)
    measure("metadata notify", lambda: metadata_store.manager.notify(), win)
    measure("theme apply", lambda: theme.manager.apply(), win)
    tw.setCurrentIndex(0)
    settle(win)
    measure("Overview select after all tabs", lambda: ov._on_select("NEE_CUT_REF_f"), win)

    print("\nSLOWEST (by max freeze):")
    for label, wall, gap, err in sorted(ROWS, key=lambda r: -r[2])[:25]:
        print(f"{gap*1000:7.0f} ms freeze | {wall*1000:8.0f} ms ready | {label} {err}")
    sys.stdout.flush()
    try:
        from diive.gui.widgets.worker import shutdown_process_pool
        shutdown_process_pool()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
