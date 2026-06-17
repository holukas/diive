"""
DETECT_AND_REMOVE_TLAG_TUI: Textual front-end for diive-tlag-pwb-detect-remove
==============================================================================

A two-column terminal UI (Textual) around ``PerFilePipeline``
(``diive.flux.hires.detect_and_remove_tlag``). The left column is a labelled
settings form; the right column is a live console: an overall progress bar,
a row per busy worker showing the file·chunk it is *currently* processing
prefixed by an animated spinner, and a ``RichLog`` into which the two-phase
pipeline (detect -> PWBOPT -> remove) streams its Rich-styled per-chunk
output (same colour coding as the CLI). A worker's spinner row appears the
instant it starts a chunk; when that chunk finishes the result line
(CH4=… HDI=…) is appended to the log below — so you see what is in flight,
not only what has already finished.

The form covers paths, wind/sonic columns, scalars, PWB & chunking
parameters, the raw-file format (skip-rows, extra header rows, separator,
file glob) and the chunk-naming rule (start-time regex/format and filename
template). The naming fields let each 30-min output chunk be named by its
own start time — e.g. ``CH-CHA_{starttime}{suffix}`` with regex
``(\\d{12})`` turns ``CH-CHA_202107271300.csv`` into per-chunk files
``CH-CHA_202107271300.csv`` (00:00), ``CH-CHA_202107271330.csv`` (00:30), …

Settings persist between sessions in a small YAML file
(``~/.diive/detect_remove_tui.yaml``): they are loaded on start, and saved on
*Run* or via the *Save* button — so column names, paths and parameters need to
be entered only once.

Convenience:

- Every field has a hover tooltip; focusing a field also echoes its help
  into the status line (so keyboard users see the same explanation).
- Drag a folder (or a file — its parent folder is used) into the window to
  fill a path field; the drop prefers an *empty* path field, so once
  *Input dir* is set the next drop fills *Output dir* without clicking it.
  Drop directly onto a field to force that one; the ✕ button clears it.
  (Some terminals "type" a dropped path into the focused field instead of
  pasting it — there, click/clear the target field first.)
- The console prefixes each line with a wall-clock time, and *Copy log*
  (or the ``c`` key) copies the whole console buffer to the clipboard.
- ``run()`` writes the summary CSV and, when *Save plots* is on, the
  batch overview figures — so they appear from the TUI exactly as from
  the CLI.
- The form exposes every CLI option. Defaults worth noting: random seed
  42 (reproducible — clear it for a random run) and line terminator
  ``auto`` (the output matches the input file's CRLF/LF convention).

A ``--demo`` mode runs a synthetic pipeline that needs no input data, purely so
the interface can be previewed.

Launch::

    uv run diive-tlag-pwb-detect-remove-tui --demo     # preview, no data
    uv run diive-tlag-pwb-detect-remove-tui            # real run (fill form)
    uv run python examples/flux/hires/flux_detect_remove_tui_demo.py

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


def _open_in_file_manager(path: Path) -> None:
    """Open a folder in the OS file manager (Windows / macOS / Linux)."""
    if sys.platform.startswith('win'):
        os.startfile(str(path))                      # noqa: S606 (Windows only)
    elif sys.platform == 'darwin':
        subprocess.run(['open', str(path)], check=False)
    else:
        subprocess.run(['xdg-open', str(path)], check=False)

import yaml
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.css.query import NoMatches
from textual.validation import Function, Integer, Number
from textual.widgets import (Button, Footer, Input, Label, OptionList,
                             ProgressBar, RichLog, Static, Switch)
from textual.widgets.option_list import Option

from diive.flux.hires.detect_and_remove_tlag import (
    _WHITESPACE_SEP, PerFilePipeline, parse_scalar_spec, window_to_lag_params)


def _scan_columns(input_dir: str, file_pattern: str, skiprows: int,
                  extra_rows: int, sep: str) -> tuple:
    """Read the first matching file's header. Return (first_file, files, cols).

    Raises ``FileNotFoundError`` if nothing matches, or other exceptions on a
    bad format. Reads only the header lines, so it is fast even for big files.
    """
    files = sorted(Path(input_dir).glob(file_pattern))
    if not files:
        raise FileNotFoundError(
            f"no files match '{file_pattern}' in {input_dir}")
    f0 = files[0]
    n_pre = skiprows + 1 + extra_rows
    with open(f0, 'r', encoding='utf-8', errors='replace') as fh:
        head = [next(fh) for _ in range(n_pre)]
    line = head[skiprows].rstrip('\r\n')
    cols = (line.split() if sep == _WHITESPACE_SEP
            else [c.strip() for c in line.split(sep)])
    return f0, files, cols


def _clean_dropped_path(raw: str) -> str | None:
    """Normalise a drag-and-drop / pasted path to a folder path string.

    Terminals deliver a dropped file or folder as its path (often quoted,
    sometimes ``file://`` URL-encoded). Returns the folder: the path itself
    when it is a directory, its parent when it is a file. Returns None when
    the text is not an existing filesystem path (so the caller can treat it
    as ordinary pasted text).
    """
    if not raw:
        return None
    cand = raw.splitlines()[0].strip().strip('"').strip("'")
    if cand.startswith('file://'):
        cand = unquote(urlparse(cand).path)
        # Windows file:///F:/x -> /F:/x : strip the leading slash before drive
        if re.match(r'^/[A-Za-z]:', cand):
            cand = cand[1:]
    if not cand:
        return None
    p = Path(cand).expanduser()
    if not p.exists():
        return None
    return str(p if p.is_dir() else p.parent)


class PathInput(Input):
    """Input that turns a dropped file/folder into a folder path.

    A dropped folder fills a path field with that folder; a dropped file
    fills it with the file's containing folder (never the file itself).
    Routing of *which* path field receives the drop is delegated to the app
    (``_apply_dropped_path``), which prefers an empty path field — so with
    Input dir already set, a drop lands in the empty Output dir without
    needing to click it first. Non-path text pastes normally into this field.
    """

    def _on_paste(self, event: events.Paste) -> None:
        app = self.app
        handled = False
        if hasattr(app, '_apply_dropped_path'):
            handled = app._apply_dropped_path(event.text or '', preferred=self.id)
        if not handled:
            # Ordinary text: paste into this field at the cursor.
            self.insert_text_at_cursor(event.text or '')
        event.stop()


def _phase_label(phase: str) -> str:
    """User-facing verb for a pipeline phase.

    The internal phase name 'remove' refers to removing the *time lag* (by
    shifting the scalar to align it with the wind), not removing any file —
    so it is shown as 'align' (the paper's term is 'temporal alignment').
    """
    return 'align' if phase == 'remove' else phase


def _unescape_sep(s: str) -> str:
    """Translate literal backslash escapes a user may type for sep/lineterm."""
    return (s.replace('\\t', '\t')
             .replace('\\s+', _WHITESPACE_SEP)
             .replace('\\r', '\r')
             .replace('\\n', '\n'))

# Soft, modern palette (Tokyo Night-ish): muted slate surfaces, pastel accents.
_FG = '#c0caf5'
_DIM = '#565f89'
_BLUE = '#7aa2f7'
_LAV = '#bb9af7'
_CYAN = '#7dcfff'
_GREEN = '#9ece6a'
_AMBER = '#e0af68'
_RED = '#f7768e'

_CSS = """
Screen { background: #1a1b26; color: #a9b1d6; }
#title {
    height: 1; background: #16161e; color: #7aa2f7; text-style: bold;
    content-align: center middle;
}
#body { height: 1fr; }
/* Responsive left column: scales with the terminal's cell grid (which varies
   with font size / DPI scaling, so a "4k monitor" can present very different
   widths), but stays within a readable band instead of a fixed 66 cells that
   gets clipped on a narrow grid. */
#settings {
    width: 44%; min-width: 42; max-width: 68;
    background: #16161e; border-right: solid #2f334d; padding: 0 1;
    scrollbar-size: 1 1;
}
#console { width: 1fr; padding: 0 1; }

.section { color: #7aa2f7; text-style: bold; height: 1; margin: 0; }
.field { height: 1; }
.flabel { width: 16; color: #565f89; content-align: left middle; }
.fin {
    width: 1fr; height: 1; border: none; background: #24283b; color: #c0caf5;
    padding: 0 1;
}
.fin:focus { background: #2f334d; color: #c0caf5; }
.clearbtn {
    width: 3; min-width: 3; height: 1; margin: 0 0 0 1; border: none;
    background: #2f334d; color: #f7768e; content-align: center middle;
}
.clearbtn:hover { background: #3b4261; color: #f7768e; }
Switch { height: 1; border: none; background: #24283b; }
Switch.-on { background: #2f334d; }

#controls { height: 1; margin: 1 0 0 0; }
#controls2 { height: 1; }
Button { margin: 0 1 0 0; min-width: 8; height: 3; }
/* Flat single-row action buttons on the left form, so the settings panel
   fits without a vertical scrollbar on high-DPI (4k-scaled) screens. The
   right-panel Copy log and the modal-dialog buttons keep the default 3-row
   height. The height:1 + border:none pattern matches .clearbtn/.pickbtn. */
#controls Button, #controls2 Button { height: 1; border: none; }
Button#run { background: #7aa2f7; color: #1a1b26; text-style: bold; }
Button#check { background: #2f334d; color: #e0af68; }
Button#stop { background: #2f334d; color: #f7768e; }
Button#open { background: #2f334d; color: #7dcfff; }
Button#save { background: #2f334d; color: #c0caf5; }
Button#quit { background: #2f334d; color: #c0caf5; }
Button:disabled { color: #565f89; text-style: none; }

#status { height: 1; color: #565f89; }
#progressrow { height: 1; }
#phase { width: 9; content-align: left middle; }
ProgressBar { width: 1fr; }
#bar Bar > .bar--bar { color: #7aa2f7; }
#bar Bar > .bar--complete { color: #9ece6a; }
#wpool {
    height: auto; max-height: 9; margin: 1 0 0 0; scrollbar-size: 1 1;
}
.wrow { height: 1; width: 1fr; color: #7dcfff; content-align: left middle; }
#log {
    height: 1fr; background: #16161e; border: round #2f334d; padding: 0 1;
    scrollbar-size: 1 1;
}
#logcontrols { height: 3; align: right middle; }
Button#copy { background: #2f334d; color: #7dcfff; min-width: 12; }
Footer { background: #16161e; color: #565f89; }

InfoScreen { align: center middle; background: #1a1b26 70%; }
#dialog {
    width: 80%; max-width: 76; height: auto; max-height: 90%;
    background: #16161e; border: round #7aa2f7; padding: 1 2;
}
#infoscroll { height: auto; max-height: 26; scrollbar-size: 1 1; }
#dialog Button { margin: 1 1 0 0; min-width: 12; }
#dialog Horizontal { height: auto; align: left middle; }
#loadinfo { height: auto; }
#loadpath {
    height: 3; border: round #2f334d; background: #24283b; color: #c0caf5;
    margin: 1 0;
}
#loadpath:focus { border: round #7aa2f7; }

.pickbtn {
    width: 3; min-width: 3; height: 1; margin: 0 0 0 1; border: none;
    background: #2f334d; color: #7aa2f7; content-align: center middle;
}
.pickbtn:hover { background: #3b4261; }
.reseedbtn {
    width: 3; min-width: 3; height: 1; margin: 0 0 0 1; border: none;
    background: #2f334d; color: #bb9af7; content-align: center middle;
}
.reseedbtn:hover { background: #3b4261; }
ColumnPickerScreen { align: center middle; background: #1a1b26 70%; }
#colpick {
    width: 80%; max-width: 72; height: auto; max-height: 80%;
    background: #16161e; border: round #7aa2f7; padding: 1 2;
}
#collist { height: auto; max-height: 22; border: round #2f334d; margin: 1 0; }
.fin.-invalid { background: #3a2330; color: #f7768e; }

/* Narrow terminals (small cell grid from high-DPI scaling / a small window):
   stack the two panes vertically so the settings form is full width and
   nothing is clipped horizontally. The -narrow class is added to the Screen
   by HORIZONTAL_BREAKPOINTS below a threshold width. */
.-narrow #body { layout: vertical; }
.-narrow #settings {
    width: 100%; min-width: 0; max-width: 100%;
    height: auto; max-height: 55%;
    border-right: none; border-bottom: solid #2f334d;
}
.-narrow #console { width: 100%; height: 1fr; }
"""

_INFO_TEXT = (
    "[b #7aa2f7]diive · PWB time-lag detect + remove[/]\n"
    "\n"
    "Corrects the [b]tube-delay time lag[/] between each gas signal and the\n"
    "vertical wind in raw eddy-covariance data, one 30-min chunk at a time.\n"
    "\n"
    "[b #bb9af7]Two phases[/]\n"
    " 1. [b]detect[/] — rotate each chunk's wind in memory, run PWB on the\n"
    "    rotated W vs each scalar -> one lag per chunk (nothing written yet).\n"
    " 2. [b]remove[/] — PWBOPT picks the best lag across all chunks; each\n"
    "    scalar is shifted by it and written as a lag-corrected file\n"
    "    (into [b]2_lag_removed/[/], the input for the next flux step).\n"
    "\n"
    "[b #bb9af7]PWB — pre-whitening bootstrap (Vitale et al. 2024)[/]\n"
    " • [b]pre-whitening[/]: an AR filter turns the auto-correlated W and\n"
    "   scalar into near-white noise, sharpening the cross-correlation peak.\n"
    " • [b]bootstrap[/]: many block-resampled series each give a peak lag; the\n"
    "   mode is the lag, and the 95% HDI (its spread) is the reliability —\n"
    "   narrow HDI = trustworthy, wide HDI = noisy.\n"
    " • robust for low-magnitude fluxes (CH4, N2O) where a single CCF is noisy.\n"
    "\n"
    "[b #bb9af7]PWBOPT[/]\n"
    " Wide-HDI (unreliable) chunks are replaced by the neighbouring good lag,\n"
    " so a spurious per-chunk value is never applied.\n"
    "\n"
    "[b #bb9af7]Per-gas search windows (Win s)[/]\n"
    " Each gas gets its own [b]LABEL:\\[lower,upper][/] window (seconds) over\n"
    " which the lag is searched. Set them by physics:\n"
    " • A [b]positive-only[/] window (e.g. N2O:\\[0,5]) keeps only physical\n"
    "   tube-delay lags (closed-path delay > 0), the paper's advice.\n"
    " • A [b]long-inlet[/] gas needs a wider window than the dry gases —\n"
    "   e.g. N2O:\\[0,5] beside H2O:\\[0,20]. EddyPro later applies one lag\n"
    "   setting to all gases, so each gas must be pre-aligned here.\n"
    " • Per gas, lag_max = the larger |bound| and the bootstrap block\n"
    "   length = max(20 s, 2x the window half-width): never below the\n"
    "   paper's 20 s (long enough to contain the lag and preserve the\n"
    "   correlation structure), but it grows for a wide window.\n"
    " • Keep the expected lag in the [b]middle[/] of the window, not on an\n"
    "   edge — a peak detected at the boundary is treated as unreliable.\n"
    "\n"
    "[#e0af68]Note:[/] downstream flux software must run with EC time-lag\n"
    "maximization DISABLED — the delay is already corrected here.\n"
    "\n"
    "[dim]Esc or Close to dismiss.[/]"
)


class InfoScreen(ModalScreen):
    """Concise 'what is this / how PWB works' overlay."""

    BINDINGS = [('escape', 'close', 'Close'), ('q', 'close', 'Close')]

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            with VerticalScroll(id='infoscroll'):
                yield Static(_INFO_TEXT, id='infotext')
            yield Button('Close', id='close', variant='primary')

    def action_close(self) -> None:
        self.dismiss()

    @on(Button.Pressed, '#close')
    def _on_close(self) -> None:
        self.dismiss()


class ColumnPickerScreen(ModalScreen):
    """Pick a column name from a scanned file header. Dismisses with the
    chosen column string, or None on cancel."""

    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, title: str, columns: list):
        super().__init__()
        self._title = title
        self._columns = columns

    def compose(self) -> ComposeResult:
        with Vertical(id='colpick'):
            yield Static(f'[b #7aa2f7]{self._title}[/]\n'
                         'Select a column (Enter), or Esc to cancel:')
            ol = OptionList(*[Option(c, id=str(i))
                              for i, c in enumerate(self._columns)],
                            id='collist')
            yield ol
            with Horizontal():
                yield Button('Cancel', id='colcancel')

    def on_mount(self) -> None:
        self.query_one('#collist', OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, '#colcancel')
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(OptionList.OptionSelected, '#collist')
    def _picked(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._columns[int(event.option.id)])


class LoadScreen(ModalScreen):
    """Prompt for a YAML settings file to load. Dismisses with the path or None."""

    BINDINGS = [('escape', 'cancel', 'Cancel')]

    def __init__(self, default_path: str = ''):
        super().__init__()
        self._default_path = default_path

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            yield Static('[b #7aa2f7]Load settings YAML[/]\n'
                         'Enter the path to a settings .yaml file:', id='loadinfo')
            yield Input(value=self._default_path,
                        placeholder='/path/to/settings.yaml', id='loadpath')
            with Horizontal():
                yield Button('Load', id='loadok', variant='primary')
                yield Button('Cancel', id='loadcancel')

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, '#loadcancel')
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, '#loadok')
    @on(Input.Submitted, '#loadpath')
    def _ok(self) -> None:
        self.dismiss(self.query_one('#loadpath', Input).value.strip())

# (field id, label, placeholder/help). For the PWB & chunking params the
# placeholder shows the DEFAULT value used when the field is left blank, so an
# empty box documents what the pipeline will fall back to (see _collect()).
_FIELDS = [
    ('input_dir', 'Input dir', 'folder with raw EC files'),
    ('output_dir', 'Output dir', 'where results are written'),
    ('col_u', 'Wind U', 'column name for U'),
    ('col_v', 'Wind V', 'column name for V'),
    ('col_w', 'Wind W', 'column name for W'),
    ('col_tsonic', 'Sonic T', 'column name for T_SONIC'),
    ('scalars', 'Scalars', 'LABEL:column,...  e.g. CH4:ch4,N2O:n2o'),
    ('hz', 'Frequency', 'default 20  (samples per second)'),
    ('chunk', 'Chunk s', 'default 1800  (= 30 min)'),
    ('minchunk', 'Min chunk s', 'default 300  (skip shorter chunks)'),
    ('nboot', 'Bootstraps', 'default 99  (PWB replicates, paper value)'),
    ('lagmax', 'Lag max s', 'default 10.0  (seeds each gas window below)'),
    ('winranges', 'Win s', 'per-gas LABEL:[lower,upper]  (⟳ re-seeds from Lag max)'),
    # --- PWBOPT (best-lag selection across chunks) ---
    ('hdithresh', 'HDI thresh', 'default 0.5  (S1: reliable if HDI range <)'),
    ('devthresh', 'Dev thresh', 'default 0.5  (S2: accept if within of prev)'),
    ('hdiprefilter', 'HDI prefilt', 'default 1.0  (drop lags HDI >; 0 = off)'),
    ('lagcol', 'Lag column', 'default {prefix}_tlag_final_pf_s  (lag removed)'),
    # --- File format (how each raw file is read) ---
    ('skiprows', 'Skip rows', 'default 0  (metadata lines before header row)'),
    ('extrarows', 'Extra rows', 'default 2  (units/source rows after header)'),
    ('sep', 'Separator', r'default ,   (use \t for tab, \s+ for whitespace)'),
    ('filepattern', 'File glob', 'default *.csv'),
    ('navalues', 'NA values', 'default -9999 -9999.0 …  (space-separated)'),
    ('narep', 'NA out', 'default -9999  (written for NaN)'),
    ('lineterm', 'Line term', r'auto = match input  (or force \r\n / \n)'),
    # --- Chunk naming (controls output filenames) ---
    ('streg', 'Start regex', r'e.g. (\d{12})  — capture file start from name'),
    ('stfmt', 'Start format', 'e.g. %Y%m%d%H%M  (parses the captured text)'),
    ('ctmpl', 'Name tmpl', '{stem}_chunk{index:02d}{suffix}'),
    # --- Output layout ---
    ('detectsub', 'Detect dir', 'default 1_lag_detection  (diagnostics)'),
    ('datasub', 'Data dir', 'default 2_lag_removed  (corrected chunks)'),
    # --- Execution ---
    ('workers', 'Workers', 'default: all CPU cores'),
    ('randomstate', 'Random seed', 'blank = non-deterministic bootstrap'),
]
_FIELD_IDS = [f[0] for f in _FIELDS]

# Path fields get a clear (✕) button and accept drag-and-drop of a folder
# or file (file -> its parent folder).
_PATH_FIELDS = ['input_dir', 'output_dir']

# Column fields that get a ▾ button to pick the name from the scanned file
# header. The four single-column fields replace their value on pick; the
# scalars field inserts the column at the cursor (so you build LABEL:column).
_COL_PICK_FIELDS = ['col_u', 'col_v', 'col_w', 'col_tsonic', 'scalars']

# The per-gas window field gets a ⟳ button that re-seeds every gas window to the
# symmetric [-Lag max, +Lag max] default (e.g. after editing Lag max s).
_WIN_FIELDS = ['winranges']

# Per-gas window text format: LABEL:[lower,upper] entries, comma-separated. The
# inner brackets carry their own comma, so split with a regex, not str.split.
_WIN_RE = re.compile(r'([^\s:,\[\]]+)\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,'
                     r'\s*(-?\d+(?:\.\d+)?)\s*\]')


def parse_win_ranges(text: str) -> dict:
    """Parse ``CH4:[-10,10],H2O:[0,25]`` -> ``{'CH4': (-10.0, 10.0), ...}``."""
    return {m.group(1): (float(m.group(2)), float(m.group(3)))
            for m in _WIN_RE.finditer(text or '')}


def _fmt_win_num(x: float) -> str:
    """Render a window bound without a trailing ``.0`` (so [-10,10] not [-10.0,10.0])."""
    return str(int(x)) if float(x).is_integer() else f'{x:g}'


def format_win_ranges(items) -> str:
    """Build the window text from ``[(label, lower, upper), ...]`` in order."""
    return ','.join(f'{lbl}:[{_fmt_win_num(a)},{_fmt_win_num(b)}]'
                    for lbl, a, b in items)


# --- Live field validation (blank is always allowed = use default) ---------
def _opt_int(v: str) -> bool:
    v = v.strip()
    return v == '' or v.lstrip('+-').isdigit()


def _opt_num(v: str) -> bool:
    v = v.strip()
    if v == '':
        return True
    try:
        float(v)
        return True
    except ValueError:
        return False


def _opt_regex(v: str) -> bool:
    v = v.strip()
    if v == '':
        return True
    try:
        re.compile(v)
        return True
    except re.error:
        return False


_VALIDATORS = {
    'hz': [Function(_opt_int, 'whole number, e.g. 20')],
    'chunk': [Function(_opt_num, 'a number of seconds')],
    'minchunk': [Function(_opt_num, 'a number of seconds')],
    'nboot': [Function(_opt_int, 'whole number, e.g. 99')],
    'lagmax': [Function(_opt_num, 'a number of seconds')],
    'workers': [Function(_opt_int, 'whole number (blank = all cores)')],
    'skiprows': [Function(_opt_int, 'whole number')],
    'extrarows': [Function(_opt_int, 'whole number')],
    'hdithresh': [Function(_opt_num, 'a number of seconds')],
    'devthresh': [Function(_opt_num, 'a number of seconds')],
    'hdiprefilter': [Function(_opt_num, 'a number of seconds')],
    'streg': [Function(_opt_regex, 'not a valid regular expression')],
}

# Boolean settings rendered as a Switch (not an Input). Persisted alongside
# the text fields.
_SWITCHES = ['saveplots', 'strict']

# Fields pre-filled with a concrete value on start (no sensible blank fallback).
# The PWB & chunking params and paths are left blank so their placeholder shows
# the default; the pipeline applies those defaults when the box is empty.
_DEFAULTS = {
    'col_u': 'u', 'col_v': 'v', 'col_w': 'w', 'col_tsonic': 'ts',
    'scalars': 'CH4:ch4,N2O:n2o',
    'sep': ',',
    # Chunk naming pre-filled for the common 12-digit YYYYMMDDHHMM filename
    # (e.g. CH-CHA_202107271300.csv -> per-chunk CH-CHA_202107271330.csv …).
    'streg': r'(\d{12})',
    'stfmt': '%Y%m%d%H%M',
    'ctmpl': 'CH-CHA_{starttime}{suffix}',
    'lagcol': '{prefix}_tlag_final_pf_s',
    'detectsub': '1_lag_detection',
    'datasub': '2_lag_removed',
    'narep': '-9999',
    'lineterm': 'auto',
    # Reproducible bootstrap by default (override with a blank field for a
    # non-deterministic run).
    'randomstate': '42',
}

# PWB & chunking params that fall back to a default when left blank (see
# _collect). The Reset button clears exactly these.
_RESET_FIELDS = ['hz', 'chunk', 'minchunk', 'nboot', 'lagmax',
                 'workers', 'hdithresh', 'devthresh', 'hdiprefilter']

# Longer per-field explanations. Shown as a hover tooltip on the field and
# its label, and echoed to the status line when the field gains focus, so
# both mouse and keyboard users see what each setting means.
_HELP = {
    'input_dir':
        'Folder with the raw (unrotated) high-frequency EC files to process.',
    'output_dir':
        'Where results go. Creates 1_lag_detection/ (diagnostics, summary, '
        'plots) and 2_lag_removed/ (the lag-corrected chunk files to feed the '
        'next flux step).',
    'col_u': 'Column name of the horizontal wind component U, exactly as it '
             'appears in the file header.',
    'col_v': 'Column name of the horizontal wind component V.',
    'col_w': 'Column name of the vertical wind component W.',
    'col_tsonic': 'Column name of the sonic temperature. Used as a second '
                  'reference signal in PWB (the 4-combination logic).',
    'scalars':
        'Gases to time-align. Format LABEL:column, comma-separated, e.g. '
        'CH4:CH4_DRY_[LGR-A],N2O:N2O_DRY_[LGR-A]. LABEL is your short name; '
        'column is the header name in the file. This field selects the gases '
        'only — set each gas\'s search window in "Win s" below.',
    'hz': 'Sampling frequency in Hz (samples per second). Typically 10 or 20.',
    'chunk': 'Length of each processing chunk in seconds. 1800 = 30 min, the '
             'standard EC averaging interval. One output file per chunk.',
    'minchunk': 'Chunks shorter than this many seconds are skipped — PWB needs '
                'enough records for the block-bootstrap. Default 300 (5 min).',
    'nboot': 'Number of block-bootstrap replicates per chunk. 99 is the paper '
             'value; fewer is faster but noisier HDIs.',
    'lagmax': 'Default lag search half-width (seconds). It only SEEDS each '
              'gas window in "Win s" with [-this, +this]; once a gas has a '
              'window there, that window is used and this value is ignored for '
              'it. The ⟳ button re-seeds all windows from this. Default 10.',
    'winranges':
        'Per-gas lag search window, one LABEL:[lower,upper] (seconds) per gas, '
        'auto-filled from your Scalars + Lag max. The lag is searched only '
        'inside [lower,upper]; an asymmetric, positive-only window like '
        'H2O:[0,25] keeps only physical tube-delay lags (delay > 0, as the '
        'paper recommends for closed-path gases) and lets a long-inlet gas use '
        'a wider range than the dry gases — e.g. N2O:[0,5] (a ~1.8 s lag) next '
        'to H2O:[0,20]. Per gas, lag_max = the larger |bound| (the CCF is still '
        'computed symmetric over +/-lag_max, only the peak search is windowed) '
        'and the block length = max(20 s, 2x the window half-width): never '
        'below the paper\'s 20 s, growing so a wide window still contains a long '
        'lag. Keep your expected lag in the MIDDLE of the window, not on an '
        'edge — a detection at the boundary is treated as unreliable. Press ⟳ '
        'to reset all windows to the symmetric Lag max default; leave a gas out '
        '(or clear the field) to use the plain symmetric Lag max for it.',
    'hdithresh':
        'S1 threshold (seconds). A chunk whose 95% HDI range is below this is '
        'flagged reliable (S1) and its detected lag is trusted directly.',
    'devthresh':
        'S2 threshold (seconds). An uncertain chunk is still accepted if its '
        'lag is within this distance of the previous reliable lag.',
    'hdiprefilter':
        'Pre-filter (seconds). Lags with an HDI range wider than this are '
        'dropped before PWBOPT (the pre-filtered variant). 0 disables it.',
    'lagcol':
        'Which PWBOPT lag column is actually removed in phase 2. Default '
        '{prefix}_tlag_final_pf_s (pre-filtered, gap-filled "best" lag). Use '
        '{prefix}_tlag_final_s for the non-pre-filtered PWBOPT lag.',
    'skiprows':
        'Number of metadata lines BEFORE the column-name (header) row. 0 if '
        'the header is the first line; EddyPro rotated files use 9.',
    'extrarows':
        'Rows AFTER the header but BEFORE the data — e.g. a units row and an '
        'instrument-source row. Typical raw EC CSV: 2.',
    'sep': r'Field separator. , for CSV, \t for tab, \s+ for any whitespace. '
           'Used for both reading and writing.',
    'filepattern': 'Glob selecting which files in the input folder to process, '
                   'e.g. *.csv or *.dat.',
    'navalues': 'Strings in the input treated as missing (NaN), '
                'space-separated. Default covers the -9999 family.',
    'narep': 'Value written for missing data in the output files. Default '
             '-9999 (the trailing rows of each shifted column become this).',
    'lineterm':
        "Line ending of the output file. 'auto' (default) reproduces the "
        "input file's convention — CRLF for typical Windows EC logger files, "
        r"LF for Unix. Force with \r\n or \n.",
    'streg':
        r'Regex capturing the file START timestamp from its name. e.g. '
        r'(\d{12}) grabs 202107271300 from CH-CHA_202107271300.csv. Capture '
        'groups are concatenated, then parsed with Start format.',
    'stfmt':
        'strptime/strftime pattern for the captured timestamp, e.g. '
        '%Y%m%d%H%M for 202107271300. Also used to format {starttime} in the '
        'output name.',
    'ctmpl':
        'Output filename template. Placeholders: {stem} {suffix} {index} '
        '{starttime}. With {starttime} each chunk is named by its own start '
        'time, e.g. CH-CHA_{starttime}{suffix} -> CH-CHA_202107271330.csv.',
    'detectsub': 'Name of the subfolder (under Output dir) holding step-1 '
                 'diagnostics: summary CSV, plots, checkpoints. Default '
                 '1_lag_detection.',
    'datasub': 'Name of the subfolder holding the deliverable: the '
               'lag-corrected chunk files. Feed THIS folder to the next flux '
               'step. Default 2_lag_removed.',
    'workers': 'Parallel worker processes. Blank = all CPU cores. 1 = '
               'sequential (slower but easier to debug).',
    'randomstate': 'Seed for the bootstrap RNG. A fixed number (default 42) '
                   'makes runs reproducible; clear it for a random run.',
    'saveplots': 'Save diagnostic figures: a 3-panel PWB plot per chunk per '
                 'gas, plus the batch overview plots.',
    'strict': 'Stop on the first error instead of recording it per chunk and '
              'continuing. Useful for debugging a misconfiguration.',
}

_SETTINGS_PATH = Path.home() / '.diive' / 'detect_remove_tui.yaml'

# Filename of the per-run settings YAML dropped into each run's output folder
# (same schema as _SETTINGS_PATH, so it loads straight back into the TUI).
_OUTPUT_SETTINGS_NAME = 'detect_remove_tui_settings.yaml'

# Number of live per-worker rows shown in the console (each with an animated
# spinner). Runs with more workers than this still work; only the first
# _MAX_WORKER_ROWS busy workers get a visible row at any moment.
_MAX_WORKER_ROWS = 16

# Braille "dots" spinner frames (the classic terminal spinner animation).
_SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'


def _load_settings() -> dict:
    try:
        data = yaml.safe_load(_SETTINGS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _summary_lines(summary, scalars, hz, cancelled: bool = False) -> list:
    """Build the post-run summary block: counts + reliability + median lag."""
    import numpy as np
    lines: list = []
    if summary is None or len(summary) == 0:
        lines.append(Text('summary: no chunks processed', style=_DIM))
        return lines
    n = len(summary)
    st = summary['status'] if 'status' in summary.columns else None
    n_ok = int((st == 'ok').sum()) if st is not None else 0
    # Count every skipped:* variant (short trailing chunks + duplicate output
    # names) under the generic "skipped" tally.
    n_skip = (int(st.astype(str).str.startswith('skipped').sum())
              if st is not None else 0)
    n_err = int((st == 'error').sum()) if st is not None else 0
    head = Text()
    head.append('── summary ──  ', style=f'bold {_BLUE}')
    head.append(f'{n_ok} aligned', style=_GREEN)
    if n_skip:
        head.append(f'  {n_skip} skipped', style=_AMBER)
    if n_err:
        head.append(f'  {n_err} errors', style=_RED)
    if cancelled:
        head.append('  (stopped early)', style=_AMBER)
    lines.append(head)
    for g in scalars:
        pfx = g.lower()
        tcol = f'{pfx}_tlag_s'
        if tcol not in summary.columns:
            continue
        vals = summary[tcol].astype(float).dropna()
        med = float(np.nanmedian(vals)) if len(vals) else float('nan')
        rcol = f'{pfx}_is_reliable'
        rel = (int(summary[rcol].fillna(False).astype(bool).sum())
               if rcol in summary.columns else 0)
        pct = (100.0 * rel / n) if n else 0.0
        t = Text()
        t.append(f'{g}: ', style=f'bold {_FG}')
        t.append(f'median lag {med:.2f}s', style=_FG)
        t.append(f'   reliable(S1) {rel}/{n} ({pct:.0f}%)',
                 style=_GREEN if pct >= 50 else _AMBER)
        lines.append(t)
    return lines


def _detect_line(stem: str, row: dict, scalars) -> Text:
    """Render one detection row as a Rich Text line (colour-coded by HDI)."""
    t = Text()
    t.append(f'{stem}  ', style=_DIM)
    for label in scalars:
        pfx = label.lower()
        v = row.get(f'{pfx}_tlag_s')
        h = row.get(f'{pfx}_hdi_range_s')
        if v is None or v != v:
            t.append(f'{label}=--  ', style=_DIM)
            continue
        col = _GREEN if (h == h and h < 0.5) else (_AMBER if (h == h and h < 1.0) else _RED)
        t.append(f'{label}=', style=_FG)
        t.append(f'{v:.2f}s ', style=f'bold {_FG}')
        t.append(f'HDI={h:.2f}  ', style=col)
    return t


class DetectRemoveTUI(App):
    """Two-column Textual UI for the PWB detect+remove pipeline."""

    CSS = _CSS
    TITLE = 'diive · PWB time-lag detect + remove'
    # Responsive layout: below 96 cells wide the Screen gets the '-narrow'
    # class and the two panes stack vertically (see the CSS). This keeps the
    # form usable whatever cell grid the terminal/monitor presents.
    HORIZONTAL_BREAKPOINTS = [(0, '-narrow'), (96, '-wide')]
    BINDINGS = [('r', 'run', 'Run'), ('k', 'check', 'Check'),
                ('x', 'stop', 'Stop'), ('o', 'open', 'Open output folder'),
                ('s', 'save', 'Save'), ('l', 'load', 'Load'),
                ('d', 'reset', 'Reset'), ('c', 'copy', 'Copy log'),
                ('i', 'info', 'Info'), ('q', 'quit', 'Quit')]

    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo = demo
        self._phase = None  # tracks detect->remove handoff for the bar
        self._logbuf: list[str] = []  # plain-text mirror of the console log
        self._last_path_field = 'input_dir'  # most recently focused path field
        self._cancel_event = None       # threading.Event for the active run
        self._last_output_dir = None    # output dir of the last run (for Open)
        self._busy = False           # a run/check is in progress

    # ---- layout: title / [settings | console] / footer ----------------
    def compose(self) -> ComposeResult:
        yield Static('diive · PWB time-lag detect + remove', id='title')
        with Horizontal(id='body'):
            with VerticalScroll(id='settings'):
                yield Static('Paths', classes='section')
                yield self._field('input_dir')
                yield self._field('output_dir')
                yield Static('Wind & sonic columns', classes='section')
                yield self._field('col_u')
                yield self._field('col_v')
                yield self._field('col_w')
                yield self._field('col_tsonic')
                yield Static('Scalars', classes='section')
                yield self._field('scalars')
                yield Static('PWB & chunking', classes='section')
                yield self._field('hz')
                yield self._field('chunk')
                yield self._field('minchunk')
                yield self._field('nboot')
                yield self._field('lagmax')
                yield self._field('winranges')
                yield Static('PWBOPT (best-lag selection)', classes='section')
                yield self._field('hdithresh')
                yield self._field('devthresh')
                yield self._field('hdiprefilter')
                yield self._field('lagcol')
                yield Static('File format', classes='section')
                yield self._field('skiprows')
                yield self._field('extrarows')
                yield self._field('sep')
                yield self._field('filepattern')
                yield self._field('navalues')
                yield self._field('narep')
                yield self._field('lineterm')
                yield Static('Chunk naming', classes='section')
                yield self._field('streg')
                yield self._field('stfmt')
                yield self._field('ctmpl')
                yield Static('Output layout', classes='section')
                yield self._field('detectsub')
                yield self._field('datasub')
                yield Static('Execution', classes='section')
                yield self._field('workers')
                yield self._field('randomstate')
                yield self._switch_row('Save plots', 'saveplots')
                yield self._switch_row('Strict', 'strict')
                with Horizontal(id='controls'):
                    yield Button('Run', id='run', variant='primary')
                    yield Button('Check', id='check')
                    yield Button('Stop', id='stop')
                    yield Button('Open', id='open')
                with Horizontal(id='controls2'):
                    yield Button('Save', id='save')
                    yield Button('Load', id='load')
                    yield Button('Reset', id='reset')
                    yield Button('Quit', id='quit')
            with Vertical(id='console'):
                yield Static('idle', id='status')
                with Horizontal(id='progressrow'):
                    yield Static(f'[{_LAV}]detect[/]', id='phase')
                    yield ProgressBar(id='bar')
                # One row per live worker: an animated spinner + the file·chunk
                # it is currently processing. Hidden until a worker is active;
                # when its chunk finishes the result line lands in the log
                # below.
                with VerticalScroll(id='wpool'):
                    for i in range(_MAX_WORKER_ROWS):
                        yield Label('', classes='wrow', id=f'wrow{i}')
                yield RichLog(id='log', wrap=True, markup=False, highlight=False)
                with Horizontal(id='logcontrols'):
                    yield Button('Copy log', id='copy')
        yield Footer()

    def _field(self, fid: str) -> Horizontal:
        label = next(lbl for i, lbl, _ in _FIELDS if i == fid)
        ph = next(p for i, _, p in _FIELDS if i == fid)
        help_txt = _HELP.get(fid, ph)
        lbl = Label(label, classes='flabel')
        lbl.tooltip = help_txt
        inp_cls = PathInput if fid in _PATH_FIELDS else Input
        inp = inp_cls(placeholder=ph, id=fid, classes='fin')
        inp.tooltip = help_txt  # hover anywhere on the field shows the help
        # Live validation on the relevant fields (Textual marks them
        # ``-invalid`` automatically; CSS paints them red).
        validators = _VALIDATORS.get(fid)
        if validators:
            inp.validators = validators
        children = [lbl, inp]
        # Path fields get a ✕ clear button; column fields get a ▾ picker; the
        # per-gas window field gets a ⟳ button to re-seed from Lag max s.
        if fid in _PATH_FIELDS:
            children.append(Button('✕', id=f'clr_{fid}', classes='clearbtn'))
        elif fid in _COL_PICK_FIELDS:
            children.append(Button('▾', id=f'pick_{fid}', classes='pickbtn'))
        elif fid in _WIN_FIELDS:
            children.append(Button('⟳', id=f'reseed_{fid}', classes='reseedbtn'))
        return Horizontal(*children, classes='field')

    def _switch_row(self, label: str, sid: str) -> Horizontal:
        help_txt = _HELP.get(sid, label)
        lbl = Label(label, classes='flabel')
        lbl.tooltip = help_txt
        sw = Switch(value=False, id=sid)
        sw.tooltip = help_txt
        return Horizontal(lbl, sw, classes='field')

    # ---- column picker (▾ next to the column fields) -------------------
    @on(Button.Pressed, '.pickbtn')
    def _on_pick(self, event: Button.Pressed) -> None:
        event.stop()
        fid = (event.button.id or '')[5:]  # strip 'pick_'
        if not fid:
            return

        def g(i):
            return self.query_one(f'#{i}', Input).value.strip()

        in_dir = g('input_dir')
        if not in_dir:
            self._status('set Input dir first, then pick columns', _AMBER)
            return
        try:
            f0, _files, cols = _scan_columns(
                in_dir, g('filepattern') or '*.csv',
                int(g('skiprows') or 0), int(g('extrarows') or 2),
                _unescape_sep(g('sep') or ','))
        except Exception as e:
            self._status(f'could not read columns: {e}', _RED)
            return
        if not cols:
            self._status('no columns found in the header', _AMBER)
            return

        def _assign(choice) -> None:
            if not choice:
                return
            inp = self.query_one(f'#{fid}', Input)
            if fid == 'scalars':
                # Build LABEL:column — insert the column at the cursor so the
                # user can prefix a label (e.g. 'CH4:' then pick the column).
                inp.insert_text_at_cursor(choice)
            else:
                inp.value = choice
            inp.focus()

        self.push_screen(ColumnPickerScreen(f'Columns in {f0.name}', cols),
                         _assign)

    # ---- per-gas window field (auto-sync to the selected scalars) -------
    def _scalar_labels(self) -> list:
        """Ordered, de-duplicated gas labels currently typed in Scalars.

        Only complete ``LABEL:column`` tokens count, so a half-typed label
        (no colon yet) does not spawn a transient window entry while typing.
        """
        labels, seen = [], set()
        for tok in self.query_one('#scalars', Input).value.split(','):
            tok = tok.strip()
            if ':' not in tok:
                continue
            lbl = tok.split(':', 1)[0].strip()
            if lbl and lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)
        return labels

    def _lagmax_value(self) -> float:
        try:
            return float(self.query_one('#lagmax', Input).value.strip() or 10.0)
        except ValueError:
            return 10.0

    def _sync_win_field(self, reseed: bool = False) -> None:
        """Reconcile the Win field to the gases in Scalars.

        Adds a default ``[-lagmax, +lagmax]`` window for each newly-typed gas,
        drops windows for gases no longer present, and keeps windows the user
        has already edited (unless ``reseed`` rewrites them all to the default).
        Only writes when the text actually changes, so it never disturbs the
        cursor while typing in another field.
        """
        try:
            win_inp = self.query_one('#winranges', Input)
        except NoMatches:
            return
        existing = {} if reseed else parse_win_ranges(win_inp.value)
        lm = self._lagmax_value()
        items = [(lbl, *existing.get(lbl, (-lm, lm)))
                 for lbl in self._scalar_labels()]
        new_text = format_win_ranges(items)
        if new_text != win_inp.value:
            win_inp.value = new_text

    @on(Input.Changed, '#scalars')
    def _on_scalars_changed(self, event: Input.Changed) -> None:
        self._sync_win_field()

    @on(Button.Pressed, '.reseedbtn')
    def _on_reseed(self, event: Button.Pressed) -> None:
        event.stop()
        self._sync_win_field(reseed=True)
        self._status('per-gas windows re-seeded from Lag max s', _LAV)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Echo a field's help into the status line when it gains focus, so
        keyboard users get the same explanation as the hover tooltip. Also
        remember the last-focused path field as a drop target."""
        wid = getattr(getattr(event, 'widget', None), 'id', None)
        if wid in _PATH_FIELDS:
            self._last_path_field = wid
        if wid in _HELP:
            self._status(_HELP[wid], _DIM)

    def _drop_target(self, preferred: str | None = None) -> str:
        """Choose which path field a dropped folder fills.

        Rule (so a drop usually lands where you expect without clicking):
        1. the *preferred* field (the one focused / dropped onto) if empty;
        2. otherwise the single empty path field, if exactly one is empty
           (Input dir filled + Output dir empty -> Output dir);
        3. otherwise the preferred field, else the last-focused path field.
        """
        def _empty(fid: str) -> bool:
            return not self.query_one(f'#{fid}', Input).value.strip()

        if preferred in _PATH_FIELDS and _empty(preferred):
            return preferred
        empties = [f for f in _PATH_FIELDS if _empty(f)]
        if len(empties) == 1:
            return empties[0]
        if preferred in _PATH_FIELDS:
            return preferred
        return self._last_path_field

    def _apply_dropped_path(self, text: str, preferred: str | None = None) -> bool:
        """Route a dropped/pasted path. Returns True if it was a path/YAML.

        A dropped ``.yaml`` / ``.yml`` is auto-loaded as settings. Any other
        existing folder/file (file -> its parent) fills the path field chosen
        by ``_drop_target``. Returns False for ordinary text so the caller
        can paste it normally.
        """
        raw = (text or '').strip()
        if not raw:
            return False
        cand = raw.splitlines()[0].strip().strip('"').strip("'")
        if cand.startswith('file://'):
            cand = unquote(urlparse(cand).path)
            if re.match(r'^/[A-Za-z]:', cand):
                cand = cand[1:]
        low = cand.lower()
        if (low.endswith('.yaml') or low.endswith('.yml')) \
                and Path(cand).expanduser().is_file():
            self._load_path(cand)
            return True
        folder = _clean_dropped_path(raw)
        if folder is None:
            return False
        target = self._drop_target(preferred)
        self.query_one(f'#{target}', Input).value = folder
        self._status(f'{target.replace("_", " ")} set from dropped folder',
                     _GREEN)
        return True

    @on(Button.Pressed, '.clearbtn')
    def _on_clear(self, event: Button.Pressed) -> None:
        fid = (event.button.id or '')[4:]  # strip 'clr_'
        if fid:
            self.query_one(f'#{fid}', Input).value = ''
            self.query_one(f'#{fid}', Input).focus()
        event.stop()

    # ---- form enable/disable + run-control button states ---------------
    def _set_form_enabled(self, enabled: bool) -> None:
        """Grey out (or restore) every settings widget during a run."""
        for fid in _FIELD_IDS:
            self.query_one(f'#{fid}', Input).disabled = not enabled
        for sid in _SWITCHES:
            self.query_one(f'#{sid}', Switch).disabled = not enabled
        for bid in ('save', 'load', 'reset',
                    *(f'clr_{f}' for f in _PATH_FIELDS),
                    *(f'pick_{f}' for f in _COL_PICK_FIELDS),
                    *(f'reseed_{f}' for f in _WIN_FIELDS)):
            try:
                self.query_one(f'#{bid}', Button).disabled = not enabled
            except NoMatches:
                pass

    def _controls_running(self, running: bool) -> None:
        """Toggle Run/Check/Stop button availability for a run in progress."""
        self.query_one('#run', Button).disabled = running
        self.query_one('#check', Button).disabled = running
        self.query_one('#stop', Button).disabled = not running
        self._set_form_enabled(not running)

    # ---- Check (preflight) ---------------------------------------------
    def action_check(self) -> None:
        if not self._busy:
            self._start_check()

    @on(Button.Pressed, '#check')
    def _on_check(self) -> None:
        self.action_check()

    # ---- Stop ----------------------------------------------------------
    def action_stop(self) -> None:
        if self._busy and self._cancel_event is not None:
            self._cancel_event.set()
            self.query_one('#stop', Button).disabled = True
            self._status('stopping… (finishing in-flight chunks)', _AMBER)

    @on(Button.Pressed, '#stop')
    def _on_stop(self) -> None:
        self.action_stop()

    # ---- Open output folder --------------------------------------------
    def action_open(self) -> None:
        d = self._last_output_dir
        if not d or not Path(d).exists():
            self._status('no output folder yet — run first', _DIM)
            return
        try:
            _open_in_file_manager(Path(d))
            self._status(f'opened {d}', _GREEN)
        except Exception as e:
            self._status(f'could not open folder: {e}', _RED)

    @on(Button.Pressed, '#open')
    def _on_open(self) -> None:
        self.action_open()

    def on_mount(self) -> None:
        # Worker-row pool: all hidden until a worker becomes active.
        self._wfree: list[int] = list(range(_MAX_WORKER_ROWS))
        self._wpid: dict[int, int] = {}    # worker pid -> row index
        self._wtext: dict[int, str] = {}   # row index -> static label markup
        self._wphase: str | None = None    # phase the current rows belong to
        self._spin = 0                     # current spinner frame index
        for i in range(_MAX_WORKER_ROWS):
            self.query_one(f'#wrow{i}', Label).display = False
        # Drive the spinner animation: re-render active rows ~12x/second.
        self.set_interval(1 / 12, self._tick_spinners)

        # Stop is only usable during a run; Open only after one produced output.
        self.query_one('#stop', Button).disabled = True
        self.query_one('#open', Button).disabled = True

        loaded = _load_settings()
        for fid in _FIELD_IDS:
            saved = loaded.get(fid)
            if saved not in (None, ''):
                value = str(saved)          # restore the saved value
            elif fid in _DEFAULTS:
                value = _DEFAULTS[fid]       # concrete default (columns/scalars)
            else:
                value = ''                  # leave blank -> placeholder shows default
            self.query_one(f'#{fid}', Input).value = value
        for sid in _SWITCHES:
            self.query_one(f'#{sid}', Switch).value = bool(loaded.get(sid, False))

        # Reconcile the per-gas window field with the loaded scalars (adds any
        # gas missing a window, drops stale ones; keeps saved/edited windows).
        self._sync_win_field()

        if self.demo:
            self.query_one('#input_dir', Input).value = '(demo — no data needed)'
            self.query_one('#output_dir', Input).value = '(demo)'
            self._status('demo mode — press Run (r)', _LAV)
        elif loaded:
            self._status(f'settings loaded from {_SETTINGS_PATH}', _DIM)
        else:
            self._status('fill the form, then Run (r)', _DIM)

    # ---- actions / buttons ---------------------------------------------
    def action_quit(self) -> None:
        self.exit()

    def action_save(self) -> None:
        self._save_settings(announce=True)

    def action_run(self) -> None:
        self._start()

    def action_info(self) -> None:
        self.push_screen(InfoScreen())

    def action_reset(self) -> None:
        self._reset_params()

    @on(Button.Pressed, '#reset')
    def _on_reset(self) -> None:
        self._reset_params()

    def _reset_params(self) -> None:
        """Clear the PWB & chunking fields so their defaults are used."""
        for fid in _RESET_FIELDS:
            self.query_one(f'#{fid}', Input).value = ''
        self._status('PWB & chunking reset — defaults will be used', _LAV)

    @on(Button.Pressed, '#quit')
    def _on_quit(self) -> None:
        self.exit()

    @on(Button.Pressed, '#save')
    def _on_save(self) -> None:
        self._save_settings(announce=True)

    def action_load(self) -> None:
        self.push_screen(LoadScreen(str(_SETTINGS_PATH)), self._on_loaded)

    @on(Button.Pressed, '#load')
    def _on_load(self) -> None:
        self.action_load()

    def _on_loaded(self, path) -> None:
        if path:
            self._load_path(path)

    def _load_path(self, path_str: str) -> bool:
        """Read a YAML settings file and overlay it onto the form."""
        s = path_str.strip().strip('"').strip("'")
        if s.startswith('file://'):
            s = s[7:]
        p = Path(s).expanduser()
        try:
            data = yaml.safe_load(p.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('file is not a YAML mapping')
        except Exception as e:
            self._status(f'could not load {p}: {e}', _RED)
            return False
        for fid in _FIELD_IDS:
            if data.get(fid) is not None:
                self.query_one(f'#{fid}', Input).value = str(data[fid])
        for sid in _SWITCHES:
            if sid in data:
                self.query_one(f'#{sid}', Switch).value = bool(data[sid])
        self._status(f'settings loaded from {p}', _GREEN)
        return True

    def on_paste(self, event: events.Paste) -> None:
        """Handle a drag-and-drop / paste that reaches the app.

        Fires for drops that don't land on a focused ``PathInput`` (which
        routes its own). A dropped ``.yaml`` is auto-loaded; any other
        folder/file fills the path field chosen by ``_drop_target`` —
        preferring an empty one, so a drop fills Output dir once Input dir
        is set, without clicking it first.
        """
        if self._apply_dropped_path(event.text or '',
                                    preferred=self._last_path_field):
            event.stop()

    @on(Button.Pressed, '#run')
    def _on_run(self) -> None:
        self._start()

    def _start(self) -> None:
        if self._busy:
            return
        # Demo needs no config.
        if self.demo:
            self.query_one('#log', RichLog).clear()
            self._logbuf.clear()
            self._clear_workers()
            self._busy = True
            self._cancel_event = threading.Event()
            self._controls_running(True)
            self._phase = None
            self._status('running…', _BLUE)
            threading.Thread(target=self._demo_impl, daemon=True).start()
            return
        # Collect + validate on the UI thread (so we never read widgets from a
        # worker thread, and a bad config is reported without disabling the
        # form first).
        try:
            cfg = self._collect()
        except Exception as e:
            self._status(f'bad input: {e}', _RED)
            return
        # Auto-preflight: cheap validation (files match + columns present) so
        # a misconfigured run aborts now instead of failing mid-way.
        ok, msg = self._quick_preflight(cfg)
        if not ok:
            self._status(f'cannot run: {msg} — press Check for details', _RED)
            return
        self.query_one('#log', RichLog).clear()
        self._logbuf.clear()
        self._clear_workers()
        self._busy = True
        self._cancel_event = threading.Event()
        self._controls_running(True)
        self._phase = None
        # Overwrite guard: note pre-existing output so a re-run into the wrong
        # folder is not silently destructive.
        existing = self._count_existing_output(cfg)
        if existing:
            self._log_only(Text(
                f'note: {cfg["data_subdir"]}/ already has {existing} file(s) '
                f'— matching chunks will be overwritten', style=_AMBER))
        self._status('running…', _BLUE)
        self._save_settings(announce=False)  # remember what we ran (home file)
        # Drop a TUI-loadable settings YAML into the run's output folder, at
        # run start, so the run is reproducible straight from its own output.
        saved = self._save_settings_to_output(cfg['output_dir'])
        if saved:
            self._log_only(Text(f'settings -> {saved}', style=_CYAN))
        threading.Thread(target=self._real_impl, args=(cfg,),
                         daemon=True).start()

    def _quick_preflight(self, cfg: dict) -> tuple:
        """Fast pre-run validation. Return (ok, message). Reads only the
        first file's header — no chunk processing."""
        try:
            f0, files, cols = _scan_columns(
                cfg['input_dir'], cfg['file_pattern'], cfg['skiprows'],
                cfg['extra_rows'], cfg['sep'])
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f'cannot read header: {e}'
        colset = set(cols)
        needed = {'U': cfg['col_u'], 'V': cfg['col_v'], 'W': cfg['col_w'],
                  'T_SONIC': cfg['col_tsonic']}
        needed.update(cfg['scalars'])
        missing = [f'{role} ({col!r})'
                   for role, col in needed.items() if col not in colset]
        if missing:
            return False, 'columns not in header: ' + ', '.join(missing)
        return True, f'{len(files)} file(s), {len(cols)} columns'

    def _count_existing_output(self, cfg: dict) -> int:
        """Count files already present in the output data subfolder."""
        try:
            d = Path(cfg['output_dir']) / cfg['data_subdir']
            return sum(1 for _ in d.glob('*')) if d.exists() else 0
        except Exception:
            return 0

    def _start_check(self) -> None:
        """Run the preflight checks (no pipeline) in a worker thread."""
        if self._busy:
            return
        try:
            cfg = self._collect()
        except Exception as e:
            self._status(f'check failed: {e}', _RED)
            return
        self.query_one('#log', RichLog).clear()
        self._logbuf.clear()
        self._busy = True
        self._controls_running(True)
        self.query_one('#stop', Button).disabled = True  # nothing to stop
        self._status('checking configuration…', _BLUE)
        threading.Thread(target=self._check_impl, args=(cfg,),
                         daemon=True).start()

    def _check_impl(self, cfg: dict) -> None:
        """Preflight: validate paths, file count, header columns, chunk plan.

        Reads only the first matching file's header (fast), so a
        misconfiguration is caught in well under a second instead of after a
        long run. ``cfg`` was already collected on the UI thread.
        """
        from diive.flux.hires.detect_and_remove_tlag import (
            _WHITESPACE_SEP as _WS, _chunk_filename, _count_data_rows)

        def log(line):
            self.call_from_thread(self._log_only, line)

        ok = True
        try:
            in_dir = Path(cfg['input_dir'])
            files = sorted(in_dir.glob(cfg['file_pattern']))
            if not files:
                log(Text(f"✗ no files match '{cfg['file_pattern']}' in {in_dir}",
                         style=_RED))
                self.call_from_thread(self._finish_check, 'check: no input files', False)
                return
            log(Text(f"✓ {len(files)} file(s) match "
                     f"'{cfg['file_pattern']}'", style=_GREEN))

            f0 = files[0]
            skiprows, extra = cfg['skiprows'], cfg['extra_rows']
            sep = cfg['sep']
            n_pre = skiprows + 1 + extra
            with open(f0, 'r', encoding='utf-8', errors='replace') as fh:
                head = [next(fh) for _ in range(n_pre)]
            header_line = head[skiprows].rstrip('\r\n')
            cols = (header_line.split() if sep == _WS
                    else [c.strip() for c in header_line.split(sep)])
            log(Text(f"✓ first file {f0.name}: {len(cols)} columns parsed "
                     f"(skiprows={skiprows}, extra-rows={extra})", style=_GREEN))
            # List every column so the exact (bracketed) names can be copied
            # or picked via the ▾ buttons.
            log(Text('  columns: ' + '  '.join(cols), style=_DIM))

            # Verify every configured column exists in the header.
            needed = {'U': cfg['col_u'], 'V': cfg['col_v'], 'W': cfg['col_w'],
                      'T_SONIC': cfg['col_tsonic']}
            for lbl, col in cfg['scalars'].items():
                needed[lbl] = col
            colset = set(cols)
            for role, col in needed.items():
                if col in colset:
                    log(Text(f"✓ {role}: '{col}' found", style=_GREEN))
                else:
                    ok = False
                    log(Text(f"✗ {role}: '{col}' NOT in header — check the "
                             f"name / separator / skiprows", style=_RED))

            # Chunk plan + first output filename.
            n_rows = _count_data_rows(f0, n_pre)
            chunk_records = int(round(cfg['chunk_seconds'] * cfg['hz']))
            n_chunks = max(1, (n_rows + chunk_records - 1) // chunk_records)
            log(Text(f"✓ {f0.name}: ~{n_rows} data rows -> ~{n_chunks} "
                     f"chunk(s)/file (≈{n_chunks * len(files)} total)",
                     style=_CYAN))
            try:
                name0, _ = _chunk_filename(
                    f0, 0, cfg['chunk_seconds'], cfg['chunk_name_template'],
                    cfg['start_time_regex'], cfg['start_time_format'])
                log(Text(f"✓ first output file would be: {name0}", style=_CYAN))
            except Exception as e:
                ok = False
                log(Text(f"✗ chunk naming: {e}", style=_RED))

            msg = ('check passed — ready to Run' if ok
                   else 'check found problems (see log)')
            self.call_from_thread(self._finish_check, msg, ok)
        except Exception as e:
            self.call_from_thread(self._finish_check,
                                  f'check error: {type(e).__name__}: {e}', False)

    def _finish_check(self, msg: str, ok: bool) -> None:
        self._busy = False
        self._controls_running(False)
        self._status(msg, _GREEN if ok else _RED)

    # ---- live per-worker rows (driven by the pipeline's on_active) ------
    def ui_active(self, active: dict, phase: str) -> None:
        """Show one spinner row per busy worker: '⠹ <phase> <file>·cNN'.

        ``active`` maps worker pid -> {'parent', 'chunk_index',
        'chunk_period'} and is delivered the instant a worker STARTS a chunk
        (not when it finishes), so the current file/chunk appears immediately.
        Each row shows an animated spinner (driven by ``_tick_spinners``)
        while the worker is busy; when its chunk finishes the result line
        (CH4=… HDI=…) is appended to the log below. Rows stay assigned to a
        pid for the whole run (workers are reused), so they don't flicker.
        """
        try:
            # Phase boundary: detect and remove run in separate process pools
            # (different pids), so drop the previous phase's rows before
            # showing the new phase's workers — otherwise stale 'detect …'
            # rows linger while 'remove' is running.
            if phase != self._wphase:
                self._clear_workers()
                self._wphase = phase
            for pid in sorted(active):
                info = active[pid]
                if pid not in self._wpid:
                    if not self._wfree:
                        continue  # more busy workers than rows; skip extras
                    idx = self._wfree.pop(0)
                    self._wpid[pid] = idx
                    self.query_one(f'#wrow{idx}', Label).display = True
                idx = self._wpid[pid]
                stem = Path(info.get('parent', '')).stem
                if len(stem) > 20:
                    stem = stem[:9] + '…' + stem[-10:]
                ci = info.get('chunk_index', 0)
                ptag = _LAV if phase == 'detect' else _CYAN
                disp = _phase_label(phase)
                # Static part of the row (spinner frame is prepended on tick).
                self._wtext[idx] = f'[{ptag}]{disp}[/] [b]{stem}[/b]·c{ci:02d}'
            self._render_spinners()
            self._status(f'{_phase_label(phase)} · {len(active)} workers busy',
                         _BLUE)
        except NoMatches:
            # App is tearing down (quit mid-run): widgets gone, nothing to do.
            return

    def _render_spinners(self) -> None:
        """Paint the current spinner frame onto every active worker row."""
        frame = _SPINNER[self._spin]
        try:
            for idx in self._wpid.values():
                txt = self._wtext.get(idx)
                if txt is not None:
                    self.query_one(f'#wrow{idx}', Label).update(
                        f'[{_CYAN}]{frame}[/] {txt}')
        except NoMatches:
            pass

    def _tick_spinners(self) -> None:
        """Advance the spinner animation (called on a timer)."""
        if not self._wpid:
            return
        self._spin = (self._spin + 1) % len(_SPINNER)
        self._render_spinners()

    def _clear_workers(self) -> None:
        """Hide and release every worker row (run start / end)."""
        try:
            for idx in range(_MAX_WORKER_ROWS):
                row = self.query_one(f'#wrow{idx}', Label)
                row.display = False
                row.update('')
        except NoMatches:
            pass
        self._wpid = {}
        self._wtext = {}
        self._wfree = list(range(_MAX_WORKER_ROWS))
        self._wphase = None

    # ---- settings persistence ------------------------------------------
    def _settings_dict(self) -> dict:
        """Current form values keyed by field id (the TUI's YAML schema)."""
        data = {fid: self.query_one(f'#{fid}', Input).value for fid in _FIELD_IDS}
        for sid in _SWITCHES:
            data[sid] = self.query_one(f'#{sid}', Switch).value
        return data

    def _save_settings(self, announce: bool) -> None:
        data = self._settings_dict()
        try:
            _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SETTINGS_PATH.write_text(yaml.safe_dump(data, sort_keys=False),
                                      encoding='utf-8')
            if announce:
                self._status(f'settings saved to {_SETTINGS_PATH}', _GREEN)
        except Exception as e:
            if announce:
                self._status(f'could not save settings: {e}', _RED)

    def _save_settings_to_output(self, output_dir: str) -> str | None:
        """Drop a TUI-loadable settings YAML into the run's output folder.

        Same schema as the persisted settings file, so it can be loaded
        straight back into the TUI (Load button, or drag the file onto the
        window) to reproduce or inspect the run. Returns the path written, or
        None on failure (never raises — a settings copy must not block a run).
        """
        try:
            path = Path(output_dir) / _OUTPUT_SETTINGS_NAME
            path.write_text(yaml.safe_dump(self._settings_dict(), sort_keys=False),
                            encoding='utf-8')
            return str(path)
        except Exception:
            return None

    # ---- UI updates (always called via call_from_thread) ---------------
    def _status(self, msg: str, color: str = _DIM) -> None:
        self.query_one('#status', Static).update(f'[{color}]{msg}[/]')

    def _scan_status(self, msg: str) -> None:
        """Surface coarse pre-phase-1 progress (the up-front file scan).

        Drives both the one-line status (live counter) and the console log
        (so the big console pane is not empty while many/large inputs are
        being counted before the first chunk completes). Guarded against
        teardown races like the other live-update methods.
        """
        try:
            self._status(msg, _BLUE)
            self._log_only(Text(msg, style=_DIM))
        except NoMatches:
            return

    def ui_update(self, phase: str, done: int, total: int, line) -> None:
        try:
            bar = self.query_one('#bar', ProgressBar)
        except NoMatches:
            return  # app tearing down
        if phase != self._phase:
            self._phase = phase
            bar.update(total=total, progress=0)
            tag = _LAV if phase == 'detect' else _CYAN
            self.query_one('#phase', Static).update(
                f'[{tag}]{_phase_label(phase)}[/]')
        bar.update(total=total, progress=done)
        if line is not None:
            # Prefix every console line with a wall-clock timestamp, matching
            # the CLI's `console.log` style, and mirror the plain text into
            # the copy buffer (issues: time in output + copyable log).
            ts = datetime.now().strftime('%H:%M:%S')
            stamped = Text.assemble((f'{ts} ', _DIM), line)
            self.query_one('#log', RichLog).write(stamped)
            self._logbuf.append(stamped.plain)

    def _log_only(self, line) -> None:
        """Write one line to the console log (timestamped) without touching
        the progress bar. Used for end-of-run notices."""
        ts = datetime.now().strftime('%H:%M:%S')
        stamped = Text.assemble((f'{ts} ', _DIM), line)
        self.query_one('#log', RichLog).write(stamped)
        self._logbuf.append(stamped.plain)

    def action_copy(self) -> None:
        """Copy the entire console log to the system clipboard."""
        if not self._logbuf:
            self._status('nothing to copy yet', _DIM)
            return
        text = '\n'.join(self._logbuf)
        try:
            self.copy_to_clipboard(text)
            self._status(f'copied {len(self._logbuf)} log lines to clipboard',
                         _GREEN)
        except Exception as e:
            self._status(f'copy failed: {e}', _RED)

    @on(Button.Pressed, '#copy')
    def _on_copy(self) -> None:
        self.action_copy()

    def _finish(self, msg: str, color: str = _GREEN) -> None:
        self._clear_workers()
        self._busy = False
        self._cancel_event = None
        self._controls_running(False)
        if self._last_output_dir and Path(self._last_output_dir).exists():
            self.query_one('#open', Button).disabled = False
        self._status(msg, color)

    def _error(self, msg: str) -> None:
        self._clear_workers()
        self._busy = False
        self._cancel_event = None
        self._controls_running(False)
        self._status(msg, _RED)

    # ---- real pipeline (worker thread) ---------------------------------
    def _real_impl(self, cfg: dict) -> None:
        try:
            pipeline = PerFilePipeline(**cfg)
            scalars = pipeline.scalars
            hz = pipeline.hz

            def _applied(row, g):
                # The applied lag in seconds = shifted records / hz (the shift
                # is by whole records, so this is the exact lag removed).
                rec = row.get(f'{g.lower()}_applied_records')
                if rec is None or rec != rec:   # NaN -> not applied
                    return f'{g}=--'
                rec = int(rec)
                return f'{g}={rec / hz:.2f}s ({rec}rec)'

            def on_progress(done, total, row, phase):
                chunk = Path(row.get('period', '')).stem
                parent = Path(row.get('parent', '')).stem
                # Show the source (6 h) file the chunk came from, then the
                # chunk itself, so each line is traceable to its input file.
                label = f'{parent} › {chunk}' if parent else chunk
                if phase == 'remove':
                    applied = '  '.join(_applied(row, g) for g in scalars)
                    line = Text(f'{label}  aligned {applied}', style=_GREEN)
                elif row.get('status') == 'error':
                    line = Text(f'{label}  ERROR {row.get("error", "")[:50]}', style=_RED)
                else:
                    line = _detect_line(label, row, scalars)
                self.call_from_thread(self.ui_update, phase, done, total, line)

            def on_active(active, phase):
                # active fires at chunk START -> show the live worker rows.
                self.call_from_thread(self.ui_active, dict(active), phase)

            def on_status(msg):
                # Coarse pre-phase-1 progress (the up-front file scan reads
                # every input file and emits no per-chunk callback, so without
                # this the console sits empty while large inputs are counted).
                self.call_from_thread(self._scan_status, msg)

            # Remember the output dir for the Open button (even on cancel).
            self._last_output_dir = cfg['output_dir']

            # run() writes the summary CSV + overview plots itself; the cancel
            # event lets the Stop button abort mid-run.
            summary = pipeline.run(on_progress=on_progress, on_active=on_active,
                                   cancel_event=self._cancel_event,
                                   on_status=on_status)
            n_ok = int((summary['status'] == 'ok').sum()) if 'status' in summary else 0
            if pipeline.summary_csv_path is not None:
                self.call_from_thread(
                    self._log_only,
                    Text(f'summary -> {pipeline.summary_csv_path}', style=_CYAN))
            if pipeline.summary_plots_dir is not None:
                self.call_from_thread(
                    self._log_only,
                    Text(f'overview plots -> {pipeline.summary_plots_dir}',
                         style=_CYAN))
            # Post-run summary block (counts, reliability, median lags).
            for ln in _summary_lines(summary, list(scalars), hz,
                                     cancelled=pipeline.cancelled):
                self.call_from_thread(self._log_only, ln)
            if pipeline.cancelled:
                self.call_from_thread(
                    self._finish,
                    f'stopped — {n_ok}/{len(summary)} chunks aligned '
                    f'-> {cfg["output_dir"]}', _AMBER)
            else:
                self.call_from_thread(
                    self._finish,
                    f'done — {n_ok}/{len(summary)} chunks -> {cfg["output_dir"]}')
        except Exception as e:
            # Surface the full error in the console log (the status line alone
            # truncates it), then mark the run failed.
            msg = f'{type(e).__name__}: {e}'
            self.call_from_thread(self._log_only, Text(f'ERROR  {msg}', style=_RED))
            self.call_from_thread(self._error, msg)

    def _collect(self) -> dict:
        def g(i):
            return self.query_one(f'#{i}', Input).value.strip()

        # Scalars field selects the gases only (LABEL:column); per-gas search
        # windows live in the Win field.
        scalars = {}
        for tok in g('scalars').split(','):
            tok = tok.strip()
            if not tok:
                continue
            label, col, _ = parse_scalar_spec(tok)
            scalars[label] = col
        if not scalars:
            raise ValueError('no scalars (use LABEL:column, comma-separated)')
        # Per-gas lag windows from the Win field: LABEL:[lower,upper] (seconds).
        # Each window sets that gas's lag_max (= window half-width) and block
        # length (2x half), so a long-inlet gas (H2O) gets its own range while
        # the dry gases keep theirs. A gas without a window uses the global
        # Lag max s symmetrically.
        per_gas_lag: dict = {}
        windows = parse_win_ranges(g('winranges'))
        for label, (lo, hi) in windows.items():
            if label not in scalars:
                continue  # stale entry for a removed gas; ignore
            try:
                per_gas_lag[label] = window_to_lag_params(lo, hi)
            except ValueError as e:
                raise ValueError(f'window for {label}: {e}')
        in_dir, out_dir = g('input_dir'), g('output_dir')
        if not in_dir or not out_dir:
            raise ValueError('input dir and output dir are required')
        # Catch a botched drag-and-drop: some terminals (e.g. Windows
        # Terminal) deliver a dropped path as typed keystrokes, so dropping
        # onto a filled field appends instead of replaces, producing a string
        # with two drive letters like 'F:\a\bF:\a\b'. That yields a cryptic
        # WinError 123 later — flag it here with a fix instead.
        for nm, val in (('Input dir', in_dir), ('Output dir', out_dir)):
            if len(re.findall(r'[A-Za-z]:[\\/]', val)) > 1 or '\x00' in val:
                raise ValueError(
                    f'{nm} looks like two paths joined together — clear the '
                    f'field (clear button next to it) and enter it once '
                    f'({val!r}).')
        ip = Path(in_dir).expanduser()
        if not ip.is_dir():
            raise ValueError(f'Input dir is not an existing folder: {in_dir}')
        op = Path(out_dir).expanduser()
        try:
            op.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f'Output dir is not a valid path: {out_dir} ({e})')
        workers = int(g('workers') or 0)
        # Chunk naming: only pass a start-time regex when given. The pipeline
        # raises a clear error if the template uses {starttime} without one.
        streg = g('streg') or None
        ctmpl = g('ctmpl') or '{stem}_chunk{index:02d}{suffix}'
        if '{starttime}' in ctmpl and not streg:
            raise ValueError(
                "Name tmpl uses {starttime} but Start regex is empty — "
                "set Start regex (e.g. (\\d{12})) and Start format, or drop "
                "{starttime} from the template.")
        rs = g('randomstate')
        navalues = g('navalues').split()  # space-separated; [] -> pipeline default
        return dict(
            input_dir=in_dir, output_dir=out_dir,
            col_u=g('col_u'), col_v=g('col_v'), col_w=g('col_w'),
            col_tsonic=g('col_tsonic'), scalars=scalars,
            hz=int(g('hz') or 20),
            lag_max_s=float(g('lagmax') or 10.0),
            n_bootstrap=int(g('nboot') or 99),
            per_gas_lag=per_gas_lag,
            chunk_seconds=float(g('chunk') or 1800),
            min_chunk_seconds=float(g('minchunk') or 300),
            n_workers=workers if workers > 0 else None,
            save_plots=self.query_one('#saveplots', Switch).value,
            strict=self.query_one('#strict', Switch).value,
            random_state=int(rs) if rs else None,
            # PWBOPT best-lag selection
            hdi_thresh=float(g('hdithresh') or 0.5),
            dev_thresh=float(g('devthresh') or 0.5),
            hdi_prefilter=float(g('hdiprefilter') or 1.0),
            lag_column_template=g('lagcol') or '{prefix}_tlag_final_pf_s',
            # File format
            skiprows=int(g('skiprows') or 0),
            extra_rows=int(g('extrarows') or 2),
            sep=_unescape_sep(g('sep') or ','),
            file_pattern=g('filepattern') or '*.csv',
            na_values=navalues if navalues else None,
            na_rep=g('narep') or '-9999',
            # 'auto' (default) reproduces the input file's CRLF/LF convention.
            lineterm=(lt if (lt := g('lineterm')) == 'auto'
                      else _unescape_sep(lt or 'auto')),
            # Chunk naming (controls the output filenames)
            start_time_regex=streg,
            start_time_format=g('stfmt') or '%Y%m%d-%H%M',
            chunk_name_template=ctmpl,
            # Output layout
            detect_subdir=g('detectsub') or '1_lag_detection',
            data_subdir=g('datasub') or '2_lag_removed',
        )

    # ---- demo (no input data) ------------------------------------------
    def _demo_impl(self) -> None:
        scalars = ['CH4', 'N2O']
        files = ['CH-CHA_202107261300', 'CH-CHA_202107271300',
                 'CH-CHA_202107281300']
        chunks = [(f, ci) for f in files for ci in range(12)]
        total = len(chunks)
        n_workers = 4
        pids = [1000 + k for k in range(n_workers)]   # fake worker pids

        def _cancelled():
            return (self._cancel_event is not None
                    and self._cancel_event.is_set())

        def _phase(phase: str, hold: float, write: bool):
            # Simulate N workers each grabbing the next chunk: emit an
            # 'active' snapshot when a chunk STARTS (so the live rows + bars
            # appear up front), then the completion line when it finishes.
            active: dict = {}
            done = 0
            it = iter(enumerate(chunks))
            buffered = list(it)
            pos = 0
            # Prime: assign the first batch of chunks to workers.
            inflight: dict = {}  # pid -> (i, f, ci)
            for pid in pids:
                if pos < len(buffered):
                    i, (f, ci) = buffered[pos]; pos += 1
                    inflight[pid] = (i, f, ci)
                    active[pid] = {'parent': f, 'chunk_index': ci,
                                   'chunk_period': f}
            self.call_from_thread(self.ui_active, dict(active), phase)
            while inflight:
                if _cancelled():
                    return
                time.sleep(hold)
                # Finish one worker's chunk, hand it the next.
                pid = next(iter(inflight))
                i, f, ci = inflight.pop(pid)
                done += 1
                if write:
                    line = Text(
                        f'{f}·c{ci:02d}  aligned '
                        f'CH4=1.75s (35rec)  N2O=1.80s (36rec)', style=_GREEN)
                else:
                    row = {}
                    for g in scalars:
                        pfx = g.lower()
                        row[f'{pfx}_tlag_s'] = 1.7 + 0.25 * random.random()
                        row[f'{pfx}_hdi_range_s'] = (
                            random.choice([0.05, 0.1, 0.6, 1.4]) * random.random())
                    line = _detect_line(f'{f}·c{ci:02d}', row, scalars)
                self.call_from_thread(self.ui_update, phase, done, total, line)
                if pos < len(buffered):
                    i, (f, ci) = buffered[pos]; pos += 1
                    inflight[pid] = (i, f, ci)
                    active[pid] = {'parent': f, 'chunk_index': ci,
                                   'chunk_period': f}
                else:
                    active.pop(pid, None)
                self.call_from_thread(self.ui_active, dict(active), phase)

        _phase('detect', 0.12, write=False)
        if not _cancelled():
            _phase('remove', 0.08, write=True)
        if _cancelled():
            self.call_from_thread(self._finish, 'demo stopped', _AMBER)
        else:
            self.call_from_thread(
                self._finish, f'demo done — {total} chunks (no files written)')


def _tui_main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog='diive-tlag-pwb-detect-remove-tui',
        description='Textual UI for diive-tlag-pwb-detect-remove. '
                    'Pass --demo to preview the interface without input data.')
    p.add_argument('--demo', action='store_true',
                   help='Run a synthetic pipeline (no input data required).')
    args = p.parse_args()
    DetectRemoveTUI(demo=args.demo).run()


if __name__ == '__main__':
    _tui_main()
