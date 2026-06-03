"""
DETECT_AND_REMOVE_TLAG_TUI: Textual front-end for diive-tlag-pwb-detect-remove
==============================================================================

A two-column terminal UI (Textual) around ``PerFilePipeline``
(``diive.flux.hires.detect_and_remove_tlag``). The left column is a labelled
settings form; the right column is a live console: a progress bar plus a
``RichLog`` into which the two-phase pipeline (detect -> PWBOPT -> remove)
streams its Rich-styled per-chunk output (same colour coding as the CLI).

Settings persist between sessions in a small YAML file
(``~/.diive/detect_remove_tui.yaml``): they are loaded on start, and saved on
*Run* or via the *Save* button — so column names, paths and parameters need to
be entered only once.

A ``--demo`` mode runs a synthetic pipeline that needs no input data, purely so
the interface can be previewed.

Launch::

    uv run diive-tlag-pwb-detect-remove-tui --demo     # preview, no data
    uv run diive-tlag-pwb-detect-remove-tui            # real run (fill form)
    uv run python examples/flux/hires/flux_detect_remove_tui_demo.py

Part of the diive library: https://github.com/holukas/diive
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import yaml
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, Footer, Input, Label, ProgressBar,
                             RichLog, Static, Switch)

from diive.flux.hires.detect_and_remove_tlag import PerFilePipeline

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
#settings {
    width: 66; background: #16161e; border-right: solid #2f334d; padding: 0 1;
    scrollbar-size: 1 1;
}
#console { width: 1fr; padding: 0 1; }

.section { color: #7aa2f7; text-style: bold; height: 1; margin: 1 0 0 0; }
.field { height: 1; }
.flabel { width: 16; color: #565f89; content-align: left middle; }
.fin {
    width: 1fr; height: 1; border: none; background: #24283b; color: #c0caf5;
    padding: 0 1;
}
.fin:focus { background: #2f334d; color: #c0caf5; }
Switch { height: 1; border: none; background: #24283b; }
Switch.-on { background: #2f334d; }

#controls { height: 3; margin: 1 0 0 0; }
Button { margin: 0 1 0 0; min-width: 8; height: 3; }
Button#run { background: #7aa2f7; color: #1a1b26; text-style: bold; }
Button#save { background: #2f334d; color: #c0caf5; }
Button#quit { background: #2f334d; color: #c0caf5; }

#status { height: 1; color: #565f89; }
#progressrow { height: 1; }
#phase { width: 9; content-align: left middle; }
ProgressBar { width: 1fr; }
#bar Bar > .bar--bar { color: #7aa2f7; }
#bar Bar > .bar--complete { color: #9ece6a; }
#log {
    height: 1fr; background: #16161e; border: round #2f334d; padding: 0 1;
    scrollbar-size: 1 1;
}
Footer { background: #16161e; color: #565f89; }

InfoScreen { align: center middle; background: #1a1b26 70%; }
#dialog {
    width: 76; height: auto; max-height: 90%;
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
    ('scalars', 'Scalars', 'LABEL:column, e.g. CH4:ch4,N2O:n2o'),
    ('hz', 'Frequency', 'default 20  (samples per second)'),
    ('chunk', 'Chunk s', 'default 1800  (= 30 min)'),
    ('minchunk', 'Min chunk s', 'default 300  (skip shorter chunks)'),
    ('nboot', 'Bootstraps', 'default 99  (PWB replicates, paper value)'),
    ('lagmax', 'Lag max s', 'default 10.0  (CCF search half-width)'),
    ('block', 'Block s', 'default 20.0  (bootstrap block length)'),
    ('workers', 'Workers', 'default: all CPU cores'),
]
_FIELD_IDS = [f[0] for f in _FIELDS]

# Fields pre-filled with a concrete value on start (no sensible blank fallback).
# The PWB & chunking params and paths are left blank so their placeholder shows
# the default; the pipeline applies those defaults when the box is empty.
_DEFAULTS = {
    'col_u': 'u', 'col_v': 'v', 'col_w': 'w', 'col_tsonic': 'ts',
    'scalars': 'CH4:ch4,N2O:n2o',
}

# PWB & chunking params that fall back to a default when left blank (see
# _collect). The Reset button clears exactly these.
_RESET_FIELDS = ['hz', 'chunk', 'minchunk', 'nboot', 'lagmax', 'block', 'workers']

_SETTINGS_PATH = Path.home() / '.diive' / 'detect_remove_tui.yaml'


def _load_settings() -> dict:
    try:
        data = yaml.safe_load(_SETTINGS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    BINDINGS = [('r', 'run', 'Run'), ('s', 'save', 'Save'),
                ('l', 'load', 'Load'), ('d', 'reset', 'Reset'),
                ('i', 'info', 'Info'), ('q', 'quit', 'Quit')]

    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo = demo
        self._phase = None  # tracks detect->remove handoff for the bar

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
                yield self._field('block')
                yield self._field('workers')
                yield Horizontal(
                    Label('Save plots', classes='flabel'),
                    Switch(value=False, id='saveplots'),
                    classes='field',
                )
                with Horizontal(id='controls'):
                    yield Button('Run', id='run', variant='primary')
                    yield Button('Save', id='save')
                    yield Button('Load', id='load')
                    yield Button('Reset', id='reset')
                    yield Button('Quit', id='quit')
            with Vertical(id='console'):
                yield Static('idle', id='status')
                with Horizontal(id='progressrow'):
                    yield Static(f'[{_LAV}]detect[/]', id='phase')
                    yield ProgressBar(id='bar')
                yield RichLog(id='log', wrap=True, markup=False, highlight=False)
        yield Footer()

    def _field(self, fid: str) -> Horizontal:
        label = next(lbl for i, lbl, _ in _FIELDS if i == fid)
        ph = next(p for i, _, p in _FIELDS if i == fid)
        return Horizontal(
            Label(label, classes='flabel'),
            Input(placeholder=ph, id=fid, classes='fin'),
            classes='field',
        )

    def on_mount(self) -> None:
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
        self.query_one('#saveplots', Switch).value = bool(loaded.get('saveplots', False))

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
        if 'saveplots' in data:
            self.query_one('#saveplots', Switch).value = bool(data['saveplots'])
        self._status(f'settings loaded from {p}', _GREEN)
        return True

    def on_paste(self, event: events.Paste) -> None:
        """Drag-and-drop: most terminals paste the dropped file's path. If it's
        an existing .yaml/.yml file, load it automatically."""
        raw = (event.text or '').strip()
        if not raw:
            return
        cand = raw.splitlines()[0].strip().strip('"').strip("'")
        if cand.startswith('file://'):
            cand = cand[7:]
        low = cand.lower()
        if (low.endswith('.yaml') or low.endswith('.yml')) \
                and Path(cand).expanduser().is_file():
            self._load_path(cand)
            event.stop()

    @on(Button.Pressed, '#run')
    def _on_run(self) -> None:
        self._start()

    def _start(self) -> None:
        self.query_one('#log', RichLog).clear()
        self.query_one('#run', Button).disabled = True
        self._phase = None
        self._status('running…', _BLUE)
        if self.demo:
            self._worker_demo()
        else:
            self._save_settings(announce=False)  # remember what we ran
            self._worker_real()

    # ---- settings persistence ------------------------------------------
    def _save_settings(self, announce: bool) -> None:
        data = {fid: self.query_one(f'#{fid}', Input).value for fid in _FIELD_IDS}
        data['saveplots'] = self.query_one('#saveplots', Switch).value
        try:
            _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SETTINGS_PATH.write_text(yaml.safe_dump(data, sort_keys=False),
                                      encoding='utf-8')
            if announce:
                self._status(f'settings saved to {_SETTINGS_PATH}', _GREEN)
        except Exception as e:
            if announce:
                self._status(f'could not save settings: {e}', _RED)

    # ---- UI updates (always called via call_from_thread) ---------------
    def _status(self, msg: str, color: str = _DIM) -> None:
        self.query_one('#status', Static).update(f'[{color}]{msg}[/]')

    def ui_update(self, phase: str, done: int, total: int, line) -> None:
        bar = self.query_one('#bar', ProgressBar)
        if phase != self._phase:
            self._phase = phase
            bar.update(total=total, progress=0)
            tag = _LAV if phase == 'detect' else _CYAN
            self.query_one('#phase', Static).update(f'[{tag}]{phase}[/]')
        bar.update(total=total, progress=done)
        if line is not None:
            self.query_one('#log', RichLog).write(line)

    def _finish(self, msg: str) -> None:
        self.query_one('#run', Button).disabled = False
        self._status(msg, _GREEN)

    def _error(self, msg: str) -> None:
        self.query_one('#run', Button).disabled = False
        self._status(msg, _RED)

    # ---- real pipeline (worker thread) ---------------------------------
    @work(thread=True)
    def _worker_real(self) -> None:
        try:
            cfg = self._collect()
        except Exception as e:
            self.call_from_thread(self._error, f'bad input: {e}')
            return
        try:
            pipeline = PerFilePipeline(**cfg)
            scalars = pipeline.scalars

            def on_progress(done, total, row, phase):
                stem = Path(row.get('period', '')).stem
                if phase == 'remove':
                    applied = '  '.join(
                        f'{g}={row.get(f"{g.lower()}_applied_records", "--")}rec'
                        for g in scalars)
                    line = Text(f'{stem}  written {applied}', style=_GREEN)
                elif row.get('status') == 'error':
                    line = Text(f'{stem}  ERROR {row.get("error", "")[:50]}', style=_RED)
                else:
                    line = _detect_line(stem, row, scalars)
                self.call_from_thread(self.ui_update, phase, done, total, line)

            def on_active(active, phase):
                self.call_from_thread(
                    self._status, f'{phase} · {len(active)} workers active', _BLUE)

            summary = pipeline.run(on_progress=on_progress, on_active=on_active)
            try:
                csv = (pipeline.output_dir / pipeline.detect_subdir
                       / 'detect_and_remove_tlag_summary.csv')
                csv.parent.mkdir(parents=True, exist_ok=True)
                summary.to_csv(csv, index=False)
            except Exception:
                pass
            n_ok = int((summary['status'] == 'ok').sum()) if 'status' in summary else 0
            self.call_from_thread(
                self._finish, f'done — {n_ok}/{len(summary)} chunks -> {cfg["output_dir"]}')
        except Exception as e:
            self.call_from_thread(self._error, f'{type(e).__name__}: {e}')

    def _collect(self) -> dict:
        def g(i):
            return self.query_one(f'#{i}', Input).value.strip()

        scalars = {}
        for tok in g('scalars').split(','):
            tok = tok.strip()
            if not tok:
                continue
            label, col = tok.split(':', 1)
            scalars[label.strip()] = col.strip()
        if not scalars:
            raise ValueError('no scalars (use LABEL:column, comma-separated)')
        if not g('input_dir') or not g('output_dir'):
            raise ValueError('input dir and output dir are required')
        workers = int(g('workers') or 0)
        return dict(
            input_dir=g('input_dir'), output_dir=g('output_dir'),
            col_u=g('col_u'), col_v=g('col_v'), col_w=g('col_w'),
            col_tsonic=g('col_tsonic'), scalars=scalars,
            hz=int(g('hz') or 20),
            lag_max_s=float(g('lagmax') or 10.0),
            n_bootstrap=int(g('nboot') or 99),
            block_length_s=float(g('block') or 20.0),
            chunk_seconds=float(g('chunk') or 1800),
            min_chunk_seconds=float(g('minchunk') or 300),
            n_workers=workers if workers > 0 else None,
            save_plots=self.query_one('#saveplots', Switch).value,
        )

    # ---- demo (no input data) ------------------------------------------
    @work(thread=True)
    def _worker_demo(self) -> None:
        scalars = ['CH4', 'N2O']
        files = ['CH-CHA_20210726', 'CH-CHA_20210727', 'CH-CHA_20210728']
        chunks = [(f, ci) for f in files for ci in range(12)]
        total = len(chunks)
        for i, (f, ci) in enumerate(chunks, start=1):
            time.sleep(0.045)
            row = {}
            for g in scalars:
                pfx = g.lower()
                row[f'{pfx}_tlag_s'] = 1.7 + 0.25 * random.random()
                row[f'{pfx}_hdi_range_s'] = random.choice([0.05, 0.1, 0.6, 1.4]) * random.random()
            line = _detect_line(f'{f}·c{ci:02d}', row, scalars)
            self.call_from_thread(self.ui_update, 'detect', i, total, line)
        for i, (f, ci) in enumerate(chunks, start=1):
            time.sleep(0.03)
            line = Text(f'{f}·c{ci:02d}  written CH4=35rec  N2O=36rec', style=_GREEN)
            self.call_from_thread(self.ui_update, 'remove', i, total, line)
        self.call_from_thread(self._finish, f'demo done — {total} chunks (no files written)')


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
