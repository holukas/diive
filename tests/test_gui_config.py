"""
TEST_GUI_CONFIG: atomic persistence of the GUI preferences file
==============================================================

Covers ``diive.gui.config`` alone: the settings blob holds theme, window
geometry, ``last_project`` and the per-dataset ``variable_metadata``, so a
half-written file loses real user state. Saving therefore serializes to a
temporary file next to the target and swaps it in with ``os.replace``; the
target must never be observable in a partial state, and neither path may leave
a temporary file behind.

No ``QApplication`` and no widget: ``config_file`` is monkeypatched and
``save_config``/``load_config`` are called directly, so this module runs in
milliseconds.

Run: pytest tests/test_gui_config.py -v

Part of the diive library: https://github.com/holukas/diive
"""
import json
import os

import pytest

config = pytest.importorskip("diive.gui.config",
                             reason="requires the 'gui' extra")


@pytest.fixture
def target(tmp_path, monkeypatch):
    """Point the config module at a throwaway settings file."""
    path = tmp_path / "gui_settings.json"
    monkeypatch.setattr(config, "config_file", lambda: path)
    return path


def _dir_entries(path):
    """Names in the settings file's directory (to catch temp-file leftovers)."""
    return sorted(p.name for p in path.parent.iterdir())


def test_save_config_roundtrip(target):
    """A normal save lands on the real target and reads back unchanged."""
    blob = {"theme": {"tokens": {"CANVAS": "#ffffff"}}, "last_project": "p.diive"}
    config.save_config(blob)

    assert config.load_config() == blob
    # The blob really is in the target file, not only in some temporary one.
    assert json.loads(target.read_text(encoding="utf-8")) == blob
    assert _dir_entries(target) == ["gui_settings.json"]


def test_failed_swap_leaves_previous_file_intact(target, monkeypatch):
    """An interrupted save must not be visible in the target file.

    Standing in for the crash/interruption: the atomic swap itself fails. The
    new content has been fully written by then, but to a temporary file, so the
    previous preferences survive complete and the temporary file is cleaned up.
    """
    config.save_config({"theme": "good", "last_project": "p.diive"})

    def _boom(src, dst):
        raise OSError("simulated interruption")

    monkeypatch.setattr(config.os, "replace", _boom)
    config.save_config({"theme": "half-written"})  # must not raise

    assert config.load_config() == {"theme": "good", "last_project": "p.diive"}
    assert _dir_entries(target) == ["gui_settings.json"]


def test_unserializable_value_leaves_previous_file_intact(target):
    """G8: an unencodable value is swallowed, with the old file left complete."""
    config.save_config({"a": 1})
    config.save_config({"theme": {"token": object()}})  # must not raise

    assert config.load_config() == {"a": 1}
    assert _dir_entries(target) == ["gui_settings.json"]
    # Saving still works afterwards.
    config.save_config({"a": 2})
    assert config.load_config() == {"a": 2}


def test_load_config_survives_corrupt_file(target):
    """A corrupt config may not stop the app from starting."""
    target.write_text('{"theme": {"tokens"', encoding="utf-8")  # truncated
    assert config.load_config() == {}

    # Valid JSON that is not an object: would reach every `cfg.get(...)` call
    # site in `app.py` as a non-dict.
    target.write_text("[1, 2]", encoding="utf-8")
    assert config.load_config() == {}
    target.write_text("null", encoding="utf-8")
    assert config.load_config() == {}


def test_save_config_creates_the_file_when_absent(target):
    """First-ever save (no existing file to replace)."""
    assert not target.exists()
    config.save_config({"last_filetype": "parquet"})

    assert config.load_config() == {"last_filetype": "parquet"}
    assert _dir_entries(target) == ["gui_settings.json"]


def test_save_config_swallows_an_unwritable_target(tmp_path, monkeypatch):
    """An unwritable config directory is swallowed and leaves nothing behind."""
    missing = tmp_path / "nope" / "gui_settings.json"  # parent does not exist
    monkeypatch.setattr(config, "config_file", lambda: missing)
    config.save_config({"a": 1})  # must not raise

    assert config.load_config() == {}
    assert not os.path.exists(missing.parent)
