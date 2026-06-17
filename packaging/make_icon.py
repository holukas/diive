"""Render the diive app icon (drawn in ``diive.gui.splash``) to a real
multi-resolution ``packaging/diive.ico`` so the packaged Windows EXE and its
taskbar button have a stable, embedded icon (procedural runtime icons leave the
Windows taskbar cache nothing to anchor to).

Run once whenever the icon artwork changes:  ``uv run python packaging/make_icon.py``
"""
from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # render without a display

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray
from PySide6.QtWidgets import QApplication

from diive.gui.splash import make_icon_pixmap

_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _pixmap_to_pil(size: int) -> Image.Image:
    pm = make_icon_pixmap(size)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    pm.toImage().save(buf, "PNG")
    buf.close()
    return Image.open(io.BytesIO(bytes(ba))).convert("RGBA")


def main() -> None:
    app = QApplication.instance() or QApplication([])  # needed to render QPixmaps
    images = [_pixmap_to_pil(s) for s in _SIZES]
    out = Path(__file__).with_name("diive.ico")
    # Embed every rendered size (each is tuned for its resolution) as one .ico.
    images[-1].save(out, format="ICO", sizes=[(s, s) for s in _SIZES],
                    append_images=images[:-1])
    print(f"wrote {out} ({out.stat().st_size} bytes, sizes {_SIZES})")
    del app


if __name__ == "__main__":
    main()
