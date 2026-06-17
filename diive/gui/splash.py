"""
GUI.SPLASH: STARTUP SPLASH SCREEN
=================================

A small, self-contained splash shown while the main window builds (it auto-loads
the example dataset, which takes a moment). Everything is drawn with `QPainter`
— no image assets — so it works regardless of what's on disk: a deep blue-teal
gradient, the diive wordmark + version + tagline, a band of layered sine
**waves**, and a credits line (author now, supporters later).

To add supporters, append names to ``SUPPORTERS``. GUI-only presentation; no
other module imports from here.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QSplashScreen, QVBoxLayout

import diive

#: The person who created and maintains diive (shown on the splash).
AUTHOR = "Lukas Hörtnagl"

#: One-line description under the wordmark.
TAGLINE = "Eddy covariance & environmental time series"

#: Supporters shown beneath the author. Append names here as they come in;
#: an empty list simply hides the supporters line. (The current entries are
#: placeholder example names — replace them with real supporters.)
SUPPORTERS: list[str] = ["Gusty McFluxface", "Anita Breeze", "Ed D. Covariance"]

_WIDTH = 620
_HEIGHT = 380
_RADIUS = 18


def build_number() -> str | None:
    """Build identifier baked into the packaged app by ``packaging/build_gui.ps1``.

    The build script writes a timestamp to ``_build_info.txt`` next to this module
    just before PyInstaller bundles it, so a packaged install can report exactly
    which build it is even when the version string is unchanged. Returns ``None``
    when running from source (no file present), where "build number" has no meaning.
    """
    try:
        text = Path(__file__).with_name("_build_info.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None

# Palette — deep water at the top fading to lighter teal; waves layered below.
_BG_TOP = QColor("#06293F")
_BG_BOTTOM = QColor("#0E5063")
_WAVE_COLORS = [
    QColor(21, 101, 192, 90),    # blue 800
    QColor(30, 136, 229, 110),   # blue 600
    QColor(38, 198, 218, 130),   # cyan 400
    QColor(128, 222, 234, 160),  # cyan 200
]
_WHITE = QColor("#FFFFFF")
_LIGHT = QColor("#B3E5FC")       # light blue 100
_FOOTER = QColor("#D6F1FC")      # near-white cyan — keeps the URL legible on the waves


def _draw_wave(p: QPainter, base_y: float, amp: float, wavelength: float,
               phase: float, color: QColor) -> None:
    """Fill one sine wave from `base_y` down to the bottom of the splash."""
    path = QPainterPath()
    path.moveTo(0.0, _HEIGHT)
    path.lineTo(0.0, base_y)
    x = 0.0
    while x <= _WIDTH:
        y = base_y + amp * math.sin(2 * math.pi * (x / wavelength) + phase)
        path.lineTo(x, y)
        x += 4.0
    path.lineTo(_WIDTH, _HEIGHT)
    path.closeSubpath()
    p.fillPath(path, color)


def make_splash_pixmap(dpr: float = 1.0) -> QPixmap:
    """Render the splash artwork to a (high-DPI aware) pixmap."""
    pm = QPixmap(int(_WIDTH * dpr), int(_HEIGHT * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Rounded card.
    card = QPainterPath()
    card.addRoundedRect(QRectF(0, 0, _WIDTH, _HEIGHT), _RADIUS, _RADIUS)
    p.setClipPath(card)

    grad = QLinearGradient(0, 0, 0, _HEIGHT)
    grad.setColorAt(0.0, _BG_TOP)
    grad.setColorAt(1.0, _BG_BOTTOM)
    p.fillRect(QRectF(0, 0, _WIDTH, _HEIGHT), grad)

    # Layered waves across the lower half (back = highest/softest, front = lowest).
    layers = [
        (218.0, 16.0, 320.0, 0.4, _WAVE_COLORS[0]),
        (250.0, 20.0, 250.0, 1.7, _WAVE_COLORS[1]),
        (288.0, 16.0, 200.0, 3.1, _WAVE_COLORS[2]),
        (322.0, 12.0, 160.0, 4.6, _WAVE_COLORS[3]),
    ]
    for base_y, amp, wavelength, phase, color in layers:
        _draw_wave(p, base_y, amp, wavelength, phase, color)

    # Wordmark.
    title = QFont()
    title.setPointSize(58)
    title.setWeight(QFont.Weight.Bold)
    title.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.0)
    p.setFont(title)
    p.setPen(_WHITE)
    p.drawText(QRectF(40, 48, _WIDTH - 80, 80),
               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "diive")

    # A small accent stroke under the wordmark.
    p.setPen(Qt.PenStyle.NoPen)
    p.fillRect(QRectF(46, 126, 86, 4), _WAVE_COLORS[2])

    # Tagline.
    tag = QFont()
    tag.setPointSize(13)
    p.setFont(tag)
    p.setPen(_LIGHT)
    p.drawText(QRectF(48, 138, _WIDTH - 96, 28),
               Qt.AlignmentFlag.AlignLeft, TAGLINE)

    # Version (top-right).
    ver = QFont()
    ver.setPointSize(11)
    ver.setWeight(QFont.Weight.DemiBold)
    p.setFont(ver)
    p.setPen(_LIGHT)
    p.drawText(QRectF(_WIDTH - 220, 56, 180, 24),
               Qt.AlignmentFlag.AlignRight, f"version {diive.__version__}")

    # Build identifier (packaged builds only) — distinguishes repeated deploys of
    # the same version. Drawn on a second, smaller line below the version.
    _build = build_number()
    if _build:
        bld = QFont()
        bld.setPointSize(8)
        p.setFont(bld)
        p.setPen(_LIGHT)
        p.drawText(QRectF(_WIDTH - 300, 80, 260, 18),
                   Qt.AlignmentFlag.AlignRight, f"build {_build}")

    # Credits, lower-left over the waves. A single plain names line: the author
    # first, then supporters.
    names_font = QFont()
    names_font.setPointSize(10)
    p.setFont(names_font)
    p.setPen(_LIGHT)
    p.drawText(QRectF(48, _HEIGHT - 98, _WIDTH - 96, 22),
               Qt.AlignmentFlag.AlignLeft, " · ".join([AUTHOR, *SUPPORTERS]))

    # Footer (the project URL). DemiBold + near-white cyan so it stays readable
    # over the bright front waves it sits on.
    foot = QFont()
    foot.setPointSize(9)
    foot.setWeight(QFont.Weight.DemiBold)
    p.setFont(foot)
    p.setPen(_FOOTER)
    p.drawText(QRectF(48, _HEIGHT - 34, _WIDTH - 96, 18),
               Qt.AlignmentFlag.AlignLeft, "github.com/holukas/diive")

    p.end()
    return pm


def _wave_path(s: float, base_y: float, amp: float, wl: float, phase: float) -> QPainterPath:
    """A closed sine-wave fill path from `base_y` down to the bottom edge `s`."""
    path = QPainterPath()
    path.moveTo(0.0, s)
    path.lineTo(0.0, base_y)
    x, step = 0.0, s / 96.0
    while x <= s:
        path.lineTo(x, base_y + amp * math.sin(2 * math.pi * (x / wl) + phase))
        x += step
    path.lineTo(s, s)
    path.closeSubpath()
    return path


def make_icon_pixmap(size: int = 256) -> QPixmap:
    """Render the diive app icon at `size` px square: a glossy blue-teal gradient
    tile with layered cyan waves and a white "[d]" wordmark above them (the splash
    motif).

    Depth comes from a diagonal gradient + a top-left light glow + a deeper water
    band under the waves, all drawn as vectors so they stay crisp from 16 px
    (taskbar) up; the front wave carries a bright crest highlight and the "[d]" a
    soft drop shadow, so the mark reads even on a busy backdrop."""
    pm = QPixmap(size, size)
    pm.setDevicePixelRatio(1.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = float(size)

    card = QPainterPath()
    card.addRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)
    p.setClipPath(card)

    # Diagonal gradient (bright sky-blue top-left → deep teal bottom-right) gives
    # the tile a light direction; brighter/more saturated than the splash so it
    # holds up on a light *or* dark taskbar.
    grad = QLinearGradient(0, 0, s, s)
    grad.setColorAt(0.0, QColor("#4FC3F7"))   # light blue 300
    grad.setColorAt(0.45, QColor("#1E88E5"))  # blue 600
    grad.setColorAt(1.0, QColor("#00838F"))   # cyan 800
    p.fillRect(QRectF(0, 0, s, s), grad)

    # Soft light glow from the top-left corner.
    glow = QRadialGradient(s * 0.28, s * 0.22, s * 0.75)
    glow.setColorAt(0.0, QColor(255, 255, 255, 70))
    glow.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.fillRect(QRectF(0, 0, s, s), glow)

    # Deeper-water band under the waves so the lower third feels like depth, not
    # just stripes (echoes the splash's darker bottom).
    deep = QLinearGradient(0, s * 0.6, 0, s)
    deep.setColorAt(0.0, QColor(0, 80, 110, 0))
    deep.setColorAt(1.0, QColor(0, 70, 100, 90))
    p.fillRect(QRectF(0, s * 0.6, s, s * 0.4), deep)

    # Layered translucent-white "foam" waves across the lower third, back→front.
    # Kept in the lower ~28% so they never crowd the "d" (matters at 16 px).
    front_base, front_amp, front_wl, front_phase = s * 0.88, s * 0.038, s * 0.46, 3.3
    p.fillPath(_wave_path(s, s * 0.72, s * 0.052, s * 0.95, 0.4), QColor(255, 255, 255, 40))
    p.fillPath(_wave_path(s, s * 0.80, s * 0.048, s * 0.62, 1.9), QColor(255, 255, 255, 70))
    p.fillPath(_wave_path(s, front_base, front_amp, front_wl, front_phase),
               QColor(255, 255, 255, 130))
    # A bright crest line tracing the front wave's surface → glossy water sheen.
    crest = QPen(QColor(255, 255, 255, 210), max(1.0, s * 0.012))
    crest.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(crest)
    p.setBrush(Qt.BrushStyle.NoBrush)
    prev = None
    x, step = 0.0, s / 96.0
    while x <= s:
        y = front_base + front_amp * math.sin(2 * math.pi * (x / front_wl) + front_phase)
        if prev is not None:
            p.drawLine(QPointF(prev[0], prev[1]), QPointF(x, y))
        prev = (x, y)
        x += step

    # Wordmark "[d]" set in real bold type, centred above the waves. The bracketed
    # "d" is the diive mark; a soft drop shadow lifts it off the gradient.
    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(s * 0.50))
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 96.0)
    p.setFont(font)
    text_rect = QRectF(0, s * 0.02, s, s * 0.74)  # upper area, clear of the waves

    p.setPen(QColor(0, 40, 70, 80))               # soft drop shadow
    p.drawText(text_rect.adjusted(s * 0.012, s * 0.018, s * 0.012, s * 0.018),
               Qt.AlignmentFlag.AlignCenter, "[d]")
    p.setPen(_WHITE)                              # the mark
    p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "[d]")
    p.end()
    return pm


def app_icon() -> QIcon:
    """The diive application/window icon (multi-size, drawn from the splash motif)."""
    icon = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(make_icon_pixmap(sz))
    return icon


class _SplashScreen(QSplashScreen):
    """Splash with a rotating loading indicator (a 12-spoke 'comet' spinner).

    A `QTimer` advances the angle and repaints, so it animates whenever the event
    loop runs — which is why the real startup defers the data load onto the loop
    (see `app.run`) instead of blocking before the splash can spin.
    """

    def __init__(self, pixmap) -> None:
        super().__init__(pixmap)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(80)  # ~12 fps; one full turn ≈ 1 s

    def _advance(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.repaint()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def drawContents(self, painter: QPainter) -> None:
        super().drawContents(painter)  # keeps showMessage() text working
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx, cy, r = _WIDTH - 52, _HEIGHT - 50, 12
        painter.translate(cx, cy)
        painter.rotate(self._angle)
        n = 12
        for i in range(n):
            alpha = int(40 + 200 * (i / (n - 1)))  # fading tail -> "comet"
            pen = QPen(QColor(255, 255, 255, alpha), 3.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(0, -(r - 5), 0, -r)
            painter.rotate(360.0 / n)
        painter.restore()


def create_splash(app=None) -> QSplashScreen:
    """Build the splash screen (high-DPI aware) ready to ``show()``."""
    dpr = 1.0
    try:
        screen = (app.primaryScreen() if app is not None else None)
        if screen is not None:
            dpr = screen.devicePixelRatio()
    except Exception:
        dpr = 1.0
    pm = make_splash_pixmap(dpr)
    splash = _SplashScreen(pm)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    # Rounded corners: clip the window to the non-transparent (rounded) region.
    mask = pm.mask()
    if not mask.isNull():
        splash.setMask(mask)
    return splash


def _screen_dpr(parent=None) -> float:
    """Device-pixel ratio of the relevant screen (for a crisp pixmap)."""
    try:
        screen = (parent.screen() if parent is not None else None) \
            or QGuiApplication.primaryScreen()
        if screen is not None:
            return screen.devicePixelRatio()
    except Exception:
        pass
    return 1.0


class _AboutDialog(QDialog):
    """Modal 'About' dialog showing the splash artwork; click or Esc to close."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("About diive")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        pm = make_splash_pixmap(_screen_dpr(parent))
        label = QLabel(self)
        label.setPixmap(pm)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label)
        self.setFixedSize(_WIDTH, _HEIGHT)

        # Faint "click to close" hint, overlaid (not baked into the shared
        # pixmap, so the startup splash stays clean).
        hint = QLabel("click anywhere to close", self)
        hint.setStyleSheet("color: rgba(255,255,255,225); background: transparent;")
        hint.adjustSize()
        hint.move(_WIDTH - hint.width() - 20, _HEIGHT - hint.height() - 14)

    def mousePressEvent(self, _event) -> None:
        self.accept()


def show_about(parent=None) -> None:
    """Show the diive 'About' dialog (the splash artwork as a modal dialog)."""
    _AboutDialog(parent).exec()


def show_message(splash: QSplashScreen, text: str) -> None:
    """Show a status line at the bottom of the splash (e.g. 'Loading data…')."""
    splash.showMessage(
        text,
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
        _LIGHT,
    )
