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

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
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

    # Credits, lower-left over the waves (white reads well on the dark teal).
    author_font = QFont()
    author_font.setPointSize(12)
    author_font.setWeight(QFont.Weight.DemiBold)
    p.setFont(author_font)
    p.setPen(_WHITE)
    p.drawText(QRectF(48, _HEIGHT - 96, _WIDTH - 96, 22),
               Qt.AlignmentFlag.AlignLeft, f"Created by {AUTHOR}")

    if SUPPORTERS:
        sup_font = QFont()
        sup_font.setPointSize(10)
        p.setFont(sup_font)
        p.setPen(_LIGHT)
        p.drawText(QRectF(48, _HEIGHT - 74, _WIDTH - 96, 20),
                   Qt.AlignmentFlag.AlignLeft,
                   "Supported by " + " · ".join(SUPPORTERS))

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


def make_icon_pixmap(size: int = 256) -> QPixmap:
    """Render the diive app icon at `size` px square: the blue-teal gradient
    tile with layered cyan waves and a white "d" above them (the splash motif)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = float(size)

    card = QPainterPath()
    card.addRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)
    p.setClipPath(card)

    # Brighter, more saturated than the splash so the icon stands out on a
    # (light or dark) taskbar; near-black navy would vanish on a dark taskbar.
    grad = QLinearGradient(0, 0, 0, s)
    grad.setColorAt(0.0, QColor("#1E88E5"))   # blue 600
    grad.setColorAt(1.0, QColor("#00ACC1"))   # cyan 600
    p.fillRect(QRectF(0, 0, s, s), grad)

    def _wave(base_f: float, amp_f: float, wl_f: float, phase: float, color: QColor) -> None:
        base_y, amp, wl = s * base_f, s * amp_f, s * wl_f
        path = QPainterPath()
        path.moveTo(0.0, s)
        path.lineTo(0.0, base_y)
        x, step = 0.0, s / 64.0
        while x <= s:
            path.lineTo(x, base_y + amp * math.sin(2 * math.pi * (x / wl) + phase))
            x += step
        path.lineTo(s, s)
        path.closeSubpath()
        p.fillPath(path, color)

    # Layered translucent-white "foam" waves across the lower third.
    _wave(0.66, 0.055, 0.95, 0.4, QColor(255, 255, 255, 45))
    _wave(0.76, 0.050, 0.62, 1.9, QColor(255, 255, 255, 75))
    _wave(0.86, 0.040, 0.46, 3.3, QColor(255, 255, 255, 120))

    # Wordmark "d" drawn as a vector (font-independent, always crisp): a ring
    # "bowl" with a vertical stem on its right, sitting above the waves.
    w = s * 0.10                 # stroke / stem width
    r = s * 0.155                # bowl radius
    cx, cy = s * 0.47, s * 0.43  # bowl centre (above the waves)
    y_top = s * 0.17             # stem reaches up to here
    pen = QPen(_WHITE, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))   # bowl (ring -> counter)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_WHITE)
    p.drawRoundedRect(QRectF(cx + r - w, y_top, w, (cy + r) - y_top),
                      w / 2.0, w / 2.0)                    # stem
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
