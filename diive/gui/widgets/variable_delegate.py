"""
GUI.WIDGETS.VARIABLE_DELEGATE: LIST ITEM RENDERER
=================================================

`VariableDelegate` paints variable-list rows itself so the panel highlight is
reliable: once a `QListWidget` is styled via a stylesheet, Qt ignores per-item
`setBackground`/`setForeground`, so role-based colouring silently fails. The
delegate also draws a colored "pill" tag for recognised variables. The
name->kind classification comes from `dv.variables.classify_variable` (library
domain knowledge); this module only maps a kind to its pill colour/label.

Each item carries its panel position in `PANEL_ROLE` (0 = not shown; 1 = the
primary panel; 2+ = additional panels), which the delegate maps to colours.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from diive.core.metadata import FAVORITE, ORIGINAL
from diive.gui import metadata_store, theme
from diive.variables import classify_variable

#: Variable name (shared with VariableList click handling).
NAME_ROLE = Qt.ItemDataRole.UserRole
#: Panel position: 0 = unselected, 1 = primary, 2+ = additional panels.
PANEL_ROLE = Qt.ItemDataRole.UserRole + 1
#: True for features created by the user (feature engineer) -> "NEW" pill.
CREATED_ROLE = Qt.ItemDataRole.UserRole + 2
#: True while the variable's plot is loading -> animated loading bar.
LOADING_ROLE = Qt.ItemDataRole.UserRole + 3

def _tok(name: str) -> QColor:
    """Live theme token as a QColor (re-read each call for live preview)."""
    return QColor(theme.manager.tokens[name])


def _pill_for(name):
    """Return ``(label, background, text_color)`` for a variable, or None.

    Reads the live theme (`theme.manager`) so Settings edits preview instantly.
    Classification (name -> kind) is the library's job; colours/labels are GUI.
    """
    vc = classify_variable(name)
    if vc is None:
        return None
    entry = theme.manager.pills.get(vc.kind)
    if entry is None:
        return None
    label, bg, fg = entry
    return label, QColor(bg), QColor(fg)


def _new_pill():
    label, bg, fg = theme.manager.new_pill
    return label, QColor(bg), QColor(fg)


class VariableDelegate(QStyledItemDelegate):
    """Render a variable row: highlight + optional pill + loading indicator."""

    def sizeHint(self, option, index) -> QSize:
        # Width 1 so long names never force a horizontal scrollbar (which would
        # push the right-aligned pill out of view). Rows still paint at the
        # full viewport width via option.rect; we elide the name ourselves.
        s = super().sizeHint(option, index)
        return QSize(1, max(s.height(), 26))

    @staticmethod
    def _paint_metadata(painter, option, rect, name, anchor_right) -> int:
        """Draw a favorite ★ and a faint extra-tag count right of the name (left
        of the pill). Returns the horizontal space consumed so the name can be
        elided clear of it."""
        md = metadata_store.manager.store.peek(name)
        if md is None:
            return 0
        tags = md.tags
        fav = FAVORITE in tags
        extra = len(tags - {ORIGINAL, FAVORITE})
        if not fav and extra == 0:
            return 0

        font = QFont(option.font)
        font.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.0))
        painter.setFont(font)
        fm = painter.fontMetrics()
        x = anchor_right
        used = 0
        if extra:
            text = f"●{extra}"
            w = fm.horizontalAdvance(text)
            r = QRect(x - w, rect.center().y() - fm.height() // 2, w, fm.height())
            painter.setPen(QColor("#90A4AE"))  # blue-grey 300 — quiet count
            painter.drawText(r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, text)
            x -= w + 6
            used += w + 6
        if fav:
            w = fm.horizontalAdvance("★")
            r = QRect(x - w, rect.center().y() - fm.height() // 2, w, fm.height())
            painter.setPen(QColor("#FFB300"))  # amber 600 — favorite star
            painter.drawText(r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "★")
            used += w + 4
        return used

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setClipRect(option.rect)  # never draw past the row
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)

        rect: QRect = option.rect
        name = index.data(NAME_ROLE) or index.data(Qt.ItemDataRole.DisplayRole) or ""
        order = index.data(PANEL_ROLE) or 0

        # Row background by selection state (or hover). Colours read live.
        bg, fg, bold = None, _tok("TEXT_FG"), False
        if order == 1:
            bg, fg, bold = _tok("PRIMARY_BG"), _tok("PRIMARY_FG"), True
        elif order > 1:
            bg, fg, bold = _tok("EXTRA_BG"), _tok("EXTRA_FG"), True
        elif option.state & QStyle.StateFlag.State_MouseOver:
            bg = _tok("HOVER_BG")
        if bg is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(rect.adjusted(2, 1, -2, -1), 4, 4)

        # Loading indicator: a translucent accent wash over the row plus a
        # solid accent bar along the bottom. Static (matplotlib renders
        # synchronously, blocking the event loop) but painted before the render
        # so it's visible during the wait.
        if index.data(LOADING_ROLE):
            band = rect.adjusted(2, 1, -2, -1)
            wash = QColor(_tok("ACCENT"))
            wash.setAlpha(45)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(wash)
            painter.drawRoundedRect(band, 4, 4)
            bar = QColor(_tok("ACCENT"))
            painter.setBrush(bar)
            painter.drawRect(band.left(), band.bottom() - 3, band.width(), 3)

        # User-created features get the "NEW" pill; otherwise classify by name.
        pill = _new_pill() if index.data(CREATED_ROLE) else _pill_for(name)

        # Pill tag (right-aligned), drawn first so we know how much width to
        # reserve for the text.
        pill_w = 0
        ind_anchor = rect.right() - 8  # right edge for metadata indicators
        if pill is not None:
            pill_label, pill_color, pill_fg = pill
            pill_font = QFont(option.font)
            pill_font.setBold(True)
            pill_font.setPointSizeF(max(7.0, option.font.pointSizeF() - 1.0))
            painter.setFont(pill_font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(pill_label)
            ph = fm.height() + 2
            pw = tw + 14
            px = rect.right() - pw - 8
            py = rect.center().y() - ph // 2
            pill_rect = QRect(px, py, pw, ph)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(pill_color)
            painter.drawRoundedRect(pill_rect, ph / 2.0, ph / 2.0)
            painter.setPen(pill_fg)
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, pill_label)
            pill_w = pw + 12
            ind_anchor = px - 6

        # Metadata indicators (left of the pill): a favorite star and a small
        # count of extra tags. Read live from the store so they track edits made
        # in other tabs (the panel repaints on the store's `changed` signal).
        pill_w += self._paint_metadata(painter, option, rect, name, ind_anchor)

        # Variable name (numbered with its panel position when shown).
        text_font = QFont(option.font)
        text_font.setBold(bold)
        painter.setFont(text_font)
        painter.setPen(fg)
        label = f"{order}  {name}" if order else str(name)
        text_rect = rect.adjusted(10, 0, -8 - pill_w, 0)
        elided = painter.fontMetrics().elidedText(
            label, Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            elided,
        )

        painter.restore()
