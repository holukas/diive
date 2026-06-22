"""
PLOTTING: RECTANGLE PATCH HELPER
================================

Add a filled, borderless rectangle patch to an axes (used to highlight ranges).

Part of the diive library: https://github.com/holukas/diive
"""
import matplotlib.patches as patches


def rectangle(ax, rect_lower_left_x, rect_lower_left_y, rect_width, rect_height, label,
              color: str = 'black', alpha: float = .2):
    """Add a filled, borderless rectangle patch to *ax* and return it (y in axis-fraction coords)."""
    rect = patches.Rectangle((rect_lower_left_x,
                              rect_lower_left_y),
                             rect_width,
                             rect_height,
                             linewidth=0, edgecolor='none', facecolor=color, alpha=alpha, zorder=1,
                             label=label, transform=ax.get_xaxis_transform(), )
    ax.add_patch(rect)
    return rect
