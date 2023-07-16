import matplotlib.patches as patches


def rectangle(ax, rect_lower_left_x, rect_lower_left_y, rect_width, rect_height, label,
              color: str = 'black', alpha: float = .2):
    rect = patches.Rectangle((rect_lower_left_x,
                              rect_lower_left_y),
                             rect_width,
                             rect_height,
                             linewidth=0, edgecolor='none', facecolor=color, alpha=alpha, zorder=1,
                             label=label, transform=ax.get_xaxis_transform(), )
    ax.add_patch(rect)
    return rect
