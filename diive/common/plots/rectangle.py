import matplotlib.patches as patches


def rectangle(ax, rect_lower_left_x, rect_lower_left_y, rect_width, rect_height, label):
    rect = patches.Rectangle((rect_lower_left_x,
                              rect_lower_left_y),
                             rect_width,
                             rect_height,
                             linewidth=0, edgecolor='none', facecolor='black', alpha=.1, zorder=1,
                             label=label)
    ax.add_patch(rect)
    return rect
