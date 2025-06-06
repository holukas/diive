"""

PLOT COLORS
-----------
LIGHT THEME

    Define default plot colors.
    *Note that this does not affect colors defined in the CSS file for the GUI.*

"""

from diive.core.plotting.styles._material_design_colors import *

# Specific colors
COLOR_GPP = '#39a7b3'  # nice blue
COLOR_RECO = '#d95318'  # nice red
COLOR_NEE = '#4CAF50'  # nice green
# COLOR_THRESHOLD = '#FFB72B'  # nice orange yellow
COLOR_THRESHOLD = '#d95318'  # nice red
# COLOR_THRESHOLD2 = '#39a7b3'  # nice blue
COLOR_THRESHOLD2 = '#BA68C8'  # purple 300
COLOR_NEP = bluegray(600)

# DEFAULT LINE PLOT
# COLOR_LINE_DEFAULT = '#026DA3'  # Line in time series plot
COLOR_LINE_DEFAULT = bluegray(500)  # Line in time series plot
WIDTH_LINE_DEFAULT = 1
WIDTH_LINE_WIDER = 4
COLOR_MARKER_DEFAULT = bluegray(500)  # Default, used in e.g. line plots
COLOR_MARKER_DEFAULT_EDGE = 'none'
SIZE_MARKER_DEFAULT = 3
SIZE_MARKER_LARGE = 30
TYPE_MARKER_DEFAULT = 'o'

# GENERAL DEFAULTS FOR ALL PLOT TYPES
LINEWIDTH_SPINES = 1  # Lines framing the plot
LINEWIDTH_ZERO = 1

# SCATTER PLOT
COLOR_SCATTER_DEFAULT = bluegray(400)

# PREVIEW MARKERS
COLOR_MARKER_PREVIEW = amber(500)
COLOR_MARKER_PREVIEW_EDGE = amber(700)
SIZE_MARKER_PREVIEW = 4
TYPE_MARKER_PREVIEW = 'o'
COLOR_AUXLINE_PREVIEW = amber(300)

# LINES
COLOR_LINE_CUMULATIVE = red(500)
COLOR_LINE_ZERO = black()

COLOR_LINE_GRID = bluegray(200)
COLOR_LINE_LIMIT = red(400)

COLOR_HISTOGRAM = lightgreen(600)
COLOR_BG_EVENTS = red(500)
COLOR_TXT_LEGEND = black()
FONTSIZE_TXT_LEGEND = 16
FONTSIZE_TXT_LEGEND_SMALLER_14 = 14
FONTSIZE_TXT_LEGEND_SMALL_9 = 9

# HEADERS & LABELS
FONTSIZE_HEADER_AXIS_LARGE = 9
FONTSIZE_HEADER_AXIS = 8
FONTSIZE_HEADER_AXIS_SMALL = 6
FONTCOLOR_HEADER_AXIS = black()

# Axis labels
AX_LABELS_FONTSIZE = 16
AX_LABELS_FONTSIZE_12 = 12
AX_LABELS_FONTCOLOR = black()
AX_LABELS_FONTWEIGHT = 'normal'

INFOTXT_FONTSIZE = 12

FONTSIZE_ANNOTATIONS_SMALL = 12

# FIGURE
FIGHEADER_FONTSIZE = 16

# Ticks
TICKS_WIDTH = 1
TICKS_LENGTH = 4
TICKS_DIRECTION = 'in'
TICKS_LABELS_FONTSIZE = 16

FONTSIZE_LEGEND = 12

FONTSIZE_INFO_LARGE_8 = 8
FONTSIZE_INFO_LARGE = 7  # Info overlay in plots, e.g. r2 = 91%
COLOR_BG_INFO_LARGE = lightgreen(400)
COLOR_TXT_INFO_LARGE = white()

FONTSIZE_INFO_DEFAULT = 6


def adjust_color_lightness(color, amount=0.5):
    # https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def colorwheel_36():
    """Create dictionary with a total of 36 colors."""
    picked_colors_dict = {}
    pick = -1
    shades = [300, 600, 900]

    for shade in shades:
        color_list = colors_12(shade=shade)
        for c in color_list:
            pick += 1
            picked_colors_dict[pick] = c

    return picked_colors_dict


def colorwheel_36_blackfirst():
    """Create dictionary with a total of 36 colors, and in addition add black as the first color."""
    picked_colors_dict = {}
    pick = 0
    picked_colors_dict[pick] = black()
    shades = [300, 600, 900]

    for shade in shades:
        color_list = colors_12(shade=shade)
        for c in color_list:
            pick += 1
            picked_colors_dict[pick] = c

    return picked_colors_dict


def colorwheel_48():
    """Create dictionary with a total of 48 colors."""
    picked_colors_dict = {}
    pick = -1
    shades = [200, 400, 600, 800]

    for shade in shades:
        color_list = colors_12(shade=shade)
        for c in color_list:
            pick += 1
            picked_colors_dict[pick] = c

    return picked_colors_dict


def colorwheel_36_wider():
    """Create dictionary with a total of 36 colors."""
    picked_colors_dict = {}
    pick = -1
    shades = [300, 600, 900]

    for shade in shades:
        color_list = colors_12(shade=shade)
        for c in color_list:
            pick += 1
            picked_colors_dict[pick] = c

    return picked_colors_dict


def colors_4_season(shade):
    list_4colors = [lightgreen(shade), red(shade), brown(shade), lightblue(shade)]
    return list_4colors


def colors_12(shade):
    """Create list of 12 colors with specific shade."""
    list_12colors = [red(shade), blue(shade), amber(shade), teal(shade),
                     purple(shade), indigo(shade), deeporange(shade), cyan(shade),
                     deeppurple(shade), green(shade), lime(shade), bluegray(shade)]
    return list_12colors


def colors_12_months():
    """Create list of 12 colors."""
    list_12colors = [lightblue(400),
                     blue(900),
                     teal(500),
                     green(600),
                     yellow(800),  # May
                     deeporange(500),  # June
                     pink(600),  # July
                     red(900),  # August
                     deeppurple(400),  # September
                     bluegray(400),
                     lime(600),
                     cyan(700)]
    return list_12colors


def colors_24():
    """Create list of 24 colors that should be more or less distinguishable."""
    list_24colors = [amber(300), deeppurple(400), indigo(300), blue(400), lightblue(300), cyan(300),
                     cyan(400), teal(300), green(400), lightgreen(300), lime(400), yellow(300),
                     purple(300), orange(300), deeporange(300), red(400), pink(400), purple(500),
                     brown(400), gray(500), bluegray(400), lime(4000), purple(2000), cyan(2000)]
    return list_24colors


def colors_6():
    list_6colors = [deeporange(500), orange(500), amber(500),
                    blue(500), cyan(500)]
    return list_6colors


def generate_plot_marker_list():
    plot_marker_dict = {0: 'o', 1: 's', 2: 'v', 3: '^',
                        4: '<', 5: '>', 6: 'd', 7: 'H',
                        8: 'D', 9: 'p'}
    return plot_marker_dict

# def colorwheel(ix):
#     # red, blue, green, amber, bluegray
#     # colorwheel500 = ['#44b2d3', '#f44336', '#2196f3', '#4caf50', '#ffc107', '#607d8b']
#     colorwheel500 = ['#78909C', '#FF9800', '#2196f3', '#4caf50', '#ffc107', '#607d8b']
#     return colorwheel500[ix]


# def generate_plot_color_dict(self, num_colors):
#     """ Generates a dict of colors based on the number of groups in the df """
#     print('[{}]'.format(self.generate_plot_color_dict.__name__))
#     # https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib
#     plot_color_dict = {}
#     cmap = matplotlib.cm.get_cmap('Reds')
#     colorstep = 1 / num_colors
#     for c in range(num_colors):
#         picked_color = 0 + (colorstep * c)
#         picked_color = cmap(picked_color)  # color as rgba (a=alpha)
#         plot_color_dict[c] = matplotlib.colors.to_hex(picked_color, keep_alpha=False)  # color as hex w/o alpha
#     return plot_color_dict

# def generate_class_color_list(self, df_with_classes, class_id_col):
#     """ Generates a dict of colors based on the number of groups in the df """
#
#     print('[{}]'.format(self.generate_class_color_list.__name__))
#
#     # https://stackoverflow.com/questions/25408393/getting-individual-colors-from-a-color-map-in-matplotlib
#     class_color_dict = {}
#     cmap = matplotlib.cm.get_cmap('plasma')
#
#     grouped_by_class = df_with_classes.groupby(class_id_col)
#     num_group_colors = len(grouped_by_class)
#     colorstep = 1 / num_group_colors
#
#     for _key1, _group1 in grouped_by_class:
#         picked_color = 1 - (colorstep * _key1)
#         picked_color = cmap(picked_color)  # color as rgba (a=alpha)
#
#         class_color_dict[_key1] = matplotlib.colors.to_hex(picked_color, keep_alpha=False)  # color as hex w/o alpha
#
#         # convert rgba to hex
#         # class_color_dict[_key1] = '#{:02x}{:02x}{:02x}'.format(picked_color[0], picked_color[1], picked_color[2])
#
#         # class_color_dict[_key1] = picked_color
#
#     return class_color_dict
