"""
Collection of functions for easy access to Material Design colors.

    For more info about Material Design see here: https://material.io/design/
    List of colors: https://material.io/design/color/#tools-for-picking-colors
    Convenient for color picking:
        https://www.materialpalette.com/colors
        https://material.io/design/color/the-color-system.html#tools-for-picking-colors

    Some Material Design color start with a letter. To keep this list integer-only,
    these colors are expressed as an integer with an additional 0 at the end of their
    id. For example, A100 has id 1000, A700 has id 7000.

"""


def red(shade):
    c = {50: '#ffebee',
         100: '#ffcdd2', 200: '#ef9a9a', 300: '#e57373',
         400: '#ef5350', 500: '#f44336', 600: '#e53935',
         700: '#d32f2f', 800: '#c62828', 900: '#b71c1c',
         1000: '#ff8a80', 2000: '#ff5252', 4000: '#ff1744', 7000: '#d50000'}
    return c[shade]


def pink(shade):
    c = {50: '#FCE4EC',
         100: '#F8BBD0', 200: '#F48FB1', 300: '#F06292',
         400: '#EC407A', 500: '#E91E63', 600: '#D81B60',
         700: '#C2185B', 800: '#AD1457', 900: '#880E4F',
         1000: '#FF80AB', 2000: '#FF4081', 4000: '#F50057', 7000: '#C51162'}
    return c[shade]


def purple(shade):
    c = {50: '#F3E5F5',
         100: '#E1BEE7', 200: '#CE93D8', 300: '#BA68C8',
         400: '#AB47BC', 500: '#9C27B0', 600: '#8E24AA',
         700: '#7B1FA2', 800: '#6A1B9A', 900: '#4A148C',
         1000: '#EA80FC', 2000: '#E040FB', 4000: '#D500F9', 7000: '#AA00FF'}
    return c[shade]


def deeppurple(shade):
    c = {50: '#EDE7F6',
         100: '#D1C4E9', 200: '#B39DDB', 300: '#9575CD',
         400: '#7E57C2', 500: '#673AB7', 600: '#5E35B1',
         700: '#512DA8', 800: '#4527A0', 900: '#311B92',
         1000: '#B388FF', 2000: '#7C4DFF', 4000: '#651FFF', 7000: '#6200EA'}
    return c[shade]


def indigo(shade):
    c = {50: '#E8EAF6',
         100: '#C5CAE9', 200: '#9FA8DA', 300: '#7986CB',
         400: '#5C6BC0', 500: '#3F51B5', 600: '#3949AB',
         700: '#303F9F', 800: '#283593', 900: '#1A237E',
         1000: '#8C9EFF', 2000: '#536DFE', 4000: '#3D5AFE', 7000: '#304FFE'}
    return c[shade]


def blue(shade):
    c = {50: '#e3f2fd',
         100: '#bbdefb', 200: '#90caf9', 300: '#64b5f6',
         400: '#42a5f5', 500: '#2196f3', 600: '#1e88e5',
         700: '#1976d2', 800: '#1565c0', 900: '#0d47a1',
         1000: '#82b1ff', 2000: '#448aff', 4000: '#2979ff', 7000: '#2962ff'}
    return c[shade]


def lightblue(shade):
    c = {50: '#E1F5FE',
         100: '#B3E5FC', 200: '#81D4FA', 300: '#4FC3F7',
         400: '#29B6F6', 500: '#03A9F4', 600: '#039BE5',
         700: '#0288D1', 800: '#0277BD', 900: '#01579B',
         1000: '#80D8FF', 2000: '#40C4FF', 4000: '#00B0FF', 7000: '#0091EA'}
    return c[shade]


def cyan(shade):
    c = {50: '#E0F7FA',
         100: '#B2EBF2', 200: '#80DEEA', 300: '#4DD0E1',
         400: '#26C6DA', 500: '#00BCD4', 600: '#00ACC1',
         700: '#0097A7', 800: '#00838F', 900: '#006064',
         1000: '#84FFFF', 2000: '#18FFFF', 4000: '#00E5FF', 7000: '#00B8D4'}
    return c[shade]


def teal(shade):
    c = {50: '#E0F2F1',
         100: '#B2DFDB', 200: '#80CBC4', 300: '#4DB6AC',
         400: '#26A69A', 500: '#009688', 600: '#00897B',
         700: '#00796B', 800: '#00695C', 900: '#004D40',
         1000: '#A7FFEB', 2000: '#64FFDA', 4000: '#1DE9B6', 7000: '#00BFA5'}
    return c[shade]


def green(shade):
    c = {50: '#E8F5E9',
         100: '#C8E6C9', 200: '#A5D6A7', 300: '#81C784',
         400: '#66BB6A', 500: '#4CAF50', 600: '#43A047',
         700: '#388E3C', 800: '#2E7D32', 900: '#1B5E20',
         1000: '#B9F6CA', 2000: '#69F0AE', 4000: '#00E676', 7000: '#00C853'
         }
    return c[shade]


def lightgreen(shade):
    c = {50: '#F1F8E9',
         100: '#DCEDC8', 200: '#C5E1A5', 300: '#AED581',
         400: '#9CCC65', 500: '#8BC34A', 600: '#7CB342',
         700: '#689F38', 800: '#558B2F', 900: '#33691E',
         1000: '#CCFF90', 2000: '#B2FF59', 4000: '#76FF03', 7000: '#64DD17'
         }
    return c[shade]


def lime(shade):
    c = {50: '#F9FBE7',
         100: '#F0F4C3', 200: '#E6EE9C', 300: '#DCE775',
         400: '#D4E157', 500: '#CDDC39', 600: '#C0CA33',
         700: '#AFB42B', 800: '#9E9D24', 900: '#827717',
         1000: '#F4FF81', 2000: '#EEFF41', 4000: '#C6FF00', 7000: '#AEEA00'}
    return c[shade]


def yellow(shade):
    c = {50: '#FFFDE7',
         100: '#FFF9C4', 200: '#FFF59D', 300: '#FFF176',
         400: '#FFEE58', 500: '#FFEB3B', 600: '#FDD835',
         700: '#FBC02D', 800: '#F9A825', 900: '#F57F17',
         1000: '#FFFF8D', 2000: '#FFFF00', 4000: '#FFEA00', 7000: '#FFD600'}
    return c[shade]


def amber(shade):
    c = {50: '#FFF8E1',
         100: '#FFECB3', 200: '#FFE082', 300: '#FFD54F',
         400: '#FFCA28', 500: '#FFC107', 600: '#FFB300',
         700: '#FFA000', 800: '#FF8F00', 900: '#FF6F00',
         1000: '#FFE57F', 2000: '#FFD740', 4000: '#FFC400', 7000: '#FFAB00'}
    return c[shade]


def orange(shade):
    c = {50: '#FFF3E0',
         100: '#FFE0B2', 200: '#FFCC80', 300: '#FFB74D',
         400: '#FFA726', 500: '#FF9800', 600: '#FB8C00',
         700: '#F57C00', 800: '#EF6C00', 900: '#E65100',
         1000: '#FFD180', 2000: '#FFAB40', 4000: '#FF9100', 7000: '#FF6D00'}
    return c[shade]


def deeporange(shade):
    c = {50: '#FBE9E7',
         100: '#FFCCBC', 200: '#FFAB91', 300: '#FF8A65',
         400: '#FF7043', 500: '#FF5722', 600: '#F4511E',
         700: '#E64A19', 800: '#D84315', 900: '#BF360C',
         1000: '#FF9E80', 2000: '#FF6E40', 4000: '#FF3D00', 7000: '#DD2C00'}
    return c[shade]


def brown(shade):
    c = {50: '#EFEBE9',
         100: '#D7CCC8', 200: '#BCAAA4', 300: '#A1887F',
         400: '#8D6E63', 500: '#795548', 600: '#6D4C41',
         700: '#5D4037', 800: '#4E342E', 900: '#3E2723'}
    return c[shade]


def gray(shade):
    c = {50: '#FAFAFA',
         100: '#F5F5F5', 200: '#EEEEEE', 300: '#E0E0E0',
         400: '#BDBDBD', 500: '#9E9E9E', 600: '#757575',
         700: '#616161', 800: '#424242', 900: '#212121'}
    return c[shade]


def bluegray(shade):
    c = {50: '#ECEFF1',
         100: '#CFD8DC', 200: '#B0BEC5', 300: '#90A4AE',
         400: '#78909C', 500: '#607D8B', 600: '#546E7A',
         700: '#455A64', 800: '#37474F', 900: '#263238'}
    return c[shade]


def black():
    return '#000000'


def white():
    return '#FFFFFF'

# def _template(shade):
#     c = {50: '#',
#          100: '#', 200: '#', 300: '#',
#          400: '#', 500: '#', 600: '#',
#          700: '#', 800: '#', 900: '#',
#          1000: '#', 2000: '#', 4000: '#', 7000: '#'}
#     return c[shade]
