"""
Sphinx configuration for diive API documentation.

This extends the Jupyter Book configuration for API reference generation.
"""

import os
import sys

# Add diive package to path for autodoc
sys.path.insert(0, os.path.abspath('../../..'))

# Project information
project = 'diive'
copyright = '2024'
author = 'Lukas Hörtnagl'
release = '0.91.0'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

# Theme
html_theme = 'sphinx_book_theme'

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Autosummary
autosummary_generate = True
autosummary_generate_overwrite = True

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# MyST settings
myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'linkify',
    'substitution',
]
