# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Add source directory to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "DIIVE"
copyright = "2025, Lukas Hörtnagl"
author = "Lukas Hörtnagl"
version = "0.91.0"
release = version

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Language and internationalization
language = "en"
locale_dirs = ["locale"]
gettext_compact = False

# HTML output options
html_theme = "furo"
html_title = "DIIVE"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#0066cc",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4da6ff",
        "color-brand-content": "#4da6ff",
    },
}
html_static_path = ["_static"]
html_css_files = []
html_logo = None

# Autodoc configuration
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
}

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"^(?!__).*\.py$",
    "ignore_pattern": r"__pycache__|\.pyc|run_all_examples",
    "plot_gallery": True,
    "abort_on_example_error": False,
    "matplotlib_animations": True,
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# MyST configuration
myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
myst_url_schemes = ("http", "https", "mailto")

# Source and suffix
source_suffix = {
    ".rst": None,
    ".md": "myst-nb",
}
