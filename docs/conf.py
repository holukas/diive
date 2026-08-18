# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import date
from pathlib import Path

# Add source directory to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imported after the path insert above, so a source checkout wins over any
# diive that happens to be installed in the build environment.
from diive import __version__ as diive_version  # noqa: E402

# Whether the example gallery runs the examples. Read here rather than at the
# gallery config below, because the stylesheet list depends on it too.
execute_gallery = os.environ.get("DIIVE_DOCS_GALLERY", "0") == "1"

# Project information
project = "diive"
# Year is derived so the footer does not silently go stale; first commit was 2021.
copyright = f"2021-{date.today().year}, Lukas Hörtnagl"
author = "Lukas Hörtnagl"
# Single source of truth: diive/__init__.py reads the installed distribution
# metadata (which hatchling fills from pyproject.toml) and falls back to a
# literal only when running from an uninstalled source tree. Hardcoding it here
# meant the docs footer drifted from the package at every release.
release = diive_version
version = release

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Required: the codebase uses Google-style docstrings, which plain docutils
    # reads as malformed indentation ("Args:" becomes a block quote).
    "sphinx.ext.napoleon",
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
html_title = "diive"
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
html_css_files = ["custom.css"]
if not execute_gallery:
    # Without execution there are no figures, so every gallery card gets the
    # same stock placeholder image. Show the example summaries instead.
    html_css_files.append("gallery_textcards.css")
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

# Napoleon renders an "Attributes:" section as .. attribute:: directives and a
# "Methods:" section as .. method:: directives. Those collide with the members
# autodoc already documents on the same page, which is fatal under
# fail_on_warning (56 "duplicate object description"). Rendering both as field
# lists instead keeps the prose and drops the duplicate directive. Napoleon has
# no dedicated option for "Methods", hence the custom section.
napoleon_use_ivar = True
napoleon_custom_sections = [("Methods", "params_style")]

# Sphinx Gallery configuration - handle nested subdirectories
# Build separate galleries for each category subdirectory
examples_base = Path(__file__).parent.parent / "examples"
examples_dirs = []
gallery_dirs = []

# Dynamically add all subdirectories as separate galleries
for subdir in sorted(examples_base.iterdir()):
    if subdir.is_dir() and not subdir.name.startswith('_'):
        examples_dirs.append(str(subdir))
        gallery_dirs.append(f"auto_examples/{subdir.name}")

sphinx_gallery_conf = {
    "examples_dirs": examples_dirs,
    "gallery_dirs": gallery_dirs,
    "filename_pattern": r"^[^_].*\.py$",
    "ignore_pattern": r"(__pycache__|\.pyc|run_all_examples|__init__)",
    # Off by default: a build would execute all 113 examples, several of them
    # minutes long, which does not fit a Read the Docs build. The gallery pages
    # are still generated, without running the code or producing figures.
    # Set DIIVE_DOCS_GALLERY=1 to execute them locally (docs/build_docs.ps1 -Gallery).
    "plot_gallery": execute_gallery,
    "abort_on_example_error": False,
    "matplotlib_animations": True,
    "backreferences_dir": "api/generated",
    "doc_module": ("diive",),
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
# ".md" maps to the parser name myst_parser registers, not to "myst-nb":
# myst-nb is a separate package that is neither installed nor in requirements.txt,
# so naming it here fails the build as soon as any .md page is added under docs/.
source_suffix = {
    ".rst": None,
    ".md": "markdown",
}
