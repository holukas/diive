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
html_css_files = ["custom.css"]
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
    "plot_gallery": True,
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
source_suffix = {
    ".rst": None,
    ".md": "myst-nb",
}

# Post-process sg_execution_times.rst to fix cross-reference warnings
def fix_execution_times_references(app, env, docnames):
    """Convert :ref: directives to :doc: in sg_execution_times.rst files."""
    import re
    from pathlib import Path

    categories = [
        'analysis', 'binary', 'corrections', 'createvar', 'echires',
        'fits', 'flux', 'gap_filling', 'outlierdetection', 'timeseries', 'visualization'
    ]

    def replace_ref(match):
        inner = match.group(1)
        inner = inner.replace('sphx_glr_auto_examples_', '').replace('.py', '')

        for cat in categories:
            if inner.startswith(cat + '_'):
                name = inner[len(cat)+1:]
                return f":doc:`/auto_examples/{cat}/{name}`"

        return match.group(0)

    # Process all sg_execution_times.rst files
    auto_examples_dir = Path(app.srcdir) / "auto_examples"
    if auto_examples_dir.exists():
        for rst_file in auto_examples_dir.rglob("sg_execution_times.rst"):
            with open(rst_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Only process if it has the problematic :ref: directives
            if ':ref:`sphx_glr_auto_examples_' in content:
                # Replace :ref: with :doc:
                pattern = r':ref:`(sphx_glr_auto_examples_[^`]+)`'
                new_content = re.sub(pattern, replace_ref, content)

                with open(rst_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

def setup(app):
    """Register Sphinx event handlers."""
    app.connect('env-before-read-docs', fix_execution_times_references)
