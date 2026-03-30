# Building Documentation Locally

This directory contains the documentation source for diive, built with Jupyter Book.

## Prerequisites

- Python 3.10+
- Poetry (recommended) or pip

## Installation

Install documentation dependencies:

```bash
# Using poetry (from project root)
cd ..
poetry install --with docs

# OR using pip
pip install jupyter-book myst-parser sphinx sphinx-book-theme
```

## Building

Build the documentation HTML:

```bash
# From project root
cd docs
poetry run jupyter-book build source/

# View in browser
start source/_build/html/index.html  # Windows
open source/_build/html/index.html   # macOS
xdg-open source/_build/html/index.html  # Linux
```

## Clean Build

To remove old build artifacts and rebuild:

```bash
cd docs
poetry run jupyter-book clean -a source/
poetry run jupyter-book build source/
```

## Project Structure

```
source/
├── _config.yml           # Jupyter Book configuration
├── _toc.yml             # Table of contents
├── intro.md             # Getting started guide
├── faq.md               # Frequently asked questions
├── changelog.md         # Version history (links to CHANGELOG.md)
├── guide/               # User guides
│   ├── architecture.md  # System overview
│   ├── workflows.md     # Common workflows
│   └── api_design.md    # API patterns & conventions
└── examples/            # Examples organized by topic
    ├── io.md            # Data loading & I/O
    ├── timeseries.md    # Time series processing
    ├── variables.md     # Creating variables
    ├── qc.md            # Quality control
    ├── analyses.md      # Data analysis & aggregation
    ├── gapfilling.md    # Gap-filling methods
    ├── flux.md          # Flux processing
    └── plotting.md      # Visualization
```

## Adding Notebooks

To embed Jupyter notebooks in the documentation:

1. Place notebook in `notebooks/` directory
2. Reference in the appropriate example page (e.g., `examples/qc.md`)
3. Use relative path: `[Link text](../../notebooks/path/to/notebook.ipynb)`

Example:
```markdown
- [Histogram](../../notebooks/plotting/Histogram.ipynb)
```

## Troubleshooting

### Unicode Encoding Error on Windows

If you see `UnicodeEncodeError` when running `jupyter-book`, this is a known Windows issue.

**Workaround**: Use the conda environment or set environment variable:

```bash
set PYTHONIOENCODING=utf-8
jupyter-book build source/
```

Or use WSL/Git Bash.

### Clear Caches

If builds seem stale:

```bash
jupyter-book clean source/
rm -rf source/_build/
jupyter-book build source/
```

## Deployment

Documentation is built locally and deployed manually to GitHub Pages using `ghp-import`.

### Local Build & Deploy Workflow

```bash
# From project root
cd docs

# Clean old build artifacts
poetry run jupyter-book clean -a source/

# Build documentation
poetry run jupyter-book build source/

# Deploy to GitHub Pages
poetry run ghp-import -n -p -f source/_build/html
```

**What each step does:**
- `jupyter-book clean -a` - Removes all build artifacts and cache
- `jupyter-book build source/` - Builds HTML from markdown source files
- `ghp-import -n -p -f` - Pushes built HTML to `gh-pages` branch and publishes to GitHub Pages
  - `-n` - Include .nojekyll file (don't process with Jekyll)
  - `-p` - Push to remote immediately
  - `-f` - Force push (overwrite previous version)

### GitHub Pages Setup (One-time)

If you haven't set up GitHub Pages yet:

1. Go to repository Settings → Pages
2. Under "Build and deployment":
   - Source: Select "Deploy from a branch"
   - Branch: Select `gh-pages` and `/ (root)`
3. Save

The documentation will then be available at: `https://holukas.github.io/diive/`

## Documentation Workflow

When adding new features or fixing bugs:

1. **Update code** with clear docstrings
2. **Create or update** notebook example in `notebooks/`
3. **Reference notebook** in appropriate `examples/*.md` page
4. **Test locally**: `jupyter-book build source/`
5. **Commit** docs changes alongside code

## Resources

- [Jupyter Book Documentation](https://jupyterbook.org)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

