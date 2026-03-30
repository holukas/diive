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
jupyter-book build source/

# View in browser
start source/_build/html/index.html  # Windows
open source/_build/html/index.html   # macOS
xdg-open source/_build/html/index.html  # Linux
```

## Clean Build

To remove old build artifacts and rebuild:

```bash
jupyter-book clean source/
jupyter-book build source/
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

### GitHub Pages (Automated with GitHub Actions)

The project includes automated CI/CD for documentation deployment. When you push to `main` or `indev` branches, GitHub Actions automatically:

1. Builds the documentation
2. Deploys to GitHub Pages
3. Makes it available at `https://username.github.io/diive/`

**Setup Instructions:**

1. **Ensure workflow file exists**: `.github/workflows/docs.yml` (already included)

2. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Under "Build and deployment":
     - Source: Select "GitHub Actions"
     - (Do NOT select "Deploy from a branch")

3. **Update domain (optional)**:
   - Edit `.github/workflows/docs.yml` line with `cname:`
   - Replace `diive.example.com` with your actual domain, or remove the line for default GitHub Pages URL

4. **Verify deployment**:
   - After pushing to `main`, check the "Actions" tab in GitHub
   - Look for "Build and Deploy Documentation" workflow
   - Once successful, visit `https://username.github.io/diive/`

**Workflow triggers** (automatic deployment on):
- Push to `main` or `indev` branches
- Changes to `diive/`, `notebooks/`, `docs/`, `pyproject.toml`, or `poetry.lock`
- Manual workflow dispatch (if needed)

### Manual Deployment

To deploy to GitHub Pages manually (if GitHub Actions is not available):

```bash
# Install ghp-import (if not already installed)
pip install ghp-import

# Build documentation
cd docs
jupyter-book build source/

# Deploy to GitHub Pages
ghp-import -n -p -f source/_build/html
```

This pushes the built HTML to the `gh-pages` branch and GitHub Pages serves it automatically.

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

