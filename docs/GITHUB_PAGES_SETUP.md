# GitHub Pages Setup Guide

This guide walks through setting up automated documentation deployment to GitHub Pages.

## Prerequisites

- Repository is on GitHub (not Gitea or other platform)
- You have push access to the repository
- The `.github/workflows/docs.yml` file exists (it does!)

## Step-by-Step Setup

### Step 1: Go to Repository Settings

1. Navigate to your GitHub repository
2. Click **Settings** (top right)
3. Look for **Pages** in the left sidebar (under "Code and automation")

### Step 2: Configure GitHub Pages

1. **Build and deployment** section:
   - **Source**: Select `GitHub Actions` from the dropdown
   - (Do NOT select "Deploy from a branch" — we're using Actions)

2. Click **Save** (if there's a save button)

### Step 3: (Optional) Configure Custom Domain

If you want docs at a custom domain (e.g., `docs.example.com`):

1. In the **Custom domain** field, enter your domain
2. Update your DNS provider to point to GitHub's servers
3. GitHub will show instructions specific to your domain type

**Or skip this** to use the default: `https://username.github.io/diive/`

### Step 4: Update Workflow Configuration (Optional)

The workflow file `.github/workflows/docs.yml` has a placeholder domain:

```yaml
cname: diive.example.com  # Update with actual domain or remove
```

**Option A**: Use custom domain
- Replace `diive.example.com` with your actual domain

**Option B**: Use default GitHub Pages URL
- Delete or comment out the `cname:` line
- Documentation will be at `https://username.github.io/diive/`

### Step 5: Test the Workflow

1. Make a small change to documentation (e.g., edit `docs/source/intro.md`)
2. Commit and push to `main` or `indev` branch
3. Go to repository **Actions** tab
4. Look for **"Build and Deploy Documentation"** workflow
5. Click it to see build progress
6. Once **green checkmark** appears, the build succeeded

### Step 6: View Your Documentation

After successful build:

- **Default GitHub Pages**: `https://username.github.io/diive/`
- **Custom domain**: `https://yourdomain.com/`

The first deployment may take a few minutes to appear.

## Workflow Behavior

The workflow **automatically** triggers on:

- ✅ Push to `main` branch → **Builds and deploys**
- ✅ Push to `indev` branch → **Builds (no deploy)** for testing
- ✅ Changes to `diive/`, `notebooks/`, `docs/`, `pyproject.toml`, `poetry.lock`
- ✅ Manual dispatch via Actions tab

The workflow **skips** deployment on:
- Pull requests (only builds to verify)
- Pushes to branches other than `main`/`indev`

## Troubleshooting

### "Source is not GitHub Actions"

**Problem**: Settings still shows "Deploy from a branch" instead of "GitHub Actions"

**Solution**:
1. In Pages settings, change Source to `GitHub Actions`
2. Save
3. Wait 30 seconds and refresh

### Workflow fails with error

**Check the Actions tab**:
1. Go to **Actions** → **Build and Deploy Documentation**
2. Click the failed run
3. Check the error message
4. Common issues:
   - Missing `poetry.lock` file → run `poetry lock`
   - Python version mismatch → check `pyproject.toml`
   - Dependencies missing → run `poetry install`

### Site not updating

**Potential causes**:

1. **Workflow not running**:
   - Check Actions tab for failed/skipped runs
   - Verify files match trigger paths in `docs.yml`

2. **Browser cache**:
   - Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or clear browser cache

3. **GitHub Pages caching**:
   - GitHub Pages caches for ~5 minutes
   - Wait a few minutes and try again

### Want to disable automatic deployment

Edit `.github/workflows/docs.yml` and:

**Option 1**: Comment out the deployment step:
```yaml
# - name: Deploy to GitHub Pages
#   if: github.event_name == 'push' && github.ref == 'refs/heads/main'
#   uses: peaceiris/actions-gh-pages@v3
#   ...
```

**Option 2**: Remove the workflow file entirely:
```bash
rm .github/workflows/docs.yml
```

Then manually deploy with `ghp-import` when needed.

## Manual Deployment (Alternative)

If you prefer not to use GitHub Actions:

```bash
# Build docs locally
poetry run jupyter-book build docs/source/

# Install ghp-import (one time)
pip install ghp-import

# Deploy
cd docs
ghp-import -n -p -f source/_build/html
```

This pushes to the `gh-pages` branch (automatic in GitHub Pages settings).

## Documentation Workflow

Once GitHub Pages is set up:

1. **Write/update documentation** in `docs/source/`
2. **Create/update notebook examples** in `notebooks/`
3. **Commit and push** to `main` (or `indev` for testing)
4. **GitHub Actions** automatically builds and deploys
5. **Check** `https://username.github.io/diive/` after ~2 minutes

No manual deployment needed!

## Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Jupyter Book Guide](https://jupyterbook.org)
- [Actions Configuration](../.github/workflows/docs.yml)
