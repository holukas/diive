<#
.SYNOPSIS
    Build the diive documentation with Sphinx, the way Read the Docs builds it.

.DESCRIPTION
    Produces  docs\_build\html\index.html

    Two things this does that a bare sphinx-build does not:

    * Fails on warnings, because .readthedocs.yml sets fail_on_warning: true.
      Without -W a build can look clean locally and still fail on RTD.
    * Wipes the generated directories first (_autosummary, api\generated,
      auto_examples, _build). They are untracked output; a stale page for a
      symbol that no longer exists hides the error that should be reported.

    The example gallery is NOT executed by default, matching RTD - running all
    113 examples does not fit an RTD build. Pass -Gallery to run them locally
    and get the figures.

.NOTES
    Requires the docs deps, which ship in the default sync:  uv sync
    Run from anywhere:  .\docs\build_docs.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoClean,          # keep the generated directories from the last build
    [switch]$Gallery,          # execute the examples and produce figures (slow)
    [switch]$NoFailOnWarning,  # do not stop on warnings (RTD still will)
    [switch]$Open              # open the built index.html when finished
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$docs = Join-Path $repo "docs"
$out = Join-Path $docs "_build\html"

if (-not $NoClean) {
    Write-Host "Cleaning generated directories ..." -ForegroundColor Cyan
    foreach ($dir in "_build", "_autosummary", "api\generated", "auto_examples") {
        $path = Join-Path $docs $dir
        if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    }
    $times = Join-Path $docs "sg_execution_times.rst"
    if (Test-Path $times) { Remove-Item -Force $times }
}

# conf.py reads this; the gallery config is a nested dict, so sphinx-build's -D
# cannot reach it.
$env:DIIVE_DOCS_GALLERY = if ($Gallery) { "1" } else { "0" }
if ($Gallery) {
    Write-Host "Gallery ON - the examples will be executed, this takes a while." -ForegroundColor Yellow
}

$sphinxArgs = @("-b", "html", $docs, $out)
if (-not $NoFailOnWarning) {
    # --keep-going reports every warning instead of stopping at the first, so one
    # run gives the whole list to work through.
    $sphinxArgs = @("-W", "--keep-going") + $sphinxArgs
}

Write-Host "Building docs -> $out" -ForegroundColor Cyan
uv run sphinx-build @sphinxArgs
if ($LASTEXITCODE -ne 0) { throw "Docs build failed." }

Write-Host "Docs built: $out\index.html" -ForegroundColor Green

if ($Open) { Start-Process (Join-Path $out "index.html") }
