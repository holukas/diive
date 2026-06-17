<#
.SYNOPSIS
    Build the standalone diive GUI (Windows, one-folder) and zip it for sharing.

.DESCRIPTION
    Produces  dist\diive-gui\          (the runnable app folder)
    and       dist\diive-gui-<ver>-win64.zip  (hand this to users)

    Users unzip the archive and run diive-gui.exe. No Python / uv / pip needed.

.NOTES
    Requires the build deps:  uv sync --extra gui --group build
    Run from the repo root:    .\packaging\build_gui.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoZip,     # build the folder but skip zipping
    [switch]$Clean      # remove build\ and dist\ first
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if ($Clean) {
    Write-Host "Cleaning build\ and dist\ ..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}

# Render the GUI user manual (MANUAL.md -> MANUAL.html) so the exe bundles a
# fresh copy. The spec includes gui/MANUAL.html in the build's data files.
Write-Host "Rendering user manual (MANUAL.md -> MANUAL.html) ..." -ForegroundColor Cyan
uv run python -m diive.gui.build_manual
if ($LASTEXITCODE -ne 0) { throw "Manual render failed." }

# Stamp a build identifier (timestamp) so each deploy of the same version is
# distinguishable. Written next to splash.py; the spec bundles it and the GUI
# reads it (splash.build_number) to show "build <stamp>". Absent in source runs.
$build = Get-Date -Format "yyyyMMdd.HHmmss"
$buildInfo = Join-Path $repo "diive\gui\_build_info.txt"
Set-Content -Path $buildInfo -Value $build -NoNewline -Encoding utf8
Write-Host "Build stamp: $build" -ForegroundColor Cyan

Write-Host "Running PyInstaller (this takes a few minutes) ..." -ForegroundColor Cyan
uv run pyinstaller "packaging\diive_gui.spec" --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$appDir = Join-Path $repo "dist\diive-gui"
if (-not (Test-Path (Join-Path $appDir "diive-gui.exe"))) {
    throw "Expected dist\diive-gui\diive-gui.exe was not produced."
}

if ($NoZip) {
    Write-Host "Build complete: $appDir" -ForegroundColor Green
    return
}

# Read the package version to name the zip.
$ver = (uv run python -c "import diive; print(diive.__version__)").Trim()
$zip = Join-Path $repo "dist\diive-gui-$ver+build.$build-win64.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue

Write-Host "Zipping -> $zip ..." -ForegroundColor Cyan
Compress-Archive -Path $appDir -DestinationPath $zip

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  App folder: $appDir"
Write-Host "  Share this: $zip"
Write-Host "  Users unzip it and run diive-gui.exe (nothing to install)."
