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
$zip = Join-Path $repo "dist\diive-gui-$ver-win64.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue

Write-Host "Zipping -> $zip ..." -ForegroundColor Cyan
Compress-Archive -Path $appDir -DestinationPath $zip

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  App folder: $appDir"
Write-Host "  Share this: $zip"
Write-Host "  Users unzip it and run diive-gui.exe (nothing to install)."
