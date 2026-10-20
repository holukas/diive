<#
.SYNOPSIS
    Build the standalone diive GUI (Windows, one-folder) and zip it for sharing.

.DESCRIPTION
    Produces  dist\diive-gui\          (the runnable app folder)
    and       dist\diive-gui-<ver>+build.<stamp>-win64.zip  (hand this to users)

    Users unzip the archive and run diive-gui.exe. No Python / uv / pip needed.

    The default build is the public one: it leaves out the InfluxDB database
    tabs. -WithDb makes an internal build with them; its zip is named
    ...-internal-win64.zip and is meant for the working group only.

.NOTES
    Requires the build deps:  uv sync --extra gui --group build
                   (-WithDb:  uv sync --extra all --group build)
    Run from the repo root:    .\packaging\build_gui.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoZip,     # build the folder but skip zipping
    [switch]$Clean,     # remove build\ and dist\ first
    [switch]$WithDb     # internal build: bundle the InfluxDB database tabs
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

# The spec bundles influxdb-client only when DIIVE_BUILD_DB=1, so a public build
# stays without the database tabs even if the build env has the db extra.
if ($WithDb) {
    $env:DIIVE_BUILD_DB = "1"
    Write-Host "Internal build: including the InfluxDB database tabs." -ForegroundColor Yellow
} else {
    Remove-Item Env:DIIVE_BUILD_DB -ErrorAction SilentlyContinue
}

Write-Host "Running PyInstaller (this takes a few minutes) ..." -ForegroundColor Cyan
try {
    uv run pyinstaller "packaging\diive_gui.spec" --noconfirm --clean
} finally {
    Remove-Item Env:DIIVE_BUILD_DB -ErrorAction SilentlyContinue
}
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
$tag = if ($WithDb) { "-internal" } else { "" }
$zip = Join-Path $repo "dist\diive-gui-$ver+build.$build$tag-win64.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue

Write-Host "Zipping -> $zip ..." -ForegroundColor Cyan
Compress-Archive -Path $appDir -DestinationPath $zip

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  App folder: $appDir"
if ($WithDb) {
    Write-Host "  Internal build (database tabs): share within the working group only, not on GitHub." -ForegroundColor Yellow
}
Write-Host "  Share this: $zip"
Write-Host "  Users unzip it and run diive-gui.exe (nothing to install)."
