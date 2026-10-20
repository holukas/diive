<#
.SYNOPSIS
    Build the standalone diive GUI (Windows, one-folder) and zip it for sharing.

.DESCRIPTION
    Produces  dist\diive-gui\          (the runnable app folder)
    and       dist\diive-gui-<ver>+build.<stamp>-win64.zip  (hand this to users)

    Users unzip the archive and run diive-gui.exe. No Python / uv / pip needed.

    The default build is the public one: it leaves out the InfluxDB database
    tabs. -WithDb makes an internal build with them, for the working group only:
    dist\diive-gui-internal\ and ...-internal-win64.zip. The two builds use
    separate folders, so either can be built without touching the other.

.NOTES
    Requires the build deps:  uv sync --extra gui --group build
                   (-WithDb:  uv sync --extra all --group build)
    Run from the repo root:    .\packaging\build_gui.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoZip,     # build the folder but skip zipping
    [switch]$Clean,     # remove this build's old app and work folders first
    [switch]$WithDb     # internal build: bundle the InfluxDB database tabs
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Public and internal builds get their own app and work folders. -Clean empties
# only the folders of the build being made, and keeps all zips.
$appName = if ($WithDb) { "diive-gui-internal" } else { "diive-gui" }
$appDir = Join-Path $repo "dist\$appName"
$workDir = Join-Path $repo "build\$appName"

if ($Clean) {
    Write-Host "Cleaning dist\$appName\ and build\$appName\ ..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $appDir, $workDir -ErrorAction SilentlyContinue
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
    uv run pyinstaller "packaging\diive_gui.spec" --noconfirm --clean --workpath $workDir
} finally {
    Remove-Item Env:DIIVE_BUILD_DB -ErrorAction SilentlyContinue
}
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

if (-not (Test-Path (Join-Path $appDir "diive-gui.exe"))) {
    throw "Expected dist\$appName\diive-gui.exe was not produced."
}
# Safety net: the public build must not contain the database client.
if (-not $WithDb -and (Get-ChildItem -Path $appDir -Recurse -Directory -Filter "influxdb_client*" -ErrorAction SilentlyContinue)) {
    throw "Public build contains influxdb_client. Rebuild with -Clean."
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
